"""
TabM 多目标回归模型（基于官方 tabm 包 + 自研 Lightning 封装）

设计要点：
  - 多目标：一次拟合全部 target（TabM 原生支持 d_out=n_targets 多输出）。
  - BatchEnsemble 架构：k 个并行 MLP 头共享主干，推理取均值，兼顾集成收益与效率。
  - 训练损失为 **mean-of-loss**：对 k 个预测独立算损失再平均（不能先取均值再算 loss，
    这是 TabM 正确性的核心）。
  - 目标缺失用 **masked loss** 处理：每个目标仅在该样本真实值非缺失处计 loss，
    互不干扰（多目标场景下缺失是按通道分组的，masked loss 最大化利用样本特征）。
  - 损失按目标后缀选择：mean/variance→MSE，max/2max→quantile 高分位，
    min/2min→quantile 低分位；多目标混合损失按目标列加权平均。
  - 新类别（OOV）处理：每列 cardinality = n_unique + 1，未知值映射到 n_unique，
    one-hot 最后一列即 OOV 表示（沿用官方参考实现的兼容做法）。
  - 标签标准化：对 48 目标分别减均值除标准差训练，推理时还原。
  - 数值缺失：填 0（TabM 无原生 missing；0 在标准化后近似中性，配合 BatchNorm 吸收）。
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from src.core.constants.train_constants import (
    RANDOM_STATE,
    TB_DEFAULT_BATCH_SIZE,
    TB_DEFAULT_MAX_EPOCHS,
    TB_DEFAULT_N_TRIALS,
    TB_DEFAULT_NUM_WORKS,
    TB_N_JOBS,
)
from src.core.models.base import BaseRegressor
from src.core.models.loss_config import LossSpec, build_loss_map

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_float32_matmul_precision("medium")


class _MaskedMultiLoss(nn.Module):
    """
    多目标混合损失：按 target 后缀选损失类型，缺失用 mask 屏蔽，最终按有效样本数加权平均。

    输入：
      y_pred: (B, k, T) TabM 输出（k 路、T 目标）
      y_true: (B, T) 真实值（含 NaN）
    输出：标量 loss（mean-of-loss：k 路独立算损失再平均）

    损失类型：
      - squared: MSE
      - quantile: pinball loss，q>0.5 偏好高估（upper），q<0.5 偏好低估（lower）
    """

    def __init__(self, loss_specs: List[LossSpec]):
        super().__init__()
        self.loss_specs = loss_specs
        # 预取分位数，便于向量化
        self._quantiles = torch.tensor(
            [s.quantile if s.loss_type == "quantile" else 0.5 for s in loss_specs],
            dtype=torch.float32,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: (B, k, T) -> (B*k, T) 使每路独立计损失
        B, k, T = y_pred.shape
        pred = y_pred.reshape(B * k, T)
        # 真实值广播到 k 路
        true = y_true.repeat_interleave(k, dim=0)  # (B*k, T)
        mask = ~torch.isnan(true)  # (B*k, T) 有效位

        if mask.sum() == 0:
            return pred.sum() * 0.0  # 无有效样本，返回 0 且保留梯度图

        # 逐目标计算损失
        losses = []
        for t, spec in enumerate(self.loss_specs):
            m = mask[:, t]
            if m.sum() == 0:
                continue
            p = pred[:, t][m]
            tr = true[:, t][m]
            if spec.loss_type == "quantile":
                q = self._quantiles[t].to(p.device)
                errors = tr - p
                # pinball: max(q*err, (q-1)*err)
                loss = torch.maximum(q * errors, (q - 1) * errors).mean()
            else:
                loss = nn.functional.mse_loss(p, tr)
            losses.append(loss)
        if not losses:
            return pred.sum() * 0.0
        return torch.stack(losses).mean()


class _TabMModule(pl.LightningModule):
    """Lightning 封装 TabM：forward + masked loss + 训练步。"""

    def __init__(
        self,
        tabm_model: nn.Module,
        loss_fn: _MaskedMultiLoss,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        self.tabm = tabm_model
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["tabm_model", "loss_fn"])

    def forward(self, x_num, x_cat):
        return self.tabm(x_num, x_cat)

    def training_step(self, batch, batch_idx):
        y_pred = self(batch["x_num"], batch["x_cat"])
        loss = self.loss_fn(y_pred, batch["y"])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch["x_num"], batch["x_cat"])
        loss = self.loss_fn(y_pred, batch["y"])
        self.log("valid_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.tabm.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class _ArrayDataset(Dataset):
    """简易数组数据集：返回 dict 形式的 batch，供 TabM 消费。"""

    def __init__(self, x_num, x_cat, y):
        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y

    def __len__(self):
        return len(self.x_num) if self.x_num is not None else len(self.x_cat)

    def __getitem__(self, idx):
        item = {}
        if self.x_num is not None:
            item["x_num"] = self.x_num[idx]
        if self.x_cat is not None:
            item["x_cat"] = self.x_cat[idx]
        item["y"] = self.y[idx]
        return item


class TabMRegressor(BaseRegressor):
    """基于官方 tabm 的多目标回归模型封装。"""

    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        k: int = 16,
        d_block: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        n_trials: Optional[int] = None,
        n_jobs: Optional[int] = None,
        num_workers: Optional[int] = None,
        acc: str = "gpu",
        use_piecewise_embedding: bool = False,
        learning_rate: float = 2e-3,
        weight_decay: float = 3e-4,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            num_cols / cat_cols: 数值/分类特征列
            k: BatchEnsemble 集成大小（默认 16，较官方 32 减半以省显存）
            d_block / n_blocks / dropout: MLP 主干宽度/深度/dropout
            batch_size / max_epochs: 训练批大小与最大轮数
            n_trials: optuna 试验数
            num_workers: DataLoader 工作进程数
            acc: 'gpu'/'cpu'，无 CUDA 自动降级
            use_piecewise_embedding: 是否启用 PiecewiseLinear 数值 embedding（对 OOD 数值
                更鲁棒，但需训练分箱，开销略增）
            learning_rate / weight_decay: AdamW 超参（optuna 搜索时可覆盖）
        """
        random_state = random_state if random_state is not None else RANDOM_STATE
        super().__init__(
            num_cols=num_cols,
            cat_cols=cat_cols,
            random_state=random_state,
            multi_target=True,
        )
        self.k = k
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.use_piecewise_embedding = use_piecewise_embedding
        self.batch_size = batch_size or TB_DEFAULT_BATCH_SIZE
        self.max_epochs = max_epochs or TB_DEFAULT_MAX_EPOCHS
        # n_trials=None 时用默认值；显式传 0 表示跳过 optuna（用默认参数直接训练）
        self.n_trials = TB_DEFAULT_N_TRIALS if n_trials is None else n_trials
        self.n_jobs = n_jobs or TB_N_JOBS
        self.acc = "gpu" if acc == "gpu" and torch.cuda.is_available() else "cpu"
        self.num_workers = num_workers or min(
            os_cpu_half(), TB_DEFAULT_NUM_WORKS
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight_decay = weight_decay

        self.model: Optional[_TabMModule] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.metrics: Optional[Dict[str, float]] = None
        # 预处理器状态（fit 拟合，推理还原）
        self._cat_encoder: Optional[OrdinalEncoder] = None
        self._cat_cardinalities: Optional[List[int]] = None
        self._num_fill_value: Optional[np.ndarray] = None  # 数值缺失填 0 的均值（标准化前）
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None
        self._loss_specs: Optional[List[LossSpec]] = None

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
    ) -> Tuple[Optional[_TabMModule], Dict[str, float]]:
        """训练 TabM 多目标模型。"""
        self._check_feature_cols()
        self.target_cols = target_cols
        self._loss_specs = [build_loss_map(target_cols)[c] for c in target_cols]

        # 拟合预处理器（基于 train）
        self._fit_preprocessors(train)
        # 编码数据
        Xn_tr, Xc_tr, Y_tr = self._encode(train)
        Xn_va, Xc_va, Y_va = self._encode(valid)

        # optuna 搜索（可选，n_trials=0 时跳过用默认参数）
        if self.n_trials and self.n_trials > 0:
            self._optuna_search(Xn_tr, Xc_tr, Y_tr, Xn_va, Xc_va, Y_va)

        # 最优参数训练
        self.model = self._train_module(Xn_tr, Xc_tr, Y_tr, Xn_va, Xc_va, Y_va, self.best_params)

        # 指标（验证集，去标准化）
        valid_pred = self._predict_raw(valid).cpu().numpy()
        valid_pred_orig = valid_pred * self._y_std + self._y_mean
        y_valid_orig = valid[target_cols].to_numpy(dtype=float)
        self.metrics = self._compute_metrics(y_valid_orig, valid_pred_orig)
        return self.model, self.metrics

    def _fit_preprocessors(self, train: pd.DataFrame) -> None:
        """拟合分类编码（OrdinalEncoder + cardinality+1 OOV）与标签标准化。"""
        # 分类编码：记录每列训练见过的类别集合（含 nan 兜底），
        # 推理时未知/缺失统一映射到末位（cardinality-1），不依赖 OrdinalEncoder 的 OOV 行为
        if self.cat_cols:
            cat_df = train[self.cat_cols].astype(str).fillna("nan")
            self._cat_categories: List[List[str]] = [
                sorted(cat_df[c].unique().tolist()) for c in self.cat_cols
            ]
            # cardinality = 每列 n_unique + 1（末位为 OOV 占位）
            self._cat_cardinalities = [len(c) + 1 for c in self._cat_categories]
        else:
            self._cat_categories = []
            self._cat_cardinalities = []
        # 标记，便于 _encode 判断（保留 _cat_encoder=None 兼容旧分支）
        self._cat_encoder = None

        # 数值缺失填 0（标准化前先记均值，但这里统一填 0，标准化后近似中性）
        if self.num_cols:
            self._num_fill_value = np.zeros(len(self.num_cols), dtype=np.float32)
        # 标签标准化：每目标减均值除标准差（标准差为 0 的列填 1 避免除零）
        y = train[self.target_cols].to_numpy(dtype=float)
        self._y_mean = np.nanmean(y, axis=0)
        self._y_std = np.nanstd(y, axis=0)
        self._y_std = np.where(self._y_std < 1e-8, 1.0, self._y_std)

    def _encode(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """将 DataFrame 编码为 TabM 输入：x_num(浮点,缺0) / x_cat(long,OOV->末位) / y(标准化)。"""
        x_num = None
        if self.num_cols:
            xn = df[self.num_cols].to_numpy(dtype=np.float32)
            xn = np.nan_to_num(xn, nan=0.0)
            x_num = xn

        x_cat = None
        if self.cat_cols and self._cat_categories:
            # 自行做 OOV 映射：训练见过的类别按其索引编码，未知/缺失映射到末位（cardinality-1）
            cat_df = df[self.cat_cols].astype(str).fillna("nan")
            n_rows = len(cat_df)
            n_cat = len(self.cat_cols)
            xc = np.zeros((n_rows, n_cat), dtype=np.int64)
            for j, cats in enumerate(self._cat_categories):
                col_vals = cat_df[self.cat_cols[j]].to_numpy()
                cat_to_idx = {c: i for i, c in enumerate(cats)}
                oov_idx = len(cats)  # 末位索引 = n_unique（cardinality-1）
                xc[:, j] = np.array(
                    [cat_to_idx.get(v, oov_idx) for v in col_vals], dtype=np.int64
                )
            x_cat = xc

        y = df[self.target_cols].to_numpy(dtype=float)
        # 标准化（NaN 保持 NaN，loss 内 mask）
        y_std = (y - self._y_mean) / self._y_std
        return x_num, x_cat, y_std.astype(np.float32)

    def _make_loaders(
        self,
        Xn_tr, Xc_tr, Y_tr,
        Xn_va, Xc_va, Y_va,
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds = _ArrayDataset(Xn_tr, Xc_tr, Y_tr)
        valid_ds = _ArrayDataset(Xn_va, Xc_va, Y_va)
        collate = self._collate_fn
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate, drop_last=False,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate,
        )
        return train_loader, valid_loader

    @staticmethod
    def _collate_fn(batch):
        item = {}
        keys = batch[0].keys()
        for k in keys:
            vals = [b[k] for b in batch]
            item[k] = torch.as_tensor(np.stack(vals))
        return item

    def _build_tabm(self, params: Optional[Dict[str, Any]]) -> nn.Module:
        """构造官方 TabM 模型实例。"""
        from tabm import TabM

        p = params or {}
        num_emb = None
        if self.use_piecewise_embedding and self.num_cols:
            import rtdl_num_embeddings as rne
            # 分箱边界由训练数据分位数决定（需在外部预计算 bins）
            bins = getattr(self, "_piecewise_bins", None)
            if bins is not None:
                num_emb = rne.PiecewiseLinearEmbeddings(bins, d_embedding=16, activation=True, version="B")
        kwargs = dict(
            n_num_features=len(self.num_cols),
            cat_cardinalities=self._cat_cardinalities,
            d_out=len(self.target_cols),
            k=p.get("k", self.k),
            d_block=p.get("d_block", self.d_block),
            n_blocks=p.get("n_blocks", self.n_blocks),
            dropout=p.get("dropout", self.dropout),
            arch_type="tabm",
        )
        if num_emb is not None:
            kwargs["num_embeddings"] = num_emb
        return TabM.make(**kwargs)

    def _train_module(
        self,
        Xn_tr, Xc_tr, Y_tr,
        Xn_va, Xc_va, Y_va,
        params: Optional[Dict[str, Any]],
    ) -> _TabMModule:
        train_loader, valid_loader = self._make_loaders(Xn_tr, Xc_tr, Y_tr, Xn_va, Xc_va, Y_va)
        tabm = self._build_tabm(params)
        loss_fn = _MaskedMultiLoss(self._loss_specs)
        lr = (params or {}).get("learning_rate", self.learning_rate)
        wd = (params or {}).get("weight_decay", self.weight_decay)
        module = _TabMModule(tabm, loss_fn, lr, wd)

        trainer = Trainer(
            max_epochs=(params or {}).get("max_epochs", self.max_epochs),
            accelerator=self.acc,
            devices=-1 if self.acc == "gpu" else 1,
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False,
            callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", mode="min", patience=5)],
        )
        trainer.fit(module, train_loader, valid_loader)
        return module

    def _optuna_search(
        self,
        Xn_tr, Xc_tr, Y_tr,
        Xn_va, Xc_va, Y_va,
    ) -> None:
        """optuna 搜索 TabM 超参，优化目标为验证集 valid_loss。"""

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
                "d_block": trial.suggest_int("d_block", 64, 512, step=64),
                "n_blocks": trial.suggest_int("n_blocks", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "max_epochs": min(15, self.max_epochs),
            }
            module = self._train_module(Xn_tr, Xc_tr, Y_tr, Xn_va, Xc_va, Y_va, params)
            # 取最后一次 valid_loss
            score = module.trainer.callback_metrics.get("valid_loss")
            score = float(score) if score is not None else float("inf")
            del module
            torch.cuda.empty_cache()
            return score

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=max(1, self.n_trials // 5),
                seed=self.random_state,
            ),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        self.best_params = study.best_params

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _predict_raw(self, df: pd.DataFrame) -> torch.Tensor:
        """预测（标准化空间），返回 (n_samples, n_targets) 的 k 路均值。"""
        self._check_fitted()
        x_num, x_cat, _ = self._encode(df)
        x_num_t = torch.as_tensor(x_num) if x_num is not None else None
        x_cat_t = torch.as_tensor(x_cat) if x_cat is not None else None
        self.model.eval()
        if self.acc == "gpu":
            x_num_t = x_num_t.cuda() if x_num_t is not None else None
            x_cat_t = x_cat_t.cuda() if x_cat_t is not None else None
            self.model = self.model.cuda()
        # 分批避免显存溢出
        preds = []
        n = len(df)
        bs = max(1, self.batch_size)
        for i in range(0, n, bs):
            xn = x_num_t[i : i + bs] if x_num_t is not None else None
            xc = x_cat_t[i : i + bs] if x_cat_t is not None else None
            out = self.model(xn, xc)  # (b, k, T)
            preds.append(out.mean(dim=1))  # k 路均值 -> (b, T)
        return torch.cat(preds, dim=0)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """预测，返回 DataFrame（列名同 target_cols，已去标准化）。"""
        raw = self._predict_raw(X).cpu().numpy()
        pred_orig = raw * self._y_std + self._y_mean
        return pd.DataFrame(pred_orig, columns=self.target_cols, index=X.index)

    # ------------------------------------------------------------------
    # 指标
    # ------------------------------------------------------------------
    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        from sklearn.metrics import r2_score, root_mean_squared_error

        # 逐目标，仅有效样本
        r2s, rmses = [], []
        for t in range(y_true.shape[1]):
            yt, yp = y_true[:, t], y_pred[:, t]
            mask = ~np.isnan(yt)
            if mask.sum() < 2:
                continue
            r2s.append(r2_score(yt[mask], yp[mask]))
            rmses.append(root_mean_squared_error(yt[mask], yp[mask]))
        return {
            "Valid_R2_mean": float(np.mean(r2s)) if r2s else float("nan"),
            "Valid_RMSE_mean": float(np.mean(rmses)) if rmses else float("nan"),
            "n_targets_evaluated": float(len(r2s)),
        }

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save_model(self, save_dir: Union[str, Path]) -> None:
        self._check_fitted()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # 权重（state_dict）：存 _TabMModule 的完整状态（含 .tabm. 前缀），
        # load 时也按 _TabMModule 还原，保证 save/load 结构对称
        torch.save(self.model.state_dict(), save_dir / "tabm.pt")
        # 预处理器与超参
        meta = {
            "target_cols": self.target_cols,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "feature_cols": self.feature_cols,
            "best_params": self.best_params or {},
            "metrics": self.metrics,
            "k": self.k,
            "d_block": self.d_block,
            "n_blocks": self.n_blocks,
            "dropout": self.dropout,
            "use_piecewise_embedding": self.use_piecewise_embedding,
            "cat_categories": (
                self._cat_categories if self._cat_categories else None
            ),
            "cat_cardinalities": self._cat_cardinalities,
            "y_mean": self._y_mean.tolist(),
            "y_std": self._y_std.tolist(),
            "loss_specs": [
                {"loss_type": s.loss_type, "quantile": s.quantile, "direction": s.direction}
                for s in (self._loss_specs or [])
            ],
            "random_state": self.random_state,
        }
        with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_model(cls, save_dir: Union[str, Path]) -> "TabMRegressor":
        save_dir = Path(save_dir)
        with open(save_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            num_cols=meta["num_cols"],
            cat_cols=meta["cat_cols"],
            k=meta.get("k", 16),
            d_block=meta.get("d_block", 256),
            n_blocks=meta.get("n_blocks", 3),
            dropout=meta.get("dropout", 0.1),
            use_piecewise_embedding=meta.get("use_piecewise_embedding", False),
            random_state=meta.get("random_state", RANDOM_STATE),
        )
        obj.feature_cols = meta["feature_cols"]
        obj.target_cols = meta["target_cols"]
        obj.best_params = meta.get("best_params", {})
        obj.metrics = meta.get("metrics")
        # 还原预处理器：仅需类别集合与 cardinality（OOV 映射在 _encode 内自行完成）
        if meta.get("cat_categories") is not None:
            obj._cat_categories = [list(c) for c in meta["cat_categories"]]
            obj._cat_cardinalities = meta.get("cat_cardinalities")
        else:
            obj._cat_categories = []
            obj._cat_cardinalities = []
        obj._cat_encoder = None
        obj._y_mean = np.array(meta["y_mean"], dtype=np.float32)
        obj._y_std = np.array(meta["y_std"], dtype=np.float32)
        obj._loss_specs = [
            LossSpec(
                loss_type=s["loss_type"],
                quantile=s.get("quantile"),
                direction=s.get("direction", "sym"),
            )
            for s in meta.get("loss_specs", [])
        ]
        # 还原模型：_TabMModule 包裹 TabM，state_dict 带 .tabm. 前缀，需按包裹结构加载
        tabm = obj._build_tabm(obj.best_params)
        loss_fn = _MaskedMultiLoss(obj._loss_specs)
        lr = (obj.best_params or {}).get("learning_rate", obj.learning_rate)
        wd = (obj.best_params or {}).get("weight_decay", obj.weight_decay)
        obj.model = _TabMModule(tabm, loss_fn, lr, wd)
        state = torch.load(save_dir / "tabm.pt", map_location="cpu")
        obj.model.load_state_dict(state)
        return obj


def os_cpu_half() -> int:
    """获取 CPU 核数的一半（用于默认 num_workers），无 CPU 信息时返回 1。"""
    n = os.cpu_count()
    return max(1, n // 2) if n else 1
