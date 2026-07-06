"""
GANDALF 多目标回归模型（基于 pytorch_tabular）

设计要点：
  - 多目标：一次拟合多个 target（GANDALF 原生支持多输出）。
  - 特征列由调用层注入（num/cat/date），不再硬编码常量。
  - 缺失/未知类别：pytorch_tabular 对分类列自动 embedding，缺失作为独立类别；
    数值缺失由模型自行处理。
  - optuna 搜索 + Lightning Trainer 早停 + 最优参数重训。
  - 损失：GANDALFConfig.loss，默认 MSE；多目标场景下统一用 MSE，分位损失暂不在此模型
    启用（多目标分位损失需自定义 loss，留待 TabM 路线统一实现）。
"""

import gc
import json
import logging
import os
import sys
import typing
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import torch
import omegaconf
import collections
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import GANDALFConfig

from src.core.models.base import BaseRegressor
from src.core.constants.train_constants import (
    RANDOM_STATE,
    TB_DEFAULT_BATCH_SIZE,
    TB_DEFAULT_MAX_EPOCHS,
    TB_DEFAULT_N_TRIALS,
    TB_DEFAULT_NUM_WORKS,
    TB_N_JOBS,
)

torch.set_float32_matmul_precision("medium")
torch.serialization.add_safe_globals(
    [
        dict,
        list,
        int,
        float,
        str,
        collections.defaultdict,
        omegaconf.dictconfig.DictConfig,
        omegaconf.listconfig.ListConfig,
        omegaconf.nodes.AnyNode,
        omegaconf.base.ContainerMetadata,
        omegaconf.base.Metadata,
        typing.Any,
    ]
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Seed set to.*")


@contextmanager
def suppress_output():
    """同时抑制 stdout 和 stderr，用于 optuna 搜索阶段屏蔽 pytorch_tabular 冗长日志。"""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class GandalfRegressor(BaseRegressor):
    """基于 pytorch_tabular GANDALF 的多目标回归模型封装。"""

    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        date_cols: Optional[List[Tuple[str, str, str]]] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        random_state: Optional[int] = None,
        n_trials: Optional[int] = None,
        n_jobs: Optional[int] = None,
        acc: Literal["gpu", "cpu"] = "gpu",
        num_workers: Optional[int] = None,
    ):
        """
        Args:
            num_cols / cat_cols: 数值/分类特征列
            date_cols: pytorch_tabular 格式 [(col, freq, fmt)]，深度模型需要时传入
            batch_size / max_epochs: 训练批大小与最大轮数
            n_trials: optuna 试验数（通路测试传小值）
            n_jobs: 并行数
            acc: 'gpu' 或 'cpu'，无 CUDA 时自动降级 cpu
            num_workers: DataLoader 工作进程数
        """
        random_state = random_state if random_state is not None else RANDOM_STATE
        super().__init__(
            num_cols=num_cols,
            cat_cols=cat_cols,
            date_cols=[d[0] for d in (date_cols or [])],
            random_state=random_state,
            multi_target=True,
        )
        self._date_cols_pt = date_cols or []
        self.batch_size = batch_size or TB_DEFAULT_BATCH_SIZE
        self.max_epochs = max_epochs or TB_DEFAULT_MAX_EPOCHS
        self.n_trials = n_trials or TB_DEFAULT_N_TRIALS
        self.n_jobs = n_jobs or TB_N_JOBS
        self.acc = "gpu" if acc == "gpu" and torch.cuda.is_available() else "cpu"
        self.num_workers = num_workers or min(
            os.cpu_count() // 2 if os.cpu_count() else 1, TB_DEFAULT_NUM_WORKS
        )
        self.optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            lr_scheduler="StepLR",
            lr_scheduler_params={"step_size": 20},
        )
        self.model: Optional[TabularModel] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.metrics: Optional[Dict[str, float]] = None

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
        check_point_dir: str = "checkpoint",
    ) -> Tuple[Optional[TabularModel], Dict[str, Any]]:
        """训练 GANDALF 多目标模型。多目标统一用 MSE 损失（分位损失路线见 TabM）。"""
        self._check_feature_cols()
        self.target_cols = target_cols
        self._optuna_search(train, valid)
        self._set_tabular_logging(logging.INFO)
        d_cfg, t_cfg, m_cfg = self._build_configs(
            self.best_params, is_tuning=False, check_point_dir=check_point_dir
        )
        self.model = TabularModel(
            data_config=d_cfg,
            trainer_config=t_cfg,
            model_config=m_cfg,
            optimizer_config=self.optimizer_config,
        )
        self.model.fit(train, valid)
        eval_metrics = self.model.evaluate(valid)
        self.metrics = {k: v for d in eval_metrics for k, v in d.items()}
        return self.model, self.metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return self.model.predict(X)

    def save_model(self, path: Union[str, Path]) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        meta = {
            "target_cols": self.target_cols,
            "metrics": self.metrics,
            "feature_cols": self.feature_cols,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "date_cols_pt": self._date_cols_pt,
            "best_params": self.best_params,
            "random_state": self.random_state,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_model(cls, save_dir: Union[str, Path]) -> "GandalfRegressor":
        save_dir = Path(save_dir)
        with open(save_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            num_cols=meta["num_cols"],
            cat_cols=meta["cat_cols"],
            date_cols=meta.get("date_cols_pt"),
            random_state=meta.get("random_state", RANDOM_STATE),
        )
        obj.target_cols = meta.get("target_cols")
        obj.metrics = meta.get("metrics")
        obj.feature_cols = meta.get("feature_cols")
        obj.best_params = meta.get("best_params")
        obj.model = TabularModel.load_model(save_dir)
        return obj

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _set_tabular_logging(self, level: int) -> None:
        """调整 pytorch_tabular / lightning 日志级别，避免训练刷屏。"""
        logger_names = [
            "pytorch_tabular",
            "pytorch_tabular.tabular_model",
            "pytorch_tabular.tabular_datamodule",
            "pytorch_tabular.models",
            "pytorch_tabular.models.gandalf.gandalf",
            "lightning",
            "lightning.pytorch",
            "lightning.fabric",
            "pytorch_lightning",
            "lightning.pytorch.utilities.rank_zero",
            "lightning.pytorch.accelerators.cuda",
            "lightning.pytorch.utilities.seed",
            "lightning.fabric.utilities.seed",
            "pytorch_lightning.utilities.seed",
        ]
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

    def _build_configs(
        self,
        params: Optional[Dict[str, Any]] = None,
        is_tuning: bool = True,
        check_point_dir: str = "checkpoint",
    ) -> Tuple[DataConfig, TrainerConfig, GANDALFConfig]:
        if not params:
            params = {}
        use_gpu = torch.cuda.is_available() and self.acc == "gpu"
        acc = "gpu" if use_gpu else "cpu"
        data_config = DataConfig(
            target=self.target_cols,
            continuous_cols=self.num_cols,
            categorical_cols=self.cat_cols,
            date_columns=self._date_cols_pt,
            num_workers=self.num_workers,
        )
        trainer_config = TrainerConfig(
            accelerator=acc,
            devices=-1,
            batch_size=self.batch_size,
            max_epochs=min(15, self.max_epochs) if is_tuning else self.max_epochs,
            early_stopping_patience=3,
            seed=self.random_state,
            checkpoints=None if is_tuning else "valid_loss",
            checkpoints_path=check_point_dir,
            progress_bar="none" if is_tuning else "simple",
            load_best=False if is_tuning else True,
            trainer_kwargs=(
                {"enable_model_summary": False, "enable_checkpointing": False}
                if is_tuning
                else {}
            ),
        )
        model_config = GANDALFConfig(
            task="regression",
            learning_rate=params.get("learning_rate", 1e-3),
            gflu_stages=params.get("gflu_stages", 6),
            gflu_dropout=params.get("gflu_dropout", 0.1),
            gflu_feature_init_sparsity=params.get("gflu_feature_init_sparsity", 0.3),
            batch_norm_continuous_input=not is_tuning,
            target_range=None,
            seed=self.random_state,
        )
        return data_config, trainer_config, model_config

    def _optuna_search(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
    ) -> None:
        """optuna 搜索 GANDALF 超参，优化目标为验证集 valid_loss。"""
        self._set_tabular_logging(logging.ERROR)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "gflu_stages": trial.suggest_int("gflu_stages", 3, 15),
                "gflu_dropout": trial.suggest_float("gflu_dropout", 0, 0.5),
                "gflu_feature_init_sparsity": trial.suggest_float(
                    "gflu_feature_init_sparsity", 0, 0.9
                ),
            }
            d_cfg, t_cfg, m_cfg = self._build_configs(params, is_tuning=True)
            with suppress_output():
                model = TabularModel(
                    data_config=d_cfg,
                    trainer_config=t_cfg,
                    model_config=m_cfg,
                    optimizer_config=self.optimizer_config,
                )
                model.fit(train, valid)
                score = model.trainer.callback_metrics["valid_loss"].item()
                del model
                gc.collect()
                torch.cuda.empty_cache()
            return float(score)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=max(1, self.n_trials // 10),
                seed=self.random_state,
                multivariate=True,
            ),
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )
        self.best_params = study.best_params
