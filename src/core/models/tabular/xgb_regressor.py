"""
XGBoost 单目标回归模型

设计要点：
  - 单目标：一次只拟合一个 target（XGBoost 原生单输出）。
  - 损失按目标后缀自动选择：mean/variance→reg:squarederror，max/2max→reg:quantileerror
    高分位（宁高勿低），min/2min→reg:quantileerror 低分位（宁低勿高）。
  - 缺失与未知类别：分类列经 cat_encoding 归一，未知类别回退 missing token；数值缺失
    由 XGBoost 原生 missing 处理（默认 NaN）。
  - optuna 搜索 + 早停；最优参数重训；记录 train/valid 指标。
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import root_mean_squared_error, r2_score

from src.core.constants.train_constants import (
    RANDOM_STATE,
    TREE_DEFAULT_EARLY_STOPPING_ROUND,
    TREE_DEFAULT_NUM_BOOST_ROUND,
    TREE_DEFAULT_N_TRIALS,
    TREE_N_JOBS,
)
from src.core.models.base import BaseRegressor
from src.core.models.loss_config import LossSpec, get_loss_spec
from src.core.models.tabular.cat_encoding import (
    fit_cat_mappings,
    to_category_dtype,
    transform_cat_cols,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBRegressor(BaseRegressor):
    """基于 XGBoost 的单目标回归模型封装。"""

    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        num_boost_round: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        n_trials: Optional[int] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            num_cols / cat_cols: 数值/分类特征列（可由 set_feature_cols 后补全）
            num_boost_round: 最大提升轮数
            early_stopping_rounds: 早停轮数
            n_trials: optuna 搜索试验数（notebook 通路测试传小值）
            n_jobs: 并行线程数
            random_state: 随机种子
        """
        random_state = random_state if random_state is not None else RANDOM_STATE
        super().__init__(
            num_cols=num_cols,
            cat_cols=cat_cols,
            random_state=random_state,
            multi_target=False,
        )
        self.num_boost_round = num_boost_round or TREE_DEFAULT_NUM_BOOST_ROUND
        self.early_stopping_rounds = (
            early_stopping_rounds or TREE_DEFAULT_EARLY_STOPPING_ROUND
        )
        self.n_trials = n_trials or TREE_DEFAULT_N_TRIALS
        self.n_jobs = n_jobs or TREE_N_JOBS
        self.model: Optional[xgb.Booster] = None
        self.loss_spec: Optional[LossSpec] = None
        # 分类列类别映射（训练时拟合，推理时还原 + OOV 兜底）
        self._cat_mappings: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
    ) -> Tuple[Optional[xgb.Booster], Dict[str, float]]:
        """
        训练单目标模型。

        Args:
            train / valid: 含 feature_cols 与 target_cols 的 DataFrame
            target_cols: 仅使用第一个（单目标），长度>1 时告警并取首项

        Returns:
            (model, metrics)
        """
        self._check_feature_cols()
        if len(target_cols) > 1:
            warnings.warn(
                f"XGBRegressor 只支持单目标，使用第一个目标: '{target_cols[0]}'",
                UserWarning,
            )
        target = target_cols[0]
        self.target_cols = [target]
        # 按目标后缀选损失
        self.loss_spec = get_loss_spec(target)

        # 拟合分类编码（基于 train，保证 valid/推理用同一映射）
        self._cat_mappings = fit_cat_mappings(train, self.cat_cols)
        dtrain, dvalid, y_train, y_valid = self._prepare_data(train, valid, target)

        # optuna 搜索最优超参
        self._optuna_search(dtrain, dvalid, y_valid)
        # 最优参数重训
        self.model = xgb.train(
            self.best_params,
            dtrain,
            evals=[(dvalid, "valid")],
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.num_boost_round // 5,
        )
        # 指标
        train_preds = self.model.predict(dtrain)
        valid_preds = self.model.predict(dvalid)
        self.metrics = {
            "Train_R2": float(r2_score(y_train, train_preds)),
            "Valid_R2": float(r2_score(y_valid, valid_preds)),
            "Valid_RMSE": float(root_mean_squared_error(y_valid, valid_preds)),
        }
        return self.model, self.metrics

    def _prepare_data(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target: str,
    ) -> Tuple[xgb.DMatrix, xgb.DMatrix, pd.Series, pd.Series]:
        """构造 XGBoost DMatrix：分类列归一 + category dtype，目标列 dropna 对齐。"""
        # 单目标：丢弃该目标缺失的样本（无监督信号不参与训练）
        tr = train.dropna(subset=[target])
        va = valid.dropna(subset=[target])

        X_train = self._encode_features(tr)
        X_valid = self._encode_features(va)
        y_train = tr[target].astype(float)
        y_valid = va[target].astype(float)

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
        return dtrain, dvalid, y_train, y_valid

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征列编码：分类列归一字符串 -> category dtype；数值列保留 NaN（XGBoost 原生 missing）。"""
        X = df[self.feature_cols].copy()
        X = transform_cat_cols(X, self.cat_cols, self._cat_mappings)
        X = to_category_dtype(X, self.cat_cols, self._cat_mappings)
        return X

    def _build_objective(self) -> Tuple[Dict[str, Any], str]:
        """按损失规格构造 XGBoost objective/eval_metric 参数。"""
        spec = self.loss_spec
        if spec.loss_type == "quantile":
            return (
                {
                    "objective": "reg:quantileerror",
                    "quantile_alpha": spec.quantile,
                    "eval_metric": "rmse",  # 分位数无可解释的 eval_metric，仍用 rmse 监控
                },
                "valid-rmse",
            )
        # squared
        return ({"objective": "reg:squarederror", "eval_metric": "rmse"}, "valid-rmse")

    def _optuna_search(
        self,
        dtrain: xgb.DMatrix,
        dvalid: xgb.DMatrix,
        y_valid: pd.Series,
    ) -> None:
        """optuna 搜索超参，优化目标为验证集 RMSE。"""

        def objective(trial: optuna.Trial) -> float:
            obj_params, prune_key = self._build_objective()
            params = {
                **obj_params,
                "eta": trial.suggest_float("eta", 1e-4, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "gamma": trial.suggest_float("gamma", 0, 2),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }
            pruning_callback = XGBoostPruningCallback(trial, observation_key=prune_key)
            model = xgb.train(
                params,
                dtrain,
                evals=[(dvalid, "valid")],
                num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
                callbacks=[pruning_callback],
            )
            preds = model.predict(dvalid)
            return float(root_mean_squared_error(y_valid, preds))

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
        obj_params, _ = self._build_objective()
        self.best_params = study.best_params | obj_params | {
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------
    def predict(self, X: Union[pd.DataFrame, np.ndarray, xgb.DMatrix]) -> np.ndarray:
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            X_mat = xgb.DMatrix(self._encode_features(X), enable_categorical=True)
        elif isinstance(X, np.ndarray):
            X_mat = xgb.DMatrix(X)
        elif isinstance(X, xgb.DMatrix):
            X_mat = X
        else:
            raise TypeError(f"无法识别的输入类型: {type(X)}")
        return self.model.predict(X_mat)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save_model(self, save_dir: Union[str, Path]) -> None:
        self._check_fitted()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        target = self.target_cols[0]
        # 权重
        model_path = save_dir / f"{target}.ubj"
        self.model.save_model(str(model_path))
        # meta
        meta = {
            "target_cols": self.target_cols,
            "metrics": self.metrics,
            "feature_cols": self.feature_cols,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "cat_mappings": self._cat_mappings,
            "best_params": self.best_params,
            "loss_spec": _loss_spec_to_dict(self.loss_spec),
            "random_state": self.random_state,
        }
        with open(save_dir / f"{target}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_model(cls, model_dir: Union[str, Path], model_name: str) -> "XGBRegressor":
        model_dir = Path(model_dir)
        with open(model_dir / f"{model_name}.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            num_cols=meta["num_cols"],
            cat_cols=meta["cat_cols"],
            random_state=meta.get("random_state", RANDOM_STATE),
        )
        obj.feature_cols = meta["feature_cols"]
        obj._cat_mappings = meta.get("cat_mappings", {})
        obj.best_params = meta.get("best_params")
        obj.target_cols = meta.get("target_cols")
        obj.metrics = meta.get("metrics")
        obj.loss_spec = _loss_spec_from_dict(meta.get("loss_spec"))
        obj.model = xgb.Booster()
        obj.model.load_model(model_dir / f"{model_name}.ubj")
        return obj


def _loss_spec_to_dict(spec: Optional[LossSpec]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    return {"loss_type": spec.loss_type, "quantile": spec.quantile, "direction": spec.direction}


def _loss_spec_from_dict(d: Optional[Dict[str, Any]]) -> Optional[LossSpec]:
    if d is None:
        return None
    return LossSpec(
        loss_type=d["loss_type"],
        quantile=d.get("quantile"),
        direction=d.get("direction", "sym"),
    )
