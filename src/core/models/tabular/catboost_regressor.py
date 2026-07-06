"""
CatBoost 单目标回归模型

设计要点：
  - 单目标：一次拟合一个 target。
  - 损失按目标后缀选择：mean/variance→RMSE，max/2max→Quantile 高分位，min/2min→Quantile 低分位。
  - 分类特征用 CatBoost 原生 cat_features（Pool 索引），缺失与未知类别经 cat_encoding 归一
    为 missing token，避免未来新类别导致 predict 崩溃。
  - optuna 搜索（CPU，GPU 不支持 pruning callback）+ 早停 + 最优参数重训。
"""

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from optuna.exceptions import ExperimentalWarning
from optuna.integration import CatBoostPruningCallback
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
    transform_cat_cols,
)

warnings.filterwarnings("ignore", category=ExperimentalWarning)


class CatBoostRegressorModel(BaseRegressor):
    """基于 CatBoost 的单目标回归模型封装。"""

    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        iterations: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        n_trials: Optional[int] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        use_gpu: bool = False,
    ):
        random_state = random_state if random_state is not None else RANDOM_STATE
        super().__init__(
            num_cols=num_cols,
            cat_cols=cat_cols,
            random_state=random_state,
            multi_target=False,
        )
        self.iterations = iterations or TREE_DEFAULT_NUM_BOOST_ROUND
        self.early_stopping_rounds = (
            early_stopping_rounds or TREE_DEFAULT_EARLY_STOPPING_ROUND
        )
        self.n_trials = n_trials or TREE_DEFAULT_N_TRIALS
        self.n_jobs = n_jobs or TREE_N_JOBS
        self.use_gpu = use_gpu
        self.model: Optional[CatBoostRegressor] = None
        self.loss_spec: Optional[LossSpec] = None
        self._cat_mappings: Dict[str, List[str]] = {}

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
    ) -> tuple:
        self._check_feature_cols()
        if len(target_cols) > 1:
            warnings.warn(
                f"CatBoostRegressor 只支持单目标，使用第一个目标: '{target_cols[0]}'",
                UserWarning,
            )
        target = target_cols[0]
        self.target_cols = [target]
        self.loss_spec = get_loss_spec(target)

        # 目标 dropna 对齐 + 分类列归一
        tr = transform_cat_cols(
            train.dropna(subset=[target]), self.cat_cols, fit_cat_mappings(train, self.cat_cols)
        )
        va = transform_cat_cols(
            valid.dropna(subset=[target]), self.cat_cols, fit_cat_mappings(train, self.cat_cols)
        )
        # 拟合映射存档（推理用）
        self._cat_mappings = fit_cat_mappings(train, self.cat_cols)

        self._optuna_search(tr, va, target)
        self.best_params = self._base_params() | self.best_params
        train_pool = self._build_pool(tr, target)
        valid_pool = self._build_pool(va, target)
        self.model = CatBoostRegressor(**self.best_params)
        self.model.fit(
            train_pool,
            eval_set=valid_pool,
            verbose=self.iterations // 10,
            use_best_model=True,
        )
        # 指标
        y_train = tr[target].astype(float)
        y_valid = va[target].astype(float)
        train_preds = self.model.predict(tr[self.feature_cols])
        valid_preds = self.model.predict(va[self.feature_cols])
        self.metrics = {
            "Train_R2": float(r2_score(y_train, train_preds)),
            "Valid_R2": float(r2_score(y_valid, valid_preds)),
            "Valid_RMSE": float(root_mean_squared_error(y_valid, valid_preds)),
        }
        return self.model, self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_enc = transform_cat_cols(X, self.cat_cols, self._cat_mappings)
        return self.model.predict(X_enc[self.feature_cols])

    # ------------------------------------------------------------------
    # CatBoost 参数与数据结构
    # ------------------------------------------------------------------
    def _build_loss(self) -> str:
        """按损失规格构造 CatBoost loss_function。"""
        spec = self.loss_spec
        if spec.loss_type == "quantile":
            return f"Quantile:alpha={spec.quantile}"
        return "RMSE"

    def _base_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(
            loss_function=self._build_loss(),
            eval_metric="RMSE",
            random_seed=self.random_state,
            iterations=self.iterations,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        if self.use_gpu:
            params.update(task_type="GPU", devices="0")
        else:
            params.update(task_type="CPU", thread_count=self.n_jobs)
        return params

    def _build_pool(self, df: pd.DataFrame, target: str) -> Pool:
        X = df[self.feature_cols]
        y = df[target]
        cat_idx = [X.columns.get_loc(c) for c in self.cat_cols]
        return Pool(data=X, label=y, cat_features=cat_idx)

    def _optuna_search(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target: str,
    ) -> None:
        train_pool = self._build_pool(train, target)
        valid_pool = self._build_pool(valid, target)

        def objective(trial: optuna.Trial) -> float:
            params = self._base_params()
            params.update(
                task_type="CPU",  # optuna pruning 仅 CPU 支持
                thread_count=self.n_jobs,
                devices=None,
                depth=trial.suggest_int("depth", 4, 12),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0, 5),
                random_strength=trial.suggest_float("random_strength", 0, 5),
                border_count=trial.suggest_int("border_count", 32, 255),
                grow_policy=trial.suggest_categorical(
                    "grow_policy", ["SymmetricTree", "Depthwise"]
                ),
            )
            model = CatBoostRegressor(**params)
            pruning_callback = CatBoostPruningCallback(trial, metric="RMSE")
            model.fit(
                train_pool,
                eval_set=valid_pool,
                verbose=False,
                use_best_model=True,
                callbacks=[pruning_callback],
            )
            pred = model.predict(valid[self.feature_cols])
            return float(root_mean_squared_error(valid[target], pred))

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

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save_model(self, save_dir: Union[str, Path]) -> None:
        self._check_fitted()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        target = self.target_cols[0]
        self.model.save_model(save_dir / f"{target}.cbm")
        meta = dict(
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            best_params=self.best_params,
            metrics=self.metrics,
            cat_mappings=self._cat_mappings,
            loss_spec=_loss_spec_to_dict(self.loss_spec),
            random_state=self.random_state,
        )
        with open(save_dir / f"{target}.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load_model(
        cls, model_dir: Union[str, Path], model_name: str
    ) -> "CatBoostRegressorModel":
        model_dir = Path(model_dir)
        with open(model_dir / f"{model_name}.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(
            num_cols=meta["num_cols"],
            cat_cols=meta["cat_cols"],
            random_state=meta.get("random_state", RANDOM_STATE),
        )
        obj.feature_cols = meta["feature_cols"]
        obj.target_cols = meta["target_cols"]
        obj.best_params = meta["best_params"]
        obj.metrics = meta["metrics"]
        obj._cat_mappings = meta.get("cat_mappings", {})
        obj.loss_spec = _loss_spec_from_dict(meta.get("loss_spec"))
        model = CatBoostRegressor()
        model.load_model(model_dir / f"{model_name}.cbm")
        obj.model = model
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
