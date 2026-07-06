"""
LightGBM 单目标回归模型

设计要点：
  - 单目标：一次拟合一个 target。
  - 损失按目标后缀选择：mean/variance→regression(MSE)，max/2max→quantile 高分位，
    min/2min→quantile 低分位。
  - 分类特征用 LightGBM 原生 categorical_feature（category dtype），缺失与未知类别
    经 cat_encoding 归一为 missing token。
  - optuna 搜索（n_jobs=1，LightGBM 与 optuna 并行有冲突）+ 早停 + 最优参数重训。
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback
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


class LGBMRegressor(BaseRegressor):
    """基于 LightGBM 的单目标回归模型封装。"""

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
        self.model: Optional[lgb.Booster] = None
        self.loss_spec: Optional[LossSpec] = None
        self._cat_mappings: Dict[str, List[str]] = {}

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
    ) -> Tuple[Optional[lgb.Booster], Dict[str, float]]:
        self._check_feature_cols()
        if len(target_cols) > 1:
            warnings.warn(
                f"LGBM 仅支持单目标，使用第一个目标: '{target_cols[0]}'",
                UserWarning,
            )
        target = target_cols[0]
        self.target_cols = [target]
        self.loss_spec = get_loss_spec(target)

        self._cat_mappings = fit_cat_mappings(train, self.cat_cols)
        dtrain, dvalid, y_train, y_valid = self._prepare_data(train, valid, target)

        self._optuna_search(dtrain, dvalid, y_valid)
        self.model = lgb.train(
            params=self.best_params,
            train_set=dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(self.num_boost_round // 5),
            ],
        )
        train_preds = self.model.predict(dtrain.data)
        valid_preds = self.model.predict(dvalid.data)
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
    ) -> Tuple[lgb.Dataset, lgb.Dataset, pd.Series, pd.Series]:
        """构造 LightGBM Dataset：目标 dropna 对齐，分类列归一 + category dtype。"""
        tr = train.dropna(subset=[target])
        va = valid.dropna(subset=[target])
        X_train = self._encode_features(tr)
        X_valid = self._encode_features(va)
        y_train = tr[target].astype(float)
        y_valid = va[target].astype(float)
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=self.cat_cols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=dtrain,
            categorical_feature=self.cat_cols,
            free_raw_data=False,
        )
        return dtrain, dvalid, y_train, y_valid

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols].copy()
        X = transform_cat_cols(X, self.cat_cols, self._cat_mappings)
        X = to_category_dtype(X, self.cat_cols, self._cat_mappings)
        return X

    def _build_objective(self) -> Dict[str, Any]:
        """按损失规格构造 LightGBM objective/metric 参数。"""
        spec = self.loss_spec
        if spec.loss_type == "quantile":
            return {
                "objective": "quantile",
                "alpha": spec.quantile,
                "metric": "rmse",  # 分位数无可解释 metric，仍用 rmse 监控
            }
        return {"objective": "regression", "metric": "rmse"}

    def _optuna_search(
        self,
        dtrain: lgb.Dataset,
        dvalid: lgb.Dataset,
        y_valid: pd.Series,
    ) -> None:
        def objective(trial: optuna.Trial) -> float:
            obj_params = self._build_objective()
            params = {
                **obj_params,
                "boosting_type": "gbdt",
                "feature_pre_filter": False,
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "max_depth": trial.suggest_int("max_depth", -1, 20),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 10.0, log=True),
                "verbosity": -1,
                "seed": self.random_state,
                "num_threads": self.n_jobs,
            }
            pruning_cb = LightGBMPruningCallback(trial, metric="rmse")
            model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                    pruning_cb,
                ],
            )
            preds = model.predict(dvalid.data)
            return float(root_mean_squared_error(y_valid, preds))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state,
                multivariate=True,
                n_startup_trials=max(1, self.n_trials // 10),
            ),
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=1,  # optuna 与 LightGBM 并行有冲突
            show_progress_bar=True,
        )
        self.best_params = study.best_params | self._build_objective() | {
            "boosting_type": "gbdt",
            "feature_pre_filter": False,
            "verbosity": -1,
            "seed": self.random_state,
            "num_threads": self.n_jobs,
        }

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            return self.model.predict(self._encode_features(X))
        if isinstance(X, np.ndarray):
            return self.model.predict(X)
        raise TypeError(f"不支持的预测输入类型: {type(X)}")

    def save_model(self, dir_path: Union[str, Path]) -> None:
        self._check_fitted()
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        target = self.target_cols[0]
        self.model.save_model(str(dir_path / f"{target}.txt"))
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
        with open(dir_path / f"{target}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_model(cls, dir_path: Union[str, Path], model_name: str) -> "LGBMRegressor":
        dir_path = Path(dir_path)
        with open(dir_path / f"{model_name}.json", "r", encoding="utf-8") as f:
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
        obj.model = lgb.Booster(model_file=dir_path / f"{model_name}.txt")
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
