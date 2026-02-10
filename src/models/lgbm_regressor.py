import warnings
import json
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import root_mean_squared_error, r2_score

from src.models.base import BaseRegressor
from src.constants.data_constants import (
    ACT_RATE_NM_SEC_COLS,
    RATE_COEF_COLS,
    BATCH_NUMBER_COLS,
    MACHINE_COLS,
)
from src.constants.train_constants import (
    RANDOM_STATE,
    TREE_DEFAULT_NUM_BOOST_ROUND,
    TREE_DEFAULT_EARLY_STOPPING_ROUND,
    TREE_DEFAULT_N_TRIALS,
    TREE_N_JOBS,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMRegressor(BaseRegressor):
    """
    基于 LGBM 的单目标回归模型
    """

    MISSING_TOKEN = "missing"
    DEFAULT_NUM_COLS = ACT_RATE_NM_SEC_COLS + RATE_COEF_COLS + BATCH_NUMBER_COLS
    DEFAULT_CAT_COLS = MACHINE_COLS
    DEFAULT_DATE_COLS = []

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
        基于 LightGBM 的单目标回归模型封装类。

        Args:
        -
        """
        self.num_cols = num_cols or self.DEFAULT_NUM_COLS
        self.cat_cols = cat_cols or self.DEFAULT_CAT_COLS
        random_state = random_state or RANDOM_STATE
        super().__init__(
            self.num_cols + self.cat_cols, random_state, multi_target=False
        )

        self.num_boost_round = num_boost_round or TREE_DEFAULT_NUM_BOOST_ROUND
        self.early_stopping_rounds = (
            early_stopping_rounds or TREE_DEFAULT_EARLY_STOPPING_ROUND
        )
        self.n_trials = n_trials or TREE_DEFAULT_N_TRIALS
        self.n_jobs = n_jobs or TREE_N_JOBS

        self.target_cols: List[str] | None = None
        self.best_params: Dict[str, Any] | None = None
        self.model: lgb.Booster | None = None
        self.metrics: Dict[str, float] | None = None

        self._cat_mappings: Dict[str, float] = {}

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
    ) -> Tuple[lgb.Booster, Dict[str, float]]:
        """
        LightGBM 对单个变量进行训练

        Args:
        - train(pd.DataFrame): 训练集
        - valid(pd.DataFrame): 验证集
        - target_cols(List[str]): 目标变量，LigntGBM 只支持单变量，传入的列表长度要为1，否则默认使用第一个目标变量进行训练。

        Returns:
        - model(lgb.Booster): 最优参数下训练的 LGBM 模型
        - metrics(Dict[str, float]): 验证集上的评估指标
        """
        # 训练数据
        dtrain, dvalid, y_train, y_valid, target = self._prepare_data(
            train, valid, target_cols
        )
        # 参数搜索
        self._optuna_search(dtrain, dvalid, y_valid)
        # 最优参数训练
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
        self.target_cols = [target]
        self.metrics = {
            "Train_R2": r2_score(y_train, train_preds),
            "Valid_R2": r2_score(y_valid, valid_preds),
            "Valid_RMSE": root_mean_squared_error(y_valid, valid_preds),
        }
        return self.model, self.metrics

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        """ """
        if self.model is None:
            raise RuntimeError("当前未训练模型，无法预测")
        if isinstance(X, pd.DataFrame):
            X_copy = X[self.feature_cols].copy()
            for col in self.cat_cols:
                cats = self._cat_mappings[col]
                X_copy[col] = (
                    X_copy[col]
                    .fillna(self.MISSING_TOKEN)
                    .astype(str)
                    .apply(lambda x: x if x in cats else self.MISSING_TOKEN)
                    .astype("category")
                )
            return self.model.predict(X_copy)
        if isinstance(X, np.ndarray):
            return self.model.predict(X)
        raise TypeError(f"不支持的预测输入类型 {type(X)}")

    def save_model(self, dir_path: Union[str, Path]):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        target = self.target_cols[0]
        # 保存模型
        model_path = dir_path / f"{target}.txt"
        self.model.save_model(str(model_path))
        # 保存模型元信息
        meta = {
            "target_cols": self.target_cols,
            "metrics": self.metrics,
            "feature_cols": self.feature_cols,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "cat_mappings": self._cat_mappings,
            "best_params": self.best_params,
            "random_state": self.random_state,
        }
        meta_path = dir_path / f"{target}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_model(cls, dir_path: Union[str, Path], model_name: str):
        """
        加载类
        """
        dir_path = Path(dir_path)
        model_path = dir_path / f"{model_name}.txt"
        meta_path = dir_path / f"{model_name}.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 元信息
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
        # 模型
        obj.model = lgb.Booster(model_file=model_path)
        return obj

    def _prepare_data(
        self, train: pd.DataFrame, valid: pd.DataFrame, target_cols: List[str]
    ) -> Tuple[lgb.Dataset, lgb.Dataset, pd.Series, pd.Series, str]:
        """
        构造 LGBM 数据集
        """
        if len(target_cols) > 1:
            warnings.warn(
                f"LGBM 仅支持单目标回归，已使用第一个目标变量：{target_cols[0]}",
                UserWarning,
            )
        target = target_cols[0]
        X_train = train[self.feature_cols].copy()
        y_train = train[target].copy()
        X_valid = valid[self.feature_cols].copy()
        y_valid = valid[target].copy()
        # 分类特征处理
        for col in self.cat_cols:
            train_vals = X_train[col].fillna(self.MISSING_TOKEN).astype(str)
            valid_vals = X_valid[col].fillna(self.MISSING_TOKEN).astype(str)
            categories = sorted(train_vals.unique().tolist())
            self._cat_mappings[col] = categories
            X_train[col] = train_vals.astype("category")
            X_valid[col] = valid_vals.apply(
                lambda x: x if x in categories else self.MISSING_TOKEN
            ).astype("category")
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
        return dtrain, dvalid, y_train, y_valid, target

    def _optuna_search(
        self,
        dtrain: lgb.Dataset,
        dvalid: lgb.Dataset,
        y_valid: pd.Series,
    ):
        """
        使用 optuna 对 LGBM 超参数进行搜索，优化目标为最小化 RMSE
        """

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "feature_pre_filter": False,
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 0.5, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "max_depth": trial.suggest_int("max_depth", -1, 20),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                # 采样
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                # 正则化
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 10.0, log=True),
                "verbosity": -1,
                "seed": self.random_state,
                "num_threads": 1,
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
            return root_mean_squared_error(y_valid, preds)

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
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )
        self.best_params = study.best_params | {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "feature_pre_filter": False,
            "seed": self.random_state,
            "num_threads": self.n_jobs,
        }
