import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import optuna
from pathlib import Path
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import root_mean_squared_error, r2_score
from typing import List, Tuple, Dict, Union, Any, Optional
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


class XGBRegressor(BaseRegressor):
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
        基于 XGBoost 的单目标回归模型封装类。
        """
        self.num_cols = num_cols or self.DEFAULT_NUM_COLS
        self.cat_cols = cat_cols or self.DEFAULT_CAT_COLS
        random_state = random_state or RANDOM_STATE
        super().__init__(
            feature_cols=self.num_cols + self.cat_cols,
            random_state=random_state,
            multi_target=False,
        )
        self.num_boost_round = num_boost_round or TREE_DEFAULT_NUM_BOOST_ROUND
        self.early_stopping_rounds = (
            early_stopping_rounds or TREE_DEFAULT_EARLY_STOPPING_ROUND
        )
        self.n_trials = n_trials or TREE_DEFAULT_N_TRIALS
        self.n_jobs = n_jobs or TREE_N_JOBS
        # 最佳模型
        self.best_params: Dict[str, Any] | None = None
        self.model: xgb.Booster | None = None
        self.target_cols: List[str] | None = None
        self.metrics: Dict[str, float] | None
        # 训练类别映射
        self._cat_mappings: Dict[str, List[str]] = {}

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
    ) -> Tuple[xgb.Booster, Dict[str, float]]:
        """
        XGB 对单个目标变量进行训练

        Args:
        - train(pd.DataFrame): 训练集
        - valid(pd.DataFrame): 验证集
        - target_cols(List[str]): 目标变量，XGBoost 只支持单变量，传入的列表长度要为1，否则默认使用第一个目标变量进行训练。

        Returns:
        - model(xgb.Booster): 最优参数下训练的 XGBoost 模型
        - metrics(Dict[str, float]): 验证集上的评估指标
        """
        dtrain, dvalid, y_train, y_valid, target = self._prepare_data(
            train, valid, target_cols
        )
        self._optuna_search(dtrain, dvalid, y_valid)
        self.model = xgb.train(
            self.best_params,
            dtrain,
            evals=[(dvalid, "valid")],
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.num_boost_round // 5,
        )
        train_preds = self.model.predict(dtrain)
        valid_preds = self.model.predict(dvalid)
        self.target_cols = [target]
        self.metrics = {
            "Train_R2": r2_score(y_train, train_preds),
            "Valid_R2": r2_score(y_valid, valid_preds),
            "Valid_RMSE": root_mean_squared_error(y_valid, valid_preds),
        }
        return self.model, self.metrics

    def _prepare_data(
        self, train: pd.DataFrame, valid: pd.DataFrame, target_cols: List[str]
    ) -> Tuple[xgb.DMatrix, xgb.DMatrix, pd.Series, str]:
        """
        准备训练和验证数据

        -Args:

        -Returns:
        -

        """
        if len(target_cols) > 1:
            warnings.warn(
                f"XGBRegressor 只支持单目标回归，已自动使用第一个目标: '{target_cols[0]}'",
                UserWarning,
            )
        target = target_cols[0]
        X_train = train[self.feature_cols].copy()
        y_train = train[target].copy()
        X_valid = valid[self.feature_cols].copy()
        y_valid = valid[target].copy()
        # 分类变量处理
        for col in self.cat_cols:
            train_vals = X_train[col].fillna(self.MISSING_TOKEN).astype(str)
            valid_vals = X_valid[col].fillna(self.MISSING_TOKEN).astype(str)
            categories = sorted(train_vals.unique().tolist())
            self._cat_mappings[col] = categories
            X_train[col] = train_vals.astype("category")
            X_valid[col] = valid_vals.apply(
                lambda x: x if x in categories else self.MISSING_TOKEN
            ).astype("category")

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
        return dtrain, dvalid, y_train, y_valid, target

    def _optuna_search(
        self, dtrain: xgb.DMatrix, dvalid: xgb.DMatrix, y_valid: pd.Series
    ):
        """
        使用 optuna 对 XGBoost 超参数进行搜索，优化目标为最小化 RMSE。
        """

        def objective(trial):
            params = {
                # 基础配置
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                # 学习率和树结构
                "eta": trial.suggest_float("eta", 1e-4, 0.5, log=True),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "gamma": trial.suggest_float("gamma", 0, 2),
                # 采样策略
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                # 正则化
                "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                # 其他参数
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }
            pruning_callback = XGBoostPruningCallback(
                trial, observation_key="valid-rmse"
            )
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
            return root_mean_squared_error(y_valid, preds)

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
        self.best_params = study.best_params | {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def predict(self, X):
        if self.model is None:
            raise Exception("当前未训练模型，无法预测")
        if isinstance(X, pd.DataFrame):
            X_copy = X[self.feature_cols].copy()
            for col in self.cat_cols:
                if col in X_copy:
                    cats = self._cat_mappings[col]
                    X_copy[col] = (
                        X_copy[col]
                        .fillna(self.MISSING_TOKEN)
                        .astype(str)
                        .apply(lambda x: x if x in cats else self.MISSING_TOKEN)
                        .astype("category")
                    )
            X_mat = xgb.DMatrix(X_copy, enable_categorical=True)
        elif isinstance(X, np.ndarray):
            X_mat = xgb.DMatrix(X)
        elif isinstance(X, xgb.DMatrix):
            X_mat = X
        else:
            raise TypeError(f"无法识别的输入类型: {type(X)}")
        return self.model.predict(X_mat)

    def save_model(self, save_dir: Union[str, Path]):
        """
        保存模型和相关参数
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        target = self.target_cols[0]
        # 保存模型
        model_path = save_dir / f"{target}.ubj"
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
        meta_path = save_dir / f"{target}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_model(cls, model_dir: Union[str, Path], model_name: str):
        """
        加载类
        """
        model_dir = Path(model_dir)
        model_path = model_dir / f"{model_name}.ubj"
        meta_path = model_dir / f"{model_name}.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 实例化
        obj = cls(
            num_cols=meta["num_cols"],
            cat_cols=meta["cat_cols"],
            random_state=meta.get("random_state", RANDOM_STATE),
        )
        obj.feature_cols = meta["feature_cols"]
        obj._cat_mappings = meta.get("cat_mappings", {})
        obj.best_params = meta.get("best_params")
        obj.target_cols = [meta.get("target_cols")]
        obj.metrics = meta.get("metrics")
        # 加载模型
        obj.model = xgb.Booster()
        obj.model.load_model(model_path)
        return obj
