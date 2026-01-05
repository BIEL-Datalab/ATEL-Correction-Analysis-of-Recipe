import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import optuna
from pathlib import Path
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import root_mean_squared_error, r2_score
from src.models.base import BaseRegressor
from typing import List, Tuple, Dict, Union

optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBRegressor(BaseRegressor):
    def __init__(
        self,
        feature_cols,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        n_trials: int = 100,
        n_jobs: int = 1,
        random_state=42,
    ):
        """
        基于 XGBoost 的单目标回归模型封装类。
        """
        # 参数
        super().__init__(feature_cols, random_state, multi_target=False)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        # 最佳模型
        self.best_params: Dict[str, float] | None = None
        self.targets: str | None = None
        self.metrics: Dict[str, float] | None

    def fit(
        self, train: pd.DataFrame, valid: pd.DataFrame, targets: List[str]
    ) -> Tuple[xgb.Booster, Dict[str, float]]:
        """
        对单个目标变量进行训练
        XGB 不支持多变量回归任务。

        Args:
        - train(pd.DataFrame): 训练集
        - valid(pd.DataFrame): 验证集
        - target_cols(List[str]): 目标变量，XGBoost 只支持单变量，传入的列表长度要为1，否则默认使用第一个目标变量进行训练。

        Returns:
        - model(xgb.Booster): 最优参数下训练的 XGBoost 模型
        - metrics(Dict[str, float]): 验证集上的评估指标
        """
        dtrain, dvalid, y_valid, target = self._prepare_data(train, valid, targets)
        self._optuna_search(dtrain, dvalid, y_valid)
        self.model = xgb.train(
            self.best_params,
            dtrain,
            evals=[(dvalid, "valid")],
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.num_boost_round // 5,
        )
        preds = self.model.predict(dvalid)
        self.targets = target
        self.metrics = {
            "RMSE": root_mean_squared_error(y_valid, preds),
            "R2": r2_score(y_valid, preds),
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
        X_train = train[self.feature_cols]
        y_train = train[target]
        X_valid = valid[self.feature_cols]
        y_valid = valid[target]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        return dtrain, dvalid, y_valid, target

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
                "eta": trial.suggest_float("eta", 1e-3, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                # 采样策略
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
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
                n_startup_trials=self.n_trials // 10,
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
            X_mat = xgb.DMatrix(X[self.feature_cols])
        elif isinstance(X, np.ndarray):
            X_mat = xgb.DMatrix(X)
        elif isinstance(X, xgb.DMatrix):
            X_mat = X
        else:
            raise TypeError(f"无法识别的输入类型: {type(X)}")
        return self.model.predict(X_mat)

    def save_model(self, path: Union[str, Path]):
        """
        保存模型和相关参数
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 保存模型
        model_file = str(path.with_suffix(".bin"))
        self.model.save_model(model_file)
        # 保存模型元信息
        meta = {
            "targets": self.targets,
            "metrics": self.metrics,
            "feature_cols": self.feature_cols,
            "best_params": self.best_params,
            "random_state": self.random_state,
        }
        meta_file = str(path.with_suffix(".json"))
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_model(cls, path: Union[str, Path]):
        """
        加载类
        """
        path = Path(path)
        meta_file = str(path.with_suffix(".json"))
        model_file = str(path.with_suffix(".bin"))
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 实例化
        obj = cls(
            feature_cols=meta["feature_cols"], random_state=meta.get("random_state", 42)
        )
        obj.best_params = meta.get("best_params")
        obj.targets = meta.get("targets")
        obj.metrics = meta.get("metrics")
        # 加载模型
        obj.model = xgb.Booster()
        obj.model.load_model(model_file)
        return obj
