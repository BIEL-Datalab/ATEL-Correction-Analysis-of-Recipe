import numpy as np
import pandas as pd
import warnings
import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.integration import CatBoostPruningCallback
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import root_mean_squared_error, r2_score

from src.core.models.base import BaseRegressor
from src.core.constants.data_constants import (
    ACT_RATE_NM_SEC_COLS,
    RATE_COEF_COLS,
    BATCH_NUMBER_COLS,
    MACHINE_COLS,
)
from src.core.constants.train_constants import (
    RANDOM_STATE,
    TREE_DEFAULT_NUM_BOOST_ROUND,
    TREE_DEFAULT_EARLY_STOPPING_ROUND,
    TREE_DEFAULT_N_TRIALS,
    TREE_N_JOBS,
)

warnings.filterwarnings("ignore", category=ExperimentalWarning)


class CatBoostRegressorModel(BaseRegressor):
    MISSING_TOKEN = "missing"
    DEFAULT_NUM_COLS = ACT_RATE_NM_SEC_COLS + RATE_COEF_COLS + BATCH_NUMBER_COLS
    DEFAULT_CAT_COLS = MACHINE_COLS
    DEFAULT_DATE_COLS = []

    def __init__(
        self,
        num_cols: List[str] = None,
        cat_cols: List[str] = None,
        iterations: Optional[int] = None,
        early_stopping_rounds: Optional[int] = None,
        n_trials: Optional[int] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        use_gpu: bool = False,
    ):
        """
        基于 CatBoost 的单目标回归模型封装类。
        """
        self.num_cols = num_cols or self.DEFAULT_NUM_COLS
        self.cat_cols = cat_cols or self.DEFAULT_CAT_COLS
        random_state = random_state or RANDOM_STATE
        super().__init__(
            feature_cols=self.num_cols + self.cat_cols,
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

        self.best_params: Dict[str, Any] | None = None
        self.model: CatBoostRegressor | None = None
        self.target_cols: List[str] | None = None
        # 未训练时 metrics 为 None，避免访问未初始化属性
        self.metrics: Dict[str, float] | None = None

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
        plot_file: Optional[Union[Path, str]] = None,
    ):
        if len(target_cols) == 0:
            raise ValueError("target_cols 长度不能为0")
        elif len(target_cols) >= 1:
            if len(target_cols) > 1:
                warnings.warn(
                    f"CatBoostRegressor 只支持单目标回归，已自动使用第一个目标: '{target_cols[0]}'",
                    UserWarning,
                )
            target = target_cols[0]
            self.target_cols = [target]
        self._optuna_search(train, valid, target)
        params = self._base_params()
        params.update(self.best_params)
        self.best_params = params
        train_pool = self._build_pool(train, target)
        valid_pool = self._build_pool(valid, target)
        self.model = CatBoostRegressor(**self.best_params)
        self.model.fit(
            train_pool,
            eval_set=valid_pool,
            verbose=self.iterations // 10,
            use_best_model=True,
            plot_file=Path(plot_file) if plot_file else None,
        )
        # 结果评估
        y_train = train[target].copy()
        y_valid = valid[target].copy()
        train_preds = self.model.predict(train[self.feature_cols])
        valid_preds = self.model.predict(valid[self.feature_cols])
        self.metrics = {
            "Train_R2": r2_score(y_train, train_preds),
            "Valid_R2": r2_score(y_valid, valid_preds),
            "Valid_RMSE": root_mean_squared_error(y_valid, valid_preds),
        }
        return self.model, self.metrics

    def predict(self, X: pd.DataFrame):
        if self.model is None:
            raise Exception("当前未训练模型，无法预测")
        return self.model.predict(X[self.feature_cols])

    def save_model(self, save_dir: Union[str, Path]):
        """
        保存类
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        target = self.target_cols[0]
        model_path = save_dir / f"{target}.cbm"
        meta_path = save_dir / f"{target}.pkl"
        self.model.save_model(model_path)
        meta = dict(
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            best_params=self.best_params,
            metrics=self.metrics,
            random_state=self.random_state,
        )
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load_model(cls, model_dir: Union[str, Path], model_name: str):
        """
        加载类
        """
        model_path = Path(model_dir) / f"{model_name}.cbm"
        meta_path = Path(model_dir) / f"{model_name}.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        obj = cls(
            num_cols=meta["num_cols"],
            cat_cols=meta["cat_cols"],
            random_state=meta["random_state"],
        )
        obj.target_cols = meta["target_cols"]
        obj.best_params = meta["best_params"]
        obj.metrics = meta["metrics"]
        model = CatBoostRegressor()
        model.load_model(model_path)
        obj.model = model
        return obj

    def _base_params(self):
        """
        基础参数
        """
        params = dict(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=self.random_state,
            thread_count=3,
            iterations=self.iterations,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        if self.use_gpu:
            params.update(
                dict(
                    task_type="GPU",
                    devices="0",
                )
            )
        else:
            params.update(
                dict(
                    task_type="CPU",
                    thread_count=3,
                )
            )
        return params

    def _build_pool(self, df: pd.DataFrame, target: str):
        """ """
        X = df[self.feature_cols]
        y = df[target]
        cat_idx = [X.columns.get_loc(c) for c in self.cat_cols]
        return Pool(data=X, label=y, cat_features=cat_idx)

    def _optuna_search(self, train: pd.DataFrame, valid: pd.DataFrame, target: str):
        """
        使用 optuna 进行超参数进行搜索，优化目标为最小化 RMSE。
        """
        train_pool = self._build_pool(train, target)
        valid_pool = self._build_pool(valid, target)

        def objective(trial):
            params = self._base_params()
            params.update(
                dict(
                    # optuna 在 gpu 上不支持 callback
                    task_type="CPU",
                    thread_count=6,
                    devices=None,
                    depth=trial.suggest_int("depth", 4, 12),
                    learning_rate=trial.suggest_float(
                        "learning_rate", 1e-4, 0.5, log=True
                    ),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10),
                    bagging_temperature=trial.suggest_float(
                        "bagging_temperature", 0, 5
                    ),
                    random_strength=trial.suggest_float("random_strength", 0, 5),
                    border_count=trial.suggest_int("border_count", 32, 255),
                    grow_policy=trial.suggest_categorical(
                        "grow_policy",
                        [
                            "SymmetricTree",
                            "Depthwise",
                        ],
                    ),
                )
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
            return root_mean_squared_error(valid[target], pred)

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
