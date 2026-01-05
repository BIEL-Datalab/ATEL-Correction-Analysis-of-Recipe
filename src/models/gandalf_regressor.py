import gc
import logging
import json
import pandas as pd
import optuna
import torch
import omegaconf
import typing
import sys
import os
import collections
from typing import List, Tuple, Dict, Any, Literal, Union, Optional
from pathlib import Path
from contextlib import contextmanager
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import GANDALFConfig

from src.models.base import BaseRegressor

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


@contextmanager
def suppress_output():
    """
    同时抑制 stdout 和 stderr 的上下文管理器
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class GandalfRegressor(BaseRegressor):
    def __init__(
        self,
        categorical_cols: List[str],
        continuous_cols: List[str],
        batch_size: int = 256,
        max_epochs: int = 100,
        date_cols: List[Tuple[str, str, str]] | None = None,
        random_state=42,
        n_trials: int = 100,
        checkpoint_dir: str = "gandalf_checkpoints",
        save_dir: Optional[str] = None,
        acc: Literal["gpu", "cpu"] = "gpu",
    ):
        """ """
        self.date_cols = date_cols or []
        date_feature_names = [d[0] for d in self.date_cols]
        feature_cols = categorical_cols + continuous_cols + date_feature_names

        super().__init__(feature_cols, random_state, multi_target=True)

        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.targets: List[str] | None = None

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_trials = n_trials
        self.checkpoint_dir = checkpoint_dir
        self.save_dir = save_dir
        self.acc = "gpu" if acc == "gpu" and torch.cuda.is_available() else "cpu"

        # 结果相关
        self.optimizer_config = OptimizerConfig(optimizer="AdamW")
        self.model: TabularModel | None = None
        self.best_params: Dict[str, float] | None = None
        self.metrics: Dict[str, float] | None = None

    def fit(
        self, train: pd.DataFrame, valid: pd.DataFrame, targets: List[str]
    ) -> Tuple[TabularModel, Dict[str, Any]]:
        """
        训练 GANDALF 模型
        """
        self.targets = targets
        # 搜索参数
        self._optuna_search(train, valid)
        self._set_tabular_logging(logging.INFO)
        # 最优参数训练
        d_cfg, t_cfg, m_cfg = self._build_configs(self.best_params, is_tuning=False)
        self.model = TabularModel(
            data_config=d_cfg,
            trainer_config=t_cfg,
            model_config=m_cfg,
            optimizer_config=self.optimizer_config,
        )
        self.model.fit(train, valid)
        # 结果评估
        eval_metrics = self.model.evaluate(valid)
        self.metrics = {k: v for d in eval_metrics for k, v in d.items()}
        # 保存模型
        if self.save_dir:
            self.model.save_model(self.save_dir)
            meta = {
                "targets": self.targets,
                "metrics": self.metrics,
                "feature_cols": self.feature_cols,
                "categorical_cols": self.categorical_cols,
                "continuous_cols": self.continuous_cols,
                "date_cols": self.date_cols,
                "best_params": self.best_params,
                "random_state": self.random_state,
            }
            meta_file = self.save_dir / "meta.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)
        return self.model, self.metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.model:
            raise RuntimeError("模型还未训练")
        return self.model.predict(X)

    @classmethod
    def load_model(cls, save_dir: Union[str, Path]):
        """
        加载模型
        """
        save_dir = Path(save_dir)
        meta_file = save_dir / "meta.json"
        # 读取 meta
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 创建实例
        obj = cls(
            categorical_cols=meta["categorical_cols"],
            continuous_cols=meta["continuous_cols"],
            date_cols=meta.get("date_cols", None),
            random_state=meta.get("random_state", 42),
        )
        obj.targets = meta.get("targets")
        obj.metrics = meta.get("metrics")
        obj.feature_cols = meta.get("feature_cols")
        obj.best_params = meta.get("best_params")
        # 加载模型
        obj.model = TabularModel.load_model(save_dir)
        return obj

    def _set_tabular_logging(self, level: int):
        """ """
        logger_names = [
            "pytorch_tabular",
            "pytorch_tabular.tabular_model",
            "pytorch_tabular.tabular_datamodule",
            "pytorch_tabular.models",
            "pytorch_tabular.models.gandalf.gandalf",
            "lightning.pytorch",
            "pytorch_lightning",
            "lightning.pytorch.utilities.rank_zero",
            "lightning.pytorch.accelerators.cuda",
        ]
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

    def _build_configs(self, params: Dict[str, Any], is_tuning: bool = True):
        """
        构建参数
        """
        use_gpu = torch.cuda.is_available() and self.acc == "gpu"
        self.acc == "gpu" if use_gpu else "cpu"
        data_config = DataConfig(
            target=self.targets,
            continuous_cols=self.continuous_cols,
            categorical_cols=self.categorical_cols,
            date_columns=self.date_cols,
        )
        trainer_config = TrainerConfig(
            accelerator=self.acc,
            devices=1,
            batch_size=self.batch_size,
            max_epochs=min(15, self.max_epochs) if is_tuning else self.max_epochs,
            early_stopping_patience=3,
            seed=self.random_state,
            checkpoints=None if is_tuning else "valid_loss",
            checkpoints_path=self.checkpoint_dir,
            progress_bar="none" if is_tuning else "simple",
            load_best=False if is_tuning else True,
            trainer_kwargs=(
                {
                    "enable_model_summary": False,
                    "enable_checkpointing": False,
                }
                if is_tuning
                else {}
            ),
        )

        model_config = GANDALFConfig(
            task="regression",
            learning_rate=params.get("learning_rate", 1e-3),
            gflu_stages=params["gflu_stages"],
            gflu_dropout=params["gflu_dropout"],
            gflu_feature_init_sparsity=params["gflu_feature_init_sparsity"],
            batch_norm_continuous_input=not is_tuning,
            target_range=None,
            seed=self.random_state,
        )

        return data_config, trainer_config, model_config

    def _optuna_search(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
    ):
        """
        使用 optuna 对 GANDALF 超参数进行搜索，优化目标为最小化 RMSE。
        """
        self._set_tabular_logging(logging.ERROR)

        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
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
                # 清理显存
                del model
                gc.collect()
                torch.cuda.empty_cache()
            return score

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
            n_jobs=1,
            show_progress_bar=True,
        )
        self.best_params = study.best_params
