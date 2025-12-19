import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import shap
from typing import List, Optional, Union, Any, Dict

from src.models.base import BaseRegressor


class RegressionEvaluator:
    """
    回归模型评估器
    """

    def __init__(
        self, feature_cols: List[str], output_dir: Optional[Union[Path, str]], eps=1e-8
    ):
        """
        args:
        - feature_cols(List[str]): 模型使用
        """
        self.feature_cols = feature_cols
        self.eps = eps
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

    def _to_numpy(
        self,
        y: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        target_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        统一转换为 (n_samples, n_targets) 的 ndarray
        """
        if isinstance(y, list):
            return np.column_stack(y)
        if isinstance(y, pd.DataFrame):
            if target_names:
                return y[target_names].values
            return y.values
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                return y.reshape(-1, 1)
            return y
        raise TypeError(f"不支持的 y 类型： {type(y)}")

    def _save_single_target_shap(
        self,
        target_name: str,
        shap_value: np.ndarray,
        X_val_np: np.ndarray,
        max_display: int,
    ):
        """
        单个 target 的 SHAP 值输出
        """
        if not self.output_dir:
            return
        target_dir = self.output_dir / target_name
        target_dir.mkdir(parents=True, exist_ok=True)
        # SHAP 值
        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_value,
            X_val_np,
            feature_names=self.feature_cols,
            show=False,
            max_display=min(max_display, len(self.feature_cols)),
        )
        plt.title(f"Shap Value of {target_name}")
        plt.tight_layout()
        plt.savefig(target_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        # 特征重要性
        mean_abs_shap = np.mean(np.abs(shap_value), axis=0)
        importance_df = pd.DataFrame(
            {"feature": self.feature_cols, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)
        importance_df.to_csv(target_dir / "shap_importance.csv", index=False)

    def compute_metrics(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        y_pred: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        target_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        计算回归指标
        """
        y_true = self._to_numpy(y_true, target_names)
        y_pred = self._to_numpy(y_pred, target_names)
        assert y_true.shape == y_pred.shape, "y_true 与 y_pred 形状不匹配"
        n_targets = y_true.shape[1]
        if not target_names:
            target_names = [f"target_{i}" for i in range(n_targets)]

        records = []
        for i, tgt in enumerate(target_names):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            rel_err = np.abs(yt - yp) / (np.abs(yt) + self.eps)
            records.append(
                {
                    "target": tgt,
                    "RMSE": float(root_mean_squared_error(yt, yp)),
                    "R2": float(r2_score(yt, yp)),
                    "MARE": float(np.mean(rel_err)),
                    "MAAE": float(mean_absolute_error(yt, yp)),
                }
            )
        metrics_df = pd.DataFrame(records)
        if self.output_dir:
            metrics_df.to_csv(self.output_dir / "metrics.csv", index=False)
        return metrics_df

    def plot_multi_target_diagnostics(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        y_pred: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        target_names: List[str],
        filename: str = "multi_target_diagnostics.png",
    ):
        """
        预测结果可视化
        """
        y_true = self._to_numpy(y_true, target_names)
        y_pred = self._to_numpy(y_pred, target_names)
        assert y_true.shape == y_pred.shape

        n_targets = y_true.shape[1]
        fig, axes = plt.subplots(
            n_targets,
            3,
            figsize=(36, 8 * n_targets),
            squeeze=False,
        )
        for i, tgt in enumerate(target_names):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            abs_err = np.abs(yt - yp)
            rel_err = np.abs(yt - yp) / (np.abs(yt) + self.eps)
            # 预测真实值对比
            ax = axes[i, 0]
            ax.scatter(yt, yp, alpha=0.5)
            min_v = min(yt.min(), yp.min())
            max_v = max(yt.max(), yp.max())
            ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
            ax.set_title(f"{tgt} | Prediction vs Truth")
            ax.set_xlabel("True Value")
            ax.set_ylabel("Predicted Value")
            ax.grid(True)
            # 相对误差图
            ax = axes[i, 1]
            ax.hist(rel_err, bins=50)
            ax.set_title(f"{tgt} | Relative Error")
            ax.set_xlabel("|y_true - y_pred| / |y_true|")
            ax.set_ylabel("Count")
            ax.grid(True)
            # 绝对误差图
            ax = axes[i, 2]
            ax.hist(abs_err, bins=50)
            ax.set_title(f"{tgt} | Absolute Error")
            ax.set_xlabel("|y_true - y_pred|")
            ax.set_ylabel("Count")
            ax.grid(True)

        plt.tight_layout()
        if self.output_dir:
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def shap_summary_all_targets(
        self,
        models: Union[BaseRegressor, Dict[str, BaseRegressor]],
        X_val: Union[pd.DataFrame, np.ndarray],
        target_names: Optional[List[str]],
        max_display: int = 15,
    ):
        """
        SHAP 值总结

        Args:
        - models: 以 model.multi_target 判断单变量回归还是多变量回归。
            - 单模型传入 model
            - 多模型传入字典 {"target": model}
        """
        X_val_np = self._to_numpy(X_val, self.feature_cols)
        single_model = not isinstance(models, dict)
        n_targets = len(target_names)

        if single_model:
            assert isinstance(models, BaseRegressor), "models 参数不合法。"
            model = models
            explainer = shap.Explainer(model.model)
            if n_targets == 1:
                sv = explainer(X_val_np).values
                self._save_single_target_shap(
                    target_names[0], sv, X_val_np, max_display
                )
            else:
                assert getattr(
                    model, "multi_target", False
                ), "单模型预测多目标预测时，model.multi_target 必须为 True"
                sv_all = explainer(X_val_np).values
                for i, tgt in enumerate(target_names):
                    sv = sv_all[:, :, i]
                    self._save_single_target_shap(tgt, sv, X_val_np, max_display)
        else:
            for tgt in target_names:
                model = models[tgt]
                assert isinstance(model, BaseRegressor), "models 参数不合法。"
                explainer = shap.Explainer(model.model)
                sv = explainer(X_val_np).values
                self._save_single_target_shap(tgt, sv, X_val_np, max_display)

    def run_full_evaluation(
        self,
        models: Union[BaseRegressor, Dict[str, BaseRegressor]],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.DataFrame, np.ndarray],
        target_names: List[str],
    ):
        """
        完成指标计算、真实值vs预测值分析、Shap 值特征重要性分析
        """
        single_model = not isinstance(models, dict)

        if single_model:
            assert isinstance(models, BaseRegressor), "models 参数不合法"
            assert len(target_names) == 1 or getattr(
                models, "multi_target", False
            ), "该模型不支持多目标预测，但传入的目标变量个数大于1"
            models_list = [models]
        else:
            models_list = [models[tgt] for tgt in target_names]
            assert len(models_list) == len(
                target_names
            ), "单变量回归模型需要模型个数与目标变量个数一致"

        y_pred = [model.predict(X_val) for model in models_list]
        self.compute_metrics(y_true, y_pred, target_names)
        self.plot_multi_target_diagnostics(y_true, y_pred, target_names)
        self.shap_summary_all_targets(models, X_val, target_names)
