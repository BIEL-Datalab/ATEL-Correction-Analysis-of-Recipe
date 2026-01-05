import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import shap
from typing import List, Optional, Union, Any, Dict, Tuple

from src.models.base import BaseRegressor


class RegressionEvaluator:
    """
    回归模型评估器
    """

    def __init__(
        self,
        feature_cols: List[str],
        output_dir: Optional[Union[Path, str]],
        date_cols: Optional[List[str]] = None,
        eps: float = 1e-8,
    ):
        """
        Args:
        - feature_cols(List[str]): 特征列名列表
        - output_dir(Optional[Union[Path, str]]): 结果保存路径，为 None 则不保存
        - eps(float): 用于计算相对误差的极小值，防止除零
        """
        self.feature_cols = feature_cols
        self.date_cols = date_cols or []
        self.eps = eps
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

    def _to_numpy(
        self,
        y: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        target_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        统一转换为 (n_samples, n_targets) 的 ndarray
        """
        if isinstance(y, list):
            return np.column_stack(y)
        if isinstance(y, pd.DataFrame):
            if target_cols:
                return y[target_cols].values
            return y.values
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                return y.reshape(-1, 1)
            return y
        raise TypeError(f"不支持的 y 类型： {type(y)}")

    def _ensure_shap_dir(self) -> Path:
        if not self.output_dir:
            raise RuntimeError("output_dir 为空")
        shap_dir = self.output_dir / "shap"
        shap_dir.mkdir(exist_ok=True)
        return shap_dir

    def _build_tabular_predict_func(
        self,
        model: BaseRegressor,
        target_cols: List[str],
    ):
        """
        为 pytorch_tabular 多目标模型构造 SHAP 所需的 predict 函数
        """

        def predict_func(X_np: np.ndarray) -> np.ndarray:
            X_df = pd.DataFrame(X_np, columns=self.feature_cols)
            preds_df = model.model.predict(X_df)
            pred_cols = [f"{t}_prediction" for t in target_cols]
            return preds_df[pred_cols].values

        return predict_func

    def _shap_multi_model_single_target(
        self,
        models: Dict[str, BaseRegressor],
        X_val_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """
        多个单目标模型输出 shap 值

        Args:
        - models(Dict[str, BaseRegressor])
        - X_val_df(pd.DataFrame):
        """
        shap_results = {}
        for tgt, model in models.items():
            explainer = shap.Explainer(model.model)
            sv = explainer(X_val_df).values
            shap_results[tgt] = {"shap_values": sv, "X": X_val_df}
        return shap_results

    def _shap_single_model_multi_target(
        self,
        model: BaseRegressor,
        X_val_df: pd.DataFrame,
        target_cols: List[str],
        bg_sample: int,
        nsamples: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        多目标模型输出 shap 值

        Args:
        -
        -
        -
        """
        assert model.multi_target, "多目标模型 multi_target 必须为 True"
        predict_func = self._build_tabular_predict_func(model, target_cols)
        background = shap.sample(X_val_df, bg_sample)
        explainer = shap.KernelExplainer(
            predict_func,
            background.values,
        )
        shap_values = explainer.shap_values(X_val_df.values, nsamples=nsamples)
        shap_results = {}
        for i, tgt in enumerate(target_cols):
            sv = (
                shap_values[i]
                if isinstance(shap_values, list)
                else shap_values[:, :, i]
            )
            shap_results[tgt] = {"shap_values": sv, "X": X_val_df}
        return shap_results

    def _plot_shap_subplots(
        self,
        shap_results: Dict[str, Dict[str, Any]],
        max_display: int,
        shap_dir: Optional[Union[str, Path]] = None,
    ):
        """ """
        for i, (tgt, data) in enumerate(shap_results.items()):
            sv = data["shap_values"]
            X_df = data["X"].copy()
            # 转换日期
            for col in self.date_cols:
                if col in X_df.columns:
                    X_df[col] = pd.to_datetime(X_df[col]).apply(
                        lambda x: x.toordinal() if pd.notnull(x) else 0
                    )
            plt.figure(figsize=(20, 12))
            shap.summary_plot(sv, X_df, max_display=max_display, show=False)
            plt.title(f"SHAP Summary | {tgt}")
            if shap_dir:
                plt.savefig(
                    shap_dir / f"shap_summary_{tgt}.png", dpi=150, bbox_inches="tight"
                )
            plt.close()

    def _save_shap_to_excel(
        self,
        shap_results: Dict[str, Dict[str, Any]],
        filename: str = "shap_values.xlsx",
    ):
        """
        保存所有的 shap 值结果
        """
        if not self.output_dir:
            return
        excel_path = self.output_dir / filename
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for tgt, data in shap_results.items():
                sv = data["shap_values"]
                X_df = data["X"]
                df = pd.DataFrame(
                    {
                        "feature": X_df.columns,
                        "mean_abs_shap": np.abs(sv).mean(axis=0),
                        "mean_shap": sv.mean(axis=0),
                    }
                ).sort_values("mean_abs_shap", ascending=False)
                df.to_excel(writer, sheet_name=tgt[:31], index=False)

    def compute_metrics(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        y_pred: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        target_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        计算回归指标
        """
        y_true = self._to_numpy(y_true, target_cols)
        y_pred = self._to_numpy(y_pred, target_cols)
        assert y_true.shape == y_pred.shape, "y_true 与 y_pred 形状不匹配"
        if not target_cols:
            target_cols = [f"target_{i}" for i in range(y_true.shape[1])]

        records = []
        for i, tgt in enumerate(target_cols):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            rel_err = np.abs(yt - yp) / (np.abs(yt) + self.eps)
            records.append(
                {
                    "target": tgt,
                    "RMSE": float(root_mean_squared_error(yt, yp)),
                    "R2": float(r2_score(yt, yp)),
                    "MARE": float(np.mean(rel_err)),
                    "MAE": float(mean_absolute_error(yt, yp)),
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
        target_cols: List[str],
        filename: str = "multi_target_diagnostics.png",
    ):
        """
        生成预测 vs 真实值、误差分布的诊断图
        """
        y_true = self._to_numpy(y_true, target_cols)
        y_pred = self._to_numpy(y_pred, target_cols)
        assert y_true.shape == y_pred.shape, "y_true 与 y_pred 形状不匹配"

        n_targets = y_true.shape[1]
        fig, axes = plt.subplots(
            n_targets,
            3,
            figsize=(36, 8 * n_targets),
            squeeze=False,
        )
        for i, tgt in enumerate(target_cols):
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
        target_cols: List[str],
        max_display: int = 15,
        bg_sample: int = 100,
        nsamples: int = 200,
    ):
        """
        SHAP 值总结，支持单模型多目标 和 多单目标模型

        Args:
        - models: 单个模型对象(多目标) 或 {target_name: model} 字典 (多个单目标模型)
        -
        """
        assert len(target_cols) >= 1, "targets 不能为空"
        X_val_df = (
            X_val
            if isinstance(X_val, pd.DataFrame)
            else pd.DataFrame(X_val, columns=self.feature_cols)
        )
        if isinstance(models, dict):
            shap_results = self._shap_multi_model_single_target(
                models=models, X_val_df=X_val_df
            )
        else:
            shap_results = self._shap_single_model_multi_target(
                model=models,
                X_val_df=X_val_df,
                target_cols=target_cols,
                bg_sample=min(bg_sample, len(X_val_df)),
                nsamples=min(nsamples, len(X_val_df)),
            )
        if self.output_dir:
            shap_dir = self._ensure_shap_dir()
            self._plot_shap_subplots(shap_results, max_display, shap_dir)
            self._save_shap_to_excel(shap_results)

    def run_full_evaluation(
        self,
        models: Union[BaseRegressor, Dict[str, BaseRegressor]],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.DataFrame, np.ndarray],
        target_cols: List[str],
    ):
        """
        完成指标计算、真实值vs预测值分析、Shap 值特征重要性分析
        """
        single_model = not isinstance(models, dict)

        if single_model:
            assert isinstance(models, BaseRegressor), "models 参数不合法"
            assert len(target_cols) == 1 or getattr(
                models, "multi_target", False
            ), "该模型不支持多目标预测，但传入的目标变量个数大于1"
            models_list = [models]
        else:
            models_list = [models[tgt] for tgt in target_cols]
            assert len(models_list) == len(
                target_cols
            ), "单变量回归模型需要模型个数与目标变量个数一致"

        y_pred = [model.predict(X_val) for model in models_list]
        self.compute_metrics(y_true, y_pred, target_cols)
        self.plot_multi_target_diagnostics(y_true, y_pred, target_cols)
        self.shap_summary_all_targets(models, X_val, target_cols)
