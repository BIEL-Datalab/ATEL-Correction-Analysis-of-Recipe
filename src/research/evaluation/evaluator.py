import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgbm
import catboost
from joblib import Parallel, delayed
from pathlib import Path
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import shap
from typing import List, Optional, Union, Any, Dict, Tuple, Literal

from src.models.base import BaseRegressor
from src.constants.eval_constants import (
    EPS,
    BG_DEFAULT_SAMPLE,
    NSAMPLES_DEFAULT,
    VAL_DEFAULT_SAMPLE,
    N_JOBS_DEFAULT,
    MAX_DISPLAY,
)


class RegressionEvaluator:
    """
    回归模型评估器
    """

    def __init__(
        self,
        num_cols: List[str],
        cat_cols: List[str],
        output_dir: Optional[Union[Path, str]],
        date_cols: Optional[List[str]] = None,
        eps: float = EPS,
        random_state: int = 42,
    ):
        """
        Args:
        -
        - output_dir(Optional[Union[Path, str]]): 结果保存路径，为 None 则不保存
        - eps(float): 用于计算相对误差的极小值，防止除零
        """
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.date_cols = date_cols or []
        self.feature_cols = num_cols + cat_cols + self.date_cols
        self.eps = eps
        self.random_state = random_state
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

    def _convert_date_cols(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        转换时间特征
        """
        X = X_df.copy()
        for col in self.date_cols:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col]).apply(
                    lambda x: x.toordinal() if pd.notnull(x) else 0
                )
        return X

    def _sample_valid_set(
        self,
        X_val_df: pd.DataFrame,
        min_samples: int = 500,
    ) -> pd.DataFrame:
        """
        验证集抽样(至少抽取三分之一的样本)
        """
        n_samples = max(min_samples, len(X_val_df) // 3)
        n_samples = min(len(X_val_df), n_samples)
        return X_val_df.sample(n=n_samples, random_state=self.random_state)

    def _is_tree_model(self, model):
        """
        判断是否为树模型
        """
        return isinstance(
            model, (xgb.Booster, lgbm.Booster, catboost.CatBoostRegressor)
        )

    def _shap_compute_general(
        self,
        regressors: Union[BaseRegressor, Dict[str, BaseRegressor]],
        X_val_df: pd.DataFrame,
        bg_sample: int = BG_DEFAULT_SAMPLE,
        nsamples: int = NSAMPLES_DEFAULT,
        val_sample: int = VAL_DEFAULT_SAMPLE,
        n_jobs: int = N_JOBS_DEFAULT,
    ) -> Dict[str, Dict[str, Any]]:
        """
        通用 SHAP 值计算函数

        Args:
        -

        Returns:
        -
        """
        X_sample = self._sample_valid_set(X_val_df, val_sample)
        if isinstance(regressors, dict):
            # 多个单目标树模型才以 dict 格式传入
            for regressor in regressors.values():
                if (
                    not self._is_tree_model(regressor.model)
                    and not regressor.multi_target
                ):
                    raise ValueError(
                        "只有 XGB / LGBM / Catboost 支持多个单目标模型 shap 值计算"
                    )

            def compute_tree_shap(regressor: BaseRegressor):
                explainer = shap.TreeExplainer(regressor.model)
                sv = explainer.shap_values(X_sample)
                if isinstance(sv, list):
                    sv = sv[0]
                return sv

            shap_values_list = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(compute_tree_shap)(regressor)
                for regressor in regressors.values()
            )
            shap_results = {
                tgt: {"shap_values": sv, "X": X_sample}
                for tgt, sv in zip(regressors.keys(), shap_values_list)
            }
            return shap_results
        elif isinstance(regressors, BaseRegressor) and regressors.multi_target:
            model = regressors
            target_cols = getattr(regressors, "target_cols", None)
            if target_cols is None:
                raise ValueError("多目标模型必须提供 target_cols 属性")

            # 定义多目标模型预测函数
            def predict_func(X_np: np.ndarray) -> np.ndarray:
                """
                为 pytorch_tabular 多目标模型构造 SHAP 所需的 predict 函数
                """
                X_df = pd.DataFrame(X_np, columns=self.feature_cols)
                preds_df = regressors.model.predict(X_df)
                pred_cols = [f"{t}_prediction" for t in target_cols]
                return preds_df[pred_cols].values

            bg_df = X_sample.sample(
                min(bg_sample, len(X_sample)), random_state=self.random_state
            )
            explainer = shap.KernelExplainer(
                predict_func,
                bg_df.values,
            )
            shap_values = explainer.shap_values(
                X_sample.values, nsamples=min(nsamples, len(X_sample))
            )
            if isinstance(shap_values, list):
                shap_values = np.array(
                    shap_values
                )  # (n_targets, n_samples, n_features)
                shap_values = np.transpose(shap_values, (1, 2, 0))
            return {
                tgt: {
                    "shap_values": (
                        shap_values[:, :, i] if shap_values.ndim == 3 else shap_values
                    ),
                    "X": X_sample,
                }
                for i, tgt in enumerate(target_cols)
            }
        else:
            raise ValueError("regressors 类型不合法")

    def _plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X_df: pd.DataFrame,
        target: str,
        max_display: int,
        shap_dir: Optional[Path],
    ):
        """
        绘制 SHAP 总结图

        Args:
        -

        """
        plt.figure(figsize=(20, 12))
        shap.summary_plot(shap_values, X_df, max_display=max_display, show=False)
        plt.title(f"SHAP Summary | {target}")
        if shap_dir:
            plt.savefig(
                shap_dir / f"shap_summary_{target}.png", dpi=150, bbox_inches="tight"
            )
        plt.close()

    def _plot_cat_shap_direction(
        self,
        shap_values: np.ndarray,
        X_df: pd.DataFrame,
        target: str,
        shap_dir: Optional[Path],
    ):
        """
        分类特征预测贡献图

        Args:
        -

        """
        for col in self.cat_cols:
            if col not in X_df.columns:
                continue
            col_idx = X_df.columns.get_loc(col)
            df = pd.DataFrame(
                {
                    "category": X_df[col].astype(str),
                    "shap": shap_values[:, col_idx],
                }
            )
            stat = (
                df.groupby("category")["shap"]
                .mean()
                .reset_index()
                .rename(columns={"shap": "mean_shap"})
                .sort_values("mean_shap", ascending=False)
            )
            plt.figure(figsize=(8, 4))
            colors = stat["mean_shap"].apply(
                lambda x: "tab:red" if x > 0 else "tab:blue"
            )
            plt.bar(stat["category"], stat["mean_shap"], color=colors)
            plt.axhline(0, linestyle="--", linewidth=1)
            plt.title(f"{target} | Mean SHAP by category: {col}")
            plt.ylabel("Mean SHAP (direction)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            if shap_dir:
                plt.savefig(
                    shap_dir / f"shap_cat_direction_{target}_{col}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
            plt.close()

    def _plot_shap_explanation(
        self,
        shap_results: Dict[str, Dict[str, Any]],
        max_display: int,
        shap_dir: Optional[Union[str, Path]] = None,
    ):
        """
        绘制 shap 解释图：
        - 全特征 SHAP 总结图
        - 分类特征预测贡献图
        """
        for tgt, data in shap_results.items():
            sv = data["shap_values"]
            X_df = self._convert_date_cols(data["X"])
            # 全特征 SHAP 总结图
            self._plot_shap_summary(sv, X_df, tgt, max_display, shap_dir)
            # 分类特征预测贡献图
            self._plot_cat_shap_direction(sv, X_df, tgt, shap_dir)

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
        train_y_true: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        train_y_pred: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        valid_y_true: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        valid_y_pred: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        target_cols: Optional[List[str]] = None,
        n_digits: int = 4,
    ) -> pd.DataFrame:
        """
        计算回归指标

        Args:
        -

        Returns:
        -
        """
        train_y_true = self._to_numpy(train_y_true, target_cols)
        train_y_pred = self._to_numpy(train_y_pred, target_cols)
        valid_y_true = self._to_numpy(valid_y_true, target_cols)
        valid_y_pred = self._to_numpy(valid_y_pred, target_cols)
        assert (
            train_y_true.shape == train_y_pred.shape
        ), f"训练集 y_true 与 y_pred 形状不匹配，{train_y_true.shape} != {train_y_pred.shape}"
        assert (
            valid_y_true.shape == valid_y_pred.shape
        ), f"验证集 y_true 与 y_pred 形状不匹配，{valid_y_true.shape} != {valid_y_pred.shape}"
        if not target_cols:
            target_cols = [f"target_{i}" for i in range(valid_y_true.shape[1])]

        records = []
        for i, tgt in enumerate(target_cols):
            t_yt = train_y_true[:, i]
            t_yp = train_y_pred[:, i]
            v_yt = valid_y_true[:, i]
            v_yp = valid_y_pred[:, i]
            v_rel_err = np.abs(v_yt - v_yp) / (np.abs(v_yt) + self.eps)
            records.append(
                {
                    "Target": tgt,
                    "Train_R2": round(r2_score(t_yt, t_yp), n_digits),
                    "Valid_R2": round(r2_score(v_yt, v_yp), n_digits),
                    "Valid_RMSE": round(root_mean_squared_error(v_yt, v_yp), n_digits),
                    "Valid_MARE": round(np.mean(v_rel_err), n_digits),
                    "Valid_MAE": round(mean_absolute_error(v_yt, v_yp), n_digits),
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
        regressors: Union[BaseRegressor, Dict[str, BaseRegressor]],
        valid: Union[pd.DataFrame, np.ndarray],
        target_cols: List[str],
        max_display: int = MAX_DISPLAY,
        bg_sample: int = BG_DEFAULT_SAMPLE,
        nsamples: int = NSAMPLES_DEFAULT,
        val_sample: int = VAL_DEFAULT_SAMPLE,
        n_jobs: int = N_JOBS_DEFAULT,
    ):
        """
        SHAP 值总结，支持单模型多目标 和 多单目标模型

        Args:
        - models: 单个模型对象(多目标) 或 {target_name: model} 字典 (多个单目标模型)
        -
        """
        assert len(target_cols) >= 1, "target_cols 不能为空"
        X_val_df = (
            valid
            if isinstance(valid, pd.DataFrame)
            else pd.DataFrame(valid, columns=self.feature_cols)
        )
        shap_results = self._shap_compute_general(
            regressors=regressors,
            X_val_df=X_val_df,
            bg_sample=bg_sample,
            nsamples=nsamples,
            val_sample=val_sample,
            n_jobs=n_jobs,
        )
        if self.output_dir:
            shap_dir = self._ensure_shap_dir()
            self._plot_shap_explanation(shap_results, max_display, shap_dir)
            self._save_shap_to_excel(shap_results)

    def run_full_evaluation(
        self,
        regressors: Union[BaseRegressor, Dict[str, BaseRegressor]],
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
    ):
        """
        完成指标计算、真实值vs预测值分析、Shap 值特征重要性分析
        """
        single_model = not isinstance(regressors, dict)

        if single_model:
            assert isinstance(regressors, BaseRegressor), "models 参数不合法"
            assert len(target_cols) == 1 or getattr(
                regressors, "multi_target", False
            ), "该模型不支持多目标预测，但传入的目标变量个数大于1"
            models_list = [regressors]
        else:
            models_list = [regressors[tgt] for tgt in target_cols]
            assert len(models_list) == len(
                target_cols
            ), "单变量回归模型需要模型个数与目标变量个数一致"
        train_y_true = train[target_cols]
        train_y_pred = [model.predict(train) for model in models_list]
        valid_y_true = valid[target_cols]
        valid_y_pred = [model.predict(valid) for model in models_list]
        # 指标
        self.compute_metrics(
            train_y_true, train_y_pred, valid_y_true, valid_y_pred, target_cols
        )
        # 诊断图
        self.plot_multi_target_diagnostics(valid_y_true, valid_y_pred, target_cols)
        # SHAP
        self.shap_summary_all_targets(regressors, valid[self.feature_cols], target_cols)
