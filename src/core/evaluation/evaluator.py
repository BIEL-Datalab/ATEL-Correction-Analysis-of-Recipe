"""
回归模型评估器

统一处理单目标树模型（{target: model} 字典）与多目标深度模型（单模型多目标）两类场景，
完成指标计算、预测诊断图、SHAP 特征重要性，以及按机器分组的区间覆盖对比。

设计要点：
  - 模型来源分两类：
      * 单目标：{target_col: BaseRegressor} 字典，每目标一模型
      * 多目标：单个 BaseRegressor（multi_target=True，一次预测全部 target）
    评估器内部统一归一为"按 target 取预测"，屏蔽差异。
  - 区间对比（interval_comparison）：按 machine_sn 分组 × time_index 排序，对每通道输出
    真实/估计的最值/次最值区间与预测均值±3σ区间，用于评估极值区间的覆盖能力。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import catboost
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.core.constants.data_constants import MACHINE_COL, TIME_INDEX_COL
from src.core.constants.eval_constants import (
    BG_DEFAULT_SAMPLE,
    EPS,
    MAX_DISPLAY,
    N_JOBS_DEFAULT,
    NSAMPLES_DEFAULT,
    VAL_DEFAULT_SAMPLE,
)
from src.core.models.base import BaseRegressor

logger = logging.getLogger(__name__)

# 中文字体兜底（matplotlib 中文显示）
plt.rcParams["axes.unicode_minus"] = False


class RegressionEvaluator:
    """回归模型评估器。"""

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
            num_cols / cat_cols / date_cols: 特征列分组（SHAP 诊断用）
            output_dir: 结果保存目录，None 则不落盘
            eps: 相对误差防除零极小值
            random_state: 抽样随机种子
        """
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.date_cols = date_cols or []
        self.feature_cols = num_cols + cat_cols + self.date_cols
        self.eps = eps
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 工具：统一 y 形状 / 模型类型判定 / 预测归一
    # ==================================================================
    @staticmethod
    def _to_numpy(
        y: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        target_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """统一转为 (n_samples, n_targets) 的 ndarray。"""
        if isinstance(y, list):
            return np.column_stack(y)
        if isinstance(y, pd.DataFrame):
            return y[target_cols].values if target_cols else y.values
        if isinstance(y, np.ndarray):
            return y.reshape(-1, 1) if y.ndim == 1 else y
        raise TypeError(f"不支持的 y 类型: {type(y)}")

    @staticmethod
    def _is_tree_model(model: Any) -> bool:
        """判断底层是否为原生树模型（可走 TreeExplainer 快速 SHAP）。"""
        return isinstance(model, (xgb.Booster, lgbm.Booster, catboost.CatBoostRegressor))

    def _collect_predictions(
        self,
        regressors: Union[BaseRegressor, Dict[str, BaseRegressor]],
        df: pd.DataFrame,
        target_cols: List[str],
    ) -> np.ndarray:
        """
        归一化预测：无论单目标字典还是多目标单模型，都返回 (n_samples, n_targets)。

        单目标字典：逐 target 调对应模型 predict，按 target 顺序拼接；
        多目标单模型：一次 predict 返回全部 target，按 target_cols 取列。
        """
        if isinstance(regressors, dict):
            cols = []
            for tgt in target_cols:
                pred = regressors[tgt].predict(df)
                pred = np.asarray(pred).reshape(-1)
                cols.append(pred)
            return np.column_stack(cols)
        # 多目标单模型
        pred_df = regressors.predict(df)
        if isinstance(pred_df, pd.DataFrame):
            return pred_df[target_cols].values
        pred_arr = np.asarray(pred_df)
        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(-1, 1)
        return pred_arr

    # ==================================================================
    # 指标计算
    # ==================================================================
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
        计算逐目标回归指标：R2 / RMSE / MARE / MAE。

        自动屏蔽 y_true 为 NaN 的位置（目标缺失时不计入该目标的指标）。
        """
        train_y_true = self._to_numpy(train_y_true, target_cols)
        train_y_pred = self._to_numpy(train_y_pred, target_cols)
        valid_y_true = self._to_numpy(valid_y_true, target_cols)
        valid_y_pred = self._to_numpy(valid_y_pred, target_cols)
        if not target_cols:
            target_cols = [f"target_{i}" for i in range(valid_y_true.shape[1])]

        records = []
        for i, tgt in enumerate(target_cols):
            t_yt, t_yp = train_y_true[:, i], train_y_pred[:, i]
            v_yt, v_yp = valid_y_true[:, i], valid_y_pred[:, i]
            # 屏蔽缺失（仅有效样本参与指标）
            t_mask = ~np.isnan(t_yt)
            v_mask = ~np.isnan(v_yt)
            rec = {"Target": tgt, "n_valid": int(v_mask.sum())}
            if t_mask.sum() >= 2:
                rec["Train_R2"] = round(r2_score(t_yt[t_mask], t_yp[t_mask]), n_digits)
            if v_mask.sum() >= 2:
                rec["Valid_R2"] = round(r2_score(v_yt[v_mask], v_yp[v_mask]), n_digits)
                rec["Valid_RMSE"] = round(
                    root_mean_squared_error(v_yt[v_mask], v_yp[v_mask]), n_digits
                )
                rec["Valid_MAE"] = round(
                    mean_absolute_error(v_yt[v_mask], v_yp[v_mask]), n_digits
                )
                rel = np.abs(v_yt[v_mask] - v_yp[v_mask]) / (np.abs(v_yt[v_mask]) + self.eps)
                rec["Valid_MARE"] = round(float(np.mean(rel)), n_digits)
            records.append(rec)
        metrics_df = pd.DataFrame(records)
        if self.output_dir:
            metrics_df.to_csv(self.output_dir / "metrics.csv", index=False)
        return metrics_df

    # ==================================================================
    # 预测诊断图
    # ==================================================================
    def plot_multi_target_diagnostics(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        y_pred: Union[np.ndarray, pd.DataFrame, List[np.ndarray]],
        target_cols: List[str],
        filename: str = "multi_target_diagnostics.png",
    ) -> None:
        """逐目标绘制 预测vs真实 / 相对误差 / 绝对误差 三联诊断图。"""
        y_true = self._to_numpy(y_true, target_cols)
        y_pred = self._to_numpy(y_pred, target_cols)
        n_targets = y_true.shape[1]
        fig, axes = plt.subplots(n_targets, 3, figsize=(24, 5 * n_targets), squeeze=False)
        for i, tgt in enumerate(target_cols):
            yt, yp = y_true[:, i], y_pred[:, i]
            mask = ~np.isnan(yt)
            yt, yp = yt[mask], yp[mask]
            abs_err = np.abs(yt - yp)
            rel_err = abs_err / (np.abs(yt) + self.eps)
            # 预测 vs 真实
            ax = axes[i, 0]
            ax.scatter(yt, yp, alpha=0.4, s=8)
            lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], "--", color="gray")
            ax.set_title(f"{tgt} | Pred vs Truth")
            ax.set_xlabel("True"); ax.set_ylabel("Pred"); ax.grid(True)
            # 相对误差
            ax = axes[i, 1]
            ax.hist(rel_err, bins=50, color="steelblue")
            ax.set_title(f"{tgt} | Relative Error"); ax.grid(True)
            # 绝对误差
            ax = axes[i, 2]
            ax.hist(abs_err, bins=50, color="salmon")
            ax.set_title(f"{tgt} | Absolute Error"); ax.grid(True)
        plt.tight_layout()
        if self.output_dir:
            plt.savefig(self.output_dir / filename, dpi=120, bbox_inches="tight")
        plt.close()

    # ==================================================================
    # SHAP 特征重要性
    # ==================================================================
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
    ) -> None:
        """SHAP 总结：树模型走 TreeExplainer，深度模型走 KernelExplainer。"""
        assert len(target_cols) >= 1, "target_cols 不能为空"
        X_val_df = (
            valid if isinstance(valid, pd.DataFrame)
            else pd.DataFrame(valid, columns=self.feature_cols)
        )
        shap_results = self._shap_compute(
            regressors, X_val_df, target_cols, bg_sample, nsamples, val_sample, n_jobs
        )
        if self.output_dir:
            shap_dir = self.output_dir / "shap"
            shap_dir.mkdir(exist_ok=True)
            self._plot_shap(shap_results, max_display, shap_dir)
            self._save_shap_excel(shap_results)

    def _shap_compute(
        self,
        regressors: Union[BaseRegressor, Dict[str, BaseRegressor]],
        X_val_df: pd.DataFrame,
        target_cols: List[str],
        bg_sample: int,
        nsamples: int,
        val_sample: int,
        n_jobs: int,
    ) -> Dict[str, Dict[str, Any]]:
        """计算 SHAP 值，分支：单目标树模型字典 / 多目标单模型。"""
        X_sample = self._sample_valid_set(X_val_df, val_sample)
        if isinstance(regressors, dict):
            # 单目标树模型：并行 TreeExplainer
            for r in regressors.values():
                if not self._is_tree_model(r.model) and not r.multi_target:
                    raise ValueError("只有 XGB/LGBM/CatBoost 单目标模型支持字典式 SHAP")

            def _tree_shap(r: BaseRegressor):
                # 单模型 SHAP 计算独立容错：某模型失败返回 None，不阻断其余模型
                try:
                    # shap 0.49 对 XGB/LGBM 的 category dtype 支持有限，统一转数值 codes
                    # （codes 与训练时 category 内部编码一致，SHAP 结果正确）
                    X_enc = _encode_for_shap(r, X_sample)
                    sv = shap.TreeExplainer(r.model).shap_values(X_enc)
                    return sv[0] if isinstance(sv, list) else sv
                except Exception as e:
                    logger.warning(f"SHAP 计算失败，跳过该模型: {e}")
                    return None

            # 逐模型串行计算（容错优先于并行；模型数通常 = 目标数，开销可接受）
            svs = [_tree_shap(r) for r in regressors.values()]
            # X 统一用第一个模型的编码结果作为绘图特征矩阵（同特征列集合）
            X_enc0 = _encode_for_shap(next(iter(regressors.values())), X_sample)
            return {
                tgt: {"shap_values": sv, "X": X_enc0}
                for tgt, sv in zip(regressors.keys(), svs)
                if sv is not None
            }

        # 多目标单模型：KernelExplainer
        model = regressors
        if not getattr(model, "multi_target", False):
            raise ValueError("单目标模型请以 {target: model} 字典形式传入")

        # 多目标模型 predict 可能返回列名带 _prediction 后缀（pytorch_tabular）或裸 target
        # 这里统一按 target_cols 顺序取预测列
        def predict_func(X_np: np.ndarray) -> np.ndarray:
            X_df = pd.DataFrame(X_np, columns=self.feature_cols)
            preds = model.predict(X_df)
            if isinstance(preds, pd.DataFrame):
                # 优先精确匹配，其次匹配 {target}_prediction 后缀列
                cols = []
                for t in target_cols:
                    if t in preds.columns:
                        cols.append(t)
                    else:
                        suf = f"{t}_prediction"
                        cols.append(suf if suf in preds.columns else None)
                if all(c is not None for c in cols):
                    return preds[cols].values
                # 兜底：取前 len(target_cols) 列
                return preds.iloc[:, : len(target_cols)].values
            return np.asarray(preds)

        bg = X_sample.sample(min(bg_sample, len(X_sample)), random_state=self.random_state)
        explainer = shap.KernelExplainer(predict_func, bg.values)
        sv = explainer.shap_values(X_sample.values, nsamples=min(nsamples, len(X_sample)))
        if isinstance(sv, list):
            sv = np.transpose(np.array(sv), (1, 2, 0))
        return {
            tgt: {"shap_values": sv[:, :, i] if sv.ndim == 3 else sv, "X": X_sample}
            for i, tgt in enumerate(target_cols)
        }

    def _plot_shap(
        self,
        shap_results: Dict[str, Dict[str, Any]],
        max_display: int,
        shap_dir: Path,
    ) -> None:
        """绘制 SHAP 总结图 + 分类特征方向图，逐 target 独立 try（单目标失败不阻断其余）。"""
        for tgt, data in shap_results.items():
            sv, X_df = data["shap_values"], data["X"]
            # 整个 target 的绘图（总结图 + 分类方向图）统一容错，单 target 失败不阻断其余
            try:
                # 总结图：默认 dot 图在某些特征 shap 全 0 时会触发 "xmin must be a single
                # scalar"（shap+matplotlib 兼容问题），优先尝试 dot，失败回退 bar 图
                plt.figure(figsize=(16, 9))
                try:
                    shap.summary_plot(sv, X_df, max_display=max_display, show=False)
                except Exception:
                    plt.close()
                    plt.figure(figsize=(16, 9))
                    shap.summary_plot(sv, X_df, max_display=max_display, plot_type="bar", show=False)
                plt.title(f"SHAP Summary | {tgt}")
                plt.savefig(shap_dir / f"shap_summary_{tgt}.png", dpi=120, bbox_inches="tight")
                plt.close()
                # 分类特征贡献方向
                for col in self.cat_cols:
                    if col not in X_df.columns:
                        continue
                    idx = X_df.columns.get_loc(col)
                    stat = (
                        pd.DataFrame({"category": X_df[col].astype(str), "shap": sv[:, idx]})
                        .groupby("category")["shap"].mean().sort_values(ascending=False)
                    )
                    plt.figure(figsize=(8, 4))
                    colors = stat.apply(lambda x: "tab:red" if x > 0 else "tab:blue")
                    plt.bar(stat.index, stat.values, color=colors)
                    plt.axhline(0, "--", linewidth=1)
                    plt.title(f"{tgt} | Mean SHAP by {col}")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(shap_dir / f"shap_cat_{tgt}_{col}.png", dpi=120, bbox_inches="tight")
                    plt.close()
            except Exception as e:
                logger.warning(f"SHAP 绘图跳过 {tgt}: {e}")
                plt.close("all")

    def _save_shap_excel(
        self, shap_results: Dict[str, Dict[str, Any]], filename: str = "shap_values.xlsx"
    ) -> None:
        if not self.output_dir:
            return
        with pd.ExcelWriter(self.output_dir / filename, engine="openpyxl") as writer:
            for tgt, data in shap_results.items():
                sv, X_df = data["shap_values"], data["X"]
                df = pd.DataFrame({
                    "feature": X_df.columns,
                    "mean_abs_shap": np.abs(sv).mean(axis=0),
                    "mean_shap": sv.mean(axis=0),
                }).sort_values("mean_abs_shap", ascending=False)
                df.to_excel(writer, sheet_name=tgt[:31], index=False)

    # ==================================================================
    # 区间覆盖对比：按 machine_sn × time_index
    # ==================================================================
    def interval_comparison(
        self,
        y_true_df: pd.DataFrame,
        y_pred_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        target_cols: List[str],
        sigma_mult: float = 3.0,
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Tuple[pd.DataFrame, None]:
        """
        按 machine_sn 分组、time_index 排序，对每通道输出五个区间的对比。

        五个区间（同一通道的 6 统计量组合）：
          1. 真实最值区间     [min, max]
          2. 真实次最值区间   [2min, 2max]
          3. 估计最值区间     [pred_min, pred_max]
          4. 估计次最值区间   [pred_2min, pred_2max]
          5. 预测均值±3σ区间  [mean-3σ, mean+3σ]  （σ 来自训练集该通道 variance）

        每个区间在每个机器内按 time_index 排序，形成序列对比。

        Args:
            y_true_df: 真实目标值（含 machine_sn/time_index 或通过 meta_df 提供）
            y_pred_df: 预测目标值
            meta_df: 含 machine_sn / time_index 的元信息（与 y_true_df 行对齐）
            target_cols: 目标列
            sigma_mult: 均值区间的 σ 倍数（默认 3）
            output_dir: 落盘目录，None 则用 self.output_dir

        Returns:
            (区间对比长表 DataFrame, None)
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        # 通道分组
        channels = _group_channels(target_cols)
        # 拼接元信息
        work = y_true_df.copy()
        for col in (MACHINE_COL, TIME_INDEX_COL):
            if col in meta_df.columns:
                work[col] = meta_df[col].values
        work_pred = y_pred_df.copy()
        # 逐通道构造区间对比记录
        records = []
        for ch, stats in channels.items():
            if not all(s in stats for s in ("min", "max", "2min", "2max", "mean")):
                continue
            # σ 取该通道 variance 列的均值（预测方差，作为不确定度估计）
            var_col = stats.get("variance")
            sigma = float(np.nanmean(work_pred[var_col].values)) ** 0.5 if var_col else 0.0
            for machine, sub in work.groupby(MACHINE_COL):
                sub = sub.sort_values(TIME_INDEX_COL)
                idx = sub.index
                pred_sub = work_pred.loc[idx]
                for i in range(len(sub)):
                    rec = {
                        MACHINE_COL: machine,
                        TIME_INDEX_COL: sub[TIME_INDEX_COL].iloc[i],
                        "channel": ch,
                        "true_min_max_lo": sub[stats["min"]].iloc[i],
                        "true_min_max_hi": sub[stats["max"]].iloc[i],
                        "true_2min_2max_lo": sub[stats["2min"]].iloc[i],
                        "true_2min_2max_hi": sub[stats["2max"]].iloc[i],
                        "pred_min_max_lo": pred_sub[stats["min"]].iloc[i],
                        "pred_min_max_hi": pred_sub[stats["max"]].iloc[i],
                        "pred_2min_2max_lo": pred_sub[stats["2min"]].iloc[i],
                        "pred_2min_2max_hi": pred_sub[stats["2max"]].iloc[i],
                        "pred_mean": pred_sub[stats["mean"]].iloc[i],
                        "pred_mean_pm3sigma_lo": pred_sub[stats["mean"]].iloc[i] - sigma_mult * sigma,
                        "pred_mean_pm3sigma_hi": pred_sub[stats["mean"]].iloc[i] + sigma_mult * sigma,
                        "sigma": sigma,
                    }
                    records.append(rec)
        result_df = pd.DataFrame(records)
        if out_dir is not None and not result_df.empty:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(out_dir / "interval_comparison.csv", index=False)
            self._plot_interval_comparison(result_df, out_dir)
        return result_df, None

    def _plot_interval_comparison(self, df: pd.DataFrame, out_dir: Path) -> None:
        """逐机器逐通道绘制五区间序列对比图（前 N 个机器/通道以控制图量）。"""
        plot_dir = out_dir / "interval_plots"
        plot_dir.mkdir(exist_ok=True)
        channels = df["channel"].unique()[:8]  # 控制图量，最多前 8 通道
        machines = df[MACHINE_COL].unique()[:6]  # 每通道最多前 6 机器
        for ch in channels:
            sub_ch = df[df["channel"] == ch]
            for machine in machines:
                sub = sub_ch[sub_ch[MACHINE_COL] == machine].sort_values(TIME_INDEX_COL)
                if len(sub) < 2:
                    continue
                x = sub[TIME_INDEX_COL].values
                fig, ax = plt.subplots(figsize=(14, 6))
                # 五个区间用 fill_between 叠加
                _fill(ax, x, sub["true_min_max_lo"], sub["true_min_max_hi"], "真实最值[min,max]", "tab:blue", 0.25)
                _fill(ax, x, sub["true_2min_2max_lo"], sub["true_2min_2max_hi"], "真实次最值[2min,2max]", "tab:cyan", 0.2)
                _fill(ax, x, sub["pred_min_max_lo"], sub["pred_min_max_hi"], "估计最值[pred_min,pred_max]", "tab:orange", 0.2)
                _fill(ax, x, sub["pred_2min_2max_lo"], sub["pred_2min_2max_hi"], "估计次最值[pred_2min,pred_2max]", "tab:red", 0.2)
                _fill(ax, x, sub["pred_mean_pm3sigma_lo"], sub["pred_mean_pm3sigma_hi"], "预测均值±3σ", "tab:green", 0.15)
                ax.plot(x, sub["pred_mean"], color="tab:green", linewidth=1.5, label="预测均值")
                ax.set_title(f"区间对比 | {ch} | machine={machine}")
                ax.set_xlabel(TIME_INDEX_COL); ax.set_ylabel("指标值")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir / f"interval_{ch}_{machine}.png", dpi=110, bbox_inches="tight")
                plt.close()

    # ==================================================================
    # 辅助
    # ==================================================================
    def _sample_valid_set(self, X_val_df: pd.DataFrame, min_samples: int = 500) -> pd.DataFrame:
        """验证集抽样（至少 1/3，下限 min_samples）。"""
        n = max(min_samples, len(X_val_df) // 3)
        n = min(len(X_val_df), n)
        return X_val_df.sample(n=n, random_state=self.random_state)

    # ==================================================================
    # 全流程编排
    # ==================================================================
    def run_full_evaluation(
        self,
        regressors: Union[BaseRegressor, Dict[str, BaseRegressor]],
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
        run_interval: bool = True,
    ) -> None:
        """
        全流程评估：指标计算 + 诊断图 + SHAP +（可选）区间对比。

        Args:
            regressors: 单目标字典 {target: model} 或多目标单模型
            train / valid: 训练/验证集（含特征与目标）
            target_cols: 目标列
            run_interval: 是否运行按机器区间对比（需要 valid 含 machine_sn/time_index）
        """
        # 预测
        train_y_pred = self._collect_predictions(regressors, train, target_cols)
        valid_y_pred = self._collect_predictions(regressors, valid, target_cols)
        # 指标 + 诊断图
        self.compute_metrics(
            train[target_cols], train_y_pred, valid[target_cols], valid_y_pred, target_cols
        )
        self.plot_multi_target_diagnostics(valid[target_cols], valid_y_pred, target_cols)
        # SHAP
        try:
            self.shap_summary_all_targets(regressors, valid[self.feature_cols], target_cols)
        except Exception as e:  # SHAP 对部分模型可能失败，不阻断主流程
            logger.warning(f"SHAP 计算失败，跳过: {e}")
        # 区间对比
        if run_interval and MACHINE_COL in valid.columns and TIME_INDEX_COL in valid.columns:
            y_pred_df = pd.DataFrame(valid_y_pred, columns=target_cols, index=valid.index)
            self.interval_comparison(
                y_true_df=valid[target_cols],
                y_pred_df=y_pred_df,
                meta_df=valid[[MACHINE_COL, TIME_INDEX_COL]],
                target_cols=target_cols,
            )
        else:
            logger.info("跳过区间对比（valid 缺少 machine_sn/time_index 或显式关闭）")


def _group_channels(target_cols: List[str]) -> Dict[str, Dict[str, str]]:
    """将目标列按通道分组：{channel: {stat: col}}。"""
    groups: Dict[str, Dict[str, str]] = {}
    for c in target_cols:
        for stat in ("2max", "2min", "mean", "variance", "max", "min"):
            if c.endswith("_" + stat):
                ch = c[: -(len(stat) + 1)]
                groups.setdefault(ch, {})[stat] = c
                break
    return groups


def _fill(ax, x, lo, hi, label, color, alpha):
    """绘制区间带（fill_between），NaN 自动跳过。"""
    lo = pd.Series(lo).astype(float).values
    hi = pd.Series(hi).astype(float).values
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, label=label)


def _encode_for_shap(regressor: BaseRegressor, X_df: pd.DataFrame) -> pd.DataFrame:
    """为树模型 SHAP 准备与训练一致的特征编码，并转为纯数值（codes）。

    XGB/LGBM 训练时分类列经 cat_mappings 归一 + category dtype；shap 0.49 对 category dtype
    支持有限，这里在编码后再把 category 列转为其 codes（整数），保证 SHAP 输入为纯数值且与
    训练分布一致。CatBoost 走 transform_cat_cols（字符串），TreeExplainer 原生支持。
    """
    # XGB / LGBM 暴露 _encode_features（返回 category dtype）
    encode = getattr(regressor, "_encode_features", None)
    if encode is not None:
        X_enc = encode(X_df)
        # category -> codes（与训练 category 顺序一致）
        for col in X_enc.columns:
            if isinstance(X_enc[col].dtype, pd.CategoricalDtype):
                X_enc[col] = X_enc[col].cat.codes.astype("float64")
        return X_enc
    # CatBoost：transform_cat_cols 保持字符串，CatBoost TreeExplainer 支持
    cat_mappings = getattr(regressor, "_cat_mappings", None)
    if cat_mappings is not None:
        from src.core.models.tabular.cat_encoding import transform_cat_cols
        return transform_cat_cols(X_df, regressor.cat_cols, cat_mappings)
    return X_df
