"""
预测后处理：物理约束修正

强制预测满足锅次统计量的物理单调性：
    min ≤ 2min ≤ 2max ≤ max       （极值/次极值嵌套）
    min ≤ mean ≤ max              （均值在极值之间）

业务上这些约束恒成立（同一锅次内，次极值必在极值之间，均值必在极值之间）。
模型各目标独立训练时可能输出违反约束的组合，本模块在预测后做最小修正使其满足约束。

修正策略（投影到可行域，最小改动）：
  1. 对 [min, 2min, 2max, max] 四列按通道整体排序，使序列非递减；
  2. 对 mean 用 [min, max] 夹逼。
排序法（isotonic-like）相比逐对 clip 更稳定：四列整体排序后必然满足两两顺序约束，
且改动量最小（与各列预测值的 L2 距离最近的形式之一）。

通道识别：目标列名形如 '{channel}_{stat}'，channel 为 c3_10_a / t_0deg_940 等。
同一 channel 的 6 个统计量为一组（mean/variance/max/min/2max/2min），其中 max/min/2max/2min
参与排序约束，mean 参与夹逼，variance 不参与（方差无单调约束）。
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

# 参与排序约束的四列统计量
ORDERED_STATS = ["min", "2min", "2max", "max"]
# mean 用 [min, max] 夹逼
MEAN_STAT = "mean"
# variance 不参与任何约束
_EXCLUDED_STAT = "variance"


def channel_of(target_col: str) -> str:
    """从目标列名提取通道名（去掉最后一个统计量后缀）。

    例：'c3_10_a_2min' -> 'c3_10_a'；'t_0deg_940_mean' -> 't_0deg_940'。
    """
    for stat in ("2max", "2min", "mean", "variance", "max", "min"):
        if target_col.endswith("_" + stat):
            return target_col[: -(len(stat) + 1)]
    return target_col


def group_by_channel(target_cols: List[str]) -> Dict[str, Dict[str, str]]:
    """
    将目标列按通道分组，返回 {channel: {stat: col}}。

    仅保留拥有全部 4 个排序统计量（min/2min/2max/max）的通道，否则跳过该通道的排序约束。
    """
    groups: Dict[str, Dict[str, str]] = defaultdict(dict)
    for c in target_cols:
        for stat in ("mean",) + tuple(ORDERED_STATS):
            if c.endswith("_" + stat):
                groups[channel_of(c)][stat] = c
                break
    # 仅保留 4 列齐全的通道
    return {
        ch: stats
        for ch, stats in groups.items()
        if all(s in stats for s in ORDERED_STATS)
    }


def _enforce_non_decreasing(values: np.ndarray) -> np.ndarray:
    """
    对 (n_samples, 4) 的数组按行整体排序，使每行非递减（min≤2min≤2max≤max）。

    排序即最小改动投影：四值排序后必然满足两两顺序，且是 L2 最近的可排序形式。
    NaN 保持原位（排序时 NaN 视为最大放末尾，排序后若该位本就缺失则回填 NaN）。
    """
    out = values.copy()
    n_rows, n_cols = out.shape
    for i in range(n_rows):
        row = out[i]
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue  # 不足两个非空值，无需排序
        sorted_vals = np.sort(row[mask])
        out[i, mask] = sorted_vals
    return out


def apply_constraints(
    predictions: pd.DataFrame,
    target_cols: List[str],
) -> pd.DataFrame:
    """
    对预测结果施加物理单调性约束，返回修正后的 DataFrame（副本）。

    Args:
        predictions: 预测值 DataFrame，列含 target_cols
        target_cols: 目标列名（用于识别通道与统计量）

    Returns:
        修正后的预测 DataFrame（不修改输入）
    """
    if predictions is None or predictions.empty:
        return predictions
    out = predictions.copy()
    groups = group_by_channel(target_cols)

    for ch, stats in groups.items():
        # 1. 排序约束：min ≤ 2min ≤ 2max ≤ max
        cols = [stats[s] for s in ORDERED_STATS]
        mat = out[cols].to_numpy(dtype=float)
        out[cols] = _enforce_non_decreasing(mat)

        # 2. 夹逼：mean 用 [min, max] 限制
        if MEAN_STAT in stats:
            mean_col = stats[MEAN_STAT]
            lo = out[stats["min"]]
            hi = out[stats["max"]]
            out[mean_col] = out[mean_col].clip(lower=lo, upper=hi)

    return out


def report_violations(
    predictions: pd.DataFrame,
    target_cols: List[str],
) -> Dict[str, int]:
    """
    统计预测中违反约束的行数（诊断用，apply 前调用看修正量）。

    Returns:
        {约束类型: 违反行数}，键如 'min>2min' / '2max>max' / 'min>mean' / 'mean>max'，
        以及 'channels_checked' 通道数。
    """
    violations: Dict[str, int] = {}
    groups = group_by_channel(target_cols)
    total_min2min = total_2min2max = total_2maxmax = 0
    total_minmean = total_meanmax = 0

    for ch, stats in groups.items():
        mn = predictions[stats["min"]]
        smin = predictions[stats["2min"]]
        smax = predictions[stats["2max"]]
        mx = predictions[stats["max"]]

        valid4 = ~(mn.isna() | smin.isna() | smax.isna() | mx.isna())
        total_min2min += int((mn > smin)[valid4].sum())
        total_2min2max += int((smin > smax)[valid4].sum())
        total_2maxmax += int((smax > mx)[valid4].sum())

        if MEAN_STAT in stats:
            me = predictions[stats[MEAN_STAT]]
            vm = ~(mn.isna() | me.isna() | mx.isna())
            total_minmean += int((mn > me)[vm].sum())
            total_meanmax += int((me > mx)[vm].sum())

    violations["channels_checked"] = len(groups)
    violations["min>2min"] = total_min2min
    violations["2min>2max"] = total_2min2max
    violations["2max>max"] = total_2maxmax
    violations["min>mean"] = total_minmean
    violations["mean>max"] = total_meanmax
    return violations
