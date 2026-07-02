"""
数据清洗

按业务规则清洗原始锅次数据。清洗策略遵循"明确坏数据丢弃、存疑离群仅标记"原则：
  - 丢弃类（无法用于监督训练或明确物理不合理）：
      1. S 整批缺失：所有 S 类列均缺失，无设参特征，无法建模
      2. Y 全缺：颜色与透过率指标全部缺失，无监督信号
      3. Y 物理越界：颜色 a/b 绝对值超阈值、透过率超出 (0,100]，明确坏数据
  - 标记类（保留信息，供模型与 EDA 自行处理）：
      cathode / incoming 的 IQR 离群点：因缺失锅次或来料特性导致，未必是错误，
      故不删除，仅在清洗报告中统计，避免丢失真实生产信息。

清洗只做"定位与剔除"，不做填充、缩放等会改变特征分布的操作（那些留到特征工程阶段）。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.core.constants.data_constants import (
    S_ALL_COLS,
    Y_COLOR_COLS,
    Y_TRANS_COLS,
    Y_ALL_COLS,
    COLOR_AB_ABS_MAX,
    TRANSMISSION_VALID_RANGE,
    Y_STATS_WITHOUT_VARIANCE,
)


@dataclass
class CleaningReport:
    """清洗过程统计，便于核对每一步丢弃了多少样本及原因。"""

    n_raw: int = 0
    # 各丢弃规则命中行数（注意：同一行可能命中多条，故各项之和可能大于实际丢弃总数）
    n_drop_s_all_missing: int = 0
    n_drop_y_all_missing: int = 0
    n_drop_y_out_of_range: int = 0
    # 离群标记统计（仅统计不删除）
    n_outlier_rows: int = 0
    outlier_per_col: Dict[str, int] = field(default_factory=dict)
    # 最终保留
    n_clean: int = 0

    def summary(self) -> str:
        """生成可读的清洗摘要文本。"""
        lines = [
            f"原始样本数: {self.n_raw}",
            f"丢弃 S 整批缺失: {self.n_drop_s_all_missing}",
            f"丢弃 Y 全缺: {self.n_drop_y_all_missing}",
            f"丢弃 Y 物理越界: {self.n_drop_y_out_of_range}",
            f"离群标记行数(仅标记不删): {self.n_outlier_rows}",
            f"清洗后样本数: {self.n_clean}",
            f"清洗保留率: {self.n_clean / max(self.n_raw, 1):.2%}",
        ]
        return "\n".join(lines)


def _detect_y_out_of_range(df: pd.DataFrame) -> pd.Series:
    """
    检测 Y 物理越界行（颜色 a/b 绝对值超阈值、透过率超出有效区间），不含 variance 列。

    Returns:
        boolean Series：True 表示该行至少有一个 Y 指标物理越界
    """
    lo, hi = TRANSMISSION_VALID_RANGE
    masks: List[pd.Series] = []

    # 颜色 a/b 列：仅 mean/max/min/2max/2min（方差无物理上下限意义）
    color_ab_cols = [
        c for c in Y_COLOR_COLS
        if c.split("_")[-1] in Y_STATS_WITHOUT_VARIANCE
        and (c.endswith("_a_" + c.split("_")[-1]) or c.endswith("_b_" + c.split("_")[-1]))
    ]
    # 上面按后缀匹配较繁琐，改为直接按通道名 a/b 过滤
    color_ab_cols = [
        c for c in Y_COLOR_COLS
        if c.split("_")[-1] in Y_STATS_WITHOUT_VARIANCE
        and any(f"_{ch}_" in c for ch in ["a", "b"])
    ]
    for col in color_ab_cols:
        if col in df.columns:
            masks.append(df[col].abs() > COLOR_AB_ABS_MAX)

    # 透过率列：所有非 variance 统计量
    trans_cols = [
        c for c in Y_TRANS_COLS if c.split("_")[-1] in Y_STATS_WITHOUT_VARIANCE
    ]
    for col in trans_cols:
        if col in df.columns:
            s = df[col]
            masks.append((s <= lo) | (s > hi))

    if not masks:
        return pd.Series(False, index=df.index)
    return pd.concat(masks, axis=1).any(axis=1)


def _mark_outliers(
    df: pd.DataFrame, cols: List[str], k: float = 5.0
) -> tuple[pd.Series, Dict[str, int]]:
    """
    对指定列做 IQR×k 离群标记（仅标记不删除）。

    Args:
        df: 数据
        cols: 参与离群检测的列
        k: IQR 倍数（默认 5，与 EDA 一致，较宽松避免误删真实生产信息）

    Returns:
        any_outlier: 命中任一列离群的行级 boolean Series
        per_col: 每列离群行数
    """
    col_masks: Dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        valid = s.dropna()
        if valid.empty:
            continue
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        # IQR=0 表示中间 50% 无变异，k×IQR 无意义，跳过
        if iqr <= 0:
            continue
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        col_masks[col] = (s < lower) | (s > upper)

    if not col_masks:
        return pd.Series(False, index=df.index), {}

    any_outlier = pd.concat(col_masks.values(), axis=1).any(axis=1)
    per_col = {c: int(m.sum()) for c, m in col_masks.items() if m.sum() > 0}
    return any_outlier, per_col


def clean_raw_data(
    df: pd.DataFrame,
    drop_s_all_missing: bool = False,
    drop_y_all_missing: bool = True,
    drop_y_out_of_range: bool = True,
    outlier_cols: Optional[List[str]] = None,
    outlier_k: float = 5.0,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    对原始数据执行清洗，返回清洗后数据与清洗报告。

    Args:
        df: 原始数据
        drop_s_all_missing: 是否丢弃 S 整批缺失的锅次；默认 False——S 全缺属真实生产中的缺失
            情况，保留以考察模型在缺失设参时凭借其他特征估计指标的能力（缺失由模型自行处理）
        drop_y_all_missing: 是否丢弃 Y 全缺的锅次
        drop_y_out_of_range: 是否丢弃 Y 物理越界的锅次
        outlier_cols: 参与离群标记的列；默认 cathode + incoming（不含 Y，Y 越界已单独处理）
        outlier_k: IQR 离群倍数

    Returns:
        cleaned_df: 清洗后的 DataFrame（保留原始列，未做填充/缩放）
        report: CleaningReport 清洗统计
    """
    report = CleaningReport(n_raw=len(df))
    if outlier_cols is None:
        # 离群标记默认覆盖 cathode 与 incoming；Y 的物理越界已单独处理，不重复纳入 IQR
        from src.core.constants.data_constants import M_CATHODE_COLS, INCOMING_QUALITY_COLS
        outlier_cols = M_CATHODE_COLS + INCOMING_QUALITY_COLS

    df_work = df.copy()

    # 1. 离群标记（仅统计，不删除；在丢弃前统计以反映原始数据真实分布）
    any_outlier, per_col = _mark_outliers(df_work, outlier_cols, k=outlier_k)
    report.n_outlier_rows = int(any_outlier.sum())
    report.outlier_per_col = per_col

    # 2. 丢弃 S 整批缺失
    drop_mask = pd.Series(False, index=df_work.index)
    if drop_s_all_missing and S_ALL_COLS:
        s_missing = df_work[S_ALL_COLS].isna().all(axis=1)
        report.n_drop_s_all_missing = int(s_missing.sum())
        drop_mask |= s_missing

    # 3. 丢弃 Y 全缺
    if drop_y_all_missing and Y_ALL_COLS:
        y_missing = df_work[Y_ALL_COLS].isna().all(axis=1)
        report.n_drop_y_all_missing = int(y_missing.sum())
        drop_mask |= y_missing

    # 4. 丢弃 Y 物理越界
    if drop_y_out_of_range:
        y_oob = _detect_y_out_of_range(df_work)
        report.n_drop_y_out_of_range = int(y_oob.sum())
        drop_mask |= y_oob

    cleaned = df_work.loc[~drop_mask].copy()
    report.n_clean = len(cleaned)
    return cleaned, report
