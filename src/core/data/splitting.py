"""
数据切分

按镀膜日期（hc_chamber_day）切分数据集。切分流程分两步：
  1. 先在全量清洗数据上按日期取最后一段作为测试集，测试集一旦确定不再随训练起点
     变化，保证不同训练起点实验在同一测试集上公平对比；
  2. 测试集之前的数据按日期区间筛选训练起点（实验旋钮，过滤久远数据），筛选后按 8:2 切
     训练/验证。

切分键选用 hc_chamber_day（镀膜日期）而非 time_index：time_index 仅为每台机器内的生产先后
相对顺序，不同机器并行生产时同一 time_index 会横跨多个月份，无法保证全局时间单调；镀膜日期
能严格保证时间先后，符合"训练早期、验证近期、测试最新"的时间外推评估语义。time_index 仍作为
生产先后相对顺序的特征保留在数据中。

同时保留随机切分作为可选，便于对照实验。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.constants.data_constants import TIME_INDEX_COL, HC_CHAMBER_DAY_COL

# 切分排序键：镀膜日期（保证全局时间单调）
SPLIT_SORT_COL = HC_CHAMBER_DAY_COL


# 测试集默认占全量清洗数据比例（取最后一段）
DEFAULT_TEST_RATIO = 0.1
# 测试集前数据中训练集占比，剩余为验证（即训练/验证 8:2）
DEFAULT_TRAIN_RATIO_IN_REST = 0.8


@dataclass
class SplitReport:
    """切分过程统计，记录每次切分后各集合的数据量与时间范围。"""

    split_method: str = ""  # "time" 或 "random"
    n_total: int = 0  # 进入本次切分的数据量（日期筛选后、切分前）
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0
    train_time_range: str = ""  # 训练集 hc_chamber_day 范围
    valid_time_range: str = ""  # 验证集 hc_chamber_day 范围
    test_time_range: str = ""  # 测试集 hc_chamber_day 范围
    train_time_index_range: str = ""
    valid_time_index_range: str = ""
    test_time_index_range: str = ""
    # 日期筛选相关（filter_by_date_range 使用）
    date_filter: str = ""  # 形如 "2025-08-01 ~ None"
    n_before_date_filter: int = 0  # 日期筛选前数据量

    def summary(self) -> str:
        lines = [f"切分方式: {self.split_method}"]
        if self.date_filter:
            lines.append(
                f"日期筛选: {self.date_filter} (筛选前 {self.n_before_date_filter} -> {self.n_total})"
            )
        lines.extend([
            f"各集合数据量: 训练 {self.n_train} / 验证 {self.n_valid} / 测试 {self.n_test}",
            f"训练集 hc_chamber_day: {self.train_time_range}",
            f"验证集 hc_chamber_day: {self.valid_time_range}",
            f"测试集 hc_chamber_day: {self.test_time_range}",
            f"训练集 time_index: {self.train_time_index_range}",
            f"验证集 time_index: {self.valid_time_index_range}",
            f"测试集 time_index: {self.test_time_index_range}",
        ])
        return "\n".join(lines)


def filter_by_date_range(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    date_col: str = HC_CHAMBER_DAY_COL,
) -> Tuple[pd.DataFrame, SplitReport]:
    """
    按日期区间筛选样本（闭区间）。用于选择训练起点，过滤久远数据。

    Args:
        df: 数据
        start: 起始日期（含），形如 "2025-08-01"；None 表示不设下界
        end: 结束日期（含），形如 "2026-01-31"；None 表示不设上界
        date_col: 日期列

    Returns:
        filtered_df, report（report 仅填充日期筛选相关字段）
    """
    report = SplitReport(date_filter=f"{start} ~ {end}", n_before_date_filter=len(df))
    if date_col not in df.columns:
        raise KeyError(f"数据中不存在日期列 {date_col}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= dates >= pd.to_datetime(start)
    if end is not None:
        mask &= dates <= pd.to_datetime(end)
    filtered = df.loc[mask].copy()
    report.n_total = len(filtered)
    return filtered, report


def holdout_test_set(
    df: pd.DataFrame,
    test_ratio: float = DEFAULT_TEST_RATIO,
    sort_col: str = SPLIT_SORT_COL,
    date_col: str = HC_CHAMBER_DAY_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitReport]:
    """
    在全量数据上按日期取最后 test_ratio 作为测试集，返回（测试集前数据, 测试集）。

    测试集一旦确定不再变化；后续对"测试集前数据"做日期筛选与训练/验证切分时，
    测试集保持固定。按镀膜日期排序，缺失日期的样本排到最后并落入测试集。

    Args:
        df: 全量清洗后数据
        test_ratio: 测试集占全量比例（取最后一段）
        sort_col: 用于排序的列（默认镀膜日期，保证全局时间单调）
        date_col: 用于报告时间范围的日期列

    Returns:
        rest_df: 测试集前数据（供后续筛选与训练/验证切分）
        test_df: 测试集
        report: 仅填充 n_total/n_test 与测试集时间范围
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio 必须在 (0,1)，当前 {test_ratio}")
    report = SplitReport(split_method="time", n_total=len(df))
    sorted_df = _sort_by_date(df, sort_col)
    n_test = int(len(sorted_df) * test_ratio)
    if n_test > 0:
        test_df = sorted_df.iloc[-n_test:].copy()
        rest_df = sorted_df.iloc[:-n_test].copy()
    else:
        test_df = sorted_df.iloc[0:0].copy()
        rest_df = sorted_df.copy()
    report.n_test = len(test_df)
    _fill_time_ranges(report, pd.DataFrame(), pd.DataFrame(), test_df, date_col)
    return rest_df, test_df, report


def split_train_valid(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO_IN_REST,
    sort_col: str = SPLIT_SORT_COL,
    date_col: str = HC_CHAMBER_DAY_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, SplitReport]:
    """
    对测试集前数据（可先经日期筛选）按日期顺序切训练/验证。

    Args:
        df: 测试集前数据（通常先经 filter_by_date_range 筛选训练起点）
        train_ratio: 训练集占比（默认 0.8，即训练/验证 8:2）
        sort_col: 用于排序的列（默认镀膜日期）
        date_col: 用于报告时间范围的日期列

    Returns:
        train_df, valid_df, report
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio 必须在 (0,1)，当前 {train_ratio}")
    report = SplitReport(split_method="time", n_total=len(df))
    sorted_df = _sort_by_date(df, sort_col)
    n_train = int(len(sorted_df) * train_ratio)
    train_df = sorted_df.iloc[:n_train].copy()
    valid_df = sorted_df.iloc[n_train:].copy()
    report.n_train = len(train_df)
    report.n_valid = len(valid_df)
    _fill_time_ranges(report, train_df, valid_df, pd.DataFrame(), date_col)
    return train_df, valid_df, report


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    sort_col: str = SPLIT_SORT_COL,
    date_col: str = HC_CHAMBER_DAY_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """
    按日期顺序一次性切分训练/验证/测试集。

    与 holdout_test_set + split_train_valid 的两步流程区别：本函数三份一次性按比例切，
    不保证测试集在多次实验间固定；适合不需要扫描训练起点的单次基线实验。

    Args:
        df: 数据
        train_ratio: 训练集占比
        valid_ratio: 验证集占比
        test_ratio: 测试集占比（三者之和应为 1.0）

    Returns:
        train_df, valid_df, test_df, report
    """
    _check_ratios(train_ratio, valid_ratio, test_ratio)
    report = SplitReport(split_method="time", n_total=len(df))
    sorted_df = _sort_by_date(df, sort_col)
    n = len(sorted_df)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train_df = sorted_df.iloc[:n_train].copy()
    valid_df = sorted_df.iloc[n_train : n_train + n_valid].copy()
    test_df = sorted_df.iloc[n_train + n_valid :].copy()
    report.n_train = len(train_df)
    report.n_valid = len(valid_df)
    report.n_test = len(test_df)
    _fill_time_ranges(report, train_df, valid_df, test_df, date_col)
    return train_df, valid_df, test_df, report


def split_random(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_state: int = 42,
    date_col: str = HC_CHAMBER_DAY_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """
    随机切分训练/验证/测试集，用于与时间切分对照实验。

    Returns:
        train_df, valid_df, test_df, report
    """
    _check_ratios(train_ratio, valid_ratio, test_ratio)
    report = SplitReport(split_method="random", n_total=len(df))
    test_size = test_ratio
    train_valid_size = 1.0 - test_ratio
    valid_size_in_train_valid = valid_ratio / train_valid_size if train_valid_size > 0 else 0.0
    train_valid_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train_df, valid_df = train_test_split(
        train_valid_df, test_size=valid_size_in_train_valid, random_state=random_state
    )
    report.n_train = len(train_df)
    report.n_valid = len(valid_df)
    report.n_test = len(test_df)
    _fill_time_ranges(report, train_df, valid_df, test_df, date_col)
    return train_df, valid_df, test_df, report


def _sort_by_date(df: pd.DataFrame, sort_col: str) -> pd.DataFrame:
    """按日期列排序，缺失日期排到最后。sort_col 为日期列（默认 hc_chamber_day）。"""
    if sort_col not in df.columns:
        raise KeyError(f"数据中不存在排序列 {sort_col}")
    # 用日期的 Timestamp 排序，避免 object 字符串排序不稳定；缺失排到最后
    sort_key = pd.to_datetime(df[sort_col], errors="coerce")
    return df.assign(_sort_key=sort_key).sort_values(
        "_sort_key", kind="mergesort", na_position="last"
    ).drop(columns="_sort_key")


def _check_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    """校验三份比例之和为 1.0（允许浮点误差）。"""
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"切分比例之和必须为 1.0，当前 train+valid+test = {total}"
        )


def _fill_time_ranges(
    report: SplitReport,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str,
) -> None:
    """填充报告中的时间范围信息，便于核对切分是否符合时间先后预期。传入空 DataFrame 时跳过该项。"""
    splits = [
        (train_df, "train_time_range", "train_time_index_range"),
        (valid_df, "valid_time_range", "valid_time_index_range"),
        (test_df, "test_time_range", "test_time_index_range"),
    ]
    for sub, date_attr, ti_attr in splits:
        if sub.empty:
            continue
        if date_col in sub.columns:
            dates = pd.to_datetime(sub[date_col], errors="coerce").dropna()
            if not dates.empty:
                setattr(report, date_attr, f"{dates.min().date()} ~ {dates.max().date()}")
        # time_index 作为生产先后相对顺序特征，在报告中记录其范围供参考
        if TIME_INDEX_COL in sub.columns:
            ti = sub[TIME_INDEX_COL].dropna()
            if not ti.empty:
                setattr(report, ti_attr, f"{int(ti.min())} ~ {int(ti.max())}")
