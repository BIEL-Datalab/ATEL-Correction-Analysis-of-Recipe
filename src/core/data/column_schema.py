"""
列分组与校验

基于 data_constants 的列名定义，提供按业务类别访问列、以及对真实数据做双向校验的能力。
列名常量集中定义在 data_constants，本模块负责"组织成可消费的结构"与"校验一致性"，
两者分离便于后续单独扩展分组逻辑而不动列名常量。
"""

from typing import Dict, List

from src.core.constants.data_constants import (
    S_ALL_COLS,
    M_CATHODE_COLS,
    M_DURATION_COLS,
    Y_COLOR_COLS,
    Y_TRANS_COLS,
    Y_ALL_COLS,
    CONTEXT_COLS,
    IDENTIFIER_COLS,
    CAT_FEATURE_COLS,
    NUM_FEATURE_COLS,
    DATE_FEATURE_COLS,
    EXCLUDE_GROUPS,
    validate_columns,
    find_unused_columns,
)


def get_column_groups() -> Dict[str, List[str]]:
    """
    返回按业务类别组织的列分组，供 EDA 与预处理统一取用。

    分组语义：
      - identifier：标识/时间列，用于定位与分组，不直接作为数值特征
      - S / M / Y：设定参数 / 监控参数 / 质检输出
      - context：环境/上下文特征（产品规格、维保、ftu、来料、测试量等）
      - cat_feature：分类特征（含 machine_sn 等低基类别）
      - num_feature：数值特征（S + M + 来料 + 测试量 + 设参计数 + 产品规格数值）
      - date_feature：由 hc_chamber_day 派生的时间特征 year/month/quarter
      - exclude：本期明确不纳入的列（o2 / ar）
    """
    return {
        "identifier": IDENTIFIER_COLS,
        "S": S_ALL_COLS,
        "M": M_CATHODE_COLS + M_DURATION_COLS,
        "Y_color": Y_COLOR_COLS,
        "Y_trans": Y_TRANS_COLS,
        "Y_all": Y_ALL_COLS,
        "context": CONTEXT_COLS,
        "cat_feature": CAT_FEATURE_COLS,
        "num_feature": NUM_FEATURE_COLS,
        "date_feature": DATE_FEATURE_COLS,
        "exclude": [c for cols in EXCLUDE_GROUPS.values() for c in cols],
    }


def get_feature_cols() -> Dict[str, List[str]]:
    """
    返回可直接用于建模的特征列（数值 + 分类 + 时间派生），按类型分组。

    用于训练时构造 num_cols / cat_cols / date_cols。
    """
    return {
        "num_cols": NUM_FEATURE_COLS,
        "cat_cols": CAT_FEATURE_COLS,
        "date_cols": DATE_FEATURE_COLS,
    }


def check_column_consistency(available_cols) -> Dict[str, Dict[str, List[str]]]:
    """
    对真实数据做双向校验，返回结构化校验报告。

    Args:
        available_cols: 真实数据列名集合或列表

    Returns:
        {
          "missing": {group -> 真实数据中缺失的定义列},   # 定义里有、数据里没有
          "unused":  {group -> 数据里有、定义里未用的列},  # 数据里有、定义里未涵盖
        }
        两者均为空时表示列名定义与真实数据完全一致。
    """
    return {
        "missing": validate_columns(available_cols),
        "unused": find_unused_columns(available_cols),
    }
