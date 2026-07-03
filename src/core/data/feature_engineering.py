"""
特征工程

基于 EDA 结论与业务规则，对清洗后数据派生/修正特征：
  1. ftu_combo：ftu_operation 与 ftu_type 业务强相关，组合为单个分类变量
     （形如 "0_none" / "1_Ar" / "1_O2"），替代原始两列分别入模型
  2. service_type 缺失填充：缺失表示数据起点未记录上次保养信息（集中在 time_index 较小段），
     填充为独立类别 unknown，区别于真实无保养
  3. pass_status 修正：fail_detail != "无" 时 pass_status 必为 fail，修正误标为 pass 的 9 行
  4. cond1_triggered / cond2_triggered：由 fail_detail 派生两个二值特征，标识是否触发条件1/2；
     当前数据两者同时触发，拆成独立二值便于未来单条件触发场景

本模块只做特征派生与修正，不改变已有列的数值分布（除上述明确的修正外）。
"""

from typing import List

import pandas as pd

from src.core.constants.data_constants import (
    FTU_COLS,
    FTU_COMBO_COL,
    SERVICE_TYPE_MISSING_FILL,
    MAINTENANCE_COLS,
    COND1_TRIGGERED_COL,
    COND2_TRIGGERED_COL,
    FAIL_DETAIL_PASS_TOKEN,
)


def build_ftu_combo(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 ftu_operation + ftu_type 组合为单个分类变量 ftu_combo。

    组合规则：ftu_operation=0 时 ftu_type 为空，组合值为 "0_none"；
    ftu_operation=1 时组合值为 "1_{ftu_type}"（如 "1_Ar" / "1_O2"）。
    缺失的 ftu_operation 归为 "unknown"。

    Returns:
        新增 ftu_combo 列的 DataFrame（副本）
    """
    df = df.copy()
    op_col, type_col = FTU_COLS
    op = df[op_col]
    tp = df[type_col]

    def _combine(o, t):
        if pd.isna(o):
            return "unknown"
        o = int(o)
        if o == 0:
            return "0_none"
        # ftu_operation=1 时 ftu_type 应非空；缺失归 unknown
        return f"1_{t}" if pd.notna(t) else "1_unknown"

    df[FTU_COMBO_COL] = [_combine(o, t) for o, t in zip(op, tp)]
    return df


def fill_service_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    填充 service_type 缺失为独立类别 unknown。

    缺失原因：数据起点未记录上次保养信息（集中在每台机器 time_index 较小段），
    填充为 unknown 与真实保养类型区分，避免模型将缺失误学为"无保养"。
    service_order 保持数值不变，让模型自学非线性。

    注意：normalize_feature_dtypes 已将分类列转为 str，故 None 变成字符串 "None"，
    需同时处理真实 NaN 与字符串 "None" 两种缺失表示。
    """
    df = df.copy()
    if "service_type" in df.columns:
        # 兼容真实 NaN 与被 str 化的 "None"（normalize_feature_dtypes 产物）
        df["service_type"] = df["service_type"].replace({"None": SERVICE_TYPE_MISSING_FILL})
        df["service_type"] = df["service_type"].fillna(SERVICE_TYPE_MISSING_FILL)
    return df


def fix_pass_status_and_derive_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    修正 pass_status 并由 fail_detail 派生条件触发二值特征。

    处理逻辑：
      1. 修正：fail_detail != "无" 时 pass_status 必为 fail（修正误标为 pass 的样本）
      2. 派生 cond1_triggered：fail_detail 含"条件1"则为 1，否则 0
      3. 派生 cond2_triggered：fail_detail 含"条件2"则为 1，否则 0

    当前数据 fail_detail 非空时均同时含条件1/条件2（两者同时触发），拆成两个独立二值特征
    便于未来出现单条件触发时直接使用。

    Returns:
        新增 cond1_triggered / cond2_triggered 列、修正 pass_status 的 DataFrame（副本）
    """
    df = df.copy()
    fd = df["fail_detail"].fillna(FAIL_DETAIL_PASS_TOKEN).astype(str)

    # 1. 修正 pass_status：fail_detail != "无" 时必为 fail
    fail_mask = fd != FAIL_DETAIL_PASS_TOKEN
    if "pass_status" in df.columns:
        df.loc[fail_mask, "pass_status"] = "fail"

    # 2. 派生条件触发二值特征
    df[COND1_TRIGGERED_COL] = (fd.str.contains("条件1", na=False)).astype(int)
    df[COND2_TRIGGERED_COL] = (fd.str.contains("条件2", na=False)).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    执行全部特征工程，返回增强后的 DataFrame。

    顺序：ftu 组合 -> service_type 填充 -> pass_status 修正与条件派生。
    各步骤独立，顺序无强依赖。
    """
    df = build_ftu_combo(df)
    df = fill_service_type(df)
    df = fix_pass_status_and_derive_conditions(df)
    return df
