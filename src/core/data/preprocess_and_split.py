"""
数据预处理与切分入口

串联完整数据流程：读取 parquet -> 列校验 -> 类型规范 -> 业务清洗 -> 日期筛选(可选)
-> 固定测试集 -> 训练/验证切分，产出训练就绪数据。

流程遵循"测试集在全量数据上固定、训练起点可扫描"的切分策略：
  1. 在全量清洗数据上按 time_index 取最后一段作测试集（固定不变）；
  2. 测试集前数据可按日期筛选训练起点（过滤久远数据）；
  3. 筛选后数据按 8:2 切训练/验证。

各步骤的细节实现分散在 column_schema / cleaning / splitting 三个模块，本文件只做编排。
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq

from src.core.constants.data_constants import (
    HC_CHAMBER_DAY_COL,
    DATE_FEATURE_COLS,
    NUM_FEATURE_COLS,
    CAT_FEATURE_COLS,
    Y_ALL_COLS,
)
from src.core.data.column_schema import check_column_consistency
from src.core.data.cleaning import clean_raw_data
from src.core.data.splitting import (
    holdout_test_set,
    split_train_valid,
    filter_by_date_range,
    SplitReport,
)

logger = logging.getLogger(__name__)


def business_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    业务相关预处理：解析时间列并派生 year/month/quarter 时间特征。

    hc_chamber_day 为镀膜日期，派生 year/month/quarter 供 EDA 分组分析与模型学习时间模式
    （真实生产存在年度高峰期、淡旺季等时间强相关情况）。day 粒度太细不派生。
    """
    df = df.copy()
    dt = pd.to_datetime(df[HC_CHAMBER_DAY_COL], errors="coerce")
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    return df


def normalize_feature_dtypes(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target_cols: List[str],
    date_feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    规范特征与目标列类型：数值特征转 numeric，分类特征转 str，目标转 numeric，
    时间派生特征（year/month/quarter）转 int。

    转换采用 errors="coerce"，非法值转 NaN 而非抛错，避免个别脏行中断流程。
    """
    df = df.copy()
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 时间派生特征应为整数类别
    for col in (date_feature_cols or DATE_FEATURE_COLS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def preprocess_and_split(
    raw_data_path: Union[Path, str],
    test_ratio: float = 0.1,
    train_ratio_in_rest: float = 0.8,
    date_filter_start: Optional[str] = None,
    date_filter_end: Optional[str] = None,
    output_dir: Optional[Union[Path, str]] = None,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    端到端数据预处理与切分，产出训练/验证/测试三份数据。

    Args:
        raw_data_path: 原始 parquet 路径
        test_ratio: 测试集占全量比例（固定取最后一段）
        train_ratio_in_rest: 测试集前数据中训练集占比（默认 0.8）
        date_filter_start: 训练起点日期（含），过滤久远数据；None 表示不筛
        date_filter_end: 训练终点日期（含）；None 表示不设上界
        output_dir: 若提供，则将三份数据与 meta 落盘到该目录
        num_cols / cat_cols / target_cols: 可选，默认取 data_constants 的全集定义

    Returns:
        train_df, valid_df, test_df, meta（含清洗与切分报告）
    """
    raw_data_path = Path(raw_data_path)
    num_cols = num_cols if num_cols is not None else NUM_FEATURE_COLS
    cat_cols = cat_cols if cat_cols is not None else CAT_FEATURE_COLS
    target_cols = target_cols if target_cols is not None else Y_ALL_COLS

    # 1. 读取 parquet（仅 schema 先做列校验，再按需加载列）
    all_cols = pq.ParquetFile(raw_data_path).schema.names
    consistency = check_column_consistency(all_cols)
    if consistency["missing"]:
        logger.warning(f"列定义存在真实数据中缺失的列: {consistency['missing']}")
    if consistency["unused"]:
        logger.info(f"真实数据中未使用的列: {consistency['unused']}")

    # 需要加载的列：特征 + 目标 + 标识/时间 + 时间派生依赖的 hc_chamber_day + ftu/fail_detail 原始列
    # 派生列（ftu_combo/cond*_triggered/year/month/quarter）不在原始数据中，加载其依赖的原始列
    from src.core.constants.data_constants import (
        IDENTIFIER_COLS, DERIVED_COLS, DERIVED_COL_SOURCES,
        FTU_COLS,
    )
    # 派生列依赖的原始列也要加载
    derived_source_cols = []
    for dcol in DERIVED_COLS:
        derived_source_cols.extend(DERIVED_COL_SOURCES.get(dcol, []))
    load_cols = list(
        dict.fromkeys(
            num_cols + cat_cols + target_cols + IDENTIFIER_COLS
            + [HC_CHAMBER_DAY_COL] + FTU_COLS + ["fail_detail"] + derived_source_cols
        )
    )
    # 去掉派生列本身（它们不在原始数据中，加载时不存在）
    from src.core.constants.data_constants import DERIVED_COLS as _DC
    load_cols = [c for c in load_cols if c not in _DC and c in all_cols]
    df = pd.read_parquet(raw_data_path, columns=load_cols)
    logger.info(f"读取原始数据: {df.shape}")

    # 2. 业务预处理 + 类型规范
    df = business_preprocess(df)
    df = normalize_feature_dtypes(
        df, num_cols, cat_cols, target_cols, date_feature_cols=DATE_FEATURE_COLS
    )

    # 3. 清洗（cathode 离群置空、丢弃 Y 全缺/越界）
    cleaned_df, cleaning_report = clean_raw_data(df)
    logger.info(f"清洗完成:\n{cleaning_report.summary()}")

    # 3.1 特征工程：ftu 组合、service_type 填充、pass_status 修正与条件触发派生
    from src.core.data.feature_engineering import engineer_features
    cleaned_df = engineer_features(cleaned_df)
    logger.info("特征工程完成: ftu_combo / service_type 填充 / pass_status 修正 / cond 触发派生")

    # 4. 固定测试集（在全量清洗数据上取最后 test_ratio）
    rest_df, test_df, test_report = holdout_test_set(cleaned_df, test_ratio=test_ratio)
    logger.info(f"测试集固定:\n{test_report.summary()}")

    # 5. 测试集前数据按日期筛选训练起点（可选）
    if date_filter_start is not None or date_filter_end is not None:
        rest_df, filter_report = filter_by_date_range(
            rest_df, start=date_filter_start, end=date_filter_end
        )
        logger.info(f"日期筛选:\n{filter_report.summary()}")

    # 6. 训练/验证切分
    train_df, valid_df, tv_report = split_train_valid(
        rest_df, train_ratio=train_ratio_in_rest
    )
    logger.info(f"训练/验证切分:\n{tv_report.summary()}")

    # 7. 汇总 meta
    from src.core.constants.data_constants import DERIVED_COLS
    meta = {
        "raw_data_path": str(raw_data_path),
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "shape": {
            "raw": df.shape,
            "cleaned": cleaned_df.shape,
            "train": train_df.shape,
            "valid": valid_df.shape,
            "test": test_df.shape,
        },
        "test_ratio": test_ratio,
        "train_ratio_in_rest": train_ratio_in_rest,
        "date_filter": {"start": date_filter_start, "end": date_filter_end},
        "column_consistency": consistency,
        "cleaning": asdict(cleaning_report),
        "test_split": _report_to_dict(test_report),
        "train_valid_split": _report_to_dict(tv_report),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_cols": target_cols,
        "date_feature_cols": DATE_FEATURE_COLS,
        "derived_cols": DERIVED_COLS,
    }

    # 8. 落盘
    if output_dir is not None:
        _save_outputs(train_df, valid_df, test_df, meta, Path(output_dir))

    return train_df, valid_df, test_df, meta


def _report_to_dict(report: SplitReport) -> dict:
    """SplitReport 转可序列化 dict。"""
    return asdict(report)


def _save_outputs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> None:
    """将三份数据与 meta 落盘到 outputs/data/{date}/。"""
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"run_{run_stamp}"
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out / "train.parquet", index=False)
    valid_df.to_parquet(out / "valid.parquet", index=False)
    test_df.to_parquet(out / "test.parquet", index=False)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"训练数据已落盘: {out}")
