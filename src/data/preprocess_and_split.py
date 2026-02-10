import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Literal, Dict, Optional
from sklearn.model_selection import train_test_split


def business_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    业务相关的数据预处理
    1. 转换日期变量并提取年月日作为特征
    """
    df = df.copy()
    # TODO 后续在原始数据获取层面解决
    # 时间变量处理
    df["hc_chamber_day"] = pd.to_datetime(
        df["hc_chamber_day"], format="%Y-%m-%d", errors="raise"
    )
    df["year"] = df["hc_chamber_day"].dt.year
    df["month"] = df["hc_chamber_day"].dt.month
    df["day"] = df["hc_chamber_day"].dt.day
    return df


def normalize_feature_dtypes(
    df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], target_cols: List[str]
) -> pd.DataFrame:
    """
    标准化特征和目标列的数值类型

    Args:
    - df(pd.DataFrame): 数据集 DataFrame
    - num_cols(List[str]): 数值特征列列表
    - cat_cols(List[str]): 分类特征列列表
    - target_cols(List[str]): 目标列列表

    Returns:
    - df(pd.DataFrame): 规范后的 DataFrame
    """
    df = df.copy()
    # 数值特征
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 分类特征
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    # 目标变量
    if target_cols:
        for col in target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preprocess_and_split(
    raw_data_path: Union[Path, str],
    num_cols: List[str],
    cat_cols: List[str],
    target_cols: List[str],
    other_info_cols: List[str],
    raw_date_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    读取数据，业务处理，特征类型规范，切分训练测试集

    Args:
    -

    Return:
    - df(pd.DataFrame): 原始数据集
    - train_df(pd.DataFrame): 训练数据集
    - valid_df(pd.DataFrame): 验证数据集
    """
    df = pd.read_csv(Path(raw_data_path))
    df = business_preprocess(df)
    need_cols = other_info_cols + raw_date_cols + num_cols + cat_cols + target_cols
    df = df[need_cols]
    df = normalize_feature_dtypes(df, num_cols, cat_cols, target_cols)
    train_df, valid_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return df, train_df, valid_df
