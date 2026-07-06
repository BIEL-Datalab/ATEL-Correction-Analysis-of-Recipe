"""
树模型分类特征编码工具

统一处理树模型（XGB/LGBM/CatBoost）的分类特征缺失填充与未知类别（OOV）映射，
避免每个树模型重复实现且行为不一致。

设计要点：
  - 训练时记录每列的类别集合 categories（有序、去重），存入 cat_mappings；
  - 缺失值统一填充为 MISSING_TOKEN；
  - 推理时遇到训练未见类别，回退为 MISSING_TOKEN（未知类别与缺失等同处理），
    避免未来新机器/新故障码导致 predict 崩溃；
  - 各树模型对 category 类型的要求略有差异（XGB/LGBM 用 pandas category dtype，
    CatBoost 用 Pool 的 cat_features 索引），本模块只负责"值"的归一，dtype 交给各模型。
"""

from typing import Dict, List

import pandas as pd

from src.core.constants.train_constants import MISSING_TOKEN


def fit_cat_mappings(
    df: pd.DataFrame,
    cat_cols: List[str],
    missing_token: str = MISSING_TOKEN,
) -> Dict[str, List[str]]:
    """
    扫描训练集，为每个分类列构建有序类别集合（含 missing_token 兜底）。

    Args:
        df: 训练集（或合并 train+valid）
        cat_cols: 分类列名
        missing_token: 缺失填充值，会作为最后一个类别加入，保证推理时未知/缺失都能命中

    Returns:
        {cat_col: [category1, category2, ..., missing_token]}
    """
    mappings: Dict[str, List[str]] = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        vals = df[col].fillna(missing_token).astype(str)
        cats = sorted(vals.unique().tolist())
        # 确保 missing_token 在类别集合中（训练时该列可能无缺失）
        if missing_token not in cats:
            cats.append(missing_token)
        mappings[col] = cats
    return mappings


def transform_cat_cols(
    df: pd.DataFrame,
    cat_cols: List[str],
    cat_mappings: Dict[str, List[str]],
    missing_token: str = MISSING_TOKEN,
) -> pd.DataFrame:
    """
    按已拟合的 cat_mappings 将 df 的分类列归一为字符串：
      - 缺失 -> missing_token
      - 训练见过的值 -> 原值
      - 训练未见过的值 -> missing_token（OOV 兜底）

    返回副本，不修改输入。dtype 留给各树模型按需转换（category / 原始）。
    """
    out = df.copy()
    for col in cat_cols:
        if col not in out.columns:
            continue
        cats = cat_mappings.get(col, [missing_token])
        out[col] = (
            out[col]
            .fillna(missing_token)
            .astype(str)
            .apply(lambda x: x if x in cats else missing_token)
        )
    return out


def to_category_dtype(
    df: pd.DataFrame,
    cat_cols: List[str],
    cat_mappings: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    将分类列转为 pandas category dtype，类别集合严格限定为 cat_mappings 中的值。
    XGB / LGBM 需要这种形式以原生支持分类特征。
    """
    out = df.copy()
    for col in cat_cols:
        if col not in out.columns:
            continue
        cats = cat_mappings.get(col)
        if cats is None:
            continue
        out[col] = pd.Categorical(out[col], categories=cats)
    return out
