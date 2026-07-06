"""
回归模型基类

定义所有回归模型（树模型 XGB/LGBM/CatBoost 与深度模型 GANDALF/TabM）的统一契约。

设计要点：
  - 特征列**不在模型构造时硬编码**，而由调用层（run_regression / notebook）从数据 meta
    传入，避免模型类与数据常量耦合，也便于未来特征列变动时只改一处。
  - 区分"单目标"与"多目标"两类模型：单目标模型一次只拟合一个 target，多目标模型一次
    拟合多个 target。基类用 multi_target 标记，供评估器等下游选择不同处理路径。
  - 统一 save/load 契约：每个模型落盘时产出模型权重文件 + meta（json/pkl），记录特征列、
    目标列、最优参数、评估指标等，保证 load 后即可 predict 且接口与训练产物对称。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class BaseRegressor(ABC):
    """
    回归模型基类。

    子类需实现 fit / predict / save_model / load_model。特征列通过 fit 的参数注入，
    而非类属性硬编码——这样同一模型类可服务不同特征集，且训练时的特征顺序在保存后
    加载时一致还原，避免特征错位。
    """

    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        date_cols: Optional[List[str]] = None,
        random_state: int = 42,
        multi_target: bool = False,
    ):
        """
        Args:
            num_cols: 数值特征列名；为 None 时表示未指定，需在 fit 前由调用层补全。
            cat_cols: 分类特征列名；同上。
            date_cols: 时间派生特征列名（如 year/month/quarter），深度模型可能需要单独
                处理；树模型通常将其作为数值或分类特征并入 num_cols/cat_cols。
            random_state: 随机种子，保证可复现。
            multi_target: 是否多目标。单目标模型 fit 时 target_cols 长度应为 1。
        """
        self.num_cols: List[str] = list(num_cols) if num_cols is not None else []
        self.cat_cols: List[str] = list(cat_cols) if cat_cols is not None else []
        self.date_cols: List[str] = list(date_cols) if date_cols is not None else []
        self.feature_cols: List[str] = self.num_cols + self.cat_cols
        self.random_state = random_state
        self.multi_target = multi_target
        # 训练产物：未训练时为 None，避免访问未初始化属性
        self.model: Any = None
        self.target_cols: Optional[List[str]] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.metrics: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # 特征列管理：允许调用层在构造后、fit 前补全特征列
    # ------------------------------------------------------------------
    def set_feature_cols(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        date_cols: Optional[List[str]] = None,
    ) -> None:
        """
        更新特征列（构造后补全或覆盖）。feature_cols 同步重建为 num + cat 的拼接顺序。

        保留动机：CLI/notebook 常先实例化模型再从数据 meta 读特征列，需要一条显式入口。
        """
        if num_cols is not None:
            self.num_cols = list(num_cols)
        if cat_cols is not None:
            self.cat_cols = list(cat_cols)
        if date_cols is not None:
            self.date_cols = list(date_cols)
        self.feature_cols = self.num_cols + self.cat_cols

    # ------------------------------------------------------------------
    # 抽象方法
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_cols: List[str],
    ) -> Any:
        """
        训练模型。

        Args:
            train: 训练集 DataFrame，需含 feature_cols 与 target_cols。
            valid: 验证集 DataFrame，同上。
            target_cols: 目标列名。单目标模型长度应为 1；多目标模型可为多个。
        """
        ...

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, Any]) -> Any:
        """
        使用训练好的模型预测。predict 前必须已 fit 或 load_model。"""
        ...

    @abstractmethod
    def save_model(self, save_dir: Union[str, Path]) -> None:
        """保存模型权重与 meta（特征列/目标列/参数/指标）到目录。"""
        ...

    # ------------------------------------------------------------------
    # 通用校验
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        """predict / save 前校验模型已训练。"""
        if self.model is None:
            raise RuntimeError("模型未训练或未加载，无法执行该操作")

    def _check_feature_cols(self) -> None:
        """校验特征列已配置（非空），避免空特征列静默训练。"""
        if not self.feature_cols:
            raise ValueError(
                "feature_cols 为空：请在实例化时传入 num_cols/cat_cols，"
                "或调用 set_feature_cols() 补全"
            )
