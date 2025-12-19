import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Any


class BaseRegressor(ABC):
    """
    回归模型基类
    """

    def __init__(self, feature_cols: List[str], random_state: int, multi_target=False):
        """
        args:
        - feature_cols(List[str]): 用于模型训练的特征列名
        - random_state(int): 随机种子
        """
        self.feature_cols = feature_cols
        self.random_state = random_state
        self.multi_target = multi_target
        self.model: Any = None

    @abstractmethod
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, target_cols: List[str]):
        """
        训练模型

        args:
        - train(pd.DataFrame): 训练数据 DataFrame
        - valid(pd.DataFrame): 验证数据 DataFrame
        - target_cols(List[str]) 目标变量
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        使用训练好的模型进行预测
        """
        pass
