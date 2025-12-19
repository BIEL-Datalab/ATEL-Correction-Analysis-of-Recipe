import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TabularPreprocessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.label_maps = {}
        self.inverse_label_maps = {}

    def fit(self, df):
        # 数值特征处理
        self.scaler.fit(df[self.numerical_cols])
        # 分类特征处理
        for col in self.categorical_cols:
            le = LabelEncoder()
            # 处理缺失值
            vals = df[col].fillna("missing").astype(str).values
            le.fit(vals)
            self.label_encoders[col] = le
            # 向前映射
            self.label_maps[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}
            # 向后映射
            self.inverse_label_maps[col] = {
                int(i): cls for i, cls in enumerate(le.classes_)
            }

    def transform(self, df):
        df = df.copy()
        # 数值特征
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols]).astype(
            np.float32
        )
        # 分类特征
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            # 新类别映射
            vals = df[col].fillna("missing").astype(str).values
            safe_vals = [v if v in le.classes_ else "missing" for v in vals]
            df[col] = le.transform(safe_vals).astype(np.int32)
        return df

    def inverse_transform_categorical(self, df):
        # 预测完后恢复原始类别
        df = df.copy()
        for col in self.categorical_cols:
            inv_map = self.inverse_label_maps[col]
            df[col] = df[col].map(inv_map)
        return df

    def save(self, path):
        obj = {
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "label_maps": self.label_maps,
            "inverse_label_maps": self.inverse_label_maps,
        }
        with open(path, "w") as f:
            json.dump(obj, f, indent=4)

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            obj = json.load(f)

        pre = TabularPreprocessor(
            numerical_cols=obj["numerical_cols"],
            categorical_cols=obj["categorical_cols"],
        )

        # 恢复 scaler
        pre.scaler.mean_ = np.array(obj["scaler_mean"])
        pre.scaler.scale_ = np.array(obj["scaler_scale"])

        # 恢复 label maps
        pre.label_maps = obj["label_maps"]
        pre.inverse_label_maps = obj["inverse_label_maps"]

        # 创建新的 LabelEncoder 并加载 classes_
        for col, mp in pre.label_maps.items():
            le = LabelEncoder()
            # 反向构造 classes_
            classes = np.array(
                [cls for cls, idx in sorted(mp.items(), key=lambda x: x[1])]
            )
            le.classes_ = classes
            pre.label_encoders[col] = le

        return pre
