import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Literal, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder


def business_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    业务相关的数据预处理
    """
    df = df.copy()
    # TODO 后续在原始数据获取层面解决
    df = df.rename(columns={"index": "row_id"})
    df = df[df["is_centralized"] != "fail"].reset_index(drop=True)
    # 时间变量处理
    df["hc_chamber_day"] = pd.to_datetime(
        df["hc_chamber_day"], format="%Y-%m-%d", errors="raise"
    )
    df["year"] = df["hc_chamber_day"].dt.year
    df["month"] = df["hc_chamber_day"].dt.month
    df["day"] = df["hc_chamber_day"].dt.day
    return df


class TabularPreprocessor:

    MISSING_TOKEN = "missing"

    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        target_cols: List[str] | None = None,
        normalize_targets: bool = False,
        target_transform: Literal["standard"] = "standard",
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_cols = target_cols or []
        self.normalize_targets = normalize_targets
        self.target_transform = target_transform
        # X
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.label_maps = {}
        self.inverse_label_maps = {}
        self.feature_cols_: Optional[List[str]] = None
        # y
        self.target_scalers: Dict[str, StandardScaler] = {}

    def fit(self, df: pd.DataFrame):
        # 数值特征处理
        df = df.copy()
        # 记录次序
        self.feature_cols_ = self.numerical_cols + self.categorical_cols
        missing = set(self.feature_cols_ + self.target_cols) - set(df.columns)
        if missing:
            raise ValueError(f"缺少列名：{missing}")
        # 数值特征处理
        if self.numerical_cols:
            self.scaler.fit(df[self.numerical_cols])
        # 分类特征处理
        for col in self.categorical_cols:
            le = LabelEncoder()
            # 处理缺失值
            vals = df[col].fillna(self.MISSING_TOKEN).astype(str).values
            if self.MISSING_TOKEN not in vals:
                vals = np.append(vals, self.MISSING_TOKEN)
            le.fit(vals)
            self.label_encoders[col] = le
            self.label_maps[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}
            self.inverse_label_maps[col] = {
                int(i): cls for i, cls in enumerate(le.classes_)
            }
        # 特征变量处理
        if self.normalize_targets and len(self.target_cols) > 0:
            for tgt in self.target_cols:
                y = df[tgt].values.reshape(-1, 1)
                if self.target_transform == "standard":
                    self.target_scalers[tgt] = StandardScaler().fit(y)
                else:
                    raise Exception("暂不支持的标准化类型")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols_ is None:
            raise RuntimeError("TabularPreprocessor 在 transform 之前需要先 fit")
        df = df.copy()
        missing = set(self.feature_cols_ + self.target_cols) - set(df.columns)
        if missing:
            raise ValueError(f"缺少列名：{missing}")
        # 数值特征
        if self.numerical_cols:
            df[self.numerical_cols] = self.scaler.transform(
                df[self.numerical_cols]
            ).astype(np.float32)
        # 分类特征
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            vals = df[col].fillna(self.MISSING_TOKEN).astype(str).values
            safe_vals = [v if v in le.classes_ else self.MISSING_TOKEN for v in vals]
            df[col] = le.transform(safe_vals).astype(np.int32)
        # 目标变量
        for tgt in self.target_cols:
            y = df[tgt].values.reshape(-1, 1)
            if self.normalize_targets:
                y = self.target_scalers[tgt].transform(y)
            df[tgt] = y.astype(np.float32)

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将处理后的数据还原为原始数据

        Args:
        - df(pd.DataFrame): transform 后的数据

        Returns:
        - pd.DataFrame: 还原后的原始数据
        """
        df = df.copy()
        # 数值变量还原
        if self.numerical_cols:
            df[self.numerical_cols] = self.scaler.inverse_transform(
                df[self.numerical_cols]
            )
        # 类别变量还原
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            df[col] = le.inverse_transform(df[col].astype(int))
        # 目标变量还原
        for tgt in self.target_cols:
            y = df[tgt].values.reshape(-1, 1)
            if self.normalize_targets:
                y = self.target_scalers[tgt].inverse_transform(y)
            df[tgt] = y

        return df

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "feature_cols": self.feature_cols_,
            "target_cols": self.target_cols,
            "normalize_target": self.normalize_targets,
            "target_transform": self.target_transform,
            "scaler": {
                "mean": self.scaler.mean_.tolist() if self.numerical_cols else None,
                "scale": self.scaler.scale_.tolist() if self.numerical_cols else None,
                "var": self.scaler.var_.tolist() if self.numerical_cols else None,
            },
            "label_maps": self.label_maps,
            "inverse_label_maps": self.inverse_label_maps,
            "target_stats": {},
        }
        if self.normalize_targets:
            for tgt in self.target_cols:
                scaler = self.target_scalers[tgt]
                obj["target_stats"][tgt] = {
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load(path: Union[str, Path]):
        path = Path(path)
        with open(
            path,
            "r",
        ) as f:
            obj = json.load(f)

        pre = TabularPreprocessor(
            numerical_cols=obj["numerical_cols"],
            categorical_cols=obj["categorical_cols"],
            target_cols=obj["target_cols"],
            normalize_target=obj["normalize_target"],
            target_transform=obj["target_transform"],
        )
        pre.feature_cols_ = obj["feature_cols"]
        # 恢复 scaler
        pre.scaler.mean_ = np.array(obj["scaler"]["mean"])
        pre.scaler.scale_ = np.array(obj["scaler"]["scale"])
        pre.scaler.var_ = np.array(obj["scaler"]["var"])

        # 恢复 label maps
        if obj["scaler"]["mean"] is not None:
            pre.scaler.mean_ = np.array(obj["scaler"]["mean"])
            pre.scaler.scale_ = np.array(obj["scaler"]["scale"])
            pre.scaler.var_ = np.array(obj["scaler"]["var"])

        pre.label_maps = obj["label_maps"]
        pre.inverse_label_maps = obj["inverse_label_maps"]

        for col, mp in pre.label_maps.items():
            le = LabelEncoder()
            classes = np.array(
                [cls for cls, idx in sorted(mp.items(), key=lambda x: x[1])]
            )
            le.classes_ = classes
            pre.label_encoders[col] = le

        # 目标变量
        if pre.normalize_target and obj["target_stats"]:
            for tgt, stats in obj["target_stats"].items():
                if pre.target_transform == "standard":
                    scaler = StandardScaler()
                    scaler.mean_ = np.array(stats["mean"])
                    scaler.scale_ = np.array(stats["scale"])
                    pre.target_scalers[tgt] = scaler
                else:
                    raise Exception("暂不支持的类型")
        return pre
