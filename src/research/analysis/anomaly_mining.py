import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Literal
from src.research.models.base import BaseRegressor


def export_prediction_anomalies(
    models: Union[BaseRegressor, Dict[str, BaseRegressor]],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    output_dir: Union[str, Path],
    error_type: Literal["rel", "abs"],
    top_k: int = 100,
    eps: float = 1e-8,
):
    """
    找出模型预测异常样本，还原原数据并储存

    Args:
    -


    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_model = not isinstance(models, dict)

    for target in target_cols:
        model = models if single_model else models[target]
        all_records = []

        for split_name, df in [("train", train), ("valid", valid)]:
            if target not in df.columns:
                raise KeyError(f"{split_name} 数据中不存在目标列 {target}")
            y_true = df[target].values
            y_pred = model.predict(df[feature_cols])

            if error_type == "rel":
                error = np.abs(y_pred - y_true) / (np.abs(y_true) + eps)
            elif error_type == "abs":
                error = np.abs(y_pred - y_true)

            top_k_idx = np.argsort(error)[-top_k:][::-1]
            top_k_df = df.iloc[top_k_idx].copy()
            top_k_df[f"{target}_pred"] = y_pred[top_k_idx].astype(float)
            top_k_df[f"{target}_{error_type}_error"] = error[top_k_idx].astype(float)
            top_k_df["data_split"] = split_name
            all_records.append(top_k_df)

        if not all_records:
            continue
        anomaly_df = pd.concat(all_records, axis=0)
        anomaly_df[[f"{target}_pred", f"{target}_{error_type}_error"]] = anomaly_df[
            [f"{target}_pred", f"{target}_{error_type}_error"]
        ].astype(float)
        save_path = output_dir / f"{target}_anomalies.csv"
        anomaly_df.to_csv(save_path, index=False, float_format="%.6f")
        print(f"目标变量 {target} 异常样本 Top {top_k} 已保至 {save_path}")
