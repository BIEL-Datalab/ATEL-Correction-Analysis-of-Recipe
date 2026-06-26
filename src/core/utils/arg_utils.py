import argparse
from typing import List, Tuple
from src.core.models.registry import MODEL_REGISTRY
from src.core.constants.data_constants import *
from src.core.constants.train_constants import *


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Unified Regression Runner")
    parser.add_argument("--model_type", required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        required=True,
        help="""
            Execution mode. 
            'train' = train model and save to {--model_dir} / {--model_type}_{date} and run evaluation; 
            'eval' = load existing model from {--model_dir} and run evaluation.
        """,
    )
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="""
            Directory path for model. 
            If mode = train, trained model and metadata will be saved here. 
            If mode = predict, model will be loaded from this directory.
        """,
    )
    parser.add_argument("--result_dir", type=str, required=True, help="")
    parser.add_argument("--log_dir", type=str, default=None)
    # 可选参数
    parser.add_argument("--target_cols", nargs="+", default=None)
    parser.add_argument("--num_cols", nargs="+", default=None)
    parser.add_argument("--cat_cols", nargs="+", default=None)
    parser.add_argument(
        "--date_cols",
        nargs="*",
        default=None,
        help="Date columns in format: col:freq:format, e.g. hc_chamber_day:ME:%Y-%m-%d",
    )
    parser.add_argument("--other_info_cols", nargs="+", default=OTHER_INFO_COLS)
    parser.add_argument("--num_boost_round", type=int, default=None)
    parser.add_argument("--early_stopping_rounds", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=TEST_SIZE)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--tgt_desc", type=bool, default=False)
    parser.add_argument("--anomaly_export", type=bool, default=False)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--acc", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument(
        "--tune_method", type=str, default="optuna", choices=["optuna", "tuner"]
    )
    return parser


def parse_date_cols(date_args: List[str]) -> List[Tuple[str, str, str]]:
    """
    将 CLI 传入的日期列参数解析为 pytorch-tabular 所需格式
    示例：
    输入为 ["hc_chamber_day:%Y-%m-%d:M"], 输出为 [("hc_chamber_day", "M", "%Y-%m-%d")]
    """
    date_cols = []
    for item in date_args:
        try:
            col, freq, fmt = item.split(":")
        except ValueError:
            raise ValueError(
                f"Invalid --date_cols format: {item}. "
                f"Expected format: col:freq:format"
            )
        date_cols.append((col, freq, fmt))
    return date_cols
