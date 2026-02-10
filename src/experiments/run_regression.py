"""
回归实验

支持：
- XGBoost (单目标模型)
- LightGBM (单目标模型)
- GANDALF (多目标模型)

功能：
1. 预处理数据，保存；
2. 模型调优、训练、保存；
3. 输出模型测评结果；
"""

import argparse
import logging
import pandas as pd
import numpy as np
import random
import warnings
import os
from typing import List, Tuple, Dict
from pathlib import Path
from datetime import datetime
from src.constants.data_constants import TARGET_GROUPS
from src.models.registry import MODEL_REGISTRY
from src.utils.arg_utils import build_arg_parser, parse_date_cols
from src.data.preprocess_and_split import preprocess_and_split
from src.models.base import BaseRegressor
from src.analysis.data_description import batch_plot_targets
from src.evaluation.evaluator import RegressionEvaluator

pd.set_option("display.max_columns", None)
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded")
logging.getLogger("shap").setLevel(logging.ERROR)
logging.getLogger("shap.explainers").setLevel(logging.ERROR)
logging.getLogger("shap.explainers._kernel").setLevel(logging.ERROR)


def choose_model(args: argparse.Namespace) -> Tuple[BaseRegressor, bool]:
    """
    选择模型
    """
    if args.model_type in ["xgb", "lgbm"]:
        return (
            MODEL_REGISTRY[args.model_type](
                num_cols=args.num_cols,
                cat_cols=args.cat_cols,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
                n_jobs=args.n_jobs,
                n_trials=args.n_trials,
                random_state=args.random_state,
            ),
            True,
        )
    else:
        return (
            MODEL_REGISTRY[args.model_type](
                continuous_cols=args.num_cols,
                categorical_cols=args.cat_cols,
                date_cols=(
                    parse_date_cols(args.date_cols)
                    if args.date_cols is not None
                    else None
                ),
                batch_size=args.batch_size,
                random_state=args.random_state,
                n_trials=args.n_trials,
                checkpoint_dir=Path(args.model_dir) / f"{args.model_type}_checkpoints",
                acc=args.acc,
            ),
            False,
        )


def prepare_target_groups(args: argparse.Namespace) -> Dict[str, List[str]]:
    """
    目标分组规则：若传入 target_cols 输出一个目标分组，未传入时输出默认目标分组
    """
    if args.target_cols:
        return {"custom": args.target_cols}
    return TARGET_GROUPS


def train_or_load_tree_models(
    args: argparse.Namespace,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_groups: Dict[str, List[str]],
    run_date: str,
) -> Dict[str, BaseRegressor]:
    """
    训练/加载树模型
    """
    all_models = {}
    model_dir = Path(args.model_dir)
    for group_name, tgts in target_groups.items():
        logging.info(
            f"训练 {args.model_type} 模型"
            if args.mode == "train"
            else f"加载 {args.model_type} 模型"
        )
        for tgt in tgts:
            print(f"训练 / 加载目标变量 {tgt} 模型")
            if args.mode == "train":
                model, _ = choose_model(args)
                model.fit(train, valid, [tgt])
                save_path = model_dir / f"{args.model_type}_{run_date}/models"
                model.save_model(save_path)
            else:
                # 加载模型目录下需要以目标变量名作为加载模型的代码路径
                modelcls = MODEL_REGISTRY[args.model_type]
                model = modelcls.load_model(model_dir, tgt)
            all_models[tgt] = model
            logging.info(
                f"目标变量 {tgt} 模型已保存在路径 {save_path} 下"
                if args.mode == "train"
                else f"目标变量 {tgt} 模型已加载"
            )
    return all_models


def train_or_load_multi_model(
    args: argparse.Namespace,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_groups: Dict[str, List[str]],
    run_date: str,
) -> Dict[str, BaseRegressor]:
    all_models = {}
    model_dir = Path(args.model_dir)
    for group_name, tgts in target_groups.items():
        if args.mode == "train":
            logging.info(f"训练 {args.model_type} 模型")
            model, _ = choose_model(args)
            save_path = (
                model_dir / f"{args.model_type}_{run_date}" / group_name / "model"
            )
            model.fit(train, valid, tgts, save_path / "checkpoint")
            model.save_model(save_path / "best_model")
            logging.info(
                f"目标变量 {tgts} 模型已保存在路径 {save_path / 'best_model'}  下"
            )
        else:
            logging.info(f"加载 {args.model_type} 模型")
            modelcls = MODEL_REGISTRY[args.model_type]
            model = modelcls.load_model(model_dir)
        all_models[group_name] = model


def run_regression(args: argparse.Namespace) -> None:
    """
    根据参数完成数据预处理、模型参数搜索、模型训练和保存，模型评估
    """
    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 随机种子
    np.random.seed(args.random_state)
    random.seed(args.random_state)
    os.environ["PYTHONHASHSEED"] = str(args.random_state)
    # 日志
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / f"{args.model_type}_{run_date}.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
            encoding="utf-8",
        )

    # 模型选择
    model, is_tree = choose_model(args)
    num_cols = model.num_cols if is_tree else model.continuous_cols
    cat_cols = model.cat_cols if is_tree else model.categorical_cols
    target_groups = prepare_target_groups(args)
    all_targets = [x for v in target_groups.values() for x in v]
    df, train, valid = preprocess_and_split(
        raw_data_path=args.raw_data_path,
        num_cols=num_cols,
        cat_cols=cat_cols,
        target_cols=all_targets,
        other_info_cols=args.other_info_cols,
        raw_date_cols=model.date_feature_names if not is_tree else [],
        test_size=args.test_size,
        random_state=args.random_state,
    )
    logging.info(
        f"""
        模型 {args.model_type} 数据集信息：\n
        分类变量：{model.cat_cols if is_tree else model.categorical_cols} \n
        数值变量：{model.num_cols if is_tree else model.continuous_cols} \n
        目标变量：{all_targets} \n
        训练集数据量 {len(train)} 条，验证集数据量 {len(valid)} 条。
    """
    )
    # 目标变量分析
    if args.tgt_desc:
        desc_path = (
            args.result_dir / f"target_analysis/target_desc/target_desc_{run_date}"
        )
        batch_plot_targets(
            df=df,
            feature_cols=num_cols + cat_cols,
            target_cols=all_targets,
            output_dir=desc_path,
        )
        logging.info(f"目标变量原始数据分析已保存，结果保存路径为 {desc_path} ")
    else:
        logging.info(f"未对目标变量原始数据进行分析")
    # 模型训练/加载和模型测评
    if is_tree:
        all_model = train_or_load_tree_models(
            args, train, valid, target_groups, run_date
        )
    else:
        all_model = train_or_load_multi_model(
            args, train, valid, target_groups, run_date
        )
    # 模型测评
    if is_tree:
        # 树模型下所有目标变量一起做测评
        eval_dir = Path(args.model_dir) / f"{args.model_type}_{run_date}/eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        evaluator = RegressionEvaluator(
            num_cols=num_cols, cat_cols=cat_cols, output_dir=eval_dir
        )
        evaluator.run_full_evaluation(
            regressors=all_model, train=train, valid=valid, target_cols=all_targets
        )
        logging.info(f"所有变量的模型测评结果已保存到 {eval_dir}")
    else:
        # 多目标模型下每个模型做一个测评
        for group_name, model in all_model:
            eval_dir = (
                Path(args.model_dir) / f"{args.model_type}_{run_date}/{group_name}/eval"
            )
            eval_dir.mkdir(parents=True, exist_ok=True)
            evaluator = RegressionEvaluator(
                num_cols=num_cols, cat_cols=cat_cols, output_dir=eval_dir
            )
            evaluator.run_full_evaluation(
                regressors=model,
                train=train,
                valid=valid,
                target_cols=target_groups[group_name],
            )
            logging.info(f"目标分类 {group_name} 的模型结果已保存到 {eval_dir}")

    # bad case 导出
    # TODO


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_regression(args)
