import math
from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec


def get_target_describe_txt(
    series: pd.Series,
    precision: int = 2,
) -> str:
    """
    得到序列的统计信息描述

    Args：
    - series(pd.Series): 目标变量
    - precision(int): 小数精度

    Returns:
    - str: 统计信息描述文本
    """
    desc = series.describe()
    fmt = f"{{:.{precision}f}}"
    return (
        f"mean = {fmt.format(desc['mean'])}\n"
        f"std  = {fmt.format(desc['std'])}\n"
        f"min  = {fmt.format(desc['min'])}\n"
        f"25%  = {fmt.format(desc['25%'])}\n"
        f"50%  = {fmt.format(desc['50%'])}\n"
        f"75%  = {fmt.format(desc['75%'])}\n"
        f"max  = {fmt.format(desc['max'])}"
    )


def plot_target_feature_scatter_with_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    max_cols: int = 4,
    output_dir: Optional[Union[str, Path]] = None,
    fig_dpi: int = 150,
    scatter_alpha: float = 0.4,
    scatter_size: int = 10,
):
    """
    绘制 target vs 所有特征散点图，并在图中右侧显示 target 统计信息

    Args:
    - df(pd.DataFrame): 输入数据
    - feature_cols(List[str]): 特征列名
    - target_col(str): 目标变量列名
    - max_cols(int): 每行最多子图数
    - output_dir(Path, optional): 保存路径(None则直接show)
    - fig_dpi (int): 图片分辨率
    - scatter_alpha (float): 散点透明度
    - scatter_size (int): 散点大小
    """
    missing = set(feature_cols + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"缺少列名：{missing}")
    df_plot = df[feature_cols + [target]].copy()
    stats_text = get_target_describe_txt(df_plot[target])
    # 作图
    n_feats = len(feature_cols)
    n_cols = min(max_cols, n_feats)
    n_rows = math.ceil(n_feats / n_cols)
    fig = plt.figure(figsize=(5 * n_cols + 4, 4 * n_rows))
    gs = GridSpec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.8], figure=fig)
    for i, feat in enumerate(feature_cols):
        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c])
        ax.scatter(df_plot[feat], df_plot[target], alpha=scatter_alpha, s=scatter_size)
        ax.set_xlabel(feat)
        ax.set_ylabel(target)
        ax.set_title(feat)
        ax.grid(True)
    info_ax = fig.add_subplot(gs[:, -1])
    info_ax.axis("off")
    info_ax.text(
        0.0,
        1.0,
        f"Target: {target}\n{stats_text}",
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    fig.suptitle(
        f"Target vs Features Scatter & Distribution Stats", fontsize=16, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 保存
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{target}_vs_features.png"
        plt.savefig(save_path, dpi=fig_dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def batch_plot_targets(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    output_dir: Union[str, Path],
    max_cols: int = 4,
):
    """
    绘制多个 target 的 target-feature 散点分布图

    Args:
    - df(pd.DataFrame): 数据集
    - feature_cols(List[str]): 特征列
    - target_cols(List[str]): 多个目标变量
    - output_dir(Path): 输出目录
    - max_cols(int): 每行子图数
    """
    for tgt in target_cols:
        print(f"Plotting {tgt}")
        plot_target_feature_scatter_with_stats(
            df=df,
            feature_cols=feature_cols,
            target=tgt,
            max_cols=max_cols,
            output_dir=output_dir,
        )
