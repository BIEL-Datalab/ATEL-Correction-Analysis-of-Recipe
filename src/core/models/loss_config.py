"""
目标列损失配置

按目标列名后缀（统计量类型）映射到合适的损失函数与方向，使不同统计量使用匹配的损失：
  - mean / variance：中心趋势统计量，用对称损失（MSE/Huber）
  - max / 2max：极值上限统计量，业务上"宁高勿低"（保守侧），用高分位损失（quantile q≈0.8）
  - min / 2min：极值下限统计量，业务上"宁低勿高"（保守侧），用低分位损失（quantile q≈0.2）

非对称方向的业务含义（保守侧）：
  - max/2max 用高分位 q：模型预测偏低于真实时受更大惩罚，从而"宁可高估极值上限"，
    避免漏报真实上限导致 NG 风险被低估。
  - min/2min 用低分位 q：模型预测偏高于真实时受更大惩罚，从而"宁可低估极值下限"，
    避免漏报真实下限。

各库原生支持分位数损失（已实测可跑）：
  - XGBoost 2.1.1：reg:quantileerror + quantile_alpha
  - LightGBM：objective='quantile' + alpha
  - CatBoost：loss_function='Quantile:alpha=...'
  - GANDALF/TabM（深度模型）：自定义 quantile loss

注意：quantile 损失的 alpha 决定方向。q=0.8 对应"高估偏好"，q=0.2 对应"低估偏好"。
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

# 统计量类型（目标列名后缀）
StatType = Literal["mean", "variance", "max", "min", "2max", "2min"]

# 默认分位数（保守侧：极值宁高勿低/宁低勿高）
DEFAULT_UPPER_QUANTILE = 0.8  # max / 2max 用
DEFAULT_LOWER_QUANTILE = 0.2  # min / 2min 用


@dataclass(frozen=True)
class LossSpec:
    """单个目标的损失规格。

    Attributes:
        loss_type: 损失类型——'squared'(MSE) / 'huber' / 'quantile'
        quantile: 分位数，仅 loss_type='quantile' 时有效。q>0.5 偏好高估，
            q<0.5 偏好低估，q=0.5 退化为中位数（对称）。
        direction: 方向标记，便于日志与下游分析：'upper'(宁高) / 'lower'(宁低) / 'sym'(对称)
    """

    loss_type: Literal["squared", "huber", "quantile"]
    quantile: Optional[float] = None
    direction: Literal["upper", "lower", "sym"] = "sym"

    def __post_init__(self) -> None:
        if self.loss_type == "quantile":
            if self.quantile is None or not 0.0 < self.quantile < 1.0:
                raise ValueError(
                    f"quantile 损失需提供 (0,1) 内的 quantile，当前: {self.quantile}"
                )


# 统计量 -> 损失规格的默认映射表
_DEFAULT_STAT_LOSS: Dict[str, LossSpec] = {
    "mean": LossSpec("squared", direction="sym"),
    "variance": LossSpec("squared", direction="sym"),
    "max": LossSpec("quantile", DEFAULT_UPPER_QUANTILE, "upper"),
    "2max": LossSpec("quantile", DEFAULT_UPPER_QUANTILE, "upper"),
    "min": LossSpec("quantile", DEFAULT_LOWER_QUANTILE, "lower"),
    "2min": LossSpec("quantile", DEFAULT_LOWER_QUANTILE, "lower"),
}


def stat_of(target_col: str) -> str:
    """
    从目标列名提取统计量类型（最后一个下划线后的部分）。

    例：'c3_10_a_2min' -> '2min'；'t_0deg_940_mean' -> 'mean'。
    若无法识别返回 'mean'（降级为对称损失，避免未知后缀中断流程）。
    """
    for stat in ("2max", "2min", "mean", "variance", "max", "min"):
        if target_col.endswith("_" + stat):
            return stat
    return "mean"


def get_loss_spec(
    target_col: str,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    override: Optional[Dict[str, LossSpec]] = None,
) -> LossSpec:
    """
    根据目标列名得到损失规格。

    Args:
        target_col: 目标列名
        upper_quantile: max/2max 用的分位数（默认 0.8，保守侧）
        lower_quantile: min/2min 用的分位数（默认 0.2，保守侧）
        override: 覆盖默认映射，{target_col: LossSpec}，优先级最高

    Returns:
        LossSpec 损失规格
    """
    if override and target_col in override:
        return override[target_col]
    stat = stat_of(target_col)
    spec = _DEFAULT_STAT_LOSS[stat]
    # 用调用层传入的分位数覆盖默认值（保持方向不变）
    if spec.loss_type == "quantile":
        q = upper_quantile if spec.direction == "upper" else lower_quantile
        return LossSpec("quantile", q, spec.direction)
    return spec


def build_loss_map(
    target_cols: list,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    override: Optional[Dict[str, LossSpec]] = None,
) -> Dict[str, LossSpec]:
    """
    批量构建 {target_col: LossSpec}，供训练/评估统一查询损失配置。

    Args:
        target_cols: 目标列名列表
        upper_quantile / lower_quantile: 见 get_loss_spec
        override: 见 get_loss_spec

    Returns:
        {target_col: LossSpec}
    """
    return {
        c: get_loss_spec(c, upper_quantile, lower_quantile, override)
        for c in target_cols
    }


def loss_map_summary(loss_map: Dict[str, LossSpec]) -> Dict[str, int]:
    """统计损失分布，便于日志输出（如 {'quantile_upper': 16, 'squared': 16, ...}）。"""
    counts: Dict[str, int] = {}
    for spec in loss_map.values():
        key = f"{spec.loss_type}_{spec.direction}" if spec.loss_type == "quantile" else spec.loss_type
        counts[key] = counts.get(key, 0) + 1
    return counts
