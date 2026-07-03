"""
原始数据列分组与命名常量

基于 raw_data/raw_data_260623/data.parquet（284 列 / 49799 行）的真实列名定义。
按业务含义划分为以下大类，供数据预处理、EDA 与建模统一引用：
  - S（设定参数）：模型寻优的调整对象，含 actual_rate / set_rate / rate_coefficient 三组（各 11 层）
  - M（监控参数）：生产中传感器实际值；本期仅纳入 cathode 耗材使用值与三类时长，
                  o2 / ar 两组传感器原始量量大且本期不分析，明确排除（见 EXCLUDE_GROUPS）
  - Y（质检输出）：颜色 c3_10 / c3_45 与透过率 t_ 三组，每组含 mean/variance/max/min/2max/2min
                  共 40 列，全部纳入预测目标（variance 同样作为锅次统计量纳入）
  - 环境/上下文：产品规格、楼层、维保、ftu 配置、来料质量等
  - 标识/时间：id（业务定位）/ machine_sn（机器差异）/ time_index（时间先后）/ start_time（定位日期）

设计说明：
  - 列名以 _l{n} 表示第 n 层（共 11 层），与真实数据一致
  - FEATURE_GROUPS / TARGET_GROUPS / IDENTIFIER_COLS 等以结构化字典组织，便于按需取用与扩展
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# 通用结构参数
# ---------------------------------------------------------------------------
N_LAYERS = 11  # 镀膜层数：S 三组与 cathode 均按 1..11 层组织
LAYER_INDEX = list(range(1, N_LAYERS + 1))


# ---------------------------------------------------------------------------
# S：设定参数（模型寻优的调整对象）
# ---------------------------------------------------------------------------
# actual_rate 为实际下发的生产速率，与 set_rate 配套使用，归入 S 设定类
S_ACTUAL_RATE_COLS: List[str] = [f"actual_rate_l{i}" for i in LAYER_INDEX]
S_SET_RATE_COLS: List[str] = [f"set_rate_l{i}" for i in LAYER_INDEX]
S_RATE_COEFFICIENT_COLS: List[str] = [f"rate_coefficient_l{i}" for i in LAYER_INDEX]

# S 全部列：用于"整批缺失"判定（所有 S 类列均缺失视为无设参的无效锅次）
S_ALL_COLS: List[str] = S_ACTUAL_RATE_COLS + S_SET_RATE_COLS + S_RATE_COEFFICIENT_COLS


# ---------------------------------------------------------------------------
# M：监控参数（生产中传感器实际值）
# ---------------------------------------------------------------------------
# cathode{k}_l{n}_usevalue：第 k 个耗材在第 n 层的使用值（k=1..3, n=1..11，共 33 列）
# 注意：缺失锅次会导致使用值出现离群点（消耗量=上一锅剩余-本锅剩余），离群仅标记不删除
M_CATHODE_COLS: List[str] = [
    f"cathode{k}_l{n}_usevalue" for k in range(1, 4) for n in LAYER_INDEX
]
# 三类时长（单位均为秒）：batch_production_duration 由 end_time - start_time 得到
M_DURATION_COLS: List[str] = [
    "batch_production_duration",
    "deposition_time",
    "pumping_time",
]

# 本期明确不纳入的 M 子组（传感器原始量大，暂不分析）
# o2_*：氧气参数；ar*_l*：ar 参数（120 列）
# o2 列形如 o2_{1,2,high,low}_l{4,11}
O2_COLS: List[str] = [
    f"o2_{tag}_l{layer}"
    for tag in ["1", "2", "high", "low"]
    for layer in ["4", "11"]
]
# ar 列形如 ar{k}_l{n}（k=1..10, n=0..11），共 120 列
AR_COLS: List[str] = [
    f"ar{k}_l{n}" for k in range(1, 11) for n in range(0, 12)
]
# 本期排除的 M 子组汇总：明确记录"暂不纳入"的列，避免被误以为遗漏
EXCLUDE_GROUPS: Dict[str, List[str]] = {"o2": O2_COLS, "ar": AR_COLS}


# ---------------------------------------------------------------------------
# Y：质检输出指标统计量（预测目标）
# ---------------------------------------------------------------------------
# 颜色指标 c3_10：a/b/y 三通道，每通道 6 统计量（mean/variance/max/min/2max/2min），共 18 列
Y_COLOR_C3_10_COLS: List[str] = [
    f"c3_10_{ch}_{stat}"
    for ch in ["a", "b", "y"]
    for stat in ["mean", "variance", "max", "min", "2max", "2min"]
]
# 颜色指标 c3_45：a/b 两通道（无 y），共 12 列
Y_COLOR_C3_45_COLS: List[str] = [
    f"c3_45_{ch}_{stat}"
    for ch in ["a", "b"]
    for stat in ["mean", "variance", "max", "min", "2max", "2min"]
]
# 透过率指标 t_：三个角度/波段组，各 6 统计量，共 18 列
Y_TRANS_T0_940_COLS: List[str] = [
    f"t_0deg_940_{stat}" for stat in ["mean", "variance", "max", "min", "2max", "2min"]
]
Y_TRANS_T40_920_960_COLS: List[str] = [
    f"t_40deg_920_960_avg_{stat}"
    for stat in ["mean", "variance", "max", "min", "2max", "2min"]
]
Y_TRANS_T0_400_770_COLS: List[str] = [
    f"t_0deg_400_770_avg_{stat}"
    for stat in ["mean", "variance", "max", "min", "2max", "2min"]
]

# 颜色 / 透过率全部列（用于"全缺"判定与分布/离群分析）
Y_COLOR_COLS: List[str] = Y_COLOR_C3_10_COLS + Y_COLOR_C3_45_COLS
Y_TRANS_COLS: List[str] = (
    Y_TRANS_T0_940_COLS + Y_TRANS_T40_920_960_COLS + Y_TRANS_T0_400_770_COLS
)
Y_ALL_COLS: List[str] = Y_COLOR_COLS + Y_TRANS_COLS  # 共 40 列

# 目标分组：颜色与透过率两大类，供多目标模型与分组评估使用
TARGET_GROUPS: Dict[str, List[str]] = {
    "color_c3_10": Y_COLOR_C3_10_COLS,
    "color_c3_45": Y_COLOR_C3_45_COLS,
    "trans": Y_TRANS_COLS,
}

# Y 物理越界阈值（用于清洗阶段丢弃明确坏数据，不含 variance）
# 颜色 a/b：CIE Lab 理论范围 [-128,128]，取宽松绝对值上限 100
COLOR_AB_ABS_MAX = 100
# 透过率：物理有效区间 (0, 100]，超出即越界
TRANSMISSION_VALID_RANGE = (0.0, 100.0)
# 越界检测排除 variance 列（方差无物理上下限意义）
Y_STATS_WITHOUT_VARIANCE: List[str] = ["mean", "max", "min", "2max", "2min"]


# ---------------------------------------------------------------------------
# 环境 / 上下文特征
# ---------------------------------------------------------------------------
# 产品规格
PRODUCT_SPEC_COLS: List[str] = [
    "l_length", "width", "area", "floor_number", "inside_code", "pcs_qty",
]
# 维保相关
# service_type：保养类型，缺失表示数据起点未记录上次保养信息（集中在 time_index 较小段），
#               预处理时填充为独立类别 unknown；service_order 保持数值让模型自学非线性
MAINTENANCE_COLS: List[str] = [
    "service_type", "service_order", "malfunction_type_code",
]
# service_type 缺失填充值（区别于真实"无保养"，本数据为起点未记录）
SERVICE_TYPE_MISSING_FILL = "unknown"

# ftu 配置：ftu_operation=0 时 ftu_type 为空，二者业务强相关，组合后作为单个分类变量处理
FTU_COLS: List[str] = ["ftu_operation", "ftu_type"]
# ftu 组合派生列：形如 "0_none" / "1_Ar" / "1_O2"，作为一个分类特征
FTU_COMBO_COL: str = "ftu_combo"

# 集中性检验与检测机台：与颜色指标关联
# pass_status：集中性检验是否通过；fail_detail：不通过原因详情
# 预处理逻辑：fail_detail!="无" 时 pass_status 必为 fail（修正误标），并派生条件触发特征
INSPECTION_COLS: List[str] = [
    "pass_status", "fail_detail", "color_station", "tr_station",
]
# fail_detail 派生的两个二值特征：是否触发条件1 / 是否触发条件2
# 文本含"条件1满足"表示触发条件1，含"条件2满足"表示触发条件2；当前数据两者同时触发，
# 拆成两个独立二值特征便于未来单条件触发场景
COND1_TRIGGERED_COL: str = "cond1_triggered"
COND2_TRIGGERED_COL: str = "cond2_triggered"
# fail_detail 中表示通过（无异常）的取值
FAIL_DETAIL_PASS_TOKEN = "无"
# 上游来料质量：平坦度 / 厚度 各 6 统计量，共 12 列
INCOMING_QUALITY_COLS: List[str] = [
    f"incoming_{prop}_{stat}"
    for prop in ["flatness", "thickness"]
    for stat in ["mean", "variance", "max", "min", "q1", "q3"]
]
# 测试数量：数量越大统计指标越稳定
TEST_QTY_COLS: List[str] = ["test_pcs_qty", "c3_test_pcs_qty", "t_test_pcs_qty"]
# 设定参数生产计数：当前 S 不变情况下生产的第几锅
PARAM_PRODUCTION_COUNT_COLS: List[str] = ["parameter_production_count"]

# 环境上下文全部特征（不含 identifier 与时间列，单独管理）
CONTEXT_COLS: List[str] = (
    PRODUCT_SPEC_COLS + MAINTENANCE_COLS + FTU_COLS + INSPECTION_COLS
    + INCOMING_QUALITY_COLS + TEST_QTY_COLS + PARAM_PRODUCTION_COUNT_COLS
)


# ---------------------------------------------------------------------------
# 标识 / 时间列
# ---------------------------------------------------------------------------
# id：锅次唯一 id，用于业务定位问题数据位置（不参与训练）
ID_COL: str = "id"
# machine_sn：设备序列号，用于考察机器间性能差异（可作为分类特征）
MACHINE_COL: str = "machine_sn"
# time_index：时间索引，仅保证相对顺序（序号小先生产），用于时间切分与时间相关评估
TIME_INDEX_COL: str = "time_index"
# start_time / end_time：锅次起止时间（含时分秒），用于计算时长与定位日期，不直接作为特征入模型
START_TIME_COL: str = "start_time"
END_TIME_COL: str = "end_time"
# hc_chamber_day：镀膜日期（物理含义=该锅次完成镀膜的日期，精确到天）
# 作为日期锚点，派生 year/month/quarter 特征供 EDA 分组分析与模型学习时间模式
HC_CHAMBER_DAY_COL: str = "hc_chamber_day"
# batch_log / batch_log_tag：锅次日志与生产类型（正常/测试），辅助 time_index 计算
BATCH_LOG_COL: str = "batch_log"
BATCH_LOG_TAG_COL: str = "batch_log_tag"

# 由 hc_chamber_day 派生的时间特征：年/月/季度
# 保留动机：真实生产存在年度高峰期、工人淡旺季等时间强相关模式，既供 EDA 按时间分组分析，
# 也作为模型输入特征让模型学到时间趋势。day 粒度太细（单日锅次多达数百）无分析价值，不派生。
DATE_FEATURE_COLS: List[str] = ["year", "month", "quarter"]

# 标识列：用于定位与分组，不直接作为数值特征
IDENTIFIER_COLS: List[str] = [
    ID_COL, MACHINE_COL, TIME_INDEX_COL, START_TIME_COL, END_TIME_COL,
    HC_CHAMBER_DAY_COL, BATCH_LOG_COL, BATCH_LOG_TAG_COL,
]


# ---------------------------------------------------------------------------
# 特征分组汇总
# ---------------------------------------------------------------------------
# 数值特征：S 全部 + M（cathode + 时长）+ 来料质量 + 测试量 + 设参计数 + 产品规格数值部分
# 说明：floor_number / inside_code 等虽为类别，归入下方分类特征
FEATURE_GROUPS: Dict[str, List[str]] = {
    "S": S_ALL_COLS,
    "M": M_CATHODE_COLS + M_DURATION_COLS,
    "incoming": INCOMING_QUALITY_COLS,
    "test_qty": TEST_QTY_COLS,
    "param_count": PARAM_PRODUCTION_COUNT_COLS,
    "product_spec_num": ["l_length", "width", "area", "pcs_qty"],
}

# 分类特征：机器、楼层、产品代码、维保类型、ftu 组合、故障代码、检测机台等
# 注：ftu 用组合派生列 ftu_combo（而非 ftu_operation/ftu_type 两列）；fail_detail 派生
#     cond1/cond2_triggered 后不再作为原始分类特征入模型（文本细节无意义）
CAT_FEATURE_COLS: List[str] = [
    MACHINE_COL,
    "floor_number", "inside_code",
    "service_type", "malfunction_type_code",
    FTU_COMBO_COL,
    "pass_status", "color_station", "tr_station",
    BATCH_LOG_TAG_COL,
    COND1_TRIGGERED_COL, COND2_TRIGGERED_COL,
]

# 数值特征：所有 FEATURE_GROUPS 展平
NUM_FEATURE_COLS: List[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]


def _all_defined_cols() -> List[str]:
    """汇总本文件所有显式定义的列名（去重保序），供双向验证使用。"""
    defined: List[str] = []
    for cols in (
        S_ALL_COLS, M_CATHODE_COLS, M_DURATION_COLS,
        Y_ALL_COLS, CONTEXT_COLS, IDENTIFIER_COLS, CAT_FEATURE_COLS,
        O2_COLS, AR_COLS,
    ):
        defined.extend(cols)
    # 去重保序
    seen = set()
    unique: List[str] = []
    for c in defined:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# 派生列：不在原始数据中，由原始列经预处理生成，双向校验时需排除
DERIVED_COLS: List[str] = [
    FTU_COMBO_COL, COND1_TRIGGERED_COL, COND2_TRIGGERED_COL,
    *DATE_FEATURE_COLS,  # year/month/quarter 由 hc_chamber_day 派生
]
# 派生列到其原始依赖列的映射（用于校验派生列的依赖列是否存在于真实数据）
DERIVED_COL_SOURCES: Dict[str, List[str]] = {
    FTU_COMBO_COL: FTU_COLS,
    COND1_TRIGGERED_COL: ["fail_detail"],
    COND2_TRIGGERED_COL: ["fail_detail"],
    "year": [HC_CHAMBER_DAY_COL],
    "month": [HC_CHAMBER_DAY_COL],
    "quarter": [HC_CHAMBER_DAY_COL],
}


def validate_columns(available_cols) -> Dict[str, List[str]]:
    """
    双向验证：本文件定义的列名是否都在真实数据中（派生列除外）。

    派生列（ftu_combo / cond1_triggered / cond2_triggered / year / month / quarter）
    不在原始数据中，由预处理生成，校验时改为检查其依赖的原始列是否存在。

    Args:
        available_cols: 真实数据列名集合或列表

    Returns:
        dict[group_name -> 缺失列名列表]；若某组无缺失则不出现在结果中。
        空字典表示所有定义列均存在于真实数据中（派生列检查其依赖列）。
    """
    available = set(available_cols)
    checks = {
        "S_ACTUAL_RATE_COLS": S_ACTUAL_RATE_COLS,
        "S_SET_RATE_COLS": S_SET_RATE_COLS,
        "S_RATE_COEFFICIENT_COLS": S_RATE_COEFFICIENT_COLS,
        "M_CATHODE_COLS": M_CATHODE_COLS,
        "M_DURATION_COLS": M_DURATION_COLS,
        "Y_COLOR_COLS": Y_COLOR_COLS,
        "Y_TRANS_COLS": Y_TRANS_COLS,
        "CONTEXT_COLS": CONTEXT_COLS,
        "IDENTIFIER_COLS": IDENTIFIER_COLS,
        "O2_COLS": O2_COLS,
        "AR_COLS": AR_COLS,
    }
    missing: Dict[str, List[str]] = {}
    for name, cols in checks.items():
        # 派生列改为检查其依赖列
        check_cols = [DERIVED_COL_SOURCES.get(c, [c]) for c in cols]
        check_cols = [c for sub in check_cols for c in sub]
        miss = [c for c in check_cols if c not in available]
        if miss:
            missing[name] = list(dict.fromkeys(miss))
    return missing


def find_unused_columns(available_cols) -> Dict[str, List[str]]:
    """
    双向验证：真实数据中哪些列名未被本文件使用，并按命名规律归类。

    Args:
        available_cols: 真实数据列名集合或列表

    Returns:
        dict[类别 -> 未使用列名列表]；空字典表示真实数据列全部已被使用。
        类别含：
          - "defined_but_未匹配规则"：无法按命名归类的未使用列
          - 其余键按 o2/ar/actual_rate 等前缀归类
    """
    available = set(available_cols)
    defined = set(_all_defined_cols())
    unused = available - defined

    if not unused:
        return {}

    # 按命名规律归类未使用列
    rules: List[tuple] = [
        ("actual_rate_l", lambda c: c.startswith("actual_rate_l")),
        ("set_rate_l", lambda c: c.startswith("set_rate_l")),
        ("rate_coefficient_l", lambda c: c.startswith("rate_coefficient_l")),
        ("cathode_usevalue", lambda c: c.startswith("cathode") and c.endswith("_usevalue")),
        ("o2_", lambda c: c.startswith("o2_")),
        ("ar_l", lambda c: c.startswith("ar") and "_l" in c),
        ("c3_10_", lambda c: c.startswith("c3_10_")),
        ("c3_45_", lambda c: c.startswith("c3_45_")),
        ("t_0deg_940", lambda c: c.startswith("t_0deg_940")),
        ("t_40deg_920_960", lambda c: c.startswith("t_40deg_920_960")),
        ("t_0deg_400_770", lambda c: c.startswith("t_0deg_400_770")),
        ("incoming_", lambda c: c.startswith("incoming_")),
    ]

    grouped: Dict[str, List[str]] = {}
    matched = set()
    for label, pred in rules:
        hits = sorted([c for c in unused if pred(c)])
        if hits:
            grouped[label] = hits
            matched.update(hits)

    rest = sorted(unused - matched)
    if rest:
        grouped["defined_but_未匹配规则"] = rest
    return grouped
