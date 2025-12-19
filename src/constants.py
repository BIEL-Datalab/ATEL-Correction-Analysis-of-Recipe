# 常数
RANDOM_STATE = 42
N_TRIALS = 1000
TEST_SIZE = 0.2

# 相关字段
MAT_NAME_COLS = [f"material_name_{i}" for i in range(1, 12)]
FILM_THICKNESS_COLS = [f"set_film_thickness_{i}" for i in range(1, 12)]
RATE_NM_SEC_COLS = [f"set_rate_nm_sec_{i}" for i in range(1, 12)]
RATE_COEF_COLS = [f"rate_coefficient_{i}" for i in range(1, 12)]
# 日期变量
DATE_COLS = ["hc_chamber_day"]
DATE_SEQ_COLS = ["year", "month", "day"]
# 分类特征
CAT_COLS = (
    [
        "machine_sn",
    ]
    + DATE_SEQ_COLS
    + MAT_NAME_COLS
)
# 数值特征
NUM_COLS = (
    [
        "batch_order",
    ]
    + RATE_NM_SEC_COLS
    + RATE_COEF_COLS
)
# 全部特征
FEATURE_COLS = CAT_COLS + NUM_COLS
# 检测结果列
INDICATOR_COLS = [
    "C3_10_Y_min",
    "C3_10_Y_max",
    "C3_10_a_min",
    "C3_10_a_max",
    "C3_10_b_min",
    "C3_10_b_max",
    "C3_45_a_min",
    "C3_45_a_max",
    "C3_45_b_min",
    "C3_45_b_max",
    "T_0Deg_400-770 AVG_min",
    "T_0Deg_940_min",
    "T_40Deg_920-960 AVG_min",
]

# 预测变量
TARGET_COLS = INDICATOR_COLS
