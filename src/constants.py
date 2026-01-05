# 常数
RANDOM_STATE = 42
N_TRIALS = 1500
TEST_SIZE = 0.2

# 相关字段
# 机台相关
MACHINE_COLS = ["machine_sn"]
# 其他信息
OTHER_INFO_COLS = ["row_id", "batch_log"]
# recipe 相关
N_LAYERS_RECIPE = 11
RECIPE_LAYER_INDEX = range(1, N_LAYERS_RECIPE + 1)
# MAT_NAME_COLS = [f"material_name_{i}" for i in RECIPE_LAYER_INDEX]
# FILM_THICKNESS_COLS = [f"set_film_thickness_{i}" for i in RECIPE_LAYER_INDEX]
RATE_NM_SEC_COLS = [f"set_rate_nm_sec_{i}" for i in RECIPE_LAYER_INDEX]
RATE_COEF_COLS = [f"rate_coefficient_{i}" for i in RECIPE_LAYER_INDEX]
ACT_RATE_NM_SEC_COLS = [f"act_rate_nm_sec_{i}" for i in RECIPE_LAYER_INDEX]
# 日期变量
RAW_DATE_COLS = ["hc_chamber_day"]
DATE_FEATURE_COLS = ["year", "month", "day"]
# 批次列
BATCH_NUMBER_COLS = ["batch_number"]

# 检测结果列
COLOR_COLS = [
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
]
TRANSMITTANCE_COLS = [
    "T_0Deg_400_770_AVG_min",
    "T_0Deg_940_min",
    "T_40Deg_920_960_AVG_min",
]
