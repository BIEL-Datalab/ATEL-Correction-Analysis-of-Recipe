# 机台相关
MACHINE_COLS = ["machine_sn"]
# 其他信息
OTHER_INFO_COLS = ["index", "batch_log"]
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
    "c3_10_y_min",
    "c3_10_y_max",
    "c3_10_a_min",
    "c3_10_a_max",
    "c3_10_b_min",
    "c3_10_b_max",
    "c3_45_a_min",
    "c3_45_a_max",
    "c3_45_b_min",
    "c3_45_b_max",
]
TRANSMITTANCE_COLS = [
    "t_0deg_400_770_avg_min",
    "t_0deg_940_min",
    "t_40deg_920_960_avg_min",
]
# 目标变量分组
TARGET_GROUPS = {"color": COLOR_COLS, "trans": TRANSMITTANCE_COLS}
