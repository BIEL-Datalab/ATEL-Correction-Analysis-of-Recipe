# 训练相关常数

# 通用
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------------------------------------------------------------------
# 树模型默认训练参数
# -------------------------------------------------------------------
TREE_DEFAULT_NUM_BOOST_ROUND = 2000
TREE_DEFAULT_EARLY_STOPPING_ROUND = 50
# optuna 默认搜索试验数。notebook 通路测试时可传小值（如 15）快速验证，
# 正式脚本再恢复较大值。
TREE_DEFAULT_N_TRIALS = 5000
TREE_N_JOBS = 10

# -------------------------------------------------------------------
# 深度模型默认训练参数（GANDALF / TabM）
# -------------------------------------------------------------------
TB_DEFAULT_BATCH_SIZE = 1024
TB_DEFAULT_MAX_EPOCHS = 100
TB_DEFAULT_N_TRIALS = 500
TB_N_JOBS = 1
TB_DEFAULT_NUM_WORKS = 20

# -------------------------------------------------------------------
# 缺失/新类别处理
# -------------------------------------------------------------------
# 分类特征缺失填充 token（树模型用字符串缺失标记，深度模型用独立类别索引）
MISSING_TOKEN = "missing"
# 深度模型分类特征未知类别（OOV）的编码索引占位：训练时未见类别统一映射到此索引。
# 与 TabM/GANDALF 的 cardinality+1 方案配合：每列 cardinality = n_unique + 1，
# 未知值映射到 n_unique（最后一个 one-hot 位即 OOV 表示）。
OOV_TOKEN = "unknown"
