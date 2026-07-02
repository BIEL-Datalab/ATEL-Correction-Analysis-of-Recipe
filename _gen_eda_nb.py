"""生成重写后的 raw_data_eda.ipynb（一次性脚本，生成后可删除）"""
import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []})

# ---------- Cell 0: 文档头 ----------
md("""# 原始数据探索分析（EDA）

数据源：`raw_data/raw_data_260623/data.parquet`（284 列，49799 行）

## 载入策略
- 复用 `src.core.constants.data_constants` 的列分组常量与阈值，不在 notebook 内重复定义。
- **不读入** `o2`(8 列)、`ar`(120 列)（传感器原始量大，本期不分析）。
- 读入时将 `actual_rate`/`set_rate`/`rate_coefficient` 三组（object 存数字字符串）转为数值列。

## 分析内容
1. **结构异常**：S 整批缺失、Y 颜色全缺、Y 透过率全缺、主键重复。
2. **Y 物理越界**：颜色 a/b `|val|>100`、透过率越出 (0,100]（均不含 variance）。
3. **分布 + IQR×5 离群**：针对 `cathode`、`incoming_quality`、所有 Y 指标（除 variance）。
4. **唯一值**：组合唯一值（machine_sn+S 三组、ftu 组合）+ 单字段唯一值。
5. **机器间 Y 对比**：按 machine_sn 分组对比 Y 分布，识别机器间性能差异。
6. **产品间 Y 对比**：按 inside_code 分组对比 Y，识别产品间指标差异。
7. **Y 相关性矩阵**：48 个 Y 指标间 Pearson 相关性 + 热力图。
8. **分类变量类别数统计**：各分类特征及组合的类别数与分布。

## 说明
- S 整批缺失属真实生产缺失情况，EDA 仅统计定位、**不作为需清洗的异常**（清洗时保留，由模型处理缺失）。
- 异常仅定位不清洗。

## 输出（`results/research/raw_data_eda/run_YYYYMMDD_HHMMSS/`）
- 每次运行创建带时间戳子目录，区分不同时间的 EDA 结果。
- `raw_data_anomalies.xlsx`：异常数据（结构/越界/离群，存整条原始行），不同 sheet。
- `raw_data_eda_summary.xlsx`：EDA 概述，每张表一个 sheet。
- `raw_data_combo_uniques.xlsx`：组合唯一值 + 单字段唯一值，每项一个 sheet。
- `eda_group_comparison.xlsx`：机器间/产品间 Y 对比。
- `y_correlation.csv` + `y_correlation_heatmap.png`：Y 相关性矩阵与热力图。
- `cat_cardinality.xlsx`：分类变量类别数统计。
""")

# ---------- Cell 1: 载入数据 ----------
code("""# ============================================================
# 1. 载入数据：读 schema 做双向列校验，按需加载列，S 三组转数值
# ============================================================
import os
import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# 将项目根目录加入 sys.path，使 notebook 可直接 import src（无论从哪个目录启动）
_PROJECT_ROOT = '/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe'
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 复用 src 常量与校验/清洗函数，不在 notebook 内重复定义
from src.core.constants.data_constants import (
    S_ACTUAL_RATE_COLS, S_SET_RATE_COLS, S_RATE_COEFFICIENT_COLS, S_ALL_COLS,
    M_CATHODE_COLS, M_DURATION_COLS,
    Y_COLOR_COLS, Y_TRANS_COLS, Y_ALL_COLS,
    INCOMING_QUALITY_COLS, TEST_QTY_COLS, CONTEXT_COLS, IDENTIFIER_COLS,
    CAT_FEATURE_COLS, NUM_FEATURE_COLS,
    COLOR_AB_ABS_MAX, TRANSMISSION_VALID_RANGE, Y_STATS_WITHOUT_VARIANCE,
    HC_CHAMBER_DAY_COL, TIME_INDEX_COL, MACHINE_COL, ID_COL, BATCH_LOG_COL,
    EXCLUDE_GROUPS,
)
from src.core.data.column_schema import get_column_groups, check_column_consistency
from src.core.data.cleaning import clean_raw_data

warnings.filterwarnings('ignore')

DATA_PATH = '/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/raw_data/raw_data_260623/data.parquet'

# 1.1 读 schema 做双向列校验（定义→真实 / 真实→未用）
_all_columns = pq.ParquetFile(DATA_PATH).schema.names
consistency = check_column_consistency(_all_columns)
print(f'原始总列数: {len(_all_columns)}')
print(f'双向校验 - 缺失(定义有数据无): {len(consistency["missing"])} 组')
print(f'双向校验 - 未用(数据有定义无): {len(consistency["unused"])} 组')

# 1.2 计算需加载列：排除 o2 / ar（不读入、不分析）
exclude_cols = set()
for cols in EXCLUDE_GROUPS.values():
    exclude_cols.update(cols)
load_cols = [c for c in _all_columns if c not in exclude_cols]
print(f'排除 o2({len(EXCLUDE_GROUPS["o2"])}) + ar({len(EXCLUDE_GROUPS["ar"])})，实际加载: {len(load_cols)} 列')

# 1.3 加载数据
df = pd.read_parquet(DATA_PATH, columns=load_cols)
print(f'\\ndf shape: {df.shape}')

# 1.4 S 三组（object 存数字字符串）转数值
S_numeric_cols = S_ACTUAL_RATE_COLS + S_SET_RATE_COLS + S_RATE_COEFFICIENT_COLS
for c in S_numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
print(f'已将 S 三组 {len(S_numeric_cols)} 列转为数值类型')
print(f'\\nY 指标列数: 颜色 {len(Y_COLOR_COLS)} + 透过率 {len(Y_TRANS_COLS)} = {len(Y_ALL_COLS)}')
df.info(max_cols=0)
""")

# ---------- Cell 2: 全局配置 ----------
code("""# ============================================================
# 2. 全局配置：定位列 / 输出路径 / 分析目标分组
# ============================================================
# 异常记录附带的定位列（贯穿后续所有 cell）
LOCATOR_COLS = [ID_COL, MACHINE_COL, BATCH_LOG_COL, HC_CHAMBER_DAY_COL]

# 输出根目录：每次 EDA 运行单独一个带时间戳子目录
OUTPUT_BASE_DIR = '/home_ext/zzx/work_file/ATEL-Correction-Analysis-of-Recipe/results/research/raw_data_eda'
RUN_SUBDIR = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, RUN_SUBDIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 各输出文件路径
ANOMALY_XLSX = os.path.join(OUTPUT_DIR, 'raw_data_anomalies.xlsx')
SUMMARY_XLSX = os.path.join(OUTPUT_DIR, 'raw_data_eda_summary.xlsx')
COMBO_XLSX = os.path.join(OUTPUT_DIR, 'raw_data_combo_uniques.xlsx')
GROUP_XLSX = os.path.join(OUTPUT_DIR, 'eda_group_comparison.xlsx')
Y_CORR_CSV = os.path.join(OUTPUT_DIR, 'y_correlation.csv')
Y_CORR_PNG = os.path.join(OUTPUT_DIR, 'y_correlation_heatmap.png')
CAT_CARD_XLSX = os.path.join(OUTPUT_DIR, 'cat_cardinality.xlsx')

# IQR 离群倍数
IQR_OUTLIER_K = 5

# 分布+离群分析的 Y 列（剔除 variance，方差无物理上下限意义）
Y_value_cols = [c for c in Y_ALL_COLS if c.split('_')[-1] in Y_STATS_WITHOUT_VARIANCE]

# 组合唯一值分析目标
COMBO_DEFS = [
    ('machine_sn+S_actual_rate', [MACHINE_COL] + S_ACTUAL_RATE_COLS),
    ('machine_sn+S_set_rate', [MACHINE_COL] + S_SET_RATE_COLS),
    ('machine_sn+S_rate_coefficient', [MACHINE_COL] + S_RATE_COEFFICIENT_COLS),
    ('ftu_operation+ftu_type', ['ftu_operation', 'ftu_type']),
]

# 单字段唯一值分析：排除 identifier 组与 service_order，纳入 test_info 组
UNIQUE_EXCLUDE_COLS = set(IDENTIFIER_COLS) | {'service_order'}

print('输出目录:', OUTPUT_DIR)
print('Y 指标列(除variance)数:', len(Y_value_cols))
print('组合唯一值分析:', [name for name, _ in COMBO_DEFS])
""")

# ---------- Cell 3: 结构异常 ----------
code("""# ============================================================
# 3. 结构异常检查：S 整批缺失 / Y 颜色全缺 / Y 透过率全缺 / 主键重复
#    每类异常存整条原始行，统一收进 structural_issues
# ============================================================
structural_issues: dict[str, pd.DataFrame] = {}

# 3.1 S 整批缺失（仅统计定位，清洗时保留——模拟真实缺失场景）
mask_s_all_missing = df[S_ALL_COLS].isna().all(axis=1)
structural_issues['结构异常_S整批缺失'] = df.loc[mask_s_all_missing].copy()
n_s_missing = int(mask_s_all_missing.sum())

# 3.2 Y 颜色全缺 / 透过率全缺
mask_color_missing = df[Y_COLOR_COLS].isna().all(axis=1)
mask_tr_missing = df[Y_TRANS_COLS].isna().all(axis=1)
structural_issues['结构异常_Y颜色全缺'] = df.loc[mask_color_missing].copy()
structural_issues['结构异常_Y透过率全缺'] = df.loc[mask_tr_missing].copy()
n_color_missing = int(mask_color_missing.sum())
n_tr_missing = int(mask_tr_missing.sum())

# 3.3 主键 / 重复检查（汇总表，非整行）
dup_rows = []
for key, label in [(ID_COL, 'id'), (BATCH_LOG_COL, 'batch_log'), (None, '整行完全重复')]:
    if key is None:
        n_dup = int(df.duplicated().sum())
    else:
        n_dup = int(df.duplicated(subset=[key]).sum())
    dup_rows.append({'检查项': label, '重复行数': n_dup, '总行数': len(df)})
structural_issues['结构异常_重复检查'] = pd.DataFrame(dup_rows)

# 输出概况
print('=' * 60)
print('  结构异常概况')
print('=' * 60)
print(f'[S整批缺失]   {n_s_missing} 行  ({n_s_missing/len(df)*100:.2f}%)  【清洗时保留，模拟真实缺失】')
print(f'[Y颜色全缺]   {n_color_missing} 行  (其中 c3_test_pcs_qty=0: {int((df.loc[mask_color_missing,"c3_test_pcs_qty"]==0).sum())})')
print(f'[Y透过率全缺] {n_tr_missing} 行  (其中 t_test_pcs_qty=0: {int((df.loc[mask_tr_missing,"t_test_pcs_qty"]==0).sum())})')
print('[重复检查]')
print(structural_issues['结构异常_重复检查'].to_string(index=False))
""")

# ---------- Cell 4: Y 物理越界 ----------
code("""# ============================================================
# 4. Y 物理越界异常：颜色 a/b |val|>100 / 透过率越出 (0,100]（均不含 variance）
#    同一锅次多列越界合并为一条整行，新增"越界列"列记录命中的列名清单
# ============================================================
# 颜色 a/b 列（剔除 variance）；透过率列（剔除 variance）
color_ab_cols = [c for c in Y_value_cols
                 if any(f'_{ch}_' in c for ch in ['a', 'b'])]
tr_value_cols = [c for c in Y_value_cols if c.startswith('t_')]
lo, hi = TRANSMISSION_VALID_RANGE

# 逐列计算越界 mask
y_boundary_masks: dict[str, pd.Series] = {}
for col in color_ab_cols:
    y_boundary_masks[col] = df[col].abs() > COLOR_AB_ABS_MAX
for col in tr_value_cols:
    s = df[col]
    y_boundary_masks[col] = (s <= lo) | (s > hi)

any_boundary = pd.concat(y_boundary_masks.values(), axis=1).any(axis=1)
n_boundary_rows = int(any_boundary.sum())

if n_boundary_rows > 0:
    hit_cols_series = pd.Series([''] * len(df), index=df.index)
    for col, m in y_boundary_masks.items():
        hit_cols_series = hit_cols_series.mask(m, hit_cols_series + col + '; ')
    hit_cols_series = hit_cols_series.str.rstrip('; ')
    df_y_boundary = df.loc[any_boundary].copy()
    df_y_boundary['越界列'] = hit_cols_series.loc[any_boundary].values
else:
    df_y_boundary = pd.DataFrame()

print('=' * 60)
print('  Y 物理越界异常概况')
print('=' * 60)
print(f'颜色 a/b 阈值: |val| > {COLOR_AB_ABS_MAX}（不含 variance）')
print(f'透过率有效区间: ({lo}, {hi}]（不含 variance）')
print(f'越界命中唯一行数: {n_boundary_rows}')
if n_boundary_rows > 0:
    per_col = pd.DataFrame([
        {'列名': col, '越界数': int(m.sum()), '占比': f"{int(m.sum())/len(df)*100:.3f}%"}
        for col, m in y_boundary_masks.items() if m.sum() > 0
    ]).sort_values('越界数', ascending=False)
    print('\\n各列越界数量:')
    print(per_col.to_string(index=False))
    structural_issues['越界_Y物理越界'] = df_y_boundary
""")

# ---------- Cell 5: 分布 + IQR 离群 ----------
code("""# ============================================================
# 5. 分布统计 + IQR×5 离群定位
#    目标：cathode / incoming_quality / 所有 Y 指标（除 variance）
#    离群记录存整条原始行 + "离群列"列标记命中列名
# ============================================================
def distribution_and_outliers(
    df_src: pd.DataFrame,
    cols: list[str],
    iqr_k: float = IQR_OUTLIER_K,
) -> tuple[pd.DataFrame, pd.Series, dict[str, pd.Series]]:
    \"\"\"对一组数值列计算分布统计 + IQR×k 离群 mask。\"\"\"
    stats_rows = []
    col_masks: dict[str, pd.Series] = {}
    for col in cols:
        s = df_src[col]
        n_missing = int(s.isna().sum())
        n_valid = int(s.count())
        n_zero = int((s == 0).sum())
        if n_valid > 0:
            q1, median, q3 = s.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            stats_rows.append({
                '列名': col, '非空数': n_valid, '缺失数': n_missing, '零值数': n_zero,
                '均值': round(float(s.mean()), 4), '标准差': round(float(s.std()), 4),
                '最小值': round(float(s.min()), 4), 'Q1': round(float(q1), 4),
                '中位数': round(float(median), 4), 'Q3': round(float(q3), 4),
                '最大值': round(float(s.max()), 4), 'IQR': round(float(iqr), 4),
            })
            if iqr > 0:
                lower = q1 - iqr_k * iqr
                upper = q3 + iqr_k * iqr
                col_masks[col] = (s < lower) | (s > upper)
            else:
                col_masks[col] = pd.Series(False, index=df_src.index)
        else:
            stats_rows.append({
                '列名': col, '非空数': 0, '缺失数': n_missing, '零值数': 0,
                '均值': np.nan, '标准差': np.nan, '最小值': np.nan,
                'Q1': np.nan, '中位数': np.nan, 'Q3': np.nan,
                '最大值': np.nan, 'IQR': np.nan,
            })
            col_masks[col] = pd.Series(False, index=df_src.index)
    stats = pd.DataFrame(stats_rows)
    any_outlier = pd.concat(col_masks.values(), axis=1).any(axis=1) if col_masks else pd.Series(False, index=df_src.index)
    return stats, any_outlier, col_masks


# 分析目标列：cathode + incoming_quality + Y 指标(除variance)
dist_target_cols = list(dict.fromkeys(M_CATHODE_COLS + INCOMING_QUALITY_COLS + Y_value_cols))
df_dist_stats, any_outlier, dist_col_masks = distribution_and_outliers(df, dist_target_cols)
n_outlier_rows = int(any_outlier.sum())

if n_outlier_rows > 0:
    hit_cols_series = pd.Series([''] * len(df), index=df.index)
    for col, m in dist_col_masks.items():
        if m.sum() == 0:
            continue
        hit_cols_series = hit_cols_series.mask(m, hit_cols_series + col + '; ')
    hit_cols_series = hit_cols_series.str.rstrip('; ')
    df_dist_outliers = df.loc[any_outlier].copy()
    df_dist_outliers['离群列'] = hit_cols_series.loc[any_outlier].values
    structural_issues['离群_IQR5离群'] = df_dist_outliers
else:
    df_dist_outliers = pd.DataFrame()

print('=' * 60)
print(f'  分布统计 + IQR×{IQR_OUTLIER_K} 离群概况')
print('=' * 60)
print(f'目标列数: {len(dist_target_cols)}  (cathode={len(M_CATHODE_COLS)}, incoming={len(INCOMING_QUALITY_COLS)}, Y(除variance)={len(Y_value_cols)})')
print(f'离群命中唯一行数: {n_outlier_rows}  ({n_outlier_rows/len(df)*100:.2f}%)')
per_col = pd.DataFrame([
    {'列名': col, '离群数': int(m.sum()), '占比': f"{int(m.sum())/len(df)*100:.3f}%"}
    for col, m in dist_col_masks.items() if m.sum() > 0
]).sort_values('离群数', ascending=False)
print('\\n各列离群数量(前15):')
print(per_col.head(15).to_string(index=False))
""")

# ---------- Cell 6: 组合/单字段唯一值 ----------
code("""# ============================================================
# 6. 组合唯一值 + 单字段唯一值
# ============================================================
combo_results: dict[str, pd.DataFrame] = {}
combo_counts: dict[str, int] = {}
for name, cols in COMBO_DEFS:
    vc = df.groupby(cols, dropna=False).size().reset_index(name='出现次数').sort_values('出现次数', ascending=False)
    combo_results[name] = vc
    combo_counts[name] = len(vc)

# 单字段唯一值：排除 identifier 组与 service_order，取 object 类型或 test_info 组
test_info_cols = set(TEST_QTY_COLS)
combo_covered_cols = set()
for _, cols in COMBO_DEFS:
    combo_covered_cols.update(cols)
single_field_cols = [
    c for c in df.columns
    if c not in combo_covered_cols
    and c not in UNIQUE_EXCLUDE_COLS
    and (df[c].dtype == 'object' or c in test_info_cols)
]
single_field_results: dict[str, pd.DataFrame] = {}
single_field_counts: dict[str, int] = {}
for col in single_field_cols:
    vc = df[col].value_counts(dropna=False).reset_index()
    vc.columns = ['取值', '出现次数']
    vc = vc.sort_values('出现次数', ascending=False)
    single_field_results[col] = vc
    single_field_counts[col] = len(vc)

print('=' * 60)
print('  组合唯一值概况')
print('=' * 60)
for name in [n for n, _ in COMBO_DEFS]:
    print(f'[{name}]  唯一组合数 = {combo_counts[name]}')
    print(combo_results[name].head(5).to_string(index=False))
    print()
print(f'单字段唯一值字段数: {len(single_field_cols)}')
print('各字段 nunique:')
for col in single_field_cols:
    print(f'  {col}: {single_field_counts[col]}')
""")

# ---------- Cell 7: 机器间 Y 对比 ----------
code("""# ============================================================
# 7. 机器间 Y 对比：按 machine_sn 分组，对比各 Y 指标均值/标准差
#    识别机器间性能差异，为是否分机器建模提供依据
# ============================================================
# 用 mean 列做对比（每个指标取 mean 统计量，避免 6 个统计量全列太宽）
y_mean_cols = [c for c in Y_ALL_COLS if c.endswith('_mean')]
machine_y = df.groupby(MACHINE_COL)[y_mean_cols].agg(['mean', 'std'])
# 展平多级列名
machine_y.columns = [f'{c[0]}_{c[1]}' for c in machine_y.columns]
machine_y = machine_y.round(3)
# 加上每台机器样本数
machine_counts = df.groupby(MACHINE_COL).size().rename('样本数')
machine_y = machine_counts.to_frame().join(machine_y)

print(f'机器数: {len(machine_y)}')
print(f'参与对比的 Y mean 列数: {len(y_mean_cols)}')
print('\\n各机器样本数分布:')
print(machine_counts.describe())
print('\\n机器间 Y 均值差异最大的 5 个指标（按 std 跨机器排序）:')
y_mean_means = machine_y[[c for c in machine_y.columns if c.endswith('_mean')]]
cross_machine_std = y_mean_means.std().sort_values(ascending=False)
print(cross_machine_std.head(5).round(3))

group_frames = {'机器间_Y对比': machine_y.reset_index()}
""")

# ---------- Cell 8: 产品间 Y 对比 ----------
code("""# ============================================================
# 8. 产品间 Y 对比：按 inside_code 分组，对比各 Y 指标均值/标准差
#    不同产品指标范围不同，对建模重要
# ============================================================
product_y = df.groupby('inside_code')[y_mean_cols].agg(['mean', 'std'])
product_y.columns = [f'{c[0]}_{c[1]}' for c in product_y.columns]
product_y = product_y.round(3)
product_counts = df.groupby('inside_code').size().rename('样本数')
product_y = product_counts.to_frame().join(product_y)

print(f'产品数(inside_code): {len(product_y)}')
print('\\n各产品样本数:')
print(product_counts.sort_values(ascending=False))
print('\\n产品间 Y 均值差异最大的 5 个指标:')
y_mean_means_p = product_y[[c for c in product_y.columns if c.endswith('_mean')]]
cross_product_std = y_mean_means_p.std().sort_values(ascending=False)
print(cross_product_std.head(5).round(3))

group_frames['产品间_Y对比'] = product_y.reset_index()
""")

# ---------- Cell 9: Y 相关性矩阵 ----------
code("""# ============================================================
# 9. Y 相关性矩阵：48 个 Y 指标间 Pearson 相关性 + 热力图
#    为后续选目标、降维、多目标共线性分析铺垫
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

y_corr = df[Y_ALL_COLS].corr(method='pearson').round(3)
y_corr.to_csv(Y_CORR_CSV, encoding='utf-8-sig')

# 热力图
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(y_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(Y_ALL_COLS)))
ax.set_xticklabels(Y_ALL_COLS, rotation=90, fontsize=6)
ax.set_yticks(range(len(Y_ALL_COLS)))
ax.set_yticklabels(Y_ALL_COLS, fontsize=6)
ax.set_title('Y 指标 Pearson 相关性矩阵', fontsize=14)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(Y_CORR_PNG, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'相关性矩阵已保存: {Y_CORR_CSV}')
print(f'热力图已保存: {Y_CORR_PNG}')

# 高相关对（|r|>0.8，剔除自身）
corr_pairs = []
for i in range(len(Y_ALL_COLS)):
    for j in range(i+1, len(Y_ALL_COLS)):
        r = y_corr.iloc[i, j]
        if abs(r) > 0.8:
            corr_pairs.append({'指标1': Y_ALL_COLS[i], '指标2': Y_ALL_COLS[j], '相关系数': round(r, 3)})
corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('相关系数', ascending=False) if corr_pairs else pd.DataFrame()
print(f'\\n高相关对(|r|>0.8)数量: {len(corr_pairs_df)}')
if len(corr_pairs_df) > 0:
    print(corr_pairs_df.head(10).to_string(index=False))
""")

# ---------- Cell 10: 分类变量类别数统计 ----------
code("""# ============================================================
# 10. 分类变量类别数统计：各分类特征及组合的类别数与分布
#     识别高基数分类变量（需合并或编码处理）与低基数变量
# ============================================================
cat_card_frames = {}
cat_card_rows = []
for col in CAT_FEATURE_COLS:
    if col not in df.columns:
        continue
    nunique = df[col].nunique(dropna=False)
    top_vals = df[col].value_counts(dropna=False).head(5)
    cat_card_rows.append({
        '字段': col, '类别数': nunique,
        'Top5取值': '; '.join([f'{k}({v})' for k, v in top_vals.items()]),
    })
    # 每个字段的完整分布单独存
    cat_card_frames[col] = df[col].value_counts(dropna=False).reset_index().rename(
        columns={'index': '取值', col: '取值', 0: '出现次数'}
    )
cat_card_summary = pd.DataFrame(cat_card_rows).sort_values('类别数', ascending=False)
print('分类变量类别数统计:')
print(cat_card_summary.to_string(index=False))

# 组合分类变量的类别数
combo_card_rows = []
for name, cols in COMBO_DEFS:
    nunique = df.groupby(cols, dropna=False).ngroups
    combo_card_rows.append({'组合': name, '类别数': nunique})
combo_card = pd.DataFrame(combo_card_rows)
print('\\n组合分类变量类别数:')
print(combo_card.to_string(index=False))
cat_card_frames['_组合类别数汇总'] = combo_card
""")

# ---------- Cell 11: 保存 ----------
code("""# ============================================================
# 11. 保存结果
# ============================================================
def _safe_sheet_name(name: str) -> str:
    for ch in '[]:*?/\\\\':
        name = name.replace(ch, '_')
    return name[:31]

# (A) 异常数据 xlsx
with pd.ExcelWriter(ANOMALY_XLSX, engine='openpyxl') as writer:
    for sheet_label, frame in structural_issues.items():
        frame.to_excel(writer, sheet_name=_safe_sheet_name(sheet_label), index=False)
print(f'异常数据已保存: {ANOMALY_XLSX}')
for k, v in structural_issues.items():
    print(f'  - {k}: {len(v)} 行, {v.shape[1]} 列')

# (B) EDA 概述 xlsx
overview_frames: dict[str, pd.DataFrame] = {}
overview_frames['概述_结构异常'] = pd.DataFrame([
    {'异常类型': 'S整批缺失', '命中行数': n_s_missing,
     '占比': f"{n_s_missing/len(df)*100:.2f}%", '说明': '所有 S 类列均缺失（清洗保留）'},
    {'异常类型': 'Y颜色全缺', '命中行数': n_color_missing,
     '占比': f"{n_color_missing/len(df)*100:.2f}%",
     '说明': f'其中 c3_test_pcs_qty=0: {int((df.loc[mask_color_missing,"c3_test_pcs_qty"]==0).sum())} (未测色)'},
    {'异常类型': 'Y透过率全缺', '命中行数': n_tr_missing,
     '占比': f"{n_tr_missing/len(df)*100:.2f}%",
     '说明': f'其中 t_test_pcs_qty=0: {int((df.loc[mask_tr_missing,"t_test_pcs_qty"]==0).sum())} (未测透)'},
])
overview_frames['概述_重复检查'] = structural_issues['结构异常_重复检查'].copy()
overview_frames['概述_Y越界各列'] = pd.DataFrame([
    {'列名': col, '越界数': int(m.sum()), '占比': f"{int(m.sum())/len(df)*100:.3f}%"}
    for col, m in y_boundary_masks.items() if m.sum() > 0
]).sort_values('越界数', ascending=False) if n_boundary_rows > 0 else pd.DataFrame(columns=['列名', '越界数', '占比'])
overview_frames['概述_分布统计'] = df_dist_stats.copy()
overview_frames['概述_离群各列'] = pd.DataFrame([
    {'列名': col, '离群数': int(m.sum()), '占比': f"{int(m.sum())/len(df)*100:.3f}%"}
    for col, m in dist_col_masks.items() if m.sum() > 0
]).sort_values('离群数', ascending=False)
overview_frames['概述_组合唯一值'] = pd.DataFrame([
    {'组合': name, '唯一组合数': combo_counts[name]} for name, _ in COMBO_DEFS
])
overview_frames['概述_单字段唯一值'] = pd.DataFrame([
    {'字段': col, 'nunique': single_field_counts[col]} for col in single_field_cols
])
overview_frames['概述_高相关Y对'] = corr_pairs_df.copy() if len(corr_pairs_df) > 0 else pd.DataFrame(columns=['指标1', '指标2', '相关系数'])
with pd.ExcelWriter(SUMMARY_XLSX, engine='openpyxl') as writer:
    for sheet_label, frame in overview_frames.items():
        frame.to_excel(writer, sheet_name=_safe_sheet_name(sheet_label), index=False)
print(f'\\nEDA 概述已保存: {SUMMARY_XLSX}')

# (C) 组合唯一值 + 单字段唯一值 xlsx
with pd.ExcelWriter(COMBO_XLSX, engine='openpyxl') as writer:
    for name, frame in combo_results.items():
        frame.to_excel(writer, sheet_name=_safe_sheet_name(name), index=False)
    for col, frame in single_field_results.items():
        frame.to_excel(writer, sheet_name=_safe_sheet_name(col), index=False)
print(f'组合唯一值已保存: {COMBO_XLSX}')

# (D) 机器间/产品间 Y 对比 xlsx
with pd.ExcelWriter(GROUP_XLSX, engine='openpyxl') as writer:
    for name, frame in group_frames.items():
        frame.to_excel(writer, sheet_name=_safe_sheet_name(name), index=False)
print(f'分组对比已保存: {GROUP_XLSX}')

# (E) 分类变量类别数 xlsx
with pd.ExcelWriter(CAT_CARD_XLSX, engine='openpyxl') as writer:
    cat_card_summary.to_excel(writer, sheet_name='分类变量类别数汇总', index=False)
    for name, frame in cat_card_frames.items():
        frame.to_excel(writer, sheet_name=_safe_sheet_name(name), index=False)
print(f'分类变量类别数已保存: {CAT_CARD_XLSX}')

print(f'\\n===== EDA 全部完成，输出目录: {OUTPUT_DIR} =====')
print('产出文件:')
for f in os.listdir(OUTPUT_DIR):
    print(f'  {f}')
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open('notebooks/research/raw_data_eda.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"notebook 已重写，共 {len(cells)} 个 cell")
