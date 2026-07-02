# Handoff — 项目结构与输出重构

> 日期：2026-06-26
> 分支：`CV-1147-param-analysis`
> 上一轮工作：项目结构重构与代码优化（已完成 8 步并提交）

---

## 当前目标

重构项目代码目录结构与输出目录，使其区分功能与任务、整理混乱输出；并顺带修复已知代码问题。整体目标已完成。

## 已确认事实

1. 业务背景：原始数据当前无法获得真实时序顺序，分三步推进：
   - 步骤1 当期 tabular 预测（用 N 锅次 S/M 预测 N 锅次指标统计量）—— 已就绪
   - 步骤2 时序预测（业务解决时序问题后预测 N+3）—— 占位
   - 步骤3 参数反向修正（搜索设定参数使预测指标落入合格区间）—— 占位
2. 源码结构采用 `src/core`（共享基础设施，任务无关）+ `src/tasks`（按业务阶段）。
3. models 按类型分 `tabular`/`timeseries`，不绑定具体任务。
4. 输出统一收口到 `outputs/`，分 `analysis/`（分析图表）、`models/`（权重与测评）、`old/`（历史归档）。
5. 运行环境为 conda 环境 `recipe`（Python 3.12），路径 `/home_ext/zzx/miniforge3/envs/recipe/bin/python`。系统默认 `python` 是 2.7，不可用。
6. `outputs/`、`logs/`、`raw_data/` 已 gitignore，不纳入版本控制。
7. `CLAUDE.md`、`PROMPT_RULES_TEMPLATE.md`、`recipe项目.txt`、`项目代码结构说明.md`、`docs/`、`notebooks/` 均未纳入 git 追踪（用户明确要求）。
8. `requirements.txt` 已纳入追踪，仅含 16 个核心依赖（库==版本）。

## 已完成内容（8 个 commit，均可独立回滚）

| commit | 类型 | 说明 |
|---|---|---|
| `3c50a3c` | backup | 备份重构前状态（3 个已追踪文件改动） |
| `1311f6a` | refactor | src 重排为 core+tasks，补 __init__.py，移除误追踪 .pyc |
| `12ce7dc` | refactor | 统一导入路径为 src.core / src.tasks.current_prediction |
| `2c3a6e7` | refactor | 输出路径收口到 outputs/{models,analysis}，脚本路径同步 |
| `f4802d4` | chore | 删除 catboost_info/.pt_tmp/output.txt，.gitignore 加 raw_data/ |
| `f1be665` | chore | requirements.txt 纳入追踪（核心依赖） |
| `9d17d71` | docs | 更新 README（结构+运行说明） |
| `6822d1b` | fix | 修复 xgb target_cols 包裹、metrics 初始化、anomaly_export 接线，删 tune_method 与 TabularModelTuner 死代码 |

### 最终目录结构

```
src/
├── core/                          # 共享基础设施
│   ├── constants/  data/  utils/
│   ├── models/{tabular, timeseries}/   # tabular: xgb/lgbm/catboost/gandalf; timeseries 占位
│   ├── evaluation/  analysis/
└── tasks/
    ├── current_prediction/run_regression.py   # 步骤1 入口
    ├── forecasting/                           # 步骤2 占位
    └── param_correction/                      # 步骤3 占位

outputs/
├── analysis/   # 新跑分析图表
├── models/     # 新跑权重与测评 {model_type}/{model_type}_{date}/{weights,eval}/
└── old/        # 历史归档 {analysis, models/{weights,eval}}/
```

## 已尝试但无效

无。

## 当前判断

重构目标已全部达成：结构清晰、导入统一、输出收口、已知 bug 已修、依赖与文档已更新。代码导入链与参数解析已通过验证（py_compile + 导入测试），但未做端到端训练验证（数据问题，非本次范围）。

## 未解决问题 / 遗留风险

1. **数据接入问题（需用户处理）**：
   - 脚本 `DATA_PATH=raw_data/df_filtered_outlier.csv` 文件不存在；
   - 当前真实数据为 `raw_data/raw_data_260623/data.parquet`；
   - `src/core/data/preprocess_and_split.py` 用 `pd.read_csv`，与 parquet 不匹配；
   - 需用户决定数据接入方式（改脚本路径 / 改 read_csv 为 read_parquet / 转格式等）。
2. **`gandalf_regressor.py:251` 笔误（用户要求暂不修）**：
   - `self.acc == "gpu" if use_gpu else "cpu"` 是无效比较表达式（`==` 应为赋值），结果被丢弃；
   - 用户说明：不同模型指定 GPU 方式不一致，此处仅想统一参数名，留待统一处理时修改；
   - 当前无功能副作用（`self.acc` 已在 `__init__` 正确设置）。
3. **未追踪文档**：`项目代码结构说明.md` 等本地文档不在版本库，他人 clone 看不到。
4. **catboost_info/.pt_tmp 未加 gitignore**：下次训练会再生成为未追踪文件（用户明确本次不加忽略项）。

## 推荐下一步

1. 解决数据接入问题（上述未解决问题 1），使脚本可端到端跑通并做整体验证；
2. 统一各模型 GPU 参数名时，一并修复 `gandalf_regressor.py:251` 笔误；
3. 业务解决时序问题后，在 `src/tasks/forecasting/` 与 `src/core/models/timeseries/` 实现步骤2；
4. 前向预测稳定后，在 `src/tasks/param_correction/` 实现步骤3 逆向寻优（DE/PSO）。

## 关键资料

- **命令**：
  - 运行训练：`bash scripts/research_scripts/run_xgb.sh`（需 recipe 环境）
  - 验证导入：`/home_ext/zzx/miniforge3/envs/recipe/bin/python -c "import src.core.models.registry"`
  - 安装依赖：`pip install -r requirements.txt`
- **文件**：
  - 结构说明：`项目代码结构说明.md`（未追踪）
  - 重构方案：`docs/REFACTOR_PLAN.md`（未追踪）
  - 协作规范：`CLAUDE.md`、`PROMPT_RULES_TEMPLATE.md`（未追踪）
  - 项目背景：`recipe项目.txt`（未追踪）
  - 任务入口：`src/tasks/current_prediction/run_regression.py`
  - 路径默认值：`src/core/utils/arg_utils.py`（DEFAULT_MODEL_DIR/RESULT_DIR）
- **日志**：`logs/`（gitignore）
- **回滚**：`git revert <commit>` 或整体 `git reset --hard 486e578`（重构前，需用户明确要求）

## 如果继续修改，下一轮优先先看什么

1. `docs/HANDOFF.md`（本文件）了解已完成进度与遗留问题；
2. `项目代码结构说明.md` 了解当前结构；
3. `src/tasks/current_prediction/run_regression.py` 了解主流程与输出路径逻辑；
4. 优先解决"未解决问题 1 数据接入"，再做端到端验证。
