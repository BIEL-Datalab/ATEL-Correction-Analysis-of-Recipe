# 项目结构与输出重构方案

> 分支：`CV-1147-param-analysis` | 起始备份 commit：`backup: before refactor src structure and outputs layout`

## 一、背景与目标

原始数据当前无法获得真实时序顺序，故分两步推进：
1. **当期 tabular 预测**：用 N 锅次设定/监控参数预测 N 锅次指标统计量；
2. **时序预测**：业务解决时序问题后，引入时序模型预测 N+3 锅次；
3. **参数反向修正**（预留）：修改参数使预测指标满足业务范围。

本次重构目标：重排 `src/` 区分功能与任务、整理混乱的输出目录、顺带修复已知代码问题。原则：最小必要修改、不改外部行为、分步可回滚。

## 二、源码结构（src/）

```
src/
├── __init__.py
├── core/                              # 共享基础设施，任务无关
│   ├── __init__.py
│   ├── constants/{data,eval,train}_constants.py
│   ├── data/preprocess_and_split.py
│   ├── models/
│   │   ├── __init__.py  base.py  registry.py
│   │   ├── tabular/                  # xgb / lgbm / catboost / gandalf
│   │   └── timeseries/               # 占位（仅 __init__.py）
│   ├── evaluation/evaluator.py
│   ├── analysis/{anomaly_mining,data_description}.py
│   └── utils/arg_utils.py
└── tasks/
    ├── current_prediction/run_regression.py   # 步骤1 当期tabular
    ├── forecasting/                           # 步骤2 时序（占位）
    └── param_correction/                      # 步骤3 参数反向修正（占位）
```

**设计依据**：`core` 放所有任务可复用能力；模型按 `tabular`/`timeseries` 分类放共享层，不绑任务，便于任意任务调用任意类型模型；`tasks` 按业务阶段切分。

## 三、输出结构（outputs/）

```
outputs/
├── analysis/              # 新跑：分析报告图表
│   ├── data_eda/  target_analysis/  anomaly_records/
├── models/                # 新跑：模型权重与测评
│   └── {model_type}/{model_type}_{date}/
│       ├── weights/       # 权重 + 元信息
│       └── eval/          # metrics / 诊断图 / shap
└── old/                   # 历史归档，内容不动只迁移
    ├── analysis/          # ← results/research/{raw_data_eda,target_analysis,anomaly_records}
    └── models/
        ├── weights/       # ← models_result/research/{xgb,lgbm,catboost,gandalf}
        └── eval/          # ← results/research/{*_eval, xgb_nodate}
logs/                      # 保留（已 gitignore）
```

## 四、文件迁移映射（核心）

| 现位置 | 新位置 |
|---|---|
| `src/research/constants/*` | `src/core/constants/*` |
| `src/research/data/*` | `src/core/data/*` |
| `src/research/models/{base,registry}.py` | `src/core/models/` |
| `src/research/models/{xgb,lgbm,catboost,gandalf}_regressor.py` | `src/core/models/tabular/` |
| `src/research/evaluation/*` | `src/core/evaluation/*` |
| `src/research/analysis/*` | `src/core/analysis/*` |
| `src/research/utils/*` | `src/core/utils/*` |
| `src/research/experiments/run_regression.py` | `src/tasks/current_prediction/run_regression.py` |

## 五、顺带修复项（经确认）

| # | 位置 | 问题 | 修法 |
|---|---|---|---|
| 1 | 全局 | 导入路径统一为 `src.core`/`src.tasks` | 统一重写 |
| 2 | 新结构 | 补齐 `__init__.py` | 必做 |
| 3 | `run_regression.py:155` | gandalf.fit 多传第5参数；`--tune_method` 无用 | 删除多余传参与无用参数 |
| 4 | `xgb_regressor.py:278` | `obj.target_cols=[meta.get(...)]` 多包列表 | 改为 `meta.get("target_cols")`，save/load 对称，单/多目标结构一致 |
| 5 | `catboost_regressor.py:70`、`xgb_regressor.py:66` | `self.metrics` 缺 `=None` 初始化 | 补 `= None` |
| 6 | `arg_utils.py:4-5` | `import *` 星号导入 | 改显式导入 |
| 7 | `run_regression.py` | `--anomaly_export` 未接线 | 接入 `export_prediction_anomalies` |
| 8 | `output.txt` | 旧报错堆栈 | 删除 |

## 六、执行与 commit 计划

每步独立 commit，commit 前与用户确认纳入文件：

0. **备份**：提交 3 个已追踪文件改动 → `backup: before refactor src structure and outputs layout`
1. 目录结构调整（`git mv`）+ 补 `__init__.py` → `refactor: reorganize src into core and tasks`
2. 统一导入路径 → `refactor: rewrite imports to src.core and src.tasks`
3. 脚本路径 + 代码默认输出路径改写到 `outputs/` → `refactor: point outputs and scripts to new structure`
4. 历史产物归档到 `outputs/old/` → `chore: archive legacy results under outputs/old`
5. 清理无用文件（catboost_info/.pt_tmp/output.txt）+ 更新 .gitignore → `chore: remove temp artifacts and update gitignore`
6. 更新 `requirements.txt`（库==版本格式）+ 纳入追踪 → `chore: add requirements with pinned versions`
7. 更新 `项目代码结构说明.md` / `README.md` → `docs: update project structure description`
8. 修复圈定 bug → `fix: cleanup regressor loading and wiring issues`

## 七、回滚方式

- 任意步骤回滚：`git revert <commit>` 或 `git reset --soft <上一commit>`；
- 整体重回重构前：`git reset --hard <备份commit>` 的父提交（需用户明确要求才执行硬重置）。
