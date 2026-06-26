# ATEL-Correction-Analysis-of-Recipe

Recipe 的相关性分析及自动调参。

## 项目简介

面向生产指标预测及逆向控制寻优：基于锅次级聚合数据，使用设定参数(S)与监控参数(M)预测指标统计量(Y)，并预留参数反向修正能力。

当前因原始数据时序顺序暂不可得，分步推进：
1. 当期 tabular 预测：用 N 锅次 S/M 预测 N 锅次指标统计量；
2. 时序预测（占位）：业务解决时序问题后预测 N+3 锅次；
3. 参数反向修正（占位）：搜索设定参数使预测指标落入合格区间。

## 目录结构

- `src/core/`：共享基础设施（constants / data / models / evaluation / analysis / utils）
- `src/tasks/`：按业务阶段划分的任务入口（current_prediction / forecasting / param_correction）
- `scripts/research_scripts/`：各模型训练运行脚本
- `outputs/`：输出结果（`analysis/` 分析图表、`models/` 权重与测评、`old/` 历史归档）
- `raw_data/`：原始数据（不纳入版本控制）

详细结构见 `项目代码结构说明.md`。

## 环境与运行

```bash
# 安装核心依赖
pip install -r requirements.txt

# 运行某模型训练与测评（需使用项目对应 Python 环境）
bash scripts/research_scripts/run_xgb.sh
```

模型权重与测评默认输出到 `outputs/models/{model_type}/`，分析报告输出到 `outputs/analysis/`。
