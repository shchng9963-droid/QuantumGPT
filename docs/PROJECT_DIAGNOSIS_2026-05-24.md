# QuantumGPT Project Diagnosis — 2026-05-24

> 状态盘点 + 路线决策记录。配合 QuantumGPT_Plan_v2.1.md 的 Decision Log 阅读。

## 一、当前状态盘点

代码体量：~20K LOC, 88 Python 文件, 103 张可视化。

P0 agent maturity 已 commit (307434a, 9ce327a)：
- agent/state.py - AgentState + ArtifactRegistry + invalidation
- agent/drift_aware.py - DriftMonitor + ReplanningPolicy
- agent/memory.py - ExperimentRecord 集成
- agent/safety.py - dry-run + audit + amplitude/duration 阈值
- agent/reporting.py - trace -> markdown / talk_track
- demos/rabi_demo.py + showcase/physics_demo_output/ 标准化
- 195 tests passing; P0 行为测试 52 passed
- benchmark runner --save-traces, guard 防 silent mock fallback

未解锁：
- real-provider smoke 卡在 API key
- 60 任务 x 6 系统 main table 没跑过
- drift-aware 一键 demo 没落到 demos/ 里
- Lab backend 还只有 Dynamics 仿真, 没接外部 calibration data

## 二、瓶颈诊断

### 可用性瓶颈
1. 入口太多没主入口: qgpt CLI / streamlit / FastAPI / Rich console / demos/dayN / showcase, 没一个被打磨到能直接发给老师。
2. 输出分散在 4 个目录: showcase/, demos/dayN/, eval/, reports/baseline_week1/。
3. demo 故事只覆盖一半: 只有 Rabi 标准化, 没有 drift-aware 标准化 demo。
4. git 治理失控: 58 untracked 文件含 node_modules/ PPT eval/ web/ api/ presentation/ demos/day14/。

### Agent 研发瓶颈
5. Planner 是 1500 行 react.py 里的硬编码 if-else, AGENT_MATURITY_ROADMAP T2 接口未落地。
6. _run_openai / _run_anthropic 路径从未 e2e 验证, paper main claim 缺 LLM 数据点。
7. Memory 影响 planner 仅 "low fidelity history -> diagnose" 一条路径, 消融差异会很小。
8. Drift-aware 是 wrapper 不是 policy, 多 policy 组合会变成 wrapper 嵌套。
9. 任务分类靠 keyword, LLM 处理模糊任务的优势没发挥。

### 路线冲突
10. v2.1 / Phase3_Plan / PROJECT_GUIDE 三份计划方向不一致, 是最隐性的研发瓶颈。

## 三、决策（2026-05-24）

A. 同意降级 60-task + XGBoost predictor 为 stretch。
   - Phase3_Plan.md 已移到 stretch/Phase3_Plan_archived.md
   - PROJECT_GUIDE.md 第五节"下一步计划"中相关内容由 v2.1 Decision Log 覆盖

B. real-provider smoke provider 选定: DeepSeek
   - key 已由用户提供 (sk-6a09... 截断)
   - 写入 .env, 不进 git
   - 第一跑: tier1 子集 3 任务 x react_drift x deepseek-chat x save-traces

C. 同意归档 demos/day1-14 / web/ / api/ / presentation/ 到 archive/。

D. P1-4 planner 重构 -> delegate 给 subagent 一次性完成（200-400 行）。

## 四、本周 P0 待办

P0-1 路线收束（本文件 + v2.1 Decision Log + 归档 Phase3）
P0-2 git 治理（.gitignore + archive/ 目录）
P0-3 DeepSeek smoke + baseline 报告

下周 P1：
P1-4 planner 重构 (delegate)
P1-5 demos/drift_demo.py + showcase/drift_demo_output/
P1-6 memory ablation 真正出数据

下下周 P2：
P2-7 drift recovery 量化
P2-8 IBM public calibration replay backend
P2-9 paper method section 占位

## 五、保持稳定的不可侵入性约束

- 现有 195 tests 不能回归
- showcase/physics_demo_output/ 是 P0 已固化产物, 不动
- agent/ 6 个文件结构 (state/drift_aware/memory/safety/reporting/react) 在 planner 重构前不动
- benchmark/runner.py 接口稳定
