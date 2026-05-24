# QuantumGPT v2.1 后续工作计划

> 定位：v2.1 不是重复 v1/v2 的愿景文档，而是把接下来 3-4 周的工程和展示目标收束到两个核心问题：
>
> 1. 把 QuantumGPT 的 agent 层做成熟，形成可写进论文的方法贡献；
> 2. 准备一个物理实验口老师能看懂、愿意进一步沟通的 demo。

---

## Decision Log

### 2026-05-24 — v2.1 钉为唯一主线

经过项目状态盘点（见 docs/PROJECT_DIAGNOSIS_2026-05-24.md），确认：

- **唯一主线**：v2.1 的 6 个 agent 能力（state / artifact / drift / memory / safety / trace）+ 2 条 demo（Rabi physics + drift-aware AI）。
- **降级为 stretch**（暂停推进，归档到 stretch/）：
  - Phase3_Plan.md 中的 "60 task × 6 系统 main table"
  - PROJECT_GUIDE.md 中的 "Fidelity Predictor (XGBoost/LightGBM)"
  - QC-Agent-Bench 60 任务 spec
  - Web Dashboard / FastAPI / Streamlit 面板
  - ICLR 2027 deadline 表（不是不投，但 v2.1 阶段不以它为节奏）
- **Benchmark 范围**：tier1 + tier2 子集（约 20 任务），不追全量 60。
- **Paper 4 个贡献点（最终叙事）**：
  1. State-aware quantum agent runtime
  2. Drift-aware replanning
  3. Experiment memory
  4. Safety-first lab copilot

后续若要回到 stretch 路线，需要先在此 Decision Log 追加新条目并说明依据。

---

## 0. 当前判断

代码现状已经不是从 0 开始：

- `agent/react.py` 已有 ReAct loop、budget、memory、drift-aware wrapper、trace。
- `agent/state.py` 已有显式 AgentState、ArtifactRegistry、artifact invalidation。
- `agent/drift_aware.py` 已有 DriftMonitor、ReplanningPolicy、DriftAwareRulePlanner。
- `agent/memory.py` 和 `data/experiment_record.py` 已有 ExperimentRecord 记忆雏形。
- `backends/lab_backend.py`、`dynamics_lab_adapter.py`、`experiments/rabi_experiment.py` 已经支撑最小 lab demo。
- `demos/rabi_demo.py`、`showcase/`、`DEMO_HANDBOOK.md` 已经有基础展示材料。

因此 v2.1 的目标不是继续横向加功能，而是纵向把 agent 做“可信、可解释、可评测、可展示”。

---

## 1. v2.1 总目标

### 1.1 研究目标

把 QuantumGPT 从“ReAct + tools”升级为：

> 一个具备显式实验状态、证据失效机制、漂移感知重规划、实验记忆、安全门控和可审计 trace 的量子实验 agent runtime。

这句话要成为后续论文和 demo 的主叙事。

### 1.2 Demo 目标

给物理实验口老师看的 demo 不能从“LLM 很聪明”讲起，而要从他们每天的痛点讲起：

> 设备参数漂移、重复 tune-up、手工记录分散、实验决策依赖经验。QuantumGPT 可以先只读观察和 dry-run，帮学生更快完成诊断、拟合、建议和记录。

demo 要让老师看到三件事：

1. 它懂基本实验流程，不只是写 Qiskit 代码；
2. 它有刹车：dry-run、安全检查、audit log，不会直接乱动设备；
3. 它能沉淀经验：ExperimentRecord 让组内知识可检索、可复现。

---

## 2. v2.1 范围控制

### 2.1 必做，不再扩散

v2.1 只做以下 6 个 agent 能力：

1. Explicit State：显式记录当前任务、backend snapshot、artifact、budget、drift、safety。
2. Artifact Validity：工具结果不是字符串，而是带依赖和状态的证据对象。
3. Drift-Aware Replanning：设备漂移后，旧结果 stale，planner 强制重新检查和 rerun。
4. Experiment Memory：每次 agent run 自动落 ExperimentRecord，下次能检索历史。
5. Safety Gate：所有 lab/pulse/calibration 风险动作先 dry-run + audit，写操作默认关闭。
6. Demonstrable Trace：每个 demo 都能导出一条老师能看懂的 trace/report。

### 2.2 暂不做或只作为 stretch

以下内容不要进入 v2.1 主线：

- 35 个工具全量完成；
- full tune-up chain 全自动闭环；
- 真机写权限；
- cross-device transfer 主实验；
- agentic RL；
- LangGraph 深度改造；
- 90 任务大 benchmark。

这些可以保留在 v2 长线里，但 v2.1 不以它们为完成标准。

---

## 3. Agent 层成熟度路线图

### Stage A：Stateful Agent Core，1-2 天

目标：让所有 agent 行为围绕 AgentState，而不是围绕纯 message history。

验收标准：

- `AgentTrace.to_dict()` 必须包含：
  - `state_summary`
  - `artifacts`
  - `drift_state`
  - `diagnostics`
- 每个关键工具调用后都能产生或更新 artifact。
- stale artifact 在 trace 里可见。
- benchmark runner 保存的 trace 也包含这些字段。

建议任务：

1. 补齐 `run_circuit`、`apply_mitigation`、`rabi_experiment`、`fit_rabi` 的 artifact 类型和依赖关系。
2. 给 `AgentState` 增加一个 `decision_log` 或把关键 decision 明确写入 `TraceStep.state_summary`。
3. 写一个 `agent explain` 风格的 trace summarizer，把 trace 转成中文/英文人类可读报告。

### Stage B：Robust Planner，2-3 天

目标：planner 不只是按固定顺序调工具，而要体现“实验决策能力”。

需要成熟的行为：

1. 任务分类：
   - circuit benchmark
   - backend diagnosis
   - drift recovery
   - rabi tune-up
   - ambiguous user request
2. 计划生成：先写出 expected plan，再执行。
3. 工具失败恢复：如果工具报错或低保真，能选择 fallback。
4. 信息不足时先感知，不直接执行。
5. 重复工具调用抑制：避免无意义 loop。

建议实现：

- 新增或重构 `AgentPlanner` 抽象：
  - `classify_task(prompt, state)`
  - `propose_plan(task, state)`
  - `next_action(state, observation)`
  - `handle_failure(state, error)`
- 保留当前 rule planner 作为 deterministic baseline。
- 真实 LLM planner 后续只要实现同一接口。

验收标准：

- 至少 5 个行为测试：
  1. 模糊任务先 health check；
  2. 低 fidelity 后尝试 mitigation；
  3. drift 后强制 recheck + rerun；
  4. rabi 任务必须 experiment 后 fit；
  5. 工具失败后不死循环。

### Stage C：Memory-Using Agent，2 天

目标：不是只“存记录”，而是 agent 会使用历史经验。

成熟行为：

- 运行任务前检索相似 ExperimentRecord；
- 若历史中某 backend/circuit 表现差，planner 会避开或先 mitigation；
- final answer 说明参考了哪些历史记录；
- benchmark 中能开关 `use_memory` 做消融。

验收标准：

- 第一次跑 GHZ-5 记录结果；
- 第二次跑相似任务时 trace 中出现 memory_context；
- 如果历史 fidelity 低，agent 的 plan 发生变化；
- `qgpt records` 可以看到对应记录。

### Stage D：Safety-Gated Lab Agent，2 天

目标：形成给物理老师看的“不会乱动设备”的基本可信度。

P0 安全规则：

- 所有 pulse/lab/calibration 工具先记录 audit event；
- `dispatch_pulse`、`update_calibration`、未来写设备动作默认 `dry_run=True`；
- 超过 amplitude/duration 简单阈值时 block；
- final answer 明确标注：本次为 dry-run / simulation / read-only。

验收标准：

- Rabi demo 的 trace/report 中出现 safety section；
- 人为设置过大 amplitude 时触发 safety block；
- safety violation 作为 artifact 或 diagnostic 进入 trace。

### Stage E：Demo-Ready Agent，2-3 天

目标：把工程能力打包成两个 5 分钟 demo。

Demo 1：漂移感知量子电路 agent

- 场景：同一个 GHZ/QFT 电路，backend drift 后标准 ReAct 继续用旧判断，QuantumGPT 检测漂移并重跑。
- 展示点：state、stale artifact、replan、恢复结果。
- 面向 AI/系统审稿人。

Demo 2：Rabi tune-up lab copilot

- 场景：老师/学生输入“帮我给 Q0 做一个 Rabi，拟合 pi pulse，并生成实验记录”。
- 展示点：实验流程、拟合图、pi/half-pi amplitude、dry-run、安全日志、ExperimentRecord。
- 面向物理实验老师。

验收标准：

- 两个 demo 都能一键运行；
- 每个 demo 输出：
  - trace JSON
  - markdown report
  - 1 张核心图
  - 30 秒讲解词
- 不依赖真机和外部 API key。

---

## 4. Demo 给物理实验口老师的叙事设计

### 4.1 不要这样讲

避免开场说：

- “我们做了一个 LLM agent”；
- “用了 ReAct / LangGraph / tool calling”；
- “可以自动控制量子设备”。

这些对物理老师要么不重要，要么会引起安全顾虑。

### 4.2 应该这样讲

推荐开场：

> 老师您好，我们现在不是想让 AI 直接接管设备，而是先做一个只读和 dry-run 的实验副驾。它可以把学生每天重复做的几件事自动串起来：看设备状态、跑标准表征、拟合参数、给出下一步建议、把实验记录成结构化日志。现在 demo 在仿真后端上跑，接口设计成后续可以接 QCoDeS/pyqum/Labber，但写设备动作默认关闭。

### 4.3 5 分钟 demo 结构

0:00-0:40 痛点

- tune-up 重复；
- 数据、拟合、笔记分散；
- 新学生依赖老学生经验；
- 设备漂移导致旧判断失效。

0:40-1:20 系统边界

- 现在只做 read-only/dry-run；
- 不直接写 calibration；
- 每步都有 audit log；
- 支持仿真、历史数据回放，未来接实验软件 adapter。

1:20-3:20 现场 demo：Rabi

命令：

```bash
cd /home/wangshuchang/quantumgpt
PYTHONPATH=. python3 demos/rabi_demo.py
```

展示输出：

- Rabi curve；
- fitted pi amplitude；
- fit R^2；
- agent trace；
- ExperimentRecord。

3:20-4:20 第二个画面：drift-aware trace

展示：

- backend health snapshot；
- drift detected；
- stale result；
- agent recheck/rerun。

4:20-5:00 合作请求

> 我们想先做低负担合作：访谈 30 分钟，了解您组 tune-up 流程；如果方便，再用一份脱敏历史数据做离线回放。短期不会接写权限，只做 read-only 分析和报告生成。

---

## 5. 近期执行计划

### Day 1：整理 agent trace 和 demo 报告

产出：

- `docs/AGENT_MATURITY_ROADMAP.md`
- `docs/PHYSICS_DEMO_PLAYBOOK.md`
- Rabi demo report 增加 safety / audit / ExperimentRecord 说明。

验证：

```bash
PYTHONPATH=. python3 demos/rabi_demo.py
python -m pytest tests/unit/test_agent_state.py tests/unit/test_agent_state_integration.py -q
```

### Day 2：补 planner 行为测试

产出：

- 补 5 个成熟 agent 行为测试。
- 明确当前 rule planner 的不足和目标行为。

测试目标：

```bash
python -m pytest tests/unit/test_agent_state_integration.py tests/unit/test_agent_trace_diagnostics.py -q
```

### Day 3：安全门控最小版

产出：

- lab/pulse 工具 audit event；
- dry-run 标志进入 trace；
- amplitude/duration 简单阈值 block。

测试目标：

```bash
python -m pytest tests/unit/test_lab_backend.py tests/unit/test_agent_state_integration.py -q
```

### Day 4：Memory 真正影响 planner

产出：

- `memory_context` 注入 planner；
- 历史低保真触发 mitigation 或 backend compare；
- 消融开关可评测。

测试目标：

```bash
python -m pytest tests/test_memory_integration.py tests/unit/test_agent_state_integration.py -q
```

### Day 5：两条 demo 一键化

产出：

- `demos/physics_lab_demo.py` 或 `showcase/run_physics_demo.sh`；
- drift demo + rabi demo 的统一输出目录；
- 一页中文 handout。

验证：

```bash
bash showcase/run_all_demos.sh
```

---

## 6. 后续论文贡献表达

建议把 agent 层写成 4 个贡献点：

1. State-aware quantum agent runtime：显式 state + artifact dependency。
2. Drift-aware replanning：硬件漂移导致证据失效，触发 recheck/rerun。
3. Experiment memory：结构化 ExperimentRecord 支持可复现和经验复用。
4. Safety-first lab copilot：dry-run、audit、guardrail 让系统能走向物理实验室。

注意：LangGraph/MCP/LLM provider 都应写成工程实现，不要写成核心贡献。

---

## 7. v2.1 完成标准

v2.1 不以工具数量为标准，而以 agent 成熟度为标准。

必须同时满足：

- 关键 agent 行为有测试；
- drift/memory/safety 三个能力都在 trace 里可见；
- 至少一个 physics-facing Rabi demo 可以一键运行；
- 至少一个 drift-aware demo 可以一键运行；
- demo 输出能直接发给物理老师，不需要解释代码；
- benchmark smoke 仍能跑通。

如果这些完成，再考虑扩展更多 lab 工具。
