# Agent 成熟度路线图

> 本文件服务于 QuantumGPT v2.1：后续优先优化 agent 层，而不是继续横向堆工具。

---

## 1. 目标

把当前 ReAct + tools 系统升级成成熟的 quantum experiment agent runtime。

成熟不是指 LLM 更大，而是指 agent 具备以下能力：

- 状态显式：知道自己依赖了哪些 backend snapshot 和实验结果；
- 证据管理：知道哪些结果有效、过期、失效或被新结果替代；
- 重规划：设备漂移或工具失败后能主动改变计划；
- 记忆：能复用过去实验，而不是每次从零开始；
- 安全：lab 相关动作默认 dry-run/audit；
- 可解释：每一步可导出给人看的 trace。

---

## 2. 当前已有基础

相关文件：

- `agent/react.py`：ReActAgent、AgentTrace、TraceStep、diagnostics。
- `agent/state.py`：AgentState、ArtifactRegistry、ArtifactType、InvalidationReason。
- `agent/drift_aware.py`：DriftMonitor、ReplanningPolicy、DriftAwareRulePlanner。
- `agent/memory.py`：AgentMemory、ExperimentRecord 集成。
- `data/experiment_record.py`：ExperimentRecord 和 ExperimentStore。
- `tests/unit/test_agent_state.py`：state 单测。
- `tests/unit/test_agent_state_integration.py`：agent/state/drift 集成测试。
- `tests/test_memory_integration.py`：memory 集成测试。

---

## 3. 成熟 agent 的模块边界

建议把 agent 层长期拆成 6 个稳定概念：

```text
User Task
  -> TaskInterpreter        任务分类、目标抽取、风险识别
  -> AgentPlanner           生成计划和下一步动作
  -> ToolExecutor           执行工具，返回结构化 observation
  -> AgentState             记录 artifact、budget、drift、safety、memory
  -> Policy Layer           budget / drift / safety / memory 等策略
  -> TraceReporter          生成人类可读报告和 benchmark trace
```

现阶段不需要大改目录，但新增功能要按这个边界写，避免把所有逻辑塞进 `ReActAgent._run_mock()`。

---

## 4. P0 行为测试清单

这些测试优先级高于新增工具。

### 4.1 状态和 artifact

文件：`tests/unit/test_agent_state_integration.py`

必须覆盖：

1. health check 产生 backend_snapshot artifact；
2. run_circuit 依赖 backend_snapshot；
3. drift 后 circuit_result 标记 stale；
4. trace JSON 包含 state_summary 和 artifacts；
5. benchmark 保存 trace 时不丢 state 字段。

### 4.2 planner 成熟行为

文件：建议新建 `tests/unit/test_agent_planner_behavior.py`

必须覆盖：

1. 模糊任务先感知：prompt = "make this circuit better"，第一步应 health/check/list，而不是直接 run；
2. 低保真后 fallback：run_circuit fidelity 低于目标时，下一步应 mitigation 或 diagnose；
3. drift 后 replan：必须重新 get_backend_health，然后 rerun；
4. rabi 任务顺序：必须 rabi_experiment -> fit_rabi -> final report；
5. 工具失败后不死循环：同一失败工具最多重试一次，然后 fallback/stop。

### 4.3 memory 行为

文件：`tests/test_memory_integration.py`

必须覆盖：

1. agent run 自动保存 ExperimentRecord；
2. 第二次相似任务能检索到 memory_context；
3. 低保真历史会影响下一次 plan；
4. `use_memory=False` 时 trace 不含 memory_context；
5. records CLI 能读到 agent 产生的记录。

### 4.4 safety 行为

文件：建议新建 `tests/unit/test_agent_safety.py`

必须覆盖：

1. lab/pulse 工具默认 dry-run；
2. amplitude 超阈值被 block；
3. update_calibration 在无人工确认时拒绝；
4. safety event 进入 trace diagnostics 或 AgentState.safety_status；
5. final report 标明 simulation/read-only/dry-run。

---

## 5. P0 实现任务

### T1. TraceReporter

新增文件：`agent/reporting.py`

职责：

- `summarize_trace_for_human(trace, audience="physics") -> str`
- `summarize_trace_for_paper(trace) -> dict`
- `extract_demo_highlights(trace) -> list[str]`

physics audience 输出应包含：

- 任务目标；
- 使用的 backend/device；
- 执行步骤；
- 关键数值；
- 是否 dry-run；
- 是否触发 drift/safety；
- 下一步建议。

### T2. Planner 接口

新增文件：`agent/planner.py`

建议接口：

```python
class AgentPlanner:
    def classify_task(self, prompt: str, state: AgentState) -> str: ...
    def propose_plan(self, task_type: str, state: AgentState) -> list[dict]: ...
    def next_action(self, state: AgentState, last_observation: str | None) -> dict | None: ...
    def handle_failure(self, state: AgentState, error: dict) -> dict | None: ...
```

先把当前 rule planner 的逻辑迁进去，不追求 LLM planner。

### T3. SafetyPolicy

新增文件：`agent/safety.py`

职责：

- 判断工具是否属于 lab/pulse/calibration risk action；
- 自动加 dry-run；
- 检查 amplitude、duration、shots 等简单阈值；
- 生成 audit event；
- 将结果写入 `AgentState.safety_status`。

### T4. Memory influence planner

修改文件：

- `agent/react.py`
- `agent/memory.py`
- 可能新增 `agent/planner.py`

目标：

- memory 不只是 prompt text；
- planner 能读 memory_context；
- 若历史记录显示某 circuit/backend fidelity consistently low，则 planner 优先 diagnose/mitigate/compare_backend。

### T5. Demo outputs standardization

修改或新增：

- `demos/rabi_demo.py`
- `demos/physics_lab_demo.py`
- `showcase/run_physics_demo.sh`
- `showcase/physics_demo_handout.md`

每次 demo 固定输出：

```text
showcase/physics_demo_output/
  trace.json
  report.md
  rabi_oscillation.png
  rabi_oscillation.pdf
  talk_track_30s.md
```

---

## 6. 验证命令

核心 agent 验证：

```bash
cd /home/wangshuchang/quantumgpt
python -m pytest tests/unit/test_agent_state.py tests/unit/test_agent_state_integration.py tests/unit/test_agent_trace_diagnostics.py -q
```

memory 验证：

```bash
python -m pytest tests/test_memory_integration.py -q
```

lab/backend 验证：

```bash
python -m pytest tests/unit/test_lab_backend.py tests/unit/test_rabi.py -q
```

benchmark smoke：

```bash
python benchmark/runner.py --systems react_drift --tiers 1 --max-tasks 2 --provider mock --save-traces --output reports/baseline_week1/mock_react_drift_state_smoke.json
```

physics demo：

```bash
PYTHONPATH=. python3 demos/rabi_demo.py
```

---

## 7. 完成定义

Agent 层成熟度达到 v2.1 的标准：

- P0 行为测试全部通过；
- trace 中可见 state/artifact/drift/memory/safety；
- demo report 可直接给非 CS 的物理老师阅读；
- benchmark smoke 不回退；
- 当前系统贡献可以用一句话讲清楚：

> QuantumGPT is a stateful, drift-aware, memory-augmented, safety-gated agent runtime for quantum experiments.
