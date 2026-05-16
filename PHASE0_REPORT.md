# QuantumGPT · Phase 0 Progress Report
### *Device-Aware Closed-Loop Agent for NISQ Workflows under Drift*
### Phase 0: Foundation — Week 1–2 (14 days)

---

## Executive Summary

Phase 0 builds the complete foundation for QuantumGPT: a shadow quantum
hardware stack, an agentic reasoning loop, a production CLI, and a
structured experiment memory system. The system can now simulate quantum
circuits on 4 realistic IBM backends (127-qubit Brisbane, Kyiv, Sherbrooke,
Torino), detect hardware drift, run an autonomous agent loop, and persist
every experiment in a queryable database — all from a single command line.

**Key number**: from `pip install -e .` to a GHZ-5 fidelity readout
on a 127-qubit noisy backend takes **one command, ~12 seconds**.

```
$ qgpt simulate ghz --shots 8192 --backend FakeBrisbane
┌──────────────── Simulation Result ────────────────┐
│ Circuit:  ghz_5                                    │
│ Backend:  FakeBrisbane (127q)                      │
│ Fidelity: 0.9269                                   │
│ Depth:    23                                       │
│ Time:     12.4s                                    │
└────────────────────────────────────────────────────┘
```

---

## What Was Built (quantitative)

| Metric                    | Value         |
|---------------------------|---------------|
| Production Python code    | 7,918 lines   |
| Test code                 | 1,647 lines   |
| Documentation             | 3,874 lines   |
| Demo scripts              | 3,979 lines   |
| **Total codebase**        | **17,418 lines** |
| Test cases                | 129 passing, 0 failing |
| Demo days                 | 13 (each with summary, figures, runnable script) |
| Visualization outputs     | 48 PNG figures |
| CLI commands              | 6 (simulate, health, list, diagnose, agent, records) |

---

## Architecture (already working)

```
User prompt / CLI
       │
       ▼
┌──────────────────────────────────────────────┐
│  Agent Loop (agent/loop.py)                  │
│  • Anthropic Claude / OpenAI / Rule Planner  │
│  • While-loop with tool calling              │
│  • Max 15 turns, auto-synthesize report      │
└──────┬───────────────────────────────────────┘
       │ tool_use
       ▼
┌──────────────────────────────────────────────┐
│  Tool Layer (tools/quantum_tools.py)         │
│  5 tools: health, qubits, run, list, diag    │
│  + MCP-compatible JSON schemas               │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Shadow Backend Layer (backends/)            │
│  FakeAdapter │ ReplayBackend │ SyntheticDrift│
│     127q    │  time-series  │  T1/T2 decay  │
│  (4 IBM     │  calibration  │  + hotswap    │
│   devices)  │  replay       │  recovery     │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Data Layer (data/)                          │
│  DuckDB: health / runs / sessions / drift    │
│  ExperimentRecord: 27-field structured memory│
│  InstrumentedExecutor: auto-log everything   │
│  W&B integration ready                       │
└──────────────────────────────────────────────┘
```

---

## 5 Highlights for Demo

### 1. Shadow Hardware Stack — 3 Backend Types, 4 IBM Devices

不是玩具模拟器，而是完整的影子硬件体系：

- **FakeBackendAdapter**: 包装 IBM 的 FakeBrisbane/Kyiv/Sherbrooke/Torino (127 qubits)，使用真实噪声模型
- **ReplayBackend**: 回放真实校准时间序列，模拟一周内 T1/T2/gate error 的自然漂移
- **SyntheticDriftBackend**: 参数化注入漂移（余弦/指数衰减/阶跃），支持 hotswap 恢复

三种后端共享同一 `ShadowBackend` 抽象接口，切到真机只需换一个 adapter。

### 2. Autonomous Agent Loop — Natural Language → Quantum Results

Agent 接收自然语言指令，自主决策工具调用顺序：

```
$ qgpt agent "Check health and run GHZ-5"

[tool] get_backend_health({})        ← 自主决定先检查健康
[tool] run_circuit({"circuit_name": "ghz_5", "shots": 4096})

## QuantumGPT Analysis Report
Backend Health: FakeBrisbane — 127 qubits
  Avg T1: 224.5 μs | Avg 2Q error: 0.009141
Circuit: ghz_5 — Fidelity: 0.9268, Depth: 23
```

支持 3 种 LLM provider (Claude / OpenAI-compatible / 离线 Rule Planner)，
无 API key 也能完整运行。

### 3. Pulse-Level Simulation — Rabi Oscillation from Hamiltonian

Day 10 用 Qiskit Dynamics 从 Hamiltonian 层面仿真 Rabi 振荡：

- 定义 transmon Hamiltonian (4.8 GHz qubit, 200 MHz 耦合)
- 扫描驱动振幅 → 拟合 Rabi 频率
- 提取 π-pulse 幅度 (0.312)，R² = 0.997
- 生成 Rabi chevron、Bloch sphere 轨迹、π-calibration 图

这展示了 agent 未来的 **跨层能力**：不仅操作 gate-level 电路，
还能深入 pulse-level 调参。

### 4. Drift Detection & Streaming Telemetry

实时校准数据流 + 变化点检测：

- `PropertiesStream`: 每 N 秒 emit 一次校准 snapshot
- `SyntheticDriftBackend`: T1 衰减 70μs → qubit "死亡"后 hotswap 恢复
- CUSUM 变化点检测，ARL0 = 500，延迟 < 3 步
- 分级告警 (nominal / warning / critical) + 自动恢复建议

### 5. ExperimentRecord — Agent's Structured Memory

不是简单的 (circuit, fidelity) 日志，而是 27 字段的结构化实验记忆：

```
ExperimentRecord:
  Identity:   id, timestamp, type
  Context:    backend, circuit, problem, tags
  Plan:       intended tool sequence
  Execution:  actual tool calls, decisions, tokens
  Results:    fidelity, outcome, metrics
  Fitting:    pi_amp, rabi_freq, R² (for pulse experiments)
  Snapshot:   backend health at experiment time
  Reflection: summary, lessons learned, failure reason
```

每次 agent 执行自动写入 DuckDB，支持 14 种查询 API。
`summary` 字段专为 RAG 向量检索设计 — Phase 3 直接接入 Qdrant。

---

## Benchmark Circuits — Fidelity Baseline

在 4 个 IBM 127-qubit 后端上测试 5 种电路 + 5 种 MQTBench 电路：

| Circuit  | Type      | FakeBrisbane | FakeKyiv | Notes |
|----------|-----------|:---:|:---:|---|
| GHZ-5    | 纠缠态    | 0.924 | 0.949 | 对噪声最敏感 |
| QFT-4    | 傅里叶变换 | 0.999 | 0.997 | 浅电路，几乎无损 |
| BV-5     | 预言机     | 0.946 | 0.954 | Oracle + Hadamard |
| VQE-4    | 变分       | 0.980 | 0.985 | 硬件高效 ansatz |
| QAOA-4   | 优化       | 0.957 | 0.965 | MaxCut ring graph |

所有电路在所有后端上 fidelity > 0.9，系统可靠。

---

## Test Coverage

```
129 passed, 1 skipped (MQTBench optional), 0 failures

test_advisor.py            — 14 tests (advisor logic + streaming)
test_backends.py           — 22 tests (3 backend types + interface)
test_cli.py                — 14 tests (all 6 CLI commands)
test_experiment_record.py  — 19 tests (CRUD + search + stats)
test_mqtbench.py           —  1 test  (optional import)
test_rabi.py               — 12 tests (Rabi simulation + fitting)
test_replay_backend.py     —  9 tests (calibration replay)
test_streaming.py          — 38 tests (telemetry + drift detection)
```

---

## CLI — 6 Commands, JSON-Ready

| Command | Purpose | Supports -j |
|---------|---------|:-----------:|
| `qgpt simulate <circuit>` | 跑电路出 fidelity | ✓ |
| `qgpt health` | 后端 T1/T2/错误率/漂移 | ✓ |
| `qgpt list` | 列出可用电路 | ✓ |
| `qgpt diagnose` | 诊断 + 建议 | ✓ |
| `qgpt agent "<prompt>"` | 自然语言 agent 模式 | ✓ |
| `qgpt records [--stats]` | 查询实验记录 | ✓ |

所有命令支持 `--backend` 切换 4 个设备，`-j` 输出机器可读 JSON。

---

## Phase 0 Checklist

| Deliverable | Status | Day |
|-------------|:------:|:---:|
| Shadow hardware (3 backend types) | ✅ | 3–4 |
| MQTBench 集成 (10 circuits) | ✅ | 5 |
| Agent loop (Claude/OpenAI/Mock) | ✅ | 6 |
| W&B + DuckDB 数据层 | ✅ | 7 |
| SyntheticDrift + telemetry stream | ✅ | 8–9 |
| Qiskit Dynamics Rabi 仿真 | ✅ | 10 |
| BACKENDS.md + CI green | ✅ | 11 |
| MVP CLI (`qgpt` command) | ✅ | 12 |
| ExperimentRecord schema | ✅ | 13 |
| 129 tests, 0 failures | ✅ | — |
| 48 visualization figures | ✅ | — |

---

## What Comes Next — Phase 1 Preview

Phase 1 (Week 3–4) 感知层:
1. 6 个感知工具 (coupling map, drift detection, backend comparison...)
2. Drift Detector 攻关 (PELT / Bayesian Online Changepoint)
3. Rabi notebook 端到端 (Hamiltonian → pulse → IQ → fit → report π-amp)
4. Operator Console 接实时数据

**目标**：drift detector 在 100 个标注点上 F1 >= 0.7。

---

## Risk Status

| Risk | Status | Mitigation |
|------|:------:|------------|
| Qiskit Dynamics 上手 | ✅ 已消除 | Day 10 Rabi 全流程跑通 |
| LLM tool calling 不稳定 | ✅ 已消除 | 3 provider + Mock fallback |
| 真机拿不到 | 🟡 预期中 | ReplayBackend 就是 Plan B |
| 12 周做不完 | 🟢 健康 | Phase 0 按期完成，进度正常 |

---

*Phase 0 completed: Day 14 of 84-day plan (17% elapsed)*
*Target: ICLR 2027 (deadline ~2026-10-03)*
