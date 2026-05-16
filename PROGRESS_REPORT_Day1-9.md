# QuantumGPT — Phase 0 Progress Report (Day 1–9)

> **项目定位**: 第一个具备设备感知（device-aware）的闭环量子 agent，能在 NISQ 漂移条件下跨 gate-level 和 pulse-level 调度工具、保持任务成功率，配套发布 QC-Agent-Bench。
>
> **Paper**: *Quantum GPT: A Device-Aware Closed-Loop Agent for NISQ Workflows under Drift*
>
> **目标会议**: ICLR 2027 Main（deadline ~2026-10-03）

---

## 1. 总览

| 指标 | 数值 |
|------|------|
| 代码量 | 8,618 行 Python（不含 venv/依赖） |
| 模块数 | 7 个核心模块 + demos |
| 单元测试 | 83 个，100% 通过 |
| Demo 场景 | 9 天，共 34 张可视化图 |
| 后端类型 | 3 种 shadow backend（Fake / Replay / SyntheticDrift） |
| Agent 工具 | 5 个 quantum-specific 工具 |
| 数据持久化 | DuckDB（5 表）+ W&B 离线跟踪 |

---

## 2. 系统架构（Day 9 状态）

```
                          ┌─────────────────────────┐
                          │     User / CLI / LLM     │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   QuantumAgent (Day 6)   │
                          │   LLM tool-calling loop  │
                          │   Anthropic / OpenAI /   │
                          │   DeepSeek / Rule-based  │
                          └────────────┬────────────┘
                                       │ tool calls
                          ┌────────────▼────────────┐
                          │   ToolExecutor (Day 6)   │
                          │   5 quantum tools        │
                          └────────────┬────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
   ┌──────────▼──────────┐  ┌─────────▼─────────┐  ┌──────────▼──────────┐
   │ CalibrationAdvisor  │  │  StreamingAdvisor  │  │ InstrumentedExecutor│
   │     (Day 8)         │  │     (Day 9)        │  │     (Day 7)         │
   │ monitor→diagnose→act│  │ stream→track→alert │  │ auto-log to DB+W&B │
   └──────────┬──────────┘  └─────────┬─────────┘  └──────────┬──────────┘
              │                        │                        │
              │              ┌─────────▼─────────┐              │
              │              │ PropertiesStream   │              │
              │              │ (Day 3, enhanced)  │              │
              │              │ background thread  │              │
              │              │ time acceleration  │              │
              │              └─────────┬─────────┘              │
              │                        │                        │
   ┌──────────▼────────────────────────▼────────────────────────▼──────────┐
   │                        ShadowBackend 抽象层 (Day 3)                   │
   │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐  │
   │  │ FakeBackendAdapt │ │  ReplayBackend   │ │ SyntheticDriftBack.  │  │
   │  │ (Day 3)          │ │  (Day 2,4)       │ │ (Day 4, enhanced)    │  │
   │  │ wraps Qiskit     │ │  历史回放 +       │ │ 参数化漂移函数:       │  │
   │  │ FakeBrisbane     │ │  合成漂移序列     │ │ STABLE / LINEAR /    │  │
   │  │ 127 qubits       │ │  20 qubits        │ │ SUDDEN / DIURNAL     │  │
   │  └──────────────────┘ └──────────────────┘ └──────────────────────┘  │
   └──────────────────────────────┬────────────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │         DuckDB (Day 7)      │
                    │ health_snapshots            │
                    │ qubit_properties            │
                    │ circuit_runs                │
                    │ agent_sessions              │
                    │ drift_events                │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │    W&B Offline (Day 7)      │
                    │ metrics / charts / sync     │
                    └────────────────────────────┘
```

---

## 3. 每日交付物明细

### Day 1 — 环境搭建
- 建仓 `quantum-gpt/` monorepo 结构
- 安装 Qiskit 1.x + Aer + Mitiq + qiskit-ibm-runtime
- `pip list` 无报错

### Day 2 — 基础后端 + 漂移模型
- `CalibrationSnapshot` 数据类：从 FakeBrisbane 提取 T1/T2/readout/gate error/coupling map
- `ReplayBackend` v1：合成历史漂移序列，支持 `set_time(hours)` 回放
- 漂移模型：随机游走 + 趋势 + 周期（正弦）+ 噪声
- **3 张图**：T1 drift, qubit heatmap, coupling map

### Day 3 — 抽象层 + Benchmark 电路 + PropertiesStream
- `ShadowBackend` ABC：统一 `get_health()` / `get_qubit_properties()` / `run()` / `get_coupling_map()` 接口
- `FakeBackendAdapter`：包装 Qiskit FakeBackendV2 为 ShadowBackend
- 5 个手写 benchmark 电路（GHZ-5, QFT-4, BV-5, VQE-4, QAOA-4）
- `PropertiesStream`：后台线程定时 emit snapshot，支持 observer 模式
- **2 张图**

### Day 4 — SyntheticDriftBackend + API 对齐
- `SyntheticDriftBackend`：参数化漂移注入，4 种预设 profile
  - `STABLE`：无漂移
  - `LINEAR_DECAY`：T1 线性下降，gate error 线性上升
  - `SUDDEN_DEGRADATION`：t=5h 阶跃退化
  - `DIURNAL_CYCLE`：24h 正弦周期
- 支持 per-qubit 覆写、profile 热替换
- 完整噪声模型构建（thermal relaxation + depolarizing + readout error）
- **4 张图**：4 种 profile 对比

### Day 5 — MQTBench 集成
- 接入 MQTBench（慕尼黑量子工具），自动生成 EfficientSU2 native-gate 电路
- 5 个门级 benchmark：GHZ-5, DJ-5, GraphState-5, QFTent-5, VQE_SU2-4
- 支持电路元数据（depth, ECR count, qubit count）
- **5 张图**：fidelity baseline 对比

### Day 6 — Agent Loop（prompt → tools → result）
- `QuantumAgent`：while-loop agent，支持 3 种 LLM 后端
  - Anthropic Claude（原生 tool_use）
  - OpenAI-compatible（DeepSeek V4-Flash 实际验证）
  - Rule-based mock（离线测试）
- 5 个量子工具的 JSON Schema 定义 + ToolExecutor 调度器
- `RulePlanner`：确定性规划器，模拟 LLM 多步工具调用
- **4 张图**：agent 工作流 + 结果

### Day 7 — W&B + DuckDB 数据管道
- `DataStore`（DuckDB）：5 表 schema + 查询 API（recent_runs, fidelity_trend, best/worst_qubits）
- `InstrumentedExecutor`：透明包装 ToolExecutor，自动持久化每次工具调用
- W&B 离线集成（无需登录，可后续 sync 到云端）
- Agent → Executor → DuckDB + W&B 全链路自动记录
- **5 张图** + DuckDB 65 行实验数据

### Day 8 — CalibrationAdvisor（自主标定顾问）⭐
- **三阶段自主推理链**: monitor → diagnose → act
- **MONITOR**：按时间点采集 health + qubit properties + 运行 probe circuit
- **DIAGNOSE**：
  - 线性回归检测 T1/T2/drift 趋势
  - 变点检测（归一化跳变检测器）
  - Qubit 质量排名（T1×0.4 + T2×0.3 + (1-readout)×0.3 复合分）
  - Fidelity 趋势分析
- **ACT**：6 种操作建议
  - `RECALIBRATE`（P1）— 变点 / 高漂移 / T1 快速衰减
  - `REMAP_LAYOUT`（P1-P2）— 低 fidelity，推荐更优 qubit
  - `APPLY_MITIGATION`（P2）— fidelity 下降趋势
  - `EXCLUDE_QUBITS`（P2）— 持续差的 qubit
  - `WAIT`（P3）— 漂移轻微
  - `OK`（P5）— 设备正常
- 完整 `AdvisoryReport` 输出（severity + actions + summary）
- **6 张图** + 25 个单测

### Day 9 — StreamingAdvisor（实时流式告警）⭐
- `StreamingAdvisor`：PropertiesStream → HealthTracker → Alert Engine → Callbacks
- **HealthTracker**：滑动窗口，实时计算 T1 slope / drift slope / drift jump
- **7 种告警类型**（带冷却防风暴）：
  - `DRIFT_SPIKE` — 漂移突跳
  - `DRIFT_HIGH` — 超阈值
  - `T1_DEGRADING` — 趋势衰减
  - `T1_LOW` — 绝对值过低
  - `ERROR_HIGH` — 2Q error 超阈值
  - `RECOVERY` — 从 critical 恢复
- **两种运行模式**：batch（同步）+ streaming（后台线程 + 时间加速）
- 与 Day 8 CalibrationAdvisor 共享 DuckDB，支持无缝接力
- **5 张图** + 21 个单测

---

## 4. 核心亮点（适合向导师展示）

### 4.1 完整的闭环能力
```
感知 → 推理 → 行动 → 验证
 │        │       │       │
 │  趋势分析   生成建议   probe circuit
 │  变点检测   优先排序   fidelity 验证
 ▼
PropertiesStream → HealthTracker → AlertEngine → AdvisoryReport
```
这不是单次工具调用，而是**多步自主推理链**，advisor 自己收集数据、分析趋势、输出决策。

### 4.2 漂移场景验证

| 场景 | Advisor 输出 | 正确性 |
|------|-------------|--------|
| STABLE（无漂移） | severity=nominal, OK | ✅ 正确无告警 |
| LINEAR_DECAY（渐进退化） | 检测到 T1 以 -7.01μs/h 衰减，预警"9h后不可用" | ✅ 定量准确 |
| SUDDEN_DEGRADATION（突变） | **精确捕获 t=5h 变点**，severity=critical，5条建议 | ✅ 时间精确 |
| DIURNAL_CYCLE（周期性） | 误报变点（已知局限，需 FFT） | ⚠️ 已识别 |

### 4.3 实时告警演示
24 小时模拟在 3 秒内完成（28800x 时间加速），实时输出：
```
t=1.8h  ⚠️  T1 degrading at -7.01 μs/h
t=5.7h  ⚠️  Drift 0.170 > 0.15 warning
t=17.1h 🚨 Drift 0.512 > 0.5 critical
t=19.9h ⚠️  T1 = 94.5μs < 100μs
t=22.7h ⚠️  2Q error 0.031 > 0.03
```

### 4.4 恢复检测
```
t=0-2.5h   健康（无告警）
t=3.0h     🚨 Drift 突跳 +1.000, T1 降到 70μs
t=3-6h     持续 critical
t=6.5h     ℹ️  RECOVERY: drift=0.000, T1=233.7μs
```
系统不仅检测退化，也能检测恢复——这对实际运营很重要。

---

## 5. 代码质量

```
模块           代码行    测试数    覆盖
───────────────────────────────────────
backends/      1,936      30      FakeAdapter + Replay + SyntheticDrift + PropertiesStream
tools/           338       —      5 个工具定义 + ToolExecutor
agent/           656       —      QuantumAgent (Anthropic/OpenAI/mock)
data/            439       —      DuckDB DataStore + InstrumentedExecutor
advisor/       1,152      46      CalibrationAdvisor + StreamingAdvisor
bench/           277       7      手写 + MQTBench 电路
tests/         1,054      83      全部通过
demos/         2,609       —      9天 × (demo + figures + summary)
───────────────────────────────────────
总计           8,618      83
```

---

## 6. 与计划对照

| 计划 Day | 计划内容 | 实际完成 | 状态 |
|----------|---------|---------|------|
| Day 1 | 建仓 + 装环境 | ✅ 完成 | ✅ |
| Day 2 | FakeBrisbane.run(GHZ_5) 出 fidelity | ✅ + CalibrationSnapshot + ReplayBackend | ✅ 超额 |
| Day 3 | ShadowBackend 抽象 + FakeBackendAdapter | ✅ + PropertiesStream + Benchmark 电路 | ✅ 超额 |
| Day 4 | ReplayBackend 单测 | ✅ SyntheticDriftBackend 重写 + API 对齐 | ✅ 超额 |
| Day 5 | MQTBench 5 个电路 baseline | ✅ 完成 | ✅ |
| Day 6 | Agent loop + 3 工具 | ✅ Agent + 5 工具 + 多 LLM 后端 | ✅ 超额 |
| Day 7 | W&B + DuckDB | ✅ 完成 | ✅ |
| Day 8 | SyntheticDriftBackend（已提前完成）→ CalibrationAdvisor | ✅ 三阶段自主推理链 | ✅ 超额 |
| Day 9 | PropertiesStream（已提前完成）→ StreamingAdvisor | ✅ 实时告警引擎 | ✅ 超额 |

**结论**: Phase 0 的 Day 1-9 均已完成，且多数超额交付。原计划中 Day 8-9 的基础任务（SyntheticDriftBackend, PropertiesStream）在前几天已提前完成，因此 Day 8-9 推进到了更高级的内容。

---

## 7. 下一步 (Day 10-14)

| Day | 任务 | 目标 |
|-----|------|------|
| Day 10 | 安装 Qiskit Dynamics，跑 Rabi 振荡 tutorial | 脉冲级仿真基础 |
| Day 11 | 写 BACKENDS.md，CI 覆盖 3 种 shadow backend | 文档 + CI |
| Day 12 | MVP CLI: `qgpt simulate ghz --shots 8192 --backend FakeBrisbane` | 一行命令出结果 |
| Day 13 | ExperimentRecord schema（DuckDB + dataclass） | 结构化实验记忆 |
| Day 14 | 缓冲 + 发物理组合作意向邮件 | Phase 0 收尾 |

**Phase 0 → Phase 1 过渡**: Day 10-14 完成后，进入 Phase 1（W3-W4），重点攻关 drift detector 评测（PELT / Bayesian changepoint detection, 100 个人工标注漂移点, F1 评测）。

---

## 8. 可运行的 Demo 命令

```bash
cd ~/quantumgpt

# 全量测试（~90秒）
.venv/bin/python -m pytest tests/ -v

# Day 8: CalibrationAdvisor（4场景，自主推理链）
.venv/bin/python demos/day8/run_day8_demo.py

# Day 9: StreamingAdvisor（实时告警，3秒模拟24小时）
.venv/bin/python demos/day9/run_day9_demo.py

# 重新生成所有可视化
.venv/bin/python demos/day8/gen_figures.py
.venv/bin/python demos/day9/gen_figures.py
```

---

*Report generated: 2026-05-16*
*Repository: /home/wangshuchang/quantumgpt/*
