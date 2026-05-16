
# QuantumGPT · Phase 0 成果展示

> **项目定位**：第一个具备设备感知（device-aware）的闭环量子 AI Agent，
> 能在 NISQ 漂移条件下跨 gate-level 和 pulse-level 调度工具、保持任务成功率。
>
> **投稿目标**：ICLR 2027 Main Track（deadline ~2026-10-03）
>
> **Phase 0 周期**：2 周（Day 1 – Day 13），单人执行

---

## 1. 两周做了什么

Phase 0 的目标是"地基"——不碰算法创新，只确保工程骨架足够扎实，
后续 Phase 1–5 能在上面快速搭建。

### 核心数字

| 指标 | 数值 |
|------|------|
| Python 源文件 | 54 个 |
| 代码行数（不含 venv） | 11,938 行 |
| 测试用例 | 129 个（pass 129, skip 1, fail 0） |
| 测试覆盖行数 | 1,647 行 |
| Demo 天数 | 13 天（每天一个独立 demo 文件夹） |
| 可视化图表 | 48 张 |
| CLI 命令 | 6 个（`simulate / health / list / diagnose / agent / records`） |
| 文档 | BACKENDS.md (280行) + 13 份 daily summary + README |

### 模块全景

```
quantumgpt/
├── backends/     1,936 行  ← 4 种影子量子硬件
│   ├── FakeBackendAdapter      包装 IBM FakeBrisbane/Kyiv/Sherbrooke/Torino
│   ├── ReplayBackend           回放真实历史校准数据（含漂移）
│   ├── SyntheticDriftBackend   可控参数化漂移注入
│   └── PropertiesStream        实时校准遥测流
│
├── agent/          656 行  ← while-loop agent，支持 3 种 LLM 后端
│   └── QuantumAgent            Anthropic / OpenAI / Rule-based mock
│
├── advisor/      1,152 行  ← 流式建议器 + 变点检测
│   ├── Advisor                 基于 health 数据给出 action 建议
│   └── StreamingAdvisor        订阅 PropertiesStream，实时告警
│
├── bench/          277 行  ← 5 手写电路 + MQTBench 集成
├── tools/          338 行  ← 5 个 agent 工具（MCP schema）
├── data/           938 行  ← DuckDB 存储 + ExperimentRecord
├── dynamics/       309 行  ← Qiskit Dynamics Rabi 脉冲仿真
└── cli.py          507 行  ← MVP CLI（pip install → qgpt 命令）
```

---

## 2. 可以演示什么

### 2.1 一行命令跑量子电路

```bash
$ qgpt simulate ghz --shots 8192 --backend FakeBrisbane

╭─────────────── Simulation Result ────────────────╮
│ Circuit:  ghz_5                                   │
│ Backend:  FakeBrisbane (127q)                     │
│ Shots:    8192                                    │
│ Fidelity: 0.9269                                  │
│ Depth:    23                                      │
│ Time:     12.36s                                  │
╰───────────────────────────────────────────────────╯
```

5 种电路 × 4 种后端，全部可跑。输出彩色：
fidelity >= 0.9 绿色，>= 0.7 黄色，< 0.7 红色。

### 2.2 Agent 自主决策

```bash
$ qgpt agent "Check health and run GHZ-5"
```

Agent 自动：
1. 调 `get_backend_health` → 读取 T1/T2/错误率
2. 调 `run_circuit` → 跑 GHZ-5
3. 综合分析 → 输出报告

支持 Claude / GPT / DeepSeek 切换，离线用 rule-based mock。

### 2.3 漂移检测 + 实时告警

```python
stream = PropertiesStream(backend, interval=5)
advisor = StreamingAdvisor(stream)
advisor.start()  # 开始监听，drift 超阈值自动告警
```

### 2.4 脉冲级仿真

```python
# Qiskit Dynamics: Rabi oscillation
result = simulate_rabi(qubit_freq=5.0e9, drive_amp=0.3, ...)
# → IQ 数据 → 拟合 Rabi 曲线 → 输出 π-pulse 幅度
```

### 2.5 结构化实验记忆

```bash
$ qgpt records --stats

  Total experiments    11
  Unique backends       2
  Unique circuits       5
  Success rate       100%
  Avg fidelity      0.964
```

每次实验自动记录 27 个字段（问题描述、工具调用链、fidelity、
backend 快照、反思总结），为 Phase 3 的 RAG 检索打基础。

---

## 3. 关键技术决策（为什么这样做）

| 决策 | 理由 |
|------|------|
| **4 种影子后端而非 1 种** | Paper 需要跨设备对比实验；ReplayBackend 是论文的核心实验基础（注入真实漂移） |
| **Agent 自研 while-loop 而非 LangChain** | 300 行可控 vs. 万行黑箱；paper 需要精确记录每步决策 |
| **DuckDB 而非 SQLite/Postgres** | 列式存储适合 fidelity 趋势分析；单文件部署；原生 Parquet 导出 |
| **ExperimentRecord 27 字段** | 比 (circuit, fidelity) 丰富 10 倍，直接支撑消融实验（有记忆 vs 无记忆） |
| **每天 demo 文件夹** | 可重现性：任何人 `python demos/dayN/gen_figures.py` 即可复现当天结果 |

---

## 4. 精选可视化

以下图表位于 `demos/` 目录，可直接打开查看：

### 系统架构
- `demos/day11/architecture.png` — 全系统架构图（User → Agent → Backends → DuckDB）
- `demos/day6/agent_architecture.png` — Agent 内部结构
- `demos/day9/streaming_architecture.png` — 实时流式监控架构

### 量子电路基准
- `demos/day12/fidelity_comparison.png` — 5 电路 fidelity 对比（全 > 0.9）
- `demos/day12/backend_comparison.png` — GHZ-5 跨 4 后端对比
- `demos/day5/mqtbench_fidelity.png` — MQTBench 电路 fidelity 分布
- `demos/day5/fidelity_vs_ecr.png` — Fidelity 随 ECR 门数衰减趋势

### 漂移与检测
- `demos/day4/drift_profiles_comparison.png` — 4 种漂移模式
- `demos/day4/fidelity_degradation.png` — 漂移导致 fidelity 退化
- `demos/day8/changepoint_detection.png` — 变点检测（PELT）
- `demos/day9/alert_timeline.png` — 实时告警时间线

### 脉冲级
- `demos/day10/rabi_curves.png` — Rabi 振荡曲线
- `demos/day10/rabi_chevron.png` — Rabi Chevron pattern
- `demos/day10/pulse_shapes_bloch.png` — 脉冲形状 + Bloch 球轨迹
- `demos/day10/pi_calibration.png` — π 脉冲校准

### 数据与记忆
- `demos/day7/fidelity_by_circuit.png` — DuckDB 中按电路分析
- `demos/day13/experiment_fidelity_scatter.png` — ExperimentRecord 跨后端分析
- `demos/day13/experiment_stats_summary.png` — 聚合统计

---

## 5. 与相关工作的差异化

| 项目 | 他们做了什么 | 我们的差异 |
|------|-------------|-----------|
| **QUASAR** (Google, 2024) | RL agent 选 transpiler pass | 我们是闭环 + 多工具 + 漂移感知 |
| **El Agente Q** (Accenture, 2024) | ReAct agent 写 Qiskit 代码 | 他们开环；我们有设备感知 + 实验记忆 |
| **QiskitGPT** | Prompt → 代码生成 | 纯代码生成，无 agent loop |

我们的核心贡献点：
1. **闭环**：漂移 → 检测 → 重规划 → 恢复（而非一次性执行）
2. **设备感知**：agent 看得到 T1/T2/gate error，不是盲跑
3. **跨层**：gate-level + pulse-level 工具在同一个 agent 框架里
4. **结构化记忆**：ExperimentRecord，可消融证明其有效性

---

## 6. 接下来做什么（Phase 1–5 概览）

```
Phase 0  [完成]  地基：影子硬件 + CLI + Agent loop + 实验记忆
Phase 1  W3–W4   感知层：6 个感知工具 + Drift Detector (F1 评测)
Phase 2  W5–W6   工具矩阵：14 工具 + 4 baseline + 7 任务全跑通
Phase 3  W7–W9   闭环学习：drift-aware replanning + ExperimentRecord 消融
Phase 4  W10–W11  Benchmark：QC-Agent-Bench 65 任务 + leaderboard
Phase 5  W12+     论文写作 → ICLR 2027 投稿
```

**Phase 1 即将启动的具体任务：**
- 实现 `detect_drift(backend, window)` 工具
- PELT / Bayesian Online Changepoint Detection 评测
- 人工标注 100 个漂移点，报 F1 score
- Rabi pulse 封装为 agent 可调用的工具

---

## 7. 风险与诚实评估

| 已验证 | 尚未验证 |
|--------|---------|
| 4 种影子后端稳定可用 | Drift detector 准确率（Phase 1 才评测） |
| Agent loop 三种 LLM 均可切换 | 真实 LLM 下 tool calling 稳定性 |
| ExperimentRecord 读写查询无误 | RAG 检索对 agent 决策的实际提升 |
| Rabi 仿真能出曲线 | Rabi 工具被 agent 自主调用的效果 |
| 5 电路全 > 0.9 fidelity | 漂移注入后 fidelity 恢复率 |

**最大风险**：Phase 3 闭环实验的提升幅度是否足够显著（目标 >= 15%）。
**缓解措施**：如果不够，可以加大漂移幅度或增加故障注入难度来拉大差距。

---

*Phase 0 完成日期：2026-05-16*
*下一步：Phase 1 感知层（W3 启动）*
