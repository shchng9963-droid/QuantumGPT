# Quantum GPT · 12 周融合版研究计划

> **策略**：以 v1 为执行骨架，从 v2 中选取 3 个高杠杆增量模块叠加，
> paper 故事定位为 **"device-aware closed-loop agent"** —— 比纯 v1 有差异化，
> 又不需要 v2 的全量工程。
>
> **核心原则**：每个 Phase 都有可发表的 checkpoint；宁可砍 scope 也不允许延期。

---

## 从 v2 中选取的 3 个增量模块

| # | 模块 | 为什么选 | 预计额外工作量 |
|---|---|---|---|
| 1 | **Rabi 端到端工具**（rabi_experiment + fit_rabi） | 最小成本展示 "跨层能力"，demo 价值极大 | 3–5 天 |
| 2 | **ExperimentRecord schema** | 替换简单 episodic memory，paper 故事升级，代码改动小 | 2–3 天 |
| 3 | **物理组合作启动**（邮件 + 访谈 + 历史数据获取） | 零工程成本的异步动作，成了是 bonus，不成不影响主线 | 累计 3–4 小时 |

**明确不做的 v2 模块**（推到后续版本）：
- 35 个工具全量实现（只做 14–16 个）
- cross-device transfer
- Active Learning experiment designer（botorch/GPyOpt）
- GST / CRAB / C3-Toolset / Pulser
- 完整 Phase L 真机部署
- 安全护栏 dry-run / power_budget / 二级确认（仿真阶段只做 audit_log）

---

## Paper 故事（W1 就要想清楚）

### 定位（一句话）
> **第一个具备设备感知（device-aware）的闭环量子 agent，能在 NISQ 漂移条件下
> 跨 gate-level 和 pulse-level 调度工具、保持任务成功率，配套发布 QC-Agent-Bench。**

### 标题候选
*Quantum GPT: A Device-Aware Closed-Loop Agent for NISQ Workflows under Drift*

### 卖点分层
1. **Closed-loop**：drift-aware replanning（vs QUASAR / El Agente Q 的开环）
2. **Device-aware**：agent 能感知设备层参数并调用 pulse-level 工具（哪怕只有 1–2 个）
3. **ExperimentRecord**：结构化实验记忆 + 消融证明其有效性
4. **Benchmark**：QC-Agent-Bench 60–70 任务，含漂移 / 故障 tier

### 投稿目标（主线先锁一个）
- **首选**：ICLR 2027 Main（deadline ~2026-10 初）→ 倒推 W12 = 9 月底必须有完整实验
- **备选**：NeurIPS 2026 Datasets & Benchmarks（如果 deadline 6 月中还来得及）
- **Lab Service paper**：视物理组合作进展决定，不列为必须产出

---

## 投稿 Deadline 倒推时间线

假设锁定 ICLR 2027（deadline ~2026-10-03）：

```
W1  = 2026-05-18    开工
W12 = 2026-08-07    代码 + 实验 freeze
W13–W15 = 写作 + 内审 + 投稿（8 月中 → 10 月初）
```

留了 ~8 周写作缓冲，比较健康。如果 NeurIPS D&B 赶得上（deadline ~6 月中），
可以先把 benchmark 部分单独投出去占位。

---

## 软件栈（与 v1 一致 + 最小增量）

| 层 | 选择 | 备注 |
|---|---|---|
| Agent runtime | LangGraph / 自研 Python loop | v1 |
| LLM | Claude / GPT-4o 主力 + DeepSeek-V3 / Qwen2.5 本地对照 | v1 |
| Inference 加速 | vLLM 0.6+ | v1 |
| 量子 SDK · 门级 | Qiskit 1.x + PennyLane（化学）+ Cirq（对照） | v1 |
| **量子 SDK · 脉冲级** | **Qiskit Dynamics**（仅此一个，不上 QuTiP/C3） | **v2 增量，最小化** |
| 噪声模拟 | Qiskit Aer + Mitiq + Stim | v1 |
| 工具协议 | MCP | v1 |
| 数据存储 | DuckDB（校准时序）+ Qdrant（电路 embedding RAG） | v1 |
| **实验记录** | **ExperimentRecord → DuckDB + Parquet** | **v2 增量** |
| 实验跟踪 | W&B | v1 |
| 可视化 | Plotly + Operator Console | v1 |

> 脉冲级只用 Qiskit Dynamics，不同时上 QuTiP-QIP / C3 / Pulser。
> 一个库跑通 Rabi 就够支撑 paper 故事。

---

## 评测体系（W1 定下来）

```python
baselines = {
    "static":        "固定 pipeline（transpile -> submit -> mitigation）",
    "llm-single":    "单次 LLM 写 Qiskit 代码",
    "react-base":    "ReAct + tools（无 memory / 无 drift-aware）",
    "qgpt":          "Quantum GPT 完整版",
    "qgpt-no-mem":   "消融 1: 去掉 ExperimentRecord",
    "qgpt-no-drift": "消融 2: 去掉 drift-aware replanning",
}

task_tiers = {
    "tier1_static":  30,  # 固定 backend snapshot
    "tier2_drift":   20,  # ReplayBackend 真实漂移
    "tier3_failure": 10,  # 注入异常（qubit 挂掉等）
    "tier4_device":   5,  # Rabi tune-up 端到端（v2 增量，只做 5 个）
    # 总计 65 个任务
}

metrics = {
    "success_rate":   "任务结果正确的比例",
    "avg_fidelity":   "端到端保真度",
    "tool_calls":     "平均工具调用数（越少越好）",
    "wall_time":      "端到端延迟",
    "drift_recovery": "漂移注入后 N 步内恢复率",
    "cost":           "token + simulator 时间成本",
}
```

**操作规范**
- 每周跑一次全套 → W&B 看趋势
- 每周一固定时间发 progress note（给自己 / 导师）
- 任何代码改动后确保 >= 1 个 baseline 还能跑通

---

## 12 周分阶段计划

### Phase 0 · 地基（W1–W2）

**目标**：影子硬件 + 评测脚手架 ready + Qiskit Dynamics 环境就绪

**W1 具体任务（按天）**

| Day | 任务 | 完成标志 |
|---|---|---|
| Day 1 | 建仓 `quantum-gpt/`（monorepo: `backends/` `tools/` `agent/` `bench/` `eval/`）；装环境 Qiskit 1.x + Aer + Mitiq | `pip list` 无报错 |
| Day 2 | 跑通 `FakeBrisbane.run(transpile(GHZ_5))`，把 fidelity 打出来 | 终端输出 fidelity 数值 |
| Day 3 | 写 `ShadowBackend` 抽象 + `FakeBackendAdapter` 子类 | 单测通过 |
| Day 4 | 写 `ReplayBackend`（接受时间戳，从历史 snapshot 取 properties） | 单测通过 |
| Day 5 | 装 MQTBench，挑 5 个门级任务能跑 baseline | 5 个电路出 fidelity |
| Day 6 | 接 Claude/GPT 做最朴素的 agent loop（while 循环 + 3 个工具） | "prompt → 调工具 → 出结果"跑通 |
| Day 7 | W&B + DuckDB 接好 | 实验结果能在 W&B dashboard 看到 |

**W2 具体任务**

| Day | 任务 | 完成标志 |
|---|---|---|
| Day 8 | `SyntheticDriftBackend`（参数化 T1/T2 漂移函数） | 单测通过 |
| Day 9 | `properties_stream.py`：每 N 秒 emit 一次 calibration snapshot | mock telemetry 能跑 |
| Day 10 | **装 Qiskit Dynamics**，跑 tutorial notebook（不急着集成） | Rabi tutorial 能跑通 |
| Day 11 | 写 `BACKENDS.md`，CI 测试覆盖 3 种影子后端 | CI 绿灯 |
| Day 12 | **MVP CLI**：`qgpt simulate ghz --shots 8192 --backend FakeBrisbane` | 一行命令出 fidelity |
| Day 13 | 写 ExperimentRecord schema（DuckDB 表 + Python dataclass） | schema 定义完，能写入读出 |
| Day 14 | 缓冲 / 补债 + **发出 1–2 封物理组合作意向邮件** | 邮件已发 |

**Phase 0 Deliverables**
- [ ] `BACKENDS.md`
- [ ] CI 测试覆盖 3 种影子后端
- [ ] MVP CLI 可演示
- [ ] ExperimentRecord schema 定义
- [ ] Qiskit Dynamics 环境就绪（tutorial 跑通）
- [ ] 物理组邮件已发

**成功判据**：5 分钟内跑通 "prompt → simulator → fidelity"，结果落 W&B + DuckDB。

---

### Phase 1 · 感知层（W3–W4）

**目标**：硬件感知工具做扎实 + drift detector 有评测数据

**具体任务**

1. 实现 6 个核心感知工具（暴露给 agent）：
   - `get_backend_health(backend)` → fidelity / queue / age
   - `get_qubit_properties(backend, qubits)` → T1/T2 / readout / gate err
   - `get_coupling_map(backend)`
   - `detect_drift(backend, window)` → drift score + 是否重校建议
   - `get_calibration_age(backend)`
   - `compare_backends(problem, candidates)` → 多后端排名

2. **Drift Detector 重点攻关**：
   - 用 ReplayBackend 喂真实 IBM 一周历史数据
   - 实现 PELT / Bayesian Online Changepoint Detection
   - 人工标注 100 个漂移点，评测 F1

3. 前端 Operator Console 接通真实数据流（dashboard 变实的）

4. **（v2 增量，穿插做）**：跑通 Qiskit Dynamics 的 Rabi pulse 仿真
   - 不急着封装成工具，先在 notebook 里跑通 "定义 Hamiltonian → 发 pulse → 读 IQ → 拟合 Rabi 曲线"
   - 这为 Phase 3 的 rabi_experiment 工具打基础

**Deliverables**
- [ ] 6 个感知工具 + 单测
- [ ] Drift detector 准确率报告（F1 on 100 labeled points）
- [ ] Console 能看到一周历史漂移回放
- [ ] Notebook：Qiskit Dynamics Rabi pulse 端到端跑通

**成功判据**：drift detector 在 100 个标注点上 F1 >= 0.7；Rabi notebook 出曲线。

---

### Phase 2 · 工具矩阵 + Agent 框架（W5–W6）

**目标**：agent 框架就位，14 个工具跑通，baseline 有数据

**具体任务**

1. 用 MCP 协议封装 14 个工具：
   - 感知层 6 个（Phase 1 已做）
   - `transpile_circuit(circ, backend, opt_level, noise_aware)`
   - `simulate(circ, backend, shots, noise_model)`
   - `apply_mitigation(circ, method)` — Mitiq ZNE / PEC / CDR
   - `predict_fidelity(circ, backend)` — lookup table 起步
   - `retrieve_similar_circuits(circ, top_k)` — Qdrant 向量库
   - `decompose_problem(prompt)` — 自然语言拆子任务
   - **`rabi_experiment(backend, qubit, amp_range)`** — v2 增量
   - **`fit_rabi(traces)`** — v2 增量

2. Agent loop 用 LangGraph ReAct 模板 + 自定义 fidelity-budget 节点

3. 双 LLM 模式：Claude/GPT 跑通逻辑 + vLLM 本地（Qwen2.5-32B 起步）

4. 实现 3 个对照 baseline：
   - **Static Pipeline**：���定流程
   - **LLM Single Shot**：一次性写 Qiskit 代码
   - **ReAct + Tools**：标准 ReAct（弱基线）

5. 跑 7 个端到端任务的全部 baseline：
   - 门级 5 个：GHZ / VQE H2 / QAOA MaxCut-6 / Bernstein-Vazirani / QFT
   - 设备级 2 个：Rabi tune-up（2 个不同 qubit 初始条件）

**Deliverables**
- [ ] 14 个工具的 MCP schema 文档
- [ ] 7 个端到端任务在 4 个系统上全跑通
- [ ] Baseline 性能基线表（fidelity / latency / success rate）
- [ ] Rabi 端到端 demo（prompt → pulse sim → IQ → fit → 报告 π-amp）

**成功判据**：7 个任务 × 4 个系统全出数据，跑一次全套 < 30 分钟。

---

### Phase 3 · 闭环 & 学习（W7–W9） ← 核心差异化阶段

**目标**：让 agent 从结果中学习 + 漂移下自适应 —— 这是拉开差距的地方

**具体任务**

1. **ExperimentRecord 落地**（v2 增量，替代 v1 的简单 episodic memory）
   - 每次任务存 `(problem, plan, tool_calls, fidelity, success, raw_data_ref, fits, decisions)`
   - 接入 Qdrant 做 RAG 检索
   - **消融实验**：开/关 ExperimentRecord，对比 50 条任务的成功率

2. **Fidelity-aware Planner**
   - 简单版：每次 tool call 前调 `predict_fidelity`，低于阈值走 fallback
   - 进阶版：训小 GNN（输入电路 + backend properties → fidelity）
   - 训练数据 = Phase 2 跑出来的所有 (circ, backend, real_fidelity)

3. **Drift-aware Replanning**
   - 监听 properties_stream，drift detector 报警时主动 invalidate transpile 结果，重做
   - **这是 paper 的核心实验**：注入漂移 → 看 agent 多快恢复 → 对比 baseline 完全不恢复

4. **（如果进度好）Stretch**：
   - 小规模 agentic RL 实验（参考 QUASAR），用 real_fidelity 作 reward 微调本地 7B
   - 如果做不了，跳过，不影响主线

**Deliverables**
- [ ] ExperimentRecord 消融报告（memory ON vs OFF 的成功率差异）
- [ ] Fidelity GNN 训练曲线 / R² 评估（或 lookup table 的 baseline 结果）
- [ ] Drift-aware replanning demo（含视频 — paper 投稿要附）
- [ ] **Paper 主图初稿**：Quantum GPT vs 3 baseline 在 7 个任务上的全面对比

**成功判据**：full Quantum GPT 在 fidelity / success rate 上较 ReAct baseline 提升 >= 15%。

---

### Phase 4 · Benchmark 化（W10–W11）

**目标**：把成果产品化为可发布的 benchmark —— 这是长期影响力的来源

**具体任务**

1. 设计 **QC-Agent-Bench**：
   - **Tier 1 · 静态**：30 个任务，固定 backend snapshot
   - **Tier 2 · 漂移**：20 个任务，ReplayBackend
   - **Tier 3 · 故障**：10 个任务，注入异常
   - **Tier 4 · 设备级**：5 个任务，Rabi tune-up（v2 增量，小规模）

2. 评测维度：success_rate / fidelity / tool_calls / wall_time / drift_recovery / cost

3. 6 个系统的完整数据（3 baseline + qgpt + 2 消融）

4. 写 dataset card / README / ground truth

5. GitHub 仓库 + leaderboard 雏形网页

6. **如果物理组合作有进展**：
   - 用他们的历史数据做 replay 评测
   - 产出 "如果当时用了 Quantum GPT 能省多少时间" 对比报告
   - 为 Lab Service paper 积累素材

**Deliverables**
- [ ] QC-Agent-Bench spec（65 个任务 + ground truth）
- [ ] GitHub 仓库（公开）
- [ ] 6 个系统的完整对比结果表
- [ ] Leaderboard 网页
- [ ] （可选）物理组 replay 评测报告

**成功判据**：一句话差异化 —— "第一个评测 quantum-LLM-agent 在漂移、故障、及设备级任务下韧性的标准集"。

---

### Phase 5 · 论文（W12 + 后续）

**W12 写作启动，后续持续打磨到投稿**

#### 主线 paper
- **Target**：ICLR 2027 Main Track（~2026-10-03 deadline）
- **Title**：*Quantum GPT: A Device-Aware Closed-Loop Agent for NISQ Workflows under Drift*
- **结构**：
  - Intro：NISQ 漂移 + 现有工作开环 + 我们闭环
  - Method：三层架构（Perception / Cognition / Execution）+ ExperimentRecord + drift-aware replanning
  - Experiments：QC-Agent-Bench 65 任务 × 6 系统 × 6 metrics
  - Ablation：去 memory / 去 drift-aware 各掉多少
  - Discussion：sim-to-real gap 诚实说明 + 真机迁移路径
- **主图**：闭环架构 + 具体 trace 示例（含 Rabi 跨层调用）
- **主表**：6 系统 × 4 tier × 6 metric

#### Benchmark paper（可选，看进度）
- **Target**：NeurIPS 2026 D&B（如果 deadline 赶得上）或 ICLR 2027 同期
- **独立贡献**：QC-Agent-Bench 定义评估范式

#### Lab Service paper（视合作进展）
- 如果物理组合作在 W10 前有实质数据 → 启动
- 否则 → 推到 v2 后续，不影响主线

---

## 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| Qiskit Dynamics 上手慢 | 中 | W2 就装环境跑 tutorial，不等到 Phase 2 才碰；最坏退化为只用 Aer（去掉 Rabi 工具，paper 只讲门级） |
| LLM tool calling 不稳定 | 中 | 双 LLM + MCP schema 严格 + Claude 兜底 |
| Drift detector F1 < 0.7 | 低 | 调参 / 换算法（PELT → BOCPD）；0.6 也能发，paper 里讨论 |
| Fidelity GNN 训不出来 | 中 | 退化为 lookup table，不卡主线 |
| Benchmark 被抢发 | 低 | Phase 4 完成立即挂 arXiv + GitHub |
| 物理组合作落空 | 高 | 不影响主线 paper；Plan B = 虚拟实验室（已在主计划中） |
| 真机最终拿不到 | 高 | 90% 工作不依赖真机；ReplayBackend 就是论证 sim-to-real 的方式 |
| 12 周做不完 | 中 | Phase 4 是最容易裁的 → 先发主线 paper，benchmark 留后续 |
| Rabi 工具做不好 | 中 | 退化为"只在 paper 中展示 notebook demo"，不影响主线评测 |

---

## 每周 Checkpoint 与 Red Line

| 周 | 必须完成 | Red Line（未完成则调整计划） |
|---|---|---|
| W2 | MVP CLI 出 fidelity + Dynamics 环境就绪 | 如果 MVP 未跑通 → W3 不开感知层，继续修地基 |
| W4 | 6 个感知工具 + drift detector F1 报告 | 如果 F1 < 0.5 → 砍 Rabi 工具时间，集中修 drift |
| W6 | 14 个工具 + 7 任务 × 4 系统有数据 | 如果 Rabi 跑不通 → 砍掉，回到 v1 的 12 工具方案 |
| W9 | 主图初稿 + 消融数据 + drift demo | 如果提升 < 10% → 检查 agent loop 逻辑 |
| W11 | Benchmark 65 任务 + 完整对比表 | 如果 Benchmark 来不及 → 先写主线 paper |
| W12 | Paper 写作启动 | 如果数据不够 → 缩小 scope 但必须开始写 |

---

## 第一周行动清单（精确到天）

| Day | 上午 | 下午 |
|---|---|---|
| Mon | 建仓 `quantum-gpt/`，写 README + `.gitignore` + CI 骨架 | 装 Qiskit 1.x + Aer + Mitiq + MQTBench |
| Tue | 跑通 `FakeBrisbane.run(transpile(GHZ_5))` | 写 `ShadowBackend` 抽象类 |
| Wed | 写 `FakeBackendAdapter` + 单测 | 写 `ReplayBackend` + 单测 |
| Thu | 装 MQTBench，挑 5 个电路跑出 fidelity | 开始接 Claude API 做最简 agent loop |
| Fri | Agent loop 跑通（prompt → 调 1 个工具 → 出结果） | 接 W&B，实验结果上 dashboard |
| Sat | 接 DuckDB，所有 calibration/job/result 落库 | 缓冲 / 补债 |
| Sun | **写 1–2 封物理组合作意向邮件** | 回顾本周进度，写 weekly note |

**W1 完成判据**：
- FakeBrisbane GHZ 出 fidelity ✓
- ShadowBackend 抽象 + 2 个子类有单测 ✓
- 最简 agent loop 跑通 ✓
- W&B + DuckDB 接好 ✓
- 物理组邮件已发 ✓

---

## 与 v1 / v2 的对比

| 维度 | v1 | v2 | 融合版 |
|---|---|---|---|
| 工具数量 | 12 | 35 | **14**（12 + Rabi + fit_rabi） |
| Backend 层级 | 门级 | 门+脉冲+设备 | **门级主力 + 脉冲级 demo** |
| 记忆系统 | (circ, fidelity) | ExperimentRecord | **ExperimentRecord** |
| Benchmark 任务 | 60 | 90 | **65**（60 + 5 Rabi） |
| Baseline 数量 | 6 | 9 | **6** |
| 投稿目标 | 2 篇 | 3 篇 | **1 篇主力 + 1 篇可选** |
| 物理组合作 | 无 | Phase L 全量 | **发邮件 + 拿历史数据（如果成了）** |
| 安全机制 | 无 | 6 项 | **audit_log（仅 1 项）** |
| 12 周可行性 | 高 | 低 | **高** |

---

## 文件说明

- `QuantumGPT_Plan.md` — v1 原版（保留作为参考）
- `QuantumGPT_Plan_v2.md` — v2 愿景版（保留作为远期路线图）
- `QuantumGPT_Plan_merged.md` — **本文件，12 周执行计划**

---

*文档版本：merged v1.0 · 2026-05-15*
*下一步：确认本计划 → 启动 Day 1*
