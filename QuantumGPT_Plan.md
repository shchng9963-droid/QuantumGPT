# Quantum GPT · 12 周细节版研究计划

> **核心思路一句话**：用「影子硬件」(shadow hardware) 替代真机 —— 拿历史 IBM 校准数据 + Qiskit FakeBackend + 噪声模拟器，复刻一台「会漂移、有真实噪声画像、能给出真实保真度反馈」的虚拟 QPU。这条路线让 95% 的 Quantum GPT 工作都能推进，且当真机到位时改一个 backend adapter 就能切换。

---

## 目录

- [一、核心策略：为什么不需要真机](#一核心策略为什么不需要真机)
- [二、影子硬件（Shadow Hardware）架构](#二影子硬件shadow-hardware架构)
- [三、软件栈](#三软件栈)
- [四、数据集 / Benchmark 资源清单](#四数据集--benchmark-资源清单)
- [五、12 周分阶段计划](#五12-周分阶段计划)
- [六、评测体系](#六评测体系)
- [七、关键风险与对策](#七关键风险与对策)
- [八、未来真机到位时如何切换](#八未来真机到位时如何切换)
- [九、第一周（本周）行动清单](#九第一周本周行动清单)
- [十、两个核心建议](#十两个核心建议)
- [附录 A · 待办看板模板](#附录-a--待办看板模板)
- [附录 B · 论文故事板（draft）](#附录-b--论文故事板draft)

---

## 一、核心策略：为什么不需要真机

**关键认知**：你的所有研究价值不在「调通真机」，而在 **「agent 如何在不确定噪声下做决策」**。这件事在模拟器上做反而更可控、更可复现、更容易写 paper。

四个支撑点：

1. **Qiskit `FakeBackendV2` 系列**就是 IBM 真实设备的快照（FakeBrisbane / FakeKyiv / FakeSherbrooke / FakeWashington…），里面带真实的 coupling map、gate error、T1/T2、readout error。直接当真机用。
2. **历史校准数据可以「回放」** —— IBM 把每天的 backend properties 保留几个月，社区 Zenodo 上有按小时存档的多设备多月数据。把这些按时间戳喂进系统就是真实漂移。
3. **Mitiq + Stim + Qiskit Aer** 把 noise simulation / error mitigation / QEC 全栈打通，速度比真机还快。
4. **AlphaQubit 这种 Nature 级工作 ≥80% 的实验都在模拟器上做** —— 模拟器 + 真机数据训练 + 模拟器验证是标准做法。

> 💡 **指导原则**：把「无真机」当成一次研究上的优势 —— 你能做对照实验、��入可控漂移、复现一切。这些是有真机的人做不到的。

---

## 二、影子硬件（Shadow Hardware）架构

把 Quantum GPT 的「执行层」换成下面这个堆栈，对 agent 完全透明：

```
┌─────────────────────────────────────────────────────────┐
│  Shadow Backend Pool                                    │
│  ├─ FakeBrisbane / FakeKyiv / FakeSherbrooke (快照)     │
│  ├─ Replay Backend (按时间戳回放历史 calibration)        │
│  └─ Synthetic Drift Backend (注入可控漂移 / 噪声谱)     │
├─────────────────────────────────────────────────────────┤
│  Noise Simulators                                       │
│  ├─ Qiskit Aer (gate-level)                             │
│  ├─ Stim (QEC, 极快)                                    │
│  └─ cuQuantum / TensorCircuit (大尺寸态)                │
├─────────────────────────────────────────────────────────┤
│  Properties Stream                                      │
│  └─ 每 N 秒 emit 一次 calibration snapshot (mock 遥测)  │
└─────────────────────────────────────────────────────────┘
```

**这一层的设计目标**：和未来真机的接口完全相同（IBM Runtime / Qiskit `BackendV2` 协议），切换只改一个 `BackendProvider`。

### 三种影子后端的语义

| Backend | 用途 | 数据源 | 适用阶段 |
|---|---|---|---|
| `FakeBackendAdapter` | 静态对照 / 单点评测 | Qiskit `fake_provider` | Phase 0–4 全程 |
| `ReplayBackend` | 真实漂移再现 | Zenodo IBM 历史校准 | Phase 1–4 |
| `SyntheticDriftBackend` | 可控压力测试 | 参数化漂移函数 | Phase 3–4 |

---

## 三、软件栈

| 层 | 选择 | 用途 |
|---|---|---|
| Agent runtime | LangGraph / 自研 Python loop | 多步 tool use |
| LLM | DeepSeek-V3 / Qwen2.5-72B（本地）+ Claude / GPT-4o（对照） | 推理核心 |
| Inference 加速 | vLLM 0.6+ | 本地部署 |
| 量子 SDK | **Qiskit 1.x** 主力 + PennyLane（化学）+ Cirq（对照） | 电路构造 / transpile |
| 噪声模拟 | Qiskit Aer + **Mitiq** + Stim | 误差缓解 + QEC |
| 工具协议 | **MCP (Model Context Protocol)** | 标准化 tool schema |
| 数据存储 | DuckDB（calibration 时序）+ Qdrant（电路 embedding RAG） | 持久化 |
| 实验跟踪 | Weights & Biases | 必须有，paper 要用 |
| 可视化 | Plotly + 已有的 Operator Console | 演示 / 调试 |

> ⚠ **决策性建议**：工具协议**优先 MCP**。这让你后期切换任意 LLM（Claude → GPT-4o → 本地 Qwen）零成本，且能与未来量子社区出现的标准对齐。

---

## 四、数据集 / Benchmark 资源清单

### 电路 Benchmark
- **MQTBench**（慕尼黑工大，约 8000 条电路，覆盖 2–130 qubits）— `pip install mqt.bench`
- **QASMBench**（PNNL，分 small/medium/large）— GitHub: `pnnl/QASMBench`
- **BenchPress**（IBM 编译器 benchmark）— GitHub: `Qiskit/benchpress`
- **HamLib**（量子化学 / 物理 Hamiltonian 库）— USC

### 校准数据
- **Qiskit FakeBackend**（自带）— `qiskit_ibm_runtime.fake_provider`
- **IBM Quantum Properties Archive on Zenodo** —— 搜 "IBM Quantum calibration data"，多设备多月时间序列存档
- **Pulser 数据**（Pasqal 中性原子）— 备用对照

### 化学 / 物理问题
- **PennyLane Datasets**（H2 / LiH / BeH2 / 哈密顿矩阵 / 基态能量）— `pennylane.data`
- **VQEzy**（12K VQE 初始化轨迹）— arXiv:2509.17322
- **OpenFermion** + Psi4 generated FCI energies

### LLM 训练 / SFT 用
- **HuggingFace 上的 OpenQASM 数据集**（如 `qiskit-textbook` 抽取版）
- **arXiv:2504.11109** 配套发布的 14k QAOA/VQE 电路 SFT 集（可复用）

### QEC
- **Stim sample data** + Google Quantum AI 的 surface code syndrome 数据集（公开）

---

## 五、12 周分阶段计划

### Phase 0 · 地基（W1–W2）

**目标**：影子硬件 + 评测脚手架 ready，不写一行业务逻辑就能跑通"prompt → 模拟结果"。

**具体任务**

1. 建仓：`quantum-gpt/` 单仓 monorepo，子模块 `backends/` `tools/` `agent/` `bench/` `eval/`
2. 实现 `ShadowBackend` 抽象类，三个具体子类：
   - `FakeBackendAdapter`（包 Qiskit FakeBackendV2）
   - `ReplayBackend`（接受时间戳，从历史 snapshot 取 properties）
   - `SyntheticDriftBackend`（参数化的 T1/T2 / gate error 漂移函数）
3. 写 `properties_stream.py`：每 4 秒 emit 一次 calibration snapshot 到 in-memory bus（mock 真机的 telemetry 流）
4. 接好 W&B + DuckDB，所有 calibration / job / result 都落库
5. **MVP**：CLI 跑 `qgpt simulate ghz --shots 8192 --backend FakeBrisbane` 出 fidelity 数

**Deliverables**
- [ ] `BACKENDS.md` 说明三种影子后端的语义
- [ ] CI 测试覆盖
- [ ] MVP CLI 可演示

**成功判据**：能在 5 分钟内跑通"prompt → simulator → fidelity"，结果落 W&B。

---

### Phase 1 · 感知层（W3–W4）

**目标**：把"实时硬件感知"做扎实，能驱动后续决策。

**具体任务**

1. 实现 6 个核心感知工具（暴露给 agent）：
   - `get_backend_health(backend)` → fidelity / queue / age
   - `get_qubit_properties(backend, qubits)` → T1/T2 / readout / gate err
   - `get_coupling_map(backend)`
   - `detect_drift(backend, window)` → 给出 drift score + 建议是否重校
   - `get_calibration_age(backend)`
   - `compare_backends(problem, candidates)` → 多后端对比
2. **重点：Drift Detector** —— 用 ReplayBackend 喂入真实 IBM 一周历史数据，做 changepoint detection（PELT / Bayesian Online CP）。这是个独立的小 paper 贡献。
3. 前端 Operator Console 接通真实数据流（PPT 里的 dashboard 现在变实的）

**Deliverables**
- [ ] 6 个感知工具 + 单测
- [ ] Drift detector 准确率（vs 人工标注）评测报告
- [ ] Console 能看到一周历史漂移回放

**成功判据**：drift detector 在 100 个人工标注的漂移点上 F1 ≥ 0.7。

---

### Phase 2 · 工具矩阵 + Agent 框架（W5–W6）

**目标**：搭好 agent 大脑，让它能调用 12+ 工具完成端到端任务。

**具体任务**

1. 用 **MCP 协议**封装所有工具（这个决策很关键，让你以后能接任何 LLM）：
   - 感知层 6 个（W3–W4 已做）
   - `transpile_circuit(circ, backend, opt_level, noise_aware)`
   - `simulate(circ, backend, shots, noise_model)`
   - `apply_mitigation(circ, method)` —— Mitiq 提供 ZNE / PEC / CDR
   - `predict_fidelity(circ, backend)` —— 简单 GNN 或 lookup 都先行
   - `retrieve_similar_circuits(circ, top_k)` —— 向量库
   - `decompose_problem(prompt)` —— 把自然语言拆成子任务
2. Agent loop 用 **LangGraph 的 ReAct 模板**起步，加自定义的 fidelity-budget 节点
3. 双 LLM 模式：default 用 Claude / GPT 跑通逻辑，并行接 vLLM-本地（用 Qwen2.5-32B 起步而不是 72B，迭代快）
4. **基线**：实现 3 个对照 baseline
   - **Static Pipeline**：固定流程（transpile → submit → mitigation）
   - **LLM Single Shot**：只让 LLM 一次性写完整 Qiskit 代码
   - **ReAct + Tools**：标准 ReAct（即将作为弱基线被你超过）

**Deliverables**
- [ ] 12+ 工具的 MCP schema 文档
- [ ] 5 个端到端任务（GHZ / VQE H2 / QAOA MaxCut-6 / Bernstein-Vazirani / QFT）能在所有 baseline 跑通
- [ ] baseline 性能基线表（含 fidelity / latency / success rate）

**成功判据**：5 个任务在 4 个系统（3 baseline + qgpt-naive）上全部出数据，跑一次全套 < 30 分钟。

---

### Phase 3 · 闭环 & 学习（W7–W9）

**目标**：让 agent 从结果中学习 —— 这是与现有 LLM-Quantum 工作（QUASAR / El Agente Q）拉开差距的地方。

**具体任务**

1. **Episodic Memory** 落地
   - 每次任务结束后存 `(problem, plan, tool calls, fidelity, success)` 到 Qdrant
   - 检索接到 `retrieve_similar_circuits` 工具
   - **小实验**：开 / 关记忆，对比 50 条任务的成功率
2. **Fidelity-aware Planner**
   - 简单版：每次 tool call 前调用 `predict_fidelity`，如果预测低于阈值就走 fallback
   - 进阶版：训一个小 GNN（输入电路 + backend properties → fidelity 标量）。**训练数据用 Phase 2 跑出来的所有 (circ, backend, real_fidelity) 元组**，自然就有了
3. **Drift-aware Replanning**
   - 监听 properties_stream，当 drift detector 报警时主动 invalidate 当前任务的 transpile 结果，重做
   - 这是 Quantum GPT 区别于普通 LLM-agent 的硬实力体现
4. **可选 stretch**：做一次小的 agentic RL 实验（参考 QUASAR），用 Phase 2 的 real_fidelity 作为 reward 微调本地 7B 模型

**Deliverables**
- [ ] memory ablation 报告
- [ ] fidelity GNN（含训练曲线 / R² 评估）
- [ ] drift-aware replanning 视频 demo（重要 —— paper 投稿要附）
- [ ] 在 5 个端到端任务上，Quantum GPT vs 三个 baseline 的对比表 ——**���就是 paper 主图**

**成功判据**：full Quantum GPT 在 fidelity / success rate 上较 ReAct baseline 提升 ≥ 15%。

---

### Phase 4 · Benchmark 化（W10–W11）

**目标**：把上面的工作产品化为可发布的 benchmark —— 这是切口 6 的卡位。

**具体任务**

1. 设计 **QC-Agent-Bench**（先内部叫这个名）
   - **Tier 1 · 静态任务**：30 个，固定 backend snapshot，看正确率
   - **Tier 2 · 漂移任务**：20 个，跑在 ReplayBackend 上，必须实时适应
   - **Tier 3 · 故障任务**：10 个，注入"qubit 突然挂掉"等异常，看 agent 能否换路线
2. 评测维度：
   - **Task success rate**（结果是否正确）
   - **Output fidelity**（结果有多接近 ground truth）
   - **Resource efficiency**（shot 数 / tool 调用数）
   - **Robustness**（漂移 / 故障下的退化曲线）
3. 把 baseline 数据 + 你的系统数据全部公开
4. 写 dataset card / leaderboard 网页框架

**Deliverables**
- [ ] Bench spec PDF
- [ ] GitHub 仓库（含 60 个任务的 ground truth）
- [ ] 五个系统的对比结果表
- [ ] leaderboard 雏形网页

**成功判据**：可以一句话概括 benchmark 的差异化（"第一个评测 quantum-LLM-agent 在漂移和故障下韧性的标准集"）。

---

### Phase 5 · 论文 / 投稿准备（W12+）

**目标**：双线投稿。

#### 主线 paper（系统侧 / 工程顶会）
- **Target**：HPCA / ASPLOS / MICRO（CCF-A）一稿；ICLR / NeurIPS Main 一稿
- **标题候选**：*Quantum GPT: A Hardware-Aware Agentic Operator for NISQ Workflows*
- **主图**：闭环架构 + 在 30 个静态任务 / 20 个漂移任务上的全面对比
- **关键卖点**：
  1. 第一个 closed-loop agent 系统
  2. drift-aware
  3. episodic memory 真有用（消融）

#### Benchmark paper（占位 / 影响力）
- **Target**：NeurIPS Datasets & Benchmarks（CCF-A）
- **关键卖点**：定义 QC-Agent 的评估范式，能成为别人引用的标准

---

## 六、评测体系

> **必须 Phase 0 就定下来。不要等到最后一刻再跑评测。**

```python
任务集 = (
    5 个 Phase-2 任务
    + 30 个 QC-Agent-Bench Tier-1
    + 20 个 Tier-2 (drift)
    + 10 个 Tier-3 (failure)
)

baseline = {
    "static":         "固定 pipeline",
    "llm-single":     "单次 LLM 写 Qiskit 代码",
    "react-base":     "ReAct + tools (无 memory / 无 fidelity-aware)",
    "qgpt":           "Quantum GPT 完整版",
    "qgpt-no-mem":    "消融 1: 去掉记忆",
    "qgpt-no-drift":  "消融 2: 去掉 drift-aware",
}

metrics = {
    "success_rate":    "任务结果正确的比例",
    "avg_fidelity":    "端到端保真度",
    "tool_calls":      "平均工具调用数 (越少越好)",
    "wall_time":       "端到端延迟",
    "drift_recovery":  "漂移注入后 N 步内恢复率",
    "cost":            "token + simulator 时间成本",
}
```

**操作规范**：
- 每周跑一次全套 → W&B 看趋势
- 每周一固定时间发 progress note
- 任何代码改动后必须确保至少一个 baseline 还能跑通（防止重构破坏对照组）

---

## 七、关键风险与对策

| 风险 | 对策 |
|---|---|
| FakeBackend 与真机差距过大 | 用 ReplayBackend 喂真实历史校准；paper 里诚实说明，并设计"sim-to-real gap"实验作为讨论 |
| LLM tool calling 不稳定 | 双 LLM 模式（云端 + 本地）；MCP 协议保证 schema 严格；一开始用 Claude 兜底 |
| 漂移检测器误报 | drift detector 单独做评估 + 人工标注 IBM 历史数据 100 个点 |
| 评测开销爆炸 | Phase 0 就上 W&B + 缓存所有 simulator 结果（同样电路同样 backend 直接命中 cache） |
| benchmark 被人抢发 | Phase 4 做完直接挂 arXiv + GitHub，别等会议 deadline |
| 真机最终拿不到 | 工作 90% 不依赖真机；最坏情况换一个 IonQ / Rigetti 公开 SDK 的"伪真机"接口验证 sim-to-real |
| GNN fidelity predictor 训不出来 | 退化为 lookup table 也能跑通系统；不要让 ML 部分卡住主线 |
| 12 周做不完 | Phase 4 是最容易裁的 —— 实在不行先发主线 paper，benchmark 留 v2 |

---

## 八、未来真机到位时如何切换

写 Phase 0 时就要保证下面这两条：

1. **Backend 抽象用 Qiskit `BackendV2` 协议** —— 真机 backend 直接进 backend pool，agent 完全无感知
2. **所有工具 input/output 用 JSON Schema** —— 任何替换不会 break agent loop

理想状态下：拿到真机当天，改一行 `provider.get_backend("ibm_brisbane")`，整个系统迁移完成。

---

## 九、第一周（本周）行动清单

按优先级，下面这五件事**这周做完**，下周一就有跑通的 MVP：

| Day | 任务 |
|---|---|
| Day 1–2 | 建仓、装环境、跑通 `FakeBrisbane.run(transpile(GHZ_5))`，把 fidelity 打出来 |
| Day 2–3 | 写 `ShadowBackend` 抽象 + `FakeBackendAdapter` |
| Day 3–4 | 把 MQTBench 装好，挑 5 个任务能跑 baseline |
| Day 4–5 | 接 Claude / GPT 做一个最朴素的"读问题 → 调 1 个工具 → 出结果"loop（不用 LangGraph，直接 while 循环就行） |
| Day 5–7 | W&B + DuckDB 接好，开始记每次���验 |

**完成判据**：Phase 0 完成 50%，Phase 2 的 baseline-3 可以开始评测。

---

## 十、两个核心建议

### 1. 不要一开始就上多 agent
先把单 agent + 12 工具做透 —— QUASAR 的论文也只是单 agent + agentic RL，已经够发 Top conference。多 agent 留到 v2。

### 2. paper 的故事要在 Phase 0 就想清楚
你的故事不是"我们做了一个量子 LLM"，而是：

> **"第一个能在 NISQ 漂移条件下保持任务成功率的 closed-loop agent，并且我们提供了评估这件事的标准 benchmark。"**

所有 12 周的工作都围绕这个故事写代码。

---

## 附录 A · 待办看板模板

建议用 GitHub Project / Linear 建一个简单看板，列分四档：

```
NOW    ── 本周必做
NEXT   ── 下周锁定
LATER  ── 已规划但未排期
ICEBOX ── stretch goal / 想到但暂不做
```

每周一从 NEXT 拉到 NOW，每周五检查 NOW 完成度并写 weekly note。

---

## 附录 B · 论文故事板（draft）

> 这部分是给最终 paper 用的故事提纲，Phase 0 就先写出来，每周根据实验结果迭代。

### Title
*Quantum GPT: A Hardware-Aware Agentic Operator for NISQ Workflows under Drift*

### Abstract（1 段）
NISQ 硬件持续漂移、误差缓解需要专家经验、现有 LLM-quantum 工具链是开环的。我们提出 Quantum GPT —— 一个具备硬件感知、工具调度、情景记忆的闭环 agent，配套发布 QC-Agent-Bench（60 个任务，覆盖静态、漂移、故障三类）。在 60 个任务上较 ReAct baseline 提升 X% success rate、Y% fidelity，且对漂移恢复时间缩短 Z%。

### 主图（Figure 1）
三层架构（Perception / Cognition / Execution）+ 闭环箭头 + 一个具体 trace 示例。

### 实验主表（Table 1）
6 个系统 × 3 个 tier × 4 个 metric 的 12 行表，关键数字加粗。

### 消融（Figure 2）
- 去 memory 掉多少
- 去 drift-aware 掉多少
- 改用本地 32B 模型掉多少

### sim-to-real 讨论（Section 6）
诚实说明所有实验在影子硬件上做；用 ReplayBackend 论证模拟与真机的真实噪声 distribution 差距 ≤ X%；列举未来真机迁移路径。

---

## 待商讨的 Open Question

下面这些是计划里目前**故意留白**的决策点，建议第一次商讨时逐一过：

1. 主 LLM 选什么？（Claude / GPT-4o / DeepSeek-V3 / Qwen2.5）
2. 论文优先级：系统会议（HPCA / ASPLOS）还是 ML 会议（NeurIPS / ICLR）先投？
3. Phase 3 的 agentic RL 实验做不做？（决定是否需要 GPU 资源 + 数据闭环时间）
4. Benchmark 是先内部用还是 Phase 4 一定开源？
5. 是否需要拉合作者（量子物理 / 系统方向）？
6. 团队人力：你一个人推还是有同学 / 师弟可分工？
7. 如何在没有真机的情况下争取后续合作（IBM / 中科院量子云 / 本源）？

---

*文档版本：v1.0 · 2026-05*
*下一步：基于本文档进行计划商讨 → 锁定 Phase 0 范围 → 启动 Day 1*
