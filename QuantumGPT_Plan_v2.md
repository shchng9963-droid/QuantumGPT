# Quantum GPT · 12 周细节版研究计划 v2

> **Lab-Service Edition** — 面向物理组真机调控的扩展版本
>
> **v1 → v2 核心变化**：把目标从「AI agent benchmark 系统」升级为「**物理实验室真机调控的智能副驾**」。技术栈从门级下沉到 **gate / pulse / device 三层**，新增 Phase L 与物理组协作主线，paper 故事从单一 ML 会议扩展到 **AI 顶会 + 系统会议 + 物理顶刊三线**。
>
> **v1 文档**：`QuantumGPT_Plan.md`（保留作为对照与回溯）
> **v2 文档**：本文件
>
> v2 不是 v1 的替代，而是 v1 的**超集** —— v1 中所有内容都有效，v2 在此基础上加了 lab 主线、设备抽象、安全护栏、物理组协作四个维度。

---

## v1 → v2 变更总览

| 维度 | v1 | v2 |
|---|---|---|
| 主用户 | AI 研究者 | **实验物理学家 + AI 研究者** |
| 目标层级 | 门级（gate-level） | **门级 + 脉冲级 + 设备级三层** |
| Backend 抽象 | ShadowBackend | **LabBackend**（多 `dispatch_pulse` / `read_measurement` 接口） |
| 工具数量 | 12+ | **25+**（新增校准 / 脉冲优化 / 数据解读 / 实验设计 / 跨设备 / 安全护栏） |
| 记忆系统 | (circuit, fidelity) episodic | **ExperimentRecord**（含 raw data / fits / decisions / human notes） |
| 主线 phase | Phase 0–5 | Phase 0–5 **+ Phase L (Lab Integration)** |
| Benchmark | Tier 1–3（静态/漂移/故障） | **+ Tier 4（Lab Tasks：tune-up / transfer / data 解读）** |
| 投稿目标 | NeurIPS / ICLR + Datasets&Benchmarks | **+ Nature Commun. / PRX Quantum（与物理组联署）** |
| 安全机制 | 无 | **Dry-run / Power budget / 二级确认 / Audit log** |

---

## 目录

- [一、战略定位（v2 升级）](#一战略定位v2-升级)
- [二、三层调控范畴：gate / pulse / device](#二三层调控范畴gate--pulse--device)
- [三、Lab Backend Pool 架构](#三lab-backend-pool-架构)
- [四、四层技术增强](#四四层技术增强)
- [五、软件栈（含 lab 扩展）](#五软件栈含-lab-扩展)
- [六、数据 / 仿真资源](#六数据--仿真资源)
- [七、12 周计划（Phase 0–5 + Phase L）](#七12-周计划phase-05--phase-l)
- [八、评测体系（含 Lab Tier）](#八评测体系含-lab-tier)
- [九、风险与对策](#九风险与对策)
- [十、真机切换路径](#十真机切换路径)
- [十一、第一周清单（v2）](#十一第一周清单v2)
- [十二、三个核心建议](#十二三个核心建议)
- [十三、论文故事板（v2 升级版）](#十三论文故事板v2-升级版)
- [十四、物理组合作策略](#十四物理组合作策略)
- [十五、待决策的 6 个问题](#十五待决策的-6-个问题)
- [附录 A · v1 / v2 对照速查](#附录-a--v1--v2-对照速查)
- [附录 B · Lab 工具矩阵详表](#附录-b--lab-工具矩阵详表)
- [附录 C · 标准 Tune-Up Sequence 范本](#附录-c--标准-tune-up-sequence-范本)

---

## 一、战略定位（v2 升级）

### 1.1 最高目标
> **让物理组每天少花 4–8 小时在重复性 tune-up 上，多产出 2–3 倍的实验数据。**

这不是抽象口号 —— 这是绝大多数超导 / 离子阱 / 中性原子物理组的真实痛点。任何能减少这部分时间的工具，都会被物理学家长期使用。

### 1.2 用户画像

| 角色 | 占比 | 主要诉求 |
|---|---|---|
| 实验物理学家（PI / 博后 / 博士生） | 70% | tune-up 自动化、数据解读、实验决策、可复现性 |
| 系统 / Infra 研究者 | 20% | 工具调度、闭环优化、benchmark |
| AI / ML 研究者 | 10% | agent 框架、tool calling、foundation model |

> 💡 **设计准则**：所有功能都要能让"物理学家拿来就用"。不能强迫他们写 Python 才能用 —— 提供 notebook / CLI / Web Console 三种交互。

### 1.3 三个差异化卖点（vs 现有 quantum-LLM 工作）

| 工作 | 层级 | 闭环 | 真机 | Lab Service |
|---|---|---|---|---|
| QUASAR (2025) | gate (QASM) | ✗ | ✗ | ✗ |
| El Agente Q (2025) | 化学层 | 部分 | ✗ | ✗ |
| AI-Mandel (2025) | 实验设计 | 部分 | ✗ | ✗ |
| **Quantum GPT v2** | **gate + pulse + device** | **✓** | **可切换** | **✓** |

---

## 二、三层调控范畴：gate / pulse / device

物理组真机调控有三个层级，v2 必须全部覆盖：

```
┌─────────��────────────────────────────────────────────────────┐
│  GATE LEVEL · 门级                                            │
│  · transpile / routing / mitigation                           │
│  · v1 已覆盖                                                  │
├──────────────────────────────────────────────────────────────┤
│  PULSE LEVEL · 脉冲级（新增）                                  │
│  · GRAPE / DRAG / CRAB pulse 优化                             │
│  · cross-resonance / parametric gate 设计                     │
│  · pulse-level transpile（Qiskit Pulse / Dynamics）           │
├──────────────────────────────────────────────────────────────┤
│  DEVICE LEVEL · 设备级（新增 · 物理组真痛点所在）              │
│  · cavity / qubit spectroscopy                                │
│  · Rabi / Ramsey / DRAG / RB / GST 等 tune-up                 │
│  · readout discrimination / IQ blob                           │
│  · drift / cross-talk / leakage characterization              │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 物理学家日常 trace 范例

> "qubit Q3 的 T1 掉了 → 检查 readout / dispersive shift → 跑 spectroscopy 找新频率 → 重新 tune DRAG → 跑 RB 验证 → 更新 calibration → 通知所有依赖任务重做 transpile"

这是一个 **30 分钟的人工流程**，涉及 7 个步骤、3 个层级、5–8 个工具。Quantum GPT 应该能 5 分钟内自动完成并出实验报告。

### 2.2 跨层流动是核心能力

agent 必须能在层级间自由跳转：
- 门级遇到瓶颈 → 下沉到脉冲级优化 specific 2Q gate
- 脉冲优化效果差 → 再下沉到设备级查频率漂移
- 设备级修好后 → 上浮重新 transpile 并 propagate 影响

**这是 Quantum GPT 区别于 QUASAR 的核心价值** —— QUASAR 只在门级，下面两层完全空白。

---

## 三、Lab Backend Pool 架构

v1 的 ShadowBackend 升级为 **LabBackend** —— 接口扩充以支持脉冲与设备层。

```
┌────────────────────────────────────────────────────────────┐
│  LAB BACKEND POOL                                          │
│                                                            │
│  ┌── Cloud / Fake (gate-level) ──────────────────────┐    │
│  │  · FakeBrisbane / FakeKyiv / FakeSherbrooke       │    │
│  │  · ReplayBackend (历史校准回放)                    │    │
│  │  · SyntheticDriftBackend                          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                            │
│  ┌── Pulse-level Simulators (新增) ──────────────────┐    │
│  │  · Qiskit Dynamics (BackendV2 兼容)               │    │
│  │  · QuTiP-QIP (开源全栈)                           │    │
│  │  · C3-Toolset (realistic transmon)                │    │
│  │  · Pulser (中性原子)                              │    │
│  └────────────────────────────────────────────────────┘    │
│                                                            │
│  ┌── Lab Software Adapters (新增) ───────────────────┐    │
│  │  · QCoDeS adapter (Delft / Microsoft 生态)        │    │
│  │  · pyqum / Labber adapter (国内常见)              │    │
│  │  · Qiskit Experiments runner                       │    │
│  │  · QM QUA adapter (商用 OPX，预留)                 │    │
│  └────────────────────────────────────────────────────┘    │
│                                                            │
│  统一接口：                                                 │
│   · dispatch_pulse(schedule)                               │
│   · run_circuit(circuit, shots)                            │
│   · read_measurement(spec) → IQ / counting / waveform     │
│   · update_calibration(params)                             │
│   · get_device_state() → 完整 snapshot                     │
└────────────────────────────────────────────────────────────┘
```

### 3.1 三种使用模式

| 模式 | 后端 | 用途 |
|---|---|---|
| **Pure Sim** | Qiskit Dynamics / QuTiP | 安全开发、无风险测试 |
| **Replay** | ReplayBackend + 真实历史数据 | 验证漂移 / 故障下的 agent 行为 |
| **Live (未来)** | QCoDeS / pyqum + 物理组真机 | Phase L 联合验证、最终落地 |

### 3.2 协议规范

`LabBackend` 使用 **JSON Schema + MCP** 双重保障，所有真机适配只需实现 5 个原语，agent 与上层逻辑完全解耦。

---

## 四、四层技术增强

### 4.1 Layer 1 · 设备抽象层

如三、所述。**设计原则**：
- 每个真机包成 `LabBackend` 子类
- 5 个原语必须覆盖：`dispatch_pulse` / `run_circuit` / `read_measurement` / `update_calibration` / `get_device_state`
- 5 个 metadata 必须暴露：`device_type` / `qubit_count` / `coupling_map` / `gate_set` / `pulse_resolution`

### 4.2 Layer 2 · 物理学家工具矩阵

详细工具表见 [附录 B](#附录-b--lab-工具矩阵详表)。这里只列分组：

| 工具组 | 数量 | 关键代表 |
|---|---|---|
| 感知（v1 已做） | 6 | get_backend_health / detect_drift |
| 门级（v1 已做） | 4 | transpile / simulate / mitigation / predict_fidelity |
| **校准 / Tune-up（v2 新增）** | 7 | rabi / ramsey / drag / RB / spectroscopy |
| **脉冲优化（v2 新增）** | 4 | grape_optimize / crab / pulse_shaper |
| **数据解读（v2 新增）** | 5 | fit_rabi / extract_t1_t2 / iq_classifier |
| **实验设计（v2 新增）** | 3 | next_best_experiment / bayesian_design |
| **跨设备迁移（v2 新增）** | 2 | transfer_calibration / port_pulse_schedule |
| **安全护栏（v2 新增）** | 4 | dry_run / power_budget / amplitude_check / audit |
| **总计** | **35** | |

### 4.3 Layer 3 · ExperimentRecord 实验记忆

v1 的 episodic memory 升级为结构化实验日志：

```python
ExperimentRecord = {
    # 设备 / 会话
    "device": "transmon_Q3",
    "session": "2026-05-15-tune-up-am",
    "timestamp": "2026-05-15T09:32:14Z",

    # 原始数据
    "raw_data": {
        "iq_traces": "<parquet path>",
        "oscilloscope": "<hdf5 path>",
        "shots": 8192,
    },

    # 拟合结果
    "fits": {
        "rabi_freq_MHz": 25.4,
        "T1_us": 84.2,
        "T2_us": 58.1,
        "DRAG_alpha": 0.42,
        "fit_quality": {"r2": 0.987, "chi2": 1.12},
    },

    # agent 决策轨迹
    "decisions": [
        {"step": 1, "tool": "qubit_spectroscopy", "result": ..., "reasoning": "..."},
        ...
    ],

    # 物理学家手动注解
    "human_notes": "Q3 出现 leakage，下次跑 RB 时 monitor",

    # 复现性
    "reproducibility": {
        "git_sha": "abc123",
        "device_state_hash": "def456",
        "software_versions": {"qiskit": "1.2.3", "pyqum": "0.5.0"},
    }
}
```

> 💡 **价值**：这本质上是物理组的"集体记忆库"。一个组里所有人的实验都进同一个 store，新来的博士生可以直接 RAG 检索"上一个人怎么调好这个 qubit 的"。

### 4.4 Layer 4 · 安全与可解释（v2 必须有）

物理组设备贵 + 易损，**这是产品化的硬刚需**。

| 机制 | 实现 | 触发时机 |
|---|---|---|
| **Dry-run** | 任何 pulse-level 改动先在 Qiskit Dynamics 仿真 | 默认 ON，不可关闭 |
| **Power budget** | 脉冲幅度 / 时长 / 占空比硬限 | dispatch_pulse 前置检查 |
| **二级确认** | 写 calibration / 影响其他用户的操作需 LLM reasoning + 人工确认 | 白名单可豁免 |
| **Audit log** | 所有 tool call 落不可篡改日志（DuckDB + hash chain） | 默认 ON |
| **Reasoning trace** | LLM 每次决策必须给出原因 | 默认 ON |
| **Rollback** | 校准库版本化（git-like），任何写都能回滚 | 默认 ON |

> ⚠ **不要让 agent 直接动设备而没有刹车系统**。这条是物理组合作的不可妥协项。

---

## 五、软件栈（含 lab 扩展）

| 层 | 选择 | v1 / v2 |
|---|---|---|
| Agent runtime | LangGraph | v1 |
| LLM | DeepSeek-V3 / Qwen2.5-72B（本地）+ Claude / GPT-4o（对照） | v1 |
| Inference 加速 | vLLM 0.6+ | v1 |
| 量子 SDK · 门级 | Qiskit 1.x + PennyLane + Cirq | v1 |
| **量子 SDK · 脉冲级** | **Qiskit Dynamics + QuTiP-QIP + C3-Toolset + Pulser** | **v2** |
| **校准 / 表征实验** | **Qiskit Experiments + 自研 tune-up runner** | **v2** |
| 噪声模拟 | Qiskit Aer + Mitiq + Stim | v1 |
| **Lab 软件 adapter** | **QCoDeS / pyqum / Labber 三选一起步** | **v2** |
| 工具协议 | MCP | v1 |
| **数据格式** | **Parquet（IQ）+ HDF5（waveform）+ DuckDB（指标时序）** | **v2** |
| 向量库 | Qdrant | v1 |
| 实验跟踪 | W&B | v1 |
| **拟合 / 数据分析** | **scipy + lmfit + 自研 IQ classifier** | **v2** |
| 可视化 | Plotly + Operator Console | v1 |
| **Notebook 接口** | **Jupyter widgets + Quantum GPT magic command** | **v2** |

> 💡 **关键决策**：物理组对 Notebook 的依赖远大于 CLI。v2 必须有 `%qgpt` magic command + Jupyter widget UI，否则推不动落地。

---

## 六、数据 / 仿真资源

### 6.1 v1 已列（保留）
- MQTBench / QASMBench / BenchPress / HamLib
- IBM Quantum Properties Archive（Zenodo）
- PennyLane Datasets / VQEzy / OpenFermion
- HuggingFace OpenQASM 数据集

### 6.2 v2 新增 · Pulse / Device 仿真
- **Qiskit Dynamics**（IBM 官方 pulse 仿真，与 BackendV2 兼容）
- **QuTiP-QIP**（开源全栈，支持任意 Hamiltonian）
- **C3-Toolset**（FZ Jülich，realistic transmon device model）
- **Pulser**（Pasqal 中性原子 pulse-level）
- **scQubits**（Princeton，superconducting qubit Hamiltonian 库）

### 6.3 v2 新增 · 校准 / 表征数据
- **Qiskit Experiments**（IBM 官方 tune-up 实验库 + 仿真数据）
- **Cirq / Stim** 的 RB / GST 标准实验
- **arXiv 论文附带的 raw data**（近两年 PRX Quantum 几乎都强制 data availability）
- **NIST / SLAC** 公开实验数据集
- **Microsoft Quantum** Azure 公开 calibration data（部分）

### 6.4 v2 新增 · Lab 软件协议参考
- **QCoDeS** / **PycQED**（Delft）— 学术界主流接口
- **pyqum** / **Labber** — 国内组常用
- **Quantum Machines QUA** — 商用 OPX 控制器
- **Zurich Instruments LabOne** — 高端实验组常见

---

## 七、12 周计划（Phase 0–5 + Phase L）

> v2 的整体节奏与 v1 相同（12 周），但每个 phase 都加入 lab 维度，并新增 Phase L 与 Phase 3/4 并行推进。

### Phase 0 · 地基（W1–W2）

**v1 任务保留**：建仓、MVP CLI、W&B + DuckDB

**v2 增加**
- [ ] `LabBackend` 抽象（升级 ShadowBackend，加 `dispatch_pulse` / `read_measurement` / `update_calibration` / `get_device_state` 四个新接口）
- [ ] 接入 **Qiskit Dynamics**（pulse-level 仿真，作为第一个非门级后端）
- [ ] 接入 **QuTiP-QIP**（更通用的开源全栈，作为第二个）
- [ ] 安全护栏雏形：`dry_run` decorator + 一个简单的 `power_budget_check`
- [ ] **MVP 升级**：`qgpt rabi --device transmon_sim --qubit 0` 能跑出 Rabi 曲线

**Deliverables**
- [ ] `LAB_BACKENDS.md`（v2 取代 v1 的 BACKENDS.md）
- [ ] CI 测试覆盖 5 种后端
- [ ] MVP CLI：能跑 GHZ（门级）+ Rabi（设备级）两个任务

**成功判据**：5 分钟内跑通"prompt → pulse-level simulator → IQ data → fit → 报告"，结果落 W&B。

---

### Phase 1 · 感知层（W3–W4）

**v1 任务保留**：6 个核心感知工具、drift detector、Console 实时数据流

**v2 增加**
- [ ] Drift detector 升级 v2：不只看 gate error，还看 **frequency drift / dispersive shift / readout fidelity drift**
- [ ] Console 加 **Device State 视图**（IQ blob 散点 + 频率漂移时间轴 + 校准 age heatmap）
- [ ] 实现 `iq_classifier_train` 工具（用历史 IQ 数据训一个 LDA / 浅层 NN 分类器，作为后续 readout 决策的底座）
- [ ] 拓扑感知工具 `analyze_crosstalk` —— 基于 ZZ coupling 估计 crosstalk

**Deliverables**
- [ ] 3 类 drift detector 准确率报告（gate / frequency / readout）
- [ ] Device State 视图能回放一周历史
- [ ] IQ classifier 在仿真数据上 ≥ 95% 准确率

**成功判据**：在仿真注入的 100 个漂移点上，三类 drift detector 的综合 F1 ≥ 0.75。

---

### Phase 2 · 工具矩阵 + Agent 框架（W5–W6）

**v1 任务保留**：12 个 MCP 工具、LangGraph ReAct、3 个 baseline

**v2 增加**
- [ ] 工具矩阵从 12 扩到 **35**（详见 [附录 B](#附录-b--lab-工具矩阵详表)）
  - Tune-up 工具组（7 个）：cavity_spec / qubit_spec / rabi / ramsey / drag / RB / GST
  - Pulse 优化工具组（4 个）：grape / crab / pulse_shaper / derivative_removal
  - 数据解读工具组（5 个）：fit_rabi / fit_ramsey / extract_t1_t2 / iq_classifier_train / parse_oscilloscope
  - 实验设计工具组（3 个）：next_best_experiment / bayesian_design / info_gain_estimate
  - 跨设备迁移（2 个）：transfer_calibration / port_pulse_schedule
  - 安全护栏（4 个）：dry_run / power_budget / amplitude_check / audit_log
- [ ] 跑通 **完整 tune-up sequence demo**：cavity spec → qubit spec → Rabi → Ramsey → DRAG → RB（在 Qiskit Dynamics 仿真上）—— [附录 C](#附录-c--标准-tune-up-sequence-范本) 给完整范本
- [ ] **新增 baseline · Manual Tune-up**：模拟物理学家手工流程，作为 lab tier 的对照基线

**Deliverables**
- [ ] 35 个工具的 MCP schema 完整文档
- [ ] 完整 tune-up sequence 一键跑通的 demo 视频
- [ ] 5 + 5 个端到端任务（5 个 v1 门级 + 5 个 v2 lab 级）在所有 baseline 跑通

**成功判据**：在 10 个端到端任务（含 lab tier）上，4 个系统全跑通，跑一次全套 < 60 分钟。

---

### Phase 3 · 闭环 & 学习（W7–W9）

**v1 任务保留**：Episodic Memory、Fidelity-aware Planner、Drift-aware Replanning

**v2 增加**
- [ ] **ExperimentRecord schema 落地**（替代 v1 的简单 episodic memory）
- [ ] **Active Learning Experiment Designer**：基于当前 fit 不确定性，决定下一个最有价值的实验（用 botorch / GPyOpt 实现 Bayesian experimental design）
- [ ] **Cross-device transfer 实验**：FakeBrisbane 调好的 calibration 自动迁到 FakeKyiv，比较迁移成功率 vs 重新 tune-up
- [ ] **Stretch · Tune-up RL 子 agent**：用 Phase 2 数据训一个小 agent 专门做 tune-up sequence 的步骤选择（参考 QUASAR 的 agentic RL 范式）

**Deliverables**
- [ ] ExperimentRecord 数据库（含 ≥ 500 条记录）
- [ ] Active learning vs random baseline 在 next-experiment 选择上的对比报告
- [ ] Cross-device transfer 报告（成功率、节省时间）
- [ ] **Paper 主图初稿**：Quantum GPT vs 4 baseline 在 10+ 任务上的全面对比

**成功判据**：full Quantum GPT 在 fidelity / success rate / tune-up 时间上较最强 baseline 全面提升；Active learning 实验数减少 ≥ 40%。

---

### **Phase L · Lab Integration（W6–W11，与 Phase 3/4 并行）** ⭐ v2 新增

> 这是 v2 最关键的新增主线。**没有 Phase L，v2 就只是 v1 的技术升级；有了 Phase L，v2 才真正是 Lab Service 产品。**

| 周次 | 任务 | 产出 |
|---|---|---|
| **W6** | 锁定 1–2 个目标物理组（详见[十四节](#十四物理组合作策略)），发出合作意向；准备 Quantum GPT 5 分钟 demo | 合作邀请回复、demo 视频 |
| **W7** | site visit / 远程沟通；拿到他们的：① 实验脚本样例 ② raw data 样例 ③ 设备配置文档 ④ 当前痛点 top-3 | 物理组需求文档（内部） |
| **W8** | 实现针对该组实际设备的 **专用 LabBackend adapter**（QCoDeS / pyqum 二选一） | 真机 adapter 代码 + 离线测试通过 |
| **W9** | 在他们的真实历史数据上跑回放评测（不需要动设备）；产出"如果当时用了 Quantum GPT，能省多少时间"对比报告 | 回放评测报告 |
| **W10** | 部署一个**只读 + dry-run** 版本到他们的实验室（agent 可以观察 + 建议，但不能写入校准库） | Lab 部署版本 v0.1 |
| **W11** | 物理组真实使用反馈收集 + 最关键 3 个 issue 修复；如果信任度足够，开放一两个安全工具的写权限 | 用户反馈 + Lab v0.2 |

**Phase L 成功判据**：物理组里有 **≥ 1 位老师 / 学生愿意把 Quantum GPT 加到日常 workflow 中**（哪怕只用其中一个工具，比如 fit_rabi 或 next_best_experiment）。

> 💡 **Phase L 的隐藏价值**：物理组合作 = 真实数据 + 联合作者 + 真机背书。这三件事每一件都能让 paper 的影响力翻倍。

---

### Phase 4 · Benchmark 化（W10–W11）

**v1 任务保留**：Tier 1（静态 30）+ Tier 2（漂移 20）+ Tier 3（故障 10）

**v2 增加 Tier 4 · Lab Tasks（20 个）**
- [ ] **Tune-up 任务（10 个）**：输入"设备初始失谐状态" → 输出"能跑高保真门"
  - 需要 agent 自主选择 spectroscopy → Rabi → Ramsey → DRAG → RB 的合适顺序
- [ ] **Cross-device transfer 任务（5 个）**：A 设备的 schedule 迁移到 B 设备的成功率
- [ ] **Raw data 解读任务（5 个）**：给 IQ trace / 噪声波形 → 提取 T1 / T2 / leakage 等参数

**Deliverables**
- [ ] **QC-Agent-Bench 完整版**（80 任务 = Tier 1–4）
- [ ] Bench spec PDF（v2，含 Lab Tier 详细说明）
- [ ] GitHub 仓库（含 80 个任务的 ground truth）
- [ ] 5 个系统 × 4 个 tier × 6 个 metric 完整对比表
- [ ] leaderboard 网页 + arXiv 抢发

**成功判据**：Lab Tier 是社区**首个**评测 quantum agent 在设备级任务上能力的标准集。

---

### Phase 5 · 论文 / 投稿（W12+）

#### 主线 paper（系统侧）
- **Target**：HPCA / ASPLOS / MICRO（CCF-A），ICLR / NeurIPS Main（CCF-A）
- **标题**：*Quantum GPT: An Agentic Copilot for Lab-Grade Quantum Hardware Operations*
- **故事**：第一个跨 gate / pulse / device 三层的闭环 agent，drift-aware + safety-by-design

#### Benchmark paper
- **Target**：NeurIPS Datasets & Benchmarks（CCF-A）
- **故事**：QC-Agent-Bench 80 任务，含独有的 Lab Tier；定义 quantum-agent 评估范式

#### **新增 · Lab Service paper** ⭐
- **Target**：Nature Communications / **PRX Quantum** / npj Quantum Information
- **故事**：与 Phase L 物理组**联署**，实证表明 AI agent 能减少 X% tune-up 时间、提升 Y% 实验产出
- **数据基础**：Phase L 收集的真实 lab usage data + 合作组联署
- **关键卖点**：**这是物理顶刊唯一接受的角度** —— 不是"我们做了一个 AI 系统"，而是"AI 帮物理组真的提了产出"

---

## 八、评测体系（含 Lab Tier）

```python
任务集 = (
      5  Phase-2 端到端任务（v1）
    + 5  Phase-2 lab 端到端任务（v2）
    + 30 QC-Agent-Bench Tier-1（静态）
    + 20 Tier-2（漂移）
    + 10 Tier-3（故障）
    + 20 Tier-4 · Lab Tasks（v2 新增）
    = 90 个任务
)

baselines = {
    "static":            "固定 pipeline（v1）",
    "llm-single":        "单次 LLM 写代码（v1）",
    "react-base":        "ReAct + tools（v1）",
    "manual-tuneup":     "模拟人工 tune-up（v2 新增 lab baseline）",
    "qgpt":              "Quantum GPT 完整版",
    "qgpt-no-mem":       "消融：去 ExperimentRecord",
    "qgpt-no-drift":     "消融：去 drift-aware",
    "qgpt-no-safety":    "消融：去安全护栏（仅评测，不上真机）",
    "qgpt-no-active":    "消融：去 active learning experiment designer",
}

metrics = {
    # v1 metrics 保留
    "success_rate":      "结果是否正确",
    "avg_fidelity":      "端到端保真度",
    "tool_calls":        "工具调用数",
    "wall_time":         "端到端延迟",
    "drift_recovery":    "漂移注入后恢复时间",
    "cost":              "token + simulator 成本",

    # v2 新增 lab-specific
    "tuneup_time":       "tune-up 端到端时间（人工 vs agent）",
    "experiment_count":  "达到目标所需实验次数（active learning 关键指标）",
    "transfer_success":  "跨设备 calibration 迁移成功率",
    "data_extract_acc":  "raw data → 物理参数提取精度",
    "safety_violations": "安全护栏触发率（应为 0）",
    "physicist_rating":  "Phase L 物理组主观评分（1-5 Likert）",
}
```

**评测纪律**
- 每周一固定时间跑全套
- 任何代码改动后必须确保 ≥ 1 个 baseline 还能跑通
- W&B 看长期趋势，**绝不能等到投稿前才补评测**
- Phase L 的物理组评分按月收集

---

## 九、风险与对策

| 风险 | v1 / v2 | 对策 |
|---|---|---|
| FakeBackend 与真机差距过大 | v1 | ReplayBackend + 诚实写在 paper 讨论 |
| LLM tool calling 不稳定 | v1 | 双 LLM + MCP schema 严格 + Claude 兜底 |
| 漂移检测器误报 | v1 | 单独评测 + 人工标注 |
| 评测开销爆炸 | v1 | W&B + 结果缓存 |
| benchmark 被人抢发 | v1 | Phase 4 完成立即 arXiv |
| 真机最终拿不到 | v1 | 90% 工作不依赖真机 |
| **物理组合作落空** | **v2** | 同时联系 2–3 个组；准备好"无合作组"的 fallback paper（仍可发 NeurIPS） |
| **Lab Backend adapter 复杂度爆炸** | **v2** | Phase L 只做 1 个组的 adapter；其他组留 v2 之后 |
| **物理学家不信任 agent** | **v2** | Phase L 先 read-only 部署 6+ 周，建立信任；safety-by-design 全文宣传 |
| **设备被 agent 调坏** | **v2** | dry-run 默认 ON；写权限默认 OFF；二级确认 + audit log |
| **pulse-level 仿真精度不够** | **v2** | C3-Toolset 提供 realistic device model；同时跑 QuTiP 做对照 |
| **跨设备 transfer 不 work** | **v2** | 退化为"提供初值，仍需局部 tune-up"也是有价值的结果 |

---

## 十、真机切换路径

写代码时严守两条：
1. **所有 Backend 走 `LabBackend` 协议** —— 真机加进 backend pool，agent 完全无感知
2. **所有工具 input/output 走 JSON Schema** —— 替换实现不破坏 agent loop

**真机迁移成本估计**
- IBM 云端真机：1 行代码（换 provider）
- 物理组真机（Phase L 合作组）：1 个 adapter 模块（W8 已实现）
- 其他物理组：每个组 1–2 周 adapter 开发

---

## 十一、第一周清单（v2）

| Day | 任务 |
|---|---|
| Day 1 | 建仓 `quantum-gpt/`；装 Qiskit 1.x + Qiskit Dynamics + QuTiP-QIP；跑通 FakeBrisbane GHZ 出 fidelity |
| Day 2 | 写 `LabBackend` 抽象（5 个原语接口） + `FakeBackendAdapter` |
| Day 2–3 | 接入 **Qiskit Dynamics**，跑通最简单的 Rabi pulse 仿真 |
| Day 3 | 装 MQTBench；挑 5 个门级 + 2 个 pulse 级任务作为 Day-1 任务集 |
| Day 4 | 接 Claude / GPT 做最朴素的 agent loop（while 循环 + 5 个工具） |
| Day 4–5 | 实现第一个 lab 工具：`rabi_experiment` + `fit_rabi`（端到端跑通：调度 pulse → 读 IQ → 拟合曲线） |
| Day 5–6 | W&B + DuckDB + ExperimentRecord schema 落库 |
| Day 6–7 | **物理组联系**：发 1–2 封合作意向邮件；准备 5 分钟 demo（哪怕只是 Day 4 的 Rabi 跑通视频） |

**第一周成功判据**
- ✅ Phase 0 完成 50%
- ✅ 能跑通门级（GHZ）+ 设备级（Rabi）两种任务
- ✅ 物理组合作邮件已发出

---

## 十二、三个核心建议

### 1. 不要一开始就做 35 个工具
**先做透 8 个**：6 个 v1 感知 + 2 个 lab 工具（rabi_experiment + fit_rabi）。这 8 个跑通 = MVP。剩下的工具按 Phase 顺序补。

### 2. paper 故事 v2 升级
原来：第一个 closed-loop quantum agent
**v2**：第一个跨 gate / pulse / device 三层、能服务物理实验室真机调控的闭环 agent

所有 12 周的工作都围绕这个故事。

### 3. **Phase L 是 v2 的灵魂，越早启动越好**
不要等到 W6 才开始联系物理组。**Day 6–7 就要发出合作意向邮件**，因为合作有 lead time（回复 / 安排会议 / 沟通需求 / 拿数据），等到 W6 才动可能已经迟了。

---

## 十三、论文故事板（v2 升级版）

### 主线 paper

**Title**
*Quantum GPT: An Agentic Copilot for Lab-Grade Quantum Hardware Operations*

**Abstract（1 段）**
NISQ 硬件持续漂移、tune-up 流程经验性、误差缓解需要专家干预；现有 LLM-quantum 工具链（QUASAR / El Agente Q）只覆盖门级，无法服务实验室真机调控。我们提出 **Quantum GPT** —— 第一个跨 **gate / pulse / device 三层** 的闭环 agent，具备硬件感知、工具调度、ExperimentRecord 记忆、active learning 实验设计、安全护栏。配套发布 **QC-Agent-Bench**（90 任务，含独有 Lab Tier）。在 Lab Tier 20 任务上，较 manual-tuneup baseline 缩短 X% tune-up 时间、减少 Y% 实验次数；与 [合作物理组] 联合实证表明能支持真实 lab workflow。

**主图（Figure 1）**
三层架构（gate / pulse / device）+ 闭环 agent + 安全护栏 + ExperimentRecord，并附一条具体物理学家日常 trace 的执行示例。

**实验主表（Table 1）**
9 个系统 × 4 个 tier × 8 个 metric 的对比表，关键数字加粗（v2 metrics 含 tuneup_time / experiment_count）。

**消融（Figure 2）**
- 去 ExperimentRecord 掉多少
- 去 active learning 掉多少
- 去 drift-aware 掉多少
- 去安全护栏：触发率从 0 升到 X%（说明护栏的实际拦截率）

**Lab Tier 专门讨论（Section 5）**
Tune-up sequence demo + cross-device transfer + raw data 解读三个子实验。

**sim-to-real 讨论（Section 6）**
诚实说明影子硬件 + ReplayBackend；用 Phase L 物理组反馈作为 sim-to-real 的早期证据；列出真机迁移路径。

### Lab Service paper（与物理组联署）

**Target**: PRX Quantum / Nature Communications / npj Quantum Information

**Title**
*An AI Copilot for Superconducting Qubit Tune-Up: Reducing Calibration Burden in Practice*

**核心主张**
- 在合作组真实历史数据 + 真实 workflow 上证明：AI agent 能让物理学家**少花 N 小时 / 周**做 tune-up
- 提供 ExperimentRecord 作为可复现性新范式
- 实证 cross-device transfer 在合作组的两台设备间的有效性

**联署作者**：Phase L 合作组的 PI + 学生

---

## 十四、物理组合作策略

### 14.1 合作目标的优先级

| 类型 | 接触难度 | 价值 | 备注 |
|---|---|---|---|
| **本校 / 同院系实验组** | 低 | 中-高 | 优先尝试，沟通成本最低 |
| **本省 / 同城物理组** | 中 | 高 | 次优先 |
| **本源量子 / 国仪量子** | 中 | 高 | 国内最易接触的真机厂商 |
| **中科院量子信息院 / 物理所** | 中-高 | 极高 | 北京 / 合肥 / 中科大 |
| **IBM Q Network 学术成员单位** | 高 | 极高 | 看你校是否成员 |
| **国外组（Delft / ETH / Yale 等）** | 极高 | 极高 | 需现有 connection |

### 14.2 接触话术模板

> 老师您好，
>
> 我是 [学校]计算机学院[课题组]的[姓名]，目前在做一个项目叫 **Quantum GPT** —— 一个面向量子实验室的 AI 副驾系统，希望能帮助物理组��少日常 tune-up 和校准的重复工作量。
>
> 我们已经在 Qiskit Dynamics + QuTiP 仿真上跑通了完整的 tune-up sequence（cavity spec → Rabi → Ramsey → DRAG → RB），现在想找物理组合作做真实场景验证。
>
> 想和您聊 30 分钟：① 您组的 tune-up 流程具体是什么样的？② 当前最耗时的环节是哪个？③ 是否有兴趣作为早期合作者，把我们的工具接到您组的实验流程上？
>
> 任何形式的合作（哪怕只是访谈 + 提供历史数据）都对我们非常有帮助。如果合适，我可以先发一个 5 分钟的 demo 视频。
>
> 谢谢！

### 14.3 合作分级（让对方容易答应）

按对物理组负担从低到高：

1. **30 分钟访谈**（零负担）—— 先建立联系
2. **历史数据共享**（低负担）—— 给我们一些过去的 tune-up 数据，我们做回放评测
3. **工具试用**（中负担）—— 物理学家在 notebook 里跑我们的 fit_rabi 等工具
4. **read-only 部署**（中负担）—— Quantum GPT 接到他们 lab，但只观察 + 建议，不写
5. **完整集成**（高负担）—— agent 真的参与 tune-up 流程
6. **联合发表**（最高价值）—— 联署 lab service paper

**策略**：从 1 谈起，逐级升级。Phase L 把 1–4 跑透就够了。

### 14.4 备选方案（如果合作落空）

**Plan B**：把所有 Phase L 工作转移到"高保真仿真物理组" —— 用 C3-Toolset / Qiskit Dynamics 搭一个**虚拟物理实验室**，所有数据 / 流程都模拟真实组的样子，仍可写主线 paper + benchmark paper，只是少了 Lab Service paper 那一线。

> 💡 **Plan B 不是失败**：Plan B 仍然是 v1 的全部 + v2 技术升级，只是少了物理顶刊投稿线。研究价值依然完整。

---

## 十五、待决策的 6 个问题

> 这 6 个问题是 v2 启动前必须锁定的决策点。建议下次商讨时按顺序过一遍。

### Q1. 物理组合作目标
- 你身边 / 学校里有哪些做超导 / 离子阱 / 中性原子的物理组？
- 最容易接触的是哪个？是否有现成 connection？
- **建议**：本周内列出 3–5 个候选，按"接触难度"排序

### Q2. 设备种类聚焦
- 优先支持哪种 qubit？
  - **超导 transmon**（推荐）：最大宗、生态最全、Qiskit Dynamics 直接支持
  - 离子阱：次之
  - 中性原子 / 光子：小众但有特色
- **建议**：v2 默认主推超导 transmon

### Q3. Lab 软件栈支持范围
- 从 Qiskit Pulse / Dynamics 起步是肯定的
- 是否承诺支持 **QCoDeS**（学术界主流）/ **pyqum**（国内常见）？
  - 这影响和国内组对接难度
- **建议**：v2 起步只承诺 Qiskit + 1 个学术 lab 软件（QCoDeS 或 pyqum，看 Q1 的合作组用什么）

### Q4. Phase L 启动时机
- 按计划 W6 启动；但合作有 lead time，**Day 6–7 就要发邮件**
- 你近期是否有时间和物理组沟通？是否有合适的导师 / 师兄帮忙引荐？
- **建议**：Day 6–7 必须发出第一封邮件

### Q5. 安全护栏的实现优先级
- v2 把安全护栏作为 P0
- 但如果你只在仿真上跑，dry-run 就够了；power_budget / 二级确认 / audit log 可后置
- **建议**：仿真阶段做 dry-run（必做） + audit log（必做） + 简单 power_budget；二级确认推到 Phase L W10 真机部署前

### Q6. paper 第三条线（Lab Service）是否启动
- 这条线**强依赖 Phase L 合作落地**
- 如果合作组在 W7–W8 锁定，Lab Service paper 就启动；否则走 Plan B
- **建议**：先按"会做"准备；W8 看合作进展再最终决定

---

## 附录 A · v1 / v2 对照速查

| 模块 | v1 状态 | v2 状态 |
|---|---|---|
| ShadowBackend 抽象 | 3 子类（Fake / Replay / SyntheticDrift） | 升级为 **LabBackend**，新增 `dispatch_pulse` / `read_measurement` / `update_calibration` / `get_device_state` 四接口 |
| Backend Pool | 仅门级（FakeBackend 等） | + Pulse-level（Qiskit Dynamics / QuTiP / C3 / Pulser）+ Lab adapter（QCoDeS / pyqum） |
| 工具数量 | 12 | **35**（[附录 B](#附录-b--lab-工具矩阵详表)） |
| 记忆系统 | (circ, fidelity) episodic | **ExperimentRecord**（含 raw_data / fits / decisions / human_notes / reproducibility） |
| Phase 数量 | Phase 0–5 | + **Phase L** 并行 |
| Benchmark Tier | 1–3（60 任务） | + **Tier 4 · Lab Tasks**（80 → 90 任务） |
| Baseline 数量 | 6 | **9**（+ manual-tuneup / + safety / + active 三个消融） |
| Metrics 数量 | 6 | **12**（+ tuneup_time / experiment_count / transfer_success / data_extract_acc / safety_violations / physicist_rating） |
| 投稿目标 | 系统会议 + Datasets&Benchmarks（2 篇） | **+ PRX Quantum / Nature Commun.**（3 篇） |
| 安全机制 | 无 | dry-run / power_budget / 二级确认 / audit log / rollback |
| 第一周清单 | 5 项 | 8 项（+ pulse 仿真 + lab 工具 + 物理组邮件） |

---

## 附录 B · Lab 工具矩阵详表

> 35 个工具的完整列表。每个工具列出：组别、名字、输入、输出、底层实现、Phase 引入。

### B.1 感知组（v1 已做，6 个）

| # | 工具 | 输入 | 输出 | 实现 | Phase |
|---|---|---|---|---|---|
| 1 | `get_backend_health` | backend | fidelity / queue / age | LabBackend.get_device_state | 1 |
| 2 | `get_qubit_properties` | backend, qubits | T1/T2/readout/gate err | 同上 | 1 |
| 3 | `get_coupling_map` | backend | graph | 同上 | 1 |
| 4 | `detect_drift` | backend, window | drift score + 建议 | PELT / BOCP | 1 |
| 5 | `get_calibration_age` | backend | minutes | properties timestamp | 1 |
| 6 | `compare_backends` | problem, candidates | ranked list | 多 backend properties 对比 | 1 |

### B.2 门级组（v1 已做，4 个）

| # | 工具 | 输入 | 输出 | 实现 | Phase |
|---|---|---|---|---|---|
| 7 | `transpile_circuit` | circ, backend, opt_level | transpiled circ | Qiskit transpile | 2 |
| 8 | `simulate` | circ, backend, shots | counts / fidelity | Qiskit Aer | 2 |
| 9 | `apply_mitigation` | circ, method | mitigated estimator | Mitiq (ZNE / PEC / CDR) | 2 |
| 10 | `predict_fidelity` | circ, backend | scalar | GNN / lookup | 2-3 |

### B.3 校准 / Tune-up 组（v2 新增，7 个）

| # | 工具 | 输入 | 输出 | 实现 | Phase |
|---|---|---|---|---|---|
| 11 | `cavity_spectroscopy` | backend, qubit, freq_range | resonance freq | Qiskit Pulse + Qiskit Experiments | 2 |
| 12 | `qubit_spectroscopy` | backend, qubit, freq_range | qubit freq | 同上 | 2 |
| 13 | `rabi_experiment` | backend, qubit, amp_range | π-amplitude | Qiskit Experiments | 2 |
| 14 | `ramsey_experiment` | backend, qubit, delays | T2*  /  detuning | 同上 | 2 |
| 15 | `drag_calibration` | backend, qubit | DRAG α | 自研 + Qiskit Experiments | 2 |
| 16 | `randomized_benchmarking` | backend, qubits, lengths | error per Clifford | Qiskit Experiments / Cirq | 2 |
| 17 | `gate_set_tomography` | backend, qubits | error process matrix | pyGSTi | 3 |

### B.4 脉冲优化组（v2 新增，4 个）

| # | 工具 | 输入 | 输出 | 实现 | Phase |
|---|---|---|---|---|---|
| 18 | `grape_optimize` | target unitary, Hamiltonian | pulse schedule | QuTiP optimal control | 2 |
| 19 | `crab_optimize` | target unitary, Hamiltonian | pulse schedule | QuTiP CRAB | 2 |
| 20 | `pulse_shaper` | target gate, constraints | shaped pulse | C3-Toolset | 2 |
| 21 | `derivative_removal` | pulse | DRAG-corrected pulse | 标准 DRAG | 2 |

### B.5 数据解读组（v2 新增，5 个）

| # | 工具 | 输入 | 输出 | 实现 | Phase |
|---|---|---|---|---|---|
| 22 | `fit_rabi` | IQ trace | π-amp / Rabi freq | scipy + lmfit | 2 |
| 23 | `fit_ramsey` | IQ trace | T2* / detuning | 同上 | 2 |
| 24 | `extract_t1_t2` | decay traces | T1 / T2 | exponential fit | 2 |
| 25 | `iq_classifier_train` | labeled IQ data | classifier | LDA / 浅层 NN | 1-2 |
| 26 | `parse_oscilloscope` | waveform file | structured signal | scipy.signal | 2 |

### B.6 实验设计组（v2 新增，3 个）

| # | 工具 | 输入 | 输出 | 实现 | Phase |
|---|---|---|---|---|---|
| 27 | `next_best_experiment` | current state | suggested experiment | botorch (BO) | 3 |
| 28 | `bayesian_design` | prior + data | posterior + design | GPyOpt | 3 |
| 29 | `info_gain_estimate` | candidate exp | expected info gain | KL divergence | 3 |

### B.7 跨设备组（v2 新增，2 个）

| # | 工具 | 输入 | 输出 | 实现 | Phase |
|---|---|---|---|---|---|
| 30 | `transfer_calibration` | src cal, dst device | adapted cal | 物理参数变换 + 局部 fine-tune | 3 |
| 31 | `port_pulse_schedule` | src schedule, dst device | dst schedule | pulse 重采样 + scaling | 3 |

### B.8 安全护栏组（v2 新增，4 个）

| # | 工具 | 输入 | 输出 | 实现 | Phase |
|---|---|---|---|---|---|
| 32 | `dry_run` | tool call | simulated result | 装饰器 + 仿真后端 | 0 |
| 33 | `power_budget_check` | pulse | pass / fail + reason | 硬限规则 | 0 |
| 34 | `amplitude_check` | pulse | pass / fail | 同上 | 0 |
| 35 | `audit_log` | tool call | log entry | DuckDB + hash chain | 0 |

---

## 附录 C · 标准 Tune-Up Sequence 范本

> 这是 Phase 2 demo 必须跑通的端到端流程。参考 IBM / Qiskit Experiments 的标准范式 + 物理组实际经验。

```
INPUT  · 一台超导芯片，Qubit Q0 状态未知（可能完全失谐）
GOAL   · 把 Q0 调到能跑高保真单 / 双比特门的状态
TARGET · 1Q gate fidelity ≥ 0.999, 2Q gate fidelity ≥ 0.99

══════════════════════════════════════════════════════════
                  AGENT REASONING TRACE
══════════════════════════════════════════════════════════

STEP 1 · Cavity Spectroscopy
├─ THOUGHT: 先确认 readout 谐振腔频率，否则后续都白做。
├�� TOOL:    cavity_spectroscopy(qubit=0, freq_range=[7.0, 7.5] GHz)
├─ OBSERVE: peak at 7.234 GHz
└─ NEXT:    locked cavity_freq, move on.

STEP 2 · Qubit Spectroscopy
├─ THOUGHT: 找比特频率，先粗扫描后细扫描。
├─ TOOL:    qubit_spectroscopy(qubit=0, freq_range=[4.5, 5.5] GHz, coarse=True)
├─ OBSERVE: candidate at 4.87 GHz
├─ TOOL:    qubit_spectroscopy(qubit=0, freq_range=[4.85, 4.89] GHz, coarse=False)
├─ OBSERVE: peak at 4.873 GHz
└─ NEXT:    locked qubit_freq.

STEP 3 · Rabi
├─ THOUGHT: 现在确定 π-脉冲幅度。
├─ TOOL:    rabi_experiment(qubit=0, amp_range=[0, 1.0])
├─ OBSERVE: IQ traces (raw)
├─ TOOL:    fit_rabi(traces)
├─ OBSERVE: π-amp = 0.42, Rabi freq = 25 MHz
└─ NEXT:    π-pulse calibrated.

STEP 4 · T1 / T2 Characterization
├─ TOOL:    extract_t1_t2(qubit=0)
├─ OBSERVE: T1 = 84 µs, T2 = 58 µs
└─ DECISION: T1/T2 within healthy range, continue.

STEP 5 · Ramsey (refine frequency)
├─ TOOL:    ramsey_experiment(qubit=0, delays=[0, 200] µs)
├─ OBSERVE: detuning = -120 kHz
├─ ACTION:  update qubit_freq → 4.873 - 0.00012 = 4.87288 GHz
└─ NEXT:    frequency refined.

STEP 6 · DRAG Calibration
├─ THOUGHT: 抑制 leakage 到 |2⟩。
├─ TOOL:    drag_calibration(qubit=0)
├─ OBSERVE: optimal α = 0.42
└─ NEXT:    DRAG locked.

STEP 7 · Randomized Benchmarking
├─ THOUGHT: 验证整体 1Q 门保真度。
├─ TOOL:    randomized_benchmarking(qubit=0, lengths=[1,2,4,8,16,32,64])
├─ OBSERVE: error per Clifford = 0.0008  (fidelity = 0.9992)
├─ DECISION: 达标 (≥ 0.999 wait, 0.9992 < 0.999? 等等，0.9992 ≥ 0.999, ✓)
└─ NEXT:    1Q tune-up complete.

STEP 8 · Update Calibration Library
├─ TOOL:    update_calibration({
│              "Q0_freq": 4.87288 GHz,
│              "Q0_pi_amp": 0.42,
│              "Q0_DRAG_alpha": 0.42,
│              "Q0_T1": 84e-6, "Q0_T2": 58e-6,
│              "Q0_1Q_fidelity": 0.9992,
│           })
├─ SAFETY:  二级确认（人类 OK）→ audit_log 记录
└─ DONE.

══════════════════════════════════════════════════════════
ELAPSED   · 6 minutes (vs 30 min 人工)
DECISIONS · 8 步骤, 11 工具调用
ARTIFACTS · 1 ExperimentRecord 入库
══════════════════════════════════════════════════════════
```

**这个 trace 就是 Phase 2 的 demo 视频脚本，也是 paper 主图的执行示例**。

---

*文档版本：v2.0 · 2026-05*
*v1 文档：`QuantumGPT_Plan.md`（保留作为对照）*
*下一步：基于本文档进行计划商讨 → 锁定 [十五节](#十五待决策的-6-个问题) 的 6 个决策点 → 启动 Day 1*








