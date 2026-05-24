# QuantumGPT 项目工程解读 & 可展示材料指南

> 生成日期: 2026-05-20
> 代码统计: 88 个 Python 文件, ~20,540 行代码, 103 张可视化图 (PNG/PDF)

---

## 一、项目架构总览

```
quantumgpt/
├── backends/          ← 影子硬件后端 (核心抽象层)
│   ├── base.py                 ShadowBackend 抽象基类 (5 接口)
│   ├── fake_adapter.py         FakeBackendAdapter (IBM FakeBrisbane 127q 等)
│   ├── replay_backend.py       ReplayBackend (历史校准数据回放)
│   ├── synthetic_drift.py      SyntheticDriftBackend (可控漂移注入)
│   ├── properties_stream.py    PropertiesStream (实时遥测流)
│   ├── calibration_data.py     CalibrationSnapshot 数据模型
│   ├── lab_backend.py          LabBackend 抽象 (脉冲级 5 原语)
│   ├── dynamics_lab_adapter.py DynamicsLabAdapter (Qiskit Dynamics 脉冲仿真)
│   ├── fake_lab_adapter.py     FakeLabAdapter (门级→LabBackend 桥接)
│   └── replay.py               旧版 replay 兼容
│
├── agent/             ← Agent 核心 (ReAct 循环 + 扩展)
│   ├── react.py                ReAct Agent (LLM tool-calling 循环)
│   ├── baselines.py            4 个 baseline 系统 (Static/LLMSingle/ReActNoBudget/ReActFull)
│   ├── budget.py               Fidelity Budget 门控
│   ├── drift_aware.py          DriftMonitor + ReplanningPolicy + DriftAwareRulePlanner
│   ├── memory.py               AgentMemory (ExperimentRecord 集成)
│   └── loop.py                 Agent 主循环入口
│
├── tools/             ← 14 个量子工具 (MCP 兼容 JSON Schema)
│   └── quantum_tools.py        ToolExecutor + 全部工具定义
│
├── experiments/       ← 脉冲级实验协议
│   ├── base.py                 ExperimentProtocol 抽象
│   ├── rabi_experiment.py      Rabi 振荡 (π-pulse 校准)
│   ├── ramsey_experiment.py    Ramsey 干涉 (T2* / 失谐)
│   └── t1_experiment.py        T1 弛豫 (能量衰减)
│
├── detection/         ← 漂移检测
│   ├── drift_detector.py       PELT + BOCPD 变点检测
│   └── evaluate.py             检测器评估
│
├── advisor/           ← 自主建议系统
│   ├── advisor.py              CalibrationAdvisor (3 阶段: MONITOR→DIAGNOSE→ACT)
│   └── streaming.py            StreamingAdvisor (实时遥测 + 告警引擎)
│
├── data/              ← 数据持久化
│   ├── store.py                DuckDB 5 表 schema
│   ├── experiment_record.py    ExperimentRecord (27 字段结构化记忆)
│   └── instrumented.py         Instrumented Executor (自动落库)
│
├── dynamics/          ← 脉冲级仿真
│   └── rabi.py                 Qiskit Dynamics Rabi 振荡模拟器
│
├── bench/             ← 基准电路
│   ├── circuits.py             5 种手写电路 (GHZ/QFT/BV/VQE/QAOA)
│   └── mqtbench.py             MQTBench 集成 (34 种 benchmark)
│
├── eval/              ← 评测框架
│   ├── run_eval.py             7 任务 × 4 系统端到端评测
│   ├── drift_experiment.py     漂移实验
│   └── drift_experiment_enhanced.py  增强版漂移实验
│
├── cli.py             ← MVP CLI (qgpt 命令)
├── api/main.py        ← FastAPI REST + WebSocket
├── web/app.py         ← Streamlit Web Dashboard
├── console/           ← Rich 终端仪表盘
│
├── demos/day1-14/     ← 每日 demo 文件夹
│   └── gen_figures.py + summary.md + 可视化图
│
├── tests/             ← 单元测试 + 集成测试
│   ├── test_drift_aware.py
│   ├── test_memory_integration.py
│   └── unit/ (11 个测试文件)
│
└── pyproject.toml     ← 包配置 + CLI 入口点
```

---

## 二、核心模块工程细节

### 2.1 Backend 层 (backends/)

**设计模式**: 策略模式 + 适配器模式

```
ShadowBackend (抽象基类)
├── FakeBackendAdapter     → 包装 Qiskit FakeBackend V2, 提供 127q 噪声模拟
├── ReplayBackend          → 回放历史校准数据序列, 模拟时间推移
└── SyntheticDriftBackend  → DriftProfile 函数式漂移, 可控注入

LabBackend (抽象基类, v2 升级)
├── DynamicsLabAdapter     → Qiskit Dynamics 脉冲级仿真 (单 qubit transmon)
└── FakeLabAdapter         → 门级 FakeBackend 桥接到 LabBackend 协议
```

**关键接口**:
- `ShadowBackend`: name / num_qubits / get_health() / get_qubit_properties() / run() / get_properties_snapshot()
- `LabBackend`: dispatch_pulse() / run_circuit() / read_measurement() / update_calibration() / get_device_state()

**SyntheticDriftBackend 漂移模型**:
- DriftProfile 数据类: t1_drift / t2_drift / gate_error_drift / readout_drift 均为 `Callable[[float], float]`
- 预设: STABLE / SUDDEN_DEGRADATION / LINEAR_DECAY / PERIODIC_OSCILLATION
- 支持 `set_time(hours)` 精确控制时间点

### 2.2 Agent 层 (agent/)

**ReActAgent** (agent/react.py):
- 支持 3 种模式: mock (规则规划器) / openai / anthropic
- Fidelity Budget 门控: 每步检查是否达标, 提前终止
- 输出 AgentTrace (含 TraceStep 列表, 每步记录 thought/action/observation/fidelity)

**4 个 Baseline** (agent/baselines.py):
1. StaticPipeline — 固定序列 (health → run → report), 无推理
2. LLMSingleShot — 一次 LLM 调用规划全部步骤, 然后执行
3. ReActNoBudget — ReAct 循环但无 fidelity 门控
4. ReActFull — 完整 ReAct + Budget (从 agent/react.py 导入)

**DriftAwareAgent** (agent/drift_aware.py):
- DriftMonitor: 每 N 步检查后端属性变化, 维护 DriftState
- ReplanningPolicy: 决定何时 invalidate + 重新规划
- DriftAwareRulePlanner: 漂移时覆盖基础规划器

**AgentMemory** (agent/memory.py):
- 自动将 AgentTrace → ExperimentRecord 并存储
- 提供 retrieve_past_experiments 工具供 agent 查询历史
- 消融开关: enabled=True/False

### 2.3 工具层 (tools/quantum_tools.py)

14 个工具, 每个有 Anthropic 兼容 JSON Schema:

| # | 工具名 | 功能 |
|---|--------|------|
| 1 | get_backend_health | 设备健康摘要 (T1/T2/errors/drift) |
| 2 | get_qubit_properties | 指定 qubit 详细属性 |
| 3 | run_circuit | 运行基准电路, 返回 fidelity + counts |
| 4 | list_benchmarks | 列出可用电路 |
| 5 | diagnose_and_suggest | 分析健康数据, 输出建议 |
| 6 | apply_mitigation | ZNE 误差缓解 |
| 7 | rabi_experiment | Rabi 振荡实验 |
| 8 | fit_rabi | 拟合 Rabi 数据, 提取 π-amp |
| 9 | ramsey_experiment | Ramsey 干涉实验 |
| 10 | t1_experiment | T1 弛豫实验 |
| 11 | check_drift_since | 检查漂移 |
| 12 | retrieve_past_experiments | 检索历史实验 |
| 13 | get_coupling_map | 耦合图 |
| 14 | compare_backends | 多后端对比 |

### 2.4 数据层 (data/)

**DuckDB 5 表**:
- health_snapshots: 设备健康时序
- qubit_properties: qubit 级别跟踪
- circuit_runs: 电路执行结果
- agent_sessions: Agent 运行元数据
- drift_events: 漂移诊断事件

**ExperimentRecord** (27 字段):
- Identity: id / timestamp / experiment_type
- Context: backend / circuit_name / num_qubits / problem / tags
- Plan: plan (步骤列表)
- Execution: tool_calls / decisions / elapsed_seconds / model / total_tokens
- Results: fidelity / success / outcome / metrics / raw_data_ref
- Fitting: fits (Rabi/T1/T2 拟合结果)
- Summary: 用于 RAG 检索的文本摘要

### 2.5 检测层 (detection/)

**DriftDetector**:
- PELT (Pruned Exact Linear Time): 离线变点检测, 给定 penalty 找最优变点
- BOCPD (Bayesian Online Changepoint Detection): 在线流式处理
- 输入: 多维信号 [mean_t1, mean_t2, mean_readout_error, mean_1q_error, mean_2q_error]
- 输出: DriftReport (changepoints / drift_scores / recalibrate_recommended)

### 2.6 实验协议层 (experiments/)

每个实验遵循统一接口: `run(backend) → analyze() → visualize()`

- **RabiExperiment**: 扫描驱动幅度, cos 拟合, 提取 π-pulse 幅度
- **T1Experiment**: 扫描延迟时间, 指数衰减拟合, 提取 T1
- **RamseyExperiment**: 扫描延迟, 阻尼振荡拟合, 提取 T2* + 失谐

---

## 三、可展示材料清单 & 使用方法

### 3.1 CLI 演示 (最快上手)

```bash
# 安装 (editable mode)
cd /home/wangshuchang/quantumgpt
pip install -e .

# 运行电路仿真
qgpt simulate ghz --shots 8192 --backend FakeBrisbane
qgpt simulate qft4 --backend FakeKyiv

# 查看后端健康
qgpt health --backend FakeBrisbane

# 列出可用电路
qgpt list

# 运行诊断
qgpt diagnose

# Agent 模式 (需要 LLM API)
qgpt agent "Run GHZ-5 and diagnose" --backend FakeBrisbane
```

### 3.2 Web Dashboard (Streamlit)

```bash
streamlit run web/app.py --server.port 8501
```
4 个页面: Dashboard / Experiment Lab / Device Monitor / History
交互式 Plotly 图表, 可运行 Rabi/Ramsey/T1 实验

### 3.3 FastAPI 后端

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- REST: /api/health, /api/backends, /api/experiments/run, /api/experiments/history
- WebSocket: /ws/telemetry (实时遥测流)

### 3.4 终端仪表盘 (Rich Console)

```bash
python -m console --profile sudden --hours 168 --speed 100
python -m console --profile stable --hours 24
```
实时显示漂移遥测、变点检测、告警

### 3.5 评测系统

```bash
# 7 任务 × 4 系统端到端评测
python eval/run_eval.py

# 漂移实验 (论文核心)
python eval/drift_experiment.py
python eval/drift_experiment_enhanced.py
```

### 3.6 单元测试

```bash
pytest tests/ -v
# 或单独运行:
python tests/test_drift_aware.py
python tests/test_memory_integration.py
```

### 3.7 Demo 可视化 (103 张图)

每个 day 文件夹都有 gen_figures.py:

```bash
# 生成某天的图
python demos/day10/gen_figures.py   # Rabi 振荡曲线
python demos/day11/gen_figures.py   # 架构图 + 漂移 profiles
python demos/day14/gen_figures.py   # 实验套件 (Rabi/Ramsey/T1)

# 生成 Phase 0 总览图
python demos/gen_showcase.py
```

**重点可视化文件**:

| 文件 | 内容 | 用途 |
|------|------|------|
| demos/day10/rabi_curves.png | Rabi 振荡曲线 | 脉冲级能力展示 |
| demos/day10/pulse_shapes_bloch.png | 脉冲形状 + Bloch 球 | 物理直觉 |
| demos/day11/architecture.png | 系统架构图 | Paper Figure 1 |
| demos/day11/drift_profiles.png | 4 种漂移 profile | 漂移实验说明 |
| demos/day14/experiment_suite.png | Rabi+Ramsey+T1 三合一 | 实验协议展示 |
| demos/day6/agent_architecture.png | Agent 架构 | Agent 设计说明 |
| demos/day6/tool_call_flow.png | 工具调用流程 | 工具链展示 |
| demos/day8/changepoint_detection.png | 变点检测 | 漂移检测能力 |
| demos/day9/streaming_architecture.png | 流式架构 | 实时监控 |
| demos/day5/mqtbench_fidelity.png | MQTBench 保真度 | Benchmark 结果 |

### 3.8 脉冲级 Rabi Demo

```bash
python demos/rabi_demo.py
# 或直接调用:
python -c "
from dynamics.rabi import sweep_rabi, RabiConfig
cfg = RabiConfig(qubit_freq_ghz=5.0, drive_freq_ghz=5.0)
result = sweep_rabi(cfg, n_amps=30)
print(f'Pi-amp: {result.pi_amplitude:.4f}')
"
```

### 3.9 Day-by-Day Demo 运行

```bash
python demos/day6/run_agent_demo.py    # Agent 端到端
python demos/day7/run_day7_demo.py     # W&B + DuckDB 数据管线
python demos/day8/run_day8_demo.py     # CalibrationAdvisor
python demos/day9/run_day9_demo.py     # StreamingAdvisor
python demos/day10/run_day10_demo.py   # Rabi 脉冲仿真
python demos/day11/run_day11_demo.py   # 多后端验证
```

---

## 四、已完成 Phase 对照

| Phase | 周次 | 状态 | 核心产出 |
|-------|------|------|----------|
| Phase 0 | W1-W2 | ✅ 完成 | 项目脚手架 + ShadowBackend + FakeBackend + CLI + DuckDB |
| Phase 1 | W3-W4 | ✅ 完成 | 14 工具 + DriftDetector + PropertiesStream + Console |
| Phase 2 | W5-W6 | ✅ 大部分完成 | ReAct Agent + 4 Baseline + 7×4 评测 + Rabi 脉冲 + ExperimentRecord schema |
| Phase 3 | W7-W9 | 🔄 进行中 | ExperimentRecord 集成 ✅ / Drift-Aware Replanning ✅ / Fidelity Predictor ❌ / 全面对比 ❌ |
| Phase 4 | W10-W11 | ❌ 未开始 | QC-Agent-Bench 60 任务 + 5 系统全面对比 |
| Phase 5 | W12+ | ❌ 未开始 | 论文写作 + 投稿 (ICLR 2027, deadline ~2026-10-03) |

---

## 五、下一步计划 (Phase 3 剩余 + Phase 4 启动)

### 5.1 Phase 3 剩余任务 (预计 4-5 天)

| 优先级 | 任务 | 预计时间 | 说明 |
|--------|------|----------|------|
| P0 | Fidelity Predictor (XGBoost/LightGBM) | 2 天 | 收集训练数据 → 特征工程 → 训练 → 评估 MAE/R² |
| P0 | 6 系统 × 7 任务全面对比表 | 2 天 | Paper Table 1 数据; 含消融 (no-mem / no-drift) |
| P1 | Demo 视频脚本 | 1 天 | drift → detect → replan → recover 完整流程录制 |

### 5.2 Phase 4: QC-Agent-Bench (预计 7-10 天)

| 步骤 | 任务 | 说明 |
|------|------|------|
| 4.1 | 定义 60 个任务 JSON spec | Tier 1 (30 静态) + Tier 2 (20 漂移) + Tier 3 (10 故障) |
| 4.2 | 实现 ground truth 生成器 | 每个任务的正确答案 / 成功判据 |
| 4.3 | 5 系统 × 60 任务 × 6 指标 | 完整评测矩阵 |
| 4.4 | Benchmark spec PDF | 任务描述 + 评测协议 + 使用说明 |
| 4.5 | GitHub 仓库整理 | README + 安装 + 复现脚本 |
| 4.6 | Leaderboard 页面 (可选) | 简单 HTML 或 HuggingFace Space |

### 5.3 Phase 5: 论文写作 (W12+, 预计 4-6 周)

| 步骤 | 任务 | Deadline |
|------|------|----------|
| 5.1 | Paper 骨架 (LaTeX) | W12 Day 1 |
| 5.2 | Figure 1 (��构图) + Figure 2 (消融) | W12 Day 3 |
| 5.3 | Table 1 (主对比表) | W12 Day 5 |
| 5.4 | 正文初稿 (9 页) | W13 |
| 5.5 | 内审 + 修改 | W14 |
| 5.6 | Camera-ready + 投稿 | W15 (≤ 2026-10-03) |

### 5.4 关键时间节点

```
现在:     2026-05-20 (W1 Day 3)
Phase 3 完成: ~2026-05-25
Phase 4 完成: ~2026-06-05
代码 freeze: ~2026-08-07
论文初稿:    ~2026-08-21
投稿:        ~2026-10-03 (ICLR 2027)
```

### 5.5 立即可做的 3 件事

1. **跑通 Fidelity Predictor**: 用现有 eval/run_eval.py 收集 (circuit_features, backend_state, fidelity) 数据, 训练 XGBoost
2. **补全 6 系统对比表**: 在 eval/ 下新增脚本, 跑 StaticPipeline / LLMSingleShot / ReActNoBudget / ReActFull / ReActFull+Memory / ReActFull+DriftAware 六个系统
3. **开始定义 QC-Agent-Bench 任务 spec**: 先写 Tier 1 的 30 个静态任务 JSON

---

## 六、风险提醒

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM API 不稳定 (DeepSeek 403) | Agent 模式无法运行 | mock 模式 (规则规划器) 可独立评测 |
| 评测时间过长 | 延误 Phase 4 | 结果缓存 + 并行跑 |
| Qiskit Dynamics 版本冲突 | 脉冲级功能受限 | 已锁定 qiskit==1.3.0 + dynamics==0.6.0 |
| 物理组合作未启动 | 少一条投稿线 | 不影响主线 paper (ICLR) |
| ICLR deadline 提前 | 时间压缩 | 保持 W12 代码 freeze 不动摇 |

---

*文档路径: /home/wangshuchang/quantumgpt/PROJECT_GUIDE.md*
