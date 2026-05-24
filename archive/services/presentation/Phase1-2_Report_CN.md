# QuantumGPT Phase 1-2 进展报告

## 1. 项目简介

QuantumGPT 是一个 LLM 驱动的自主量子计算 Agent 系统。它能够自主监控量子设备、运行电路、诊断问题、应用误差缓解，并执行脉冲级量子比特表征实验。

**目标会议:** ICLR 2027 (截稿 ~2026-10-03)

---

## 2. Phase 1 完成内容

### 2.1 Shadow Backend (影子后端)

- 基于 Qiskit FakeBrisbane 构建 127-qubit 模拟后端
- 完整噪声模型: 门错误、读出错误、T1/T2 退相干
- 合成漂移注入: 模拟真实设备的校准漂移
- 统一接口 `ShadowBackend` 抽象基类

### 2.2 量子工具集 (9 个基础工具)

| 工具 | 功能 |
|------|------|
| `get_backend_health` | 获取后端整体健康状态 |
| `get_qubit_properties` | 查询指定量子比特的 T1/T2/错误率 |
| `get_coupling_map` | 获取耦合拓扑 |
| `detect_drift` | 检测校准漂移 |
| `get_calibration_age` | 查询校准数据年龄 |
| `compare_backends` | 对比多个后端 |
| `run_circuit` | 运行量子电路并返回保真度 |
| `list_benchmarks` | 列出可用基准电路 |
| `diagnose_and_suggest` | 诊断问题并给出建议 |

### 2.3 Rabi 脉冲模拟

- 基于 Schrödinger 方程的数值求解器
- 支持 square 和 gaussian 脉冲形状
- 振幅扫描 + π-pulse 自动提取

### 2.4 Agent Loop v1

- While-loop 架构 + RulePlanner (离线测试)
- 支持 OpenAI / Anthropic / DeepSeek API
- InstrumentedExecutor + DuckDB 数据存储

---

## 3. Phase 2 完成内容

### 3.1 新增 5 个工具 (共 14 个)

| 新工具 | 功能 |
|--------|------|
| `transpile_circuit` | 转译电路，报告优化后深度和门数 |
| `apply_mitigation` | ZNE 零噪声外推误差缓解 |
| `predict_fidelity` | 不运行电路即预测保真度 |
| `rabi_experiment` | 运行 Rabi 振荡实验 |
| `fit_rabi` | 拟合 Rabi 数据提取 π-pulse 幅度 |

### 3.2 ReAct Agent

升级为 Thought → Action → Observation 结构化推理:

```
💭 Thought: 需要检查后端健康状态
🔧 Action: get_backend_health({})
👁 Observation: {backend: FakeBrisbane, avg_1q_error: 0.006...}
💭 Thought: 运行 GHZ-5 电路测量保真度
🔧 Action: run_circuit({circuit_name: "ghz_5"})
👁 Observation: {fidelity: 0.93, ...}
💭 Thought: 保真度达标，总结结果
```

### 3.3 Fidelity Budget Controller

- 跟踪目标保真��� (默认 0.85)
- 监控工具调用次数和时间预算
- 自动决策: PROCEED / MITIGATE / STOP / GOAL_MET
- 每轮将预算状态注入 LLM prompt

### 3.4 四系统对照实验

| 系统 | 描述 | 成功率 |
|------|------|--------|
| Static Pipeline | 固定序列，无推理 | 86% |
| LLM Single-Shot | 单次规划，盲执行 | 100% |
| ReAct (no budget) | ReAct 推理，无预算 | 100% |
| ReAct + Budget | 完整系统 | 100% |

**关键发现:** Static Pipeline 在需要诊断+缓解的任务上失败 (0%)，验证了 Agent 推理能力的必要性。

### 3.5 7×4 评测框架

7 个端到端任务:
1. Health Check + GHZ-5
2. Multi-Circuit Benchmark (3 电路对比)
3. Diagnose + Mitigate
4. Fidelity Prediction
5. Transpile + Run
6. Rabi Experiment
7. Full Pipeline (健康→运行→缓解→诊断)

### 3.6 Rabi 端到端 Demo

完整工作流: 自然语言 prompt → Agent 推理 → 脉冲模拟 → 数据拟合 → 报告

**结果:**
- π-pulse 幅度: 52.52 MHz
- π/2-pulse 幅度: 26.26 MHz
- 拟合 R²: 1.0000
- 总耗时: 11.3s, 4 次工具调用

---

## 4. 可运行 Demo

### 4.1 环境准备

```bash
cd /home/wangshuchang/quantumgpt
# 已有 .venv 环境，包含 qiskit, qiskit-aer, numpy, scipy, matplotlib
source .venv/bin/activate
```

### 4.2 运行命令

```bash
# Rabi 端到端 demo (生成图 + 报告)
PYTHONPATH=. python3 demos/rabi_demo.py

# 7×4 系统评测 (打印对比表)
PYTHONPATH=. python3 eval/run_eval.py

# ReAct Agent 交互
PYTHONPATH=. python3 -c "
from agent.react import ReActAgent
from backends.fake_adapter import FakeBackendAdapter
be = FakeBackendAdapter('FakeBrisbane')
agent = ReActAgent(be, provider='mock', target_fidelity=0.85)
trace = agent.run('检查后端健康，运行 GHZ-5 电路，如果保真度低于 0.95 则应用误差缓解')
"

# 单工具测试
PYTHONPATH=. python3 -c "
from tools.quantum_tools import ToolExecutor
from backends.fake_adapter import FakeBackendAdapter
ex = ToolExecutor(FakeBackendAdapter('FakeBrisbane'))
import json
print(json.dumps(json.loads(ex.execute('get_backend_health', {})), indent=2))
"
```

### 4.3 输出文件

| 文件 | 说明 |
|------|------|
| `demos/output/rabi_oscillation.png` | Rabi 振荡图 (Nature 风格) |
| `demos/output/rabi_oscillation.pdf` | 矢量版 |
| `demos/output/rabi_report.md` | 实验报告 |
| `demos/output/rabi_trace.json` | Agent trace 数据 |
| `eval_results.json` | 评测原始数据 |
| `presentation/QuantumGPT_Phase1-2.pptx` | 本 PPT |

---

## 5. 关键代码文件

```
agent/react.py         — ReAct Agent 主循环 (530 行)
agent/budget.py        — Fidelity Budget 控制器
agent/baselines.py     — 4 系统对照实现
agent/loop.py          — 基础 Agent (Phase 1)
tools/quantum_tools.py — 14 工具定义 + ToolExecutor (920 行)
backends/fake_adapter.py — Shadow Backend 适配器
dynamics/rabi.py       — Rabi 脉冲模拟器
mitigation/__init__.py — ZNE 误差缓解
eval/run_eval.py       — 评测框架
demos/rabi_demo.py     — Rabi demo 脚本
```

---

## 6. 下一步 (Phase 3)

1. 接入 DeepSeek API 运行完整 ReAct loop (非 mock)
2. 对比 LLM 推理 vs Rule-based 在复杂任务上的差异
3. 增加更多量子算法 benchmark
4. 实现 T1/T2 衰减实验 + Ramsey 干涉
5. 论文写作: 系统设计 + 实验结果 + 消融分析
