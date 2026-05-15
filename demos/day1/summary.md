# Day 1 — 项目脚手架 + ShadowBackend 抽象

**日期**: 2026-05-15  
**Commit**: `27214c9`  
**完成标志**: `pip list` 无报错，项目结构就绪

---

## 交付件

### 1. Monorepo 项目结构

```
quantumgpt/
├── backends/    ← 影子硬件后端（核心）
├── tools/       ← Agent 工具定义
├── agent/       ← Agent loop（待实现）
├── bench/       ← 基准电路
├── eval/        ← 评估框架
├── tests/       ← 单元/集成测试
├── scripts/     ← 运行脚本
├── configs/     ← 配置文件
├── data/        ← 校准数据/电路/结果
├── docs/        ← 文档
└── notebooks/   ← 实验笔记本
```

完整目录树见 `project_tree.txt`。

### 2. ShadowBackend 抽象基类

所有影子后端的统一接口（`backends/base.py`）：

| 方法 | 用途 |
|------|------|
| `name` | 后端名称 |
| `num_qubits` | 量子比特数 |
| `get_health()` | 返回 BackendHealth 摘要 |
| `get_qubit_properties(qubits)` | 指定比特的 T1/T2/gate error |
| `get_coupling_map()` | 耦合图 |
| `run(circuit, shots)` | 转译+模拟+返回 SimulationResult |
| `get_properties_snapshot()` | 完整校准快照 dict |

### 3. 三个核心数据类

- **BackendHealth**: avg_1q_error, avg_2q_error, avg_t1_us, drift_score...
- **QubitProperties**: 单量子比特 T1/T2/readout_error/gate_errors
- **SimulationResult**: counts + shots + fidelity + metadata

### 4. 环境配置

```
Python 3.11.15
qiskit==1.4.2
qiskit-aer==0.17.0
qiskit-ibm-runtime==0.37.0
mitiq==0.42.0
numpy, pytest, etc.
```

---

## 设计决策

- **不用 Qiskit Provider 接口**：ShadowBackend 更精简，只暴露 agent 需要的信息
- **数据类而非 dict**：类型安全，IDE 补全，单测清晰
- **drift_score 字段**：为后续漂移感知预留，所有后端统一报告
