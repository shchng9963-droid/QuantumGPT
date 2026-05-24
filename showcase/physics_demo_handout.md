# QuantumGPT v2.1 物理实验合作一页说明

## 我们在做什么

QuantumGPT 是一个面向量子实验室的 AI 实验副驾。当前阶段不是让 AI 直接控制真机，而是先做 read-only / dry-run：帮助学生和老师自动完成设备状态查看、标准表征流程、参数拟合、实验记录和漂移提醒。

## v2.1 新增能力

### 1. Memory 因果闭环 (Stage C)
- Agent 会记住上一次实验结果；
- 若上次 fidelity 不达标，自动触发 `diagnose_and_suggest`；
- Diagnose 返回 actionable overrides（shots ↑, optimization_level ↑, initial_layout）；
- 下一次 `run_circuit` 自动消费这些 overrides，无需人工干预。

**Ablation 实验验证 (P1-6, 90 runs):**

|            | memory_off | memory_on |
|------------|-----------|-----------|
| fid_mean   | 0.9512    | 0.9540    |
| fid_std    | 0.0331    | 0.0255    |
| diagnose%  | 0%        | 66.7%     |

memory_on 逐 trial 递增: 0.947→0.956→0.959 (diagnose 在 trial2-3 100% 触发)

### 2. Safety 增强 (Stage D)
- Pulse amplitude 超限 → 自动 block，不执行；
- Safety violation 作为 `SAFETY_VIOLATION` artifact 记入 trace；
- 每个 final answer 标注 `simulation — no physical hardware was modified`；
- 7 个 safety 单元测试全部覆盖。

### 3. Drift-aware 三阶段 Demo
- Phase 1: 正常设备 → fidelity ~0.91
- Phase 2: 突发退化 → fidelity 降至 ~0.77 (SUDDEN_DEGRADATION)
- Phase 3: Memory 注入 → agent 检测退化并自动调参

## 当前 demo 能展示什么

1. **Rabi tune-up copilot** — 自然语言 → Rabi 仿真 → 拟合 pi/pi2 pulse → 图 + 报告
2. **Drift-aware agent** — 设备健康 → 漂移检测 → 标记 stale → 重跑
3. **Memory-aware agent** — 冷跑 vs 热跑 vs 控制组，因果链可视化
4. **Safety demo** — 超限 pulse → block → artifact in trace
5. **ExperimentRecord** — 可检索的实验记忆库

一键运行: `bash showcase/run_all_demos.sh`

## 为什么是安全的

- 当前 demo 在仿真/历史回放后端上运行；
- lab/pulse/calibration 动作默认 dry-run；
- 写 calibration 或影响设备的动作保留人工确认；
- 超限参数自动 block，violation 记入 trace artifact；
- final answer 显式标注 simulation/dry-run 模式；
- 所有步骤都有 trace/audit log；
- 可以本地部署，不需要上传实验数据。

## 我们希望从物理组获得什么

最低负担版本：

1. 30 分钟访谈：了解实际 tune-up 流程和痛点；
2. 一份脱敏历史数据样例：用于离线回放和自动报告；
3. 对 read-only 工具的反馈：哪个最有用，哪个不可信。

## 可选合作方向

- Rabi/Ramsey/T1 自动拟合报告；
- 设备漂移提醒 + memory-informed 自动调参；
- 历史实验记录检索；
- read-only notebook assistant；
- 后续接入 QCoDeS / pyqum / Labber adapter。

## 当前系统边界

我们不承诺：
- 直接自动控制真机；
- 替代实验人员判断；
- 一开始覆盖完整 tune-up chain；
- 没有验证就写入校准库。

我们承诺：
- 从 read-only / dry-run 开始；
- 先做一个有用的小流程；
- 任何写操作都需要人工确认；
- 结果和中间过程可追踪、可复现。

## 技术栈

Qiskit 1.x + Aer | FakeBackend | ReAct agent (rule-based mock + LLM) | DuckDB memory | ZNE mitigation | SyntheticDriftBackend | SafetyPolicy + SafetyAwareExecutor

47 unit tests passing | 3 demo scripts | 1 ablation experiment (90 runs)
