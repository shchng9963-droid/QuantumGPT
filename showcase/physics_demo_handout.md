# QuantumGPT 物理实验合作一页说明

## 我们在做什么

QuantumGPT 是一个面向量子实验室的 AI 实验副驾。当前阶段不是让 AI 直接控制真机，而是先做 read-only / dry-run：帮助学生和老师自动完成设备状态查看、标准表征流程、参数拟合、实验记录和漂移提醒。

## 当前 demo 能展示什么

1. Rabi tune-up copilot
   - 输入自然语言任务；
   - 自动运行 Rabi 仿真实验；
   - 拟合 pi pulse 和 pi/2 pulse amplitude；
   - 生成图、报告和 agent trace。

2. Drift-aware agent trace
   - 记录 backend health snapshot；
   - 追踪 circuit result 依赖哪些设备状态；
   - 当设备漂移后，把旧结果标记为 stale；
   - 强制重新检查设备并 rerun。

3. ExperimentRecord
   - 每次实验记录任务、工具调用、结果、拟合参数和经验总结；
   - 目标是形成组内可检索的实验记忆库。

## 为什么是安全的

- 当前 demo 在仿真/历史回放后端上运行；
- lab/pulse/calibration 动作默认 dry-run；
- 写 calibration 或影响设备的动作保留人工确认；
- 所有步骤都有 trace/audit log；
- 可以本地部署，不需要上传实验数据。

## 我们希望从物理组获得什么

最低负担版本：

1. 30 分钟访谈：了解实际 tune-up 流程和痛点；
2. 一份脱敏历史数据样例：用于离线回放和自动报告；
3. 对 read-only 工具的反馈：哪个最有用，哪个不可信。

## 可选合作方向

- Rabi/Ramsey/T1 自动拟合报告；
- 设备漂移提醒；
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
