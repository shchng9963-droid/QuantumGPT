# 面向物理实验老师的 QuantumGPT Demo Playbook

> 目标：准备一个让物理实验口老师愿意继续聊的 5 分钟 demo。重点不是展示 LLM，而是展示“安全、可复现、能减少重复劳动的实验副驾”。

---

## 1. Demo 核心信息

一句话版本：

> QuantumGPT 目前是一个 read-only / dry-run 的量子实验副驾：它可以观察设备状态、执行仿真或历史回放、跑标准表征流程、拟合实验参数、生成结构化记录，并在设备漂移时提醒旧结论可能失效。

不要承诺：

- 不承诺直接闭环控制真机；
- 不承诺替代学生或老师；
- 不承诺完全自动 tune-up；
- 不承诺当前已经支持某组的全部软件栈。

可以承诺：

- 先从离线历史数据和仿真开始；
- 写权限默认关闭；
- 所有动作有 trace 和 audit；
- 后续可接 QCoDeS / pyqum / Labber adapter；
- 目标是减少重复拟合、诊断和记录工作。

---

## 2. 5 分钟展示流程

### 0:00-0:40 痛点开场

话术：

> 我们观察到实验组日常 tune-up 有很多重复步骤：看设备状态、跑 Rabi/Ramsey/T1、拟合参数、判断是否需要重校准、把结果记录到 notebook 或表格里。这个过程很依赖经验，也容易因为设备漂移导致旧判断失效。我们做 QuantumGPT，不是为了让 AI 直接控制设备，而是先做一个只读和 dry-run 的实验副驾。

### 0:40-1:20 系统边界和安全

话术：

> 现在系统默认在仿真或历史回放后端上运行。涉及 pulse 或 calibration 的动作都会先 dry-run，并且写入 audit log。未来如果接真机，写 calibration 这类动作会保持人工确认，不会让 agent 直接改设备。

需要展示的关键词：

- read-only
- dry-run
- audit log
- ExperimentRecord
- rollback-ready design

### 1:20-3:20 Demo A：Rabi tune-up copilot

运行命令：

```bash
cd /home/wangshuchang/quantumgpt
PYTHONPATH=. python3 demos/rabi_demo.py
```

展示文件：

```text
demos/output/rabi_oscillation.png
demos/output/rabi_report.md
demos/output/rabi_trace.json
```

讲解重点：

1. 输入自然语言任务：帮我做 Rabi 并拟合 pi pulse；
2. agent 自动拆成：rabi_experiment -> fit_rabi -> report；
3. 输出 pi amplitude、pi/2 amplitude、fit R^2；
4. 报告记录了模型、backend、工具调用、实验参数；
5. 这类记录未来可以进入组内 ExperimentRecord 库。

建议展示句：

> 这不是让 AI 猜一个参数，而是让它调用标准实验和标准拟合流程，并把中间步骤、结果和不确定性保留下来。

### 3:20-4:20 Demo B：漂移感知 agent trace

建议展示已有 drift-aware benchmark smoke 的 trace 或重新运行：

```bash
cd /home/wangshuchang/quantumgpt
python benchmark/runner.py --systems react_drift --tiers 1 --max-tasks 2 --provider mock --save-traces --output reports/baseline_week1/mock_react_drift_state_smoke.json
```

讲解重点：

1. backend health snapshot 是一个 artifact；
2. circuit result 依赖这个 snapshot；
3. 设备漂移后，旧 artifact 变 stale；
4. agent 不是继续沿用旧结论，而是重新 health check + rerun；
5. 这就是和普通 LLM tool calling 的区别。

建议展示句：

> 对实验来说，旧结论什么时候失效很重要。我们把这个问题显式建模了，而不是让 LLM 从上下文里猜。

### 4:20-5:00 合作请求

低负担请求：

> 我们希望先做一个非常低负担的合作：访谈 30 分钟，了解您组实际 tune-up 流程；如果方便，后续提供一小段脱敏历史数据，我们只做离线回放和报告生成。短期不会接写权限，也不会直接控制设备。

可以问老师的 5 个问题：

1. 您组最耗时间的 tune-up/校准步骤是哪几个？
2. 数据一般存在什么格式里？HDF5、csv、Labber、QCoDeS、pyqum 还是 notebook？
3. 学生通常如何记录一次 tune-up 的结果？
4. 您最担心 AI 接入实验系统的风险是什么？
5. 如果先做 read-only 工具，哪一个最有用：fit_rabi、fit_ramsey、drift alert、实验记录检索，还是自动报告？

---

## 3. Demo 前检查清单

运行环境：

```bash
cd /home/wangshuchang/quantumgpt
python -m pytest tests/unit/test_agent_state.py tests/unit/test_agent_state_integration.py tests/unit/test_lab_backend.py tests/unit/test_rabi.py -q
PYTHONPATH=. python3 demos/rabi_demo.py
```

确认文件存在：

```text
demos/output/rabi_oscillation.png
demos/output/rabi_report.md
demos/output/rabi_trace.json
```

确认报告里至少有：

- experiment parameters；
- pi pulse amplitude；
- pi/2 pulse amplitude；
- fit R^2；
- agent trace steps。

v2.1 后续要补：

- safety/dry-run section；
- ExperimentRecord ID；
- human-readable next-step suggestion；
- 30 秒 talk track。

---

## 4. 一页 handout 结构

标题：

> QuantumGPT: A Read-Only AI Copilot for Quantum Lab Tune-Up

三块内容：

1. What it does now
   - health check
   - Rabi simulation + fit
   - drift-aware trace
   - structured experiment record

2. Why it is safe
   - simulation/replay first
   - dry-run by default
   - audit log
   - human approval for write actions

3. What we need from collaborators
   - 30 min workflow interview
   - sample historical tune-up data
   - feedback on read-only report usefulness

---

## 5. 风险回答模板

### Q: 你们是不是要让 AI 自动控制设备？

答：

> 不是当前阶段目标。我们先做 read-only 和 dry-run，主要帮助诊断、拟合、记录和建议。写 calibration 或 dispatch pulse 到真机这类动作会保持人工确认。

### Q: 仿真结果和真机差距怎么办？

答：

> 我们不把仿真当最终证据。仿真用于开发 agent 行为和安全流程；真正有价值的是后续接历史数据回放，验证它是否能减少重复分析时间。

### Q: 物理实验流程很复杂，你们能覆盖吗？

答：

> 不会一开始覆盖全部。我们希望先挑一个最小但有用的流程，例如 Rabi/Ramsey 拟合或 drift alert，把一个点做扎实，再逐步扩展。

### Q: 数据安全怎么办？

答：

> 可以从完全脱敏的历史数据开始，只需要字段和波形，不需要设备身份信息。系统也可以本地部署，不上传数据。

---

## 6. v2.1 Demo 完成标准

- 5 分钟内讲完；
- 不需要解释代码细节；
- 老师能理解当前系统边界；
- 老师能指出一个可能有用的 read-only 场景；
- 老师愿意进行下一次 30 分钟 workflow 访谈。
