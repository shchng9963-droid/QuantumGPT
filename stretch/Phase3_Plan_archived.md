# QuantumGPT Phase 3 规划: 闭环 & 学习

> **时间**: W7–W9 (约 3 周)
> **核心目标**: 让 agent 从历史中学习 + 漂移下自适应 —— 这是论文的核心差异化

---

## 当前状态 (Phase 1-2 已完成)

| 模块 | 状态 |
|------|------|
| Shadow Backend (FakeBrisbane 127q) | ✅ |
| 14 个量子工具 | ✅ |
| ReAct Agent + Fidelity Budget | ✅ |
| Rabi 脉冲模拟 (Schrödinger) | ✅ |
| ZNE 误差缓解 | ✅ |
| 4 系统对照 + 7×4 评测 | ✅ |
| ExperimentRecord schema | ✅ (已定义，未集成到 agent loop) |
| PropertiesStream | ✅ (已实现) |
| DriftDetector | ✅ (已实现) |
| ReplayBackend | ✅ (已实现) |

---

## Phase 3 任务分解

### P3-T1: ExperimentRecord 集成到 ReAct Agent (2天)

**目标**: Agent 每次运行自动存储结构化实验记录，并能检索历史

**具体工作**:
1. 在 `ReActAgent._run_mock()` / `_run_openai()` / `_run_anthropic()` 结束时自动写入 ExperimentRecord
2. 新增工具 `retrieve_past_experiments(query, top_k)` — 从历史中检索相似实验
3. Agent 在规划时自动查询历史: "上次跑这个电路保真度多少？用了什么缓解？"
4. 消融开关: `use_memory=True/False`

**验证**: 跑 10 次 GHZ-5 → 第 11 次 agent 能引用历史数据做决策

---

### P3-T2: Drift-Aware Replanning (3天) ⭐ 论文核心实验

**目标**: Agent 检测到漂移后主动 invalidate 旧结果并重新规划

**具体工作**:
1. 在 ReAct loop 中集成 `PropertiesStream` 监听
2. 新增工具 `check_drift_since(timestamp)` — 检查自上次执行以来是否有漂移
3. 实现 replanning 逻辑:
   - 每次 tool call 前检查 drift score
   - 如果 drift > threshold → invalidate 之前的 transpile/fidelity 预测
   - 自动重新 transpile + 重新运行
4. 用 `SyntheticDriftBackend` 注入可控漂移:
   - t=0: 正常 (fidelity ~0.93)
   - t=5s: 注入 T1 衰减 (fidelity 降到 ~0.7)
   - 观察 agent 是否检测到并恢复

**评测指标**:
- `drift_recovery_steps`: 漂移注入后多少步恢复到目标保真度
- `drift_recovery_rate`: 在 N 步内恢复的比例
- 对比: ReAct+Budget (有 drift-aware) vs ReAct (无 drift-aware) vs Static

**验证**: 注入漂移后，完整系统在 3 步内恢复；baseline 永远不恢复

---

### P3-T3: Fidelity Predictor 升级 (2天)

**目标**: 从简单解析模型升级为数据驱动预测

**具体工作**:
1. 收集训练数据: 跑 Phase 2 的 5 个电路 × 多种噪声条件 × 多次 → 得到 (circuit_features, backend_state, real_fidelity) 三元组
2. 特征工程:
   - 电路特征: depth, n_2q_gates, n_qubits, connectivity
   - 后端特征: avg_1q_error, avg_2q_error, avg_readout_error, T1, T2, drift_score
3. 训练简单模型:
   - 方案 A: XGBoost/LightGBM (快速, 可解释)
   - 方案 B: 小 MLP (如果数据够多)
   - 方案 C: 保持 lookup table + 线性修正 (最保守)
4. 评估: 预测 vs 实际 fidelity 的 MAE / R²

**验证**: 预测误差 MAE < 0.05 (比当前解析模型的 ~0.1 好)

---

### P3-T4: 完整消融实验 (2天)

**目标**: 产出论文主表 — 6 系统 × 多任务 × 多指标

**6 个系统**:
1. Static Pipeline (无推理)
2. LLM Single-Shot (单次规划)
3. ReAct + Tools (标准 ReAct)
4. **QuantumGPT Full** (ReAct + Budget + Memory + Drift-aware)
5. QuantumGPT − Memory (消融: 去掉 ExperimentRecord)
6. QuantumGPT − Drift (消融: 去掉 drift-aware replanning)

**任务集扩展到 ~30 个**:
- Tier 1 (静态, 15个): 5 电路 × 3 噪声水平
- Tier 2 (漂移, 10个): 5 电路 × 2 漂移模式
- Tier 3 (故障, 3个): qubit 挂掉 / 读出错误飙升 / 耦合断开
- Tier 4 (设备级, 2个): Rabi tune-up 不同初始条件

**指标**:
- success_rate (任务完成率)
- avg_fidelity (平均保真度)
- tool_calls (效率)
- wall_time (延迟)
- drift_recovery_steps (漂移恢复速度)

**验证**: Full QuantumGPT 在 success_rate 上比 ReAct baseline 提升 >= 15%

---

### P3-T5: 真实 LLM 集成 + 对比 (2天)

**目标**: 用 DeepSeek API 跑完整 ReAct loop，对比 mock vs real LLM

**具体工作**:
1. 用 DeepSeek API (已有 key) 跑 7 个核心任务
2. 对比:
   - Rule-based mock: 确定性, 快, 但不灵活
   - DeepSeek-chat: 有推理能力, 能处理意外情况
3. 记录 token 消耗和延迟
4. 测试 LLM 在以下场景的表现:
   - 模糊指令 ("让电路跑得更好")
   - 多步推理 ("如果 A 不行就试 B")
   - 异常处理 ("工具返回错误")

**验证**: DeepSeek 在模糊指令任务上成功率 > mock planner

---

### P3-T6: 论文主图 + 可视化 (1天)

**目标**: 产出论文需要的核心图表

**图表清单**:
1. **Fig 1**: 系统架构图 (三层: Perception / Cognition / Execution)
2. **Fig 2**: ReAct trace 示例 (含 drift 检测 → replanning 的完整流程)
3. **Fig 3**: 6 系统 × 4 tier 的 success_rate 柱状图
4. **Fig 4**: Drift recovery 曲线 (fidelity vs time, 注入漂移后各系统的恢复)
5. **Fig 5**: Rabi 端到端 demo (已有)
6. **Table 1**: 主结果表 (6 系统 × 6 指标)
7. **Table 2**: 消融表 (Full vs −Memory vs −Drift)

---

## 时间线

```
Week 7 (Day 1-5):
  Day 1-2: P3-T1 ExperimentRecord 集成
  Day 3-5: P3-T2 Drift-Aware Replanning

Week 8 (Day 6-10):
  Day 6-7: P3-T3 Fidelity Predictor 升级
  Day 8-9: P3-T4 完整消融实验
  Day 10:  P3-T5 DeepSeek 真实 LLM 集成

Week 9 (Day 11-14):
  Day 11:  P3-T5 续 + 数据收集
  Day 12:  P3-T6 论文图表
  Day 13:  整合 + 补充实验
  Day 14:  Phase 3 报告 + 代码清理
```

---

## 成功判据

| 指标 | 目标 |
|------|------|
| Full QuantumGPT vs ReAct baseline success_rate 提升 | >= 15% |
| ExperimentRecord 消融 (有 vs 无) 差异 | >= 10% |
| Drift-aware 消融 (有 vs 无) 在 Tier 2 差异 | >= 25% |
| Fidelity 预测 MAE | < 0.05 |
| Drift recovery steps (Full system) | <= 3 |
| DeepSeek 在模糊任务上 vs mock | 成功率更高 |

---

## 风险与对策

| 风险 | 对策 |
|------|------|
| ExperimentRecord RAG 检索不准 | 退化为精确匹配 (circuit_name + backend)，不用向量搜索 |
| Drift-aware 提升不明显 | 加大漂移注入幅度；或改为"故障注入"场景 |
| Fidelity predictor 训不出来 | 保持 lookup table + 线性修正，paper 中诚实报告 |
| DeepSeek API 不稳定 | 本地缓存 + 重试；最坏用 mock 数据 + 少量真实 LLM 样本 |
| 30 个任务跑太慢 | 减少 shots (1024)；并行化；只跑 1 seed |

---

## 产出物

- [ ] `agent/memory.py` — ExperimentRecord 集成到 agent
- [ ] `agent/drift_aware.py` — Drift-aware replanning 逻辑
- [ ] `models/fidelity_predictor.py` — 数据驱动保真度预测
- [ ] `eval/ablation.py` — 6 系统 × 30 任务消融评测
- [ ] `eval/drift_experiment.py` — 漂移注入实验
- [ ] `figures/` — 论文图表 (PDF + PNG)
- [ ] `Phase3_Report_CN.md` — Phase 3 报告
- [ ] 论文主图初稿
