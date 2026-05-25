# QuantumGPT v2.2 Plan

> **继 v2.1 之后的下一个 sprint。主题：Ramsey/T1 实验 + LLM 替换 mock + Eval 自动化。**

---

## v2.1 交付回顾

| 模块 | 状态 |
|------|------|
| Memory 因果闭环 (diagnose → override → run) | ✅ 验证通过 |
| Safety artifact in trace | ✅ 7 tests |
| Final answer execution-mode label | ✅ |
| Ablation P1-6 (90 runs) | ✅ diagnose 0→100%, fid +0.01 |
| memory_demo / drift_demo / showcase runner | ✅ |
| Handout v2.1 | ✅ |

**当前工具清单 (18 个)**:
get_backend_health, get_qubit_properties, run_circuit, list_benchmarks,
get_coupling_map, detect_drift, get_calibration_age, compare_backends,
transpile_circuit, apply_mitigation, predict_fidelity,
rabi_experiment, fit_rabi, ramsey_experiment, fit_ramsey,
t1_experiment, fit_t1, diagnose_and_suggest

**当前 backend**: FakeBackendAdapter (门级), SyntheticDriftBackend, ReplayBackend

---

## v2.2 目标 (2 周)

### Stage E · 新实验工具: Ramsey + T1 + fit ✅
> v2 Plan 附录 B 的 #14 (ramsey_experiment), #24 (extract_t1_t2), #23 (fit_ramsey)

| 任务 | 交付物 | 状态 |
|------|--------|------|
| E1. `ramsey_experiment` 工具 | 仿真跑 Ramsey 并输出 IQ 衰减曲线 | ✅ |
| E2. `fit_ramsey` 工具 | 拟合 T2* + detuning | ✅ |
| E3. `t1_experiment` + `fit_t1` | 输出 T1 衰减 + 拟合 | ✅ |
| E4. Planner 集成 | agent 能根据 prompt 自主选择 rabi/ramsey/t1 | → Stage F |
| E5. Demo: `tune_up_demo.py` | cavity→rabi→ramsey→T1 全链 | ✅ |

**验收标准**: `qgpt "measure T2* of qubit 0"` 自动跑 Ramsey + fit，报告 T2* 和 detuning。

---

### Stage F · LLM Provider (替换 mock planner)
> v2.1 所有 demo 都跑 `provider="mock"` (规则引擎)。v2.2 接入真 LLM。

| 任务 | 交付物 | 优先级 |
|------|--------|--------|
| F1. Claude / GPT-4o API 接入 | 修复 `_run_anthropic` / `_run_openai` | P0 |
| F2. DeepSeek-V3 / Qwen-3 本地 vLLM 接入 | defer to v2.3 (GPU 资源待定) | P2 |
| F3. Tool-calling schema 完善 | 14 工具 → JSON function calling schema | P0 |
| F4. LLM vs Mock 对比测试 | 相同 10 个 prompt, mock vs LLM 对比表 | P1 |
| F5. Cost / latency 追踪 | per-call token count + wall time 进 trace | P1 |

**验收标准**: `qgpt --provider deepseek "Run GHZ-5 and mitigate if fidelity < 0.95"` 端到端通。

---

### Stage G · Eval 自动化 (QC-Agent-Bench skeleton)
> v2 Plan Phase 4 的准备。先把框架搭好。

| 任务 | 交付物 | 优先级 |
|------|--------|--------|
| G1. Task schema 定义 | `benchmark/tasks.yaml` — 20 个种子任务 | P0 |
| G2. Eval runner | `eval/run_eval.py` 批量跑 + 结果持久化 | P0 (已有雏形) |
| G3. 自动评分器 | fidelity / success / tool_count / wall_time 自动打分 | P0 |
| G4. CI 集成 | `pytest tests/test_eval.py` 5 个 smoke task < 3 min | P1 |
| G5. W&B 集成 | 每次 eval 自动上传 summary table | P2 |

**验收标准**: `python eval/run_eval.py --tasks 20 --provider mock` 输出标准对比表。

---

### Stage H · 技术债 / 可靠性

| 任务 | 交付物 | 优先级 |
|------|--------|--------|
| H1. test_perception_tools 超时修复 | 目前卡住 | P0 |
| H2. `detection/` → `agent/drift_detection/` 统一 | 避免 import 路径问题 | P1 |
| H3. eval/ vs experiments/ 清理 | 目录职责明确化 | P2 |
| H4. `pyproject.toml` 补全 entry_points + deps | `pip install -e .` 直接可用 | P1 |
| H5. CI (GitHub Actions): lint + unit tests | `.github/workflows/ci.yml` | P2 |

---

## 时间线

```
Week 1 (Day 1-7):
  Day 1-2: E1-E3 (Ramsey/T1 工具 + fit)
  Day 3:   E4 (Planner 集成) + E5 (tune_up_demo)
  Day 4-5: F1-F3 (LLM 接入 + schema)
  Day 6:   H1-H2 (技术债)
  Day 7:   F4 (LLM vs Mock 对比)

Week 2 (Day 8-14):
  Day 8-9:  G1-G3 (Eval framework)
  Day 10:   G4 (CI smoke)
  Day 11:   F5 + G5 (cost/latency + W&B)
  Day 12:   Full eval run (mock + 1 LLM)
  Day 13:   Paper outline draft (系统论文骨架)
  Day 14:   commit + merge + handout v2.2
```

---

## 与 v2 Plan 的对应关系

| v2 Plan Phase | v2.2 覆盖 | 备注 |
|------|------|------|
| Phase 0 (地基) | ✅ v2.0-2.1 已完成 | |
| Phase 1 (感知) | ✅ 6 感知工具已全 | detect_drift 升级留 v2.3 |
| Phase 2 (工具矩阵 35个) | **v2.2 新增 4 个 → 18 总** | ramsey/fit_ramsey/t1/fit_t1 |
| Phase 2 (Agent 框架) | **v2.2 接真 LLM** | mock→DeepSeek/Claude |
| Phase 3 (闭环学习) | v2.1 Memory 已做基础 | Active Learning 留 v2.3 |
| Phase 4 (Benchmark) | **v2.2 搭骨架** | 20 task skeleton |
| Phase L (Lab) | v2.3+ | 先把工具做全再找组 |

---

## v2.3 预告 (v2.2 之后)

- `drag_calibration` + `randomized_benchmarking` 工具 (→22 工具)
- Active Learning experiment designer (botorch)
- Cross-device transfer 实验
- LLM fine-tune on quantum tool-use data
- 物理组联系 + demo video 录制
- Paper 主图 / 实验主表初稿

---

## 风险

| 风险 | 对策 |
|------|------|
| vLLM 本地部署 GPU 资源不足 | 退化为 API-only (Claude/GPT-4o) |
| Ramsey/T1 仿真精度低 | 参考 Qiskit Experiments 标准实现 |
| Eval 任务定义主观 | 先做 5 个确定性任务，后续加 LLM-judge |
| LLM tool-calling 不稳定 | 加 retry + fallback to mock |

---

*v2.2 · 2026-05-25*
