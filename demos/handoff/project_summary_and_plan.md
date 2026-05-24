# QuantumGPT Project Summary and Work Plan

> Generated: 2026-05-19
> Purpose: Copy into new conversation as context handoff document

---

## 1. Project Positioning

**One-liner**: First device-aware closed-loop quantum agent that maintains task success rate under NISQ drift by dispatching tools across gate-level and pulse-level, with QC-Agent-Bench benchmark.

**Paper title**: *Quantum GPT: A Device-Aware Closed-Loop Agent for NISQ Workflows under Drift*

**Submission target**: ICLR 2027 Main (deadline ~2026-10-03)

---

## 2. Plan Versions

Three plan documents exist:

| File | Role |
|------|------|
| QuantumGPT_Plan.md (v1) | Original: AI agent benchmark focus, Shadow Hardware, 12-week closed-loop agent + QC-Agent-Bench |
| QuantumGPT_Plan_v2.md (v2) | Upgrade: Physics lab real-device control, 3-layer (gate/pulse/device), 35 tools, Phase L collab, safety rails, 3 papers |
| QuantumGPT_Plan_merged.md | EXECUTION VERSION: v1 skeleton + 3 high-leverage v2 additions (Rabi E2E, ExperimentRecord, physics group outreach) |

**Strategy**: Execute merged version. v2 full toolset/Phase L deferred.

---

## 3. v1 vs v2 Core Differences

| Dimension | v1 | v2 |
|-----------|----|----|
| Primary user | AI researchers | Experimental physicists + AI researchers |
| Target level | Gate-level | Gate + Pulse + Device (3 layers) |
| Backend abstraction | ShadowBackend (3 subclasses) | LabBackend (5 primitives + Lab adapters) |
| Tool count | 12+ | 35 (calibration/pulse-opt/data-interpret/experiment-design/safety) |
| Memory | (circuit, fidelity) episodic | ExperimentRecord (raw_data/fits/decisions/human_notes) |
| Benchmark | 60 tasks (Tier 1-3) | 90 tasks (+ Tier 4 Lab Tasks) |
| Papers | NeurIPS/ICLR + D&B (2) | + PRX Quantum/Nature Commun. (3) |
| Safety | None | dry-run / power_budget / 2-level confirm / audit log |
| Physics collab | None | Phase L (W6-W11 parallel) |

**3 additions from v2 in merged**:
1. Rabi E2E tool (rabi_experiment + fit_rabi) -- cross-layer capability demo
2. ExperimentRecord schema -- replaces simple episodic memory
3. Physics group outreach (email+interview) -- zero-engineering async action

---

## 4. Current Progress (Phase 0-1 complete, Phase 2 mostly done)

### Completed Modules

| Module | Day |
|--------|-----|
| Project scaffold + ShadowBackend abstraction | Day 1 |
| FakeBackendAdapter (FakeBrisbane 127q) | Day 1 |
| ReplayBackend + CalibrationData + drift model | Day 2 |
| SyntheticDriftBackend (DriftProfile+presets) | Day 4 |
| PropertiesStream (real-time telemetry sim) | Day 3 |
| MQTBench integration + 5 gate-level benchmarks | Day 5 |
| Agent Loop (ReAct + while loop + tool calling) | Day 6 |
| W&B + DuckDB data pipeline | Day 7 |
| CalibrationAdvisor (multi-step autonomous advisory) | Day 8 |
| StreamingAdvisor (real-time telemetry + alert engine) | Day 9 |
| Pulse-Level Rabi simulation (Schrodinger) | Day 10 |
| BACKENDS.md + CI green + dependency resolution | Day 11 |
| MVP CLI (qgpt simulate/analyze/advise) | Day 12 |
| ExperimentRecord Schema (DuckDB table + dataclass) | Day 13 |
| LabBackend + Experiment Protocols + Web Frontend | Day 14 |
| 14 quantum tools, ZNE mitigation, DriftDetector | - |
| 4-system comparison + 7x4 evaluation | - |

### Code Stats
- Python files: 88
- Lines of code: ~20,540
- Visualizations: 103 (PNG/PDF)
- Git commits: 9 major

---

## 5. Phase 3 Plan (Next, W7-W9) -- Paper Core

### P3-T1: ExperimentRecord Integration (2 days)
- Agent auto-writes structured experiment records
- New tool: retrieve_past_experiments(query, top_k)
- Ablation switch: use_memory=True/False
- Verify: 10x GHZ-5 then 11th run references history

### P3-T2: Drift-Aware Replanning (3 days) -- CORE EXPERIMENT
- Integrate PropertiesStream into ReAct loop
- New tool: check_drift_since(timestamp)
- Logic: drift > threshold -> invalidate old transpile -> redo
- Use SyntheticDriftBackend for controlled drift injection
- Metrics: drift_recovery_steps / drift_recovery_rate
- Verify: Full system recovers in 3 steps; baseline never recovers

### P3-T3: Fidelity Predictor Upgrade (2 days)
- Train XGBoost/LightGBM on (circuit_features, backend_state, real_fidelity)
- Evaluate: MAE / R-squared

### P3-T4: Full Comparison (2 days)
- 6 systems x 5 tasks x 4 metrics
- Paper Table 1 + ablation Figure 2

### P3-T5: Demo Video (1 day)
- drift -> detect -> replan -> recover flow

---

## 6. Phase 4-5 (W10-W12+)

### Phase 4: Benchmark (W10-W11)
- QC-Agent-Bench: Tier 1(30) + Tier 2(20) + Tier 3(10) = 60 tasks
- 5-system full comparison + GitHub + arXiv

### Phase 5: Submission (W12+)
- ICLR 2027 Main (deadline ~2026-10-03)
- Code freeze: ~2026-08-07
- Writing buffer: 8 weeks

---

## 7. Evaluation System

Baselines: static | llm-single | react-base | qgpt | qgpt-no-mem | qgpt-no-drift

Task tiers: tier1_static(30) | tier2_drift(20) | tier3_failure(10)

Metrics: success_rate | avg_fidelity | tool_calls | wall_time | drift_recovery | cost

---

## 8. Tech Stack

| Layer | Choice |
|-------|--------|
| Agent runtime | Custom Python ReAct loop |
| LLM | DeepSeek-V3 (deepseek-chat) + Claude/GPT-4o |
| Quantum SDK gate | Qiskit 1.x |
| Quantum SDK pulse | Qiskit Dynamics (Schrodinger) |
| Noise sim | Qiskit Aer + Mitiq (ZNE) |
| Tool protocol | MCP |
| Storage | DuckDB |
| Tracking | W&B |
| Viz | Plotly + Web Console |
| CLI | Click (qgpt) |

---

## 9. Key Decisions

1. Shadow hardware: FakeBackend + ReplayBackend + SyntheticDriftBackend, 90% work needs no real QPU
2. Single agent first: no multi-agent, single agent + 14 tools
3. Pulse-level: Qiskit Dynamics only (no QuTiP/C3/Pulser)
4. DeepSeek API primary: base https://api.deepseek.com, model deepseek-chat
5. ExperimentRecord: structured storage (raw_data/fits/decisions)
6. Paper story: closed-loop + device-aware + drift-aware + benchmark

---

## 10. RTX 5090 Notes

- Complex tensor ops (.abs(), negation, division) trigger NVRTC JIT fail (sm_120) -> use (real,imag) float pairs
- PyTorch cu128 no cuDNN -> pip install nvidia-cudnn-cu12==9.19.0.56
- Linalg ops (svdvals, eigvalsh) work fine

---

## 11. File Paths

/home/wangshuchang/quantumgpt/
- QuantumGPT_Plan.md (v1)
- QuantumGPT_Plan_v2.md (v2)
- QuantumGPT_Plan_merged.md (execution)
- Phase3_Plan.md
- cli.py
- backends/ agent/ demos/day1-14/

---

## 12. Next Actions (Phase 3)

| Priority | Task | Time |
|----------|------|------|
| P0 | ExperimentRecord into ReAct Agent | 2d |
| P0 | Drift-Aware Replanning + experiment | 3d |
| P1 | Fidelity Predictor (XGBoost) | 2d |
| P1 | 6-system x 5-task comparison | 2d |
| P2 | Demo video | 1d |
| P2 | Paper main figure + table draft | 1d |

---

## 13. User Preferences

- Task instructions in Chinese
- Paper: 9pp body (excl. references)
- Figures: Nature-like (vector PDF+PNG, white bg, thin axes, muted colors)
- Daily outputs: demos/dayN/ (summary.md + gen_figures.py + viz + tests)
- Systematic review with numbered issues
- Language: natural, plain, clear, fewer long sentences
- Run tests after code changes
