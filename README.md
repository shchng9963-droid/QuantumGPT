# QuantumGPT

**A device-aware, closed-loop LLM agent for NISQ quantum computing.**

QuantumGPT connects a ReAct-style language model agent to simulated (and real) quantum hardware.
Given a natural-language prompt — *"Calibrate qubit Q0 and verify gate quality"* — the agent
autonomously selects tools, runs pulse-level experiments, fits data, and iterates until
a target fidelity is reached.

```
  User prompt                   ┌──────────────┐
  "Run Rabi on Q0" ────────────►│  ReAct Agent  │──── Thought / Action / Observation loop
                                └──────┬───────┘
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              21 MCP Tools      Shadow Backend     Fidelity Budget
              (perception,      (FakeBrisbane,     (stop / refine /
               pulse, RB,       Replay, Drift)      replan logic)
               BO, ...)
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **21 MCP-compatible tools** | Perception, gate-level, pulse-level, fitting, diagnosis, BO |
| **Shadow backends** | `FakeBackendAdapter`, `ReplayBackend`, `SyntheticDrift` — no real QPU needed |
| **Physics simulators** | Rabi, Ramsey, T1, DRAG, Randomized Benchmarking, Active Learning (Bayesian Opt) |
| **ReAct agent** | Thought→Action→Observation loop with fidelity budget and drift-aware replanning |
| **Agent memory** | `ExperimentRecord`-based memory for cross-task learning |
| **QC-Agent-Bench** | 65-task benchmark (4 tiers) for evaluating quantum-LLM agents |
| **Works offline** | `mock` provider runs the full agent with rule-based decisions, no API key needed |

---

## Quick Start

```bash
git clone https://github.com/shchng9963-droid/QuantumGPT.git
cd QuantumGPT
cp .env.example .env
make setup
make doctor
make smoke-test       # offline ReAct loop; no API key required
```

All developer commands use `.venv/bin/...`; see [INSTALL.md](INSTALL.md) for environment details.
For a new contributor, follow the [documentation reading order](docs/README.md),
starting with [docs/HANDOFF.md](docs/HANDOFF.md).

---

## Demos

| Demo | Command | What it does |
|------|---------|--------------|
| **Tune-up** | `python demos/tuneup_demo.py` | Full 6-step calibration: Health → Rabi → Ramsey → DRAG → RB → BO |
| **Rabi** | `python demos/rabi_demo.py` | Single Rabi experiment with fit and publication figure |
| **Drift** | `python demos/drift_demo.py` | Drift detection and re-calibration |
| **Memory** | `python demos/memory_demo.py` | Cross-experiment memory and learning |

---

## Tool Inventory (21 tools)

### Perception & Diagnosis
- `get_backend_health` — T1/T2, gate errors, drift score
- `get_qubit_properties` — per-qubit T1, T2, error rates
- `get_coupling_map` — device topology
- `detect_drift` — statistical drift detection
- `get_calibration_age` — staleness check
- `compare_backends` — side-by-side backend comparison
- `diagnose_and_suggest` — root cause analysis + action recommendations

### Gate & Circuit Operations
- `run_circuit` — execute a circuit on the backend
- `transpile_circuit` — map to hardware topology
- `apply_mitigation` — ZNE error mitigation
- `predict_fidelity` — pre-run fidelity estimate
- `list_benchmarks` — available benchmark circuits

### Pulse-Level Experiments
- `rabi_experiment` / `fit_rabi` — drive sweep → π-pulse amplitude
- `ramsey_experiment` / `fit_ramsey` — delay sweep → T2*, detuning
- `t1_experiment` / `fit_t1` — relaxation sweep → T1
- `drag_calibration` — DRAG α sweep → leakage suppression

### Verification & Optimization
- `randomized_benchmarking` — Clifford RB → error per Clifford (EPC)
- `next_best_experiment` — Bayesian optimization for next calibration point

---

## Project Structure

```
quantumgpt/
├── agent/           # ReAct agent, budget, memory, state, drift-aware planner
│   ├── react.py     #   Main ReAct loop (Thought→Action→Observation)
│   ├── budget.py    #   Fidelity budget (stop/refine/replan decisions)
│   ├── memory.py    #   ExperimentRecord-based agent memory
│   ├── state.py     #   Agent state with artifact tracking
│   └── safety.py    #   Safety guardrails
├── backends/        # Shadow hardware backends
│   ├── fake_adapter.py      # FakeBrisbane wrapper
│   ├── replay.py            # Replay from recorded data
│   └── synthetic_drift.py   # Programmable drift injection
├── tools/           # 21 MCP-compatible tool definitions + executor
│   └── quantum_tools.py
├── dynamics/        # Physics simulators
│   ├── rabi.py      #   Rabi oscillation (Schrödinger + noise)
│   ├── drag.py      #   DRAG pulse (3-level transmon)
│   ├── rb.py        #   Randomized Benchmarking
│   └── active_learning.py  # Gaussian Process + BO
├── bench/           # QC-Agent-Bench (65 tasks, 4 tiers)
├── eval/            # Evaluation scripts & metrics
├── demos/           # Ready-to-run demos
│   ├── tuneup_demo.py   # Full tune-up sequence (6 steps)
│   ├── rabi_demo.py     # Rabi experiment
│   ├── drift_demo.py    # Drift detection
│   └── memory_demo.py   # Agent memory
├── data/            # Instrumented executor, data store
├── experiments/     # Standalone experiment scripts
└── configs/         # Backend & agent configurations
```

---

## QC-Agent-Bench

65 tasks across 4 difficulty tiers:

| Tier | Tasks | Description |
|------|-------|-------------|
| **Tier 1** | 30 | Static backend — perception, circuit execution, transpilation |
| **Tier 2** | 20 | Drift scenarios — detect, diagnose, recalibrate |
| **Tier 3** | 10 | Injected failures — fault diagnosis under anomalies |
| **Tier 4** | 5 | Device-level — full Rabi/Ramsey/DRAG tune-up sequences |

```bash
# Run benchmark evaluation
python eval/f4_compare.py
```

---

## Architecture

The agent follows the **ReAct** (Reason + Act) pattern:

1. **Thought** — LLM reasons about what to do next, considering fidelity budget
2. **Action** — calls one of 21 tools via structured function calling
3. **Observation** — receives tool output, updates agent state
4. **Decision** — fidelity budget decides: stop (good enough), refine (iterate), or replan

The fidelity budget tracks remaining tool calls and target fidelity, enabling
the agent to make cost-aware decisions about when to stop iterating.

---

## Requirements

- Python 3.11 (Qiskit compatibility)
- Qiskit 1.3.0 + qiskit-dynamics 0.6.0 + qiskit-aer 0.17.2
- numpy, scipy, matplotlib, duckdb
- Optional: OpenAI / Anthropic / DeepSeek API key for LLM-driven mode

---

## License

Research use. See LICENSE for details.
