# QuantumGPT P0 Agent Maturity Handoff

Date: 2026-05-23
Scope: P0 agent maturity closure for the physics-facing QuantumGPT prototype.

## Executive summary

P0 is now in a runnable and auditable state for a mock/FakeBrisbane physics demo.
The current agent can:

1. plan a physics-facing workflow with a deterministic ReAct rule planner;
2. inspect backend health before acting;
3. run Rabi characterization and fit the resulting oscillation;
4. track explicit runtime state and artifacts;
5. record memory context when memory is enabled and avoid leaking it when disabled;
6. apply safety and dry-run diagnostics to risky lab-style actions;
7. export human-readable reports and full JSON traces;
8. run benchmark smoke tests with `--save-traces`.

This does not mean the system is ready for real hardware or paper-grade claims. It means the P0 engineering substrate is now test-covered enough to support the next phase: real-provider smoke tests, lab adapter hardening, richer benchmark tasks, and paper/demo packaging.

## Main code areas

Agent runtime:

- `/home/wangshuchang/quantumgpt/agent/react.py`
- `/home/wangshuchang/quantumgpt/agent/drift_aware.py`
- `/home/wangshuchang/quantumgpt/agent/state.py`
- `/home/wangshuchang/quantumgpt/agent/safety.py`
- `/home/wangshuchang/quantumgpt/agent/memory.py`
- `/home/wangshuchang/quantumgpt/agent/reporting.py`

Benchmark/demo:

- `/home/wangshuchang/quantumgpt/benchmark/runner.py`
- `/home/wangshuchang/quantumgpt/demos/rabi_demo.py`
- `/home/wangshuchang/quantumgpt/showcase/run_all_demos.sh`
- `/home/wangshuchang/quantumgpt/showcase/README.md`

Key tests:

- `/home/wangshuchang/quantumgpt/tests/unit/test_agent_planner_behavior.py`
- `/home/wangshuchang/quantumgpt/tests/unit/test_agent_safety.py`
- `/home/wangshuchang/quantumgpt/tests/unit/test_agent_safety_integration.py`
- `/home/wangshuchang/quantumgpt/tests/unit/test_agent_state.py`
- `/home/wangshuchang/quantumgpt/tests/unit/test_agent_state_integration.py`
- `/home/wangshuchang/quantumgpt/tests/unit/test_agent_trace_diagnostics.py`
- `/home/wangshuchang/quantumgpt/tests/unit/test_agent_reporting.py`
- `/home/wangshuchang/quantumgpt/tests/unit/test_agent_failure_recovery.py`
- `/home/wangshuchang/quantumgpt/tests/unit/test_benchmark_runner.py`
- `/home/wangshuchang/quantumgpt/tests/unit/test_rabi_demo_outputs.py`
- `/home/wangshuchang/quantumgpt/tests/test_memory_integration.py`

## What is covered in P0

### 1. Planner maturity

Covered behavior:

- Ambiguous optimization requests inspect backend health before running circuits.
- Low-fidelity circuit results can trigger mitigation.
- Drift-aware planner rechecks backend health after drift.
- Rabi tasks follow the intended order:
  `get_backend_health -> rabi_experiment -> fit_rabi -> final report`.
- Failed tool calls do not loop forever; repeated `run_circuit` failure transitions to diagnosis/stop behavior.

Primary test:

```bash
cd /home/wangshuchang/quantumgpt
pytest tests/unit/test_agent_planner_behavior.py -q
```

### 2. Explicit state and artifacts

The agent now records a separate `AgentState`, rather than relying only on message history.
State export includes:

- backend snapshot id;
- current step;
- artifact counts and statuses;
- recent observations;
- budget state;
- drift state;
- safety status;
- artifacts with dependency metadata.

Trace JSON includes:

- `state_summary`
- `artifacts`
- step-level `state_summary`

Primary tests:

```bash
pytest tests/unit/test_agent_state.py tests/unit/test_agent_state_integration.py -q
```

### 3. Safety and dry-run diagnostics

The report and trace expose:

- `safety_block_count`
- `safety_dry_run_count`
- state-level safety status when present
- tool-level safety errors when a tool is blocked

Primary tests:

```bash
pytest tests/unit/test_agent_safety.py tests/unit/test_agent_safety_integration.py -q
```

### 4. Memory behavior

Covered behavior:

- Memory entries are used to inform planning when enabled.
- Previous low-fidelity outcomes can cause the planner to diagnose before repeating a similar run.
- `use_memory=False` prevents memory context from leaking into exported traces.
- Structured `trace.state.memory_context` is preserved when memory is enabled.

Primary test:

```bash
pytest tests/test_memory_integration.py -q
```

### 5. Human-readable reporting

`TraceReporter` now writes:

- markdown report;
- raw trace JSON;
- short talk track.

The report contains:

- request;
- run summary;
- key physics results;
- trace steps;
- tool issues if any;
- unparsed observation warnings if any;
- reliability warnings;
- runtime state;
- memory context;
- safety and dry-run section;
- diagnostics;
- final answer.

Primary tests:

```bash
pytest tests/unit/test_agent_reporting.py tests/unit/test_rabi_demo_outputs.py -q
```

### 6. Benchmark trace export

The benchmark runner supports `--save-traces`, and the saved result rows include:

- top-level task/system metrics;
- diagnostic fields;
- full trace payload when requested;
- full observations in trace steps;
- trace diagnostics;
- state summary;
- artifacts.

Smoke command used:

```bash
cd /home/wangshuchang/quantumgpt
python benchmark/runner.py \
  --systems react_drift \
  --tiers 1 \
  --max-tasks 1 \
  --provider mock \
  --save-traces \
  --output reports/baseline_week1/mock_react_drift_save_traces_smoke.json \
  --verbose
```

Output:

- `/home/wangshuchang/quantumgpt/reports/baseline_week1/mock_react_drift_save_traces_smoke.json`

Current smoke result:

- 1 task x 1 system
- react_drift success: 100%
- output has full trace, observations, diagnostics, state summary, and artifacts.

## Physics demo artifacts

Standardized output directory:

- `/home/wangshuchang/quantumgpt/showcase/physics_demo_output/`

Files:

- `/home/wangshuchang/quantumgpt/showcase/physics_demo_output/README.md`
- `/home/wangshuchang/quantumgpt/showcase/physics_demo_output/rabi_report.md`
- `/home/wangshuchang/quantumgpt/showcase/physics_demo_output/rabi_trace.json`
- `/home/wangshuchang/quantumgpt/showcase/physics_demo_output/rabi_talk_track.txt`
- `/home/wangshuchang/quantumgpt/showcase/physics_demo_output/rabi_oscillation.png`
- `/home/wangshuchang/quantumgpt/showcase/physics_demo_output/rabi_oscillation.pdf`

Run the standardized physics demo:

```bash
cd /home/wangshuchang/quantumgpt
PYTHONPATH=. python - <<'PY'
from demos.rabi_demo import run_rabi_demo
run_rabi_demo(save_dir='showcase/physics_demo_output')
PY
```

Current demo result:

- Backend: FakeBrisbane
- Workflow: `get_backend_health -> rabi_experiment -> fit_rabi -> final report`
- pi-pulse amplitude: 52.52 MHz
- pi/2-pulse amplitude: 0.026261 GHz
- fit R^2: 1.0

Run all showcase demos:

```bash
cd /home/wangshuchang/quantumgpt
bash showcase/run_all_demos.sh
```

## Full P0 regression command

Use this before claiming P0 still works:

```bash
cd /home/wangshuchang/quantumgpt
pytest \
  tests/unit/test_agent_planner_behavior.py \
  tests/unit/test_agent_safety.py \
  tests/unit/test_agent_safety_integration.py \
  tests/unit/test_agent_state.py \
  tests/unit/test_agent_state_integration.py \
  tests/unit/test_agent_trace_diagnostics.py \
  tests/unit/test_agent_reporting.py \
  tests/unit/test_agent_failure_recovery.py \
  tests/unit/test_benchmark_runner.py \
  tests/unit/test_rabi_demo_outputs.py \
  tests/test_memory_integration.py \
  -q
```

Most recent result:

- 52 passed
- 226009 warnings
- runtime: 100.70 s

The warnings are dominated by Qiskit and qiskit_ibm_runtime deprecations around BackendV1, Qobj, Pulse, and calibration APIs. They are not current logic failures.

## Current limitations and next steps

### Not yet real-hardware ready

The P0 demo uses FakeBrisbane/mock provider paths. Before real hardware:

1. harden lab backend adapters;
2. define a dry-run/approval boundary for real pulse or calibration actions;
3. add tests that prove dangerous parameters are blocked before hardware submission;
4. add credentials/config checks that fail loudly instead of silently falling back to mock.

### Not yet paper-claim ready

Current evidence supports an engineering demo, not a strong scientific claim. Before paper claims:

1. expand benchmark tasks beyond smoke coverage;
2. run multiple systems and tiers with saved traces;
3. define metrics that measure physics utility, not only tool-call success;
4. compare against static pipeline, single-shot LLM, ReAct without budget, ReAct with memory, and drift-aware variants;
5. inspect failed traces manually and summarize failure modes.

### Real-provider evaluation still needs smoke tests

The benchmark runner has provider/model/base-url controls and guards against silent mock fallback, but P0 was validated with `provider=mock`.
Next smoke should use a real API-backed provider with a tiny task subset and saved traces.

### Qiskit deprecation warnings should be isolated

Warnings are noisy enough to hide real issues in CI output. A future task should either:

- filter known third-party deprecation warnings in pytest config, or
- migrate affected fake backend/pulse code toward BackendV2/Qiskit Dynamics-compatible paths.

### Showcase still needs a polished narrative layer

The standardized artifacts exist, but a final presentation package should include:

1. one short Chinese demo script;
2. one architecture diagram;
3. one table mapping P0 capabilities to evidence files/tests;
4. one risk slide separating mock demo, real-provider eval, and real-hardware execution.

## Quick handoff checklist

Before handing to another agent or collaborator:

1. Run full P0 regression command above.
2. Run benchmark save-traces smoke command above.
3. Run standardized physics demo command above.
4. Open `/home/wangshuchang/quantumgpt/showcase/physics_demo_output/rabi_report.md` and verify it contains:
   - Runtime state
   - Memory context
   - Safety and dry-run
   - Diagnostics
5. Open `/home/wangshuchang/quantumgpt/showcase/physics_demo_output/rabi_trace.json` and verify it contains:
   - `steps`
   - `diagnostics`
   - `state_summary`
   - `artifacts`
6. Do not claim real-hardware readiness unless a real lab adapter and approval flow have been tested.
