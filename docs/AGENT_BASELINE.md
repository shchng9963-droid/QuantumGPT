# QuantumGPT Agent Optimization Baseline

Date: 2026-05-23 20:04:43 CST

Scope: start of the near-term agent-optimization track. This baseline records the current test status and the first cleanup needed before AgentState / ToolRuntime work.

## Current repository state

Branch: master

Important caveat: the working tree already contains many pre-existing modified and untracked files. I only changed:

- tests/unit/test_perception_tools.py
- reports/baseline_week1/mock_react_drift_tier1_smoke.json
- docs/AGENT_BASELINE.md

## Test baseline before cleanup

Command:

```bash
python -m pytest
```

Result before cleanup:

- Collected: 195 tests, 1 skipped
- Passed: 194
- Failed: 1
- Runtime: 293.04 s
- Failure: tests/unit/test_perception_tools.py::TestToolDefinitions::test_nine_tools_defined

Root cause:

The test still expected the old 9-tool surface, while tools/quantum_tools.py now defines 14 tools:

- get_backend_health
- get_qubit_properties
- run_circuit
- list_benchmarks
- get_coupling_map
- detect_drift
- get_calibration_age
- compare_backends
- transpile_circuit
- apply_mitigation
- predict_fidelity
- rabi_experiment
- fit_rabi
- diagnose_and_suggest

Fix:

Renamed the test to test_current_tool_surface_is_defined_without_duplicates and updated it to assert the current 14-tool surface plus duplicate-name protection.

## Verification after cleanup

Specific test:

```bash
python -m pytest tests/unit/test_perception_tools.py::TestToolDefinitions::test_current_tool_surface_is_defined_without_duplicates -v
```

Result:

- 1 passed

Full suite:

```bash
python -m pytest
```

Result after cleanup:

- Collected: 195 tests, 1 skipped
- Passed: 195
- Failed: 0
- Runtime: 302.10 s
- Warnings: many Qiskit / qiskit-aer / qiskit-ibm-runtime deprecation warnings, currently non-blocking

## Benchmark smoke baseline

Command:

```bash
python benchmark/runner.py --systems react_drift --tiers 1 --max-tasks 3 --provider mock --save-traces --output reports/baseline_week1/mock_react_drift_tier1_smoke.json
```

Result:

- Tasks: 3 Tier-1 tasks
- System: react_drift
- Provider: mock
- Success: 100.0%
- Average fidelity: 0.9099
- Average tool calls: 2.7
- Average wall time: 0.40 s
- Traces saved in reports/baseline_week1/mock_react_drift_tier1_smoke.json

## Immediate next step

Start Week-2 implementation early:

1. Add agent/state.py.
2. Define AgentState, Artifact, ArtifactRegistry, ArtifactStatus, InvalidationReason.
3. Add tests/unit/test_agent_state.py using TDD.
4. First target behavior:
   - add artifact
   - query artifact by type/status
   - invalidate artifact
   - recursively invalidate dependents
   - export compact state summary for trace/prompt use

This keeps the project moving from ReAct-loop-plus-tools toward an explicitly stateful domain agent.
