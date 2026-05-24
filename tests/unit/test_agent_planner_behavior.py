"""Behavior tests for the rule planner maturity checklist."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.budget import FidelityBudget
from agent.drift_aware import DriftAwareRulePlanner, DriftState, ReplanningPolicy
from agent.react import ReActRulePlanner


def _budget(target=0.85):
    budget = FidelityBudget(target_fidelity=target, max_tool_calls=20, max_seconds=120)
    budget.start()
    return budget


def _call_names(calls):
    return [call["name"] for call in calls]


def test_ambiguous_improvement_task_starts_with_backend_sensing():
    planner = ReActRulePlanner()

    thought, calls, final = planner.plan("make this circuit better", [], _budget())

    assert final is None
    assert _call_names(calls)[0] in {"get_backend_health", "get_qubit_properties", "list_benchmarks"}
    assert "run_circuit" not in _call_names(calls)


def test_low_fidelity_run_falls_back_to_mitigation():
    planner = ReActRulePlanner()
    budget = _budget(target=0.90)
    history = [
        {"tool": "get_backend_health", "input": {}, "result": json.dumps({"backend": "fake"})},
        {"tool": "run_circuit", "input": {"circuit_name": "ghz_5"}, "result": json.dumps({"circuit": "ghz_5", "fidelity": 0.42})},
    ]
    budget.record_tool_call("get_backend_health")
    budget.record_tool_call("run_circuit", fidelity=0.42)

    thought, calls, final = planner.plan("run ghz_5 with high fidelity", history, budget)

    assert final is None
    assert _call_names(calls) == ["apply_mitigation"]
    assert calls[0]["input"]["circuit_name"] == "ghz_5"


class _Monitor:
    def __init__(self):
        self.state = DriftState(
            drift_score=0.8,
            is_drifting=True,
            invalidated_results=["run_circuit"],
            consecutive_drift_checks=1,
        )
        self.acknowledged = False

    def acknowledge_replan(self):
        self.acknowledged = True
        self.state.invalidated_results = []


def test_drift_forces_backend_health_recheck_before_rerun():
    monitor = _Monitor()
    planner = DriftAwareRulePlanner(
        ReActRulePlanner(),
        monitor,
        ReplanningPolicy(strategy="aggressive"),
    )
    history = [
        {"tool": "get_backend_health", "input": {}, "result": json.dumps({"backend": "fake"})},
        {"tool": "run_circuit", "input": {"circuit_name": "ghz_5"}, "result": json.dumps({"fidelity": 0.80})},
    ]

    thought, calls, final = planner.plan("run ghz_5 and monitor drift", history, _budget())

    assert final is None
    assert _call_names(calls) == ["get_backend_health"]
    assert "DRIFT DETECTED" in thought


def test_rabi_task_orders_experiment_before_fit_before_final_report():
    planner = ReActRulePlanner()
    budget = _budget()
    prompt = "Run a Rabi experiment on a 5.0 GHz qubit and fit the pi pulse."
    history = [{"tool": "get_backend_health", "input": {}, "result": json.dumps({"backend": "fake"})}]

    thought, calls, final = planner.plan(prompt, history, budget)
    assert final is None
    assert _call_names(calls) == ["rabi_experiment"]

    history.append({
        "tool": "rabi_experiment",
        "input": {"qubit_freq_ghz": 5.0},
        "result": json.dumps({"amplitudes": [0.1, 0.2], "populations": [0.2, 0.9]}),
    })
    thought, calls, final = planner.plan(prompt, history, budget)
    assert final is None
    assert _call_names(calls) == ["fit_rabi"]

    history.append({
        "tool": "fit_rabi",
        "input": {"amplitudes": [0.1, 0.2], "populations": [0.2, 0.9]},
        "result": json.dumps({"pi_amplitude_mhz": 0.2, "r_squared": 0.99}),
    })
    thought, calls, final = planner.plan(prompt, history, budget)
    assert calls == []
    assert final is not None
    assert "Rabi Fit Result" in final


def test_failed_run_circuit_retries_once_then_falls_back_without_looping():
    planner = ReActRulePlanner()
    prompt = "run ghz_5 with high fidelity"
    one_failure_history = [
        {"tool": "get_backend_health", "input": {}, "result": json.dumps({"backend": "fake"})},
        {"tool": "run_circuit", "input": {"circuit_name": "ghz_5", "shots": 4096}, "result": json.dumps({"error": "backend_timeout"})},
    ]

    thought, calls, final = planner.plan(prompt, one_failure_history, _budget())
    assert final is None
    assert _call_names(calls) == ["run_circuit"]
    assert calls[0]["input"]["circuit_name"] == "ghz_5"

    two_failure_history = one_failure_history + [
        {"tool": "run_circuit", "input": {"circuit_name": "ghz_5", "shots": 4096}, "result": json.dumps({"error": "backend_timeout"})},
    ]
    thought, calls, final = planner.plan(prompt, two_failure_history, _budget())

    assert _call_names(calls) != ["run_circuit"]
    assert final is not None or _call_names(calls) == ["diagnose_and_suggest"]
