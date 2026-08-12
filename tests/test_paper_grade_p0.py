"""P0 paper-grade consistency tests.

These tests guard the credibility fixes required before paper-grade
experiments: drift detection must not mutate backend time or report future
telemetry as current state, and budget summaries must reflect mitigation calls.
"""

import json

from agent.budget import BudgetDecision, FidelityBudget
from backends.synthetic_drift import STABLE, SUDDEN_DEGRADATION, SyntheticDriftBackend
from tools.quantum_tools import ToolExecutor


def _execute_json(executor: ToolExecutor, name: str, args: dict | None = None) -> dict:
    return json.loads(executor.execute(name, args or {}))


def test_detect_drift_preserves_current_backend_time_and_calibration_age():
    backend = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
    backend.set_time(6.0)
    executor = ToolExecutor(backend)

    before_health = _execute_json(executor, "get_backend_health")
    drift = _execute_json(executor, "detect_drift", {"window_hours": 24, "step_hours": 0.5})
    after_health = _execute_json(executor, "get_backend_health")
    age = _execute_json(executor, "get_calibration_age")

    assert before_health["calibration_age_minutes"] == 360.0
    assert after_health["calibration_age_minutes"] == before_health["calibration_age_minutes"]
    assert age["calibration_age_minutes"] == before_health["calibration_age_minutes"]
    assert drift["drift_score_current"] == before_health["drift_score"]


def test_detect_drift_uses_history_ending_at_current_time_not_future_window():
    backend = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
    backend.set_time(3.0)  # before the sudden degradation at t=5h
    executor = ToolExecutor(backend)

    health = _execute_json(executor, "get_backend_health")
    drift = _execute_json(executor, "detect_drift", {"window_hours": 24, "step_hours": 0.5})
    age = _execute_json(executor, "get_calibration_age")

    assert health["drift_score"] == 0.0
    assert drift["drift_score_current"] == 0.0
    assert drift["changepoints"] == []
    assert age["calibration_age_minutes"] == 180.0


def test_fidelity_budget_marks_apply_mitigation_as_attempted():
    budget = FidelityBudget(target_fidelity=0.85, max_tool_calls=5, max_seconds=60)
    budget.start()

    budget.record_tool_call("run_circuit", fidelity=0.67)
    assert budget.decide() == BudgetDecision.MITIGATE

    budget.record_tool_call("apply_mitigation", fidelity=0.70)

    summary = budget.summary()
    assert summary["mitigation_attempted"] is True
    assert summary["decision"] == BudgetDecision.PROCEED.name
