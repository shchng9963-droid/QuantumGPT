"""B4 — VerifierNode tests.

Layered tests:

  Trace probe unit tests:
    - extract_claimed_fidelity from text (numeric / percentage / no number)
    - max_observed_step_fidelity from a trace dict
    - trace_best_fidelity reads multiple field names

  Budget gate unit tests:
    - retry_count cap (>=2 blocks)
    - cost / time / tool_calls each block at 80% margin

  Verifier node behavior:
    - happy path: claimed == observed, meets target → satisfied
    - hallucinated claim → flagged, retry requested if budget allows
    - inconsistent best_fidelity (claim < trace.best_fidelity > steps) → flagged
    - below target → not satisfied, retry requested
    - empty trace → not satisfied, retry if budget allows
    - resimulator override path: passes a custom recomputed value
    - resimulator raises → handled gracefully (resim_failed reason)

  Integration with graph: verifier rejection routes back through reflector
    → planner → executor (via custom executor stub).
"""

from __future__ import annotations

import time

import pytest

from agent.orchestrator.graph import (
    NODE_EXECUTOR,
    NODE_FINALIZER,
    NODE_PLANNER,
    NODE_REFLECTOR,
    NODE_TASK_ROUTER,
    NODE_VERIFIER,
    build_graph,
)
from agent.orchestrator.state import (
    make_task,
    new_orchestrator_state,
)
from agent.orchestrator.verifier_node import (
    DEFAULT_HALLUCINATION_TOLERANCE,
    budget_has_room,
    extract_claimed_fidelity,
    make_verifier_node,
    max_observed_step_fidelity,
    passthrough_resimulator,
    trace_best_fidelity,
)


# ── trace probes ───────────────────────────────────────────────────────────


def test_extract_claimed_fidelity_basic():
    assert extract_claimed_fidelity("Fidelity: 0.92") == 0.92
    assert extract_claimed_fidelity("fidelity = 0.876") == 0.876
    # picks the largest if multiple
    assert extract_claimed_fidelity(
        "Initial fidelity 0.6, after mitigation Fidelity 0.92"
    ) == 0.92


def test_extract_claimed_fidelity_percentage_normalized():
    assert extract_claimed_fidelity("Final fidelity: 92") == 0.92
    assert extract_claimed_fidelity("fidelity 88%") == 0.88


def test_extract_claimed_fidelity_returns_none_when_absent():
    assert extract_claimed_fidelity("") is None
    assert extract_claimed_fidelity("circuit ran successfully") is None
    # Out-of-range value is rejected
    assert extract_claimed_fidelity("fidelity 199") is None


def test_max_observed_step_fidelity():
    td = {
        "steps": [
            {"action": "transpile", "fidelity_observed": None},
            {"action": "run_circuit", "fidelity_observed": 0.74},
            {"action": "apply_mitigation", "fidelity_observed": 0.91},
        ]
    }
    assert max_observed_step_fidelity(td) == 0.91
    # No fidelities anywhere
    assert max_observed_step_fidelity({"steps": [{"action": "x"}]}) is None
    assert max_observed_step_fidelity({}) is None


def test_trace_best_fidelity_reads_multiple_locations():
    assert trace_best_fidelity({"best_fidelity": 0.9}) == 0.9
    # Falls back to budget_summary
    assert trace_best_fidelity({"budget_summary": {"best_fidelity": 0.85}}) == 0.85
    assert trace_best_fidelity({}) is None


# ── budget gate ────────────────────────────────────────────────────────────


def _state_with_budget(*, retries=0, cost_used=0.0, walltime=0.0, calls_used=0,
                      max_usd=0.5, max_seconds=60.0, max_tool_calls=20):
    s = new_orchestrator_state(
        make_task("x", max_usd=max_usd, max_seconds=max_seconds, max_tool_calls=max_tool_calls)
    )
    s["verifier_retry_count"] = retries
    s["budget_used"] = {
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": cost_used,
        "walltime_seconds": walltime,
        "tool_calls": calls_used,
        "started_at": time.time() - walltime if walltime > 0 else time.time(),
        "deadline_at": None,
    }
    return s


def test_budget_has_room_happy():
    has_room, reason = budget_has_room(_state_with_budget())
    assert has_room is True
    assert reason == "ok"


def test_budget_blocks_on_retry_cap():
    has_room, reason = budget_has_room(_state_with_budget(retries=2))
    assert has_room is False
    assert "retry_cap" in reason


def test_budget_blocks_on_cost_near_cap():
    has_room, reason = budget_has_room(
        _state_with_budget(cost_used=0.45, max_usd=0.5)  # 90% of cap
    )
    assert has_room is False
    assert "cost" in reason


def test_budget_blocks_on_walltime_near_cap():
    has_room, reason = budget_has_room(
        _state_with_budget(walltime=55.0, max_seconds=60.0)  # 91% of cap
    )
    assert has_room is False
    assert "walltime" in reason


def test_budget_blocks_on_toolcalls_near_cap():
    has_room, reason = budget_has_room(
        _state_with_budget(calls_used=18, max_tool_calls=20)  # 90% of cap
    )
    assert has_room is False
    assert "toolcalls" in reason


# ── verifier node ──────────────────────────────────────────────────────────


def _trace(*, final_answer="", best=None, steps_fids=None, num_tool_calls=None):
    """Build a trace-shaped dict for tests."""
    if steps_fids is None:
        steps_fids = []
    steps = [
        {"step_num": i, "action": f"tool{i}", "fidelity_observed": f}
        for i, f in enumerate(steps_fids)
    ]
    td = {
        "user_prompt": "test",
        "model": "rule-mock",
        "provider": "mock",
        "backend": "synth",
        "final_answer": final_answer,
        "steps": steps,
        "num_tool_calls": num_tool_calls if num_tool_calls is not None else len(steps),
    }
    if best is not None:
        td["best_fidelity"] = best
    return td


def _state_with_trace(trace, *, target=0.0, retries=0):
    s = _state_with_budget(retries=retries)
    s["task"]["target_fidelity"] = target
    s["executor_traces"] = [trace]
    return s


def test_verifier_happy_path():
    node = make_verifier_node()
    state = _state_with_trace(
        _trace(final_answer="Fidelity: 0.91, mitigation OK", best=0.91, steps_fids=[0.7, 0.91]),
        target=0.85,
    )
    update = node(state)
    v = update["verification"]
    assert v["is_satisfied"] is True
    assert v["hallucinated"] is False
    assert v["confidence"] >= 0.9
    fc = v["fidelity_check"]
    assert fc["claimed"] == 0.91
    assert fc["max_observed"] == 0.91
    assert fc["recomputed"] == 0.91
    assert update["re_execution_needed"] is False


def test_verifier_flags_hallucination_when_claim_exceeds_observed():
    """Agent claims 0.95 but no step observed > 0.70 — hallucination."""
    node = make_verifier_node()
    state = _state_with_trace(
        _trace(final_answer="Mitigated fidelity = 0.95", best=0.70, steps_fids=[0.6, 0.7]),
        target=0.85,
    )
    update = node(state)
    v = update["verification"]
    assert v["is_satisfied"] is False
    assert v["hallucinated"] is True
    assert "claim" in v["reason"]
    # budget allows retry
    assert update["re_execution_needed"] is True


def test_verifier_flags_internal_inconsistency():
    """Trace's headline best_fidelity exceeds anything actually observed in steps."""
    node = make_verifier_node()
    state = _state_with_trace(
        _trace(final_answer="ok", best=0.99, steps_fids=[0.6, 0.7]),
        target=0.5,
    )
    update = node(state)
    v = update["verification"]
    assert v["hallucinated"] is True
    assert v["is_satisfied"] is False


def test_verifier_below_target_requests_retry_if_budget_allows():
    node = make_verifier_node()
    state = _state_with_trace(
        _trace(final_answer="Fidelity: 0.55", best=0.55, steps_fids=[0.55]),
        target=0.85,
    )
    update = node(state)
    v = update["verification"]
    assert v["is_satisfied"] is False
    assert v["hallucinated"] is False
    assert "below_target" in v["reason"]
    assert update["re_execution_needed"] is True


def test_verifier_below_target_no_retry_when_retries_exhausted():
    node = make_verifier_node()
    state = _state_with_trace(
        _trace(final_answer="Fidelity: 0.55", best=0.55, steps_fids=[0.55]),
        target=0.85,
        retries=2,
    )
    update = node(state)
    assert update["verification"]["is_satisfied"] is False
    assert update["re_execution_needed"] is False


def test_verifier_handles_empty_trace():
    node = make_verifier_node()
    s = _state_with_budget()
    s["executor_traces"] = []
    update = node(s)
    v = update["verification"]
    assert v["is_satisfied"] is False
    assert v["reason"] == "no_executor_trace"


def test_verifier_uses_resimulator_for_recomputed():
    captured = {}

    def my_resim(trace, task):
        captured["called"] = True
        # Pretend re-sim says fidelity is actually 0.65 (lower than claim)
        return {"recomputed_fidelity": 0.65, "ok": True, "reason": "re_simulated"}

    node = make_verifier_node(resimulator=my_resim)
    state = _state_with_trace(
        _trace(final_answer="Fidelity 0.92", best=0.92, steps_fids=[0.92]),
        target=0.85,  # claim/best satisfy this, but re-sim should win
    )
    update = node(state)
    assert captured.get("called") is True
    v = update["verification"]
    # Recomputed value (0.65) is the one that gates the target
    assert v["fidelity_check"]["recomputed"] == 0.65
    assert v["is_satisfied"] is False  # 0.65 < 0.85
    assert "below_target" in v["reason"]


def test_verifier_handles_resimulator_exception():
    def bad_resim(trace, task):
        raise RuntimeError("resim crashed")

    node = make_verifier_node(resimulator=bad_resim)
    state = _state_with_trace(
        _trace(final_answer="Fidelity 0.92", best=0.92, steps_fids=[0.92]),
        target=0.0,
    )
    update = node(state)
    v = update["verification"]
    assert v["is_satisfied"] is False
    assert "resim_failed" in v["reason"]


def test_passthrough_resimulator_just_echoes_best():
    out = passthrough_resimulator({"best_fidelity": 0.88}, {})
    assert out["recomputed_fidelity"] == 0.88
    assert out["ok"] is True
    out_empty = passthrough_resimulator({}, {})
    assert out_empty["recomputed_fidelity"] is None


def test_verifier_diagnose_task_passes_with_no_fidelity():
    """Diagnose tasks have no fidelity to gate on — accept on non-empty answer."""
    node = make_verifier_node()
    state = _state_with_budget()
    state["task"]["task_type"] = "diagnose"
    state["task"]["target_fidelity"] = 0.0
    state["executor_traces"] = [
        _trace(final_answer="Q73 has the lowest T2; degree-3 is most connected.",
               best=None, steps_fids=[])
    ]
    update = node(state)
    assert update["verification"]["is_satisfied"] is True
    assert update["verification"]["reason"] == "ok"


def test_verifier_diagnose_task_rejects_empty_answer():
    """Diagnose tasks still need an answer — empty trace text fails."""
    node = make_verifier_node()
    state = _state_with_budget()
    state["task"]["task_type"] = "diagnose"
    state["task"]["target_fidelity"] = 0.0
    state["executor_traces"] = [_trace(final_answer="", best=None, steps_fids=[])]
    update = node(state)
    assert update["verification"]["is_satisfied"] is False
    assert "below_target" in update["verification"]["reason"]


def test_verifier_target_zero_passes_without_fidelity():
    """target_fidelity=0 means fidelity isn't gating — non-empty answer suffices."""
    node = make_verifier_node()
    state = _state_with_budget()
    state["task"]["task_type"] = "benchmark"
    state["task"]["target_fidelity"] = 0.0
    state["executor_traces"] = [
        _trace(final_answer="Run complete; budget exhausted before any fid measurement.",
               best=None, steps_fids=[])
    ]
    update = node(state)
    assert update["verification"]["is_satisfied"] is True


# ── existing tests below ──────────────────────────────────────────────────


def test_verifier_rejection_drives_full_replan_retry_through_graph():
    """Wire a real verifier into the graph; first executor produces a hallucinated
    trace, second produces a clean one; verify the loop runs exactly once."""

    exec_calls = {"n": 0}

    def stub_executor(state):
        exec_calls["n"] += 1
        if exec_calls["n"] == 1:
            # Hallucinated: claim 0.95, but max observed 0.6
            t = _trace(final_answer="Fidelity 0.95", best=0.95, steps_fids=[0.5, 0.6])
        else:
            # Clean: claim 0.90, observed 0.90
            t = _trace(final_answer="Fidelity 0.90", best=0.90, steps_fids=[0.7, 0.90])
        traces = list(state.get("executor_traces") or []) + [t]
        return {
            "executor_traces": traces,
            "last_action": (t["steps"][-1]["action"] if t["steps"] else ""),
            "last_observation": {},
            "last_fidelity": t.get("best_fidelity"),
            "history": list(state.get("history") or [])
            + [{"node": NODE_EXECUTOR, "ts": 0.0, "info": {"call": exec_calls["n"]}}],
        }

    app = build_graph(executor_fn=stub_executor, use_stub_planner=True)  # default real verifier
    task = make_task("verify-loop", target_fidelity=0.85, max_seconds=60.0, max_tool_calls=20)
    out = app.invoke(new_orchestrator_state(task))

    assert out["done"] is True
    assert exec_calls["n"] == 2
    # Final verification should accept on second pass
    assert out["verification"]["is_satisfied"] is True
    assert out["verification"]["hallucinated"] is False

    visited = [h["node"] for h in out["history"]]
    # Sequence should include: executor → verifier (reject) → reflector → planner → executor → verifier (accept) → finalizer
    assert visited.count(NODE_EXECUTOR) == 2
    assert visited.count(NODE_VERIFIER) == 2
    assert NODE_REFLECTOR in visited
    assert visited[-1] == NODE_FINALIZER


def test_verifier_rejection_finalizes_when_budget_blocks_retry():
    """If retry budget is already exhausted, verifier rejects but graph still finalizes."""

    def stub_executor(state):
        # Always produce a hallucinated trace
        t = _trace(final_answer="Fidelity 0.99", best=0.99, steps_fids=[0.4, 0.5])
        traces = list(state.get("executor_traces") or []) + [t]
        return {
            "executor_traces": traces,
            "last_fidelity": 0.99,
            "history": list(state.get("history") or [])
            + [{"node": NODE_EXECUTOR, "ts": 0.0, "info": {}}],
        }

    app = build_graph(executor_fn=stub_executor, use_stub_planner=True)
    # Pre-set retries to 2 so budget gate blocks immediately
    state = new_orchestrator_state(make_task("retry-blocked", target_fidelity=0.85))
    state["verifier_retry_count"] = 2
    out = app.invoke(state)

    assert out["done"] is True
    assert out["verification"]["is_satisfied"] is False
    visited = [h["node"] for h in out["history"]]
    # No reflector run because retry was blocked
    assert NODE_REFLECTOR not in visited
    assert visited[-1] == NODE_FINALIZER
