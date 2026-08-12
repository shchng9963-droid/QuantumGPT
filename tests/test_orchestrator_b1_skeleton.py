"""Integration test for B1 — orchestrator hello-world.

Goals:
  1. The graph compiles.
  2. ``invoke`` runs end-to-end with stub nodes.
  3. State terminates with ``done=True`` and a non-empty history.
  4. Re-execution loop works (force the verifier to demand a retry once
     and confirm the planner runs again, then we accept on second pass).

These tests pin the topology so future PRs (B2..B6) can swap real impls
into the same slots without breaking anything. They explicitly opt into
the stub executor so they stay fast and offline; the real ReAct-backed
executor is exercised by ``test_orchestrator_b2_executor.py``.
"""

from __future__ import annotations

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
    VerificationResult,
    make_task,
    new_orchestrator_state,
)


def _node_names(history: list[dict]) -> list[str]:
    return [h["node"] for h in history]


def _stub_app(**overrides):
    """All B1 tests run with stub executor + verifier + planner — fully hermetic."""
    return build_graph(
        use_stub_executor=True,
        use_stub_verifier=True,
        use_stub_planner=True,
        **overrides,
    )


# ── happy path ─────────────────────────────────────────────────────────────


def test_graph_compiles_and_runs_helloworld():
    app = _stub_app()
    task = make_task("smoke test the orchestrator", task_type="benchmark")
    state = new_orchestrator_state(task)

    out = app.invoke(state)

    assert out["done"] is True
    assert out["termination_reason"] == "satisfied"
    # Stub executor sets a placeholder final_answer
    assert "stub run" in out["final_answer"]
    # Stub plan should exist and have been bumped to revision 1
    assert out["plan_revision"] == 1
    assert len(out["plan"]) == 1
    # Exactly one execution + one verification on the happy path
    assert len(out["executor_traces"]) == 1
    assert out["verification"]["is_satisfied"] is True


def test_history_records_each_node_exactly_once_on_happy_path():
    app = _stub_app()
    state = new_orchestrator_state(make_task("hi", task_type="benchmark"))
    out = app.invoke(state)

    visited = _node_names(out["history"])
    assert visited == [
        NODE_TASK_ROUTER,
        NODE_PLANNER,
        NODE_EXECUTOR,
        NODE_VERIFIER,
        NODE_FINALIZER,
    ]


def test_budget_walltime_is_populated():
    app = _stub_app()
    state = new_orchestrator_state(make_task("budget test"))
    out = app.invoke(state)

    bs = out["budget_used"]
    assert bs["started_at"] > 0
    assert bs["walltime_seconds"] >= 0.0


# ── re-execution loop ──────────────────────────────────────────────────────


def test_verifier_can_request_retry_and_loop_back():
    """First verifier call rejects → reflector → planner → executor → verifier accepts."""

    calls = {"verifier": 0}

    def reject_then_accept_verifier(state):
        calls["verifier"] += 1
        first_pass = calls["verifier"] == 1
        return {
            "verification": VerificationResult(
                is_satisfied=not first_pass,
                confidence=0.9,
                reason="forced reject" if first_pass else "ok on retry",
                fidelity_check={"claimed": 0.0, "recomputed": 0.0, "abs_diff": 0.0},
                hallucinated=False,
            ),
            "re_execution_needed": first_pass,
            "history": list(state.get("history") or [])
            + [{"node": NODE_VERIFIER, "ts": 0.0, "info": {"forced_reject": first_pass}}],
        }

    app = _stub_app(verifier_fn=reject_then_accept_verifier)
    state = new_orchestrator_state(make_task("retry loop"))
    out = app.invoke(state)

    assert out["done"] is True
    # Verifier should have been called twice (reject, then accept)
    assert calls["verifier"] == 2
    # Plan should have been revised twice (initial + replan)
    assert out["plan_revision"] == 2
    # Executor should have run twice
    assert len(out["executor_traces"]) == 2

    visited = _node_names(out["history"])
    # Expected sequence
    assert visited == [
        NODE_TASK_ROUTER,
        NODE_PLANNER,
        NODE_EXECUTOR,
        NODE_VERIFIER,  # reject
        NODE_REFLECTOR,
        NODE_PLANNER,
        NODE_EXECUTOR,
        NODE_VERIFIER,  # accept
        NODE_FINALIZER,
    ]


def test_verifier_retry_capped_to_one_in_stub_routing():
    """If verifier keeps rejecting, the stub router caps at 1 retry then finalizes."""

    def always_reject(state):
        return {
            "verification": VerificationResult(
                is_satisfied=False,
                confidence=0.5,
                reason="always reject",
                fidelity_check={},
                hallucinated=False,
            ),
            "re_execution_needed": True,
            "history": list(state.get("history") or [])
            + [{"node": NODE_VERIFIER, "ts": 0.0, "info": {"reject": True}}],
        }

    app = _stub_app(verifier_fn=always_reject)
    state = new_orchestrator_state(make_task("infinite reject"))
    out = app.invoke(state)

    assert out["done"] is True
    # Should bail out via finalizer (best-effort) after 1 retry
    visited = _node_names(out["history"])
    assert visited.count(NODE_VERIFIER) == 2
    assert visited[-1] == NODE_FINALIZER


# ── plumbing checks ───────────────────────────────────────────────────────


def test_node_overrides_are_used():
    """Custom node functions must replace the default stubs."""

    captured = {}

    def custom_router(state):
        captured["router_ran"] = True
        return {"history": list(state.get("history") or []) + [{"node": "task_router", "ts": 0, "info": {}}]}

    app = _stub_app(task_router_fn=custom_router)
    out = app.invoke(new_orchestrator_state(make_task("override")))
    assert captured.get("router_ran") is True
    assert out["done"] is True


@pytest.mark.parametrize(
    "task_type",
    ["calibration", "mitigation", "benchmark", "diagnose"],
)
def test_all_advertised_task_types_run_through_stub(task_type: str):
    """B1 stub treats all four task types identically — sanity smoke."""
    app = _stub_app()
    state = new_orchestrator_state(make_task("smoke", task_type=task_type))
    out = app.invoke(state)
    assert out["done"] is True
    assert out["task"]["task_type"] == task_type
