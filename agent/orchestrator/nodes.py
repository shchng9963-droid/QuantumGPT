"""Orchestrator node implementations (Sprint B / v2.5 plan §3.1).

This module is the **stub layer** delivered with B1. Each node is a
plain function ``node(state) -> dict`` that returns the partial update
LangGraph will merge back into ``OrchestratorState``.

Stubs are designed so the hello-world graph compiles, runs end-to-end,
and produces a deterministic final state. Real logic lands in:
  B2 → ExecutorNode (wraps ReActAgent)
  B3 → PlannerNode  (LLM JSON-mode subgoal generation)
  B4 → VerifierNode (deterministic re-simulation)
  B5 → ReflectorNode
  B6 → TaskRouterNode  (real routing)
  B8 → BudgetGuard

Each future PR will replace exactly one stub with the real impl + tests
without touching the graph topology defined in ``graph.py``.
"""

from __future__ import annotations

import time
from typing import Any

from agent.orchestrator.state import (
    OrchestratorState,
    SubGoal,
    VerificationResult,
)


# ── small helpers ──────────────────────────────────────────────────────────


def _record_step(node: str, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a history entry with a UTC-ish wall timestamp."""
    return {
        "node": node,
        "ts": round(time.time(), 3),
        "info": info or {},
    }


def _bump_walltime(state: OrchestratorState) -> dict[str, Any]:
    """Compute updated walltime based on budget.started_at."""
    bs = dict(state.get("budget_used") or {})
    started = bs.get("started_at") or time.time()
    bs["walltime_seconds"] = round(time.time() - started, 3)
    return bs


# ── B6 stub: TaskRouter ────────────────────────────────────────────────────


def task_router(state: OrchestratorState) -> dict[str, Any]:
    """Stub router — passes the task through unchanged.

    Real implementation (B6) will inspect ``task.task_type`` and route to
    one of the four subgraphs.
    """
    task = state.get("task") or {}
    return {
        "history": list(state.get("history") or []) + [
            _record_step("task_router", {"task_type": task.get("task_type", "")})
        ],
    }


# ── B3 stub: Planner ───────────────────────────────────────────────────────


def planner(state: OrchestratorState) -> dict[str, Any]:
    """Stub planner — emits a 1-step plan: just "execute".

    Real impl (B3) will JSON-prompt DeepSeek-Chat to break the task into
    [diagnose_drift, mitigate, run, verify] style subgoals.
    """
    rev = int(state.get("plan_revision") or 0) + 1
    plan: list[SubGoal] = [
        SubGoal(
            id="sg-execute",
            description="Run the task with the flat ReAct executor.",
            intent="execute",
            status="pending",
            notes="stub plan: single execute step",
        )
    ]
    return {
        "plan": plan,
        "plan_revision": rev,
        "history": list(state.get("history") or []) + [
            _record_step("planner", {"plan_revision": rev, "n_subgoals": len(plan)})
        ],
    }


# ── B2 stub: Executor ──────────────────────────────────────────────────────


def executor(state: OrchestratorState) -> dict[str, Any]:
    """Stub executor — produces an empty trace + zero fidelity.

    Real impl (B2) wraps the existing ``ReActAgent`` and stores
    ``AgentTrace.to_dict()`` into ``executor_traces``.
    """
    fake_trace = {
        "user_prompt": (state.get("task") or {}).get("user_prompt", ""),
        "model": "stub",
        "provider": "stub",
        "backend": (state.get("task") or {}).get("backend_id", "mock"),
        "final_answer": "stub run: no real execution",
        "best_fidelity": 0.0,
        "num_tool_calls": 0,
        "elapsed_seconds": 0.0,
        "cost_summary": {"total_cost_usd": 0.0, "total_tokens": 0},
        "steps": [],
    }
    traces = list(state.get("executor_traces") or []) + [fake_trace]
    return {
        "executor_traces": traces,
        "last_action": "",
        "last_observation": {},
        "last_fidelity": 0.0,
        "budget_used": _bump_walltime(state),
        "history": list(state.get("history") or []) + [
            _record_step("executor", {"trace_idx": len(traces) - 1, "fidelity": 0.0})
        ],
    }


# ── B4 stub: Verifier ──────────────────────────────────────────────────────


def verifier(state: OrchestratorState) -> dict[str, Any]:
    """Stub verifier — accepts any execution.

    Real impl (B4) will:
      - re-simulate the last circuit deterministically
      - compare against ``last_fidelity``
      - flag hallucination if abs(claimed - recomputed) > threshold
    """
    fid = state.get("last_fidelity")
    result = VerificationResult(
        is_satisfied=True,
        confidence=1.0,
        reason="stub verifier always accepts",
        fidelity_check={"claimed": fid, "recomputed": fid, "abs_diff": 0.0},
        hallucinated=False,
    )
    return {
        "verification": result,
        "re_execution_needed": False,
        "history": list(state.get("history") or []) + [
            _record_step("verifier", {"satisfied": True})
        ],
    }


# ── B5 stub: Reflector ─────────────────────────────────────────────────────


def reflector(state: OrchestratorState) -> dict[str, Any]:
    """Stub reflector — increments the retry counter (single source of truth).

    The retry counter is owned by the reflector: each reflection ⇔ one
    full replan-execute-verify cycle attempted. Verifier nodes should
    *not* mutate ``verifier_retry_count``.
    """
    retries = int(state.get("verifier_retry_count") or 0) + 1
    return {
        "reflections": list(state.get("reflections") or []),
        "memory_writes": list(state.get("memory_writes") or []),
        "verifier_retry_count": retries,
        "history": list(state.get("history") or []) + [
            _record_step("reflector", {"new_notes": 0, "retry_count": retries})
        ],
    }


# ── Finalizer (always runs before END) ─────────────────────────────────────


def finalizer(state: OrchestratorState) -> dict[str, Any]:
    """Mark the run done and freeze final_answer from the last trace."""
    traces = state.get("executor_traces") or []
    final = ""
    if traces:
        final = traces[-1].get("final_answer", "") or ""
    bs = _bump_walltime(state)
    reason = state.get("termination_reason") or "satisfied"
    return {
        "done": True,
        "termination_reason": reason,
        "final_answer": final,
        "budget_used": bs,
        "history": list(state.get("history") or []) + [
            _record_step("finalizer", {"reason": reason})
        ],
    }


# ── Routing predicates ─────────────────────────────────────────────────────


def route_after_verifier(state: OrchestratorState) -> str:
    """Decide what happens after Verifier.

    - satisfied=True → finalize
    - re_execution_needed=True and budget not exhausted → reflector → planner
    - otherwise → finalize (best-effort)
    """
    v = state.get("verification") or {}
    if v.get("is_satisfied"):
        return "finalize"

    if state.get("re_execution_needed"):
        # check budget; stub doesn't really track, default to allow once
        retries = int(state.get("verifier_retry_count") or 0)
        if retries < 1:  # max 1 replan in stub
            return "reflect"
    return "finalize"


def route_after_reflector(state: OrchestratorState) -> str:
    """After reflection we always replan once, then re-execute."""
    return "replan"
