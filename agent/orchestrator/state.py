"""Orchestrator state schema (Sprint B / v2.5 plan §3.1).

The state is a TypedDict so LangGraph can merge node updates by key.
We deliberately keep nested objects as plain dicts (not dataclasses) inside
the state so the JSON serializer LangGraph uses for checkpoints stays happy.

Field-by-field semantics:
  task              : TaskSpec — input, immutable across the run.
  backend_snapshot  : dict — frozen at task admission for reproducibility.
  plan              : list[SubGoal] — written by Planner; empty == no plan yet.
  plan_revision     : int — incremented every time Planner re-runs.
  executor_traces   : list[dict] — each is AgentTrace.to_dict() from a ReAct run.
  last_observation  : dict — last tool observation surfaced by Executor.
  last_action       : str — name of last tool called, "" if none.
  last_fidelity     : float | None — best_fidelity reported by last execution.
  verification      : VerificationResult — written by Verifier.
  re_execution_needed : bool — Verifier asks for re-run.
  reflections       : list[str] — short notes from Reflector.
  memory_writes     : list[dict] — proposed memory entries (key/value/scope).
  budget_used       : BudgetState — accumulated cost/tokens/walltime.
  history           : list[dict] — chronological node executions for tracing.
  done              : bool — terminal flag, edges check this to route to END.
  termination_reason : str — "satisfied" | "budget_exhausted" | "best_effort"
                       | "verifier_max_retries" | "" while running.

The TaskSpec / SubGoal / VerificationResult / BudgetState helper TypedDicts
are exported as the canonical shapes; we never instantiate dataclasses.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict


class TaskSpec(TypedDict, total=False):
    """A unit of work the orchestrator runs end-to-end.

    Mandatory:
      task_id          : stable ID across logs.
      task_type        : "calibration" | "mitigation" | "benchmark" | "diagnose".
      user_prompt      : natural-language description forwarded to Executor.

    Optional:
      circuit          : circuit name (e.g. "ghz_5", "qft_ent_5"). Used by router.
      target_fidelity  : float, default 0.85.
      profile          : drift profile id ("static", "mild_gradual", "multi_shock").
      backend_id       : backend identifier (mock / real).
      max_usd          : per-task hard cost cap.
      max_seconds      : wall time limit.
      max_tool_calls   : tool-call ceiling per execution.
      meta             : free-form dict for benchmark bookkeeping.
    """

    task_id: str
    task_type: str
    user_prompt: str
    circuit: str
    target_fidelity: float
    profile: str
    backend_id: str
    max_usd: float
    max_seconds: float
    max_tool_calls: int
    meta: dict[str, Any]


class SubGoal(TypedDict, total=False):
    """One step in a Planner-emitted plan."""

    id: str
    description: str
    intent: str  # "diagnose" | "mitigate" | "run" | "verify" | ...
    status: str  # "pending" | "in_progress" | "done" | "skipped"
    notes: str


class VerificationResult(TypedDict, total=False):
    """Output of Verifier."""

    is_satisfied: bool
    confidence: float
    reason: str
    fidelity_check: dict[str, Any]  # {claimed, recomputed, abs_diff}
    hallucinated: bool


class BudgetState(TypedDict, total=False):
    """Running totals, summed across nodes."""

    tokens_in: int
    tokens_out: int
    cost_usd: float
    walltime_seconds: float
    tool_calls: int
    started_at: float
    deadline_at: float | None


class OrchestratorState(TypedDict, total=False):
    """LangGraph state — see module docstring for field semantics."""

    task: TaskSpec
    backend_snapshot: dict[str, Any]

    plan: list[SubGoal]
    plan_revision: int

    executor_traces: list[dict[str, Any]]
    last_observation: dict[str, Any]
    last_action: str
    last_fidelity: float | None

    verification: VerificationResult
    re_execution_needed: bool
    verifier_retry_count: int

    reflections: list[str]
    memory_writes: list[dict[str, Any]]

    budget_used: BudgetState
    history: list[dict[str, Any]]

    done: bool
    termination_reason: str
    final_answer: str


# ── helpers ────────────────────────────────────────────────────────────────


def new_budget_state(max_seconds: float | None = None) -> BudgetState:
    """Initial budget snapshot. ``deadline_at`` is None when no wall cap."""
    started = time.time()
    return BudgetState(
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        walltime_seconds=0.0,
        tool_calls=0,
        started_at=started,
        deadline_at=(started + max_seconds) if max_seconds else None,
    )


def new_orchestrator_state(task: TaskSpec) -> OrchestratorState:
    """Bootstrap a fresh state for a given task."""
    return OrchestratorState(
        task=task,
        backend_snapshot={},
        plan=[],
        plan_revision=0,
        executor_traces=[],
        last_observation={},
        last_action="",
        last_fidelity=None,
        verification=VerificationResult(),
        re_execution_needed=False,
        verifier_retry_count=0,
        reflections=[],
        memory_writes=[],
        budget_used=new_budget_state(task.get("max_seconds")),
        history=[],
        done=False,
        termination_reason="",
        final_answer="",
    )


def make_task(
    user_prompt: str,
    *,
    task_type: str = "benchmark",
    circuit: str = "",
    profile: str = "static",
    target_fidelity: float = 0.85,
    backend_id: str = "mock",
    max_usd: float = 0.50,
    max_seconds: float = 120.0,
    max_tool_calls: int = 20,
    meta: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> TaskSpec:
    """Convenience builder so tests don't have to hand-roll dicts."""
    return TaskSpec(
        task_id=task_id or f"task_{uuid.uuid4().hex[:8]}",
        task_type=task_type,
        user_prompt=user_prompt,
        circuit=circuit,
        target_fidelity=target_fidelity,
        profile=profile,
        backend_id=backend_id,
        max_usd=max_usd,
        max_seconds=max_seconds,
        max_tool_calls=max_tool_calls,
        meta=meta or {},
    )
