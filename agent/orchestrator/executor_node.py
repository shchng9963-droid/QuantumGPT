"""ExecutorNode — wraps the existing ReActAgent for the orchestrator (B2).

Contract (state in → state update out):
  Reads
    state["task"]                 — TaskSpec; provides user_prompt + budgets
    state["task"]["backend_id"]   — used by default_backend_factory
    state["task"]["provider"]     — used by default_agent_factory ("mock"/"deepseek")
    state["budget_used"]          — running totals; we add to them
    state["plan"]                 — surfaced as memory_context (best-effort)

  Writes
    state["executor_traces"]      — appends AgentTrace.to_dict()
    state["last_action"]          — last tool name (or "")
    state["last_observation"]     — small dict {"tool", "fidelity", "result"}
    state["last_fidelity"]        — float | None (best fidelity reported)
    state["budget_used"]          — totals updated (tokens, cost, walltime, calls)
    state["history"]              — appends one entry
    state["re_execution_needed"]  — True iff agent.run raised; otherwise unchanged

Why a factory?
  In tests we want determinism (provider="mock") and no network. In paired
  benchmarks we want real DeepSeek/OpenAI. Factories let the orchestrator
  callers swap one for the other without touching graph wiring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

from agent.orchestrator.state import OrchestratorState

logger = logging.getLogger(__name__)

FIRST_ATTEMPT_RETRY_MARGIN = 0.80


# ── factory protocol ──────────────────────────────────────────────────────


BackendFactory = Callable[[dict[str, Any]], Any]
"""Builds a ShadowBackend from a TaskSpec dict (or compatible)."""

AgentFactory = Callable[[Any, dict[str, Any]], Any]
"""Builds something with a ``.run(prompt, memory_context="") -> AgentTrace``
interface, given (backend, task)."""


def default_backend_factory(task: dict[str, Any]):
    """Default: use FakeBackendAdapter with task['backend_id'] (fall back to FakeBrisbane)."""
    from backends.fake_adapter import FakeBackendAdapter

    backend_id = task.get("backend_id") or "FakeBrisbane"
    if backend_id in ("mock", ""):
        backend_id = "FakeBrisbane"
    return FakeBackendAdapter(backend_id)


def default_agent_factory(backend: Any, task: dict[str, Any]):
    """Default: ReActAgent with provider from task (mock/deepseek/auto).

    Mock is preferred when ``provider == "mock"`` since it makes tests
    deterministic and offline.
    """
    from agent.factory import build_react_agent

    provider = task.get("provider") or task.get("meta", {}).get("provider") or "mock"
    max_calls = int(task.get("max_tool_calls") or 12)

    return build_react_agent(
        backend,
        provider=provider,
        model=("deepseek-chat" if provider == "deepseek" else "rule-planner-v1"),
        verbose=False,
        target_fidelity=float(task.get("target_fidelity") or 0.85),
        max_tool_calls=max_calls,
        max_seconds=float(task.get("max_seconds") or 60.0),
        use_memory=False,
    )


# ── trace helpers ─────────────────────────────────────────────────────────


def _trace_to_dict(trace: Any) -> dict[str, Any]:
    """Convert AgentTrace (or trace-like) into a plain JSON-friendly dict."""
    if hasattr(trace, "to_dict"):
        try:
            return trace.to_dict()
        except Exception as exc:  # defensive: keep orchestrator going
            logger.warning("trace.to_dict() failed: %s", exc)

    if is_dataclass(trace):
        return asdict(trace)

    if isinstance(trace, dict):
        return trace

    # Final fallback — best-effort string repr
    return {"final_answer": str(trace), "best_fidelity": 0.0}


def _trace_best_fidelity(trace: Any) -> Optional[float]:
    """Pull best fidelity from a trace, supporting both objects and dicts."""
    if hasattr(trace, "best_fidelity"):
        try:
            return float(trace.best_fidelity)
        except (TypeError, ValueError):
            pass
    if isinstance(trace, dict):
        bf = trace.get("best_fidelity")
        if bf is not None:
            try:
                return float(bf)
            except (TypeError, ValueError):
                return None
        bs = trace.get("budget_summary") or {}
        if "best_fidelity" in bs:
            try:
                return float(bs["best_fidelity"])
            except (TypeError, ValueError):
                return None
    return None


def _trace_last_step(trace_dict: dict[str, Any]) -> dict[str, Any]:
    """Return ``{tool, fidelity, observation}`` for the last actionable step."""
    steps = trace_dict.get("steps") or []
    for s in reversed(steps):
        if s.get("action"):
            return {
                "tool": s.get("action") or "",
                "fidelity": s.get("fidelity_observed"),
                "observation": (s.get("observation") or "")[:500],
                "step_num": s.get("step_num"),
            }
    return {"tool": "", "fidelity": None, "observation": "", "step_num": None}


def _trace_cost(trace_dict: dict[str, Any]) -> dict[str, float]:
    """Pull (tokens_in, tokens_out, cost_usd, elapsed_seconds, tool_calls) from a trace dict."""
    cost = trace_dict.get("cost_summary") or {}
    return {
        "tokens_in": int(cost.get("prompt_tokens") or trace_dict.get("prompt_tokens_total") or 0),
        "tokens_out": int(cost.get("completion_tokens") or trace_dict.get("completion_tokens_total") or 0),
        "cost_usd": float(cost.get("total_cost_usd") or 0.0),
        "elapsed": float(trace_dict.get("elapsed_seconds") or 0.0),
        "tool_calls": int(trace_dict.get("num_tool_calls") or 0),
    }


def _accumulate_budget(prev: dict[str, Any] | None, addition: dict[str, float]) -> dict[str, Any]:
    bs = dict(prev or {})
    bs["tokens_in"] = int(bs.get("tokens_in") or 0) + addition["tokens_in"]
    bs["tokens_out"] = int(bs.get("tokens_out") or 0) + addition["tokens_out"]
    bs["cost_usd"] = float(bs.get("cost_usd") or 0.0) + addition["cost_usd"]
    bs["tool_calls"] = int(bs.get("tool_calls") or 0) + addition["tool_calls"]
    started = bs.get("started_at") or time.time()
    bs["started_at"] = started
    bs["walltime_seconds"] = round(time.time() - started, 3)
    bs["deadline_at"] = bs.get("deadline_at")
    return bs


def _format_memory_context(state: OrchestratorState) -> str:
    """Best-effort serialization of plan + recent reflections for the agent prompt.

    The legacy ReActAgent expects a string. We keep this short to avoid
    blowing the prompt budget.
    """
    parts: list[str] = []
    plan = state.get("plan") or []
    if plan:
        parts.append("Plan:")
        for sg in plan[:6]:
            parts.append(f"- [{sg.get('intent', '?')}] {sg.get('description', '')}".strip())
    refl = state.get("reflections") or []
    if refl:
        parts.append("Recent notes:")
        for note in refl[-3:]:
            parts.append(f"- {note}")
    return "\n".join(parts)


def _remaining_tool_budget(task: dict[str, Any], state: OrchestratorState) -> int:
    """Project global tool budget into a per-execution allowance.

    First attempt keeps 20% headroom so the verifier can trigger one more pass.
    Retry attempts consume only the globally remaining tool-call budget.
    """
    max_calls = max(0, int(task.get("max_tool_calls") or 0))
    used_calls = max(0, int((state.get("budget_used") or {}).get("tool_calls") or 0))
    retries = max(0, int(state.get("verifier_retry_count") or 0))

    if max_calls <= 0:
        return 0

    if retries > 0:
        return max(0, max_calls - used_calls)

    reserved = max(1, int(max_calls * FIRST_ATTEMPT_RETRY_MARGIN))
    return min(max_calls, reserved)


# ── the node ──────────────────────────────────────────────────────────────


def make_executor_node(
    *,
    backend_factory: BackendFactory = default_backend_factory,
    agent_factory: AgentFactory = default_agent_factory,
) -> Callable[[OrchestratorState], dict[str, Any]]:
    """Return a LangGraph node function that runs ReActAgent against the task."""

    def executor_node(state: OrchestratorState) -> dict[str, Any]:
        task = dict(state.get("task") or {})
        prompt = task.get("user_prompt") or ""
        history = list(state.get("history") or [])

        try:
            task["max_tool_calls"] = _remaining_tool_budget(task, state)
            backend = backend_factory(task)
            agent = agent_factory(backend, task)
            memory_ctx = _format_memory_context(state)
            t0 = time.time()
            trace = agent.run(prompt, memory_context=memory_ctx)
            elapsed = time.time() - t0

            trace_dict = _trace_to_dict(trace)
            # Stamp orchestrator-side wall time in case the trace's own field is empty
            trace_dict.setdefault("elapsed_seconds", round(elapsed, 3))
            trace_dict.setdefault("backend", task.get("backend_id") or "FakeBrisbane")

            best_fid = _trace_best_fidelity(trace) or _trace_best_fidelity(trace_dict)
            last = _trace_last_step(trace_dict)
            cost = _trace_cost(trace_dict)

            traces = list(state.get("executor_traces") or []) + [trace_dict]

            update: dict[str, Any] = {
                "executor_traces": traces,
                "last_action": last["tool"],
                "last_observation": last,
                "last_fidelity": best_fid,
                "budget_used": _accumulate_budget(state.get("budget_used"), cost),
                "history": history
                + [
                    {
                        "node": "executor",
                        "ts": round(time.time(), 3),
                        "info": {
                            "trace_idx": len(traces) - 1,
                            "fidelity": best_fid,
                            "tool_calls": cost["tool_calls"],
                            "elapsed_s": round(cost["elapsed"], 3),
                            "cost_usd": cost["cost_usd"],
                        },
                    }
                ],
                # Successful execution clears any prior re-execution flag
                "re_execution_needed": False,
            }
            return update

        except Exception as exc:  # do not crash the graph
            logger.exception("ExecutorNode failed: %s", exc)
            err_trace = {
                "user_prompt": prompt,
                "model": task.get("provider", "unknown"),
                "provider": task.get("provider", "unknown"),
                "backend": task.get("backend_id", ""),
                "final_answer": f"[executor error] {exc}",
                "best_fidelity": 0.0,
                "num_tool_calls": 0,
                "elapsed_seconds": 0.0,
                "cost_summary": {"total_cost_usd": 0.0},
                "steps": [],
                "error": str(exc)[:500],
            }
            traces = list(state.get("executor_traces") or []) + [err_trace]
            return {
                "executor_traces": traces,
                "last_action": "",
                "last_observation": {"error": str(exc)[:200]},
                "last_fidelity": 0.0,
                "re_execution_needed": True,
                "history": history
                + [
                    {
                        "node": "executor",
                        "ts": round(time.time(), 3),
                        "info": {"error": str(exc)[:200]},
                    }
                ],
            }

    return executor_node


# Default node — used by graph.build_graph() when no override is passed.
executor_node = make_executor_node()


__all__ = [
    "make_executor_node",
    "executor_node",
    "default_backend_factory",
    "default_agent_factory",
    "BackendFactory",
    "AgentFactory",
]
