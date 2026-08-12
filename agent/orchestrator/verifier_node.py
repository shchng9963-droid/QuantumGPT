"""VerifierNode — does the executor's claim hold up? (B4)

The verifier is the orchestrator's truth-check. Its job is *not* to
re-do the agent's work; it is to detect three failure modes cheaply:

  1. **Hallucinated fidelity** — final_answer claims a number that
     no step actually observed. Detected via trace-claim consistency.
  2. **Below-target outcome** — trace's best_fidelity < task.target.
     Re-execution may help, gated by budget.
  3. **Inconsistent best_fidelity** — trace's headline number is higher
     than any observed step (catches reporting bugs).

A pluggable ``resimulator`` hook can do deterministic re-simulation
or LLM-judge cross-checks. Its signature is::

    resim(trace_dict, task_dict) -> {
        "recomputed_fidelity": float,
        "ok": bool,
        "reason": str,
    }

The default resimulator is a passthrough that trusts the trace
(``recomputed = claimed``); production wiring can replace it.

Output (written to state['verification']):
    is_satisfied : bool      — terminal happy path?
    confidence   : float     — 0..1, derived from how cleanly checks pass
    reason       : str       — short human-readable summary
    fidelity_check : {
        "claimed": float | None,        # in final_answer text
        "best_in_trace": float | None,  # trace.best_fidelity
        "max_observed": float | None,   # max(step.fidelity_observed)
        "recomputed": float | None,     # from resimulator
        "abs_diff": float,
        "target": float,
    }
    hallucinated : bool

Routing knobs (also written):
    re_execution_needed : bool — True iff retry is justified AND budget allows
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Optional

from agent.orchestrator.state import OrchestratorState, VerificationResult

logger = logging.getLogger(__name__)


# ── public knobs ──────────────────────────────────────────────────────────


# Tolerance: claim is "consistent" if claim - max_observed <= this.
DEFAULT_HALLUCINATION_TOLERANCE = 0.05

# Tolerance: trace.best_fidelity must not exceed max_observed by more than this.
DEFAULT_TRACE_INTERNAL_TOLERANCE = 0.02

# Resimulator: receives a trace dict and the task dict, returns a verdict dict.
Resimulator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def passthrough_resimulator(trace_dict: dict[str, Any], task_dict: dict[str, Any]) -> dict[str, Any]:
    """No-op resimulator: trusts the trace's reported fidelity."""
    bf = trace_dict.get("best_fidelity")
    if bf is None:
        bs = trace_dict.get("budget_summary") or {}
        bf = bs.get("best_fidelity")
    try:
        bf = float(bf) if bf is not None else None
    except (TypeError, ValueError):
        bf = None
    return {
        "recomputed_fidelity": bf,
        "ok": True,
        "reason": "passthrough",
    }


# ── trace probes ──────────────────────────────────────────────────────────


_FIDELITY_NUM_RE = re.compile(
    r"""
    fidelity                # the word
    \s*                     # optional whitespace
    (?:                     # optional connector phrase
        (?:is|was|of|equals|equal\s+to|reached|achieved|approximately|approx)
        \s+
      |
        [:=≈~≃]+            # punctuation connectors incl. ≈ / ~ / ≃
        \s*
      |
        \s+                 # bare whitespace fallback ("fidelity 0.95")
    )?
    (?P<num>                # capture the number
        \d+(?:\.\d+)?       # int or float
        (?:e[+-]?\d+)?      # optional sci notation
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_claimed_fidelity(text: str) -> Optional[float]:
    """Best-effort extraction of a fidelity figure from agent's final answer.

    Picks the *largest* match (agents typically state their best result).
    Returns None if nothing parseable.
    """
    if not text:
        return None
    candidates: list[float] = []
    for m in _FIDELITY_NUM_RE.finditer(text):
        try:
            v = float(m.group("num"))
        except ValueError:
            continue
        # Sometimes 'fidelity 92%' shows up — normalize to fraction.
        if v > 1.0 and v <= 100.0:
            v = v / 100.0
        if 0.0 <= v <= 1.0:
            candidates.append(v)
    return max(candidates) if candidates else None


def max_observed_step_fidelity(trace_dict: dict[str, Any]) -> Optional[float]:
    """Largest ``fidelity_observed`` across all steps. None if none recorded."""
    seen: list[float] = []
    for s in trace_dict.get("steps") or []:
        v = s.get("fidelity_observed")
        if v is None:
            continue
        try:
            seen.append(float(v))
        except (TypeError, ValueError):
            continue
    return max(seen) if seen else None


def trace_best_fidelity(trace_dict: dict[str, Any]) -> Optional[float]:
    bf = trace_dict.get("best_fidelity")
    if bf is None:
        bs = trace_dict.get("budget_summary") or {}
        bf = bs.get("best_fidelity")
    try:
        return float(bf) if bf is not None else None
    except (TypeError, ValueError):
        return None


# ── budget probe ──────────────────────────────────────────────────────────


def budget_has_room(state: OrchestratorState) -> tuple[bool, str]:
    """Return (has_room, reason).

    Hard caps come from ``task``: ``max_usd``, ``max_seconds``, ``max_tool_calls``.
    Each is checked against ``budget_used``. We refuse retry if any cap is
    within 80% of the limit (leave a margin for the next exec to actually run).

    The ``verifier_retry_count`` state field also bounds infinite loops at
    a hard cap of 2 retries (3 total attempts).
    """
    task = state.get("task") or {}
    bs = state.get("budget_used") or {}
    retries = int(state.get("verifier_retry_count") or 0)

    if retries >= 2:
        return False, f"retry_cap_reached(retries={retries})"

    margin = 0.80
    cost_used = float(bs.get("cost_usd") or 0.0)
    cost_max = float(task.get("max_usd") or 0.0)
    if cost_max > 0 and cost_used >= cost_max * margin:
        return False, f"cost_near_cap({cost_used:.4f}/{cost_max:.4f})"

    started = float(bs.get("started_at") or 0.0)
    walltime = float(bs.get("walltime_seconds") or 0.0)
    if started == 0.0:
        walltime = 0.0
    elapsed = walltime if walltime > 0 else max(time.time() - started, 0.0)
    sec_max = float(task.get("max_seconds") or 0.0)
    if sec_max > 0 and elapsed >= sec_max * margin:
        return False, f"walltime_near_cap({elapsed:.1f}/{sec_max:.1f})"

    calls_used = int(bs.get("tool_calls") or 0)
    calls_max = int(task.get("max_tool_calls") or 0)
    if calls_max > 0 and calls_used >= calls_max * margin:
        return False, f"toolcalls_near_cap({calls_used}/{calls_max})"

    return True, "ok"


# ── the node ──────────────────────────────────────────────────────────────


def make_verifier_node(
    *,
    resimulator: Resimulator = passthrough_resimulator,
    hallucination_tolerance: float = DEFAULT_HALLUCINATION_TOLERANCE,
    internal_tolerance: float = DEFAULT_TRACE_INTERNAL_TOLERANCE,
) -> Callable[[OrchestratorState], dict[str, Any]]:
    """Build a verifier node with optional resimulation and tolerances."""

    def verifier_node(state: OrchestratorState) -> dict[str, Any]:
        traces = state.get("executor_traces") or []
        task = dict(state.get("task") or {})
        target = float(task.get("target_fidelity") or 0.0)
        history = list(state.get("history") or [])

        if not traces:
            # No execution ever happened — verifier can't accept this.
            result = VerificationResult(
                is_satisfied=False,
                confidence=0.0,
                reason="no_executor_trace",
                fidelity_check={
                    "claimed": None,
                    "best_in_trace": None,
                    "max_observed": None,
                    "recomputed": None,
                    "abs_diff": 0.0,
                    "target": target,
                },
                hallucinated=False,
            )
            has_room, _ = budget_has_room(state)
            return {
                "verification": result,
                "re_execution_needed": has_room,
                "history": history
                + [{"node": "verifier", "ts": round(time.time(), 3),
                    "info": {"reason": "no_trace", "retry": has_room}}],
            }

        last = traces[-1]
        claimed = extract_claimed_fidelity(last.get("final_answer", ""))
        best_in_trace = trace_best_fidelity(last)
        max_obs = max_observed_step_fidelity(last)

        # 1) Hallucination check — claim vs observed step fidelities
        hallucinated = False
        halluc_reason = ""
        if claimed is not None and max_obs is not None:
            if claimed - max_obs > hallucination_tolerance:
                hallucinated = True
                halluc_reason = (
                    f"claim {claimed:.3f} exceeds max observed step {max_obs:.3f} "
                    f"by > {hallucination_tolerance}"
                )
        elif claimed is not None and max_obs is None:
            # Agent reports a fidelity number but no tool step ever observed
            # one. Treat as hallucinated unless the trace is empty (handled
            # above) — claim with no evidence is by definition unsupported.
            n_steps = len([s for s in (last.get("steps") or []) if s.get("action")])
            if n_steps > 0:
                hallucinated = True
                halluc_reason = (
                    f"claim {claimed:.3f} has no supporting fidelity_observed "
                    f"in any of {n_steps} tool step(s)"
                )

        # 2) Internal consistency — trace.best_fidelity vs observed
        internal_inconsistent = False
        if best_in_trace is not None and max_obs is not None:
            if best_in_trace - max_obs > internal_tolerance:
                internal_inconsistent = True
                hallucinated = True  # treat as hallucinated for routing
                halluc_reason = (
                    halluc_reason or
                    f"best_fidelity {best_in_trace:.3f} > max observed step {max_obs:.3f}"
                )

        # 3) Resimulation
        try:
            resim = resimulator(last, task)
        except Exception as exc:  # never crash the graph
            logger.warning("resimulator raised: %s", exc)
            resim = {"recomputed_fidelity": best_in_trace, "ok": False, "reason": f"resim_error:{exc}"}
        recomputed = resim.get("recomputed_fidelity")
        try:
            recomputed = float(recomputed) if recomputed is not None else None
        except (TypeError, ValueError):
            recomputed = None
        resim_ok = bool(resim.get("ok", True))
        resim_reason = str(resim.get("reason", ""))

        abs_diff = 0.0
        if claimed is not None and recomputed is not None:
            abs_diff = abs(claimed - recomputed)

        # 4) Target gate
        # Two task semantics:
        #   - "fidelity tasks" (mitigation/benchmark/calibration): need a numeric
        #     fidelity that meets the target. None ⇒ below_target.
        #   - "non-fidelity tasks" (diagnose, or any task with target == 0.0):
        #     fidelity is informational, not gating. Pass as long as the trace
        #     has a non-empty final_answer.
        ttype = (task.get("task_type") or "").lower()
        is_diagnostic = ttype == "diagnose"
        target_is_zero = target <= 0.0

        effective_fid = recomputed if recomputed is not None else best_in_trace
        if is_diagnostic or target_is_zero:
            # Non-fidelity gate: accept if executor produced an answer.
            non_empty_answer = bool((last.get("final_answer") or "").strip())
            meets_target = non_empty_answer
        else:
            meets_target = effective_fid is not None and effective_fid >= target

        # 5) Verdict — satisfied iff: not hallucinated, resim ok, meets target
        is_satisfied = (not hallucinated) and resim_ok and meets_target

        # Confidence heuristic: each passing check contributes.
        confidence = (
            (0.4 if not hallucinated else 0.0)
            + (0.3 if resim_ok else 0.0)
            + (0.3 if meets_target else 0.0)
        )

        if is_satisfied:
            reason = "ok"
        elif hallucinated:
            reason = f"hallucinated: {halluc_reason}"
        elif not resim_ok:
            reason = f"resim_failed: {resim_reason}"
        elif not meets_target:
            reason = f"below_target(eff={effective_fid}, target={target})"
        else:
            reason = "unknown_failure"

        # Routing: re-execute only if (a) not satisfied and (b) budget allows
        re_exec = False
        budget_reason = ""
        if not is_satisfied:
            has_room, budget_reason = budget_has_room(state)
            re_exec = has_room

        result = VerificationResult(
            is_satisfied=is_satisfied,
            confidence=round(confidence, 3),
            reason=reason,
            fidelity_check={
                "claimed": claimed,
                "best_in_trace": best_in_trace,
                "max_observed": max_obs,
                "recomputed": recomputed,
                "abs_diff": round(abs_diff, 4),
                "target": target,
            },
            hallucinated=hallucinated,
        )
        return {
            "verification": result,
            "re_execution_needed": re_exec,
            "history": history
            + [
                {
                    "node": "verifier",
                    "ts": round(time.time(), 3),
                    "info": {
                        "satisfied": is_satisfied,
                        "hallucinated": hallucinated,
                        "claimed": claimed,
                        "max_obs": max_obs,
                        "recomputed": recomputed,
                        "target": target,
                        "retry": re_exec,
                        "budget": budget_reason or "ok",
                        "reason": reason[:120],
                    },
                }
            ],
        }

    return verifier_node


# Default node used by the graph.
verifier_node = make_verifier_node()


__all__ = [
    "make_verifier_node",
    "verifier_node",
    "passthrough_resimulator",
    "extract_claimed_fidelity",
    "max_observed_step_fidelity",
    "trace_best_fidelity",
    "budget_has_room",
    "Resimulator",
    "DEFAULT_HALLUCINATION_TOLERANCE",
    "DEFAULT_TRACE_INTERNAL_TOLERANCE",
]
