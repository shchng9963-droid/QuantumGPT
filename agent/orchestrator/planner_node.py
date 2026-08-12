"""PlannerNode — turn a TaskSpec into an ordered list of SubGoals (B3).

Design goals:
  1. **Cheap by default** — Planner is on the hot path, so it must be
     fast and not waste tokens. We constrain the LLM to JSON output and
     budget ~600 tokens.
  2. **Always succeeds** — if the LLM is unreachable or the JSON is
     malformed, fall back to a deterministic rule-based planner. Bad
     plans are fine; *no plan* would deadlock the graph.
  3. **Pluggable** — accepts a ``llm_call`` callable so tests stay
     offline and so we can later swap in a smaller, cheaper model
     without touching graph wiring.

Subgoal vocabulary (intent enum) is fixed:
    diagnose   — gather info about backend/circuit/drift
    mitigate   — apply error mitigation (transpile / ZNE / dd / etc.)
    run        — execute the circuit and report fidelity
    verify     — explicit re-check (rare; verifier node usually handles)
    finalize   — synthesize the final answer

The Planner does *not* execute anything; it only emits subgoals. Execution
is the Executor's job. Subgoals are advisory — they appear in the
memory_context the Executor passes to ReActAgent.

Re-planning behavior:
  - First call: emits the initial plan from the task description.
  - Subsequent calls (after Reflector): refines based on prior trace +
    reflections. The schema is identical; only the prompt changes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Optional

from agent.orchestrator.state import OrchestratorState, SubGoal

logger = logging.getLogger(__name__)


# ── intent enum ────────────────────────────────────────────────────────────


VALID_INTENTS = ("diagnose", "mitigate", "run", "verify", "finalize")


# ── LLM client ─────────────────────────────────────────────────────────────


LLMCall = Callable[[list[dict[str, str]]], str]
"""Takes OpenAI-style chat messages, returns the assistant's raw string."""


def make_deepseek_call(
    *,
    model: str = "deepseek-chat",
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    max_tokens: int = 600,
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> LLMCall:
    """Return an LLMCall that talks to DeepSeek (or any OpenAI-compatible API).

    Pulls api_key from arg or DEEPSEEK_API_KEY env. Raises on missing key
    only at first call time (so tests that don't trigger LLM never need keys).
    """
    cached: dict[str, Any] = {}

    def _call(messages: list[dict[str, str]]) -> str:
        if "client" not in cached:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai package missing for DeepSeek") from exc
            key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError("DEEPSEEK_API_KEY not set")
            cached["client"] = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        client = cached["client"]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    return _call


# ── prompts ───────────────────────────────────────────────────────────────


_SYSTEM = """\
You are a planner for a quantum-computing AI agent. Decompose the user's
task into a short ordered list of subgoals using ONLY these intents:

  diagnose   — gather info about backend health, drift, qubit properties
  mitigate   — transpile / apply error mitigation (ZNE) / dynamical decoupling
  run        — execute the circuit and observe the result
  verify     — explicitly re-check a claimed result (rare)
  finalize   — synthesize the final answer for the user

Rules:
  * 1 to 5 subgoals total. Prefer fewer.
  * The last subgoal MUST be "finalize".
  * Each subgoal needs a concise description (≤ 12 words).
  * Output ONLY a JSON object with this shape:

    {"subgoals": [
       {"id": "s1", "intent": "diagnose", "description": "..."},
       ...
       {"id": "sN", "intent": "finalize", "description": "..."}
    ]}

  * No prose outside the JSON. No commentary. No markdown fences.
"""


def _build_initial_prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    user_lines = [
        f"Task type: {task.get('task_type', 'benchmark')}",
        f"User prompt: {task.get('user_prompt', '')}",
    ]
    if task.get("circuit"):
        user_lines.append(f"Circuit: {task['circuit']}")
    if task.get("profile"):
        user_lines.append(f"Drift profile: {task['profile']}")
    if task.get("target_fidelity") is not None:
        user_lines.append(f"Target fidelity: {task['target_fidelity']}")
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def _build_replan_prompt(
    task: dict[str, Any],
    last_trace: dict[str, Any] | None,
    last_verification: dict[str, Any] | None,
    reflections: list[str],
) -> list[dict[str, str]]:
    parts = [f"Task: {task.get('user_prompt', '')}"]
    if last_trace:
        parts.append(
            f"Previous run: best_fidelity={last_trace.get('best_fidelity')}, "
            f"tool_calls={last_trace.get('num_tool_calls', 0)}"
        )
        # Surface the last 3 actions so planner can see what was tried
        actions = [
            s.get("action") for s in (last_trace.get("steps") or [])[-3:] if s.get("action")
        ]
        if actions:
            parts.append("Recent actions: " + ", ".join(actions))
    if last_verification:
        parts.append(f"Verifier said: {last_verification.get('reason', '?')}")
    if reflections:
        parts.append("Recent reflections:")
        for r in reflections[-3:]:
            parts.append(f"  - {r}")
    parts.append("\nProduce a refined plan that addresses the verifier's concern.")
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


# ── parsing ────────────────────────────────────────────────────────────────


def _strip_code_fences(text: str) -> str:
    """Remove any markdown code fences the LLM may have produced anyway."""
    t = text.strip()
    if t.startswith("```"):
        # drop opening fence (```json or ```)
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -3].rstrip()
    return t


def parse_plan_json(text: str) -> Optional[list[SubGoal]]:
    """Parse LLM output into a list of SubGoals.

    Tolerant of:
      - leading/trailing whitespace
      - ```json fences (despite our prompt forbidding them)
      - extra fields per subgoal (we keep id/intent/description/notes)

    Returns None if structure is unrecoverable.
    """
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("subgoals")
    if not isinstance(raw, list) or not raw:
        return None

    plan: list[SubGoal] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        intent = str(item.get("intent", "")).strip().lower()
        if intent not in VALID_INTENTS:
            # Unknown intent → drop subgoal (safer than guessing)
            continue
        plan.append(
            SubGoal(
                id=str(item.get("id") or f"s{i + 1}"),
                intent=intent,
                description=str(item.get("description") or "")[:200],
                notes=str(item.get("notes") or "")[:200],
                status="pending",
            )
        )

    if not plan:
        return None

    # Enforce: last subgoal MUST be finalize.
    if plan[-1]["intent"] != "finalize":
        plan.append(
            SubGoal(
                id=f"s{len(plan) + 1}",
                intent="finalize",
                description="Synthesize the final answer.",
                status="pending",
            )
        )

    # Cap at 5 (mirror prompt rule)
    if len(plan) > 5:
        plan = plan[:4] + [plan[-1]]

    return plan


# ── rule-based fallback ───────────────────────────────────────────────────


def rule_based_plan(task: dict[str, Any]) -> list[SubGoal]:
    """Deterministic fallback planner.

    Uses task_type + a tiny keyword scan over user_prompt to pick a sensible
    skeleton. Works without any LLM.
    """
    ttype = (task.get("task_type") or "").lower()
    prompt = (task.get("user_prompt") or "").lower()

    plan: list[SubGoal] = []

    def _add(intent: str, desc: str):
        plan.append(
            SubGoal(
                id=f"s{len(plan) + 1}",
                intent=intent,
                description=desc,
                status="pending",
                notes="rule-fallback",
            )
        )

    has_drift_kw = any(k in prompt for k in ("drift", "calibrat", "t1", "t2", "rabi", "ramsey"))
    has_mitigate_kw = any(k in prompt for k in ("mitigat", "zne", "transpile"))
    has_run_kw = any(k in prompt for k in ("run", "fidelity", "execute", "circuit"))

    if ttype == "calibration" or has_drift_kw:
        _add("diagnose", "Probe backend health and qubit properties.")

    if ttype == "diagnose":
        _add("diagnose", "Inspect backend snapshot and recent drift.")

    if ttype in ("mitigation", "benchmark") or has_mitigate_kw or has_run_kw:
        if not plan:
            _add("diagnose", "Check backend health before running.")
        if ttype == "mitigation" or has_mitigate_kw:
            _add("mitigate", "Transpile and apply error mitigation.")
        _add("run", "Execute the circuit and report fidelity.")

    # Default skeleton if nothing matched
    if not plan:
        _add("diagnose", "Check backend health.")
        _add("run", "Execute the requested circuit.")

    _add("finalize", "Summarize the result for the user.")
    return plan


# ── the node ──────────────────────────────────────────────────────────────


def make_planner_node(
    *,
    llm_call: Optional[LLMCall] = None,
    use_llm_when: Callable[[OrchestratorState], bool] | None = None,
) -> Callable[[OrchestratorState], dict[str, Any]]:
    """Build a planner node.

    Args:
      llm_call: optional LLMCall. If omitted, we lazily build a DeepSeek
        client when the task asks for an LLM-backed plan.
      use_llm_when: predicate to decide whether to call the LLM. Defaults
        to: provider != "mock" and (task_type != "benchmark" OR plan_revision > 0).
        The default keeps simple benchmark tasks cheap on the first pass.
    """
    # Use a mutable holder so we can lazy-init the default DeepSeek client.
    holder: dict[str, Any] = {"llm": llm_call}

    def _resolve_llm() -> Optional[LLMCall]:
        if holder["llm"] is not None:
            return holder["llm"]
        try:
            holder["llm"] = make_deepseek_call()
            return holder["llm"]
        except Exception as exc:
            logger.warning("Planner: LLM init failed (%s) — using rule fallback", exc)
            holder["llm"] = None
            return None

    def _should_use_llm(state: OrchestratorState) -> bool:
        if use_llm_when is not None:
            return use_llm_when(state)
        task = state.get("task") or {}
        provider = (task.get("provider") or task.get("meta", {}).get("provider") or "").lower()
        if provider == "mock":
            return False
        # On replans, always use LLM if available (need to react to verifier).
        if int(state.get("plan_revision") or 0) > 0:
            return True
        # On first pass of simple benchmarks, rule fallback is sufficient.
        ttype = (task.get("task_type") or "").lower()
        return ttype != "benchmark"

    def planner_node(state: OrchestratorState) -> dict[str, Any]:
        task = dict(state.get("task") or {})
        prev_rev = int(state.get("plan_revision") or 0)
        is_replan = prev_rev > 0
        history = list(state.get("history") or [])

        plan: list[SubGoal] | None = None
        source = "rule"
        llm_error: str | None = None

        if _should_use_llm(state):
            llm = _resolve_llm()
            if llm is not None:
                if is_replan:
                    msgs = _build_replan_prompt(
                        task,
                        (state.get("executor_traces") or [None])[-1],
                        state.get("verification"),
                        state.get("reflections") or [],
                    )
                else:
                    msgs = _build_initial_prompt(task)
                try:
                    raw = llm(msgs)
                    plan = parse_plan_json(raw)
                    if plan is not None:
                        source = "llm"
                    else:
                        llm_error = "json_parse_failed"
                except Exception as exc:
                    llm_error = f"llm_call_error: {str(exc)[:120]}"
                    logger.warning("Planner LLM call failed: %s", exc)

        if plan is None:
            plan = rule_based_plan(task)
            if source != "llm":
                source = "rule" if llm_error is None else f"rule_after_{llm_error}"

        new_rev = prev_rev + 1
        return {
            "plan": plan,
            "plan_revision": new_rev,
            "history": history
            + [
                {
                    "node": "planner",
                    "ts": round(time.time(), 3),
                    "info": {
                        "source": source,
                        "n_subgoals": len(plan),
                        "plan_revision": new_rev,
                        "is_replan": is_replan,
                        "llm_error": llm_error,
                    },
                }
            ],
        }

    return planner_node


# Default node: lazy DeepSeek + the smart "use LLM when" rule
planner_node = make_planner_node()


__all__ = [
    "make_planner_node",
    "planner_node",
    "make_deepseek_call",
    "parse_plan_json",
    "rule_based_plan",
    "VALID_INTENTS",
    "LLMCall",
]
