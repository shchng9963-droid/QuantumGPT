"""ChatLLM-NoTools baseline (v2.5 Sprint A, A3).

A pure chat baseline: hand the LLM a textual observation (health snapshot,
drift score, action catalog) and ask for a single action name back. No
function-calling, no tool execution. The chosen action's post-drift
fidelity is then read from the oracle JSONL.

Token usage and DeepSeek pricing (0.27 / 1.10 USD per 1M in/out, matching
agent.react.cost_summary) are recorded in the result row, so this baseline
is directly cost-comparable with the full agent.

Why this baseline:
  Agent − Static-Pipeline       isolates LLM + tools + planning + drift-aware
  Agent − ChatLLM-NoTools       isolates tools + planning  (LLM contribution removed)
  ChatLLM-NoTools − Static       isolates pure LLM action choice over the same grid
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from eval.public_mqtbench_agent_eval import PublicAgentRunResult


# Pricing matches agent.react.cost_summary (USD per 1M tokens)
_PRICING = {
    "deepseek": {"input": 0.27, "output": 1.10},
    "openai":   {"input": 3.00, "output": 15.00},
    "mock":     {"input": 0.00, "output": 0.00},
}


_DEFAULT_FALLBACK = "transpile_o1_s1024"


def _build_prompt(oracle_row: dict[str, Any]) -> tuple[str, str]:
    """Build the (system, user) prompt pair shown to the chat LLM.

    Discloses every action and the device's current health, but NOT the
    actions' post-drift fidelities. The LLM has to pick blind, just like
    the full agent before its first tool call.
    """

    actions = [a["action"] for a in oracle_row.get("actions", [])]
    md = oracle_row.get("metadata") or {}
    drift = md.get("drift_score")
    avg_2q = md.get("avg_2q_error")
    avg_t1 = md.get("avg_t1_us")

    sys_prompt = (
        "You are a quantum experiment scheduler. Given a circuit, a backend "
        "health snapshot, and a list of candidate actions, you must pick "
        "exactly one action that you believe maximizes post-drift fidelity. "
        "Respond with strictly valid JSON of the form {\"action\": \"<name>\"} "
        "and nothing else. No prose."
    )

    user_prompt = (
        f"Circuit: {oracle_row.get('circuit')}\n"
        f"Backend: {oracle_row.get('backend', 'FakeBrisbane')}\n"
        f"Drift profile: {oracle_row.get('profile')}\n"
        f"Target fidelity: {oracle_row.get('target_fidelity', 0.85)}\n"
        f"Health: drift_score={drift}, avg_2q_error={avg_2q}, avg_t1_us={avg_t1}\n"
        f"Candidate actions:\n"
        + "\n".join(f"  - {a}" for a in actions)
        + "\n\nReturn JSON: {\"action\": \"<name>\"}"
    )
    return sys_prompt, user_prompt


def _parse_action(raw: str, valid: set[str]) -> tuple[str | None, bool]:
    """Try to parse an action name from the LLM's raw response.

    Strategy:
      1. Try strict JSON parse and pull "action".
      2. If JSON fails, scan the text for any whole-word valid action name.

    Returns (action_or_None, is_valid).
    """

    raw = (raw or "").strip()
    # 1. JSON path
    try:
        obj = json.loads(raw)
        candidate = obj.get("action") if isinstance(obj, dict) else None
        if isinstance(candidate, str) and candidate in valid:
            return candidate, True
        if isinstance(candidate, str):
            return candidate, False  # parsed but unknown name
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # 2. Substring scan — find the first valid action name that appears verbatim
    for name in valid:
        if name in raw:
            return name, True
    return None, False


def run_chat_llm_no_tools_task(
    oracle_row: dict[str, Any],
    *,
    client: Any,
    model: str = "deepseek-chat",
    provider: str = "deepseek",
    fallback_action: str = _DEFAULT_FALLBACK,
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> PublicAgentRunResult:
    """Run the ChatLLM-NoTools baseline against one oracle row.

    The client must be an OpenAI-compatible chat client (matches the
    object built by agent.react when provider='deepseek').
    """

    actions_idx = {a["action"]: a for a in oracle_row.get("actions", [])}
    if not actions_idx:
        raise ValueError("ChatLLM-NoTools requires non-empty oracle_row['actions']")

    sys_prompt, user_prompt = _build_prompt(oracle_row)

    t0 = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed_ms = (time.time() - t0) * 1000.0

    raw_text = completion.choices[0].message.content if completion.choices else ""
    parsed, is_valid = _parse_action(raw_text, set(actions_idx))
    chosen = parsed if (is_valid and parsed in actions_idx) else fallback_action
    if chosen not in actions_idx:
        # If even the fallback is missing, bail loudly — the caller mis-configured us
        raise KeyError(
            f"Fallback action {fallback_action!r} not present in oracle row; "
            f"available={sorted(actions_idx)!r}"
        )

    fid = float(actions_idx[chosen]["fidelity"])

    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

    rates = _PRICING.get(provider, _PRICING["deepseek"])
    cost = (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000

    seed = int(oracle_row.get("seed", 0))
    target = float(oracle_row.get("target_fidelity", 0.85))

    return PublicAgentRunResult(
        system="ChatLLM-NoTools",
        task_id=str(
            oracle_row.get(
                "task_id",
                f"{oracle_row.get('backend', '')}:"
                f"{oracle_row.get('profile', '')}:"
                f"{oracle_row.get('circuit', '')}:seed{seed}",
            )
        ),
        circuit=str(oracle_row.get("circuit", "")),
        backend=str(oracle_row.get("backend", "FakeBrisbane")),
        drift_profile=str(oracle_row.get("profile", "moderate")),
        seed=seed,
        target_fidelity=target,
        oracle_best=float(oracle_row.get("best_post_drift_score", 0.0)),
        oracle_feasible=bool(oracle_row.get("oracle_feasible", False)),
        oracle_best_action=str(oracle_row.get("best_action", "unknown")),
        pre_fidelities=[float(oracle_row.get("pre_fidelity", 0.0))],
        post_fidelities=[fid],
        provider=provider,
        model=model,
        tool_calls=[chosen],
        trace_files=[],
        metadata={
            "execution_mode": "real",
            "use_mock": False,
            "seed": seed,
            "max_turns": 1,
            "max_tool_calls": 0,
            "pre_fidelity_count": 1,
            "post_fidelity_count": 1,
            "pre_final_answer_chars": 0,
            "post_final_answer_chars": len(raw_text or ""),
            "elapsed_seconds": float(oracle_row.get("elapsed_seconds", 0.0)),
            "oracle_row": oracle_row,
            "chosen_action": chosen,
            "llm_returned_raw": raw_text,
            "llm_returned_action": parsed,
            "llm_action_invalid": (not is_valid) or (parsed is None) or (parsed not in actions_idx),
            "fallback_action": fallback_action,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": round(float(cost), 6),
            "api_latency_ms": elapsed_ms,
            "drift_alert_count": 0,
            "replan_triggered": False,
            "trace_diagnostics": [],
            "trace_cost_summaries": [
                {
                    "provider": provider,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(float(cost), 6),
                }
            ],
        },
    )


def build_deepseek_client():
    """Convenience helper: build an OpenAI-compatible client for DeepSeek.

    Reads ``DEEPSEEK_API_KEY`` from environment. Raises RuntimeError if key
    is missing — caller catches and either skips, mocks, or aborts.
    """

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY env var is not set")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai SDK is required for ChatLLM-NoTools baseline") from e
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")
