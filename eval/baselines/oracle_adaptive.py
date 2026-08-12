"""Oracle-Adaptive baseline (v2.5 Sprint A, A2).

Picks the deterministic action with the maximum post-drift fidelity from
the oracle row's action list. By construction it matches the oracle upper
bound — its purpose is to upper-bound any single-action selector and let
us report (Agent − Oracle-Adaptive) as decision-making slack.

  best_post_fidelity == oracle_best
  oracle_gap == 0
  total_cost_usd == 0
  provider == "none"
  model == "oracle-adaptive"
"""

from __future__ import annotations

from typing import Any

from eval.public_mqtbench_agent_eval import PublicAgentRunResult


def run_oracle_adaptive_task(oracle_row: dict[str, Any]) -> PublicAgentRunResult:
    """Pick the action with maximum fidelity in the oracle row, no LLM.

    The oracle_row must come from eval.public_mqtbench_oracle's JSONL output
    (or the test fixture mirror), and contain a non-empty ``actions`` list.
    """

    actions = oracle_row.get("actions", []) or []
    if not actions:
        raise ValueError(
            "Oracle-Adaptive requires a non-empty oracle_row['actions'] list; "
            f"got {actions!r}"
        )

    chosen = max(actions, key=lambda a: float(a.get("fidelity", 0.0)))
    fid = float(chosen["fidelity"])
    chosen_name = str(chosen.get("action", "unknown"))

    seed = int(oracle_row.get("seed", 0))
    target = float(oracle_row.get("target_fidelity", 0.85))

    return PublicAgentRunResult(
        system="Oracle-Adaptive",
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
        oracle_best=float(oracle_row.get("best_post_drift_score", fid)),
        oracle_feasible=bool(oracle_row.get("oracle_feasible", fid >= target)),
        oracle_best_action=str(oracle_row.get("best_action", chosen_name)),
        pre_fidelities=[float(oracle_row.get("pre_fidelity", 0.0))],
        post_fidelities=[fid],
        provider="none",
        model="oracle-adaptive",
        tool_calls=[chosen_name],
        trace_files=[],
        metadata={
            "execution_mode": "simulator",
            "use_mock": False,
            "seed": seed,
            "max_turns": 0,
            "max_tool_calls": 1,
            "pre_fidelity_count": 1,
            "post_fidelity_count": 1,
            "pre_final_answer_chars": 0,
            "post_final_answer_chars": 0,
            "elapsed_seconds": float(oracle_row.get("elapsed_seconds", 0.0)),
            "oracle_row": oracle_row,
            "chosen_action": chosen_name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "api_latency_ms": 0.0,
            "drift_alert_count": 0,
            "replan_triggered": False,
            "trace_diagnostics": [],
            "trace_cost_summaries": [],
        },
    )
