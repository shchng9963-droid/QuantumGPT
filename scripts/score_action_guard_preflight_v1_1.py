#!/usr/bin/env python3
"""Score the development-only ActionGuard v1.1 real-LLM preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.reliability_bench.schema import episode_from_dict  # noqa: E402
from eval.reliability_bench.task_predicate_evaluator_v2 import (  # noqa: E402
    aggregate_selective_metrics,
    evaluate_task_decision,
)
from eval.reliability_bench.terminal_schema_v2 import parse_terminal_decision  # noqa: E402
from eval.reliability_bench.trace_evidence_evaluator_v2 import (  # noqa: E402
    evaluate_trace_evidence,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rate(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    episodes = {
        item.episode_id: item
        for item in (episode_from_dict(raw) for raw in _read_jsonl(run / "episodes.jsonl"))
    }
    traces = _read_jsonl(run / "traces.jsonl")
    rows = []
    score_errors = []
    for trace in traces:
        if not trace["trace_complete"] or not trace.get("terminal"):
            rows.append({"trace_id": trace["trace_id"], "arm": trace["arm"], "scoreable": False})
            continue
        try:
            decision = parse_terminal_decision(trace["terminal"])
            episode = episodes[trace["episode_id"]]
            task = evaluate_task_decision(episode, decision)
            evidence = evaluate_trace_evidence(
                episode=episode,
                decision=decision,
                trace_id=trace["trace_id"],
                accepted_tool_calls=trace["accepted_tool_calls"],
            )
        except Exception as exc:
            score_errors.append(
                {"trace_id": trace["trace_id"], "error_type": type(exc).__name__, "error": str(exc)}
            )
            rows.append({"trace_id": trace["trace_id"], "arm": trace["arm"], "scoreable": False})
            continue
        rows.append(
            {
                "trace_id": trace["trace_id"],
                "episode_id": trace["episode_id"],
                "task_type": trace["schedule"]["task_type"],
                "arm": trace["arm"],
                "scoreable": True,
                "task": asdict(task),
                "evidence": asdict(evidence),
            }
        )

    traces_by_arm = defaultdict(list)
    rows_by_arm = defaultdict(list)
    for trace in traces:
        traces_by_arm[trace["arm"]].append(trace)
    for row in rows:
        rows_by_arm[row["arm"]].append(row)
    group_report = {}
    for arm, selected in sorted(traces_by_arm.items()):
        scored = [item for item in rows_by_arm[arm] if item["scoreable"]]
        task_evals = [
            evaluate_task_decision(
                episodes[item["episode_id"]], parse_terminal_decision(next(t for t in selected if t["trace_id"] == item["trace_id"])["terminal"])
            )
            for item in scored
        ]
        costs = [
            float(item["evidence"]["actual_cost_units"])
            for item in scored
        ]
        aggregate = aggregate_selective_metrics(task_evals, costs=costs) if task_evals else {}
        group_report[arm] = {
            "runs": len(selected),
            "complete": sum(item["trace_complete"] for item in selected),
            "scoreable": len(scored),
            "answered": sum(item["task"]["answered"] for item in scored),
            "abstained": sum(not item["task"]["answered"] for item in scored),
            "task_predicate_correct": sum(item["task"]["terminal_correct"] for item in scored),
            "unsafe_or_unknown": sum(item["evidence"]["unsafe_or_unknown"] for item in scored),
            "definite_stale_dependence": sum(item["evidence"]["definite_stale_dependence"] for item in scored),
            "unsupported_decision": sum(item["evidence"]["unsupported_decision"] for item in scored),
            "provenance_incomplete": sum(item["evidence"]["provenance_incomplete"] for item in scored),
            "realized_revalidation_cost_units": sum(costs),
            "api_error_attempts": sum(len(item["llm"]["api_errors"]) for item in selected),
            "budget_exhausted_runs": sum("budget_exhausted" in str(item["unscorable_reason"]) for item in selected),
            "parse_failure_runs": sum(bool(item["llm"]["parser_errors"]) for item in selected),
            "guard_interventions": sum(
                event["outcome"] != "ALLOW" for item in selected for event in item["guard_events"]
            ),
            "guard_repair_attempts": sum(
                event["repair_allowed"] for item in selected for event in item["guard_events"]
            ),
            "guard_overhead_seconds": sum(
                float(event["check_latency_seconds"])
                for item in selected
                for event in item["guard_events"]
            ),
            "llm_turns": sum(item["budget_usage"]["turns"] for item in selected),
            "llm_tokens": sum(item["llm"]["usage"].get("total_tokens", 0) for item in selected),
            "estimated_cost_usd": round(sum(item["llm"]["estimated_cost_usd"] for item in selected), 10),
            "safety_guardrails": aggregate,
        }

    prompt_hashes = defaultdict(set)
    frozen_configs = set()
    returned_models = set()
    feedback_leaks = []
    for trace in traces:
        prompt_hashes[trace["episode_id"]].add(trace["public_prompt_sha256"])
        frozen_configs.add(json.dumps(trace["frozen_config"], sort_keys=True))
        returned_models.update(trace["llm"]["returned_model_ids"])
        episode = episodes[trace["episode_id"]]
        forbidden_values = {
            episode.episode_id,
            episode.circuit,
            *episode.task_constraints["candidate_backends"],
            *(item.evidence_id for item in episode.evidence),
        }
        for event in trace["guard_events"]:
            feedback = json.dumps(event.get("feedback"), ensure_ascii=False)
            if any(value and value in feedback for value in forbidden_values):
                feedback_leaks.append({"trace_id": trace["trace_id"], "check_index": event["check_index"]})

    initial_by_episode_arm = {
        (item["episode_id"], item["arm"]): item["initial_belief_ledger_B_t"] for item in traces
    }
    boundary_failures = []
    for episode_id in episodes:
        ledger = initial_by_episode_arm[(episode_id, "ledger_only_plus_action_guard")]
        full = initial_by_episode_arm[(episode_id, "full_plus_action_guard")]
        if any(item["belief_validity"] != "believed_valid" for item in ledger):
            boundary_failures.append(f"ledger_guard_not_last_known_valid:{episode_id}")
        if not any(item["belief_validity"] == "believed_invalid" for item in full):
            boundary_failures.append(f"full_guard_missing_dependency_update:{episode_id}")

    guard_exhausted = [
        item["trace_id"] for item in traces if item["unscorable_reason"] == "guard_repair_budget_exhausted"
    ]
    ledger_guard = group_report.get("ledger_only_plus_action_guard", {})
    all_complete = len(traces) == 36 and all(item["trace_complete"] for item in traces)
    all_scoreable = len(rows) == 36 and all(item["scoreable"] for item in rows)
    no_all_abstain = all(item["answered"] > 0 for item in group_report.values())
    no_systemic_failures = all(
        item["api_error_attempts"] <= 1
        and item["budget_exhausted_runs"] == 0
        and item["parse_failure_runs"] <= 1
        for item in group_report.values()
    )
    acceptance = {
        "run_count_exactly_36": len(traces) == 36,
        "trace_completion_36_of_36": all_complete,
        "scoreable_36_of_36": all_scoreable,
        "ledger_guard_not_structural_zero_allow": ledger_guard.get("complete", 0) > 0,
        "guard_repair_budget_exhaustion_absent": not guard_exhausted,
        "no_group_all_abstains": no_all_abstain,
        "no_systemic_api_budget_or_parse_failure": no_systemic_failures,
        "public_prompt_equal_within_episode": all(len(items) == 1 for items in prompt_hashes.values()),
        "runtime_config_identical_all_arms": len(frozen_configs) == 1,
        "guard_feedback_answer_leaks": len(feedback_leaks) == 0,
        "belief_component_boundary": not boundary_failures,
    }
    report = {
        "status": "development-only real-LLM preflight",
        "paper_performance_claims_allowed": False,
        "run_count": len(traces),
        "scoreable_count": sum(item["scoreable"] for item in rows),
        "returned_model_ids": sorted(returned_models),
        "group_report": group_report,
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
        "score_errors": score_errors,
        "guard_repair_exhausted_trace_ids": guard_exhausted,
        "feedback_leaks": feedback_leaks,
        "boundary_failures": boundary_failures,
        "trace_file_sha256": _sha256(run / "traces.jsonl"),
        "method_validation_v2_1_accessed": False,
    }
    (run / "scores.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )
    (run / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["acceptance_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
