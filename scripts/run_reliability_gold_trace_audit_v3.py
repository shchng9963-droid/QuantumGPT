"""Run executable evaluators against independently declared Gold traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from eval.reliability_bench.gold_trace_spec_v3 import (
    GOLD_SPEC_VERSION,
    build_development_cases,
)
from eval.reliability_bench.schema import episode_from_dict
from eval.reliability_bench.task_predicate_evaluator_v3 import evaluate_task_decision
from eval.reliability_bench.terminal_schema_v3 import parse_terminal_decision
from eval.reliability_bench.trace_evidence_evaluator_v3 import evaluate_trace_evidence


CORE_FIELDS = (
    "evidence_state",
    "definite_stale_dependence",
    "unsupported_decision",
    "provenance_incomplete",
    "necessary_revalidation_slots",
    "valid_revalidation_slots",
    "necessary_revalidation_recall",
    "snapshot_lineage_valid",
    "evidence_closure_complete",
    "intervention_requested",
    "intervention_attempted",
    "intervention_executed",
    "intervention_verified",
    "actual_tool_call_count",
    "failed_tool_call_count",
    "repeated_tool_call_count",
    "irrelevant_tool_call_count",
    "actual_cost_units",
    "actual_latency_seconds",
)


def canonical(value):
    if isinstance(value, tuple):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    return value


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run() -> dict:
    cases = build_development_cases()
    rows = []
    failures = []
    for case in cases:
        episode = episode_from_dict(case["episode"])
        decision = parse_terminal_decision(case["terminal"])
        task = evaluate_task_decision(episode, decision)
        trace = evaluate_trace_evidence(
            episode=episode,
            decision=decision,
            trace_id=case["case_id"],
            accepted_tool_calls=case["accepted_tool_calls"],
        )
        actual = {"task_terminal_correct": task.terminal_correct}
        trace_values = asdict(trace)
        actual.update({field: canonical(trace_values[field]) for field in CORE_FIELDS})
        expected = canonical(case["expected"])
        mismatches = {
            field: {"expected": expected[field], "actual": actual[field]}
            for field in expected
            if actual[field] != expected[field]
        }
        row = {
            "case_id": case["case_id"],
            "tags": case["tags"],
            "passed": not mismatches,
            "mismatches": mismatches,
        }
        rows.append(row)
        if mismatches:
            failures.append(row)
    tag_counts = {}
    for case in cases:
        for tag in case["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {
        "audit_version": "reliabilitybench-q/development-gold-audit-3.0",
        "gold_spec_version": GOLD_SPEC_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_case_count": len(cases),
        "gold_payload_sha256": sha256_json(cases),
        "passed_count": len(cases) - len(failures),
        "failed_count": len(failures),
        "all_unique_truth_gates_passed": not failures,
        "coverage_tag_counts": dict(sorted(tag_counts.items())),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "gold_case_count", "gold_payload_sha256", "passed_count", "failed_count",
        "all_unique_truth_gates_passed"
    )}, ensure_ascii=False, indent=2))
    return 0 if result["all_unique_truth_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
