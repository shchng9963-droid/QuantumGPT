#!/usr/bin/env python3
"""Run the independent declarative development Gold audit for Measurement v3.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.reliability_bench.gold_runtime_terminal_spec_v3_1 import (  # noqa: E402
    GOLD_SPEC_VERSION,
    UTILITY_WEIGHTS,
    build_development_delta_cases,
)
from eval.reliability_bench.public_runtime_v3 import (  # noqa: E402
    bind_authoritative_tool_observation,
)
from eval.reliability_bench.public_runtime_v3_1 import (  # noqa: E402
    parse_agent_terminal_runtime_output,
)
from eval.reliability_bench.runtime_terminal_v1 import RuntimeExecutionFailure  # noqa: E402
from eval.reliability_bench.schema import episode_from_dict  # noqa: E402
from eval.reliability_bench.task_predicate_evaluator_v3_1 import (  # noqa: E402
    aggregate_selective_metrics,
    evaluate_terminal_outcome,
)
from eval.reliability_bench.terminal_outcome_v3_1 import (  # noqa: E402
    parse_terminal_outcome,
)
from eval.reliability_bench.trace_evidence_evaluator_v3_1 import (  # noqa: E402
    evaluate_trace_outcome,
)


FIELDS = (
    "task_terminal_correct",
    "predicate_satisfied",
    "wrong_decision",
    "answered",
    "execution_failure",
    "feasible",
    "feasible_completion",
    "correct_rejection",
    "unnecessary_abstain",
    "terminal_source",
    "runtime_generated",
    "failure_reason",
    "decision_is_null",
    "decision_support_applicable",
    "evidence_state",
    "definite_stale_dependence",
    "unsupported_decision",
    "provenance_incomplete",
    "unsafe_or_unknown",
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
    "runtime_failure_reason",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def predict_case(case: dict[str, Any]):
    episode = episode_from_dict(case["episode"])
    bound_calls = [
        bind_authoritative_tool_observation(
            trace_id=case["case_id"],
            call_index=raw["call_index"],
            accepted_call=raw,
        )
        for raw in case["accepted_tool_calls"]
    ]
    if case["terminal"].get("runtime_generated") is True:
        outcome = parse_terminal_outcome(case["terminal"])
    else:
        parsed = parse_agent_terminal_runtime_output(
            json.dumps(case["terminal"]),
            known_evidence_ids=[item["evidence_id"] for item in case["episode"]["evidence"]],
            accepted_tool_calls=bound_calls,
        )
        if not parsed.accepted or parsed.decision is None:
            raise RuntimeError(f"Gold Agent terminal rejected: {case['case_id']}")
        outcome = parsed.decision
    task = evaluate_terminal_outcome(episode, outcome)
    trace = evaluate_trace_outcome(
        episode=episode,
        outcome=outcome,
        trace_id=case["case_id"],
        accepted_tool_calls=case["accepted_tool_calls"],
    )
    task_raw = asdict(task)
    trace_raw = asdict(trace)
    runtime = isinstance(outcome, RuntimeExecutionFailure)
    prediction = {
        "task_terminal_correct": task.terminal_correct,
        "predicate_satisfied": task.predicate_satisfied,
        "wrong_decision": task.wrong_decision,
        "answered": task.answered,
        "execution_failure": task.execution_failure,
        "feasible": task.feasible,
        "feasible_completion": task.feasible_completion,
        "correct_rejection": task.correct_rejection,
        "unnecessary_abstain": task.unnecessary_abstain,
        "terminal_source": "runtime" if runtime else "agent",
        "runtime_generated": runtime,
        "failure_reason": outcome.failure_reason if runtime else None,
        "decision_is_null": outcome.decision is None if runtime else False,
    }
    prediction.update({field: trace_raw[field] for field in FIELDS if field in trace_raw})
    return prediction, task, trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = build_development_delta_cases()
    gold_payload = {
        "gold_spec_version": GOLD_SPEC_VERSION,
        "cases": cases,
        "field_count_per_case": len(FIELDS),
    }
    gold_bytes = canonical_bytes(gold_payload)
    (out / "gold_cases.json").write_bytes(gold_bytes)
    gold_hash = sha256_bytes(gold_bytes)
    predictions = {}
    mismatches = []
    task_evaluations = []
    costs = []
    for case in cases:
        prediction, task, trace = predict_case(case)
        predictions[case["case_id"]] = prediction
        task_evaluations.append(task)
        costs.append(float(trace.actual_cost_units))
        for field in FIELDS:
            expected = case["expected"][field]
            actual = prediction[field]
            if actual != expected:
                mismatches.append(
                    {
                        "case_id": case["case_id"],
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    aggregate = aggregate_selective_metrics(
        task_evaluations, costs=costs, utility_weights=UTILITY_WEIGHTS
    )
    prediction_payload = {
        "gold_sha256_before_prediction": gold_hash,
        "predictions": predictions,
        "aggregate_metrics": aggregate,
    }
    prediction_bytes = canonical_bytes(prediction_payload)
    (out / "predictions.json").write_bytes(prediction_bytes)
    report = {
        "status": "passed" if not mismatches else "failed",
        "development_only": True,
        "gold_sha256_before_prediction": gold_hash,
        "prediction_sha256": sha256_bytes(prediction_bytes),
        "case_count": len(cases),
        "field_count_per_case": len(FIELDS),
        "comparison_count": len(cases) * len(FIELDS),
        "exact_agreement": len(cases) * len(FIELDS) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "aggregate_metrics": aggregate,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
