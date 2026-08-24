from __future__ import annotations

import copy
import json
from dataclasses import replace

from eval.reliability_bench.measurement_spec_v3 import (
    PRIMARY_UTILITY_WEIGHTS,
    TASK_REQUIRED_SLOTS,
)
from eval.reliability_bench.public_runtime_v3 import (
    COMPONENT_INFORMATION_ACCESS,
    GENERIC_FORMAT_RETRY,
    PUBLIC_SYSTEM_PROMPT,
    StudyArm,
    bind_authoritative_tool_observation,
    generate_fairness_audit,
    parse_terminal_runtime_output,
    validate_fairness_contract,
)
from eval.reliability_bench.schema import GroundTruth, TaskType
from eval.reliability_bench.task_predicate_evaluator_v3 import (
    aggregate_selective_metrics,
    evaluate_task_decision,
)
from eval.reliability_bench.trace_evidence_evaluator_v3 import evaluate_trace_evidence
from tests.unit.test_reliability_bench_measurement_v3 import (
    _decision,
    _episode,
    _raw_terminal,
)


def test_six_arm_public_runtime_is_treatment_neutral_and_guard_is_external():
    validate_fairness_contract()
    audit = generate_fairness_audit()
    assert len(StudyArm) == 6
    assert len(COMPONENT_INFORMATION_ACCESS) == 6
    assert audit["all_public_prompts_identical"] is True
    assert len(set(audit["public_prompt_hash_by_arm"].values())) == 1
    assert audit["action_guard_core_implemented"] is True
    assert audit["action_guard_implementation_location"].startswith("external")
    assert audit["semantic_repair_capability"] is False
    assert "invalidated_evidence_ids" not in PUBLIC_SYSTEM_PROMPT
    assert "recommended_action" not in PUBLIC_SYSTEM_PROMPT


def test_runtime_binds_real_call_without_rewriting_request_or_response():
    raw = {
        "tool_name": "compare_backends",
        "request": {"backend_ids": ["a", "b"]},
        "response": {"ranked_backends": ["b", "a"]},
    }
    bound = bind_authoritative_tool_observation(
        trace_id="runtime-dev", call_index=2, accepted_call=raw
    )
    assert bound["tool_call_id"] == "call:2"
    assert bound["output_evidence_id"] == "obs:runtime-dev:2"
    assert bound["request"] == raw["request"]
    assert bound["response"] == raw["response"]
    assert "tool_call_id" not in raw


def test_runtime_format_retry_is_generic_and_identifier_failures_are_not_repaired():
    malformed = parse_terminal_runtime_output(
        "not-json", known_evidence_ids=["e-primary"], accepted_tool_calls=[]
    )
    assert malformed.accepted is False
    assert malformed.retry_message == GENERIC_FORMAT_RETRY
    assert "e-primary" not in malformed.retry_message

    raw = _raw_terminal(
        task=TaskType.BACKEND_SELECTION,
        action="select_backend",
        payload={"backend_id": "deliberately-wrong-but-syntactically-valid"},
        support=("fabricated-evidence",),
    )
    parsed = parse_terminal_runtime_output(
        json.dumps(raw), known_evidence_ids=["e-primary"], accepted_tool_calls=[]
    )
    assert parsed.accepted is False
    assert parsed.decision is not None
    assert parsed.decision.payload["backend_id"] == "deliberately-wrong-but-syntactically-valid"
    assert parsed.decision.supporting_evidence_ids == ("fabricated-evidence",)
    assert parsed.retry_message is None
    assert parsed.identifier_failures == (
        "unknown_supporting_evidence_id:fabricated-evidence",
    )


def test_failed_or_timed_out_current_call_is_unknown_not_definite_stale():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    observation = "obs:timeout-dev:1"
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-c"},
        support=(observation,),
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="timeout-dev",
        accepted_tool_calls=[
            {
                "call_index": 1,
                "tool_name": "compare_backends",
                "request": {"backend_ids": ["a", "b"]},
                "response": {"status": "timeout"},
                "cost_units": 1,
                "latency_seconds": 2.5,
            }
        ],
    )
    assert result.evidence_state == "unknown"
    assert result.definite_stale_dependence is False
    assert result.unsupported_decision is True
    assert result.failed_tool_call_count == 1
    assert result.actual_latency_seconds == 2.5


def test_snapshot_lineage_counterfactual_invalidates_current_observation_support():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    payload = copy.deepcopy(episode.ground_truth.acceptable_payload)
    payload["evidence_policy"]["snapshot_lineage"][1]["parent_snapshot_id"] = "wrong-parent"
    broken = replace(
        episode,
        ground_truth=replace(episode.ground_truth, acceptable_payload=payload),
    )
    observation = "obs:lineage-dev:1"
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-c"},
        support=(observation,),
    )
    result = evaluate_trace_evidence(
        episode=broken,
        decision=decision,
        trace_id="lineage-dev",
        accepted_tool_calls=[
            {
                "call_index": 1,
                "tool_name": "compare_backends",
                "request": {},
                "response": {"ranked_backends": ["backend-c"]},
                "observed_snapshot_id": "snapshot-current",
                "cost_units": 1,
            }
        ],
    )
    assert result.snapshot_lineage_valid is False
    assert result.evidence_state == "unknown"
    assert result.unsupported_decision is True
    assert result.definite_stale_dependence is False


def test_missing_parent_breaks_evidence_closure_without_becoming_stale():
    episode = _episode(TaskType.FIDELITY_CLAIM, related=True)
    observation = "obs:closure-dev:1"
    decision = _decision(
        task=episode.task_type,
        action="rerun_circuit",
        payload={"claimed_success": False},
        support=(observation,),
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="closure-dev",
        accepted_tool_calls=[
            {
                "call_index": 1,
                "tool_name": "run_circuit",
                "request": {},
                "response": {"success": True},
                "parent_evidence_ids": ["missing-parent"],
                "cost_units": 5,
            }
        ],
    )
    assert result.evidence_closure_complete is False
    assert result.provenance_incomplete is True
    assert result.evidence_state == "unknown"
    assert result.definite_stale_dependence is False
    assert set(result.provenance_closure_ids) == {observation, "missing-parent"}


def test_querying_old_evidence_does_not_mean_terminal_depends_on_it():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    observation = "obs:query-only-dev:1"
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-c"},
        support=(observation,),
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="query-only-dev",
        accepted_tool_calls=[
            {
                "call_index": 1,
                "tool_name": "compare_backends",
                "trigger_evidence_id": "e-primary",
                "request": {},
                "response": {"ranked_backends": ["backend-c"]},
                "cost_units": 1,
            }
        ],
    )
    assert result.evidence_state == "safe"
    assert result.definite_stale_dependence is False


def test_cost_latency_repeat_and_irrelevant_boundaries_are_trace_derived():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-c"},
        support=("obs:cost-dev:1",),
    )
    calls = [
        {
            "call_index": 1,
            "tool_name": "compare_backends",
            "request": {"x": 1},
            "response": {"ranked_backends": ["backend-c"]},
            "cost_units": 1,
            "latency_seconds": 0.25,
        },
        {
            "call_index": 2,
            "tool_name": "compare_backends",
            "request": {"x": 1},
            "response": {"ranked_backends": ["backend-c"]},
            "cost_units": 1,
            "latency_seconds": 0.5,
        },
        {
            "call_index": 3,
            "tool_name": "refresh_task_context",
            "request": {},
            "response": {"context": "unchanged"},
            "cost_units": 1,
            "latency_seconds": 0.75,
        },
    ]
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="cost-dev",
        accepted_tool_calls=calls,
    )
    assert result.actual_tool_call_count == 3
    assert result.actual_cost_units == 3
    assert result.actual_latency_seconds == 1.5
    assert result.repeated_tool_call_count == 1
    assert result.irrelevant_tool_call_count == 1


def test_primary_utility_weights_are_frozen_and_defaulted():
    episode = _episode(TaskType.BACKEND_SELECTION, related=False)
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-b"},
        support=("e-primary",),
    )
    evaluation = evaluate_task_decision(episode, decision)
    result = aggregate_selective_metrics([evaluation], costs=[1.0])
    assert dict(PRIMARY_UTILITY_WEIGHTS) == {
        "correct": 1.0,
        "wrong": 1.0,
        "unnecessary_abstain": 0.5,
        "cost": 0.01,
    }
    assert result["coverage"] == 1.0
    assert result["overall_utility"] == 0.99
    assert set(TASK_REQUIRED_SLOTS) == {task.value for task in TaskType}
