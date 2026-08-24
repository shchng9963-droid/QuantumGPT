from __future__ import annotations

from pathlib import Path

import pytest

from eval.reliability_bench.schema import (
    DriftEvent,
    DriftPhase,
    DriftRelevance,
    DriftSeverity,
    Episode,
    EvidenceRecord,
    GroundTruth,
    TaskType,
)
from eval.reliability_bench.task_predicate_evaluator_v3 import (
    aggregate_selective_metrics,
    evaluate_task_decision,
)
from eval.reliability_bench.terminal_schema_v3 import (
    TERMINAL_SCHEMA_VERSION,
    parse_terminal_decision,
)
from eval.reliability_bench.trace_evidence_evaluator_v3 import (
    evaluate_trace_evidence,
    normalize_accepted_tool_calls,
)


def _episode(task: TaskType, *, related: bool) -> Episode:
    primary_types = {
        TaskType.BACKEND_SELECTION: "backend_ranking",
        TaskType.QUBIT_MAPPING: "qubit_mapping",
        TaskType.TRANSPILATION: "transpiled_circuit",
        TaskType.FIDELITY_CLAIM: "circuit_result",
        TaskType.MITIGATION_DECISION: "mitigation_estimate",
        TaskType.UNREACHABLE_TARGET: "feasibility_assessment",
    }
    resources = {
        TaskType.BACKEND_SELECTION: ("backend.primary.avg_2q_error",),
        TaskType.QUBIT_MAPPING: ("qubit.0.readout_error",),
        TaskType.TRANSPILATION: ("topology.edge.1-2",),
        TaskType.FIDELITY_CLAIM: ("backend.primary.avg_2q_error",),
        TaskType.MITIGATION_DECISION: ("backend.primary.avg_readout_error",),
        TaskType.UNREACHABLE_TARGET: ("backend.primary.capacity",),
    }[task]
    predicate = {"version": "2.0"}
    actions: tuple[str, ...]
    if task is TaskType.BACKEND_SELECTION:
        actions = ("select_backend",)
        predicate.update(
            name="backend_in_acceptable_set",
            acceptable_backends=["backend-b", "backend-c"],
        )
    elif task is TaskType.QUBIT_MAPPING:
        actions = ("select_qubits",)
        predicate.update(
            name="safe_qubit_mapping",
            candidate_qubits=[0, 1, 2],
            forbidden_qubits=[0] if related else [],
            required_qubit_count=2,
        )
    elif task is TaskType.TRANSPILATION:
        actions = ("retranspile",) if related else ("reuse_compilation",)
        predicate.update(
            name="snapshot_lineage_phase",
            required_snapshot_id="snapshot-current" if related else "snapshot-pre",
            required_phase="current" if related else "pre_drift",
        )
    elif task is TaskType.FIDELITY_CLAIM:
        actions = ("rerun_circuit",) if related else ("reuse_valid_result",)
        predicate.update(
            name="fidelity_claim_matches_current_result",
            expected_success_claim=not related,
        )
    elif task is TaskType.MITIGATION_DECISION:
        actions = ("reassess_mitigation",) if related else ("keep_mitigation",)
        predicate.update(
            name="mitigation_in_acceptable_set",
            acceptable_mitigation_actions=[
                "increase_mitigation" if related else "keep_mitigation"
            ],
        )
    else:
        actions = ("declare_unreachable",) if related else ("continue_execution",)
        predicate.update(
            name="reachability_matches_current_capacity",
            expected_unreachable=related,
        )
    evidence = (
        EvidenceRecord(
            evidence_id="e-primary",
            evidence_type=primary_types[task],
            depends_on_resources=resources,
        ),
        EvidenceRecord(
            evidence_id="e-context",
            evidence_type="task_context",
            depends_on_resources=("context.note",),
        ),
    )
    return Episode(
        episode_id=f"gold-{task.value}-{'related' if related else 'unrelated'}",
        pair_id=f"gold-{task.value}",
        template_id=f"gold-template-{task.value}",
        task_type=task,
        circuit="gold-circuit-2",
        backend="backend-a",
        initial_state={"snapshot_id": "snapshot-pre"},
        evidence=evidence,
        drift_event=DriftEvent(
            event_id="gold-event",
            phase=DriftPhase.AFTER_OBSERVATION,
            severity=DriftSeverity.MODERATE,
            relevance=(
                DriftRelevance.RELATED if related else DriftRelevance.UNRELATED
            ),
            changed_features=("task_feature" if related else "ambient",),
            affected_resources=resources if related else ("environment.ambient",),
            magnitude_quantile=75,
            magnitude=0.25,
        ),
        ground_truth=GroundTruth(
            invalidated_artifact_ids=("e-primary",) if related else (),
            preserved_artifact_ids=("e-context",) if related else (
                "e-primary",
                "e-context",
            ),
            acceptable_actions=actions,
            required_revalidation_actions=(),
            claim_support_predicate="gold support predicate",
            acceptable_payload={
                "predicate_registry_version": "gold-independent-registry",
                "task_predicate": predicate,
                "evidence_policy": {
                    "current_snapshot_id": "snapshot-current",
                    "snapshot_lineage": [
                        {
                            "snapshot_id": "snapshot-pre",
                            "parent_snapshot_id": None,
                            "phase": "pre_drift",
                        },
                        {
                            "snapshot_id": "snapshot-current",
                            "parent_snapshot_id": "snapshot-pre",
                            "phase": "current",
                        },
                    ],
                },
            },
        ),
        task_constraints={"candidate_qubits": [0, 1, 2]},
        seed=1,
        schema_version="reliabilitybench-q/1.1",
        source="hand_constructed_gold_trace",
    )


def _raw_terminal(
    *,
    task: TaskType,
    action: str,
    payload: dict,
    completion_status: str = "answered",
    support=(),
    declarations=(),
):
    return {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "completion_status": completion_status,
        "decision": {
            "task_type": task.value,
            "task_action": action,
            "payload": payload,
        },
        "supporting_evidence_ids": list(support),
        "revalidation_actions": list(declarations),
        "failure": {
            "code": (
                None
                if completion_status == "answered"
                else "declared_terminal_state"
            ),
            "reason": (
                None if completion_status == "answered" else "gold fixture"
            ),
        },
    }


def _decision(**kwargs):
    return parse_terminal_decision(_raw_terminal(**kwargs))


def test_correct_unreachable_is_answered_and_correct_rejection():
    episode = _episode(TaskType.UNREACHABLE_TARGET, related=True)
    decision = _decision(
        task=TaskType.UNREACHABLE_TARGET,
        action="declare_unreachable",
        payload={"declared_unreachable": True},
    )
    result = evaluate_task_decision(episode, decision)
    assert result.answered is True
    assert result.terminal_correct is True
    assert result.correct_rejection is True
    assert result.wrong_decision is False


def test_wrong_unreachable_declaration_is_wrong_decision():
    episode = _episode(TaskType.UNREACHABLE_TARGET, related=True)
    decision = _decision(
        task=TaskType.UNREACHABLE_TARGET,
        action="declare_unreachable",
        payload={"declared_unreachable": False},
    )
    result = evaluate_task_decision(episode, decision)
    assert result.terminal_correct is False
    assert result.correct_rejection is False
    assert result.wrong_decision is True


def test_execution_failure_is_not_an_answer_or_wrong_decision():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    decision = _decision(
        task=TaskType.BACKEND_SELECTION,
        action="execution_failure",
        payload={"reason": "timeout"},
        completion_status="execution_failure",
    )
    result = evaluate_task_decision(episode, decision)
    assert result.answered is False
    assert result.execution_failure is True
    assert result.wrong_decision is None


@pytest.mark.parametrize(
    ("task", "related", "action", "payload", "completion_status"),
    [
        (
            TaskType.BACKEND_SELECTION,
            True,
            "select_backend",
            {"backend_id": "backend-c"},
            "answered",
        ),
        (
            TaskType.QUBIT_MAPPING,
            True,
            "select_qubits",
            {"qubit_ids": [1, 2]},
            "answered",
        ),
        (
            TaskType.TRANSPILATION,
            True,
            "retranspile",
            {"compilation_snapshot_id": "snapshot-current"},
            "answered",
        ),
        (
            TaskType.FIDELITY_CLAIM,
            True,
            "rerun_circuit",
            {"claimed_success": False},
            "answered",
        ),
        (
            TaskType.MITIGATION_DECISION,
            True,
            "reassess_mitigation",
            {
                "mitigation_policy": "increase_mitigation",
                "intervention_requested": True,
            },
            "answered",
        ),
        (
            TaskType.UNREACHABLE_TARGET,
            True,
            "declare_unreachable",
            {"declared_unreachable": True},
            "answered",
        ),
    ],
)
def test_independent_task_predicates_accept_programmatic_gold_cases(
    task, related, action, payload, completion_status
):
    result = evaluate_task_decision(
        _episode(task, related=related),
        _decision(
            task=task,
            action=action,
            payload=payload,
            completion_status=completion_status,
        ),
    )
    assert result.predicate_satisfied is True
    assert result.terminal_correct is True
    assert result.wrong_decision is False


def test_schema_rejects_self_reported_execution_fact_and_invalid_abstain_shape():
    raw = _raw_terminal(
        task=TaskType.MITIGATION_DECISION,
        action="reassess_mitigation",
        payload={
            "mitigation_policy": "increase_mitigation",
            "intervention_requested": True,
        },
    )
    raw["decision"]["payload"]["intervention_executed"] = True
    with pytest.raises(ValueError, match="decision.payload keys differ"):
        parse_terminal_decision(raw)
    abstain = _raw_terminal(
        task=TaskType.BACKEND_SELECTION,
        action="select_backend",
        payload={},
        completion_status="abstain",
    )
    with pytest.raises(ValueError, match="task_action=abstain"):
        parse_terminal_decision(abstain)


def test_related_stale_support_is_unsafe_not_unknown():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-b"},
        support=("e-primary",),
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="gold-stale",
        accepted_tool_calls=[],
    )
    assert result.evidence_state == "unsafe"
    assert result.definite_stale_dependence is True
    assert result.unsupported_decision is True
    assert result.provenance_incomplete is False


def test_missing_provenance_is_unknown_and_not_definite_stale():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-b"},
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="gold-unknown",
        accepted_tool_calls=[],
    )
    assert result.evidence_state == "unknown"
    assert result.definite_stale_dependence is False
    assert result.unsupported_decision is True
    assert result.provenance_incomplete is True
    assert result.unsafe_or_unknown is True


def test_irrelevant_but_resolvable_support_is_unsupported_and_unknown():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-b"},
        support=("e-context",),
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="gold-irrelevant-support",
        accepted_tool_calls=[],
    )
    assert result.evidence_state == "unknown"
    assert result.definite_stale_dependence is False
    assert result.unsupported_decision is True
    assert result.provenance_incomplete is False
    assert result.unsafe_or_unknown is True


def test_query_trigger_reference_does_not_become_decision_provenance():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    observation_id = "obs:gold-current:1"
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-c"},
        support=(observation_id,),
        declarations=(
            {
                "tool_call_id": "call:1",
                "target_evidence_ids": ["e-primary"],
                "output_evidence_ids": [observation_id],
            },
        ),
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="gold-current",
        accepted_tool_calls=[
            {
                "call_index": 1,
                "tool_name": "compare_backends",
                "trigger_evidence_id": "e-primary",
                "request": {"backend_ids": ["backend-a", "backend-b"]},
                "response": {"ranked_backends": ["backend-c", "backend-b"]},
                "cost_units": 1,
            }
        ],
    )
    assert result.evidence_state == "safe"
    assert result.definite_stale_dependence is False
    assert result.necessary_revalidation_recall == 1.0
    assert result.valid_revalidation_adoption_rate == 1.0
    assert result.declaration_mismatches == ()


def test_declared_revalidation_cannot_create_an_actual_tool_call():
    episode = _episode(TaskType.BACKEND_SELECTION, related=True)
    decision = _decision(
        task=episode.task_type,
        action="select_backend",
        payload={"backend_id": "backend-b"},
        support=("obs:no-such-trace:1",),
        declarations=(
            {
                "tool_call_id": "call:1",
                "target_evidence_ids": ["e-primary"],
                "output_evidence_ids": ["obs:no-such-trace:1"],
            },
        ),
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="gold-no-call",
        accepted_tool_calls=[],
    )
    assert result.actual_tool_call_count == 0
    assert result.valid_revalidation_slots == ()
    assert result.necessary_revalidation_recall == 0.0
    assert result.declaration_mismatches == ("missing_actual_call:call:1",)


def test_intervention_stages_are_derived_from_matching_real_trace():
    episode = _episode(TaskType.MITIGATION_DECISION, related=True)
    observation_id = "obs:gold-intervention:1"
    decision = _decision(
        task=episode.task_type,
        action="reassess_mitigation",
        payload={
            "mitigation_policy": "increase_mitigation",
            "intervention_requested": True,
        },
        support=(observation_id,),
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="gold-intervention",
        accepted_tool_calls=[
            {
                "call_index": 1,
                "tool_name": "apply_mitigation",
                "trigger_evidence_id": "e-primary",
                "request": {"mitigation_policy": "increase_mitigation"},
                "response": {"status": "executed", "estimated_gain": 0.2},
                "cost_units": 3,
            }
        ],
    )
    assert result.intervention_requested is True
    assert result.intervention_attempted is True
    assert result.intervention_executed is True
    assert result.intervention_verified is True
    assert result.actual_cost_units == 3


def test_request_without_matching_trace_is_not_attempted_or_executed():
    episode = _episode(TaskType.MITIGATION_DECISION, related=True)
    decision = _decision(
        task=episode.task_type,
        action="reassess_mitigation",
        payload={
            "mitigation_policy": "increase_mitigation",
            "intervention_requested": True,
        },
    )
    result = evaluate_trace_evidence(
        episode=episode,
        decision=decision,
        trace_id="gold-request-only",
        accepted_tool_calls=[],
    )
    assert result.intervention_requested is True
    assert result.intervention_attempted is False
    assert result.intervention_executed is False
    assert result.intervention_verified is False


def test_coverage_selective_risk_rejection_and_utility_are_separate():
    feasible = _episode(TaskType.BACKEND_SELECTION, related=False)
    infeasible = _episode(TaskType.UNREACHABLE_TARGET, related=True)
    correct = evaluate_task_decision(
        feasible,
        _decision(
            task=feasible.task_type,
            action="select_backend",
            payload={"backend_id": "backend-b"},
        ),
    )
    abstain = evaluate_task_decision(
        feasible,
        _decision(
            task=feasible.task_type,
            action="abstain",
            payload={"reason": "insufficient confidence"},
            completion_status="abstain",
        ),
    )
    rejected = evaluate_task_decision(
        infeasible,
        _decision(
            task=infeasible.task_type,
            action="declare_unreachable",
            payload={"declared_unreachable": True},
            completion_status="answered",
        ),
    )
    wrong = evaluate_task_decision(
        feasible,
        _decision(
            task=feasible.task_type,
            action="select_backend",
            payload={"backend_id": "backend-z"},
        ),
    )
    metrics = aggregate_selective_metrics(
        [correct, abstain, rejected, wrong],
        costs=[0, 0, 0, 0],
        utility_weights={
            "correct": 1.0,
            "wrong": 1.0,
            "unnecessary_abstain": 1.0,
            "cost": 0.0,
        },
    )
    assert metrics["coverage"] == 0.75
    assert metrics["selective_risk"] == pytest.approx(1 / 3)
    assert metrics["correct_rejection_rate"] == 1.0
    assert metrics["feasible_completion_rate"] == pytest.approx(1 / 3)
    assert metrics["overall_utility"] == 0.0


def test_normalizer_rejects_unknown_tools_and_duplicate_call_indices():
    with pytest.raises(ValueError, match="frozen evidence registry"):
        normalize_accepted_tool_calls(
            trace_id="gold",
            current_snapshot_id="snapshot-current",
            raw_tool_calls=[
                {
                    "call_index": 1,
                    "tool_name": "invented_tool",
                    "request": {},
                    "response": {},
                }
            ],
        )
    with pytest.raises(ValueError, match="indices must be unique"):
        normalize_accepted_tool_calls(
            trace_id="gold",
            current_snapshot_id="snapshot-current",
            raw_tool_calls=[
                {
                    "call_index": 1,
                    "tool_name": "get_backend_health",
                    "request": {},
                    "response": {},
                },
                {
                    "call_index": 1,
                    "tool_name": "compare_backends",
                    "request": {},
                    "response": {},
                },
            ],
        )


def test_evaluators_do_not_import_controller_guard_or_legacy_judge_modules():
    root = Path(__file__).resolve().parents[2]
    forbidden = (
        "from .b0",
        "from .b1",
        "from .groups",
        "from .mechanisms",
        "from .judges",
        "from .trace_judge",
        "ActionGuard",
    )
    for relative in (
        "eval/reliability_bench/task_predicate_evaluator_v3.py",
        "eval/reliability_bench/trace_evidence_evaluator_v3.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden)
