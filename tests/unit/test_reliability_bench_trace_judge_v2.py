from __future__ import annotations

import json
from dataclasses import replace

from eval.reliability_bench.generator_v2 import DEVELOPMENT_CONFIG, generate_method_validation_candidates
from eval.reliability_bench.schema import (
    SCHEMA_VERSION,
    DriftRelevance,
    Episode,
    GroundTruth,
    TaskType,
)
from eval.reliability_bench.trace_judge_v2 import judge_trace_v2
from scripts.score_reliability_bench_judge_v2_development import (
    _assert_development_input,
)


def _episode(task: TaskType, relevance: DriftRelevance) -> Episode:
    return next(
        item
        for item in generate_method_validation_candidates(DEVELOPMENT_CONFIG)
        if item.task_type is task and item.drift_event.relevance is relevance
    )


def _trace(episode: Episode, decision: dict, calls: list[dict] | None = None) -> dict:
    public = {
        "task": {
            "type": episode.task_type.value,
            "circuit": episode.circuit,
            "backend": episode.backend,
            "constraints": episode.task_constraints,
        },
        "initial_device_state": episode.initial_state,
        "device_change": episode.drift_event.neutral_payload(),
        "evidence_refs": [
            {"evidence_id": item.evidence_id, "evidence_type": item.evidence_type}
            for item in episode.evidence
        ],
    }
    return {
        "trace_id": "trace-test",
        "public_prompt": json.dumps({"input": public}),
        "tool_calls": calls or [],
        "model_final_decision": decision,
        "trace_complete": True,
        "unscorable": False,
        "controller": {"group": "must-not-be-read", "oracle_private": "must-not-be-read"},
        "program_judge": {"correct": False},
    }


def _decision(**updates):
    base = {
        "action": "",
        "used_artifact_ids": [],
        "revalidation_actions": ["self_report_must_not_count"],
        "selected_backend": None,
        "selected_qubits": [],
        "compilation_snapshot_id": None,
        "claimed_success": None,
        "supporting_artifact_ids": [],
        "mitigation_action": None,
        "declared_unreachable": None,
        "metadata": {},
    }
    base.update(updates)
    return base


def _call(index: int, name: str, evidence_id: str, response: dict) -> dict:
    return {
        "call_index": index,
        "tool_name": name,
        "trigger_evidence_id": evidence_id,
        "request": {"run_id": "run-test"},
        "response": response,
    }


def test_backend_ranking_tool_is_primary_and_controller_identity_is_ignored():
    episode = _episode(TaskType.BACKEND_SELECTION, DriftRelevance.RELATED)
    primary = episode.evidence[0].evidence_id
    trace = _trace(
        episode,
        _decision(action="select_backend", selected_backend="FakeSherbrooke", used_artifact_ids=[primary]),
        [_call(1, "compare_backends", primary, {"ranked_backends": ["FakeSherbrooke", "FakeBrisbane"]})],
    )
    first = judge_trace_v2(trace, episode)
    trace["controller"] = {"group": "different", "ground_truth": "poison"}
    trace["program_judge"] = {"correct": True}
    second = judge_trace_v2(trace, episode)
    assert first.correct is True
    assert second.to_dict() == first.to_dict()
    assert first.provenance["controller_identity_read"] is False


def test_self_reported_revalidation_never_replaces_actual_tool_call():
    episode = _episode(TaskType.BACKEND_SELECTION, DriftRelevance.RELATED)
    primary = episode.evidence[0].evidence_id
    result = judge_trace_v2(
        _trace(
            episode,
            _decision(
                action="select_backend",
                selected_backend="FakeSherbrooke",
                used_artifact_ids=[primary],
                revalidation_actions=["compare_backends"],
            ),
        ),
        episode,
    )
    assert result.correct is False
    assert "missing_required_revalidation" in result.error_codes
    assert result.provenance["self_reported_revalidation_used_as_execution_evidence"] is False


def test_mapping_accepts_every_safe_set_but_enforces_circuit_width():
    episode = _episode(TaskType.QUBIT_MAPPING, DriftRelevance.RELATED)
    primary = episode.evidence[0].evidence_id
    width = episode.task_constraints["required_qubit_count"]
    forbidden = episode.ground_truth.acceptable_payload["task_predicate"]["forbidden_qubits"][0]
    available = episode.task_constraints["candidate_qubits"]
    safe = [q for q in available if q != forbidden][:width]
    call = _call(1, "get_qubit_properties", primary, {"available_qubits": available, "degraded_qubits": [forbidden]})
    good = judge_trace_v2(_trace(episode, _decision(action="select_qubits", selected_qubits=list(reversed(safe))), [call]), episode)
    short = judge_trace_v2(_trace(episode, _decision(action="select_qubits", selected_qubits=safe[:-1]), [call]), episode)
    unsafe = judge_trace_v2(_trace(episode, _decision(action="select_qubits", selected_qubits=[forbidden, *safe[:-1]]), [call]), episode)
    assert good.correct is True
    assert short.correct is False and "unsafe_qubit_mapping" in short.error_codes
    assert unsafe.correct is False and "unsafe_qubit_mapping" in unsafe.error_codes


def test_legacy_mapping_allocation_may_be_a_safe_superset_without_weakening_v11():
    current = _episode(TaskType.QUBIT_MAPPING, DriftRelevance.UNRELATED)
    payload = dict(current.ground_truth.acceptable_payload)
    payload.pop("task_predicate", None)
    legacy = replace(
        current,
        schema_version=SCHEMA_VERSION,
        ground_truth=GroundTruth(
            invalidated_artifact_ids=current.ground_truth.invalidated_artifact_ids,
            preserved_artifact_ids=current.ground_truth.preserved_artifact_ids,
            acceptable_actions=current.ground_truth.acceptable_actions,
            required_revalidation_actions=current.ground_truth.required_revalidation_actions,
            claim_support_predicate=current.ground_truth.claim_support_predicate,
            acceptable_payload=payload,
        ),
    )
    width = current.task_constraints["required_qubit_count"]
    safe_superset = current.task_constraints["candidate_qubits"][: width + 1]
    legacy_result = judge_trace_v2(
        _trace(legacy, _decision(action="select_qubits", selected_qubits=safe_superset)),
        legacy,
    )
    v11_result = judge_trace_v2(
        _trace(current, _decision(action="select_qubits", selected_qubits=safe_superset)),
        current,
    )
    assert legacy_result.correct is True
    assert v11_result.correct is False


def test_initial_snapshot_id_and_pre_drift_alias_are_same_legal_source():
    episode = _episode(TaskType.TRANSPILATION, DriftRelevance.UNRELATED)
    primary = episode.evidence[0].evidence_id
    by_id = judge_trace_v2(
        _trace(episode, _decision(action="reuse_compilation", compilation_snapshot_id=episode.initial_state["snapshot_id"], used_artifact_ids=[primary])),
        episode,
    )
    by_alias = judge_trace_v2(
        _trace(episode, _decision(action="reuse_compilation", compilation_snapshot_id="pre-drift", used_artifact_ids=[primary])),
        episode,
    )
    assert by_id.correct is True
    assert by_alias.correct is True


def test_success_claim_uses_current_run_result_not_self_reported_support():
    episode = _episode(TaskType.FIDELITY_CLAIM, DriftRelevance.RELATED)
    primary = episode.evidence[0].evidence_id
    run = _call(1, "run_circuit", primary, {"execution_snapshot": "observed-after-event", "measured_success": False})
    correct = judge_trace_v2(_trace(episode, _decision(action="rerun_circuit", claimed_success=False), [run]), episode)
    contradicted = judge_trace_v2(_trace(episode, _decision(action="rerun_circuit", claimed_success=True, supporting_artifact_ids=[primary]), [run]), episode)
    assert correct.correct is True
    assert contradicted.correct is False
    assert "incorrect_success_claim" in contradicted.error_codes


def test_unreachable_capacity_is_derived_from_tool_trace_and_initial_snapshot_is_registered():
    episode = _episode(TaskType.UNREACHABLE_TARGET, DriftRelevance.RELATED)
    primary = episode.evidence[0].evidence_id
    call = _call(1, "get_backend_health", primary, {"available_capacity": 1, "snapshot": "observed-after-event"})
    result = judge_trace_v2(
        _trace(episode, _decision(action="declare_unreachable", declared_unreachable=True, used_artifact_ids=[episode.initial_state["snapshot_id"]]), [call]),
        episode,
    )
    assert result.correct is True
    assert "unknown_evidence_reference" not in result.error_codes


def test_repeated_and_irrelevant_calls_are_secondary_not_decision_overrides():
    episode = _episode(TaskType.UNREACHABLE_TARGET, DriftRelevance.RELATED)
    primary = episode.evidence[0].evidence_id
    health = _call(1, "get_backend_health", primary, {"available_capacity": 1, "snapshot": "observed-after-event"})
    duplicate = dict(health, call_index=2)
    extra = _call(3, "compare_backends", primary, {"ranked_backends": ["FakeSherbrooke"]})
    result = judge_trace_v2(_trace(episode, _decision(action="declare_unreachable", declared_unreachable=True), [health, duplicate, extra]), episode)
    assert result.correct is True
    assert "repeated_tool_call" in result.secondary_error_tags
    assert "irrelevant_tool_call" in result.secondary_error_tags


def test_incomplete_trace_is_unscorable():
    episode = _episode(TaskType.BACKEND_SELECTION, DriftRelevance.UNRELATED)
    trace = _trace(episode, _decision(action="select_backend", selected_backend="FakeBrisbane"))
    trace["trace_complete"] = False
    result = judge_trace_v2(trace, episode)
    assert result.status == "unscorable"
    assert result.correct is None


def test_development_scorer_rejects_sealed_method_validation_inputs(tmp_path):
    prohibited = tmp_path / "sealed-method-validation-v2.1" / "data.jsonl"
    prohibited.parent.mkdir()
    prohibited.write_text("{}\n", encoding="utf-8")
    try:
        _assert_development_input(prohibited)
    except ValueError as exc:
        assert "prohibited" in str(exc)
    else:
        raise AssertionError("sealed method-validation path must be rejected")
