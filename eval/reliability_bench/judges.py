"""Deterministic, task-specific judges for ReliabilityBench-Q."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import Episode, TaskType


@dataclass(frozen=True)
class DecisionSubmission:
    action: str
    used_artifact_ids: tuple[str, ...] = ()
    revalidation_actions: tuple[str, ...] = ()
    selected_backend: str | None = None
    selected_qubits: tuple[int, ...] = ()
    compilation_snapshot_id: str | None = None
    claimed_success: bool | None = None
    supporting_artifact_ids: tuple[str, ...] = ()
    mitigation_action: str | None = None
    declared_unreachable: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeResult:
    correct: bool
    error_codes: tuple[str, ...]
    stale_evidence_reuse: bool
    unsupported_success_claim: bool
    details: dict[str, Any]


def _judge_backend(episode: Episode, decision: DecisionSubmission) -> list[str]:
    acceptable = set(episode.ground_truth.acceptable_payload["acceptable_backends"])
    if decision.selected_backend not in acceptable:
        return ["unacceptable_backend"]
    return []


def _judge_qubits(episode: Episode, decision: DecisionSubmission) -> list[str]:
    acceptable = {
        tuple(item)
        for item in episode.ground_truth.acceptable_payload["acceptable_qubit_sets"]
    }
    if tuple(decision.selected_qubits) not in acceptable:
        return ["unsafe_qubit_mapping"]
    return []


def _judge_transpilation(
    episode: Episode, decision: DecisionSubmission
) -> list[str]:
    required = episode.ground_truth.acceptable_payload[
        "required_compilation_snapshot_id"
    ]
    if decision.compilation_snapshot_id != required:
        return ["outdated_compilation"]
    return []


def _judge_fidelity(episode: Episode, decision: DecisionSubmission) -> list[str]:
    expected = bool(episode.ground_truth.acceptable_payload["expected_success_claim"])
    errors: list[str] = []
    if decision.claimed_success is not expected:
        errors.append("incorrect_success_claim")
    if decision.claimed_success and not decision.supporting_artifact_ids:
        errors.append("unsupported_success_claim")
    return errors


def _judge_mitigation(episode: Episode, decision: DecisionSubmission) -> list[str]:
    acceptable = set(
        episode.ground_truth.acceptable_payload["acceptable_mitigation_actions"]
    )
    if decision.mitigation_action not in acceptable:
        return ["incorrect_mitigation_decision"]
    return []


def _judge_unreachable(episode: Episode, decision: DecisionSubmission) -> list[str]:
    expected = bool(episode.ground_truth.acceptable_payload["expected_unreachable"])
    errors: list[str] = []
    if decision.declared_unreachable is not expected:
        errors.append("incorrect_reachability_decision")
    if expected and decision.claimed_success:
        errors.append("unreachable_target_false_success")
    return errors


TASK_JUDGES: dict[
    TaskType, Callable[[Episode, DecisionSubmission], list[str]]
] = {
    TaskType.BACKEND_SELECTION: _judge_backend,
    TaskType.QUBIT_MAPPING: _judge_qubits,
    TaskType.TRANSPILATION: _judge_transpilation,
    TaskType.FIDELITY_CLAIM: _judge_fidelity,
    TaskType.MITIGATION_DECISION: _judge_mitigation,
    TaskType.UNREACHABLE_TARGET: _judge_unreachable,
}


def judge_episode(episode: Episode, decision: DecisionSubmission) -> JudgeResult:
    errors: list[str] = []
    if decision.action not in episode.ground_truth.acceptable_actions:
        errors.append("unacceptable_task_action")

    evidence_ids = {item.evidence_id for item in episode.evidence}
    referenced = set(decision.used_artifact_ids) | set(
        decision.supporting_artifact_ids
    )
    unknown = referenced - evidence_ids
    if unknown:
        errors.append("unknown_evidence_reference")
    stale = bool(
        referenced.intersection(episode.ground_truth.invalidated_artifact_ids)
    )
    if stale:
        errors.append("stale_evidence_reuse")

    required = set(episode.ground_truth.required_revalidation_actions)
    performed = set(decision.revalidation_actions)
    if not required.issubset(performed):
        errors.append("missing_required_revalidation")

    errors.extend(TASK_JUDGES[episode.task_type](episode, decision))
    errors = list(dict.fromkeys(errors))
    unsupported = "unsupported_success_claim" in errors or (
        decision.claimed_success is True and stale
    )
    return JudgeResult(
        correct=not errors,
        error_codes=tuple(errors),
        stale_evidence_reuse=stale,
        unsupported_success_claim=unsupported,
        details={
            "episode_id": episode.episode_id,
            "task_type": episode.task_type.value,
            "required_revalidation_actions": sorted(required),
            "performed_revalidation_actions": sorted(performed),
        },
    )


def reference_submission(episode: Episode) -> DecisionSubmission:
    payload = episode.ground_truth.acceptable_payload
    preserved = episode.ground_truth.preserved_artifact_ids
    common = {
        "action": episode.ground_truth.acceptable_actions[0],
        "used_artifact_ids": preserved[:1],
        "revalidation_actions": episode.ground_truth.required_revalidation_actions,
    }
    if episode.task_type is TaskType.BACKEND_SELECTION:
        return DecisionSubmission(
            **common, selected_backend=payload["acceptable_backends"][0]
        )
    if episode.task_type is TaskType.QUBIT_MAPPING:
        return DecisionSubmission(
            **common, selected_qubits=tuple(payload["acceptable_qubit_sets"][0])
        )
    if episode.task_type is TaskType.TRANSPILATION:
        return DecisionSubmission(
            **common,
            compilation_snapshot_id=payload["required_compilation_snapshot_id"],
        )
    if episode.task_type is TaskType.FIDELITY_CLAIM:
        expected = bool(payload["expected_success_claim"])
        return DecisionSubmission(
            **common,
            claimed_success=expected,
            supporting_artifact_ids=preserved[:1] if expected else (),
        )
    if episode.task_type is TaskType.MITIGATION_DECISION:
        return DecisionSubmission(
            **common,
            mitigation_action=payload["acceptable_mitigation_actions"][0],
        )
    expected = bool(payload["expected_unreachable"])
    return DecisionSubmission(
        **common,
        declared_unreachable=expected,
        claimed_success=False if expected else None,
    )


def counterexample_submission(episode: Episode) -> DecisionSubmission:
    payload = episode.ground_truth.acceptable_payload
    stale_or_any = (
        episode.ground_truth.invalidated_artifact_ids[:1]
        or tuple(item.evidence_id for item in episode.evidence[:1])
    )
    common = {
        "action": "reuse_without_checking",
        "used_artifact_ids": stale_or_any,
        "revalidation_actions": (),
    }
    if episode.task_type is TaskType.BACKEND_SELECTION:
        return DecisionSubmission(
            **common, selected_backend=payload["unacceptable_backend"]
        )
    if episode.task_type is TaskType.QUBIT_MAPPING:
        return DecisionSubmission(
            **common, selected_qubits=tuple(payload["unsafe_qubit_set"])
        )
    if episode.task_type is TaskType.TRANSPILATION:
        return DecisionSubmission(
            **common,
            compilation_snapshot_id=payload["outdated_compilation_snapshot_id"],
        )
    if episode.task_type is TaskType.FIDELITY_CLAIM:
        return DecisionSubmission(
            **common, claimed_success=True, supporting_artifact_ids=stale_or_any
        )
    if episode.task_type is TaskType.MITIGATION_DECISION:
        return DecisionSubmission(
            **common, mitigation_action=payload["unacceptable_mitigation_action"]
        )
    expected = bool(payload["expected_unreachable"])
    return DecisionSubmission(
        **common,
        declared_unreachable=not expected,
        claimed_success=True if expected else None,
    )


__all__ = [
    "DecisionSubmission",
    "JudgeResult",
    "counterexample_submission",
    "judge_episode",
    "reference_submission",
]
