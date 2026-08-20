"""Independent executable task predicates for ReliabilityBench-Q outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .measurement_spec_v2 import PRIMARY_UTILITY_WEIGHTS
from .schema import Episode, TaskType
from .terminal_schema_v2 import TerminalDecision, TerminalStatus


TASK_EVALUATOR_VERSION = "reliabilitybench-q/task-evaluator-3.0-dev"


@dataclass(frozen=True)
class TaskEvaluation:
    evaluator_version: str
    predicate_satisfied: bool | None
    terminal_correct: bool
    wrong_decision: bool | None
    answered: bool
    feasible: bool
    feasible_completion: bool
    correct_rejection: bool
    unnecessary_abstain: bool
    error_codes: tuple[str, ...]


def _payload_errors(episode: Episode, decision: TerminalDecision) -> list[str]:
    predicate = episode.ground_truth.acceptable_payload["task_predicate"]
    payload = decision.payload
    expected_payload_keys = {
        TaskType.BACKEND_SELECTION: {"backend_id"},
        TaskType.QUBIT_MAPPING: {"qubit_ids"},
        TaskType.TRANSPILATION: {"compilation_snapshot_id"},
        TaskType.FIDELITY_CLAIM: {"claimed_success"},
        TaskType.MITIGATION_DECISION: {
            "mitigation_policy",
            "intervention_requested",
        },
        TaskType.UNREACHABLE_TARGET: {"declared_unreachable"},
    }[episode.task_type]
    if set(payload) != expected_payload_keys:
        return ["payload_schema_mismatch"]
    if episode.task_type is TaskType.BACKEND_SELECTION:
        return [] if payload.get("backend_id") in set(
            predicate["acceptable_backends"]
        ) else ["unacceptable_backend"]
    if episode.task_type is TaskType.QUBIT_MAPPING:
        selected = payload.get("qubit_ids")
        if not isinstance(selected, list):
            return ["invalid_qubit_payload"]
        candidates = set(predicate["candidate_qubits"])
        forbidden = set(predicate["forbidden_qubits"])
        safe = (
            len(selected) == predicate["required_qubit_count"]
            and len(selected) == len(set(selected))
            and set(selected).issubset(candidates)
            and not set(selected).intersection(forbidden)
        )
        return [] if safe else ["unsafe_qubit_mapping"]
    if episode.task_type is TaskType.TRANSPILATION:
        return [] if payload.get("compilation_snapshot_id") == predicate[
            "required_snapshot_id"
        ] else ["outdated_compilation"]
    if episode.task_type is TaskType.FIDELITY_CLAIM:
        return [] if payload.get("claimed_success") is predicate[
            "expected_success_claim"
        ] else ["incorrect_success_claim"]
    if episode.task_type is TaskType.MITIGATION_DECISION:
        requested = payload.get("intervention_requested")
        if not isinstance(requested, bool):
            return ["invalid_intervention_request"]
        return [] if payload.get("mitigation_policy") in set(
            predicate["acceptable_mitigation_actions"]
        ) else ["incorrect_mitigation_policy"]
    return [] if payload.get("declared_unreachable") is predicate[
        "expected_unreachable"
    ] else ["incorrect_reachability_decision"]


def _is_feasible(episode: Episode) -> bool:
    if episode.task_type is not TaskType.UNREACHABLE_TARGET:
        return True
    predicate = episode.ground_truth.acceptable_payload["task_predicate"]
    return not bool(predicate["expected_unreachable"])


def evaluate_task_decision(
    episode: Episode, decision: TerminalDecision
) -> TaskEvaluation:
    """Evaluate set-valued task semantics without reading controller identity."""
    errors: list[str] = []
    if decision.task_type != episode.task_type.value:
        errors.append("task_type_mismatch")
    if decision.status is TerminalStatus.ABSTAIN:
        feasible = _is_feasible(episode)
        return TaskEvaluation(
            evaluator_version=TASK_EVALUATOR_VERSION,
            predicate_satisfied=None,
            terminal_correct=False,
            wrong_decision=None,
            answered=False,
            feasible=feasible,
            feasible_completion=False,
            correct_rejection=False,
            unnecessary_abstain=feasible,
            error_codes=tuple(errors or ["abstained"]),
        )

    if decision.task_action not in set(episode.ground_truth.acceptable_actions):
        errors.append("unacceptable_task_action")
    errors.extend(_payload_errors(episode, decision))
    errors = list(dict.fromkeys(errors))
    predicate_satisfied = not errors
    feasible = _is_feasible(episode)
    expected_status = TerminalStatus.SUCCESS if feasible else TerminalStatus.FAILURE
    if decision.status is not expected_status:
        errors.append("terminal_status_mismatch")
    terminal_correct = predicate_satisfied and decision.status is expected_status
    return TaskEvaluation(
        evaluator_version=TASK_EVALUATOR_VERSION,
        predicate_satisfied=predicate_satisfied,
        terminal_correct=terminal_correct,
        wrong_decision=not terminal_correct,
        answered=True,
        feasible=feasible,
        feasible_completion=bool(feasible and terminal_correct),
        correct_rejection=bool((not feasible) and terminal_correct),
        unnecessary_abstain=False,
        error_codes=tuple(dict.fromkeys(errors)),
    )


def aggregate_selective_metrics(
    evaluations: Sequence[TaskEvaluation],
    *,
    costs: Sequence[float] | None = None,
    utility_weights: Mapping[str, float] = PRIMARY_UTILITY_WEIGHTS,
) -> dict[str, float | int | None]:
    """Aggregate coverage/risk and an explicitly weighted overall utility."""
    if not evaluations:
        raise ValueError("at least one task evaluation is required")
    if costs is None:
        costs = [0.0] * len(evaluations)
    if len(costs) != len(evaluations):
        raise ValueError("costs must align one-to-one with evaluations")
    if any(cost < 0 for cost in costs):
        raise ValueError("costs must be non-negative")
    weights = dict(utility_weights)
    if set(weights) != {"correct", "wrong", "unnecessary_abstain", "cost"}:
        raise ValueError(
            "utility weights must define correct/wrong/unnecessary_abstain/cost"
        )
    if any(
        not isinstance(value, (int, float)) or value < 0
        for value in weights.values()
    ):
        raise ValueError("utility weights must be non-negative numbers")
    count = len(evaluations)
    answered = [item for item in evaluations if item.answered]
    feasible = [item for item in evaluations if item.feasible]
    infeasible = [item for item in evaluations if not item.feasible]
    wrong_count = sum(item.wrong_decision is True for item in answered)
    utility_total = sum(
        weights["correct"] * int(item.terminal_correct)
        - weights["wrong"] * int(item.wrong_decision is True)
        - weights["unnecessary_abstain"] * int(item.unnecessary_abstain)
        - weights["cost"] * float(cost)
        for item, cost in zip(evaluations, costs)
    )
    return {
        "episode_count": count,
        "answered_count": len(answered),
        "coverage": len(answered) / count,
        "selective_risk": wrong_count / len(answered) if answered else None,
        "feasible_completion_rate": (
            sum(item.feasible_completion for item in feasible) / len(feasible)
            if feasible
            else None
        ),
        "correct_rejection_rate": (
            sum(item.correct_rejection for item in infeasible) / len(infeasible)
            if infeasible
            else None
        ),
        "unnecessary_abstain_rate": (
            sum(item.unnecessary_abstain for item in feasible) / len(feasible)
            if feasible
            else None
        ),
        "overall_utility": utility_total / count,
    }


__all__ = [
    "TASK_EVALUATOR_VERSION",
    "TaskEvaluation",
    "aggregate_selective_metrics",
    "evaluate_task_decision",
]
