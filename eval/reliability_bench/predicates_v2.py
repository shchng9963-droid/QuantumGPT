"""Task predicates for ReliabilityBench-Q schema v1.1.

The predicates define sets of acceptable decisions. They intentionally avoid
encoding one arbitrary answer when several decisions satisfy the task.
"""

from __future__ import annotations

from typing import Any

from .schema import Episode, TaskType


PREDICATE_VERSION = "reliabilitybench-q/task-predicates-2.0"


def _decision_value(decision: Any, name: str) -> Any:
    if isinstance(decision, dict):
        return decision.get(name)
    return getattr(decision, name, None)


def evaluate_task_predicate(episode: Episode, decision: Any) -> list[str]:
    """Return task-semantic errors without inspecting controller identity."""
    predicate = episode.ground_truth.acceptable_payload["task_predicate"]
    if episode.task_type is TaskType.BACKEND_SELECTION:
        selected = _decision_value(decision, "selected_backend")
        return [] if selected in set(predicate["acceptable_backends"]) else [
            "unacceptable_backend"
        ]

    if episode.task_type is TaskType.QUBIT_MAPPING:
        selected = tuple(_decision_value(decision, "selected_qubits") or ())
        candidates = set(predicate["candidate_qubits"])
        forbidden = set(predicate["forbidden_qubits"])
        required_count = predicate["required_qubit_count"]
        safe = (
            len(selected) == required_count
            and len(selected) == len(set(selected))
            and set(selected).issubset(candidates)
            and not set(selected).intersection(forbidden)
        )
        return [] if safe else ["unsafe_qubit_mapping"]

    if episode.task_type is TaskType.TRANSPILATION:
        selected = _decision_value(decision, "compilation_snapshot_id")
        return [] if selected == predicate["required_snapshot_id"] else [
            "outdated_compilation"
        ]

    if episode.task_type is TaskType.FIDELITY_CLAIM:
        selected = _decision_value(decision, "claimed_success")
        return [] if selected is predicate["expected_success_claim"] else [
            "incorrect_success_claim"
        ]

    if episode.task_type is TaskType.MITIGATION_DECISION:
        selected = _decision_value(decision, "mitigation_action")
        return [] if selected in set(predicate["acceptable_mitigation_actions"]) else [
            "incorrect_mitigation_decision"
        ]

    selected = _decision_value(decision, "declared_unreachable")
    return [] if selected is predicate["expected_unreachable"] else [
        "incorrect_reachability_decision"
    ]


__all__ = ["PREDICATE_VERSION", "evaluate_task_predicate"]
