"""Measurement v3.1 task outcomes including runtime-owned execution failure."""

from __future__ import annotations

from typing import Mapping, Sequence

from .measurement_spec_v3_1 import PRIMARY_UTILITY_WEIGHTS
from .runtime_terminal_v1 import RuntimeExecutionFailure
from .schema import Episode, TaskType
from .task_predicate_evaluator_v3 import TaskEvaluation, evaluate_task_decision
from .terminal_outcome_v3_1 import TerminalOutcome


TASK_EVALUATOR_VERSION = "reliabilitybench-q/task-evaluator-3.1-candidate"


def _is_feasible(episode: Episode) -> bool:
    if episode.task_type is not TaskType.UNREACHABLE_TARGET:
        return True
    predicate = episode.ground_truth.acceptable_payload["task_predicate"]
    return not bool(predicate["expected_unreachable"])


def evaluate_terminal_outcome(
    episode: Episode, outcome: TerminalOutcome
) -> TaskEvaluation:
    if not isinstance(outcome, RuntimeExecutionFailure):
        return evaluate_task_decision(episode, outcome)
    feasible = _is_feasible(episode)
    return TaskEvaluation(
        evaluator_version=TASK_EVALUATOR_VERSION,
        predicate_satisfied=None,
        terminal_correct=False,
        wrong_decision=None,
        answered=False,
        execution_failure=True,
        feasible=feasible,
        feasible_completion=False,
        correct_rejection=False,
        unnecessary_abstain=False,
        error_codes=(outcome.failure_reason,),
    )


def aggregate_selective_metrics(
    evaluations: Sequence[TaskEvaluation],
    *,
    costs: Sequence[float] | None = None,
    utility_weights: Mapping[str, float] = PRIMARY_UTILITY_WEIGHTS,
) -> dict[str, float | int | None]:
    if not evaluations:
        raise ValueError("at least one task evaluation is required")
    if costs is None:
        costs = [0.0] * len(evaluations)
    if len(costs) != len(evaluations):
        raise ValueError("costs must align one-to-one with evaluations")
    if any(cost < 0 for cost in costs):
        raise ValueError("costs must be non-negative")
    weights = dict(utility_weights)
    required = {
        "correct",
        "wrong",
        "unnecessary_abstain",
        "execution_failure",
        "cost",
    }
    if set(weights) != required:
        raise ValueError("utility weights must define the frozen v3.1 fields")
    if any(
        not isinstance(value, (int, float)) or value < 0
        for value in weights.values()
    ):
        raise ValueError("utility weights must be non-negative numbers")
    count = len(evaluations)
    answered = [item for item in evaluations if item.answered]
    failures = [item for item in evaluations if item.execution_failure]
    feasible = [item for item in evaluations if item.feasible]
    infeasible = [item for item in evaluations if not item.feasible]
    wrong_count = sum(item.wrong_decision is True for item in answered)
    utility_total = sum(
        weights["correct"] * int(item.terminal_correct)
        - weights["wrong"] * int(item.wrong_decision is True)
        - weights["unnecessary_abstain"] * int(item.unnecessary_abstain)
        - weights["execution_failure"] * int(item.execution_failure)
        - weights["cost"] * float(cost)
        for item, cost in zip(evaluations, costs)
    )
    return {
        "episode_count": count,
        "answered_count": len(answered),
        "coverage": len(answered) / count,
        "execution_failure_rate": len(failures) / count,
        "selective_risk": (
            wrong_count / len(answered) if answered else None
        ),
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
    "aggregate_selective_metrics",
    "evaluate_terminal_outcome",
]
