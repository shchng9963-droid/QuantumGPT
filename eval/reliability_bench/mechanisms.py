"""Deterministic Stage-A audit of invalidation mechanisms."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .groups import ExperimentGroup
from .schema import Episode, expected_invalidated_artifacts


@dataclass(frozen=True)
class InvalidationOutcome:
    group: ExperimentGroup
    episode_id: str
    invalidated_artifact_ids: tuple[str, ...]
    preserved_artifact_ids: tuple[str, ...]
    automatic: bool


@dataclass(frozen=True)
class InvalidationMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    false_invalidation_rate: float
    valid_evidence_preservation_rate: float
    classification_accuracy: float


def run_invalidation_mechanism(
    group: ExperimentGroup, episode: Episode
) -> InvalidationOutcome:
    all_ids = tuple(item.evidence_id for item in episode.evidence)
    if group is ExperimentGroup.GLOBAL_REVALIDATE:
        invalidated = all_ids
        automatic = True
    elif group is ExperimentGroup.FULL_SELECTIVE:
        invalidated = expected_invalidated_artifacts(episode)
        automatic = True
    elif group is ExperimentGroup.ORACLE:
        invalidated = episode.ground_truth.invalidated_artifact_ids
        automatic = True
    else:
        # ReAct, Monitor, Ledger, and Prompt-Oracle do not mutate controller
        # validity state. Prompt-Oracle supplies text only and is evaluated later.
        invalidated = ()
        automatic = False
    invalidated_set = set(invalidated)
    preserved = tuple(item for item in all_ids if item not in invalidated_set)
    return InvalidationOutcome(
        group=group,
        episode_id=episode.episode_id,
        invalidated_artifact_ids=invalidated,
        preserved_artifact_ids=preserved,
        automatic=automatic,
    )


def score_invalidation_outcomes(
    episodes: Iterable[Episode],
    outcomes: Iterable[InvalidationOutcome],
) -> InvalidationMetrics:
    episode_map = {item.episode_id: item for item in episodes}
    tp = fp = fn = tn = 0
    preservation_correct = preservation_total = 0
    for outcome in outcomes:
        episode = episode_map[outcome.episode_id]
        expected = set(episode.ground_truth.invalidated_artifact_ids)
        predicted = set(outcome.invalidated_artifact_ids)
        all_ids = {item.evidence_id for item in episode.evidence}
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        tn += len(all_ids - expected - predicted)
        expected_preserved = set(episode.ground_truth.preserved_artifact_ids)
        actual_preserved = set(outcome.preserved_artifact_ids)
        preservation_correct += len(expected_preserved & actual_preserved)
        preservation_total += len(expected_preserved)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_rate = fp / (fp + tn) if fp + tn else 0.0
    preservation_rate = (
        preservation_correct / preservation_total if preservation_total else 1.0
    )
    total = tp + fp + fn + tn
    classification_accuracy = (tp + tn) / total if total else 1.0
    return InvalidationMetrics(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=round(precision, 6),
        recall=round(recall, 6),
        false_invalidation_rate=round(false_rate, 6),
        valid_evidence_preservation_rate=round(preservation_rate, 6),
        classification_accuracy=round(classification_accuracy, 6),
    )


def build_stage_a_mechanism_audit(episodes: list[Episode]) -> dict:
    groups: dict[str, dict] = {}
    for group in ExperimentGroup:
        outcomes = [run_invalidation_mechanism(group, item) for item in episodes]
        metrics = score_invalidation_outcomes(episodes, outcomes)
        groups[group.value] = {
            **asdict(metrics),
            "automatic_invalidation": any(item.automatic for item in outcomes),
            "invalidated_artifact_count": sum(
                len(item.invalidated_artifact_ids) for item in outcomes
            ),
        }

    full = groups[ExperimentGroup.FULL_SELECTIVE.value]
    oracle = groups[ExperimentGroup.ORACLE.value]
    global_group = groups[ExperimentGroup.GLOBAL_REVALIDATE.value]
    acceptance = {
        "full_precision_is_one": full["precision"] == 1.0,
        "full_recall_is_one": full["recall"] == 1.0,
        "oracle_matches_ground_truth": (
            oracle["precision"] == 1.0 and oracle["recall"] == 1.0
        ),
        "full_has_zero_false_invalidation": (
            full["false_invalidation_rate"] == 0.0
        ),
        "global_invalidates_more_than_full": (
            global_group["invalidated_artifact_count"]
            > full["invalidated_artifact_count"]
        ),
        "global_has_more_false_positives": (
            global_group["false_positive"] > full["false_positive"]
        ),
    }
    return {
        "episode_count": len(episodes),
        "pair_count": len({item.pair_id for item in episodes}),
        "artifact_count": sum(len(item.evidence) for item in episodes),
        "groups": groups,
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
        "relevance_counts": dict(
            sorted(Counter(item.drift_event.relevance.value for item in episodes).items())
        ),
    }


def write_stage_a_mechanism_audit(
    episodes: list[Episode], output_path: Path
) -> dict:
    audit = build_stage_a_mechanism_audit(episodes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


__all__ = [
    "InvalidationMetrics",
    "InvalidationOutcome",
    "build_stage_a_mechanism_audit",
    "run_invalidation_mechanism",
    "score_invalidation_outcomes",
    "write_stage_a_mechanism_audit",
]
