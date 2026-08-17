"""Versioned schema and invariants for ReliabilityBench-Q episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


SCHEMA_VERSION = "reliabilitybench-q/1.0"


class TaskType(str, Enum):
    BACKEND_SELECTION = "backend_selection"
    QUBIT_MAPPING = "qubit_mapping"
    TRANSPILATION = "transpilation"
    FIDELITY_CLAIM = "fidelity_claim"
    MITIGATION_DECISION = "mitigation_decision"
    UNREACHABLE_TARGET = "unreachable_target"


class DriftRelevance(str, Enum):
    RELATED = "related"
    UNRELATED = "unrelated"


class DriftSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class DriftPhase(str, Enum):
    AFTER_OBSERVATION = "after_observation"
    AFTER_COMPILATION = "after_compilation"
    AFTER_EXECUTION = "after_execution"
    BEFORE_FINAL_ANSWER = "before_final_answer"


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    evidence_type: str
    depends_on_resources: tuple[str, ...]
    created_at_phase: str = "pre_drift"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftEvent:
    event_id: str
    phase: DriftPhase
    severity: DriftSeverity
    relevance: DriftRelevance
    changed_features: tuple[str, ...]
    affected_resources: tuple[str, ...]
    magnitude_quantile: int
    magnitude: float
    severity_basis: str = "synthetic_stage_a_placeholder"

    def neutral_payload(self) -> dict[str, Any]:
        """Return the action-neutral event shown to eligible methods."""
        return {
            "event_id": self.event_id,
            "phase": self.phase.value,
            "changed_features": list(self.changed_features),
            "change_magnitude": self.magnitude,
            "magnitude_quantile": self.magnitude_quantile,
        }


@dataclass(frozen=True)
class GroundTruth:
    invalidated_artifact_ids: tuple[str, ...]
    preserved_artifact_ids: tuple[str, ...]
    acceptable_actions: tuple[str, ...]
    required_revalidation_actions: tuple[str, ...]
    claim_support_predicate: str
    acceptable_payload: dict[str, Any]
    failure_reason: str | None = None


@dataclass(frozen=True)
class Episode:
    episode_id: str
    pair_id: str
    template_id: str
    task_type: TaskType
    circuit: str
    backend: str
    initial_state: dict[str, Any]
    evidence: tuple[EvidenceRecord, ...]
    drift_event: DriftEvent
    ground_truth: GroundTruth
    task_constraints: dict[str, Any]
    seed: int
    schema_version: str = SCHEMA_VERSION
    source: str = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def episode_from_dict(data: dict[str, Any]) -> Episode:
    evidence = tuple(
        EvidenceRecord(
            evidence_id=item["evidence_id"],
            evidence_type=item["evidence_type"],
            depends_on_resources=tuple(item["depends_on_resources"]),
            created_at_phase=item.get("created_at_phase", "pre_drift"),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in data["evidence"]
    )
    event_data = data["drift_event"]
    event = DriftEvent(
        event_id=event_data["event_id"],
        phase=DriftPhase(event_data["phase"]),
        severity=DriftSeverity(event_data["severity"]),
        relevance=DriftRelevance(event_data["relevance"]),
        changed_features=tuple(event_data["changed_features"]),
        affected_resources=tuple(event_data["affected_resources"]),
        magnitude_quantile=int(event_data["magnitude_quantile"]),
        magnitude=float(event_data["magnitude"]),
        severity_basis=event_data.get(
            "severity_basis", "synthetic_stage_a_placeholder"
        ),
    )
    truth_data = data["ground_truth"]
    truth = GroundTruth(
        invalidated_artifact_ids=tuple(truth_data["invalidated_artifact_ids"]),
        preserved_artifact_ids=tuple(truth_data["preserved_artifact_ids"]),
        acceptable_actions=tuple(truth_data["acceptable_actions"]),
        required_revalidation_actions=tuple(
            truth_data["required_revalidation_actions"]
        ),
        claim_support_predicate=truth_data["claim_support_predicate"],
        acceptable_payload=dict(truth_data["acceptable_payload"]),
        failure_reason=truth_data.get("failure_reason"),
    )
    return Episode(
        episode_id=data["episode_id"],
        pair_id=data["pair_id"],
        template_id=data["template_id"],
        task_type=TaskType(data["task_type"]),
        circuit=data["circuit"],
        backend=data["backend"],
        initial_state=dict(data["initial_state"]),
        evidence=evidence,
        drift_event=event,
        ground_truth=truth,
        task_constraints=dict(data["task_constraints"]),
        seed=int(data["seed"]),
        schema_version=data.get("schema_version", SCHEMA_VERSION),
        source=data.get("source", "synthetic"),
    )


def expected_invalidated_artifacts(episode: Episode) -> tuple[str, ...]:
    affected = set(episode.drift_event.affected_resources)
    return tuple(
        item.evidence_id
        for item in episode.evidence
        if affected.intersection(item.depends_on_resources)
    )


def validate_episode(episode: Episode) -> None:
    errors: list[str] = []
    if episode.schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version={episode.schema_version}")
    if not episode.episode_id or not episode.pair_id or not episode.template_id:
        errors.append("episode, pair, and template identifiers must be non-empty")
    evidence_ids = [item.evidence_id for item in episode.evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("evidence_id values must be unique")
    invalidated = set(episode.ground_truth.invalidated_artifact_ids)
    preserved = set(episode.ground_truth.preserved_artifact_ids)
    if invalidated & preserved:
        errors.append("invalidated and preserved evidence must be disjoint")
    if invalidated | preserved != set(evidence_ids):
        errors.append("ground truth must partition every evidence record")
    expected = set(expected_invalidated_artifacts(episode))
    if invalidated != expected:
        errors.append(
            "invalidated evidence does not match the resource dependency graph"
        )
    if episode.drift_event.relevance is DriftRelevance.RELATED and not invalidated:
        errors.append("related drift must invalidate at least one evidence record")
    if episode.drift_event.relevance is DriftRelevance.UNRELATED and invalidated:
        errors.append("unrelated drift must not invalidate task evidence")
    if episode.drift_event.magnitude_quantile not in {50, 75, 95}:
        errors.append("magnitude_quantile must be one of 50, 75, or 95")
    if not episode.ground_truth.acceptable_actions:
        errors.append("at least one acceptable action is required")
    if errors:
        raise ValueError(f"invalid episode {episode.episode_id}: " + "; ".join(errors))


def validate_counterfactual_pairs(episodes: Iterable[Episode]) -> None:
    pairs: dict[str, list[Episode]] = {}
    episode_ids: set[str] = set()
    for episode in episodes:
        validate_episode(episode)
        if episode.episode_id in episode_ids:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        episode_ids.add(episode.episode_id)
        pairs.setdefault(episode.pair_id, []).append(episode)

    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"counterfactual pair {pair_id} must contain two episodes")
        by_relevance = {item.drift_event.relevance: item for item in pair}
        if set(by_relevance) != {
            DriftRelevance.RELATED,
            DriftRelevance.UNRELATED,
        }:
            raise ValueError(f"counterfactual pair {pair_id} lacks both relevance arms")
        related = by_relevance[DriftRelevance.RELATED]
        unrelated = by_relevance[DriftRelevance.UNRELATED]
        invariant_fields = (
            related.template_id == unrelated.template_id,
            related.task_type == unrelated.task_type,
            related.circuit == unrelated.circuit,
            related.backend == unrelated.backend,
            related.initial_state == unrelated.initial_state,
            related.evidence == unrelated.evidence,
            related.task_constraints == unrelated.task_constraints,
            related.seed == unrelated.seed,
            related.drift_event.phase == unrelated.drift_event.phase,
            related.drift_event.severity == unrelated.drift_event.severity,
            related.drift_event.magnitude == unrelated.drift_event.magnitude,
        )
        if not all(invariant_fields):
            raise ValueError(
                f"counterfactual pair {pair_id} changes fields beyond relevance"
            )


__all__ = [
    "SCHEMA_VERSION",
    "DriftEvent",
    "DriftPhase",
    "DriftRelevance",
    "DriftSeverity",
    "Episode",
    "EvidenceRecord",
    "GroundTruth",
    "TaskType",
    "episode_from_dict",
    "expected_invalidated_artifacts",
    "validate_counterfactual_pairs",
    "validate_episode",
]
