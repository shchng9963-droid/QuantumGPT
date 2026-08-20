"""Versioned schema and invariants for ReliabilityBench-Q episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


SCHEMA_VERSION = "reliabilitybench-q/1.0"
LATEST_SCHEMA_VERSION = "reliabilitybench-q/1.1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION, LATEST_SCHEMA_VERSION})


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


def _validate_v11_semantics(episode: Episode) -> list[str]:
    """Validate hidden task predicates and evidence lineage introduced in v1.1."""
    errors: list[str] = []
    payload = episode.ground_truth.acceptable_payload
    predicate = payload.get("task_predicate")
    if not isinstance(predicate, dict):
        errors.append("v1.1 requires a task_predicate object")
        predicate = {}
    if not str(predicate.get("name", "")).strip():
        errors.append("task_predicate.name must be non-empty")
    if predicate.get("version") != "2.0":
        errors.append("task_predicate.version must be 2.0")
    expected_predicate_names = {
        TaskType.BACKEND_SELECTION: "backend_in_acceptable_set",
        TaskType.QUBIT_MAPPING: "safe_qubit_mapping",
        TaskType.TRANSPILATION: "snapshot_lineage_phase",
        TaskType.FIDELITY_CLAIM: "fidelity_claim_matches_current_result",
        TaskType.MITIGATION_DECISION: "mitigation_in_acceptable_set",
        TaskType.UNREACHABLE_TARGET: "reachability_matches_current_capacity",
    }
    if predicate.get("name") != expected_predicate_names[episode.task_type]:
        errors.append("task_predicate.name is inconsistent with task_type")
    if (
        payload.get("predicate_registry_version")
        != "reliabilitybench-q/task-predicates-2.0"
    ):
        errors.append("predicate_registry_version must be task-predicates-2.0")

    evidence_policy = payload.get("evidence_policy")
    if not isinstance(evidence_policy, dict):
        errors.append("v1.1 requires an evidence_policy object")
        return errors
    registered = evidence_policy.get("registered_source_ids")
    lineage = evidence_policy.get("snapshot_lineage")
    current_snapshot_id = evidence_policy.get("current_snapshot_id")
    if (
        not isinstance(registered, list)
        or len(registered) != len(set(registered))
        or any(not isinstance(item, str) or not item for item in registered)
    ):
        errors.append("registered_source_ids must be a unique non-empty string list")
        registered_set: set[str] = set()
    else:
        registered_set = set(registered)
    required_sources = {episode.initial_state.get("snapshot_id")}
    required_sources.update(item.evidence_id for item in episode.evidence)
    if None in required_sources or not required_sources.issubset(registered_set):
        errors.append("initial snapshot and evidence records must be registered sources")

    if not isinstance(lineage, list) or not lineage:
        errors.append("snapshot_lineage must be a non-empty list")
        lineage_by_id: dict[str, dict[str, Any]] = {}
    else:
        lineage_by_id = {
            str(item.get("snapshot_id")): item
            for item in lineage
            if isinstance(item, dict) and item.get("snapshot_id")
        }
        if len(lineage_by_id) != len(lineage):
            errors.append("snapshot_lineage IDs must be unique and non-empty")
    initial_snapshot_id = episode.initial_state.get("snapshot_id")
    initial_lineage = lineage_by_id.get(str(initial_snapshot_id))
    if not initial_lineage or initial_lineage.get("phase") != "pre_drift":
        errors.append("initial snapshot must be registered at pre_drift phase")
    elif initial_lineage.get("parent_snapshot_id") is not None:
        errors.append("initial snapshot must not have a parent")
    current_lineage = lineage_by_id.get(str(current_snapshot_id))
    if not current_lineage or current_lineage.get("phase") != "current":
        errors.append("current snapshot must be registered at current phase")
    elif current_lineage.get("parent_snapshot_id") != initial_snapshot_id:
        errors.append("current snapshot must descend from the initial snapshot")
    if not set(lineage_by_id).issubset(registered_set):
        errors.append("every snapshot lineage node must be a registered source")

    if episode.task_type is TaskType.QUBIT_MAPPING:
        if predicate.get("name") != "safe_qubit_mapping":
            errors.append("qubit_mapping requires safe_qubit_mapping predicate")
        candidates = predicate.get("candidate_qubits")
        forbidden = predicate.get("forbidden_qubits")
        required_count = predicate.get("required_qubit_count")
        if (
            not isinstance(candidates, list)
            or len(candidates) != len(set(candidates))
            or candidates != episode.task_constraints.get("candidate_qubits")
        ):
            errors.append("mapping candidates must equal unique task candidate_qubits")
        if not isinstance(forbidden, list) or not set(forbidden).issubset(
            set(candidates or [])
        ):
            errors.append("forbidden_qubits must be a subset of mapping candidates")
        if not isinstance(required_count, int) or required_count <= 0:
            errors.append("required_qubit_count must be positive")
        elif len(set(candidates or []) - set(forbidden or [])) < required_count:
            errors.append("mapping predicate must admit at least one safe set")
        if "acceptable_qubit_sets" in payload:
            errors.append("v1.1 mapping must use a predicate, not one enumerated answer")

    if episode.task_type is TaskType.TRANSPILATION:
        if predicate.get("name") != "snapshot_lineage_phase":
            errors.append("transpilation requires snapshot_lineage_phase predicate")
        required_snapshot_id = predicate.get("required_snapshot_id")
        if required_snapshot_id not in lineage_by_id:
            errors.append("required compilation snapshot must exist in lineage")
        required_phase = predicate.get("required_phase")
        if required_phase not in {"pre_drift", "current"}:
            errors.append("required compilation phase must be pre_drift or current")
        elif lineage_by_id.get(str(required_snapshot_id), {}).get("phase") != required_phase:
            errors.append("required compilation snapshot phase is inconsistent")
    return errors


def validate_episode(episode: Episode) -> None:
    errors: list[str] = []
    if episode.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
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
    if episode.schema_version == LATEST_SCHEMA_VERSION:
        errors.extend(_validate_v11_semantics(episode))
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
    "LATEST_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
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
