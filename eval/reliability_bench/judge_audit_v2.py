"""Independent audit generator and frozen protocol helpers for judge v2."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .generator_v2 import DEVELOPMENT_CONFIG, generate_method_validation_candidates
from .predicates_v2 import PREDICATE_VERSION
from .schema import (
    LATEST_SCHEMA_VERSION,
    DriftEvent,
    DriftPhase,
    DriftRelevance,
    DriftSeverity,
    Episode,
    EvidenceRecord,
    GroundTruth,
    TaskType,
    expected_invalidated_artifacts,
    validate_counterfactual_pairs,
)


AUDIT_VERSION = "reliabilitybench-q/judge-v2-independent-audit-1.1"
AUDIT_GENERATOR_VERSION = "reliabilitybench-q/judge-audit-generator-1.1"
AUDIT_EPISODE_COUNT = 24

RETIRED_AUDIT_CIRCUITS_V1 = (
    "AuditBackendA-4",
    "AuditBackendB-5",
    "AuditMapA-4",
    "AuditMapB-4",
    "AuditTranspileA-4",
    "AuditTranspileB-5",
    "AuditFidelityA-4",
    "AuditFidelityB-5",
    "AuditMitigationA-4",
    "AuditMitigationB-5",
    "AuditReachabilityA-5",
    "AuditReachabilityB-5",
)
AUDIT_CIRCUITS = (
    "IndependentBackendC-5",
    "IndependentBackendD-4",
    "IndependentMapC-5",
    "IndependentMapD-4",
    "IndependentTranspileC-5",
    "IndependentTranspileD-4",
    "IndependentFidelityC-5",
    "IndependentFidelityD-4",
    "IndependentMitigationC-5",
    "IndependentMitigationD-4",
    "IndependentReachabilityC-4",
    "IndependentReachabilityD-5",
)
BACKENDS = ("FakeKyiv", "FakeSherbrooke", "FakeBrisbane")


@dataclass(frozen=True)
class JudgeAuditGeneratorConfig:
    random_seed: int = 2026082011
    schedule_seed: int = 2026082012
    pairs_per_task: int = 2
    candidate_qubit_slack: int = 2
    target_fidelity: float = 0.85
    cost_budget: int = 10
    episode_namespace: str = "RBQJA2"
    dataset_revision: int = 2
    source: str = "independent_judge_audit_v2_1"


FROZEN_AUDIT_CONFIG = JudgeAuditGeneratorConfig()
RETIRED_AUDIT_CONFIG_V1 = JudgeAuditGeneratorConfig(
    random_seed=2026082001,
    schedule_seed=2026082002,
    episode_namespace="RBQJA",
    dataset_revision=1,
    source="independent_judge_audit_v2",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _width(circuit: str) -> int:
    width = int(circuit.rsplit("-", 1)[1])
    if width <= 0:
        raise ValueError("circuit width must be positive")
    return width


def _circuits(config: JudgeAuditGeneratorConfig) -> tuple[str, ...]:
    if config.dataset_revision == 1:
        return RETIRED_AUDIT_CIRCUITS_V1
    if config.dataset_revision == 2:
        return AUDIT_CIRCUITS
    raise ValueError("unsupported judge-audit dataset revision")


def _resources(task: TaskType, degraded: int) -> tuple[str, tuple[str, ...], str]:
    if task is TaskType.BACKEND_SELECTION:
        return "avg_2q_error", ("backend.primary.avg_2q_error",), "backend_ranking"
    if task is TaskType.QUBIT_MAPPING:
        return "qubit_readout_error", (f"qubit.{degraded}.readout_error",), "qubit_mapping"
    if task is TaskType.TRANSPILATION:
        return "coupling_map", ("topology.edge.1-2",), "transpiled_circuit"
    if task is TaskType.FIDELITY_CLAIM:
        return "avg_2q_error", ("backend.primary.avg_2q_error",), "circuit_result"
    if task is TaskType.MITIGATION_DECISION:
        return "avg_readout_error", ("backend.primary.avg_readout_error",), "mitigation_estimate"
    return "backend_capacity", ("backend.primary.capacity",), "feasibility_assessment"


def _semantics(
    task: TaskType,
    relevance: DriftRelevance,
    candidates: list[int],
    width: int,
    degraded: int,
    pre_snapshot: str,
    current_snapshot: str,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    related = relevance is DriftRelevance.RELATED
    common = {"version": "2.0"}
    if task is TaskType.BACKEND_SELECTION:
        return (
            ("select_backend",),
            ("get_backend_health", "compare_backends") if related else (),
            {
                **common,
                "name": "backend_in_acceptable_set",
                "acceptable_backends": ["FakeSherbrooke"] if related else ["FakeBrisbane"],
            },
        )
    if task is TaskType.QUBIT_MAPPING:
        return (
            ("select_qubits",),
            ("get_qubit_properties",) if related else (),
            {
                **common,
                "name": "safe_qubit_mapping",
                "candidate_qubits": candidates,
                "forbidden_qubits": [degraded] if related else [],
                "required_qubit_count": width,
            },
        )
    if task is TaskType.TRANSPILATION:
        return (
            ("retranspile",) if related else ("reuse_compilation",),
            ("get_coupling_map", "transpile_circuit") if related else (),
            {
                **common,
                "name": "snapshot_lineage_phase",
                "required_phase": "current" if related else "pre_drift",
                "required_snapshot_id": current_snapshot if related else pre_snapshot,
            },
        )
    if task is TaskType.FIDELITY_CLAIM:
        return (
            ("rerun_circuit",) if related else ("reuse_valid_result",),
            ("run_circuit",) if related else (),
            {
                **common,
                "name": "fidelity_claim_matches_current_result",
                "expected_success_claim": not related,
            },
        )
    if task is TaskType.MITIGATION_DECISION:
        return (
            ("reassess_mitigation",) if related else ("keep_mitigation",),
            ("apply_mitigation",) if related else (),
            {
                **common,
                "name": "mitigation_in_acceptable_set",
                "acceptable_mitigation_actions": [
                    "increase_mitigation" if related else "keep_mitigation"
                ],
            },
        )
    return (
        ("declare_unreachable",) if related else ("continue_execution",),
        ("get_backend_health",) if related else (),
        {
            **common,
            "name": "reachability_matches_current_capacity",
            "expected_unreachable": related,
        },
    )


def _build_episode(
    task: TaskType,
    pair_index: int,
    relevance: DriftRelevance,
    config: JudgeAuditGeneratorConfig,
) -> Episode:
    task_index = list(TaskType).index(task)
    circuit = _circuits(config)[task_index * config.pairs_per_task + pair_index]
    width = _width(circuit)
    candidates = list(range(width + config.candidate_qubit_slack))
    degraded = candidates[(task_index * 2 + pair_index + 1) % len(candidates)]
    pair_id = f"{config.episode_namespace}-{task.value}-{pair_index:02d}"
    backend = BACKENDS[(task_index + pair_index) % len(BACKENDS)]
    feature, dependencies, evidence_type = _resources(task, degraded)
    pre_snapshot = f"snapshot-{pair_id}-pre"
    current_snapshot = f"snapshot-{pair_id}-current"
    actions, required, predicate = _semantics(
        task,
        relevance,
        candidates,
        width,
        degraded,
        pre_snapshot,
        current_snapshot,
    )
    evidence = (
        EvidenceRecord(
            evidence_id=f"{pair_id}-primary-evidence",
            evidence_type=evidence_type,
            depends_on_resources=dependencies,
            metadata={"backend": backend, "circuit": circuit, "pair_index": pair_index},
        ),
        EvidenceRecord(
            evidence_id=f"{pair_id}-context-evidence",
            evidence_type="task_context",
            depends_on_resources=("context.audit_note",),
            metadata={"pair_index": pair_index},
        ),
    )
    severity = DriftSeverity.MODERATE if pair_index == 0 else DriftSeverity.SEVERE
    quantile, magnitude = (75, 0.30) if pair_index == 0 else (95, 0.65)
    phase = tuple(DriftPhase)[(task_index + pair_index + 1) % len(tuple(DriftPhase))]
    related = relevance is DriftRelevance.RELATED
    event_id = "evt-audit-" + hashlib.sha256(
        f"{config.random_seed}|{pair_id}|{relevance.value}".encode("utf-8")
    ).hexdigest()[:16]
    event = DriftEvent(
        event_id=event_id,
        phase=phase,
        severity=severity,
        relevance=relevance,
        changed_features=(feature,) if related else ("ambient_temperature",),
        affected_resources=dependencies if related else ("environment.ambient_temperature",),
        magnitude_quantile=quantile,
        magnitude=magnitude,
        severity_basis="independent judge audit preregistration",
    )
    initial_state = {
        "snapshot_id": pre_snapshot,
        "snapshot_phase": "pre_drift",
        "backend": backend,
        "circuit": circuit,
        "avg_1q_error": round(0.0012 + task_index * 0.00001 + pair_index * 0.00002, 6),
        "avg_2q_error": round(0.012 + task_index * 0.0002 + pair_index * 0.0003, 6),
        "avg_readout_error": round(0.024 + task_index * 0.0003 + pair_index * 0.0004, 6),
        "available_qubits": candidates,
    }
    constraints = {
        "target_fidelity": config.target_fidelity,
        "candidate_backends": list(BACKENDS),
        "candidate_qubits": candidates,
        "required_qubit_count": width,
        "cost_budget": config.cost_budget,
    }
    provisional = Episode(
        episode_id=f"{pair_id}-{relevance.value}",
        pair_id=pair_id,
        template_id=f"template-judge-audit-{task.value}-{pair_index:02d}",
        task_type=task,
        circuit=circuit,
        backend=backend,
        initial_state=initial_state,
        evidence=evidence,
        drift_event=event,
        ground_truth=GroundTruth((), (), (), (), "", {}),
        task_constraints=constraints,
        seed=config.random_seed + task_index * 100 + pair_index,
        schema_version=LATEST_SCHEMA_VERSION,
        source=config.source,
    )
    invalidated = expected_invalidated_artifacts(provisional)
    invalidated_set = set(invalidated)
    preserved = tuple(item.evidence_id for item in evidence if item.evidence_id not in invalidated_set)
    evidence_policy = {
        "policy_version": "2.0",
        "registered_source_ids": [
            *(item.evidence_id for item in evidence),
            pre_snapshot,
            current_snapshot,
        ],
        "current_snapshot_id": current_snapshot,
        "snapshot_lineage": [
            {"snapshot_id": pre_snapshot, "phase": "pre_drift", "parent_snapshot_id": None},
            {"snapshot_id": current_snapshot, "phase": "current", "parent_snapshot_id": pre_snapshot},
        ],
    }
    return Episode(
        **{
            **provisional.__dict__,
            "ground_truth": GroundTruth(
                invalidated_artifact_ids=invalidated,
                preserved_artifact_ids=preserved,
                acceptable_actions=actions,
                required_revalidation_actions=required,
                claim_support_predicate="latest valid trace evidence supports the final decision",
                acceptable_payload={
                    "predicate_registry_version": PREDICATE_VERSION,
                    "task_predicate": predicate,
                    "evidence_policy": evidence_policy,
                },
            ),
        }
    )


def generate_judge_audit_episodes(
    config: JudgeAuditGeneratorConfig = FROZEN_AUDIT_CONFIG,
) -> list[Episode]:
    if config.pairs_per_task != 2:
        raise ValueError("judge audit v1 requires exactly two pairs per task")
    episodes = [
        _build_episode(task, pair_index, relevance, config)
        for task in TaskType
        for pair_index in range(config.pairs_per_task)
        for relevance in (DriftRelevance.RELATED, DriftRelevance.UNRELATED)
    ]
    validate_counterfactual_pairs(episodes)
    if len(episodes) != AUDIT_EPISODE_COUNT:
        raise AssertionError("unexpected judge audit episode count")
    return episodes


def semantic_fingerprint(episode: Episode) -> str:
    payload = episode.ground_truth.acceptable_payload
    predicate = dict(payload.get("task_predicate") or {})
    predicate.pop("required_snapshot_id", None)
    semantic = {
        "task_type": episode.task_type.value,
        "circuit": episode.circuit,
        "backend": episode.backend,
        "initial_metrics": {
            key: value
            for key, value in episode.initial_state.items()
            if key not in {"snapshot_id", "snapshot_phase", "backend", "circuit"}
        },
        "evidence": [
            {"type": item.evidence_type, "dependencies": list(item.depends_on_resources)}
            for item in episode.evidence
        ],
        "drift": {
            "phase": episode.drift_event.phase.value,
            "severity": episode.drift_event.severity.value,
            "relevance": episode.drift_event.relevance.value,
            "changed_features": list(episode.drift_event.changed_features),
            "magnitude_quantile": episode.drift_event.magnitude_quantile,
            "magnitude": episode.drift_event.magnitude,
        },
        "constraints": episode.task_constraints,
        "predicate": predicate,
    }
    return _sha256(semantic)


def audit_deduplication(
    audit_episodes: Iterable[Episode],
    old_development_episodes: Iterable[Episode],
) -> dict[str, Any]:
    audit = list(audit_episodes)
    old = list(old_development_episodes)
    method_reference = generate_method_validation_candidates(DEVELOPMENT_CONFIG)
    retired_reference = generate_judge_audit_episodes(RETIRED_AUDIT_CONFIG_V1)
    audit_ids = {item.episode_id for item in audit}
    old_ids = {item.episode_id for item in old}
    method_ids = {item.episode_id for item in method_reference}
    retired_ids = {item.episode_id for item in retired_reference}
    audit_semantic = {semantic_fingerprint(item) for item in audit}
    old_semantic = {semantic_fingerprint(item) for item in old}
    method_semantic = {semantic_fingerprint(item) for item in method_reference}
    retired_semantic = {semantic_fingerprint(item) for item in retired_reference}
    report = {
        "audit_episode_count": len(audit),
        "old_development_id_overlap": sorted(audit_ids.intersection(old_ids)),
        "method_namespace_id_overlap": sorted(audit_ids.intersection(method_ids)),
        "retired_audit_id_overlap": sorted(audit_ids.intersection(retired_ids)),
        "old_development_semantic_hash_overlap": sorted(audit_semantic.intersection(old_semantic)),
        "method_generator_semantic_hash_overlap": sorted(audit_semantic.intersection(method_semantic)),
        "retired_audit_semantic_hash_overlap": sorted(
            audit_semantic.intersection(retired_semantic)
        ),
        "audit_episode_ids_sha256": _sha256(sorted(audit_ids)),
        "audit_semantic_hashes_sha256": _sha256(sorted(audit_semantic)),
        "method_validation_archive_accessed": False,
        "method_independence_basis": (
            "disjoint RBQJA namespace plus semantic fingerprints compared against the "
            "frozen public method-generator structure; sealed archive remains unopened"
        ),
    }
    report["passed"] = not any(
        report[key]
        for key in (
            "old_development_id_overlap",
            "method_namespace_id_overlap",
            "retired_audit_id_overlap",
            "old_development_semantic_hash_overlap",
            "method_generator_semantic_hash_overlap",
            "retired_audit_semantic_hash_overlap",
        )
    )
    return report


def _pair_index(episode: Episode) -> int:
    return int(episode.pair_id.rsplit("-", 1)[1])


def assigned_controller(episode: Episode) -> str:
    pair_index = _pair_index(episode)
    if episode.drift_event.relevance is DriftRelevance.RELATED:
        return "ledger_only" if pair_index == 0 else "full_selective"
    return "react" if pair_index == 0 else "ledger_only"


def build_audit_schedule(
    episodes: Iterable[Episode],
    seed: int = FROZEN_AUDIT_CONFIG.schedule_seed,
) -> list[dict[str, Any]]:
    ordered = sorted(episodes, key=lambda item: item.episode_id)
    random.Random(seed).shuffle(ordered)
    schedule = []
    for index, episode in enumerate(ordered):
        group = assigned_controller(episode)
        opaque = hashlib.sha256(
            f"{seed}|{episode.episode_id}|{group}".encode("utf-8")
        ).hexdigest()[:16]
        schedule.append(
            {
                "schedule_index": index,
                "episode_index": index,
                "within_episode_index": 0,
                "episode_id": episode.episode_id,
                "controller_group": group,
                "hidden_group_id": f"audit-arm-{opaque}",
            }
        )
    if len(schedule) != AUDIT_EPISODE_COUNT:
        raise ValueError("audit schedule must contain 24 fixed traces")
    return schedule


def validate_audit_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_version") != AUDIT_VERSION:
        raise ValueError("audit protocol version mismatch")
    if config.get("judge_candidate_commit") != "e2ceb18a610436f01b5f572a3dc75a36816b6c3b":
        raise ValueError("judge candidate commit is not frozen")
    if int(config.get("episode_count", 0)) != AUDIT_EPISODE_COUNT:
        raise ValueError("audit requires exactly 24 traces")
    if config.get("legacy_program_judge_enabled") is not False:
        raise ValueError("independent audit must disable development-only judge v1")
    if config.get("sampling", {}).get("post_result_selection_allowed") is not False:
        raise ValueError("post-result selection must be prohibited")
    thresholds = config.get("thresholds") or {}
    if thresholds.get("outcome", {}).get("agreement_min") != 0.90:
        raise ValueError("outcome agreement threshold must remain 0.90")
    if thresholds.get("outcome", {}).get("cohen_kappa_min") != 0.80:
        raise ValueError("outcome kappa threshold must remain 0.80")
    if thresholds.get("critical_labels", {}).get("f1_min") != 0.90:
        raise ValueError("critical-label F1 threshold must remain 0.90")


__all__ = [
    "AUDIT_EPISODE_COUNT",
    "AUDIT_GENERATOR_VERSION",
    "AUDIT_VERSION",
    "FROZEN_AUDIT_CONFIG",
    "JudgeAuditGeneratorConfig",
    "assigned_controller",
    "audit_deduplication",
    "build_audit_schedule",
    "generate_judge_audit_episodes",
    "semantic_fingerprint",
    "validate_audit_config",
]
