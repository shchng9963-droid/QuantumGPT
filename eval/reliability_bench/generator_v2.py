"""Leakage-safe candidate generator for ReliabilityBench-Q schema v1.1."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

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


GENERATOR_VERSION = "reliabilitybench-q/method-validation-generator-2.0"

BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeKyiv")
CIRCUITS = (
    "GHZ-4",
    "DJ-5",
    "GraphState-4",
    "QFTent-5",
    "VQE_SU2-4",
    "QAOA-5",
    "BV-4",
    "QFT-5",
    "VQE_HEA-4",
    "MaxCut-5",
    "GHZ-6",
    "DJ-4",
)
PHASES = tuple(DriftPhase)
SEVERITIES = (
    (DriftSeverity.MODERATE, 75, 0.25),
    (DriftSeverity.SEVERE, 95, 0.60),
)


@dataclass(frozen=True)
class GeneratorV2Config:
    random_seed: int = 20260824
    pairs_per_task: int = 2
    candidate_qubit_slack: int = 2
    target_fidelity: float = 0.85
    cost_budget: int = 10
    source: str = "synthetic_method_validation_v2"


DEFAULT_CONFIG = GeneratorV2Config()


def canonical_config(config: GeneratorV2Config) -> dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "schema_version": LATEST_SCHEMA_VERSION,
        **asdict(config),
    }


def config_sha256(config: GeneratorV2Config) -> str:
    payload = json.dumps(
        canonical_config(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def circuit_width(circuit: str) -> int:
    match = re.search(r"[-_](\d+)$", circuit)
    if not match:
        raise ValueError(f"circuit width is not encoded in {circuit!r}")
    width = int(match.group(1))
    if width <= 0:
        raise ValueError("circuit width must be positive")
    return width


def _task_resources(task: TaskType, degraded_qubit: int) -> tuple[str, tuple[str, ...], str]:
    if task is TaskType.BACKEND_SELECTION:
        return "avg_2q_error", ("backend.primary.avg_2q_error",), "backend_ranking"
    if task is TaskType.QUBIT_MAPPING:
        return (
            "qubit_readout_error",
            (f"qubit.{degraded_qubit}.readout_error",),
            "qubit_mapping",
        )
    if task is TaskType.TRANSPILATION:
        return "coupling_map", ("topology.edge.1-2",), "transpiled_circuit"
    if task is TaskType.FIDELITY_CLAIM:
        return "avg_2q_error", ("backend.primary.avg_2q_error",), "circuit_result"
    if task is TaskType.MITIGATION_DECISION:
        return (
            "avg_readout_error",
            ("backend.primary.avg_readout_error",),
            "mitigation_estimate",
        )
    return "backend_capacity", ("backend.primary.capacity",), "feasibility_assessment"


def _task_semantics(
    *,
    task: TaskType,
    relevance: DriftRelevance,
    candidates: list[int],
    required_qubit_count: int,
    degraded_qubit: int,
    pre_snapshot_id: str,
    current_snapshot_id: str,
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
                "forbidden_qubits": [degraded_qubit] if related else [],
                "required_qubit_count": required_qubit_count,
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
                "required_snapshot_id": current_snapshot_id if related else pre_snapshot_id,
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
    config: GeneratorV2Config,
) -> Episode:
    task_index = list(TaskType).index(task)
    circuit = CIRCUITS[task_index * config.pairs_per_task + pair_index]
    width = circuit_width(circuit)
    candidates = list(range(width + config.candidate_qubit_slack))
    degraded_qubit = candidates[(task_index + pair_index) % len(candidates)]
    pair_id = f"RBQ2-{task.value}-{pair_index:02d}"
    backend = BACKENDS[(task_index + pair_index) % len(BACKENDS)]
    feature, resources, evidence_type = _task_resources(task, degraded_qubit)
    pre_snapshot_id = f"snapshot-{pair_id}-pre"
    current_snapshot_id = f"snapshot-{pair_id}-current"
    actions, revalidation, task_predicate = _task_semantics(
        task=task,
        relevance=relevance,
        candidates=candidates,
        required_qubit_count=width,
        degraded_qubit=degraded_qubit,
        pre_snapshot_id=pre_snapshot_id,
        current_snapshot_id=current_snapshot_id,
    )
    evidence = (
        EvidenceRecord(
            evidence_id=f"{pair_id}-primary-evidence",
            evidence_type=evidence_type,
            depends_on_resources=resources,
            metadata={"backend": backend, "circuit": circuit},
        ),
        EvidenceRecord(
            evidence_id=f"{pair_id}-context-evidence",
            evidence_type="task_context",
            depends_on_resources=("context.calibration_note",),
            metadata={"pair_index": pair_index},
        ),
    )
    severity, quantile, magnitude = SEVERITIES[pair_index % len(SEVERITIES)]
    phase = PHASES[(task_index + pair_index) % len(PHASES)]
    related = relevance is DriftRelevance.RELATED
    event_digest = hashlib.sha256(
        f"{config.random_seed}|{pair_id}|{relevance.value}".encode("utf-8")
    ).hexdigest()[:16]
    event = DriftEvent(
        event_id=f"evt-{event_digest}",
        phase=phase,
        severity=severity,
        relevance=relevance,
        changed_features=(feature,) if related else ("ambient_temperature",),
        affected_resources=resources if related else ("environment.ambient_temperature",),
        magnitude_quantile=quantile,
        magnitude=magnitude,
        severity_basis="development synthetic thresholds; calibrate before main study",
    )
    initial_state = {
        "snapshot_id": pre_snapshot_id,
        "snapshot_phase": "pre_drift",
        "backend": backend,
        "circuit": circuit,
        "avg_1q_error": round(0.001 + pair_index * 0.00001, 6),
        "avg_2q_error": round(0.010 + pair_index * 0.0002, 6),
        "avg_readout_error": round(0.020 + pair_index * 0.0003, 6),
        "available_qubits": candidates,
    }
    task_constraints = {
        "target_fidelity": config.target_fidelity,
        "candidate_backends": list(BACKENDS),
        "candidate_qubits": candidates,
        "required_qubit_count": width,
        "cost_budget": config.cost_budget,
    }
    provisional = Episode(
        episode_id=f"{pair_id}-{relevance.value}",
        pair_id=pair_id,
        template_id=f"template-v2-{task.value}-{pair_index:02d}",
        task_type=task,
        circuit=circuit,
        backend=backend,
        initial_state=initial_state,
        evidence=evidence,
        drift_event=event,
        ground_truth=GroundTruth((), (), (), (), "", {}),
        task_constraints=task_constraints,
        seed=config.random_seed + task_index * 100 + pair_index,
        schema_version=LATEST_SCHEMA_VERSION,
        source=config.source,
    )
    invalidated = expected_invalidated_artifacts(provisional)
    invalidated_set = set(invalidated)
    preserved = tuple(
        item.evidence_id
        for item in evidence
        if item.evidence_id not in invalidated_set
    )
    evidence_policy = {
        "policy_version": "2.0",
        "registered_source_ids": [
            *(item.evidence_id for item in evidence),
            pre_snapshot_id,
            current_snapshot_id,
        ],
        "current_snapshot_id": current_snapshot_id,
        "snapshot_lineage": [
            {
                "snapshot_id": pre_snapshot_id,
                "phase": "pre_drift",
                "parent_snapshot_id": None,
            },
            {
                "snapshot_id": current_snapshot_id,
                "phase": "current",
                "parent_snapshot_id": pre_snapshot_id,
            },
        ],
    }
    truth = GroundTruth(
        invalidated_artifact_ids=invalidated,
        preserved_artifact_ids=preserved,
        acceptable_actions=actions,
        required_revalidation_actions=revalidation,
        claim_support_predicate=(
            "claims may rely on registered sources that remain valid or were refreshed"
        ),
        acceptable_payload={
            "predicate_registry_version": PREDICATE_VERSION,
            "task_predicate": task_predicate,
            "evidence_policy": evidence_policy,
        },
        failure_reason=(
            f"{feature} changed for a task-dependent resource" if related else None
        ),
    )
    return replace(provisional, ground_truth=truth)


def generate_method_validation_candidates(
    config: GeneratorV2Config = DEFAULT_CONFIG,
) -> list[Episode]:
    if config.pairs_per_task != 2:
        raise ValueError("the preregistered v2 protocol requires two pairs per task")
    if config.candidate_qubit_slack < 1:
        raise ValueError("candidate_qubit_slack must leave at least one alternative")
    episodes = [
        _build_episode(task, pair_index, relevance, config)
        for task in TaskType
        for pair_index in range(config.pairs_per_task)
        for relevance in (DriftRelevance.RELATED, DriftRelevance.UNRELATED)
    ]
    validate_counterfactual_pairs(episodes)
    return episodes


__all__ = [
    "DEFAULT_CONFIG",
    "GENERATOR_VERSION",
    "GeneratorV2Config",
    "canonical_config",
    "circuit_width",
    "config_sha256",
    "generate_method_validation_candidates",
]
