"""Deterministic Stage-A generator for matched Synthetic episodes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import (
    SCHEMA_VERSION,
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


CIRCUITS = (
    "GHZ-5",
    "DJ-5",
    "GraphState-5",
    "QFTent-5",
    "VQE_SU2-4",
    "ghz_5",
    "qft_4",
    "bv_5",
    "vqe_4",
    "qaoa_4",
)
BACKENDS = ("FakeBrisbane", "FakeSherbrooke", "FakeKyiv")
SEVERITY_CYCLE = (
    (DriftSeverity.MILD, 50, 0.10),
    (DriftSeverity.MODERATE, 75, 0.25),
    (DriftSeverity.SEVERE, 95, 0.60),
)
PHASE_CYCLE = tuple(DriftPhase)


def _task_spec(task: TaskType, index: int) -> dict[str, Any]:
    if task is TaskType.BACKEND_SELECTION:
        return {
            "feature": "avg_2q_error",
            "resources": ("backend.primary.avg_2q_error",),
            "evidence_type": "backend_ranking",
            "related_actions": ("select_backend",),
            "unrelated_actions": ("select_backend",),
            "related_revalidation": ("get_backend_health", "compare_backends"),
            "payload_related": {
                "acceptable_backends": ["FakeSherbrooke"],
                "unacceptable_backend": "FakeBrisbane",
            },
            "payload_unrelated": {
                "acceptable_backends": ["FakeBrisbane"],
                "unacceptable_backend": "FakeKyiv",
            },
        }
    if task is TaskType.QUBIT_MAPPING:
        bad = index % 4
        safe = [item for item in range(5) if item != bad][:3]
        return {
            "feature": "qubit_readout_error",
            "resources": (f"qubit.{bad}.readout_error",),
            "evidence_type": "qubit_mapping",
            "related_actions": ("select_qubits",),
            "unrelated_actions": ("select_qubits",),
            "related_revalidation": ("get_qubit_properties",),
            "payload_related": {
                "acceptable_qubit_sets": [safe],
                "unsafe_qubit_set": [bad, (bad + 1) % 5, (bad + 2) % 5],
            },
            "payload_unrelated": {
                "acceptable_qubit_sets": [[0, 1, 2]],
                "unsafe_qubit_set": [2, 3, 4],
            },
        }
    if task is TaskType.TRANSPILATION:
        return {
            "feature": "coupling_map",
            "resources": ("topology.edge.1-2",),
            "evidence_type": "transpiled_circuit",
            "related_actions": ("retranspile",),
            "unrelated_actions": ("reuse_compilation",),
            "related_revalidation": ("get_coupling_map", "transpile_circuit"),
            "payload_related": {
                "required_compilation_snapshot_id": "post-drift",
                "outdated_compilation_snapshot_id": "pre-drift",
            },
            "payload_unrelated": {
                "required_compilation_snapshot_id": "pre-drift",
                "outdated_compilation_snapshot_id": "unknown",
            },
        }
    if task is TaskType.FIDELITY_CLAIM:
        return {
            "feature": "avg_2q_error",
            "resources": ("backend.primary.avg_2q_error",),
            "evidence_type": "circuit_result",
            "related_actions": ("rerun_circuit",),
            "unrelated_actions": ("reuse_valid_result",),
            "related_revalidation": ("run_circuit",),
            "payload_related": {"expected_success_claim": False},
            "payload_unrelated": {"expected_success_claim": True},
        }
    if task is TaskType.MITIGATION_DECISION:
        return {
            "feature": "avg_readout_error",
            "resources": ("backend.primary.avg_readout_error",),
            "evidence_type": "mitigation_estimate",
            "related_actions": ("reassess_mitigation",),
            "unrelated_actions": ("keep_mitigation",),
            "related_revalidation": ("apply_mitigation",),
            "payload_related": {
                "acceptable_mitigation_actions": ["increase_mitigation"],
                "unacceptable_mitigation_action": "keep_mitigation",
            },
            "payload_unrelated": {
                "acceptable_mitigation_actions": ["keep_mitigation"],
                "unacceptable_mitigation_action": "increase_mitigation",
            },
        }
    return {
        "feature": "backend_capacity",
        "resources": ("backend.primary.capacity",),
        "evidence_type": "feasibility_assessment",
        "related_actions": ("declare_unreachable",),
        "unrelated_actions": ("continue_execution",),
        "related_revalidation": ("get_backend_health",),
        "payload_related": {"expected_unreachable": True},
        "payload_unrelated": {"expected_unreachable": False},
    }


def _build_episode(
    task: TaskType,
    template_index: int,
    relevance: DriftRelevance,
) -> Episode:
    task_index = list(TaskType).index(task)
    pair_id = f"RBQ-{task.value}-{template_index:02d}"
    spec = _task_spec(task, template_index)
    severity, quantile, magnitude = SEVERITY_CYCLE[template_index % 3]
    phase = PHASE_CYCLE[template_index % len(PHASE_CYCLE)]
    backend = BACKENDS[template_index % len(BACKENDS)]
    circuit = CIRCUITS[template_index]
    seed = 10_000 + task_index * 100 + template_index
    evidence = (
        EvidenceRecord(
            evidence_id=f"{pair_id}-primary-evidence",
            evidence_type=spec["evidence_type"],
            depends_on_resources=tuple(spec["resources"]),
            metadata={"circuit": circuit, "backend": backend},
        ),
        EvidenceRecord(
            evidence_id=f"{pair_id}-context-evidence",
            evidence_type="task_context",
            depends_on_resources=("context.calibration_note",),
            metadata={"template_index": template_index},
        ),
    )
    if relevance is DriftRelevance.RELATED:
        changed_features = (spec["feature"],)
        affected_resources = tuple(spec["resources"])
        actions = tuple(spec["related_actions"])
        revalidation = tuple(spec["related_revalidation"])
        payload = dict(spec["payload_related"])
        reason = f"{spec['feature']} changed for a task-dependent resource"
    else:
        changed_features = ("ambient_temperature",)
        affected_resources = ("environment.ambient_temperature",)
        actions = tuple(spec["unrelated_actions"])
        revalidation = ()
        payload = dict(spec["payload_unrelated"])
        reason = None

    event = DriftEvent(
        event_id=f"{pair_id}-{relevance.value}-event",
        phase=phase,
        severity=severity,
        relevance=relevance,
        changed_features=changed_features,
        affected_resources=affected_resources,
        magnitude_quantile=quantile,
        magnitude=magnitude,
    )
    initial_state = {
        "snapshot_id": f"snapshot-{pair_id}",
        "backend": backend,
        "circuit": circuit,
        "avg_1q_error": round(0.001 + template_index * 0.00001, 6),
        "avg_2q_error": round(0.010 + template_index * 0.0002, 6),
        "avg_readout_error": round(0.020 + template_index * 0.0003, 6),
        "available_qubits": list(range(5)),
    }
    provisional = Episode(
        episode_id=f"{pair_id}-{relevance.value}",
        pair_id=pair_id,
        template_id=f"template-{task.value}-{template_index:02d}",
        task_type=task,
        circuit=circuit,
        backend=backend,
        initial_state=initial_state,
        evidence=evidence,
        drift_event=event,
        ground_truth=GroundTruth((), (), (), (), "", {}),
        task_constraints={
            "target_fidelity": 0.85,
            "candidate_backends": list(BACKENDS),
            "candidate_qubits": list(range(5)),
            "cost_budget": 10,
        },
        seed=seed,
    )
    invalidated = expected_invalidated_artifacts(provisional)
    invalidated_set = set(invalidated)
    preserved = tuple(
        item.evidence_id
        for item in evidence
        if item.evidence_id not in invalidated_set
    )
    truth = GroundTruth(
        invalidated_artifact_ids=invalidated,
        preserved_artifact_ids=preserved,
        acceptable_actions=actions,
        required_revalidation_actions=revalidation,
        claim_support_predicate=(
            "final claims may cite only evidence not invalidated by the event"
        ),
        acceptable_payload=payload,
        failure_reason=reason,
    )
    return Episode(
        **{
            **provisional.__dict__,
            "ground_truth": truth,
        }
    )


def generate_stage_a_episodes() -> list[Episode]:
    episodes = [
        _build_episode(task, template_index, relevance)
        for task in TaskType
        for template_index in range(10)
        for relevance in (DriftRelevance.RELATED, DriftRelevance.UNRELATED)
    ]
    validate_counterfactual_pairs(episodes)
    return episodes


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_stage_a_dataset(output_path: Path, manifest_path: Path) -> dict[str, Any]:
    episodes = generate_stage_a_episodes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(_canonical_json(item.to_dict()) for item in episodes) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": output_path.name,
        "episode_count": len(episodes),
        "pair_count": len({item.pair_id for item in episodes}),
        "sha256": digest,
        "source": "synthetic_stage_a",
        "severity_basis": "synthetic placeholder; calibrate from replay before main study",
        "counts_by_task": dict(
            sorted(Counter(item.task_type.value for item in episodes).items())
        ),
        "counts_by_relevance": dict(
            sorted(Counter(item.drift_event.relevance.value for item in episodes).items())
        ),
        "counts_by_severity": dict(
            sorted(Counter(item.drift_event.severity.value for item in episodes).items())
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "generate_stage_a_episodes",
    "write_stage_a_dataset",
]
