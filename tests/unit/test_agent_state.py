"""Tests for explicit agent state and artifact invalidation."""

from agent.state import (
    AgentState,
    Artifact,
    ArtifactRegistry,
    ArtifactStatus,
    ArtifactType,
    InvalidationReason,
)


def test_registry_adds_and_queries_artifacts_by_type_and_status():
    registry = ArtifactRegistry()

    snapshot = registry.add_artifact(
        artifact_type=ArtifactType.BACKEND_SNAPSHOT,
        created_at_step=1,
        payload_ref="snapshot:abc",
        metadata={"backend": "FakeBrisbane"},
    )
    prediction = registry.add_artifact(
        artifact_type=ArtifactType.PREDICTED_FIDELITY,
        created_at_step=2,
        depends_on=[snapshot.artifact_id],
        backend_snapshot_hash="abc",
        payload_ref="prediction:001",
    )

    assert snapshot.artifact_id != prediction.artifact_id
    assert registry.get(snapshot.artifact_id) == snapshot
    assert registry.by_type(ArtifactType.BACKEND_SNAPSHOT) == [snapshot]
    assert registry.by_status(ArtifactStatus.VALID) == [snapshot, prediction]


def test_invalidate_artifact_records_reason_and_keeps_dependents_valid_by_default():
    registry = ArtifactRegistry()
    snapshot = registry.add_artifact(ArtifactType.BACKEND_SNAPSHOT, created_at_step=1)
    prediction = registry.add_artifact(
        ArtifactType.PREDICTED_FIDELITY,
        created_at_step=2,
        depends_on=[snapshot.artifact_id],
    )

    registry.invalidate_artifact(
        snapshot.artifact_id,
        reason=InvalidationReason.BACKEND_DRIFT,
        detail="T1 changed by 30%",
    )

    assert snapshot.status is ArtifactStatus.INVALID
    assert snapshot.invalidation_reason is InvalidationReason.BACKEND_DRIFT
    assert snapshot.invalidation_detail == "T1 changed by 30%"
    assert prediction.status is ArtifactStatus.VALID


def test_invalidate_dependents_recursively_marks_derived_artifacts_stale():
    registry = ArtifactRegistry()
    snapshot = registry.add_artifact(ArtifactType.BACKEND_SNAPSHOT, created_at_step=1)
    transpiled = registry.add_artifact(
        ArtifactType.TRANSPILED_CIRCUIT,
        created_at_step=2,
        depends_on=[snapshot.artifact_id],
    )
    prediction = registry.add_artifact(
        ArtifactType.PREDICTED_FIDELITY,
        created_at_step=3,
        depends_on=[transpiled.artifact_id],
    )
    unrelated = registry.add_artifact(ArtifactType.COUPLING_MAP, created_at_step=1)

    invalidated = registry.invalidate_dependents(
        snapshot.artifact_id,
        reason=InvalidationReason.BACKEND_DRIFT,
        status=ArtifactStatus.STALE,
    )

    assert [a.artifact_id for a in invalidated] == [
        transpiled.artifact_id,
        prediction.artifact_id,
    ]
    assert transpiled.status is ArtifactStatus.STALE
    assert prediction.status is ArtifactStatus.STALE
    assert unrelated.status is ArtifactStatus.VALID


def test_agent_state_summary_is_compact_and_counts_artifacts():
    state = AgentState(
        task_id="T1-01",
        user_prompt="Run GHZ with fidelity target 0.9",
        backend_name="FakeBrisbane",
    )
    snapshot = state.artifacts.add_artifact(ArtifactType.BACKEND_SNAPSHOT, created_at_step=1)
    state.artifacts.add_artifact(
        ArtifactType.PREDICTED_FIDELITY,
        created_at_step=2,
        depends_on=[snapshot.artifact_id],
    )
    state.artifacts.invalidate_dependents(
        snapshot.artifact_id,
        reason=InvalidationReason.BACKEND_DRIFT,
        status=ArtifactStatus.STALE,
    )
    state.current_step = 3
    state.observations.append("drift detected")

    summary = state.summary_for_prompt()

    assert summary["task_id"] == "T1-01"
    assert summary["backend_name"] == "FakeBrisbane"
    assert summary["current_step"] == 3
    assert summary["artifact_counts"] == {"backend_snapshot": 1, "predicted_fidelity": 1}
    assert summary["artifact_status_counts"] == {"valid": 1, "stale": 1}
    assert summary["recent_observations"] == ["drift detected"]


def test_artifact_to_dict_uses_stable_string_values():
    artifact = Artifact(
        artifact_id="artifact-001",
        artifact_type=ArtifactType.CIRCUIT_RESULT,
        created_at_step=4,
        depends_on=["artifact-000"],
        backend_snapshot_hash="hash-123",
        status=ArtifactStatus.VALID,
        payload_ref="result:001",
        metadata={"fidelity": 0.91},
    )

    assert artifact.to_dict() == {
        "artifact_id": "artifact-001",
        "artifact_type": "circuit_result",
        "created_at_step": 4,
        "depends_on": ["artifact-000"],
        "backend_snapshot_hash": "hash-123",
        "status": "valid",
        "invalidation_reason": None,
        "invalidation_detail": None,
        "payload_ref": "result:001",
        "metadata": {"fidelity": 0.91},
    }
