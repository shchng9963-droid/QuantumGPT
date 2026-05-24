"""Explicit state model for the QuantumGPT agent runtime.

The state layer tracks experimental artifacts separately from the LLM
message history. This makes drift, calibration expiry, and tool failures
visible to the controller instead of leaving them implicit in prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from typing import Any


class ArtifactType(str, Enum):
    """Artifact categories produced or consumed by agent tool calls."""

    BACKEND_SNAPSHOT = "backend_snapshot"
    QUBIT_PROPERTIES = "qubit_properties"
    COUPLING_MAP = "coupling_map"
    TRANSPILED_CIRCUIT = "transpiled_circuit"
    PREDICTED_FIDELITY = "predicted_fidelity"
    CIRCUIT_RESULT = "circuit_result"
    MITIGATION_RESULT = "mitigation_result"
    DRIFT_REPORT = "drift_report"
    LAB_EXPERIMENT_RESULT = "lab_experiment_result"
    FIT_RESULT = "fit_result"
    CALIBRATION_UPDATE = "calibration_update"
    SAFETY_VIOLATION = "safety_violation"


class ArtifactStatus(str, Enum):
    """Validity state for an artifact."""

    VALID = "valid"
    STALE = "stale"
    INVALID = "invalid"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class InvalidationReason(str, Enum):
    """Reason why an artifact can no longer be trusted."""

    BACKEND_DRIFT = "backend_drift"
    CALIBRATION_EXPIRED = "calibration_expired"
    TOOL_FAILURE = "tool_failure"
    SUPERSEDED_BY_NEW_SNAPSHOT = "superseded_by_new_snapshot"
    MANUAL_VETO = "manual_veto"
    SAFETY_VIOLATION = "safety_violation"


@dataclass
class Artifact:
    """A stateful piece of evidence created during an agent run."""

    artifact_id: str
    artifact_type: ArtifactType
    created_at_step: int
    depends_on: list[str] = field(default_factory=list)
    backend_snapshot_hash: str | None = None
    status: ArtifactStatus = ArtifactStatus.VALID
    invalidation_reason: InvalidationReason | None = None
    invalidation_detail: str | None = None
    payload_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-stable representation for traces and reports."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "created_at_step": self.created_at_step,
            "depends_on": list(self.depends_on),
            "backend_snapshot_hash": self.backend_snapshot_hash,
            "status": self.status.value,
            "invalidation_reason": self.invalidation_reason.value if self.invalidation_reason else None,
            "invalidation_detail": self.invalidation_detail,
            "payload_ref": self.payload_ref,
            "metadata": dict(self.metadata),
        }


@dataclass
class ArtifactRegistry:
    """Stores artifacts and manages dependency-based invalidation."""

    artifacts: dict[str, Artifact] = field(default_factory=dict)

    def add_artifact(
        self,
        artifact_type: ArtifactType,
        created_at_step: int,
        depends_on: list[str] | None = None,
        backend_snapshot_hash: str | None = None,
        payload_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> Artifact:
        artifact = Artifact(
            artifact_id=artifact_id or f"artifact-{uuid4().hex[:12]}",
            artifact_type=artifact_type,
            created_at_step=created_at_step,
            depends_on=list(depends_on or []),
            backend_snapshot_hash=backend_snapshot_hash,
            payload_ref=payload_ref,
            metadata=dict(metadata or {}),
        )
        if artifact.artifact_id in self.artifacts:
            raise ValueError(f"duplicate artifact_id: {artifact.artifact_id}")
        self.artifacts[artifact.artifact_id] = artifact
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        return self.artifacts.get(artifact_id)

    def by_type(self, artifact_type: ArtifactType) -> list[Artifact]:
        return [a for a in self.artifacts.values() if a.artifact_type is artifact_type]

    def by_status(self, status: ArtifactStatus) -> list[Artifact]:
        return [a for a in self.artifacts.values() if a.status is status]

    def invalidate_artifact(
        self,
        artifact_id: str,
        reason: InvalidationReason,
        detail: str | None = None,
        status: ArtifactStatus = ArtifactStatus.INVALID,
    ) -> Artifact:
        artifact = self.artifacts[artifact_id]
        artifact.status = status
        artifact.invalidation_reason = reason
        artifact.invalidation_detail = detail
        return artifact

    def invalidate_dependents(
        self,
        artifact_id: str,
        reason: InvalidationReason,
        detail: str | None = None,
        status: ArtifactStatus = ArtifactStatus.INVALID,
    ) -> list[Artifact]:
        """Invalidate all direct and transitive dependents in creation order."""
        invalidated: list[Artifact] = []
        seen: set[str] = set()

        def visit(parent_id: str) -> None:
            dependents = [
                artifact
                for artifact in self.artifacts.values()
                if parent_id in artifact.depends_on and artifact.artifact_id not in seen
            ]
            dependents.sort(key=lambda a: a.created_at_step)
            for artifact in dependents:
                seen.add(artifact.artifact_id)
                self.invalidate_artifact(
                    artifact.artifact_id,
                    reason=reason,
                    detail=detail,
                    status=status,
                )
                invalidated.append(artifact)
                visit(artifact.artifact_id)

        visit(artifact_id)
        return invalidated

    def to_dict(self) -> dict[str, Any]:
        return {artifact_id: artifact.to_dict() for artifact_id, artifact in self.artifacts.items()}

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for artifact in self.artifacts.values():
            key = artifact.artifact_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def count_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for artifact in self.artifacts.values():
            key = artifact.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts


@dataclass
class AgentState:
    """Explicit runtime state for one agent task."""

    task_id: str | None = None
    user_prompt: str = ""
    backend_name: str | None = None
    backend_snapshot_id: str | None = None
    current_step: int = 0
    budget_state: dict[str, Any] = field(default_factory=dict)
    memory_context: list[dict[str, Any]] = field(default_factory=list)
    drift_state: dict[str, Any] = field(default_factory=dict)
    artifacts: ArtifactRegistry = field(default_factory=ArtifactRegistry)
    observations: list[str] = field(default_factory=list)
    safety_status: dict[str, Any] = field(default_factory=dict)

    def summary_for_prompt(self, max_observations: int = 3) -> dict[str, Any]:
        """Return compact state for prompt injection or trace summaries."""
        return {
            "task_id": self.task_id,
            "backend_name": self.backend_name,
            "backend_snapshot_id": self.backend_snapshot_id,
            "current_step": self.current_step,
            "artifact_counts": self.artifacts.count_by_type(),
            "artifact_status_counts": self.artifacts.count_by_status(),
            "recent_observations": self.observations[-max_observations:],
            "budget_state": dict(self.budget_state),
            "drift_state": dict(self.drift_state),
            "safety_status": dict(self.safety_status),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return full JSON-stable state for trace export."""
        return {
            "task_id": self.task_id,
            "user_prompt": self.user_prompt,
            "backend_name": self.backend_name,
            "backend_snapshot_id": self.backend_snapshot_id,
            "current_step": self.current_step,
            "budget_state": dict(self.budget_state),
            "memory_context": list(self.memory_context),
            "drift_state": dict(self.drift_state),
            "artifacts": self.artifacts.to_dict(),
            "observations": list(self.observations),
            "safety_status": dict(self.safety_status),
        }
