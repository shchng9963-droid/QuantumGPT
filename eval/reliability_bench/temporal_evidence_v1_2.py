"""Controller-belief producer B_t(e), independent of objective V*_t(e)."""

from __future__ import annotations

from dataclasses import dataclass

from .action_guard_contract_v1_2 import (
    BeliefValidity,
    ControllerBeliefLedger,
    ControllerEvidenceBelief,
)


TEMPORAL_EVIDENCE_VERSION = "reliabilitybench-q/temporal-evidence-layer-1.2"


@dataclass(frozen=True)
class EvidenceBeliefDependency:
    evidence_id: str
    slots: tuple[str, ...]
    resources: tuple[str, ...]
    source: str
    last_known_belief: BeliefValidity = BeliefValidity.BELIEVED_VALID


def ledger_only_belief_state(
    evidence: tuple[EvidenceBeliefDependency, ...],
) -> ControllerBeliefLedger:
    """No drift update: preserve each pre-drift last-known belief."""
    return ControllerBeliefLedger(
        tuple(
            ControllerEvidenceBelief(
                item.evidence_id,
                item.slots,
                item.last_known_belief,
                item.source,
            )
            for item in evidence
        )
    )


def dependency_scoped_belief_update(
    evidence: tuple[EvidenceBeliefDependency, ...],
    *,
    changed_features: tuple[str, ...],
) -> ControllerBeliefLedger:
    """Full arm update: invalidate only beliefs whose dependencies are affected."""
    aliases = {
        "avg_2q_error": "avg_2q_error",
        "qubit_readout_error": "readout_error",
        "coupling_map": "topology.",
        "avg_readout_error": "avg_readout_error",
        "backend_capacity": "capacity",
        "task_context": "context.",
    }

    def impacted(item: EvidenceBeliefDependency) -> bool:
        return any(
            aliases.get(feature, feature) in resource
            for feature in changed_features
            for resource in item.resources
        )

    return ControllerBeliefLedger(
        tuple(
            ControllerEvidenceBelief(
                item.evidence_id,
                item.slots,
                (
                    BeliefValidity.BELIEVED_INVALID
                    if impacted(item)
                    else item.last_known_belief
                ),
                item.source,
            )
            for item in evidence
        )
    )


def register_revalidated_belief(
    ledger: ControllerBeliefLedger,
    *,
    evidence_id: str,
    slots: tuple[str, ...],
    source: str,
    tool_call_id: str,
) -> ControllerBeliefLedger:
    """Append successful current evidence as believed valid without changing old beliefs."""
    return ControllerBeliefLedger(
        (
            *ledger.records,
            ControllerEvidenceBelief(
                evidence_id=evidence_id,
                slots=slots,
                belief_validity=BeliefValidity.BELIEVED_VALID,
                source=source,
                produced_by_tool_call_id=tool_call_id,
            ),
        )
    )


__all__ = [
    "TEMPORAL_EVIDENCE_VERSION",
    "EvidenceBeliefDependency",
    "dependency_scoped_belief_update",
    "ledger_only_belief_state",
    "register_revalidated_belief",
]
