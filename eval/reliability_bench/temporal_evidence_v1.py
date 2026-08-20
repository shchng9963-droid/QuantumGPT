"""Temporal Evidence Layer: the sole producer of post-drift validity states."""

from __future__ import annotations

from dataclasses import dataclass

from .action_guard_contract_v1 import (
    EvidenceValidity,
    LedgerEvidence,
    VisibleEvidenceLedger,
)


TEMPORAL_EVIDENCE_VERSION = "reliabilitybench-q/temporal-evidence-layer-1.0"


@dataclass(frozen=True)
class EvidenceDependency:
    evidence_id: str
    slots: tuple[str, ...]
    resources: tuple[str, ...]
    source: str


def dependency_scoped_validity(
    evidence: tuple[EvidenceDependency, ...],
    *,
    affected_resources: tuple[str, ...],
) -> VisibleEvidenceLedger:
    """Full/Temporal arm validity producer; never called by ActionGuard."""
    affected = set(affected_resources)
    return VisibleEvidenceLedger(
        tuple(
            LedgerEvidence(
                evidence_id=item.evidence_id,
                slots=item.slots,
                validity=(
                    EvidenceValidity.INVALID
                    if affected.intersection(item.resources)
                    else EvidenceValidity.VALID
                ),
                source=item.source,
            )
            for item in evidence
        )
    )


def ledger_only_visible_state(
    evidence: tuple[EvidenceDependency, ...],
) -> VisibleEvidenceLedger:
    """Ledger arm exposes unchecked state and performs no drift inference."""
    return VisibleEvidenceLedger(
        tuple(
            LedgerEvidence(
                evidence_id=item.evidence_id,
                slots=item.slots,
                validity=EvidenceValidity.UNKNOWN,
                source=item.source,
            )
            for item in evidence
        )
    )


__all__ = [
    "TEMPORAL_EVIDENCE_VERSION",
    "EvidenceDependency",
    "dependency_scoped_validity",
    "ledger_only_visible_state",
]
