"""ActionGuard v1.1: a shield over controller belief B_t(e), not V*_t(e)."""

from __future__ import annotations

from typing import Sequence

from .action_guard_contract_v1 import EvidenceValidity, LedgerEvidence, VisibleEvidenceLedger
from .action_guard_contract_v1_1 import (
    AcceptedToolCall,
    BeliefValidity,
    CandidateFinalAction,
    CandidateToolAction,
    ControllerBeliefLedger,
    GuardResult,
    PublicTaskContract,
)
from .action_guard_v1 import ActionGuard as _ActionGuardV1


ACTION_GUARD_VERSION = "reliabilitybench-q/action-guard-1.1-frozen-candidate"


def _belief_as_legacy_ledger(ledger: ControllerBeliefLedger) -> VisibleEvidenceLedger:
    """Mechanical type bridge; it performs no drift or objective-validity inference."""
    mapping = {
        BeliefValidity.BELIEVED_VALID: EvidenceValidity.VALID,
        BeliefValidity.BELIEVED_INVALID: EvidenceValidity.INVALID,
        BeliefValidity.UNKNOWN: EvidenceValidity.UNKNOWN,
    }
    return VisibleEvidenceLedger(
        tuple(
            LedgerEvidence(
                evidence_id=item.evidence_id,
                slots=item.slots,
                validity=mapping[item.belief_validity],
                source=item.source,
                produced_by_tool_call_id=item.produced_by_tool_call_id,
            )
            for item in ledger.records
        )
    )


class ActionGuard:
    """Pure decision procedure whose validity input is exclusively B_t(e)."""

    def __init__(self) -> None:
        self._syntax_and_support_guard = _ActionGuardV1()

    def check_tool_action(
        self,
        candidate: CandidateToolAction,
        *,
        task: PublicTaskContract,
        ledger: ControllerBeliefLedger,
    ) -> GuardResult:
        return self._syntax_and_support_guard.check_tool_action(
            candidate,
            task=task,
            ledger=_belief_as_legacy_ledger(ledger),
        )

    def check_final_action(
        self,
        candidate: CandidateFinalAction,
        *,
        task: PublicTaskContract,
        ledger: ControllerBeliefLedger,
        accepted_tool_calls: Sequence[AcceptedToolCall],
    ) -> GuardResult:
        return self._syntax_and_support_guard.check_final_action(
            candidate,
            task=task,
            ledger=_belief_as_legacy_ledger(ledger),
            accepted_tool_calls=accepted_tool_calls,
        )


__all__ = ["ACTION_GUARD_VERSION", "ActionGuard"]
