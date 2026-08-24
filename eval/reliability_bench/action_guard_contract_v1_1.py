"""Controller-belief contract for ActionGuard v1.1.

The ledger below represents B_t(e), never the evaluator's objective V*_t(e).
This module intentionally has no evaluator, Oracle, ground-truth, drift, or
recovery-planner dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from .action_guard_contract_v1 import (
    AcceptedToolCall,
    CandidateFinalAction,
    CandidateToolAction,
    GuardOutcome,
    GuardResult,
    PUBLIC_REQUIRED_SLOTS,
    PUBLIC_TOOL_SCHEMAS,
    ParameterRule,
    PublicTaskContract,
    ReasonCode,
    candidate_sha256,
)


ACTION_GUARD_CONTRACT_VERSION = "reliabilitybench-q/action-guard-contract-1.1"

PUBLIC_EVIDENCE_TYPE_SLOTS = MappingProxyType(
    {
        "backend_ranking": ("backend_ranking",),
        "qubit_mapping": ("qubit_properties",),
        "transpiled_circuit": ("compilation",),
        "circuit_result": ("circuit_result",),
        "mitigation_estimate": ("mitigation_estimate",),
        "feasibility_assessment": ("backend_capacity",),
        "task_context": ("task_context",),
    }
)

PUBLIC_TOOL_EVIDENCE_SLOTS = MappingProxyType(
    {
        "get_backend_health": ("backend_health", "backend_capacity"),
        "compare_backends": ("backend_ranking",),
        "get_qubit_properties": ("qubit_properties",),
        "get_coupling_map": ("coupling_map",),
        "transpile_circuit": ("compilation",),
        "run_circuit": ("circuit_result",),
        "apply_mitigation": ("mitigation_estimate",),
        "refresh_task_context": ("task_context",),
    }
)


class BeliefValidity(str, Enum):
    """Validity state visible to the controller, B_t(e)."""

    BELIEVED_VALID = "believed_valid"
    BELIEVED_INVALID = "believed_invalid"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ControllerEvidenceBelief:
    evidence_id: str
    slots: tuple[str, ...]
    belief_validity: BeliefValidity
    source: str
    produced_by_tool_call_id: str | None = None


@dataclass(frozen=True)
class ControllerBeliefLedger:
    """The complete evidence-validity belief visible to one controller."""

    records: tuple[ControllerEvidenceBelief, ...]

    def __post_init__(self) -> None:
        ids = [item.evidence_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("belief ledger evidence IDs must be unique")

    def by_id(self) -> dict[str, ControllerEvidenceBelief]:
        return {item.evidence_id: item for item in self.records}


__all__ = [
    "ACTION_GUARD_CONTRACT_VERSION",
    "AcceptedToolCall",
    "BeliefValidity",
    "CandidateFinalAction",
    "CandidateToolAction",
    "ControllerBeliefLedger",
    "ControllerEvidenceBelief",
    "GuardOutcome",
    "GuardResult",
    "PUBLIC_EVIDENCE_TYPE_SLOTS",
    "PUBLIC_REQUIRED_SLOTS",
    "PUBLIC_TOOL_EVIDENCE_SLOTS",
    "PUBLIC_TOOL_SCHEMAS",
    "ParameterRule",
    "PublicTaskContract",
    "ReasonCode",
    "candidate_sha256",
]
