"""Public contracts consumed by ActionGuard, independent of evaluation code.

This module intentionally does not import Measurement v2, task predicates,
ground truth, Oracle data, the Temporal Evidence producer, or a recovery planner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .terminal_schema_v2 import TerminalDecision


ACTION_GUARD_CONTRACT_VERSION = "reliabilitybench-q/action-guard-contract-1.0"


class GuardOutcome(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REPAIR_REQUEST = "REPAIR_REQUEST"


class ReasonCode(str, Enum):
    INVALID_SCHEMA = "invalid_schema"
    UNKNOWN_ENTITY = "unknown_entity"
    INVALID_REFERENCE = "invalid_reference"
    MISSING_REQUIRED_SUPPORT = "missing_required_support"
    STALE_CITED_EVIDENCE = "stale_cited_evidence"
    UNVERIFIED_REVALIDATION = "unverified_revalidation"
    PREMATURE_FINALIZATION = "premature_finalization"


class EvidenceValidity(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


# Public decision-support contract.  This is deliberately duplicated rather
# than imported from the evaluator, so method and measurement implementations
# remain code-level independent.
PUBLIC_REQUIRED_SLOTS = MappingProxyType(
    {
        "backend_selection": ("backend_ranking",),
        "qubit_mapping": ("qubit_properties",),
        "transpilation": ("compilation",),
        "fidelity_claim": ("circuit_result",),
        "mitigation_decision": ("mitigation_estimate",),
        "unreachable_target": ("backend_capacity",),
    }
)


@dataclass(frozen=True)
class ParameterRule:
    kind: str
    entity_domain: str | None = None
    non_empty: bool = True


PUBLIC_TOOL_SCHEMAS = MappingProxyType(
    {
        "get_backend_health": MappingProxyType(
            {"backend_id": ParameterRule("string", "backend")}
        ),
        "compare_backends": MappingProxyType(
            {"backend_ids": ParameterRule("string_list", "backend")}
        ),
        "get_qubit_properties": MappingProxyType(
            {
                "backend_id": ParameterRule("string", "backend"),
                "qubit_ids": ParameterRule("integer_list", "qubit"),
            }
        ),
        "get_coupling_map": MappingProxyType(
            {"backend_id": ParameterRule("string", "backend")}
        ),
        "transpile_circuit": MappingProxyType(
            {
                "circuit_id": ParameterRule("string", "circuit"),
                "backend_id": ParameterRule("string", "backend"),
            }
        ),
        "run_circuit": MappingProxyType(
            {
                "circuit_id": ParameterRule("string", "circuit"),
                "backend_id": ParameterRule("string", "backend"),
            }
        ),
        "apply_mitigation": MappingProxyType(
            {"mitigation_policy": ParameterRule("string", "mitigation_policy")}
        ),
        "refresh_task_context": MappingProxyType(
            {"task_id": ParameterRule("string", "task")}
        ),
    }
)


@dataclass(frozen=True)
class PublicTaskContract:
    task_id: str
    task_type: str
    backend_ids: tuple[str, ...]
    qubit_ids: tuple[int, ...]
    circuit_ids: tuple[str, ...]
    snapshot_ids: tuple[str, ...]
    mitigation_policy_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task_type not in PUBLIC_REQUIRED_SLOTS:
            raise ValueError("unknown public task type")
        for field_name in (
            "backend_ids",
            "qubit_ids",
            "circuit_ids",
            "snapshot_ids",
            "mitigation_policy_ids",
        ):
            value = getattr(self, field_name)
            if len(value) != len(set(value)):
                raise ValueError(f"{field_name} must contain unique entities")

    @property
    def required_slots(self) -> tuple[str, ...]:
        return PUBLIC_REQUIRED_SLOTS[self.task_type]

    def entity_domain(self, name: str) -> set[Any]:
        domains = {
            "backend": self.backend_ids,
            "qubit": self.qubit_ids,
            "circuit": self.circuit_ids,
            "snapshot": self.snapshot_ids,
            "mitigation_policy": self.mitigation_policy_ids,
            "task": (self.task_id,),
        }
        if name not in domains:
            raise ValueError("unknown public entity domain")
        return set(domains[name])


@dataclass(frozen=True)
class LedgerEvidence:
    evidence_id: str
    slots: tuple[str, ...]
    validity: EvidenceValidity
    source: str
    produced_by_tool_call_id: str | None = None


@dataclass(frozen=True)
class VisibleEvidenceLedger:
    records: tuple[LedgerEvidence, ...]

    def __post_init__(self) -> None:
        ids = [item.evidence_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("ledger evidence IDs must be unique")

    def by_id(self) -> dict[str, LedgerEvidence]:
        return {item.evidence_id: item for item in self.records}


@dataclass(frozen=True)
class AcceptedToolCall:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    successful: bool
    output_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateToolAction:
    tool_name: str
    arguments: dict[str, Any]
    trigger_evidence_id: str | None = None


@dataclass(frozen=True)
class CandidateFinalAction:
    terminal: TerminalDecision


@dataclass(frozen=True)
class GuardResult:
    outcome: GuardOutcome
    reason_code: ReasonCode | None
    violated_public_constraint: str | None
    candidate_sha256: str

    def __post_init__(self) -> None:
        if self.outcome is GuardOutcome.ALLOW:
            if self.reason_code is not None or self.violated_public_constraint is not None:
                raise ValueError("ALLOW must not include an error reason")
        elif self.reason_code is None or not self.violated_public_constraint:
            raise ValueError("blocked outcomes require an abstract public reason")


def candidate_sha256(value: CandidateToolAction | CandidateFinalAction) -> str:
    if isinstance(value, CandidateToolAction):
        payload: Mapping[str, Any] = {
            "type": "tool",
            "tool_name": value.tool_name,
            "arguments": value.arguments,
            "trigger_evidence_id": value.trigger_evidence_id,
        }
    else:
        payload = {"type": "final", "terminal": asdict(value.terminal)}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ACTION_GUARD_CONTRACT_VERSION",
    "AcceptedToolCall",
    "CandidateFinalAction",
    "CandidateToolAction",
    "EvidenceValidity",
    "GuardOutcome",
    "GuardResult",
    "LedgerEvidence",
    "PUBLIC_REQUIRED_SLOTS",
    "PUBLIC_TOOL_SCHEMAS",
    "ParameterRule",
    "PublicTaskContract",
    "ReasonCode",
    "VisibleEvidenceLedger",
    "candidate_sha256",
]
