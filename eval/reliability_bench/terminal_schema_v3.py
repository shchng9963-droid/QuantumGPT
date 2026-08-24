"""Shared terminal-decision v3 schema for all ReliabilityBench-Q controllers.

This module contains no controller, guard, oracle, or evaluator policy.  It only
parses the common public output contract and preserves declarations separately
from authoritative execution traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


TERMINAL_SCHEMA_VERSION = "reliabilitybench-q/terminal-decision-3.0"

TASK_PAYLOAD_KEYS: dict[str, set[str]] = {
    "backend_selection": {"backend_id"},
    "qubit_mapping": {"qubit_ids"},
    "transpilation": {"compilation_snapshot_id"},
    "fidelity_claim": {"claimed_success"},
    "mitigation_decision": {"mitigation_policy", "intervention_requested"},
    "unreachable_target": {"declared_unreachable"},
}

TASK_ACTIONS: dict[str, set[str]] = {
    "backend_selection": {"select_backend"},
    "qubit_mapping": {"select_qubits"},
    "transpilation": {"retranspile", "reuse_compilation"},
    "fidelity_claim": {"rerun_circuit", "reuse_valid_result"},
    "mitigation_decision": {"reassess_mitigation", "keep_mitigation"},
    "unreachable_target": {"declare_unreachable", "continue_execution"},
}


class CompletionStatus(str, Enum):
    """Whether the controller produced an answer, not whether the target is feasible."""

    ANSWERED = "answered"
    ABSTAIN = "abstain"
    EXECUTION_FAILURE = "execution_failure"


@dataclass(frozen=True)
class RevalidationDeclaration:
    tool_call_id: str
    target_evidence_ids: tuple[str, ...]
    output_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class TerminalDecision:
    schema_version: str
    completion_status: CompletionStatus
    task_type: str
    task_action: str
    payload: dict[str, Any]
    supporting_evidence_ids: tuple[str, ...]
    revalidation_actions: tuple[RevalidationDeclaration, ...]
    failure_code: str | None
    failure_reason: str | None

    @property
    def answered(self) -> bool:
        return self.completion_status is CompletionStatus.ANSWERED


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, location: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _string_tuple(value: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{location} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{location} must not contain duplicates")
    return tuple(value)


def _validate_task_payload(
    *, task_type: str, task_action: str, payload: Mapping[str, Any]
) -> None:
    if task_type not in TASK_PAYLOAD_KEYS:
        raise ValueError("decision.task_type is not a registered task")
    if task_action not in TASK_ACTIONS[task_type]:
        raise ValueError("decision.task_action is invalid for the task type")
    _require_exact_keys(
        payload, TASK_PAYLOAD_KEYS[task_type], location="decision.payload"
    )
    if task_type == "backend_selection":
        if not isinstance(payload["backend_id"], str) or not payload["backend_id"]:
            raise ValueError("backend_id must be a non-empty string")
    elif task_type == "qubit_mapping":
        qubits = payload["qubit_ids"]
        if (
            not isinstance(qubits, list)
            or any(
                not isinstance(item, int) or isinstance(item, bool)
                for item in qubits
            )
            or len(qubits) != len(set(qubits))
        ):
            raise ValueError("qubit_ids must be a unique integer list")
    elif task_type == "transpilation":
        if (
            not isinstance(payload["compilation_snapshot_id"], str)
            or not payload["compilation_snapshot_id"]
        ):
            raise ValueError("compilation_snapshot_id must be a non-empty string")
    elif task_type == "fidelity_claim":
        if not isinstance(payload["claimed_success"], bool):
            raise ValueError("claimed_success must be boolean")
    elif task_type == "mitigation_decision":
        if (
            not isinstance(payload["mitigation_policy"], str)
            or not payload["mitigation_policy"]
        ):
            raise ValueError("mitigation_policy must be a non-empty string")
        if not isinstance(payload["intervention_requested"], bool):
            raise ValueError("intervention_requested must be boolean")
    elif not isinstance(payload["declared_unreachable"], bool):
        raise ValueError("declared_unreachable must be boolean")


def parse_terminal_decision(value: Mapping[str, Any]) -> TerminalDecision:
    """Parse one strict terminal object shared by every experimental arm."""
    if not isinstance(value, Mapping):
        raise ValueError("terminal decision must be an object")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "completion_status",
            "decision",
            "supporting_evidence_ids",
            "revalidation_actions",
            "failure",
        },
        location="terminal",
    )
    if value["schema_version"] != TERMINAL_SCHEMA_VERSION:
        raise ValueError("unsupported terminal schema version")
    try:
        completion_status = CompletionStatus(value["completion_status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid completion_status") from exc

    decision = value["decision"]
    if not isinstance(decision, Mapping):
        raise ValueError("decision must be an object")
    _require_exact_keys(
        decision,
        {"task_type", "task_action", "payload"},
        location="decision",
    )
    task_type = decision["task_type"]
    task_action = decision["task_action"]
    payload = decision["payload"]
    if not isinstance(task_type, str) or not task_type:
        raise ValueError("decision.task_type must be a non-empty string")
    if not isinstance(task_action, str) or not task_action:
        raise ValueError("decision.task_action must be a non-empty string")
    if not isinstance(payload, dict):
        raise ValueError("decision.payload must be an object")
    if task_type not in TASK_PAYLOAD_KEYS:
        raise ValueError("decision.task_type is not a registered task")
    if completion_status is CompletionStatus.ABSTAIN and task_action != "abstain":
        raise ValueError("abstain completion_status requires task_action=abstain")
    if completion_status is CompletionStatus.EXECUTION_FAILURE and task_action != "execution_failure":
        raise ValueError(
            "execution_failure completion_status requires task_action=execution_failure"
        )
    if completion_status is CompletionStatus.ANSWERED and task_action in {
        "abstain",
        "execution_failure",
    }:
        raise ValueError("answered completion_status requires a registered task action")
    if completion_status in {
        CompletionStatus.ABSTAIN,
        CompletionStatus.EXECUTION_FAILURE,
    }:
        _require_exact_keys(payload, {"reason"}, location="decision.payload")
        if (
            not isinstance(payload["reason"], str)
            or not payload["reason"].strip()
        ):
            raise ValueError("terminal reason must be a non-empty string")
    else:
        _validate_task_payload(
            task_type=task_type, task_action=task_action, payload=payload
        )

    declarations = value["revalidation_actions"]
    if not isinstance(declarations, list):
        raise ValueError("revalidation_actions must be a list")
    parsed_declarations: list[RevalidationDeclaration] = []
    seen_call_ids: set[str] = set()
    for index, item in enumerate(declarations):
        if not isinstance(item, Mapping):
            raise ValueError(f"revalidation_actions[{index}] must be an object")
        _require_exact_keys(
            item,
            {"tool_call_id", "target_evidence_ids", "output_evidence_ids"},
            location=f"revalidation_actions[{index}]",
        )
        call_id = item["tool_call_id"]
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool_call_id must be a non-empty string")
        if call_id in seen_call_ids:
            raise ValueError("revalidation_actions must not duplicate tool_call_id")
        seen_call_ids.add(call_id)
        parsed_declarations.append(
            RevalidationDeclaration(
                tool_call_id=call_id,
                target_evidence_ids=_string_tuple(
                    item["target_evidence_ids"],
                    location=f"revalidation_actions[{index}].target_evidence_ids",
                ),
                output_evidence_ids=_string_tuple(
                    item["output_evidence_ids"],
                    location=f"revalidation_actions[{index}].output_evidence_ids",
                ),
            )
        )

    failure = value["failure"]
    if not isinstance(failure, Mapping):
        raise ValueError("failure must be an object")
    _require_exact_keys(failure, {"code", "reason"}, location="failure")
    for field in ("code", "reason"):
        if failure[field] is not None and not isinstance(failure[field], str):
            raise ValueError(f"failure.{field} must be null or a string")
    if completion_status is CompletionStatus.ANSWERED and any(
        failure[field] is not None for field in ("code", "reason")
    ):
        raise ValueError("answered completion_status cannot contain a failure description")
    if completion_status is not CompletionStatus.ANSWERED and not str(
        failure["code"] or ""
    ).strip():
        raise ValueError("abstain and execution_failure require failure.code")

    return TerminalDecision(
        schema_version=TERMINAL_SCHEMA_VERSION,
        completion_status=completion_status,
        task_type=task_type,
        task_action=task_action,
        payload=dict(payload),
        supporting_evidence_ids=_string_tuple(
            value["supporting_evidence_ids"], location="supporting_evidence_ids"
        ),
        revalidation_actions=tuple(parsed_declarations),
        failure_code=failure["code"],
        failure_reason=failure["reason"],
    )


__all__ = [
    "TASK_ACTIONS",
    "TASK_PAYLOAD_KEYS",
    "TERMINAL_SCHEMA_VERSION",
    "RevalidationDeclaration",
    "TerminalDecision",
    "CompletionStatus",
    "parse_terminal_decision",
]
