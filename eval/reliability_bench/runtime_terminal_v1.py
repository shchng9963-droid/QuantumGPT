"""Treatment-neutral runtime-generated terminal states.

Runtime failures are authoritative workflow outcomes, not Agent decisions.  The
runtime may create this object only after the ordinary Agent path can no longer
close normally.  It deliberately contains no task decision or evidence claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


RUNTIME_TERMINAL_SCHEMA_VERSION = "reliabilitybench-q/runtime-terminal-1.0"
RUNTIME_TERMINAL_VERSION = "reliabilitybench-q/runtime-terminal-builder-1.0"


@dataclass(frozen=True)
class RuntimeExecutionFailure:
    schema_version: str
    completion_status: str
    failure_reason: str
    decision: None
    runtime_generated: bool
    last_candidate_hash: str | None
    last_guard_outcome: str | None
    last_reason_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be null or a non-empty string")
    return value


def build_runtime_execution_failure(
    *,
    failure_reason: str,
    last_candidate_hash: str | None,
    last_guard_outcome: str | None,
    last_reason_code: str | None,
) -> RuntimeExecutionFailure:
    """Build a non-semantic terminal without inventing an Agent decision."""
    return parse_runtime_execution_failure(
        {
            "schema_version": RUNTIME_TERMINAL_SCHEMA_VERSION,
            "completion_status": "execution_failure",
            "failure_reason": failure_reason,
            "decision": None,
            "runtime_generated": True,
            "last_candidate_hash": last_candidate_hash,
            "last_guard_outcome": last_guard_outcome,
            "last_reason_code": last_reason_code,
        }
    )


def parse_runtime_execution_failure(
    value: Mapping[str, Any],
) -> RuntimeExecutionFailure:
    if not isinstance(value, Mapping):
        raise ValueError("runtime terminal must be an object")
    expected = {
        "schema_version",
        "completion_status",
        "failure_reason",
        "decision",
        "runtime_generated",
        "last_candidate_hash",
        "last_guard_outcome",
        "last_reason_code",
    }
    if set(value) != expected:
        raise ValueError("runtime terminal keys differ from the frozen schema")
    if value["schema_version"] != RUNTIME_TERMINAL_SCHEMA_VERSION:
        raise ValueError("unsupported runtime terminal schema version")
    if value["completion_status"] != "execution_failure":
        raise ValueError("runtime terminal must be execution_failure")
    if value["decision"] is not None:
        raise ValueError("runtime execution_failure cannot contain a decision")
    if value["runtime_generated"] is not True:
        raise ValueError("runtime execution_failure must be runtime generated")
    failure_reason = _optional_string(value["failure_reason"], field="failure_reason")
    if failure_reason is None:
        raise ValueError("failure_reason is required")
    candidate_hash = _optional_string(
        value["last_candidate_hash"], field="last_candidate_hash"
    )
    if candidate_hash is not None and (
        len(candidate_hash) != 64
        or any(character not in "0123456789abcdef" for character in candidate_hash)
    ):
        raise ValueError("last_candidate_hash must be a lowercase SHA-256")
    outcome = _optional_string(value["last_guard_outcome"], field="last_guard_outcome")
    if outcome not in {None, "BLOCK", "REPAIR_REQUEST"}:
        raise ValueError("last_guard_outcome is not a blocked Guard outcome")
    reason_code = _optional_string(value["last_reason_code"], field="last_reason_code")
    if (outcome is None) != (reason_code is None):
        raise ValueError("Guard outcome and reason code must be jointly present or absent")
    return RuntimeExecutionFailure(
        schema_version=RUNTIME_TERMINAL_SCHEMA_VERSION,
        completion_status="execution_failure",
        failure_reason=failure_reason,
        decision=None,
        runtime_generated=True,
        last_candidate_hash=candidate_hash,
        last_guard_outcome=outcome,
        last_reason_code=reason_code,
    )


__all__ = [
    "RUNTIME_TERMINAL_SCHEMA_VERSION",
    "RUNTIME_TERMINAL_VERSION",
    "RuntimeExecutionFailure",
    "build_runtime_execution_failure",
    "parse_runtime_execution_failure",
]
