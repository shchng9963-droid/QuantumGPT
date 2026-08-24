"""ActionGuard package v1.3 runtime wrapper.

The Guard session and two-repair budget are inherited byte-for-behavior from
v1.2.  Version 1.3 adds only treatment-neutral runtime terminal closure in the
public preflight state machine; no safety decision is changed here.
"""

from __future__ import annotations

from .action_guard_runtime_v1_2 import (
    SHARED_GUARD_BUDGET,
    GuardRuntimeSessionV1_2,
    GuardTraceEvent,
    SharedGuardBudget,
)
from .runtime_terminal_v1 import (
    RuntimeExecutionFailure,
    build_runtime_execution_failure,
)


ACTION_GUARD_RUNTIME_VERSION = "reliabilitybench-q/action-guard-runtime-1.3"


class GuardRuntimeSessionV1_3(GuardRuntimeSessionV1_2):
    """Version marker; safety checks and repair accounting remain v1.2."""


def close_runtime_failure_state(
    *,
    failure_reason: str,
    last_candidate_hash: str | None,
    last_guard_outcome: str | None,
    last_reason_code: str | None,
) -> RuntimeExecutionFailure:
    """Uniform closure for Guard and non-Guard fatal workflow outcomes."""
    return build_runtime_execution_failure(
        failure_reason=failure_reason,
        last_candidate_hash=last_candidate_hash,
        last_guard_outcome=last_guard_outcome,
        last_reason_code=last_reason_code,
    )


__all__ = [
    "ACTION_GUARD_RUNTIME_VERSION",
    "SHARED_GUARD_BUDGET",
    "GuardRuntimeSessionV1_3",
    "GuardTraceEvent",
    "SharedGuardBudget",
    "close_runtime_failure_state",
]
