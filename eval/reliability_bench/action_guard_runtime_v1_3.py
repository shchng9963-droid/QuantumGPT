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


ACTION_GUARD_RUNTIME_VERSION = "reliabilitybench-q/action-guard-runtime-1.3"


class GuardRuntimeSessionV1_3(GuardRuntimeSessionV1_2):
    """Version marker; safety checks and repair accounting remain v1.2."""


__all__ = [
    "ACTION_GUARD_RUNTIME_VERSION",
    "SHARED_GUARD_BUDGET",
    "GuardRuntimeSessionV1_3",
    "GuardTraceEvent",
    "SharedGuardBudget",
]
