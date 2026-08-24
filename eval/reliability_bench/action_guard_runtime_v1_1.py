"""Budget-neutral repair runtime for ActionGuard v1.1."""

from __future__ import annotations

from .action_guard_runtime_v1 import (
    SHARED_GUARD_BUDGET,
    GuardRuntimeSession,
    GuardTraceEvent,
    SharedGuardBudget,
)
from .action_guard_v1_1 import ActionGuard


ACTION_GUARD_RUNTIME_VERSION = "reliabilitybench-q/action-guard-runtime-1.1"


class GuardRuntimeSessionV1_1(GuardRuntimeSession):
    def __init__(
        self,
        *,
        guard: ActionGuard | None = None,
        budget: SharedGuardBudget = SHARED_GUARD_BUDGET,
    ) -> None:
        super().__init__(guard=guard or ActionGuard(), budget=budget)


__all__ = [
    "ACTION_GUARD_RUNTIME_VERSION",
    "SHARED_GUARD_BUDGET",
    "GuardRuntimeSessionV1_1",
    "GuardTraceEvent",
    "SharedGuardBudget",
]
