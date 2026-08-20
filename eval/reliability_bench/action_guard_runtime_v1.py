"""Shared, budget-neutral repair loop accounting for Guard-enabled arms."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Sequence

from .action_guard_contract_v1 import (
    AcceptedToolCall,
    CandidateFinalAction,
    CandidateToolAction,
    GuardOutcome,
    GuardResult,
    PublicTaskContract,
    VisibleEvidenceLedger,
)
from .action_guard_v1 import ActionGuard


ACTION_GUARD_RUNTIME_VERSION = "reliabilitybench-q/action-guard-runtime-1.0"


@dataclass(frozen=True)
class SharedGuardBudget:
    max_repair_attempts: int = 2
    max_total_llm_turns: int = 12
    max_tool_calls: int = 10


SHARED_GUARD_BUDGET = SharedGuardBudget()


@dataclass(frozen=True)
class GuardTraceEvent:
    check_index: int
    candidate_sha256: str
    outcome: str
    reason_code: str | None
    violated_public_constraint: str | None
    repair_allowed: bool
    repair_attempt_index: int | None
    llm_turn_count: int
    tool_call_count: int
    check_latency_seconds: float
    feedback: dict[str, str] | None


class GuardRuntimeSession:
    def __init__(
        self,
        *,
        guard: ActionGuard | None = None,
        budget: SharedGuardBudget = SHARED_GUARD_BUDGET,
    ) -> None:
        self.guard = guard or ActionGuard()
        self.budget = budget
        self.events: list[GuardTraceEvent] = []
        self.repair_attempts = 0

    def check_tool(
        self,
        candidate: CandidateToolAction,
        *,
        task: PublicTaskContract,
        ledger: VisibleEvidenceLedger,
        llm_turn_count: int,
        tool_call_count: int,
    ) -> GuardResult:
        started = time.perf_counter()
        result = self.guard.check_tool_action(candidate, task=task, ledger=ledger)
        self._record(
            result,
            llm_turn_count=llm_turn_count,
            tool_call_count=tool_call_count,
            latency=time.perf_counter() - started,
        )
        return result

    def check_final(
        self,
        candidate: CandidateFinalAction,
        *,
        task: PublicTaskContract,
        ledger: VisibleEvidenceLedger,
        accepted_tool_calls: Sequence[AcceptedToolCall],
        llm_turn_count: int,
        tool_call_count: int,
    ) -> GuardResult:
        started = time.perf_counter()
        result = self.guard.check_final_action(
            candidate,
            task=task,
            ledger=ledger,
            accepted_tool_calls=accepted_tool_calls,
        )
        self._record(
            result,
            llm_turn_count=llm_turn_count,
            tool_call_count=tool_call_count,
            latency=time.perf_counter() - started,
        )
        return result

    def _record(
        self,
        result: GuardResult,
        *,
        llm_turn_count: int,
        tool_call_count: int,
        latency: float,
    ) -> None:
        intercepted = result.outcome is not GuardOutcome.ALLOW
        repair_allowed = bool(
            intercepted
            and self.repair_attempts < self.budget.max_repair_attempts
            and llm_turn_count < self.budget.max_total_llm_turns
            and tool_call_count <= self.budget.max_tool_calls
        )
        repair_index = None
        feedback = None
        if repair_allowed:
            self.repair_attempts += 1
            repair_index = self.repair_attempts
            feedback = {
                "guard_outcome": result.outcome.value,
                "reason_code": result.reason_code.value,
                "violated_public_constraint": result.violated_public_constraint,
            }
        self.events.append(
            GuardTraceEvent(
                check_index=len(self.events) + 1,
                candidate_sha256=result.candidate_sha256,
                outcome=result.outcome.value,
                reason_code=(result.reason_code.value if result.reason_code else None),
                violated_public_constraint=result.violated_public_constraint,
                repair_allowed=repair_allowed,
                repair_attempt_index=repair_index,
                llm_turn_count=llm_turn_count,
                tool_call_count=tool_call_count,
                check_latency_seconds=latency,
                feedback=feedback,
            )
        )

    def overhead(self) -> dict[str, float | int]:
        return {
            "guard_check_count": len(self.events),
            "guard_intervention_count": sum(
                item.outcome != GuardOutcome.ALLOW.value for item in self.events
            ),
            "repair_attempt_count": self.repair_attempts,
            "guard_latency_seconds": sum(item.check_latency_seconds for item in self.events),
        }

    def trace_json(self) -> str:
        return json.dumps(
            [asdict(item) for item in self.events],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


__all__ = [
    "ACTION_GUARD_RUNTIME_VERSION",
    "SHARED_GUARD_BUDGET",
    "GuardRuntimeSession",
    "GuardTraceEvent",
    "SharedGuardBudget",
]
