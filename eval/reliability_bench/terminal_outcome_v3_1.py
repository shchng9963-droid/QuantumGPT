"""Measurement-facing union of Agent terminals and runtime failures."""

from __future__ import annotations

from typing import Any, Mapping, TypeAlias

from .runtime_terminal_v1 import (
    RuntimeExecutionFailure,
    parse_runtime_execution_failure,
)
from .terminal_schema_v3 import (
    CompletionStatus,
    TerminalDecision,
    parse_terminal_decision,
)


TERMINAL_OUTCOME_VERSION = "reliabilitybench-q/terminal-outcome-3.1"
TerminalOutcome: TypeAlias = TerminalDecision | RuntimeExecutionFailure


def parse_terminal_outcome(value: Mapping[str, Any]) -> TerminalOutcome:
    if value.get("runtime_generated") is True:
        return parse_runtime_execution_failure(value)
    decision = parse_terminal_decision(value)
    if decision.completion_status is CompletionStatus.EXECUTION_FAILURE:
        raise ValueError("Agent-authored execution_failure is not scoreable in v3.1")
    return decision


__all__ = [
    "TERMINAL_OUTCOME_VERSION",
    "TerminalOutcome",
    "parse_terminal_outcome",
]
