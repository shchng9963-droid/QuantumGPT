"""Minimal safety policy for physics-facing QuantumGPT tool execution.

The policy is intentionally small: it provides a dry-run mode, an audit trail,
and a few conservative guards for costly or potentially risky experiment calls.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class SafetyDecision:
    """Decision returned before a tool call is executed."""

    allowed: bool
    reason: str = "allowed"


class ToolLike(Protocol):
    def execute(self, tool_name: str, tool_input: dict) -> str: ...


class SafetyPolicy:
    """Guard tool calls and keep an audit log for demo and lab workflows."""

    def __init__(
        self,
        *,
        dry_run: bool = False,
        audit_enabled: bool = True,
        max_rabi_amp_ghz: float = 0.10,
        max_shots: int = 20000,
    ):
        self.dry_run = dry_run
        self.audit_enabled = audit_enabled
        self.max_rabi_amp_ghz = max_rabi_amp_ghz
        self.max_shots = max_shots
        self.audit_log: list[dict[str, Any]] = []

    def assess(self, tool_name: str, tool_input: dict) -> SafetyDecision:
        """Return whether a tool call is allowed under the current policy."""
        if tool_name == "rabi_experiment":
            amp_max = float(tool_input.get("amp_max", 0.08))
            if amp_max > self.max_rabi_amp_ghz:
                return SafetyDecision(
                    allowed=False,
                    reason=(
                        f"amp_max={amp_max:g} GHz exceeds safety limit "
                        f"{self.max_rabi_amp_ghz:g} GHz"
                    ),
                )

        shots = tool_input.get("shots")
        if shots is not None and int(shots) > self.max_shots:
            return SafetyDecision(
                allowed=False,
                reason=f"shots={int(shots)} exceeds safety limit {self.max_shots}",
            )

        return SafetyDecision(allowed=True)

    def record(self, tool_name: str, tool_input: dict, decision: str, reason: str) -> None:
        if not self.audit_enabled:
            return
        self.audit_log.append(
            {
                "ts": time.time(),
                "tool": tool_name,
                "input": dict(tool_input),
                "decision": decision,
                "reason": reason,
                "dry_run": self.dry_run,
            }
        )


class SafetyAwareExecutor:
    """Wrap any ToolExecutor-like object with dry-run, audit, and block behavior."""

    def __init__(self, inner: ToolLike, policy: SafetyPolicy | None = None):
        self.inner = inner
        self.policy = policy or SafetyPolicy()

    def execute(self, tool_name: str, tool_input: dict) -> str:
        decision = self.policy.assess(tool_name, tool_input)
        if not decision.allowed:
            self.policy.record(tool_name, tool_input, "block", decision.reason)
            return json.dumps(
                {
                    "error": "blocked_by_safety_policy",
                    "tool": tool_name,
                    "reason": decision.reason,
                    "safety": {"status": "blocked", "would_execute": False},
                }
            )

        if self.policy.dry_run:
            self.policy.record(tool_name, tool_input, "dry_run", decision.reason)
            return json.dumps(
                {
                    "tool": tool_name,
                    "input": tool_input,
                    "safety": {
                        "status": "dry_run",
                        "would_execute": True,
                        "reason": decision.reason,
                    },
                }
            )

        self.policy.record(tool_name, tool_input, "allow", decision.reason)
        return self.inner.execute(tool_name, tool_input)
