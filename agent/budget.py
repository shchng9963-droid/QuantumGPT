"""Fidelity budget tracker for the ReAct agent.

Tracks a target fidelity and remaining "budget" (number of tool calls,
wall-clock time, or both). The agent consults the budget before each
action to decide whether to:
  - proceed normally
  - try mitigation (if fidelity is below target)
  - stop early (budget exhausted)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class BudgetDecision(Enum):
    PROCEED = auto()
    MITIGATE = auto()
    STOP = auto()
    GOAL_MET = auto()


@dataclass
class FidelityBudget:
    """Tracks fidelity target and resource budget.

    Attributes:
        target_fidelity: desired fidelity threshold (0..1)
        max_tool_calls: maximum number of tool calls allowed
        max_seconds: maximum wall-clock time
        mitigation_attempted: whether mitigation has been tried
    """
    target_fidelity: float = 0.85
    max_tool_calls: int = 20
    max_seconds: float = 120.0

    # Internal state
    tool_calls_used: int = 0
    best_fidelity: float | None = None
    mitigation_attempted: bool = False
    _start_time: float = 0.0
    _observations: list[dict] = field(default_factory=list)

    def start(self):
        self._start_time = time.time()
        self.tool_calls_used = 0
        self.best_fidelity = None
        self.mitigation_attempted = False
        self._observations = []

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls_used)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    def record_tool_call(self, tool_name: str, fidelity: float | None = None):
        """Record a tool call and optionally update best fidelity."""
        self.tool_calls_used += 1
        if tool_name == "apply_mitigation":
            self.mitigation_attempted = True
        obs = {"tool": tool_name, "t": round(self.elapsed, 2)}
        if fidelity is not None:
            obs["fidelity"] = fidelity
            if self.best_fidelity is None or fidelity > self.best_fidelity:
                self.best_fidelity = fidelity
        self._observations.append(obs)

    def decide(self) -> BudgetDecision:
        """Decide what to do next based on current state."""
        # Goal already met?
        if self.best_fidelity is not None and self.best_fidelity >= self.target_fidelity:
            return BudgetDecision.GOAL_MET

        # Budget exhausted?
        if self.remaining_calls <= 0 or self.remaining_seconds <= 0:
            return BudgetDecision.STOP

        # Fidelity below target and mitigation not tried yet?
        if (self.best_fidelity is not None
                and self.best_fidelity < self.target_fidelity
                and not self.mitigation_attempted):
            return BudgetDecision.MITIGATE

        return BudgetDecision.PROCEED

    def summary(self) -> dict:
        return {
            "target_fidelity": self.target_fidelity,
            "best_fidelity": self.best_fidelity,
            "tool_calls_used": self.tool_calls_used,
            "remaining_calls": self.remaining_calls,
            "elapsed_seconds": round(self.elapsed, 1),
            "remaining_seconds": round(self.remaining_seconds, 1),
            "mitigation_attempted": self.mitigation_attempted,
            "decision": self.decide().name,
        }

    def budget_prompt_insert(self) -> str:
        """Generate a prompt fragment that tells the LLM about budget state."""
        d = self.decide()
        s = self.summary()
        lines = [
            f"[FIDELITY BUDGET STATUS]",
            f"  Target: {s['target_fidelity']:.2f}",
            f"  Best achieved: {s['best_fidelity'] if s['best_fidelity'] is not None else 'N/A'}",
            f"  Tool calls: {s['tool_calls_used']}/{self.max_tool_calls}",
            f"  Time: {s['elapsed_seconds']}s / {self.max_seconds}s",
            f"  Decision: {d.name}",
        ]
        if d == BudgetDecision.MITIGATE:
            lines.append("  → You should try apply_mitigation or transpile_circuit to improve fidelity.")
        elif d == BudgetDecision.STOP:
            lines.append("  → Budget exhausted. Provide your final answer now.")
        elif d == BudgetDecision.GOAL_MET:
            lines.append("  → Fidelity target met! Summarize results and stop.")
        return "\n".join(lines)
