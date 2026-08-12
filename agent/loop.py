"""Backward-compatible facade for the canonical ReAct runtime.

Historically this module contained a second, independent agent loop. Production
entry points now use :class:`agent.react.ReActAgent`; ``QuantumAgent`` remains
only to preserve the original public API for archived demos and external users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TURNS = 15


@dataclass
class AgentTurn:
    """One legacy-format conversation turn."""

    role: str
    content: Any
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class AgentResult:
    """Legacy result view produced from an ``AgentTrace``."""

    final_answer: str
    turns: list[AgentTurn]
    tool_calls_made: list[dict]
    total_tokens: int
    elapsed_seconds: float
    model: str
    best_fidelity: float | None = None
    total_cost_usd: float = 0.0


class RulePlanner:
    """Compatibility adapter for the former two-value planner API."""

    def __init__(self):
        self._planner = None
        self._budget = None

    def plan(
        self, user_prompt: str, history: list[dict]
    ) -> tuple[list[dict], Optional[str]]:
        from agent.budget import FidelityBudget
        from agent.react import ReActRulePlanner

        if self._planner is None:
            self._planner = ReActRulePlanner()
            self._budget = FidelityBudget(
                target_fidelity=0.85,
                max_tool_calls=MAX_TURNS,
                max_seconds=120.0,
            )
            self._budget.start()
        _, calls, final_text = self._planner.plan(
            user_prompt, history, self._budget
        )
        return calls, final_text


class QuantumAgent:
    """Compatibility facade delegating every run to ``ReActAgent``.

    New code should import ``ReActAgent`` or use
    :func:`agent.factory.build_react_agent` directly.
    """

    def __init__(
        self,
        backend,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = "auto",
        max_turns: int = MAX_TURNS,
        verbose: bool = True,
        use_mock: bool = False,
        db_path: Optional[str] = None,
        wandb_run=None,
    ):
        from agent.factory import build_react_agent

        self._agent = build_react_agent(
            backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider=provider,
            max_turns=max_turns,
            max_tool_calls=max_turns,
            verbose=verbose,
            use_mock=use_mock or provider == "mock",
            db_path=db_path,
            wandb_run=wandb_run,
            use_memory=False,
        )

    def __getattr__(self, name: str):
        """Expose legacy runtime attributes from the canonical agent."""
        agent = object.__getattribute__(self, "_agent")
        return getattr(agent, name)

    def run(self, user_prompt: str) -> AgentResult:
        trace = self._agent.run(user_prompt)
        result = trace.to_agent_result()
        result.best_fidelity = trace.best_fidelity
        result.total_cost_usd = float(
            trace.cost_summary.get("total_cost_usd") or 0.0
        )
        return result


__all__ = [
    "AgentResult",
    "AgentTurn",
    "QuantumAgent",
    "RulePlanner",
]
