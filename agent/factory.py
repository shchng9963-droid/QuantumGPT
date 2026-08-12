"""Canonical construction helpers for QuantumGPT agent runtimes.

All production entry points use this module so provider resolution, budgets,
and mock-mode selection do not drift between the CLI and orchestrator.
"""

from __future__ import annotations

from typing import Any


def build_react_agent(
    backend: Any,
    *,
    model: str = "rule-planner-v1",
    provider: str = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
    verbose: bool = False,
    use_mock: bool | None = None,
    target_fidelity: float = 0.85,
    max_tool_calls: int = 20,
    max_seconds: float = 120.0,
    max_turns: int | None = None,
    db_path: str | None = None,
    wandb_run: Any = None,
    use_memory: bool = False,
    use_drift_aware: bool = False,
    safety_policy: Any = None,
):
    """Build the canonical :class:`ReActAgent` used by runtime entry points."""
    from agent.react import ReActAgent

    if use_mock is None:
        use_mock = model == "rule-planner-v1" or provider == "mock"
    if max_turns is None:
        max_turns = max_tool_calls + 5

    return ReActAgent(
        backend=backend,
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        max_turns=max_turns,
        verbose=verbose,
        use_mock=use_mock,
        target_fidelity=target_fidelity,
        max_tool_calls=max_tool_calls,
        max_seconds=max_seconds,
        db_path=db_path,
        wandb_run=wandb_run,
        use_memory=use_memory,
        use_drift_aware=use_drift_aware,
        safety_policy=safety_policy,
    )


__all__ = ["build_react_agent"]
