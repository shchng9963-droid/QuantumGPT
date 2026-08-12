"""Structured trace models shared by QuantumGPT agents and reports."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agent.state import AgentState

@dataclass
class TraceStep:
    """One step in the ReAct trace."""
    step_num: int
    thought: str | None = None
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    fidelity_observed: float | None = None
    budget_decision: str | None = None
    state_summary: dict[str, Any] | None = None
    timestamp: float = 0.0
    # F5: per-step token and latency tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_latency_ms: float = 0.0

    def to_dict(self, include_observation: bool = False) -> dict:
        data = {k: v for k, v in {
            "step": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation_length": len(self.observation) if self.observation else 0,
            "fidelity_observed": self.fidelity_observed,
            "budget_decision": self.budget_decision,
            "state_summary": self.state_summary,
            "timestamp": round(self.timestamp, 3),
            "prompt_tokens": self.prompt_tokens or None,
            "completion_tokens": self.completion_tokens or None,
            "api_latency_ms": round(self.api_latency_ms, 1) if self.api_latency_ms else None,
        }.items() if v is not None}
        if include_observation and self.observation is not None:
            data["observation"] = self.observation
        return data


@dataclass
class TraceDiagnostics:
    """Structured reliability diagnostics for one agent run."""
    requested_provider: str = "mock"
    resolved_provider: str = "mock"
    model: str = ""
    final_answer_length: int = 0
    no_final_answer: bool = False
    invalid_tool_call_count: int = 0
    malformed_json_count: int = 0
    repeated_tool_count: int = 0
    max_turns_exceeded: bool = False
    hallucinated_tool_names: list[str] = field(default_factory=list)
    safety_block_count: int = 0
    safety_dry_run_count: int = 0
    drift_alert_count: int = 0
    replan_triggered: bool = False
    drift_alert_steps: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_provider": self.requested_provider,
            "resolved_provider": self.resolved_provider,
            "model": self.model,
            "final_answer_length": self.final_answer_length,
            "no_final_answer": self.no_final_answer,
            "invalid_tool_call_count": self.invalid_tool_call_count,
            "malformed_json_count": self.malformed_json_count,
            "repeated_tool_count": self.repeated_tool_count,
            "max_turns_exceeded": self.max_turns_exceeded,
            "hallucinated_tool_names": self.hallucinated_tool_names,
            "safety_block_count": self.safety_block_count,
            "safety_dry_run_count": self.safety_dry_run_count,
            "drift_alert_count": self.drift_alert_count,
            "replan_triggered": self.replan_triggered,
            "drift_alert_steps": list(self.drift_alert_steps),
        }


@dataclass
class AgentTrace:
    """Full trace of a ReAct agent run."""
    user_prompt: str
    model: str
    provider: str
    backend: str
    steps: list[TraceStep] = field(default_factory=list)
    final_answer: str = ""
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    budget_summary: dict = field(default_factory=dict)
    diagnostics: TraceDiagnostics = field(default_factory=TraceDiagnostics)
    state: AgentState | None = None
    # F5: detailed token breakdown
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0

    @property
    def best_fidelity(self) -> float:
        return self.budget_summary.get("best_fidelity", 0.0)

    @property
    def cost_summary(self) -> dict:
        """Estimate API cost based on provider pricing (USD per 1M tokens)."""
        PRICING = {
            "deepseek": {"input": 0.27, "output": 1.10},  # DeepSeek-V3
            "openai": {"input": 3.00, "output": 15.00},   # GPT-4o
            "anthropic": {"input": 3.00, "output": 15.00}, # Claude Sonnet
            "mock": {"input": 0.0, "output": 0.0},
        }
        rates = PRICING.get(self.provider, PRICING["openai"])
        input_cost = self.prompt_tokens_total * rates["input"] / 1_000_000
        output_cost = self.completion_tokens_total * rates["output"] / 1_000_000
        return {
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens_total,
            "completion_tokens": self.completion_tokens_total,
            "total_tokens": self.total_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
        }

    @property
    def num_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action is not None)

    @property
    def tool_calls_made(self) -> list[dict]:
        return [
            {"turn": s.step_num, "tool": s.action, "input": s.action_input}
            for s in self.steps if s.action is not None
        ]

    def to_dict(self, include_observations: bool = False) -> dict[str, Any]:
        """Serialize trace with compact steps and structured diagnostics."""
        data = {
            "user_prompt": self.user_prompt,
            "model": self.model,
            "provider": self.provider,
            "backend": self.backend,
            "final_answer": self.final_answer,
            "total_tokens": self.total_tokens,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "num_tool_calls": self.num_tool_calls,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "cost_summary": self.cost_summary,
            "budget_summary": self.budget_summary,
            "diagnostics": self.diagnostics.to_dict(),
            "steps": [
                s.to_dict(include_observation=include_observations)
                for s in self.steps
            ],
        }
        if self.state is not None:
            data["state_summary"] = self.state.summary_for_prompt()
            data["artifacts"] = self.state.artifacts.to_dict()
        return data

    def to_agent_result(self):
        """Convert to legacy AgentResult for compatibility."""
        from agent.loop import AgentResult, AgentTurn
        turns = [AgentTurn(role="user", content=self.user_prompt, timestamp=0)]
        for s in self.steps:
            if s.thought:
                turns.append(AgentTurn(role="assistant", content=f"Thought: {s.thought}", timestamp=s.timestamp))
            if s.action:
                turns.append(AgentTurn(
                    role="assistant",
                    content=f"[tool] {s.action}",
                    tool_calls=[{"name": s.action, "input": s.action_input}],
                    timestamp=s.timestamp,
                ))
        turns.append(AgentTurn(role="assistant", content=self.final_answer, timestamp=time.time()))

        return AgentResult(
            final_answer=self.final_answer,
            turns=turns,
            tool_calls_made=self.tool_calls_made,
            total_tokens=self.total_tokens,
            elapsed_seconds=self.elapsed_seconds,
            model=self.model,
            best_fidelity=self.best_fidelity,
            total_cost_usd=float(self.cost_summary.get("total_cost_usd") or 0.0),
        )


__all__ = ["TraceStep", "TraceDiagnostics", "AgentTrace"]
