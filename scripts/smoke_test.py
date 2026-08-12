#!/usr/bin/env python3
"""Run one deterministic, offline QuantumGPT Agent loop."""

from __future__ import annotations

import json

from agent.factory import build_react_agent
from backends.fake_adapter import FakeBackendAdapter


def run_smoke() -> dict:
    backend = FakeBackendAdapter("FakeBrisbane")
    agent = build_react_agent(
        backend,
        model="rule-planner-v1",
        provider="mock",
        use_mock=True,
        use_memory=False,
        verbose=False,
        max_tool_calls=3,
        max_seconds=30.0,
    )
    trace = agent.run("Check backend health and summarize.")
    tool_names = [call["tool"] for call in trace.tool_calls_made]
    if tool_names != ["get_backend_health"]:
        raise RuntimeError(f"Unexpected smoke tool sequence: {tool_names}")
    if not trace.final_answer.strip():
        raise RuntimeError("Smoke run returned an empty final answer")
    if trace.diagnostics.no_final_answer:
        raise RuntimeError("Smoke diagnostics reports no final answer")
    return {
        "status": "ok",
        "backend": backend.name,
        "provider": trace.provider,
        "tool_calls": tool_names,
        "final_answer_length": len(trace.final_answer),
        "elapsed_seconds": round(trace.elapsed_seconds, 3),
    }


def main() -> int:
    print(json.dumps(run_smoke(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
