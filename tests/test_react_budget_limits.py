"""Regression tests for enforcing tool-call caps inside a single ReAct turn."""

from __future__ import annotations

from agent.react import ReActAgent


class _DummyBackend:
    name = "FakeTest"
    num_qubits = 5


class _OvereagerPlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, user_prompt, history, budget, memory_context=""):
        self.calls += 1
        if self.calls == 1:
            return (
                "Try several tools at once.",
                [
                    {"name": "get_backend_health", "input": {}},
                    {"name": "detect_drift", "input": {}},
                    {"name": "list_benchmarks", "input": {}},
                ],
                None,
            )
        return ("Budget exhausted. Summarizing findings.", [], "done")


def test_mock_react_agent_does_not_exceed_tool_cap_with_multi_tool_turn(monkeypatch):
    agent = ReActAgent(
        _DummyBackend(),
        provider="mock",
        use_mock=True,
        use_memory=False,
        verbose=False,
        max_turns=4,
        max_tool_calls=2,
        max_seconds=60.0,
    )
    agent.planner = _OvereagerPlanner()

    def _fake_execute(_tool_name, _tool_input):
        return '{"ok": true}'

    monkeypatch.setattr(agent, "_execute_tool", _fake_execute)

    trace = agent.run("Check backend and drift")

    assert trace.num_tool_calls == 2
    assert trace.budget_summary["tool_calls_used"] == 2
