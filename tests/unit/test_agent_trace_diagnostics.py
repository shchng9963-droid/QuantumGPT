"""Tests for AgentTrace diagnostics and serialization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.react import AgentTrace, ReActAgent, TraceDiagnostics, TraceStep


class DummyBackend:
    name = "FakeTest"
    num_qubits = 5


def test_agent_trace_to_dict_includes_diagnostics():
    trace = AgentTrace(user_prompt="p", model="m", provider="mock", backend="FakeBrisbane")
    trace.final_answer = "done"
    trace.diagnostics.requested_provider = "deepseek"
    trace.diagnostics.resolved_provider = "openai"
    trace.diagnostics.model = "deepseek-chat"

    data = trace.to_dict()

    assert data["diagnostics"]["requested_provider"] == "deepseek"
    assert data["diagnostics"]["resolved_provider"] == "openai"
    assert data["diagnostics"]["model"] == "deepseek-chat"
    assert data["diagnostics"]["no_final_answer"] is False


def test_trace_step_to_dict_can_include_observation_when_requested():
    trace = AgentTrace(user_prompt="p", model="m", provider="mock", backend="FakeBrisbane")
    trace.steps.append(TraceStep(step_num=0, action="get_backend_health", observation='{"ok": true}'))

    compact = trace.to_dict(include_observations=False)
    verbose = trace.to_dict(include_observations=True)

    assert compact["steps"][0]["observation_length"] == len('{"ok": true}')
    assert "observation" not in compact["steps"][0]
    assert verbose["steps"][0]["observation"] == '{"ok": true}'


def test_trace_diagnostics_marks_empty_final_answer():
    diag = TraceDiagnostics(model="deepseek-chat")
    diag.final_answer_length = 0
    diag.no_final_answer = True

    data = diag.to_dict()

    assert data["model"] == "deepseek-chat"
    assert data["final_answer_length"] == 0
    assert data["no_final_answer"] is True


def test_react_agent_populates_provider_and_max_turn_diagnostics():
    agent = ReActAgent(
        DummyBackend(),
        provider="mock",
        use_memory=False,
        max_turns=0,
        verbose=False,
    )

    trace = agent.run("Check backend health")

    assert trace.diagnostics.requested_provider == "mock"
    assert trace.diagnostics.resolved_provider == "mock"
    assert trace.diagnostics.model == "react-rule-planner-v1"
    assert trace.diagnostics.max_turns_exceeded is True
    assert trace.diagnostics.final_answer_length == len(trace.final_answer)
    assert trace.diagnostics.no_final_answer is False


def test_react_agent_records_tool_call_diagnostics():
    agent = ReActAgent(DummyBackend(), provider="mock", use_memory=False, verbose=False)
    trace = agent._new_trace("p")
    seen_calls = set()

    agent._record_tool_diagnostics(trace, "get_backend_health", {}, seen_calls)
    agent._record_tool_diagnostics(trace, "get_backend_health", {}, seen_calls)
    agent._record_tool_diagnostics(trace, "made_up_tool", {"x": 1}, seen_calls)

    assert trace.diagnostics.repeated_tool_count == 1
    assert trace.diagnostics.invalid_tool_call_count == 1
    assert trace.diagnostics.hallucinated_tool_names == ["made_up_tool"]
