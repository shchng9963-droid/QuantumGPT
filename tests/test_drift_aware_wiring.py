"""Drift-aware wiring tests: ensure DriftMonitor transitions are surfaced
into trace diagnostics and into the LLM prompt for real LLM agents."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_drift_aware_agent():
    from agent.react import ReActAgent
    from backends.synthetic_drift import STABLE, SyntheticDriftBackend

    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    backend.set_time(0.0)
    agent = ReActAgent(
        backend=backend,
        provider="mock",
        use_mock=True,
        use_memory=False,
        use_drift_aware=True,
        target_fidelity=0.85,
        max_tool_calls=24,
        verbose=False,
    )
    return agent, backend


def _make_no_drift_agent():
    from agent.react import ReActAgent
    from backends.synthetic_drift import STABLE, SyntheticDriftBackend

    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    backend.set_time(0.0)
    agent = ReActAgent(
        backend=backend,
        provider="mock",
        use_mock=True,
        use_memory=False,
        use_drift_aware=False,
        target_fidelity=0.85,
        max_tool_calls=24,
        verbose=False,
    )
    return agent, backend


def _new_trace(agent):
    return agent._new_trace("test prompt")


def _new_step(step_num):
    from agent.react import TraceStep

    return TraceStep(step_num=step_num)


def test_drift_aware_records_alert_and_replan_on_transition():
    """When drift monitor transitions stable→drifting, trace diagnostics
    must record a drift alert, set replan_triggered=True, and queue a
    pending drift-alert text for the next LLM turn."""

    from backends.synthetic_drift import SUDDEN_DEGRADATION

    agent, backend = _make_drift_aware_agent()
    assert agent.drift_monitor is not None

    trace = _new_trace(agent)

    # Stable tool call: no drift, no alert
    agent._record_drift_after_tool(trace, _new_step(0))
    assert trace.diagnostics.replan_triggered is False
    assert trace.diagnostics.drift_alert_count == 0
    assert not getattr(trace, "_pending_drift_alert", "")

    # Inject hard drift, simulate next tool call
    backend.profile = SUDDEN_DEGRADATION
    backend.set_time(6.0)
    agent._record_drift_after_tool(trace, _new_step(1))

    assert trace.diagnostics.replan_triggered is True
    assert trace.diagnostics.drift_alert_count >= 1
    pending = getattr(trace, "_pending_drift_alert", "")
    assert isinstance(pending, str) and "DRIFT" in pending.upper()
    assert "drift_score" in pending.lower() or "drift score" in pending.lower()


def test_drift_aware_does_not_double_count_persistent_drift():
    from backends.synthetic_drift import SUDDEN_DEGRADATION

    agent, backend = _make_drift_aware_agent()
    trace = _new_trace(agent)

    backend.profile = SUDDEN_DEGRADATION
    backend.set_time(6.0)
    agent._record_drift_after_tool(trace, _new_step(0))
    first = trace.diagnostics.drift_alert_count
    # Another tool call while still drifting must NOT increment alert count again
    agent._record_drift_after_tool(trace, _new_step(1))
    assert trace.diagnostics.drift_alert_count == first
    assert trace.diagnostics.replan_triggered is True


def test_no_drift_system_records_no_alerts_even_after_drift():
    from backends.synthetic_drift import SUDDEN_DEGRADATION

    agent, backend = _make_no_drift_agent()
    assert agent.drift_monitor is None
    trace = _new_trace(agent)

    backend.profile = SUDDEN_DEGRADATION
    backend.set_time(6.0)
    agent._record_drift_after_tool(trace, _new_step(0))

    assert trace.diagnostics.replan_triggered is False
    assert trace.diagnostics.drift_alert_count == 0
    assert not getattr(trace, "_pending_drift_alert", "")


def test_pending_drift_alert_is_injected_into_openai_messages():
    """In _run_openai, a pending drift alert must be appended to the
    chat messages as a system message before the next API call so the
    LLM actually sees it."""

    from agent.react import ReActAgent, AgentTrace, TraceStep
    from agent.state import AgentState
    from backends.synthetic_drift import STABLE, SyntheticDriftBackend

    class _FakeMessage:
        def __init__(self, content):
            self.content = content
            self.tool_calls = None

        def model_dump(self):
            return {"role": "assistant", "content": self.content}

    class _FakeUsage:
        prompt_tokens = 1
        completion_tokens = 1

    class _FakeChoice:
        def __init__(self, content):
            self.message = _FakeMessage(content)

    class _FakeResponse:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()

    class _RecordingChat:
        def __init__(self):
            self.captured: list[list[dict]] = []

        def create(self, *, model, max_tokens, tools, messages):
            # Capture a snapshot of messages exactly as sent.
            import copy
            self.captured.append(copy.deepcopy(messages))
            return _FakeResponse("done")

    class _FakeClient:
        def __init__(self):
            self.chat = type("C", (), {"completions": _RecordingChat()})()

    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    backend.set_time(0.0)
    agent = ReActAgent(
        backend=backend,
        provider="deepseek",
        api_key="dummy",
        use_mock=False,
        use_memory=False,
        use_drift_aware=True,
        target_fidelity=0.85,
        max_turns=1,
        max_tool_calls=4,
        verbose=False,
    )
    # Force OpenAI-compatible path with a recording client.
    fake = _FakeClient()
    agent.client = fake
    agent.use_mock = False
    agent.provider = "deepseek"

    # Pre-seed a pending drift alert as if a previous turn triggered it.
    # Hook through a wrapper so the alert is set just before _run_openai.
    original = agent._run_openai

    def patched(user_prompt, memory_context=""):
        trace = AgentTrace(
            user_prompt=user_prompt, model=agent.model, provider=agent.provider, backend=backend.name,
            state=AgentState(user_prompt=user_prompt, backend_name=backend.name),
        )
        # We can't easily reach the trace from outside; instead, monkeypatch _new_trace.
        return original(user_prompt, memory_context=memory_context)

    seeded = {}

    def patched_new_trace(user_prompt):
        trace = AgentTrace(
            user_prompt=user_prompt, model=agent.model, provider=agent.provider, backend=backend.name,
            state=AgentState(user_prompt=user_prompt, backend_name=backend.name),
        )
        trace._pending_drift_alert = "[DRIFT ALERT] Device parameters have shifted; replan now."
        seeded["trace"] = trace
        return trace

    agent._new_trace = patched_new_trace  # type: ignore

    trace = agent._run_openai("hi")
    captured = fake.chat.completions.captured
    assert captured, "_run_openai never called the recording client"
    first_call_messages = captured[0]
    contents = " ".join((m.get("content") or "") for m in first_call_messages if isinstance(m, dict))
    assert "DRIFT ALERT" in contents


def test_paper_grade_row_surfaces_drift_alert_and_replan():
    """public_mqtbench_agent_eval.paper_grade_row_from_public_result must
    expose replan_triggered and drift_alert_count in diagnostics so the
    JSONL captures wiring evidence."""

    from eval.public_mqtbench_agent_eval import (
        PublicAgentRunResult,
        paper_grade_row_from_public_result,
    )

    result = PublicAgentRunResult(
        system="QuantumGPT-Full",
        task_id="t",
        circuit="ghz_5",
        backend="FakeBrisbane",
        drift_profile="severe_sudden",
        seed=0,
        target_fidelity=0.85,
        oracle_best=0.89,
        oracle_feasible=True,
        oracle_best_action="zne_s1024",
        pre_fidelities=[0.92],
        post_fidelities=[0.7, 0.86],
        provider="deepseek",
        model="deepseek-chat",
        tool_calls=["run_circuit"],
        trace_files=["pre.json", "post.json"],
        metadata={
            "execution_mode": "real",
            "use_mock": False,
            "drift_alert_count": 2,
            "replan_triggered": True,
        },
    )

    row = paper_grade_row_from_public_result(result).to_dict()
    assert row["diagnostics"]["replan_triggered"] is True
    assert row["diagnostics"]["drift_alert_count"] == 2
