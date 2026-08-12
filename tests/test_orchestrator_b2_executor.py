"""B2 — ExecutorNode tests (real ReAct wrapping).

Two layers:

  Unit (fast, fully synthetic):
    - feed mock backend + mock agent factories
    - assert ExecutorNode correctly:
        * appends a trace dict to executor_traces
        * propagates best_fidelity / last_action / last_observation
        * accumulates budget (tokens, cost, tool_calls, walltime)
        * recovers from agent.run() exceptions and sets re_execution_needed

  Integration (slower, uses Qiskit FakeBrisbane + rule-based mock LLM):
    - end-to-end orchestrator graph runs a mitigation prompt
    - asserts the run completes and produces a non-trivial trace

The integration test uses ``provider="mock"`` so it stays offline and
deterministic. Marked with ``slow`` so we can selectively skip it.
"""

from __future__ import annotations

import pytest

from agent.orchestrator.executor_node import (
    _accumulate_budget,
    _trace_best_fidelity,
    _trace_cost,
    _trace_last_step,
    make_executor_node,
)
from agent.orchestrator.graph import build_graph
from agent.orchestrator.state import (
    make_task,
    new_orchestrator_state,
)


# ── synthetic fixtures ─────────────────────────────────────────────────────


class _FakeTrace:
    """Minimal stand-in for AgentTrace that supports ``to_dict()``."""

    def __init__(self, *, fidelity=0.92, tool_calls=3, tokens_in=120, tokens_out=80, cost_usd=0.001):
        self._fidelity = fidelity
        self._tool_calls = tool_calls
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._cost_usd = cost_usd

    @property
    def best_fidelity(self):
        return self._fidelity

    def to_dict(self):
        return {
            "user_prompt": "synthetic",
            "model": "rule-mock",
            "provider": "mock",
            "backend": "synthetic",
            "final_answer": "Mitigation applied; fidelity 0.92",
            "best_fidelity": self._fidelity,
            "num_tool_calls": self._tool_calls,
            "elapsed_seconds": 0.5,
            "prompt_tokens_total": self._tokens_in,
            "completion_tokens_total": self._tokens_out,
            "cost_summary": {
                "provider": "mock",
                "prompt_tokens": self._tokens_in,
                "completion_tokens": self._tokens_out,
                "total_tokens": self._tokens_in + self._tokens_out,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": self._cost_usd,
            },
            "budget_summary": {"best_fidelity": self._fidelity},
            "steps": [
                {"step_num": 0, "action": "transpile", "fidelity_observed": 0.85, "observation": "ok"},
                {"step_num": 1, "action": "zne_extrapolate", "fidelity_observed": self._fidelity, "observation": "fid=0.92"},
            ],
        }


class _FakeAgent:
    def __init__(self, *, trace=None, raise_exc=None):
        self._trace = trace or _FakeTrace()
        self._raise = raise_exc
        self.run_calls: list[tuple[str, str]] = []
        self.last_task: dict | None = None

    def run(self, prompt, memory_context=""):
        self.run_calls.append((prompt, memory_context))
        if self._raise is not None:
            raise self._raise
        return self._trace


def _backend_factory(task):
    return object()


def _agent_factory_with(trace=None, raise_exc=None):
    fake = _FakeAgent(trace=trace, raise_exc=raise_exc)

    def _factory(backend, task):
        fake.last_task = dict(task)
        return fake

    return _factory, fake


# ── unit tests ─────────────────────────────────────────────────────────────


def test_executor_node_appends_trace_and_extracts_fidelity():
    factory, fake = _agent_factory_with(trace=_FakeTrace(fidelity=0.91))
    node = make_executor_node(backend_factory=_backend_factory, agent_factory=factory)

    state = new_orchestrator_state(make_task("mitigate ghz_5", task_type="mitigation"))
    update = node(state)

    assert len(update["executor_traces"]) == 1
    trace = update["executor_traces"][0]
    assert trace["best_fidelity"] == 0.91
    assert trace["final_answer"].startswith("Mitigation")
    assert update["last_fidelity"] == 0.91
    assert update["last_action"] == "zne_extrapolate"  # last actionable step
    assert update["re_execution_needed"] is False

    # Agent saw the user prompt
    assert fake.run_calls[0][0] == "mitigate ghz_5"


def test_executor_node_accumulates_budget():
    factory, _ = _agent_factory_with(
        trace=_FakeTrace(tool_calls=4, tokens_in=200, tokens_out=150, cost_usd=0.005)
    )
    node = make_executor_node(backend_factory=_backend_factory, agent_factory=factory)

    state = new_orchestrator_state(make_task("first run"))
    upd1 = node(state)
    state.update(upd1)

    # Run a second time — budget should accumulate, not reset
    factory2, _ = _agent_factory_with(
        trace=_FakeTrace(tool_calls=2, tokens_in=100, tokens_out=50, cost_usd=0.002)
    )
    node2 = make_executor_node(backend_factory=_backend_factory, agent_factory=factory2)
    upd2 = node2(state)

    bs = upd2["budget_used"]
    assert bs["tokens_in"] == 300
    assert bs["tokens_out"] == 200
    assert bs["tool_calls"] == 6
    assert abs(bs["cost_usd"] - 0.007) < 1e-9
    assert bs["walltime_seconds"] >= 0.0
    assert bs["started_at"] > 0  # preserved across runs


def test_executor_node_handles_agent_exception_and_signals_retry():
    factory, _ = _agent_factory_with(raise_exc=RuntimeError("API down"))
    node = make_executor_node(backend_factory=_backend_factory, agent_factory=factory)

    state = new_orchestrator_state(make_task("flaky"))
    update = node(state)

    assert update["re_execution_needed"] is True
    assert update["last_fidelity"] == 0.0
    assert "executor error" in update["executor_traces"][-1]["final_answer"]
    assert "API down" in update["executor_traces"][-1]["error"]
    # History records the failure mode
    last_history = update["history"][-1]
    assert last_history["node"] == "executor"
    assert "error" in last_history["info"]


def test_executor_node_passes_plan_as_memory_context():
    factory, fake = _agent_factory_with()
    node = make_executor_node(backend_factory=_backend_factory, agent_factory=factory)

    state = new_orchestrator_state(make_task("plan-aware"))
    state["plan"] = [
        {"id": "s1", "intent": "diagnose", "description": "check drift", "status": "pending"},
        {"id": "s2", "intent": "mitigate", "description": "apply ZNE", "status": "pending"},
    ]
    state["reflections"] = ["last time ZNE beat transpile_o1 at high noise"]

    node(state)

    memory_ctx = fake.run_calls[0][1]
    assert "diagnose" in memory_ctx
    assert "ZNE" in memory_ctx
    assert "last time" in memory_ctx


def test_executor_node_reserves_retry_budget_on_first_attempt():
    factory, fake = _agent_factory_with()
    node = make_executor_node(backend_factory=_backend_factory, agent_factory=factory)

    state = new_orchestrator_state(make_task("reserve-budget", max_tool_calls=12))

    node(state)

    assert fake.last_task is not None
    assert fake.last_task["max_tool_calls"] == 9


def test_executor_node_passes_only_remaining_tool_budget_on_retry():
    factory, fake = _agent_factory_with()
    node = make_executor_node(backend_factory=_backend_factory, agent_factory=factory)

    state = new_orchestrator_state(make_task("retry-aware", max_tool_calls=12))
    state["verifier_retry_count"] = 1
    state["budget_used"] = {
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "walltime_seconds": 1.0,
        "tool_calls": 9,
        "started_at": 1.0,
        "deadline_at": None,
    }

    node(state)

    assert fake.last_task is not None
    assert fake.last_task["max_tool_calls"] == 3


# ── helper coverage ────────────────────────────────────────────────────────


def test_trace_helpers_handle_dict_input():
    td = _FakeTrace().to_dict()
    assert _trace_best_fidelity(td) == 0.92
    last = _trace_last_step(td)
    assert last["tool"] == "zne_extrapolate"
    cost = _trace_cost(td)
    assert cost["tokens_in"] == 120
    assert cost["tokens_out"] == 80
    assert cost["tool_calls"] == 3  # uses num_tool_calls field, not derived from steps


def test_accumulate_budget_initializes_when_empty():
    out = _accumulate_budget(None, {"tokens_in": 5, "tokens_out": 3, "cost_usd": 0.01, "tool_calls": 1, "elapsed": 1.0})
    assert out["tokens_in"] == 5
    assert out["tool_calls"] == 1
    assert out["started_at"] > 0


# ── integration: real ReActAgent + FakeBrisbane (slow, offline) ───────────


@pytest.mark.slow
def test_executor_e2e_with_react_mock_provider():
    """End-to-end: orchestrator graph runs ReActAgent (mock provider) on FakeBrisbane.

    This is the B2 acceptance criterion: 'flat ReAct equivalent' invocation
    completes and produces a non-trivial trace through the orchestrator wiring.
    """
    pytest.importorskip("qiskit_ibm_runtime")  # FakeBrisbane lives there

    app = build_graph()
    task = make_task(
        "transpile and run ghz_5 on this backend, report the fidelity",
        task_type="mitigation",
        circuit="ghz_5",
        backend_id="FakeBrisbane",
        target_fidelity=0.0,  # accept anything; mock planner is rule-based
        max_tool_calls=8,
        max_seconds=60.0,
        meta={"provider": "mock"},
    )
    # Inject provider into the spot the default agent factory reads
    task["provider"] = "mock"
    state = new_orchestrator_state(task)

    out = app.invoke(state)

    assert out["done"] is True
    assert len(out["executor_traces"]) >= 1

    trace = out["executor_traces"][0]
    # Trace shape contract
    assert "final_answer" in trace
    assert "steps" in trace
    # Agent should have made at least one tool call to reach fidelity
    assert trace["num_tool_calls"] >= 1
    # Budget should have been touched
    bs = out["budget_used"]
    assert bs["tool_calls"] >= 1
    assert bs["walltime_seconds"] > 0.0
