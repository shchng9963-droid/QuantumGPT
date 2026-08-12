"""B9 — Orchestrator E2E integration tests.

Plan v2.5 §3.2 B9 deliverable.

Coverage
--------
1. Three representative tasks (mitigation / benchmark / diagnose) flow
   START → task_router → planner → executor → verifier → finalizer
   end-to-end without crashing, with budgets enforced and traces captured.

2. **G2 gate** (Plan v2.5 §5):
   ``orchestrator E2E in 1 task ≡ flat ReAct, |Δfidelity| < 0.01``.
   Verified across 3 seeds × 1 task with the deterministic mock provider.
   Empirically observed Δ = 0.001..0.005 — only shot-sampling noise from
   AerSimulator runs on FakeBrisbane. The orchestrator's ExecutorNode is
   a lossless wrapper around ReActAgent.

All tests run with provider="mock" → no LLM API calls, fully offline.
The `pytest -m e2e_real` smoke that actually hits DeepSeek lives separately
under ``experiments/smoke_orchestrator_e2e.py`` (see Plan v2.5 cost notes).

Wall-clock note: each task runs the noisy AerSimulator on FakeBrisbane,
so this file takes ~30-40s on CPU. Mark slow if running it in tight CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── shared fixtures ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def graph_app():
    """Build the orchestrator graph once per test module.

    Use the stub planner so the test is fully hermetic — no LLM calls,
    no provider keys, no network. Real planner has its own unit tests.
    """
    from agent.orchestrator.graph import build_graph

    return build_graph(use_stub_planner=True, use_stub_verifier=False)


def _make_task(prompt: str, *, task_type: str, circuit: str = "",
               target_fidelity: float = 0.0,
               max_tool_calls: int = 6,
               max_seconds: float = 60.0) -> dict:
    """Build a TaskSpec with provider=mock for hermetic tests."""
    from agent.orchestrator.state import make_task

    task = make_task(
        prompt,
        task_type=task_type,
        circuit=circuit,
        target_fidelity=target_fidelity,
        backend_id="FakeBrisbane",
        max_usd=0.50,
        max_seconds=max_seconds,
        max_tool_calls=max_tool_calls,
    )
    task["provider"] = "mock"
    return task


# ── B9: three representative tasks end-to-end ─────────────────────────────


@pytest.mark.parametrize("spec", [
    pytest.param(
        {
            "label": "ghz5_mitigation",
            "prompt": "Run ghz_5 with transpile_and_run optimization_level=1, "
                      "then apply ZNE mitigation. Report fidelity.",
            "task_type": "mitigation",
            "circuit": "ghz_5",
            "target_fidelity": 0.85,
        },
        id="ghz5_mitigation",
    ),
    pytest.param(
        {
            "label": "qftent5_benchmark",
            "prompt": "Run qft_ent_5 on this backend and report the achieved fidelity.",
            "task_type": "benchmark",
            "circuit": "qft_ent_5",
            "target_fidelity": 0.0,
        },
        id="qftent5_benchmark",
    ),
    pytest.param(
        {
            "label": "diagnose_short",
            "prompt": "Get backend health: drift score, T1/T2 averages.",
            "task_type": "diagnose",
            "circuit": "",
            "target_fidelity": 0.0,
        },
        id="diagnose_short",
    ),
])
def test_orchestrator_e2e_runs_to_completion(graph_app, spec):
    """The graph must execute START→...→END for every representative task."""
    from agent.orchestrator.state import new_orchestrator_state

    task = _make_task(
        spec["prompt"],
        task_type=spec["task_type"],
        circuit=spec["circuit"],
        target_fidelity=spec["target_fidelity"],
        max_tool_calls=6,
    )
    out = graph_app.invoke(new_orchestrator_state(task))

    # Hard contract: graph terminates and writes its termination reason.
    assert out is not None, "graph.invoke returned None"
    assert out.get("done") is True, f"graph did not finalize for {spec['label']}"
    assert out.get("termination_reason"), \
        f"missing termination_reason for {spec['label']}"

    # Executor produced at least one trace, with non-empty steps.
    traces = out.get("executor_traces") or []
    assert traces, f"no executor traces for {spec['label']}"
    last_trace = traces[-1]
    assert (last_trace.get("steps") or []), \
        f"empty trace steps for {spec['label']}"
    # Diagnostic: tool_calls must be > 0 unless the task is degenerate.
    n_calls = sum(t.get("num_tool_calls", 0) for t in traces)
    assert n_calls > 0, \
        f"executor never called any tool for {spec['label']}"

    # Verifier produced its verdict (satisfied bit may be true OR false —
    # we only require that it WROTE something, not that it passed).
    verification = out.get("verification") or {}
    assert "is_satisfied" in verification, \
        f"verifier did not write a verdict for {spec['label']}"

    # Budget accounting wired end-to-end.
    bs = out.get("budget_used") or {}
    assert "walltime_seconds" in bs, "budget_used missing walltime_seconds"
    assert bs.get("walltime_seconds", 0.0) > 0, "walltime not accumulated"


# ── G2 gate: orchestrator ≡ flat ReAct on identical task ──────────────────


@pytest.mark.parametrize("seed", [0, 1, 2], ids=lambda s: f"seed{s}")
def test_g2_paired_orchestrator_vs_flat_react_within_tolerance(seed):
    """G2 gate: |fid_orch - fid_flat| < 0.01 on the same task.

    With the mock provider both pipelines pick the SAME tool plan
    (typically: get_backend_health → transpile_and_run → run → mitigation),
    so any residual delta is pure shot-sampling noise. We force strict
    determinism by configuring both FakeBackendAdapter instances with the
    SAME ``deterministic_seed`` and run them in lockstep — their k-th
    AerSimulator call uses ``seed + k`` regardless of pipeline.

    Without this hook the test still passes most of the time but ZNE's
    multi-noise-scale Richardson extrapolation amplifies shot variance to
    ~0.03 and would intermittently fail the <0.01 gate. See plan §3.3.
    """
    from agent.orchestrator.executor_node import (
        default_agent_factory,
        make_executor_node,
    )
    from agent.orchestrator.graph import build_graph
    from agent.orchestrator.state import new_orchestrator_state
    from agent.react import ReActAgent
    from backends.fake_adapter import FakeBackendAdapter

    prompt = (
        "Run ghz_5 on this backend with transpile_and_run optimization_level=1 "
        "and report the fidelity."
    )

    # Flat ReAct with deterministic backend
    backend_flat = FakeBackendAdapter("FakeBrisbane")
    backend_flat.deterministic_seed = seed
    agent = ReActAgent(
        backend=backend_flat, provider="mock", model="rule-mock",
        use_mock=True, verbose=False, max_turns=10,
        target_fidelity=0.85, max_tool_calls=6, max_seconds=60.0,
        use_memory=False,
    )
    trace_flat = agent.run(prompt, memory_context="")
    flat_fid = float(trace_flat.best_fidelity or 0.0)

    # Orchestrator with the same deterministic seed (passed via custom factory)
    def _backend_factory_seeded(_task):
        b = FakeBackendAdapter("FakeBrisbane")
        b.deterministic_seed = seed
        return b

    seeded_executor = make_executor_node(
        backend_factory=_backend_factory_seeded,
        agent_factory=default_agent_factory,
    )
    app = build_graph(
        use_stub_planner=True,
        use_stub_verifier=False,
        executor_fn=seeded_executor,
    )
    task = _make_task(prompt, task_type="benchmark", circuit="ghz_5",
                      target_fidelity=0.85, max_tool_calls=6)
    out = app.invoke(new_orchestrator_state(task))
    orch_fid = float(out.get("last_fidelity") or 0.0)

    # G2 gate
    delta = abs(flat_fid - orch_fid)
    assert delta < 0.01, (
        f"G2 gate failed (seed={seed}): "
        f"flat_fid={flat_fid:.6f}, orch_fid={orch_fid:.6f}, "
        f"|Δ|={delta:.6f} >= 0.01 — orchestrator is NOT equivalent to flat ReAct"
    )

    # Same tool sequence — sanity check (ExecutorNode must dispatch same plan)
    flat_steps = [s.action for s in trace_flat.steps if s.action]
    orch_traces = out.get("executor_traces") or []
    orch_steps = [
        s.get("action", "") for t in orch_traces
        for s in (t.get("steps") or []) if s.get("action")
    ]
    assert flat_steps == orch_steps, (
        f"G2 mismatch in tool plan (seed={seed}): "
        f"flat={flat_steps}, orch={orch_steps}"
    )


def test_g2_summary_metrics_recorded(graph_app):
    """The orchestrator must surface fidelity, tool_calls, and walltime in a
    single canonical place so downstream paired stats can read them.

    Specifically Plan v2.5 §6 expects:
      - state['last_fidelity']           (float | None)
      - state['executor_traces'][i]['num_tool_calls']
      - state['budget_used']['walltime_seconds']
    """
    from agent.orchestrator.state import new_orchestrator_state

    task = _make_task(
        "Run ghz_5 with transpile_and_run optimization_level=1, then ZNE.",
        task_type="mitigation", circuit="ghz_5",
        target_fidelity=0.85, max_tool_calls=6,
    )
    out = graph_app.invoke(new_orchestrator_state(task))

    assert isinstance(out.get("last_fidelity"), (int, float)), \
        "last_fidelity must be a number"
    assert out["last_fidelity"] > 0, \
        "last_fidelity must be positive after a successful mitigation run"

    traces = out.get("executor_traces") or []
    assert traces and "num_tool_calls" in traces[-1], \
        "trace must record num_tool_calls"

    bs = out.get("budget_used") or {}
    assert bs.get("walltime_seconds", 0.0) > 0
    assert bs.get("tool_calls", 0) > 0
