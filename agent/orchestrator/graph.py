"""LangGraph wiring for the QuantumGPT orchestrator.

Topology (Sprint B / v2.5 plan §3.1, B1 stub version):

    START → task_router → planner → executor → verifier
                                                  │
                                  ┌───────────────┼───────────────┐
                                  ▼               ▼               ▼
                              finalize        reflect          finalize
                                  │              │               (best-effort)
                                  │              ▼               │
                                  │           planner            │
                                  │              │               │
                                  │              ▼               │
                                  │           executor           │
                                  │              │               │
                                  │              ▼               │
                                  │           verifier ──────────┤
                                  │                              │
                                  ▼                              ▼
                                          END

Nodes are imported from :mod:`agent.orchestrator.nodes`. In B1 every node is
a stub; B2..B6 swap real implementations into the same slots.
"""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from agent.orchestrator.executor_node import executor_node as _real_executor
from agent.orchestrator.planner_node import planner_node as _real_planner
from agent.orchestrator.verifier_node import verifier_node as _real_verifier
from agent.orchestrator.nodes import (
    executor as _stub_executor,
    finalizer,
    planner as _stub_planner,
    reflector,
    route_after_reflector,
    route_after_verifier,
    task_router,
    verifier as _stub_verifier,
)
from agent.orchestrator.state import OrchestratorState


# ── node names (single source of truth — referenced in tests) ──────────────

NODE_TASK_ROUTER = "task_router"
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_VERIFIER = "verifier"
NODE_REFLECTOR = "reflector"
NODE_FINALIZER = "finalizer"


def build_graph(
    *,
    task_router_fn: Callable[..., dict] = task_router,
    planner_fn: Callable[..., dict] | None = None,
    executor_fn: Callable[..., dict] | None = None,
    verifier_fn: Callable[..., dict] | None = None,
    reflector_fn: Callable[..., dict] = reflector,
    finalizer_fn: Callable[..., dict] = finalizer,
    use_stub_executor: bool = False,
    use_stub_verifier: bool = False,
    use_stub_planner: bool = False,
):
    """Compile the orchestrator graph.

    Each node is overridable so tests / future PRs can inject mocks
    without forking the topology. Returns a compiled LangGraph runnable
    with ``.invoke(state)`` and ``.stream(state)`` methods.

    Args:
      use_stub_executor: when True (and no ``executor_fn`` given), use the
        original B1 stub executor that returns a placeholder trace without
        any quantum stack. Useful for hello-world tests.
      use_stub_verifier: when True (and no ``verifier_fn`` given), use the
        always-accept stub verifier (B1 behaviour). Useful when testing
        upstream nodes in isolation.
      use_stub_planner: when True (and no ``planner_fn`` given), use the
        single-step stub planner that doesn't call any LLM. Useful for
        offline / hermetic tests.
    """
    if planner_fn is None:
        planner_fn = _stub_planner if use_stub_planner else _real_planner
    if executor_fn is None:
        executor_fn = _stub_executor if use_stub_executor else _real_executor
    if verifier_fn is None:
        verifier_fn = _stub_verifier if use_stub_verifier else _real_verifier

    g: StateGraph = StateGraph(OrchestratorState)

    g.add_node(NODE_TASK_ROUTER, task_router_fn)
    g.add_node(NODE_PLANNER, planner_fn)
    g.add_node(NODE_EXECUTOR, executor_fn)
    g.add_node(NODE_VERIFIER, verifier_fn)
    g.add_node(NODE_REFLECTOR, reflector_fn)
    g.add_node(NODE_FINALIZER, finalizer_fn)

    g.add_edge(START, NODE_TASK_ROUTER)
    g.add_edge(NODE_TASK_ROUTER, NODE_PLANNER)
    g.add_edge(NODE_PLANNER, NODE_EXECUTOR)
    g.add_edge(NODE_EXECUTOR, NODE_VERIFIER)

    g.add_conditional_edges(
        NODE_VERIFIER,
        route_after_verifier,
        {
            "finalize": NODE_FINALIZER,
            "reflect": NODE_REFLECTOR,
        },
    )

    g.add_conditional_edges(
        NODE_REFLECTOR,
        route_after_reflector,
        {
            "replan": NODE_PLANNER,
        },
    )

    g.add_edge(NODE_FINALIZER, END)

    return g.compile()


__all__ = [
    "build_graph",
    "NODE_TASK_ROUTER",
    "NODE_PLANNER",
    "NODE_EXECUTOR",
    "NODE_VERIFIER",
    "NODE_REFLECTOR",
    "NODE_FINALIZER",
]
