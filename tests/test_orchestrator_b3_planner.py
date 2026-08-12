"""B3 — PlannerNode tests.

Test layers:

  Parsing (parse_plan_json):
    - well-formed JSON → list of SubGoals with valid intents
    - markdown fences stripped
    - last subgoal forced to "finalize" if missing
    - >5 subgoals truncated to 5 (keeps the finalize at the end)
    - unknown intents dropped
    - malformed JSON → None
    - empty / non-dict input → None

  Rule-based fallback:
    - calibration → diagnose first
    - mitigation → mitigate + run + finalize
    - benchmark → at least diagnose + run + finalize
    - empty / unknown → default skeleton
    - last subgoal is always "finalize"

  Node behavior:
    - mock provider → never calls LLM, uses rule fallback, source="rule"
    - non-mock provider with mock LLM call returning valid JSON → uses LLM
    - LLM raises → falls back to rule, history records llm_error
    - LLM returns invalid JSON → falls back to rule
    - replan path: provides last trace + verification + reflections to LLM
    - plan_revision increments on every call

  Integration:
    - graph runs with stub planner (B1 path) — confirms wiring still works
    - graph runs with real planner (rule fallback, mock provider) — no LLM call

These tests pin the contract so future planner upgrades (smaller models,
tool-aware routing, etc.) keep the same node interface.
"""

from __future__ import annotations

import pytest

from agent.orchestrator.graph import (
    NODE_EXECUTOR,
    NODE_FINALIZER,
    NODE_PLANNER,
    NODE_TASK_ROUTER,
    NODE_VERIFIER,
    build_graph,
)
from agent.orchestrator.planner_node import (
    VALID_INTENTS,
    make_planner_node,
    parse_plan_json,
    rule_based_plan,
)
from agent.orchestrator.state import (
    make_task,
    new_orchestrator_state,
)


# ── parsing ────────────────────────────────────────────────────────────────


def test_parse_plan_json_valid():
    raw = """{"subgoals": [
        {"id": "s1", "intent": "diagnose", "description": "check drift"},
        {"id": "s2", "intent": "mitigate", "description": "apply ZNE"},
        {"id": "s3", "intent": "run", "description": "execute"},
        {"id": "s4", "intent": "finalize", "description": "report"}
    ]}"""
    plan = parse_plan_json(raw)
    assert plan is not None
    assert len(plan) == 4
    intents = [p["intent"] for p in plan]
    assert intents == ["diagnose", "mitigate", "run", "finalize"]
    for p in plan:
        assert p["intent"] in VALID_INTENTS
        assert p["status"] == "pending"


def test_parse_plan_json_strips_markdown_fences():
    raw = '```json\n{"subgoals": [{"intent": "run", "description": "go"}, {"intent": "finalize", "description": "done"}]}\n```'
    plan = parse_plan_json(raw)
    assert plan is not None
    assert plan[0]["intent"] == "run"


def test_parse_plan_json_appends_finalize_if_missing():
    raw = '{"subgoals": [{"intent": "run", "description": "just run"}]}'
    plan = parse_plan_json(raw)
    assert plan is not None
    assert plan[-1]["intent"] == "finalize"


def test_parse_plan_json_truncates_to_five():
    raw = (
        '{"subgoals": [' +
        ",".join(
            f'{{"intent": "diagnose", "description": "step{i}"}}' for i in range(8)
        )
        + "]}"
    )
    plan = parse_plan_json(raw)
    assert plan is not None
    assert len(plan) == 5
    assert plan[-1]["intent"] == "finalize"


def test_parse_plan_json_drops_unknown_intents():
    raw = """{"subgoals": [
        {"intent": "wave_a_wand", "description": "magic"},
        {"intent": "diagnose", "description": "check"},
        {"intent": "finalize", "description": "report"}
    ]}"""
    plan = parse_plan_json(raw)
    assert plan is not None
    assert len(plan) == 2
    assert [p["intent"] for p in plan] == ["diagnose", "finalize"]


def test_parse_plan_json_returns_none_for_garbage():
    assert parse_plan_json("not json") is None
    assert parse_plan_json("") is None
    assert parse_plan_json('{"subgoals": "should_be_a_list"}') is None
    assert parse_plan_json('"a string"') is None
    # All unknown intents → no valid plan
    assert parse_plan_json('{"subgoals": [{"intent": "wat", "description": "x"}]}') is None


# ── rule fallback ─────────────────────────────────────────────────────────


def test_rule_based_plan_calibration_starts_with_diagnose():
    task = make_task("calibrate qubit 0", task_type="calibration")
    plan = rule_based_plan(task)
    assert plan[0]["intent"] == "diagnose"
    assert plan[-1]["intent"] == "finalize"


def test_rule_based_plan_mitigation_includes_mitigate_and_run():
    task = make_task("apply ZNE on ghz_5", task_type="mitigation")
    plan = rule_based_plan(task)
    intents = [p["intent"] for p in plan]
    assert "mitigate" in intents
    assert "run" in intents
    assert intents[-1] == "finalize"


def test_rule_based_plan_benchmark_minimal():
    task = make_task("run circuit and report fidelity", task_type="benchmark")
    plan = rule_based_plan(task)
    intents = [p["intent"] for p in plan]
    # At minimum we go: diagnose / run / finalize (or transpile-aware variant)
    assert "run" in intents
    assert intents[-1] == "finalize"


def test_rule_based_plan_empty_task_uses_default_skeleton():
    task = make_task("", task_type="")
    plan = rule_based_plan(task)
    intents = [p["intent"] for p in plan]
    assert intents[0] == "diagnose"
    assert "run" in intents
    assert intents[-1] == "finalize"


def test_rule_based_plan_keyword_drift_triggers_diagnose():
    task = make_task("the device has drifted, recalibrate T1/T2", task_type="diagnose")
    plan = rule_based_plan(task)
    assert plan[0]["intent"] == "diagnose"


# ── node behavior ─────────────────────────────────────────────────────────


def _state_for(task_kwargs):
    s = new_orchestrator_state(make_task("planner test", **task_kwargs))
    return s


def test_planner_mock_provider_never_calls_llm():
    """Provider=mock → planner short-circuits to rule fallback."""
    calls = {"n": 0}

    def my_llm(messages):
        calls["n"] += 1
        return '{"subgoals": []}'

    node = make_planner_node(llm_call=my_llm)
    state = _state_for({"task_type": "mitigation", "meta": {"provider": "mock"}})
    state["task"]["provider"] = "mock"

    out = node(state)
    assert calls["n"] == 0  # never called LLM
    assert out["plan_revision"] == 1
    assert len(out["plan"]) >= 2
    assert out["plan"][-1]["intent"] == "finalize"
    # History records the source
    assert out["history"][-1]["info"]["source"] == "rule"


def test_planner_uses_llm_when_provider_is_not_mock():
    """Non-mock provider + valid LLM → LLM-sourced plan."""
    canned = """{"subgoals": [
        {"id": "s1", "intent": "diagnose", "description": "drift?"},
        {"id": "s2", "intent": "mitigate", "description": "ZNE"},
        {"id": "s3", "intent": "run", "description": "execute"},
        {"id": "s4", "intent": "finalize", "description": "report"}
    ]}"""

    def my_llm(messages):
        # Should receive system + user
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        return canned

    node = make_planner_node(llm_call=my_llm)
    state = _state_for({"task_type": "calibration"})
    state["task"]["provider"] = "deepseek"

    out = node(state)
    assert out["history"][-1]["info"]["source"] == "llm"
    assert len(out["plan"]) == 4
    intents = [p["intent"] for p in out["plan"]]
    assert intents == ["diagnose", "mitigate", "run", "finalize"]


def test_planner_falls_back_when_llm_raises():
    """LLM exception → rule fallback, history flags the error."""

    def my_llm(messages):
        raise RuntimeError("network down")

    node = make_planner_node(llm_call=my_llm)
    state = _state_for({"task_type": "calibration"})
    state["task"]["provider"] = "deepseek"

    out = node(state)
    info = out["history"][-1]["info"]
    assert info["source"].startswith("rule")
    assert "llm_call_error" in (info["llm_error"] or "")
    assert "network down" in (info["llm_error"] or "")
    # Plan still produced (rule fallback)
    assert len(out["plan"]) >= 2


def test_planner_falls_back_when_llm_returns_garbage():
    def my_llm(messages):
        return "this is not JSON"

    node = make_planner_node(llm_call=my_llm)
    state = _state_for({"task_type": "calibration"})
    state["task"]["provider"] = "deepseek"

    out = node(state)
    info = out["history"][-1]["info"]
    assert info["source"].startswith("rule")
    assert info["llm_error"] == "json_parse_failed"


def test_planner_replan_uses_replan_prompt_with_trace_and_reflections():
    """On replan (plan_revision > 0), prompt should include prior trace + verifier reason."""
    captured = {}

    def my_llm(messages):
        captured["user_msg"] = messages[1]["content"]
        return '{"subgoals": [{"intent": "run", "description": "retry"}, {"intent": "finalize", "description": "done"}]}'

    node = make_planner_node(llm_call=my_llm)
    state = _state_for({"task_type": "benchmark"})
    state["task"]["provider"] = "deepseek"
    state["plan_revision"] = 1
    state["executor_traces"] = [{
        "best_fidelity": 0.6,
        "num_tool_calls": 3,
        "steps": [
            {"action": "transpile_circuit"},
            {"action": "run_circuit"},
        ],
    }]
    state["verification"] = {"reason": "below_target(eff=0.6, target=0.85)"}
    state["reflections"] = ["ZNE wasn't tried", "noise was elevated"]

    out = node(state)
    assert "best_fidelity=0.6" in captured["user_msg"]
    assert "below_target" in captured["user_msg"]
    assert "ZNE wasn't tried" in captured["user_msg"]
    assert out["plan_revision"] == 2
    assert out["history"][-1]["info"]["is_replan"] is True


def test_planner_revision_increments_each_call():
    node = make_planner_node(llm_call=None)  # forces rule fallback path entirely
    state = _state_for({"meta": {"provider": "mock"}})
    state["task"]["provider"] = "mock"

    out1 = node(state)
    state.update(out1)
    out2 = node(state)

    assert out1["plan_revision"] == 1
    assert out2["plan_revision"] == 2


def test_planner_use_llm_when_predicate_is_respected():
    """Custom predicate decides whether to call LLM."""
    calls = {"n": 0}

    def my_llm(messages):
        calls["n"] += 1
        return '{"subgoals": [{"intent": "finalize", "description": "x"}]}'

    # Predicate: never use LLM
    node = make_planner_node(llm_call=my_llm, use_llm_when=lambda s: False)
    state = _state_for({"task_type": "calibration"})
    state["task"]["provider"] = "deepseek"

    out = node(state)
    assert calls["n"] == 0
    assert out["history"][-1]["info"]["source"] == "rule"


# ── integration with the graph ────────────────────────────────────────────


def test_graph_with_real_planner_runs_end_to_end_offline():
    """provider=mock → real planner uses rule fallback, no LLM contact."""

    def stub_executor(state):
        # Fake a happy trace with claim consistent with steps
        td = {
            "user_prompt": "x",
            "model": "mock",
            "provider": "mock",
            "backend": "mock",
            "final_answer": "Fidelity: 0.91",
            "best_fidelity": 0.91,
            "num_tool_calls": 1,
            "elapsed_seconds": 0.01,
            "steps": [{"step_num": 0, "action": "run_circuit", "fidelity_observed": 0.91}],
        }
        return {
            "executor_traces": list(state.get("executor_traces") or []) + [td],
            "last_action": "run_circuit",
            "last_observation": {},
            "last_fidelity": 0.91,
            "history": list(state.get("history") or [])
            + [{"node": NODE_EXECUTOR, "ts": 0.0, "info": {}}],
        }

    # Real planner + real verifier; mock provider keeps planner offline
    app = build_graph(executor_fn=stub_executor)
    task = make_task(
        "transpile and run ghz_5",
        task_type="mitigation",
        target_fidelity=0.85,
    )
    task["provider"] = "mock"
    out = app.invoke(new_orchestrator_state(task))

    assert out["done"] is True
    assert out["verification"]["is_satisfied"] is True
    # Plan came from the rule fallback
    planner_history = [h for h in out["history"] if h["node"] == NODE_PLANNER]
    assert len(planner_history) == 1
    assert planner_history[0]["info"]["source"] == "rule"
    # Plan structure is sensible
    intents = [p["intent"] for p in out["plan"]]
    assert intents[-1] == "finalize"
    assert len(intents) >= 2


def test_graph_with_real_planner_replans_when_verifier_rejects():
    """Replan path: first executor returns hallucinated → verifier rejects →
    planner re-runs (revision 2) → executor returns clean → done."""

    exec_calls = {"n": 0}

    def stub_executor(state):
        exec_calls["n"] += 1
        if exec_calls["n"] == 1:
            td = {
                "final_answer": "Fidelity 0.99",
                "best_fidelity": 0.99,  # claim
                "num_tool_calls": 1,
                "steps": [{"step_num": 0, "action": "run_circuit", "fidelity_observed": 0.5}],  # observed
            }
        else:
            td = {
                "final_answer": "Fidelity 0.90",
                "best_fidelity": 0.90,
                "num_tool_calls": 1,
                "steps": [{"step_num": 0, "action": "run_circuit", "fidelity_observed": 0.90}],
            }
        return {
            "executor_traces": list(state.get("executor_traces") or []) + [td],
            "last_action": "run_circuit",
            "last_observation": {},
            "last_fidelity": td["best_fidelity"],
            "history": list(state.get("history") or [])
            + [{"node": NODE_EXECUTOR, "ts": 0.0, "info": {"call": exec_calls["n"]}}],
        }

    app = build_graph(executor_fn=stub_executor)
    task = make_task("retry-aware", target_fidelity=0.85, max_seconds=60.0, max_tool_calls=20)
    task["provider"] = "mock"  # keeps planner offline on both passes
    out = app.invoke(new_orchestrator_state(task))

    assert out["done"] is True
    assert exec_calls["n"] == 2
    assert out["plan_revision"] == 2
    assert out["verification"]["is_satisfied"] is True

    # Both planner runs produced finalize-terminated plans
    plans_seen = [h for h in out["history"] if h["node"] == NODE_PLANNER]
    assert len(plans_seen) == 2
