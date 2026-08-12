import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _oracle_row(**overrides):
    row = {
        "task_id": "FakeBrisbane:moderate:ghz_5:seed0",
        "circuit": "ghz_5",
        "circuit_family": "hand_written",
        "backend": "FakeBrisbane",
        "profile": "moderate",
        "seed": 0,
        "target_fidelity": 0.85,
        "pre_fidelity": 0.91,
        "first_post_drift_score": 0.80,
        "best_post_drift_score": 0.89,
        "final_score": 0.89,
        "post_drift_improvement": 0.09,
        "target_success": True,
        "oracle_feasible": True,
        "best_action": "zne_s1024",
        "actions": [],
        "metadata": {"drift_score": 0.15},
    }
    row.update(overrides)
    return row


def test_load_oracle_rows_filters_feasible_and_profile(tmp_path):
    from eval.public_mqtbench_agent_eval import load_oracle_rows

    path = tmp_path / "oracle.jsonl"
    path.write_text("\n".join([
        json.dumps(_oracle_row(task_id="t1", circuit="ghz_5", profile="moderate", oracle_feasible=True)),
        json.dumps(_oracle_row(task_id="t2", circuit="GHZ-5", profile="severe_sudden", oracle_feasible=False)),
        json.dumps(_oracle_row(task_id="t3", circuit="qft_4", profile="moderate", oracle_feasible=True)),
    ]))

    rows = load_oracle_rows(path, profiles={"moderate"}, only_feasible=True, max_tasks=1)

    assert len(rows) == 1
    assert rows[0]["task_id"] == "t1"
    assert rows[0]["oracle_feasible"] is True


def test_build_agent_prompt_names_circuit_target_and_oracle_best():
    from eval.public_mqtbench_agent_eval import build_agent_prompt

    prompt = build_agent_prompt(_oracle_row(circuit="GHZ-5", best_post_drift_score=0.8694))

    assert "GHZ-5" in prompt
    assert "0.85" in prompt
    assert "0.8694" in prompt
    assert "best observed" in prompt.lower()
    assert "backend health" in prompt.lower()
    assert "drift" in prompt.lower()


def test_public_agent_row_computes_post_metrics_and_oracle_gap():
    from eval.public_mqtbench_agent_eval import PublicAgentRunResult, paper_grade_row_from_public_result

    result = PublicAgentRunResult(
        system="QuantumGPT-Full",
        task_id="FakeBrisbane:moderate:ghz_5:seed0",
        circuit="ghz_5",
        backend="FakeBrisbane",
        drift_profile="moderate",
        seed=0,
        target_fidelity=0.85,
        oracle_best=0.89,
        oracle_feasible=True,
        oracle_best_action="zne_s1024",
        pre_fidelities=[0.92],
        post_fidelities=[0.80, 0.86],
        provider="deepseek",
        model="deepseek-chat",
        tool_calls=["get_backend_health", "run_circuit"],
        trace_files=["pre.json", "post.json"],
        metadata={"prompt_tokens": 10, "completion_tokens": 5, "total_cost_usd": 0.001, "elapsed_seconds": 2.0},
    )

    row = paper_grade_row_from_public_result(result)
    data = row.to_dict()

    assert data["execution_mode"] == "real"
    assert data["target_success"] is True
    assert data["final_score"] == pytest.approx(0.86)
    assert data["best_score"] == pytest.approx(0.86)
    assert data["metrics"]["first_post_drift_score"] == pytest.approx(0.80)
    assert data["metrics"]["best_post_drift_score"] == pytest.approx(0.86)
    assert data["metrics"]["post_drift_improvement"] == pytest.approx(0.06)
    assert data["metrics"]["oracle_best"] == pytest.approx(0.89)
    assert data["metrics"]["oracle_gap"] == pytest.approx(0.03)
    assert data["diagnostics"]["oracle_best_action"] == "zne_s1024"


def test_run_public_agent_task_uses_only_post_fidelities_for_success(monkeypatch, tmp_path):
    import eval.public_mqtbench_agent_eval as agent_eval
    from agent.react import AgentTrace, TraceStep

    class FakeAgent:
        provider = "deepseek"
        model = "deepseek-chat"
        planner = None

        def __init__(self):
            self.calls = 0

        def run(self, prompt):
            trace = AgentTrace(user_prompt=prompt, model=self.model, provider=self.provider, backend="FakeBrisbane")
            if self.calls == 0:
                trace.steps.append(TraceStep(step_num=0, action="run_circuit", fidelity_observed=0.93))
            else:
                trace.steps.append(TraceStep(step_num=0, action="run_circuit", fidelity_observed=0.50))
                trace.steps.append(TraceStep(step_num=1, action="run_circuit", fidelity_observed=0.72))
            self.calls += 1
            return trace

    monkeypatch.setattr(agent_eval, "make_real_agent", lambda *args, **kwargs: FakeAgent())

    result = agent_eval.run_public_agent_task(
        _oracle_row(best_post_drift_score=0.89),
        system="QuantumGPT-Full",
        provider="deepseek",
        out_dir=tmp_path,
        verbose=False,
        max_turns=3,
        max_tool_calls=6,
    )

    assert result.pre_fidelities == [0.93]
    assert result.post_fidelities == [0.50, 0.72]
    assert result.best_post_fidelity == pytest.approx(0.72)
    assert result.target_success is False
    assert result.oracle_gap == pytest.approx(0.17)


def test_write_public_agent_results_jsonl(tmp_path):
    from eval.public_mqtbench_agent_eval import PublicAgentRunResult, write_public_agent_results

    result = PublicAgentRunResult(
        system="QuantumGPT-No-Drift",
        task_id="t1",
        circuit="ghz_5",
        backend="FakeBrisbane",
        drift_profile="moderate",
        seed=2,
        target_fidelity=0.85,
        oracle_best=0.89,
        oracle_feasible=True,
        oracle_best_action="zne_s1024",
        pre_fidelities=[0.91],
        post_fidelities=[0.86],
        provider="deepseek",
        model="deepseek-chat",
        tool_calls=["run_circuit"],
        trace_files=["post.json"],
        metadata={"prompt_tokens": 1, "completion_tokens": 1, "total_cost_usd": 0.0},
    )

    out = tmp_path / "result_summary.jsonl"
    write_public_agent_results([result], out)
    row = json.loads(out.read_text())

    assert row["run_id"] == "public_mqtbench:QuantumGPT-No-Drift:t1:seed2"
    assert row["metrics"]["oracle_gap"] == pytest.approx(0.03)
    assert row["diagnostics"]["oracle_feasible"] is True
