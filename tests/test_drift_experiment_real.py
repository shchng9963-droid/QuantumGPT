"""Regression checks for real drift experiments.

These tests enforce the user-facing requirement that drift experiments use
real ReAct/LLM plumbing and real ToolExecutor/AerSimulator execution, not a
mock executor or hard-coded fidelity boosts.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_drift_experiment_has_no_mock_executor():
    source = (ROOT / "eval" / "drift_experiment.py").read_text()
    assert "MockExecutor" not in source
    assert "DriftAwareMockExecutor" not in source
    assert "base_fidelity = 0.93" not in source


def test_enhanced_experiment_has_no_hardcoded_boost_or_mock_executor():
    source = (ROOT / "eval" / "drift_experiment_enhanced.py").read_text()
    assert "MockExecutor" not in source
    assert "DriftAwareMockExecutor" not in source
    assert "fid + 0.10" not in source
    assert "+ 0.10" not in source


def test_experiments_require_real_llm_provider_when_agentic():
    source = (ROOT / "eval" / "drift_experiment.py").read_text()
    assert "require_real_llm_provider" in source
    assert "use_mock" in source
    assert "False" in source
    assert "deepseek" in source


def test_drift_experiment_writes_paper_grade_jsonl(tmp_path):
    from eval.drift_experiment import DriftExperimentResult, write_paper_grade_results

    result = DriftExperimentResult(
        system="Static-Pipeline",
        task="Run GHZ-5",
        drift_profile="SUDDEN_DEGRADATION",
        fidelities=[0.91, 0.72],
        drift_injected_at=1,
        recovery_step=0,
        final_fidelity=0.72,
        total_steps=2,
        replan_triggered=False,
        replan_at_step=0,
        provider="none",
        model="static-tool-executor",
        tool_calls=["get_backend_health", "run_circuit"],
        trace_files=[],
        metadata={"seed": 7, "elapsed_seconds": 1.5, "pre_fidelity_count": 1},
    )

    out = tmp_path / "result_summary.jsonl"
    write_paper_grade_results([result], out)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["system"] == "Static-Pipeline"
    assert row["execution_mode"] == "simulator"
    assert row["use_mock"] is False
    assert row["provider"] == "none"
    assert row["trace_file"] == "static-pipeline:no-trace"
    assert row["final_score"] == pytest.approx(0.72)
    assert row["best_score"] == pytest.approx(0.91)
    assert row["target_success"] is False
    assert row["metrics"]["first_post_drift_score"] == pytest.approx(0.72)
    assert row["metrics"]["best_post_drift_score"] == pytest.approx(0.72)
    assert row["metrics"]["post_drift_improvement"] == pytest.approx(0.0)
    assert row["metrics"]["post_drift_target_success"] is False
    assert row["diagnostics"]["tool_calls"] == ["get_backend_health", "run_circuit"]


def test_drift_experiment_paper_grade_writer_aggregates_trace_usage(tmp_path):
    from agent.react import AgentTrace
    from eval.drift_experiment import DriftExperimentResult, _usage_metadata_from_traces, write_paper_grade_results

    pre = AgentTrace(user_prompt="pre", model="deepseek-chat", provider="deepseek", backend="FakeBrisbane")
    pre.elapsed_seconds = 1.2
    pre.prompt_tokens_total = 100
    pre.completion_tokens_total = 20
    pre.total_tokens = 120
    pre.final_answer = "pre ok"
    pre.diagnostics.final_answer_length = len(pre.final_answer)

    post = AgentTrace(user_prompt="post", model="deepseek-chat", provider="deepseek", backend="FakeBrisbane")
    post.elapsed_seconds = 2.3
    post.prompt_tokens_total = 200
    post.completion_tokens_total = 30
    post.total_tokens = 230
    post.final_answer = ""
    post.diagnostics.no_final_answer = True
    post.diagnostics.max_turns_exceeded = True

    metadata = _usage_metadata_from_traces([pre, post])
    metadata["pre_fidelity_count"] = 1
    assert metadata["prompt_tokens"] == 300
    assert metadata["completion_tokens"] == 50
    assert metadata["total_tokens"] == 350
    assert metadata["elapsed_seconds"] == pytest.approx(3.5)
    assert metadata["total_cost_usd"] > 0
    assert metadata["trace_diagnostics"][1]["no_final_answer"] is True
    assert metadata["trace_diagnostics"][1]["max_turns_exceeded"] is True

    result = DriftExperimentResult(
        system="QuantumGPT-Full",
        task="Run GHZ-5",
        drift_profile="SUDDEN_DEGRADATION",
        fidelities=[0.9, 0.5, 0.72],
        drift_injected_at=2,
        recovery_step=0,
        final_fidelity=0.72,
        total_steps=3,
        replan_triggered=True,
        replan_at_step=2,
        provider="deepseek",
        model="deepseek-chat",
        tool_calls=["run_circuit"],
        trace_files=["pre.json", "post.json"],
        metadata=metadata,
    )
    out = tmp_path / "usage.jsonl"
    write_paper_grade_results([result], out)
    row = json.loads(out.read_text())
    assert row["prompt_tokens"] == 300
    assert row["completion_tokens"] == 50
    assert row["total_cost_usd"] == pytest.approx(metadata["total_cost_usd"])
    assert row["metrics"]["first_post_drift_score"] == pytest.approx(0.5)
    assert row["metrics"]["best_post_drift_score"] == pytest.approx(0.72)
    assert row["metrics"]["post_drift_improvement"] == pytest.approx(0.22)
    assert row["metrics"]["post_drift_target_success"] is False
    assert row["diagnostics"]["trace_diagnostics"][1]["max_turns_exceeded"] is True


def test_llm_drift_result_does_not_count_pre_drift_fidelity_as_recovery(monkeypatch, tmp_path):
    import eval.drift_experiment as drift
    from agent.react import AgentTrace, TraceStep
    from backends.synthetic_drift import SyntheticDriftBackend

    class FakeAgent:
        provider = "deepseek"
        model = "deepseek-chat"
        planner = None

        def __init__(self):
            self.calls = 0

        def run(self, prompt):
            trace = AgentTrace(user_prompt=prompt, model=self.model, provider=self.provider, backend="FakeBrisbane")
            if self.calls == 0:
                trace.steps.append(TraceStep(step_num=0, action="run_circuit", fidelity_observed=0.80))
                trace.steps.append(TraceStep(step_num=1, action="run_circuit", fidelity_observed=0.91))
                trace.steps.append(TraceStep(step_num=2, action="run_circuit", fidelity_observed=0.89))
            else:
                trace.steps.append(TraceStep(step_num=0, action="run_circuit", fidelity_observed=0.5))
                trace.steps.append(TraceStep(step_num=1, action="run_circuit", fidelity_observed=0.72))
            self.calls += 1
            return trace

    monkeypatch.setattr(drift, "make_real_agent", lambda *args, **kwargs: FakeAgent())

    result = drift.run_llm_drift_experiment(
        SyntheticDriftBackend("FakeBrisbane"),
        target_fidelity=0.85,
        out_dir=tmp_path,
        verbose=False,
    )

    assert result.fidelities == [0.80, 0.91, 0.89, 0.5, 0.72]
    assert result.recovered is False
    assert result.recovery_step == 0
    assert result.metadata["pre_fidelity_count"] == 3
    assert result.metadata["post_fidelity_count"] == 2


def test_drift_experiment_paper_grade_writer_rejects_mock_rows(tmp_path):
    from eval.drift_experiment import DriftExperimentResult, write_paper_grade_results

    result = DriftExperimentResult(
        system="QuantumGPT-Full",
        task="Run GHZ-5",
        drift_profile="SUDDEN_DEGRADATION",
        fidelities=[0.9],
        drift_injected_at=1,
        recovery_step=1,
        final_fidelity=0.9,
        total_steps=1,
        replan_triggered=True,
        replan_at_step=1,
        provider="mock",
        model="rule-planner",
        tool_calls=["run_circuit"],
        metadata={"use_mock": True},
    )

    with pytest.raises(ValueError, match="paper-grade evaluation forbids mock"):
        write_paper_grade_results([result], tmp_path / "bad.jsonl")
