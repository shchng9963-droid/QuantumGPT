"""P0 tests for paper-grade result schema and no-mock guard."""

import json
from pathlib import Path

import pytest

from eval.paper_grade.result_schema import (
    PaperGradeResultRow,
    assert_paper_grade_execution,
    write_result_jsonl,
)


def _valid_row(**overrides):
    data = dict(
        run_id="smoke-ghz5-full-seed0",
        system="Full QuantumGPT",
        task="ghz_5",
        circuit="ghz_5",
        drift_profile="sudden_degradation",
        seed=0,
        execution_mode="real",
        provider="deepseek",
        model="deepseek-chat",
        use_mock=False,
        trace_file="/tmp/trace.json",
        target_fidelity=0.85,
        final_score=0.68,
        best_score=0.70,
        target_success=False,
        tool_calls=12,
        max_tool_calls=24,
        elapsed_seconds=32.5,
        prompt_tokens=1000,
        completion_tokens=200,
        total_cost_usd=0.01,
        diagnostics={"invalid_tool_call_count": 0},
        metrics={"stale_reuse_rate": 0.0},
    )
    data.update(overrides)
    return PaperGradeResultRow(**data)


def test_paper_grade_result_row_has_required_schema_fields():
    row = _valid_row()
    as_dict = row.to_dict()

    required = {
        "run_id",
        "system",
        "task",
        "circuit",
        "drift_profile",
        "seed",
        "execution_mode",
        "provider",
        "model",
        "use_mock",
        "trace_file",
        "target_fidelity",
        "final_score",
        "best_score",
        "target_success",
        "tool_calls",
        "max_tool_calls",
        "elapsed_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_cost_usd",
        "diagnostics",
        "metrics",
    }
    assert required <= set(as_dict)
    assert as_dict["execution_mode"] == "real"
    assert as_dict["provider"] == "deepseek"


def test_paper_grade_execution_guard_rejects_mock_or_rule_planner():
    with pytest.raises(ValueError, match="paper-grade.*mock"):
        assert_paper_grade_execution(execution_mode="mock", provider="deepseek", use_mock=False)

    with pytest.raises(ValueError, match="paper-grade.*mock"):
        assert_paper_grade_execution(execution_mode="real", provider="mock", use_mock=False)

    with pytest.raises(ValueError, match="paper-grade.*mock"):
        assert_paper_grade_execution(execution_mode="real", provider="deepseek", use_mock=True)

    assert_paper_grade_execution(execution_mode="real", provider="deepseek", use_mock=False)


def test_paper_grade_result_row_validation_rejects_incomplete_or_mock_rows():
    with pytest.raises(ValueError, match="trace_file"):
        _valid_row(trace_file="")

    with pytest.raises(ValueError, match="paper-grade.*mock"):
        _valid_row(provider="mock")

    with pytest.raises(ValueError, match="execution_mode"):
        _valid_row(execution_mode="simulated")


def test_write_result_jsonl_round_trips_rows(tmp_path: Path):
    out = tmp_path / "result_summary.jsonl"
    rows = [_valid_row(run_id="r0", seed=0), _valid_row(run_id="r1", seed=1)]

    write_result_jsonl(out, rows)

    loaded = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["run_id"] for r in loaded] == ["r0", "r1"]
    assert all(r["execution_mode"] == "real" for r in loaded)
    assert all(r["provider"] != "mock" for r in loaded)
