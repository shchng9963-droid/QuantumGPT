"""Static-Pipeline baseline tests for public MQTBench eval.

Static-Pipeline applies one fixed deterministic action (no LLM, no oracle
selection, no drift awareness) and reads its post-drift fidelity directly
from the oracle JSONL. This is the right "non-adaptive baseline" line:
agent gain over Static-Pipeline isolates the contribution of LLM + drift
module, while gain over oracle is bounded above (oracle has full
hindsight).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _oracle_row(**overrides):
    row = {
        "task_id": "FakeBrisbane:severe_sudden:ghz_5:seed0",
        "circuit": "ghz_5",
        "circuit_family": "hand_written",
        "backend": "FakeBrisbane",
        "profile": "severe_sudden",
        "seed": 0,
        "target_fidelity": 0.85,
        "pre_fidelity": 0.91,
        "first_post_drift_score": 0.70,
        "best_post_drift_score": 0.727,
        "final_score": 0.727,
        "post_drift_improvement": 0.027,
        "target_success": False,
        "oracle_feasible": False,
        "best_action": "zne_s1024",
        "actions": [
            {"action": "raw_s1024", "fidelity": 0.7020, "shots": 1024,
             "depth": 8, "metadata": {"kind": "raw"}},
            {"action": "transpile_o0_s1024", "fidelity": 0.6796, "shots": 1024,
             "depth": 8, "metadata": {"kind": "transpile", "opt_level": 0}},
            {"action": "transpile_o1_s1024", "fidelity": 0.6691, "shots": 1024,
             "depth": 8, "metadata": {"kind": "transpile", "opt_level": 1}},
            {"action": "transpile_o2_s1024", "fidelity": 0.6737, "shots": 1024,
             "depth": 8, "metadata": {"kind": "transpile", "opt_level": 2}},
            {"action": "transpile_o3_s1024", "fidelity": 0.6816, "shots": 1024,
             "depth": 8, "metadata": {"kind": "transpile", "opt_level": 3}},
            {"action": "zne_s1024", "fidelity": 0.727, "shots": 1024,
             "depth": None, "metadata": {"kind": "zne"}},
        ],
        "metadata": {"drift_score": 1.0},
    }
    row.update(overrides)
    return row


def test_run_static_pipeline_task_uses_fixed_action():
    """Static-Pipeline must read fidelity for the configured action and
    NOT do oracle selection, NOT call any LLM, and NOT charge tokens."""

    from eval.public_mqtbench_agent_eval import run_static_pipeline_task

    row = _oracle_row()
    result = run_static_pipeline_task(row, action="transpile_o1_s1024")

    assert result.system == "Static-Pipeline"
    assert result.provider in {"none", "static"}
    assert result.model == "static-transpile_o1_s1024"
    assert result.target_fidelity == pytest.approx(0.85)
    # Post fidelity must equal the action's fidelity, not oracle best
    assert result.first_post_fidelity == pytest.approx(0.6691)
    assert result.best_post_fidelity == pytest.approx(0.6691)
    assert result.target_success is False
    assert result.oracle_best == pytest.approx(0.727)
    assert result.oracle_gap == pytest.approx(0.727 - 0.6691)
    # No tokens / no cost
    assert result.metadata["prompt_tokens"] == 0
    assert result.metadata["completion_tokens"] == 0
    assert result.metadata["total_cost_usd"] == 0.0
    assert result.metadata["execution_mode"] == "simulator"


def test_run_static_pipeline_task_default_action_is_qiskit_o1():
    from eval.public_mqtbench_agent_eval import run_static_pipeline_task

    row = _oracle_row()
    result = run_static_pipeline_task(row)
    # Default static action must be a fixed identifier, not the row's best.
    assert result.model == "static-transpile_o1_s1024"
    assert result.best_post_fidelity == pytest.approx(0.6691)


def test_run_static_pipeline_task_raises_when_action_missing():
    from eval.public_mqtbench_agent_eval import run_static_pipeline_task

    row = _oracle_row(actions=[{"action": "raw_s1024", "fidelity": 0.7,
                                 "shots": 1024, "depth": 8, "metadata": {}}])
    with pytest.raises(KeyError):
        run_static_pipeline_task(row, action="transpile_o1_s1024")


def test_static_pipeline_writes_paper_grade_jsonl(tmp_path):
    from eval.public_mqtbench_agent_eval import (
        run_static_pipeline_task,
        write_public_agent_results,
    )

    row = _oracle_row()
    result = run_static_pipeline_task(row, action="transpile_o1_s1024")
    out = tmp_path / "static.jsonl"
    write_public_agent_results([result], out)
    written = json.loads(out.read_text())
    assert written["system"] == "Static-Pipeline"
    assert written["execution_mode"] == "simulator"
    assert written["use_mock"] is False
    assert written["provider"] == "none"
    assert written["total_cost_usd"] == 0.0
    assert written["metrics"]["first_post_drift_score"] == pytest.approx(0.6691)
    assert written["metrics"]["best_post_drift_score"] == pytest.approx(0.6691)
    assert written["metrics"]["oracle_gap"] == pytest.approx(0.727 - 0.6691)
    # Static-Pipeline never replans
    assert written["diagnostics"]["replan_triggered"] is False
    assert written["diagnostics"]["drift_alert_count"] == 0
