"""Oracle-Adaptive baseline tests (v2.5 Sprint A, A2).

Oracle-Adaptive is the *upper-bound, zero-LLM* deterministic baseline: given
the oracle row's action list (which already contains every action's post-drift
fidelity computed by the deterministic oracle module), Oracle-Adaptive picks
the action with maximum fidelity. By construction:

  best_post_fidelity == oracle_best == row.best_post_drift_score
  oracle_gap == 0
  total_cost_usd == 0
  provider == "none"
  model == "oracle-adaptive"

Why this baseline matters:
- It strictly upper-bounds any system that picks a single action from the
  same action grid, so reporting (Agent − Oracle-Adaptive) shows how much
  performance is being left on the table by the agent's planning.
- Unlike Static-Pipeline (which fixes one action), Oracle-Adaptive uses
  hindsight per-task — agent gain over Oracle-Adaptive is impossible by
  construction; the gap quantifies decision-making quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _oracle_row(**overrides):
    """Same shape as v2.4 Static-Pipeline tests; ZNE is the row best at 0.727."""
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


def test_oracle_adaptive_picks_max_fidelity_action():
    """Oracle-Adaptive picks the action with max post-drift fidelity."""
    from eval.baselines.oracle_adaptive import run_oracle_adaptive_task

    row = _oracle_row()
    result = run_oracle_adaptive_task(row)

    assert result.system == "Oracle-Adaptive"
    assert result.provider == "none"
    assert result.model == "oracle-adaptive"
    assert result.first_post_fidelity == pytest.approx(0.727)
    assert result.best_post_fidelity == pytest.approx(0.727)
    assert result.oracle_best == pytest.approx(0.727)
    # Tight gap: by construction Oracle-Adaptive matches oracle best
    assert result.oracle_gap == pytest.approx(0.0, abs=1e-9)
    # No LLM cost
    assert result.metadata["prompt_tokens"] == 0
    assert result.metadata["completion_tokens"] == 0
    assert result.metadata["total_cost_usd"] == 0.0
    assert result.metadata["execution_mode"] == "simulator"
    # The chosen action name must match the oracle best
    assert result.metadata["chosen_action"] == "zne_s1024"


def test_oracle_adaptive_respects_target_success():
    """target_success follows from best_post_fidelity >= target_fidelity."""
    from eval.baselines.oracle_adaptive import run_oracle_adaptive_task

    # Lift the row best above target — should flip to success
    actions_overrides = [
        {"action": "raw_s1024", "fidelity": 0.86, "shots": 1024,
         "depth": 8, "metadata": {"kind": "raw"}},
    ]
    row = _oracle_row(actions=actions_overrides, best_post_drift_score=0.86,
                      best_action="raw_s1024", oracle_feasible=True,
                      target_success=True)
    result = run_oracle_adaptive_task(row)
    assert result.target_success is True
    assert result.best_post_fidelity == pytest.approx(0.86)
    assert result.metadata["chosen_action"] == "raw_s1024"


def test_oracle_adaptive_raises_on_empty_actions():
    """An oracle row with no actions is a malformed row — fail loudly."""
    from eval.baselines.oracle_adaptive import run_oracle_adaptive_task

    row = _oracle_row(actions=[])
    with pytest.raises(ValueError):
        run_oracle_adaptive_task(row)


def test_oracle_adaptive_writes_paper_grade_jsonl(tmp_path):
    """Result must round-trip through write_public_agent_results unchanged."""
    from eval.baselines.oracle_adaptive import run_oracle_adaptive_task
    from eval.public_mqtbench_agent_eval import write_public_agent_results

    row = _oracle_row()
    result = run_oracle_adaptive_task(row)
    out = tmp_path / "oracle_adaptive.jsonl"
    write_public_agent_results([result], out)

    written = json.loads(out.read_text())
    assert written["system"] == "Oracle-Adaptive"
    assert written["provider"] == "none"
    assert written["model"] == "oracle-adaptive"
    assert written["use_mock"] is False
    assert written["execution_mode"] == "simulator"
    assert written["total_cost_usd"] == 0.0
    assert written["metrics"]["best_post_drift_score"] == pytest.approx(0.727)
    assert written["metrics"]["oracle_gap"] == pytest.approx(0.0, abs=1e-9)
    # Oracle-Adaptive does not engage drift detection
    assert written["diagnostics"]["replan_triggered"] is False
    assert written["diagnostics"]["drift_alert_count"] == 0
