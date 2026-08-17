"""B0 deterministic controller and audit contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.reliability_bench.b0 import (
    B0_GROUPS,
    SHARED_BUDGET,
    _public_input,
    _select_invalidated,
    build_b0_report,
    run_b0,
    write_b0_outputs,
)
from eval.reliability_bench.generator import generate_stage_a_episodes
from eval.reliability_bench.groups import ExperimentGroup


def _traces():
    return run_b0(generate_stage_a_episodes())


def test_b0_runs_120_episodes_across_exactly_six_groups():
    traces = _traces()
    assert len(traces) == 720
    assert {item["controller"]["group"] for item in traces} == {
        item.value for item in B0_GROUPS
    }
    assert ExperimentGroup.PROMPT_ORACLE not in B0_GROUPS


def test_b0_public_prompts_tools_and_budgets_are_identical_per_episode():
    report = build_b0_report(_traces())
    audit = report["prompt_fairness_audit"]
    assert audit["passed"] is True
    assert audit["all_six_groups_share_exact_prompt"] is True
    assert audit["shared_tool_catalog"] is True
    assert audit["shared_budget"] is True
    assert audit["prompt_leakage_trace_ids"] == []


def test_b0_opaque_event_ids_do_not_reveal_relevance():
    for episode in generate_stage_a_episodes():
        event_id = episode.drift_event.neutral_payload()["event_id"].lower()
        assert "related" not in event_id
        assert "unrelated" not in event_id


def test_b0_label_firewall_excludes_oracle_data_from_full_and_other_groups():
    report = build_b0_report(_traces())
    audit = report["label_isolation_audit"]
    assert audit["passed"] is True
    assert audit["ordinary_controller_visible_context_violations"] == []
    assert audit["full_oracle_channel_used"] is False


def test_b0_label_firewall_rejects_accidental_oracle_labels_for_full():
    episode = generate_stage_a_episodes()[0]
    with pytest.raises(ValueError, match="non-Oracle"):
        _select_invalidated(
            ExperimentGroup.FULL_SELECTIVE,
            _public_input(episode),
            [],
            episode.ground_truth.invalidated_artifact_ids,
        )


def test_b0_all_decisions_are_scored_and_traces_are_complete():
    traces = _traces()
    assert all(item["controller"]["trace_complete"] for item in traces)
    assert all(
        isinstance(item["evaluator"]["program_judge"]["correct"], bool)
        for item in traces
    )
    assert {item["evaluator"]["task_type"] for item in traces} == {
        "backend_selection",
        "qubit_mapping",
        "transpilation",
        "fidelity_claim",
        "mitigation_decision",
        "unreachable_target",
    }


def test_b0_global_over_invalidation_becomes_real_tool_cost():
    report = build_b0_report(_traces())
    metrics = report["group_metrics"]
    global_group = metrics[ExperimentGroup.GLOBAL_REVALIDATE.value]
    full = metrics[ExperimentGroup.FULL_SELECTIVE.value]
    assert global_group["false_invalidated_artifacts"] == 180
    assert global_group["tool_calls"] > full["tool_calls"]
    assert global_group["cost_units"] > full["cost_units"]
    assert report["acceptance"]["global_false_invalidations_trigger_tools"]


def test_b0_state_changes_action_and_ledger_can_reuse_stale_evidence():
    report = build_b0_report(_traces())
    metrics = report["group_metrics"]
    assert report["acceptance"][
        "full_changes_decision_after_related_invalidation"
    ]
    assert metrics[ExperimentGroup.LEDGER_ONLY.value][
        "stale_evidence_reuse"
    ] > 0


def test_b0_every_run_respects_the_same_budget():
    for trace in _traces():
        usage = trace["controller"]["budget_usage"]
        assert usage["tool_calls"] <= SHARED_BUDGET.max_tool_calls
        assert usage["cost_units"] <= SHARED_BUDGET.max_cost_units
        assert usage["turns"] <= SHARED_BUDGET.max_turns
        assert usage["tokens"] == 0


def test_b0_human_audit_export_is_blind_and_pending(tmp_path):
    report = write_b0_outputs(generate_stage_a_episodes(), tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "b0_human_audit_sample.jsonl")
        .read_text()
        .splitlines()
    ]
    keys = [
        json.loads(line)
        for line in (tmp_path / "b0_human_audit_key.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(keys) == 24
    assert all(item["human_correct"] is None for item in rows)
    assert all("program_judge" not in item for item in rows)
    assert all("group" not in item for item in rows)
    assert report["b1_readiness"]["ready"] is False


def test_tracked_b0_outputs_match_the_deterministic_runner():
    root = Path(__file__).resolve().parents[2]
    data = root / "eval/reliability_bench/data"
    tracked = [
        json.loads(line)
        for line in (data / "b0_traces.jsonl").read_text().splitlines()
    ]
    generated = _traces()
    summary = json.loads((data / "b0_summary.json").read_text())
    assert tracked == generated
    assert summary == build_b0_report(generated)
    assert summary["acceptance_passed"] is True
    assert summary["development_only"] is True
    assert summary["performance_claim_allowed"] is False
