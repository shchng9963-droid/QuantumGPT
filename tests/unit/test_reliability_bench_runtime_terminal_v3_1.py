from __future__ import annotations

import hashlib
from pathlib import Path

from eval.reliability_bench.action_guard_contract_v1_2 import ReasonCode
from eval.reliability_bench.action_guard_preflight_v1_3 import (
    CompletionResult,
    generate_development_preflight_episodes,
    run_preflight_trace,
)
from eval.reliability_bench.action_guard_runtime_v1_3 import (
    close_runtime_failure_state,
)
from eval.reliability_bench.gold_trace_spec_v3 import build_development_cases
from eval.reliability_bench.public_runtime_v3_1 import (
    StudyArm,
    parse_agent_terminal_runtime_output,
)
from eval.reliability_bench.runtime_terminal_v1 import (
    parse_runtime_execution_failure,
)
from eval.reliability_bench.schema import episode_from_dict
from eval.reliability_bench.task_predicate_evaluator_v3_1 import (
    aggregate_selective_metrics,
    evaluate_terminal_outcome,
)
from eval.reliability_bench.terminal_outcome_v3_1 import parse_terminal_outcome
from eval.reliability_bench.trace_evidence_evaluator_v3_1 import (
    evaluate_trace_outcome,
)


ROOT = Path(__file__).resolve().parents[2]
CORE_SHA256 = "3cf75ca5045e5e01504597c121a155cbe1ee0086931fa826ff4d8c9b03dec1a2"


class _AlwaysInvalidToolClient:
    def __init__(self):
        self.index = 0

    def complete(self, messages, max_tokens):
        self.index += 1
        return CompletionResult(
            content='{"type":"tool","tool_name":"not_a_tool","arguments":{},"trigger_evidence_id":null}',
            response_id=f"fake-{self.index}",
            returned_model="fake-model",
            system_fingerprint="fake-fingerprint",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            latency_seconds=0.01,
            request_started_at="2026-08-24T00:00:00+00:00",
            request_finished_at="2026-08-24T00:00:01+00:00",
            raw_response={"fake": True},
        )


def _config():
    return {
        "config_sha256": "a" * 64,
        "requested_model": "fake-model",
        "thinking": "disabled",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "max_turns": 12,
        "max_tool_calls": 10,
        "max_cost_units": 10,
        "request_timeout_seconds": 120.0,
        "retry": {"max_attempts": 2, "fixed_backoff_seconds": 0.0},
        "format_max_attempts": 2,
        "guard_max_repair_attempts": 2,
        "pricing": {
            "per_million_tokens": {
                "input_cache_hit": 0.0,
                "input_cache_miss": 0.0,
                "output": 0.0,
            }
        },
    }


def _runtime_failure(reason_code: str):
    return close_runtime_failure_state(
        failure_reason="guard_repair_budget_exhausted",
        last_candidate_hash="b" * 64,
        last_guard_outcome="BLOCK",
        last_reason_code=reason_code,
    )


def test_all_seven_reason_codes_close_without_a_decision():
    for reason in ReasonCode:
        outcome = parse_runtime_execution_failure(_runtime_failure(reason.value).to_dict())
        assert outcome.completion_status == "execution_failure"
        assert outcome.decision is None
        assert outcome.runtime_generated is True


def test_agent_cannot_self_report_execution_failure():
    case = next(
        item
        for item in build_development_cases()
        if item["terminal"]["completion_status"] == "execution_failure"
    )
    result = parse_agent_terminal_runtime_output(
        __import__("json").dumps(case["terminal"]),
        known_evidence_ids=[item["evidence_id"] for item in case["episode"]["evidence"]],
        accepted_tool_calls=[],
    )
    assert result.accepted is False
    assert result.decision is None
    assert result.format_failure == "agent_reported_execution_failure_reserved_for_runtime"


def test_runtime_failure_is_scoreable_conservative_itt_for_feasible_task():
    case = next(
        item for item in build_development_cases()
        if item["episode"]["task_type"] == "mitigation_decision"
    )
    episode = episode_from_dict(case["episode"])
    outcome = parse_terminal_outcome(
        _runtime_failure(ReasonCode.MISSING_REQUIRED_SUPPORT.value).to_dict()
    )
    task = evaluate_terminal_outcome(episode, outcome)
    assert task.terminal_correct is False
    assert task.answered is False
    assert task.execution_failure is True
    assert task.feasible is True
    assert task.feasible_completion is False
    assert task.correct_rejection is False
    aggregate = aggregate_selective_metrics([task], costs=[3.0])
    assert aggregate["coverage"] == 0.0
    assert aggregate["selective_risk"] is None
    assert aggregate["execution_failure_rate"] == 1.0
    assert aggregate["overall_utility"] == -1.03


def test_runtime_failure_is_not_correct_rejection_for_infeasible_task():
    case = next(
        item for item in build_development_cases()
        if item["episode"]["task_type"] == "unreachable_target"
        and item["episode"]["ground_truth"]["acceptable_payload"]["task_predicate"]["expected_unreachable"]
    )
    task = evaluate_terminal_outcome(
        episode_from_dict(case["episode"]),
        _runtime_failure(ReasonCode.PREMATURE_FINALIZATION.value),
    )
    assert task.feasible is False
    assert task.correct_rejection is False
    assert task.terminal_correct is False


def test_runtime_failure_trace_preserves_real_cost_latency_and_intervention():
    case = next(
        item for item in build_development_cases()
        if item["episode"]["task_type"] == "mitigation_decision"
    )
    episode = episode_from_dict(case["episode"])
    call = {
        "call_index": 1,
        "tool_name": "apply_mitigation",
        "request": {"mitigation_policy": "increase_mitigation"},
        "response": {"success": True, "estimated_gain": 0.2, "verified": True},
        "trigger_evidence_id": None,
        "observed_snapshot_id": episode.ground_truth.acceptable_payload["evidence_policy"]["current_snapshot_id"],
        "cost_units": 3,
        "latency_seconds": 0.25,
    }
    result = evaluate_trace_outcome(
        episode=episode,
        outcome=_runtime_failure(ReasonCode.UNVERIFIED_REVALIDATION.value),
        trace_id="runtime-failure-gold",
        accepted_tool_calls=[call],
    )
    assert result.decision_support_applicable is False
    assert result.evidence_state == "not_applicable"
    assert result.actual_tool_call_count == 1
    assert result.actual_cost_units == 3
    assert result.actual_latency_seconds == 0.25
    assert result.intervention_requested is None
    assert result.intervention_attempted is True
    assert result.intervention_executed is True
    assert result.intervention_verified is True


def test_preflight_repair_exhaustion_closes_as_scoreable_runtime_failure():
    episode = generate_development_preflight_episodes(20264011)[0]
    trace = run_preflight_trace(
        episode=episode,
        arm=StudyArm.LEDGER_GUARD,
        schedule_item={
            "schedule_index": 0,
            "episode_id": episode.episode_id,
            "task_type": episode.task_type.value,
            "arm": StudyArm.LEDGER_GUARD.value,
            "hidden_arm_id": "arm-test",
        },
        client=_AlwaysInvalidToolClient(),
        config=_config(),
    )
    assert trace["trace_complete"] is True
    assert trace["terminal_source"] == "runtime"
    assert trace["terminal"]["completion_status"] == "execution_failure"
    assert trace["terminal"]["decision"] is None
    assert trace["runtime_failure_reason"] == "guard_repair_budget_exhausted"
    assert trace["unscorable_reason"] is None
    assert len(trace["guard_events"]) == 3
    assert sum(event["repair_allowed"] for event in trace["guard_events"]) == 2


def test_non_guard_exhaustion_uses_the_same_runtime_failure_envelope():
    episode = generate_development_preflight_episodes(20264011)[0]
    trace = run_preflight_trace(
        episode=episode,
        arm=StudyArm.LEDGER_ONLY,
        schedule_item={
            "schedule_index": 1,
            "episode_id": episode.episode_id,
            "task_type": episode.task_type.value,
            "arm": StudyArm.LEDGER_ONLY.value,
            "hidden_arm_id": "arm-test-nonguard",
        },
        client=_AlwaysInvalidToolClient(),
        config=_config(),
    )
    assert trace["trace_complete"] is True
    assert trace["terminal_source"] == "runtime"
    assert trace["terminal"]["runtime_generated"] is True
    assert trace["terminal"]["decision"] is None
    assert trace["terminal"]["last_guard_outcome"] is None
    assert trace["terminal"]["last_reason_code"] is None
    assert trace["unscorable_reason"] is None


def test_frozen_action_guard_core_hash_is_unchanged():
    path = ROOT / "eval/reliability_bench/action_guard_v1_2.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA256
