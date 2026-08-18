"""Offline contract tests for the ReliabilityBench-Q B1 LLM runner."""

from __future__ import annotations

import json
from pathlib import Path

from eval.reliability_bench.b1 import (
    CompletionResult,
    _private_controller_state,
    build_b1_report,
    build_schedule,
    estimate_cost_usd,
    load_b1_config,
    run_b1_trace,
    select_frozen_episodes,
)
from eval.reliability_bench.b0 import B0_GROUPS, _public_input
from eval.reliability_bench.generator import generate_stage_a_episodes
from eval.reliability_bench.groups import ExperimentGroup


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "eval/reliability_bench/data/b1_config.json"


class FinalOnlyClient:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, max_tokens):
        self.calls += 1
        decision = {
            "type": "final",
            "decision": {
                "action": "select_backend",
                "used_artifact_ids": [],
                "revalidation_actions": [],
                "selected_backend": "FakeBrisbane",
                "selected_qubits": [],
                "compilation_snapshot_id": None,
                "claimed_success": None,
                "supporting_artifact_ids": [],
                "mitigation_action": None,
                "declared_unreachable": None,
                "metadata": {},
            },
        }
        raw = {
            "id": f"fake-{self.calls}",
            "model": "deepseek-v4-flash",
            "choices": [{"message": {"content": json.dumps(decision)}}],
        }
        return CompletionResult(
            content=json.dumps(decision),
            response_id=f"fake-{self.calls}",
            returned_model="deepseek-v4-flash",
            system_fingerprint="fp-test",
            finish_reason="stop",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
            latency_seconds=0.01,
            request_started_at="2026-08-18T00:00:00+00:00",
            request_finished_at="2026-08-18T00:00:00.010000+00:00",
            raw_response=raw,
        )


class InvalidThenCorrectedClient(FinalOnlyClient):
    def __init__(self, invalid_evidence_id):
        super().__init__()
        self.invalid_evidence_id = invalid_evidence_id

    def complete(self, messages, max_tokens):
        result = super().complete(messages, max_tokens)
        payload = json.loads(result.content)
        if self.calls == 1:
            payload["decision"]["used_artifact_ids"] = [self.invalid_evidence_id]
        return CompletionResult(
            **{
                **result.__dict__,
                "content": json.dumps(payload),
                "raw_response": {
                    **result.raw_response,
                    "choices": [{"message": {"content": json.dumps(payload)}}],
                },
            }
        )


class CorrectFullClient(FinalOnlyClient):
    def __init__(self, primary_id, context_id):
        super().__init__()
        self.primary_id = primary_id
        self.context_id = context_id

    def complete(self, messages, max_tokens):
        if self.calls < 2:
            self.calls += 1
            tool_name = "compare_backends" if self.calls == 1 else "get_backend_health"
            payload = {
                "type": "tool",
                "tool_name": tool_name,
                "trigger_evidence_id": self.primary_id,
            }
            return CompletionResult(
                content=json.dumps(payload),
                response_id=f"fake-{self.calls}",
                returned_model="deepseek-v4-flash",
                system_fingerprint="fp-test",
                finish_reason="stop",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "total_tokens": 140,
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 100,
                },
                latency_seconds=0.01,
                request_started_at="2026-08-18T00:00:00+00:00",
                request_finished_at="2026-08-18T00:00:00.010000+00:00",
                raw_response={
                    "id": f"fake-{self.calls}",
                    "model": "deepseek-v4-flash",
                    "choices": [{"message": {"content": json.dumps(payload)}}],
                },
            )
        result = super().complete(messages, max_tokens)
        payload = json.loads(result.content)
        payload["decision"]["selected_backend"] = "FakeSherbrooke"
        payload["decision"]["used_artifact_ids"] = [self.context_id]
        return CompletionResult(
            **{
                **result.__dict__,
                "content": json.dumps(payload),
                "raw_response": {
                    **result.raw_response,
                    "choices": [{"message": {"content": json.dumps(payload)}}],
                },
            }
        )


def _config():
    return load_b1_config(CONFIG_PATH)


def _episode():
    config = _config()
    selected = select_frozen_episodes(generate_stage_a_episodes(), config)
    return next(
        item for item in selected if item.episode_id == config["preflight_episode_id"]
    )


def _manifest(config, mode="preflight"):
    return {
        "mode": mode,
        "config": config,
        "config_sha256": "config-test",
        "benchmark_sha256": "benchmark-test",
        "git": {"commit": "commit-test"},
    }


def test_b1_config_freezes_24_episodes_and_exactly_six_groups():
    config = _config()
    episodes = select_frozen_episodes(generate_stage_a_episodes(), config)
    assert len(episodes) == 24
    assert config["groups"] == [item.value for item in B0_GROUPS]
    assert "prompt_oracle" not in config["groups"]


def test_b1_schedule_is_interleaved_and_re_randomizes_each_episode():
    config = _config()
    schedule = build_schedule(config, preflight=False)
    assert len(schedule) == 144
    chunks = [schedule[index : index + 6] for index in range(0, 144, 6)]
    expected = set(config["groups"])
    assert all(len({item["episode_id"] for item in chunk}) == 1 for chunk in chunks)
    assert all(
        {item["controller_group"] for item in chunk} == expected for chunk in chunks
    )
    assert (
        len({tuple(item["controller_group"] for item in chunk) for chunk in chunks}) > 1
    )
    assert schedule == build_schedule(config, preflight=False)


def test_full_selective_derives_state_without_oracle_channel():
    episode = _episode()
    public = _public_input(episode)
    full_state, full_invalid, full_oracle = _private_controller_state(
        episode, ExperimentGroup.FULL_SELECTIVE, public
    )
    oracle_state, oracle_invalid, oracle_used = _private_controller_state(
        episode, ExperimentGroup.ORACLE, public
    )
    assert full_invalid == episode.ground_truth.invalidated_artifact_ids
    assert oracle_invalid == episode.ground_truth.invalidated_artifact_ids
    assert full_oracle is False
    assert oracle_used is True
    assert "ground_truth" not in json.dumps(full_state)
    assert full_state == oracle_state


def test_b1_trace_is_complete_scoreable_and_preserves_actual_tool_actions():
    config = _config()
    episode = _episode()
    item = build_schedule(config, preflight=True)[0]
    group = ExperimentGroup(item["controller_group"])
    trace = run_b1_trace(
        episode,
        group,
        FinalOnlyClient(),
        config,
        item,
        _manifest(config),
        sleep_fn=lambda _: None,
    )
    assert trace["trace_complete"] is True
    assert trace["unscorable"] is False
    assert trace["program_judge"] is not None
    assert trace["budget_usage"]["turns"] == 1
    assert trace["budget_usage"]["tool_calls"] == 0
    assert trace["model_final_decision"]["revalidation_actions"] == []
    assert trace["frozen_context"]["temperature"] == 0.0
    assert trace["llm"]["returned_model_ids"] == ["deepseek-v4-flash"]


def test_full_controller_rejects_invalid_pre_event_evidence_reference():
    config = _config()
    episode = _episode()
    item = next(
        item
        for item in build_schedule(config, preflight=True)
        if item["controller_group"] == "full_selective"
    )
    invalid_id = episode.ground_truth.invalidated_artifact_ids[0]
    client = InvalidThenCorrectedClient(invalid_id)
    trace = run_b1_trace(
        episode,
        ExperimentGroup.FULL_SELECTIVE,
        client,
        config,
        item,
        _manifest(config),
        sleep_fn=lambda _: None,
    )
    assert client.calls == 2
    assert trace["trace_complete"] is True
    assert trace["model_final_decision"]["used_artifact_ids"] == []
    assert trace["llm"]["controller_validation_errors"] == [
        {
            "turn": 1,
            "error": "invalid_pre_event_evidence_reference",
            "invalid_evidence_ids": [invalid_id],
            "instruction": (
                "These IDs identify invalid pre-event artifacts. Current tool "
                "observations are already recorded by the controller; remove the "
                "invalid IDs from used_artifact_ids and supporting_artifact_ids, "
                "and do not invent replacement IDs."
            ),
        }
    ]


def test_preflight_report_accepts_six_complete_offline_traces():
    config = _config()
    episode = _episode()
    manifest = _manifest(config)
    schedule = build_schedule(config, preflight=True)
    traces = []
    for item in schedule:
        if item["controller_group"] == "full_selective":
            primary_id = episode.ground_truth.invalidated_artifact_ids[0]
            context_id = episode.ground_truth.preserved_artifact_ids[0]
            client = CorrectFullClient(primary_id, context_id)
        else:
            client = FinalOnlyClient()
        traces.append(
            run_b1_trace(
                episode,
                ExperimentGroup(item["controller_group"]),
                client,
                config,
                item,
                manifest,
                sleep_fn=lambda _: None,
            )
        )
    report = build_b1_report(traces, manifest, preflight=True)
    assert report["run_count"] == 6
    assert report["acceptance_passed"] is True
    assert report["prompt_and_label_audit"]["passed"] is True
    assert report["acceptance"]["single_returned_model_id"] is True
    assert report["acceptance"]["returned_model_matches_request"] is True


def test_deepseek_v4_flash_cost_uses_frozen_cache_rates():
    pricing = _config()["pricing"]
    result = estimate_cost_usd(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "prompt_cache_hit_tokens": 200,
            "prompt_cache_miss_tokens": 800,
        },
        pricing,
    )
    expected = (200 * 0.0028 + 800 * 0.14 + 500 * 0.28) / 1_000_000
    assert result["estimated_cost"] == round(expected, 10)
