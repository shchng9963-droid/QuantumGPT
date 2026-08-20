from __future__ import annotations

import base64
import json
from dataclasses import asdict
from pathlib import Path

from eval.reliability_bench.b1 import CompletionResult, run_b1_trace
from eval.reliability_bench.b0 import _public_input, _tool_output
from eval.reliability_bench.generator import generate_stage_a_episodes
from eval.reliability_bench.judge_audit_package_v2 import (
    PREDICTION_MAGIC,
    build_blinded_review_package,
    seal_judge_predictions,
)
from eval.reliability_bench.judge_audit_v2 import (
    AUDIT_EPISODE_COUNT,
    AUDIT_VERSION,
    assigned_controller,
    audit_deduplication,
    build_audit_schedule,
    generate_judge_audit_episodes,
    semantic_fingerprint,
    validate_audit_config,
)
from eval.reliability_bench.groups import ExperimentGroup
from eval.reliability_bench.schema import DriftRelevance, TaskType


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "eval/reliability_bench/data/judge_audit_v2_config.json"


class _FinalAuditClient:
    def __init__(self, episode):
        self.episode = episode

    def complete(self, messages, max_tokens):
        payload = {"type": "final", "decision": _decision(self.episode)}
        content = json.dumps(payload)
        return CompletionResult(
            content=content,
            response_id="audit-fake-response",
            returned_model="deepseek-v4-flash",
            system_fingerprint="audit-fake-fingerprint",
            finish_reason="stop",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 100,
            },
            latency_seconds=0.01,
            request_started_at="2026-08-20T00:00:00+00:00",
            request_finished_at="2026-08-20T00:00:00.010000+00:00",
            raw_response={
                "id": "audit-fake-response",
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": content}}],
            },
        )


def _decision(episode):
    predicate = episode.ground_truth.acceptable_payload["task_predicate"]
    common = {
        "used_artifact_ids": [],
        "revalidation_actions": [],
        "selected_backend": None,
        "selected_qubits": [],
        "compilation_snapshot_id": None,
        "claimed_success": None,
        "supporting_artifact_ids": [],
        "mitigation_action": None,
        "declared_unreachable": None,
        "metadata": {},
    }
    if episode.task_type is TaskType.BACKEND_SELECTION:
        return {**common, "action": "select_backend", "selected_backend": predicate["acceptable_backends"][0]}
    if episode.task_type is TaskType.QUBIT_MAPPING:
        forbidden = set(predicate["forbidden_qubits"])
        selected = [q for q in predicate["candidate_qubits"] if q not in forbidden][: predicate["required_qubit_count"]]
        return {**common, "action": "select_qubits", "selected_qubits": selected}
    if episode.task_type is TaskType.TRANSPILATION:
        return {
            **common,
            "action": "retranspile" if predicate["required_phase"] == "current" else "reuse_compilation",
            "compilation_snapshot_id": predicate["required_snapshot_id"],
        }
    if episode.task_type is TaskType.FIDELITY_CLAIM:
        return {
            **common,
            "action": "rerun_circuit" if episode.drift_event.relevance is DriftRelevance.RELATED else "reuse_valid_result",
            "claimed_success": predicate["expected_success_claim"],
        }
    if episode.task_type is TaskType.MITIGATION_DECISION:
        return {
            **common,
            "action": "reassess_mitigation" if episode.drift_event.relevance is DriftRelevance.RELATED else "keep_mitigation",
            "mitigation_action": predicate["acceptable_mitigation_actions"][0],
        }
    expected = predicate["expected_unreachable"]
    return {
        **common,
        "action": "declare_unreachable" if expected else "continue_execution",
        "declared_unreachable": expected,
    }


def _trace(episode, index):
    public = _public_input(episode)
    primary = episode.evidence[0].evidence_id
    calls = []
    for call_index, name in enumerate(episode.ground_truth.required_revalidation_actions, start=1):
        calls.append(
            {
                "call_index": call_index,
                "turn": call_index,
                "tool_name": name,
                "trigger_evidence_id": primary,
                "request": {"run_id": public["run_id"]},
                "response": _tool_output(name, episode, True),
                "cost_units": 1,
            }
        )
    return {
        "trace_id": f"audit-test-trace-{index:02d}",
        "episode": {
            "episode_id": episode.episode_id,
            "task_type": episode.task_type.value,
            "relevance": episode.drift_event.relevance.value,
        },
        "controller": {"group": assigned_controller(episode)},
        "public_prompt": public["actual_prompt"],
        "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": public["actual_prompt"]}],
        "tool_calls": calls,
        "model_final_decision": _decision(episode),
        "budget_usage": {"turns": len(calls) + 1, "tool_calls": len(calls), "cost_units": len(calls), "completion_tokens": 10},
        "llm": {"parser_errors": [], "api_errors": []},
        "trace_complete": True,
        "unscorable": False,
    }


def test_audit_generator_is_balanced_valid_and_deterministic():
    first = generate_judge_audit_episodes()
    second = generate_judge_audit_episodes()
    assert len(first) == AUDIT_EPISODE_COUNT
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len({item.episode_id for item in first}) == 24
    for task in TaskType:
        selected = [item for item in first if item.task_type is task]
        assert len(selected) == 4
        assert sum(item.drift_event.relevance is DriftRelevance.RELATED for item in selected) == 2


def test_audit_semantics_do_not_overlap_old_or_method_generator_structure():
    episodes = generate_judge_audit_episodes()
    report = audit_deduplication(episodes, generate_stage_a_episodes())
    assert report["passed"] is True
    assert report["method_validation_archive_accessed"] is False
    assert report["retired_audit_id_overlap"] == []
    assert report["retired_audit_semantic_hash_overlap"] == []
    assert len({semantic_fingerprint(item) for item in episodes}) == 24


def test_schedule_and_controller_assignment_are_frozen_before_results():
    episodes = generate_judge_audit_episodes()
    first = build_audit_schedule(episodes)
    second = build_audit_schedule(episodes)
    assert first == second
    assert len(first) == 24
    assert {row["controller_group"] for row in first} == {"react", "ledger_only", "full_selective"}
    assert [row["episode_id"] for row in first] != sorted(row["episode_id"] for row in first)


def test_frozen_config_contains_three_layer_gates_and_failure_policy():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_audit_config(config)
    assert config["protocol_version"] == AUDIT_VERSION
    assert config["thresholds"]["outcome"]["minimum_agreements_out_of_24"] == 22
    assert config["thresholds"]["critical_labels"]["f1_min"] == 0.9
    assert config["legacy_program_judge_enabled"] is False
    assert config["class_balance"]["post_hoc_resampling_allowed"] is False
    assert config["failure_policy"]["replacement_requires_new_episodes"] is True


def test_b1_trace_disables_incompatible_development_judge_v1():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    episode = next(
        item
        for item in generate_judge_audit_episodes()
        if item.task_type is TaskType.UNREACHABLE_TARGET
    )
    schedule_item = next(
        row for row in build_audit_schedule([*generate_judge_audit_episodes()])
        if row["episode_id"] == episode.episode_id
    )
    group = ExperimentGroup(schedule_item["controller_group"])
    trace = run_b1_trace(
        episode,
        group,
        _FinalAuditClient(episode),
        config,
        schedule_item,
        {
            "mode": "independent_judge_audit",
            "config_sha256": "audit-config-test",
            "benchmark_sha256": "audit-benchmark-test",
            "git": {"commit": "audit-commit-test"},
        },
    )
    assert trace["trace_complete"] is True
    assert trace["program_judge"] is None
    assert trace["frozen_context"]["legacy_program_judge_enabled"] is False


def test_tool_trace_uses_episode_candidate_qubits_not_legacy_fixed_five():
    episode = next(
        item
        for item in generate_judge_audit_episodes()
        if item.task_type is TaskType.QUBIT_MAPPING
    )
    response = _tool_output("get_qubit_properties", episode, True)
    assert response["available_qubits"] == episode.task_constraints["candidate_qubits"]


def test_predictions_are_sealed_before_blind_packages_and_never_written_plaintext(tmp_path):
    episodes = generate_judge_audit_episodes()
    by_id = {episode.episode_id: episode for episode in episodes}
    traces = [_trace(episode, index) for index, episode in enumerate(episodes)]
    prediction_dir = tmp_path / "predictions"
    key_path = tmp_path / "keys" / "predictions.key"
    manifest = seal_judge_predictions(
        traces=traces,
        episodes=by_id,
        output_dir=prediction_dir,
        key_path=key_path,
        judge_candidate_commit="e2ceb18a610436f01b5f572a3dc75a36816b6c3b",
        judge_source_path=ROOT / "eval/reliability_bench/trace_judge_v2.py",
        audit_config_path=CONFIG,
        generated_at="2026-08-20T00:00:00+00:00",
    )
    archive = (prediction_dir / "judge_predictions.jsonl.aesgcm").read_bytes()
    assert archive.startswith(PREDICTION_MAGIC)
    assert len(base64.urlsafe_b64decode(key_path.read_bytes().strip())) == 32
    assert not (prediction_dir / "judge_predictions.jsonl").exists()
    assert manifest["human_annotations_seen"] is False

    codebook = {"codebook_version": "0.2-frozen", "language": "zh-CN"}
    package = build_blinded_review_package(
        traces=traces,
        output_dir=tmp_path / "blind",
        frozen_codebook=codebook,
        prediction_commitment=manifest,
        audit_config_sha256="test-config-hash",
        generated_at="2026-08-20T00:01:00+00:00",
    )
    assert package["judge_predictions_in_reviewer_files"] is False
    for slot in ("A", "B"):
        rows = [
            json.loads(line)
            for line in (tmp_path / "blind" / f"给标注者{slot}" / f"标注者{slot}_待标注.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 24
        serialized = json.dumps(rows, ensure_ascii=False)
        assert "controller_group" not in serialized
        assert "program_judge" not in serialized
        assert "judge_predictions" not in serialized
