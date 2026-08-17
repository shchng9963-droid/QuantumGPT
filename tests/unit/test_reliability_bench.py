"""Stage-A contract tests for ReliabilityBench-Q."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from eval.reliability_bench.generator import (
    generate_stage_a_episodes,
    write_stage_a_dataset,
)
from eval.reliability_bench.groups import (
    FairnessContract,
    GROUP_CONFIGS,
    ExperimentGroup,
    validate_group_matrix,
)
from eval.reliability_bench.judges import (
    counterexample_submission,
    judge_episode,
    reference_submission,
)
from eval.reliability_bench.mechanisms import (
    build_stage_a_mechanism_audit,
    run_invalidation_mechanism,
)
from eval.reliability_bench.schema import (
    DriftRelevance,
    TaskType,
    episode_from_dict,
    validate_counterfactual_pairs,
    validate_episode,
)


def test_stage_a_generates_120_unique_matched_episodes():
    episodes = generate_stage_a_episodes()
    assert len(episodes) == 120
    assert len({item.episode_id for item in episodes}) == 120
    assert len({item.pair_id for item in episodes}) == 60
    validate_counterfactual_pairs(episodes)


def test_stage_a_is_balanced_by_task_and_relevance():
    episodes = generate_stage_a_episodes()
    by_task = Counter(item.task_type for item in episodes)
    by_relevance = Counter(item.drift_event.relevance for item in episodes)
    assert by_task == {task: 20 for task in TaskType}
    assert by_relevance == {
        DriftRelevance.RELATED: 60,
        DriftRelevance.UNRELATED: 60,
    }


def test_related_drift_invalidates_only_dependent_evidence():
    for episode in generate_stage_a_episodes():
        validate_episode(episode)
        if episode.drift_event.relevance is DriftRelevance.RELATED:
            assert episode.ground_truth.invalidated_artifact_ids
            assert episode.ground_truth.preserved_artifact_ids
        else:
            assert episode.ground_truth.invalidated_artifact_ids == ()
            assert len(episode.ground_truth.preserved_artifact_ids) == len(
                episode.evidence
            )


def test_schema_round_trip_is_lossless():
    episode = generate_stage_a_episodes()[0]
    restored = episode_from_dict(episode.to_dict())
    assert restored == episode


def test_reference_decisions_pass_all_six_task_judges():
    for episode in generate_stage_a_episodes():
        result = judge_episode(episode, reference_submission(episode))
        assert result.correct, (episode.episode_id, result.error_codes)


def test_counterexamples_fail_all_six_task_judges():
    observed = set()
    for episode in generate_stage_a_episodes():
        result = judge_episode(episode, counterexample_submission(episode))
        assert not result.correct
        observed.add(episode.task_type)
    assert observed == set(TaskType)


def test_related_counterexamples_trigger_each_task_specific_error():
    expected_error = {
        TaskType.BACKEND_SELECTION: "unacceptable_backend",
        TaskType.QUBIT_MAPPING: "unsafe_qubit_mapping",
        TaskType.TRANSPILATION: "outdated_compilation",
        TaskType.FIDELITY_CLAIM: "incorrect_success_claim",
        TaskType.MITIGATION_DECISION: "incorrect_mitigation_decision",
        TaskType.UNREACHABLE_TARGET: "incorrect_reachability_decision",
    }
    related = [
        item
        for item in generate_stage_a_episodes()
        if item.drift_event.relevance is DriftRelevance.RELATED
    ]
    for task, error_code in expected_error.items():
        episode = next(item for item in related if item.task_type is task)
        result = judge_episode(episode, counterexample_submission(episode))
        assert error_code in result.error_codes
        if task is TaskType.FIDELITY_CLAIM:
            assert result.stale_evidence_reuse is True
            assert result.unsupported_success_claim is True


def test_group_matrix_and_neutral_event_fairness_contract():
    validate_group_matrix()
    assert GROUP_CONFIGS[ExperimentGroup.REACT].sees_neutral_drift_event is False
    assert all(
        GROUP_CONFIGS[group].sees_neutral_drift_event
        for group in set(ExperimentGroup) - {ExperimentGroup.REACT}
    )
    event = generate_stage_a_episodes()[0].drift_event.neutral_payload()
    assert tuple(event) == FairnessContract().neutral_event_fields
    rendered = json.dumps(event).lower()
    for forbidden in ("rerun", "revalidate", "invalidated", "stale", "action_required"):
        assert forbidden not in rendered


def test_selective_and_oracle_match_ground_truth_while_global_over_invalidates():
    episodes = generate_stage_a_episodes()
    audit = build_stage_a_mechanism_audit(episodes)
    assert audit["acceptance_passed"] is True
    full = audit["groups"][ExperimentGroup.FULL_SELECTIVE.value]
    oracle = audit["groups"][ExperimentGroup.ORACLE.value]
    global_group = audit["groups"][ExperimentGroup.GLOBAL_REVALIDATE.value]
    assert full["precision"] == full["recall"] == 1.0
    assert oracle["precision"] == oracle["recall"] == 1.0
    assert full["false_invalidation_rate"] == 0.0
    assert global_group["false_positive"] > 0

    for episode in episodes:
        full_outcome = run_invalidation_mechanism(
            ExperimentGroup.FULL_SELECTIVE, episode
        )
        global_outcome = run_invalidation_mechanism(
            ExperimentGroup.GLOBAL_REVALIDATE, episode
        )
        assert set(full_outcome.invalidated_artifact_ids).issubset(
            global_outcome.invalidated_artifact_ids
        )


def test_dataset_writer_is_deterministic_and_manifest_is_verifiable(tmp_path):
    output = tmp_path / "episodes.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = write_stage_a_dataset(output, manifest_path)
    payload = output.read_bytes()
    assert manifest["episode_count"] == 120
    assert manifest["pair_count"] == 60
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 120
    assert json.loads(manifest_path.read_text()) == manifest


def test_tracked_stage_a_dataset_matches_generator():
    root = Path(__file__).resolve().parents[2]
    data_path = root / "eval/reliability_bench/data/stage_a_synthetic_v1.jsonl"
    manifest_path = root / "eval/reliability_bench/data/stage_a_manifest.json"
    audit_path = root / "eval/reliability_bench/data/stage_a_mechanism_audit.json"
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    generated = [item.to_dict() for item in generate_stage_a_episodes()]
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert rows == generated
    assert manifest["episode_count"] == 120
    assert manifest["pair_count"] == 60
    assert manifest["sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert audit == build_stage_a_mechanism_audit(generate_stage_a_episodes())
    assert audit["acceptance_passed"] is True
