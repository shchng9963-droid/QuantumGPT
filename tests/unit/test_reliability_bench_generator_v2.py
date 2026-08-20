"""Semantic regression tests for generator and task predicates v2."""

from __future__ import annotations

from dataclasses import replace

from eval.reliability_bench.generator_v2 import (
    DEVELOPMENT_CONFIG,
    config_sha256,
    generate_method_validation_candidates,
)
from eval.reliability_bench.judges import DecisionSubmission
from eval.reliability_bench.predicates_v2 import evaluate_task_predicate
from eval.reliability_bench.schema import (
    LATEST_SCHEMA_VERSION,
    DriftRelevance,
    TaskType,
    validate_counterfactual_pairs,
    validate_episode,
)


def _episode(task, relevance):
    return next(
        item
        for item in generate_method_validation_candidates(DEVELOPMENT_CONFIG)
        if item.task_type is task and item.drift_event.relevance is relevance
    )


def test_v2_generates_24_matched_heldout_candidates_without_public_truth_leakage():
    episodes = generate_method_validation_candidates(DEVELOPMENT_CONFIG)
    assert len(episodes) == 24
    assert len({item.pair_id for item in episodes}) == 12
    assert {item.schema_version for item in episodes} == {LATEST_SCHEMA_VERSION}
    validate_counterfactual_pairs(episodes)
    for episode in episodes:
        validate_episode(episode)
        assert "task_predicate" not in episode.initial_state
        assert "current_snapshot_id" not in episode.initial_state
        assert "acceptable_qubit_sets" not in episode.ground_truth.acceptable_payload


def test_mapping_accepts_every_safe_set_and_rejects_forbidden_or_wrong_size():
    episode = _episode(TaskType.QUBIT_MAPPING, DriftRelevance.RELATED)
    predicate = episode.ground_truth.acceptable_payload["task_predicate"]
    candidates = predicate["candidate_qubits"]
    forbidden = set(predicate["forbidden_qubits"])
    required = predicate["required_qubit_count"]
    safe_candidates = [item for item in candidates if item not in forbidden]
    first = tuple(safe_candidates[:required])
    second = tuple(safe_candidates[-required:])
    assert first != second
    assert evaluate_task_predicate(
        episode, DecisionSubmission(action="select_qubits", selected_qubits=first)
    ) == []
    assert evaluate_task_predicate(
        episode, DecisionSubmission(action="select_qubits", selected_qubits=second)
    ) == []
    unsafe = (next(iter(forbidden)), *safe_candidates[: required - 1])
    assert evaluate_task_predicate(
        episode, DecisionSubmission(action="select_qubits", selected_qubits=unsafe)
    ) == ["unsafe_qubit_mapping"]
    assert evaluate_task_predicate(
        episode,
        DecisionSubmission(action="select_qubits", selected_qubits=first[:-1]),
    ) == ["unsafe_qubit_mapping"]


def test_snapshot_lineage_uses_registered_real_ids_not_symbolic_aliases():
    related = _episode(TaskType.TRANSPILATION, DriftRelevance.RELATED)
    unrelated = _episode(TaskType.TRANSPILATION, DriftRelevance.UNRELATED)
    for episode in (related, unrelated):
        payload = episode.ground_truth.acceptable_payload
        predicate = payload["task_predicate"]
        required = predicate["required_snapshot_id"]
        registered = payload["evidence_policy"]["registered_source_ids"]
        assert required in registered
        assert required not in {"pre-drift", "post-drift", "current"}
        assert evaluate_task_predicate(
            episode,
            DecisionSubmission(
                action=episode.ground_truth.acceptable_actions[0],
                compilation_snapshot_id=required,
            ),
        ) == []
        assert evaluate_task_predicate(
            episode,
            DecisionSubmission(
                action=episode.ground_truth.acceptable_actions[0],
                compilation_snapshot_id="unknown-snapshot",
            ),
        ) == ["outdated_compilation"]


def test_initial_snapshot_is_a_registered_evidence_source_with_current_descendant():
    for episode in generate_method_validation_candidates(DEVELOPMENT_CONFIG):
        policy = episode.ground_truth.acceptable_payload["evidence_policy"]
        initial = episode.initial_state["snapshot_id"]
        current = policy["current_snapshot_id"]
        assert initial in policy["registered_source_ids"]
        assert current in policy["registered_source_ids"]
        lineage = {item["snapshot_id"]: item for item in policy["snapshot_lineage"]}
        assert lineage[initial] == {
            "snapshot_id": initial,
            "phase": "pre_drift",
            "parent_snapshot_id": None,
        }
        assert lineage[current]["parent_snapshot_id"] == initial
        assert lineage[current]["phase"] == "current"


def test_v11_schema_rejects_unregistered_initial_snapshot_boundary_case():
    episode = _episode(TaskType.UNREACHABLE_TARGET, DriftRelevance.UNRELATED)
    payload = dict(episode.ground_truth.acceptable_payload)
    policy = dict(payload["evidence_policy"])
    policy["registered_source_ids"] = [
        item
        for item in policy["registered_source_ids"]
        if item != episode.initial_state["snapshot_id"]
    ]
    payload["evidence_policy"] = policy
    broken = replace(
        episode,
        ground_truth=replace(episode.ground_truth, acceptable_payload=payload),
    )
    try:
        validate_episode(broken)
    except ValueError as error:
        assert "registered sources" in str(error)
    else:
        raise AssertionError("unregistered initial snapshot was accepted")


def test_generator_config_hash_is_deterministic_and_sensitive():
    assert config_sha256(DEVELOPMENT_CONFIG) == config_sha256(DEVELOPMENT_CONFIG)
    changed = replace(
        DEVELOPMENT_CONFIG, random_seed=DEVELOPMENT_CONFIG.random_seed + 1
    )
    assert config_sha256(changed) != config_sha256(DEVELOPMENT_CONFIG)
