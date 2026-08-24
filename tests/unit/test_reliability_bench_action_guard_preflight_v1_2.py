from __future__ import annotations

from collections import Counter

from eval.reliability_bench.action_guard_preflight_v1_2 import (
    DEVELOPMENT_NAMESPACE,
    SHARED_AGENT_SYSTEM_PROMPT_SHA256,
    build_schedule,
    controller_belief_state,
    generate_development_preflight_episodes,
    public_episode_prompt,
)
from eval.reliability_bench.public_runtime_v3 import StudyArm


def test_preflight_has_six_counterfactual_pairs_and_72_runs():
    episodes = generate_development_preflight_episodes(20263001)
    assert len(episodes) == 12
    assert len({item.task_type for item in episodes}) == 6
    assert Counter(item.drift_event.relevance.value for item in episodes) == {
        "related": 6,
        "unrelated": 6,
    }
    assert all(item.episode_id.startswith(DEVELOPMENT_NAMESPACE) for item in episodes)
    assert all(item.source == "action_guard_v1_2_real_llm_preflight_source_v3" for item in episodes)
    schedule = build_schedule(episodes, 20263002)
    assert len(schedule) == 72
    assert Counter(item["arm"] for item in schedule) == {arm.value: 12 for arm in StudyArm}
    assert Counter(item["episode_id"] for item in schedule) == {
        episode.episode_id: 6 for episode in episodes
    }


def test_core_2x2_belief_boundary_is_present_in_every_development_task():
    for episode in generate_development_preflight_episodes(20263001):
        ledger = controller_belief_state(episode, StudyArm.LEDGER_GUARD)
        full = controller_belief_state(episode, StudyArm.FULL_GUARD)
        assert {item.belief_validity.value for item in ledger.records} == {"believed_valid"}
        if episode.drift_event.relevance.value == "related":
            assert any(
                item.belief_validity.value == "believed_invalid"
                for item in full.records
            )
        else:
            assert {
                item.belief_validity.value for item in full.records
            } == {"believed_valid"}


def test_public_prompt_excludes_hidden_ground_truth_payload():
    forbidden = {
        "ground_truth",
        "acceptable_actions",
        "acceptable_payload",
        "task_predicate",
        "invalidated_artifact_ids",
        "required_revalidation_actions",
        "affected_resources",
        "failure_reason",
    }
    for episode in generate_development_preflight_episodes(20263001):
        prompt = public_episode_prompt(episode)
        assert not any(token in prompt for token in forbidden)


def test_shared_prompt_hash_is_frozen_and_nonempty():
    assert len(SHARED_AGENT_SYSTEM_PROMPT_SHA256) == 64
    int(SHARED_AGENT_SYSTEM_PROMPT_SHA256, 16)
