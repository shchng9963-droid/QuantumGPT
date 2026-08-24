from __future__ import annotations

from collections import Counter

from eval.reliability_bench.action_guard_preflight_v1_1 import (
    DEVELOPMENT_NAMESPACE,
    SHARED_AGENT_SYSTEM_PROMPT_SHA256,
    build_schedule,
    controller_belief_state,
    generate_development_preflight_episodes,
)
from eval.reliability_bench.public_runtime_v2 import StudyArm


def test_preflight_has_six_namespace_isolated_tasks_and_36_runs():
    episodes = generate_development_preflight_episodes(20260921)
    assert len(episodes) == 6
    assert len({item.task_type for item in episodes}) == 6
    assert all(item.episode_id.startswith(DEVELOPMENT_NAMESPACE) for item in episodes)
    assert all(item.source == "action_guard_v1_1_real_llm_preflight_source" for item in episodes)
    schedule = build_schedule(episodes, 20260922)
    assert len(schedule) == 36
    assert Counter(item["arm"] for item in schedule) == {arm.value: 6 for arm in StudyArm}
    assert Counter(item["episode_id"] for item in schedule) == {
        episode.episode_id: 6 for episode in episodes
    }


def test_core_2x2_belief_boundary_is_present_in_every_development_task():
    for episode in generate_development_preflight_episodes(20260921):
        ledger = controller_belief_state(episode, StudyArm.LEDGER_GUARD)
        full = controller_belief_state(episode, StudyArm.FULL_GUARD)
        assert {item.belief_validity.value for item in ledger.records} == {"believed_valid"}
        assert any(item.belief_validity.value == "believed_invalid" for item in full.records)


def test_shared_prompt_hash_is_frozen_and_nonempty():
    assert len(SHARED_AGENT_SYSTEM_PROMPT_SHA256) == 64
    int(SHARED_AGENT_SYSTEM_PROMPT_SHA256, 16)
