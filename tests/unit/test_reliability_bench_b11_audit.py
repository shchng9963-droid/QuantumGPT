"""Tests for the blinded B1.1 failure-taxonomy calibration package."""

from __future__ import annotations

import json
from collections import Counter

from eval.reliability_bench.b11_audit import (
    PRIMARY_FAILURE_STAGES,
    SECONDARY_ERROR_TAGS,
    build_calibration_package,
    empty_annotation,
    validate_annotation,
    validate_reviewer_blinding,
    write_calibration_package,
)


TASKS = (
    "backend_selection",
    "fidelity_claim",
    "mitigation_decision",
    "qubit_mapping",
    "transpilation",
    "unreachable_target",
)
GROUPS = (
    "react",
    "monitor_only",
    "ledger_only",
    "global_revalidate",
    "full_selective",
    "oracle",
)


def _fake_traces():
    traces = []
    index = 0
    for task_index, task in enumerate(TASKS):
        for relevance in ("related", "unrelated"):
            for episode_copy in range(2):
                episode_id = f"hidden-{task_index}-{relevance}-{episode_copy}"
                for group_index, group in enumerate(GROUPS):
                    index += 1
                    correct = (task_index + group_index + episode_copy) % 3 == 0
                    public = {
                        "input": {
                            "task": {"type": task},
                            "device_change": {"changed_features": ["feature-x"]},
                            "evidence_refs": [
                                {"evidence_id": f"e-{task_index}-{episode_copy}"}
                            ],
                        }
                    }
                    traces.append(
                        {
                            "trace_id": f"trace-{index:03d}",
                            "episode": {
                                "episode_id": episode_id,
                                "task_type": task,
                                "relevance": relevance,
                            },
                            "controller": {"group": group},
                            "public_prompt": json.dumps(public),
                            "messages": [
                                {"role": "system", "content": "instructions"},
                                {"role": "user", "content": json.dumps(public)},
                                {
                                    "role": "system",
                                    "content": json.dumps(
                                        {"controller_state": {"evidence_ledger": []}}
                                    ),
                                },
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {
                                            "type": "final",
                                            "decision": {"action": "stop"},
                                        }
                                    ),
                                },
                            ],
                            "tool_calls": [],
                            "model_final_decision": {"action": "stop"},
                            "budget_usage": {"turns": 1, "tool_calls": 0},
                            "llm": {"parser_errors": [], "api_errors": []},
                            "trace_complete": True,
                            "unscorable": False,
                            "program_judge": {
                                "correct": correct,
                                "error_codes": [] if correct else ["synthetic_error"],
                            },
                        }
                    )
    assert len(traces) == 144
    return traces


def test_calibration_selection_is_balanced_and_contains_controls():
    package = build_calibration_package(_fake_traces())
    key = package["study_manager_key"]
    assert len(key) == 24
    assert Counter(row["task_type"] for row in key) == {task: 4 for task in TASKS}
    assert Counter(row["relevance"] for row in key) == {
        "related": 12,
        "unrelated": 12,
    }
    assert Counter(row["group"] for row in key) == {group: 4 for group in GROUPS}
    assert 6 <= sum(row["program_correct"] for row in key) <= 10
    assert {row["task_type"] for row in key if not row["program_correct"]} == set(TASKS)


def test_reviewer_rows_are_blinded_and_independently_shuffled():
    package = build_calibration_package(_fake_traces())
    reviewer_a = package["reviewer_a"]
    reviewer_b = package["reviewer_b"]
    assert validate_reviewer_blinding(reviewer_a)["passed"] is True
    assert validate_reviewer_blinding(reviewer_b)["passed"] is True
    assert [row["item_id"] for row in reviewer_a] != [
        row["item_id"] for row in reviewer_b
    ]
    assert all(row["annotation"] == empty_annotation() for row in reviewer_a)
    assert all("program_judge" not in row["scene"] for row in reviewer_a + reviewer_b)


def test_annotation_contract_enforces_one_primary_failure_stage():
    correct = {
        "annotator_id": "reviewer-a",
        "trajectory_outcome": "correct",
        "primary_failure_stage": None,
        "secondary_error_tags": ["irrelevant_tool_call"],
        "confidence": 4,
        "rationale": "最终动作正确，但存在无关查询。",
    }
    assert validate_annotation(correct, complete=True) == []

    incorrect = {
        **correct,
        "trajectory_outcome": "incorrect",
        "primary_failure_stage": "Revalidation",
        "secondary_error_tags": ["missing_required_revalidation"],
    }
    assert validate_annotation(incorrect, complete=True) == []
    assert set(PRIMARY_FAILURE_STAGES)
    assert set(SECONDARY_ERROR_TAGS)

    invalid = {**incorrect, "primary_failure_stage": None}
    assert "primary_failure_stage" in validate_annotation(invalid, complete=True)


def test_written_manifest_hashes_every_draft_file(tmp_path):
    output = tmp_path / "audit"
    manifest = write_calibration_package(
        _fake_traces(), output, generated_at="2026-08-18T00:00:00+00:00"
    )
    assert manifest["codebook_frozen"] is False
    assert manifest["validation_set_generation_allowed"] is False
    assert set(manifest["files_sha256"]) == {
        "codebook_zh.json",
        "protocol.json",
        "reviewer_a.jsonl",
        "reviewer_b.jsonl",
        "study_manager_key.jsonl",
    }
    assert all(len(value) == 64 for value in manifest["files_sha256"].values())
