"""Blind-review selection, redaction, annotation, and gate tests."""

from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

from eval.reliability_bench.b0 import run_b0
from eval.reliability_bench.generator import generate_stage_a_episodes
from eval.reliability_bench.human_audit import (
    CORE_LABELS,
    _program_labels,
    build_blind_review_package,
    score_blind_review,
    validate_reviewer_blinding,
)
from eval.reliability_bench.judges import judge_episode


def _package():
    return build_blind_review_package(run_b0(generate_stage_a_episodes()))


def _complete_rows(rows, key, slot, mutate=None):
    key_field = f"reviewer_{slot}_item_id"
    by_item = {item[key_field]: item for item in key}
    completed = []
    for row in rows:
        labels = dict(by_item[row["item_id"]]["program_labels"])
        annotation = {
            "annotator_id": f"reviewer-{slot}",
            **labels,
            "confidence": 5,
            "rationale": "Independent assessment from the visible scene and trace.",
        }
        if mutate:
            mutate(row, annotation)
        completed.append({**row, "annotation": annotation})
    return completed


def test_blind_sample_is_balanced_and_contains_errors_for_every_task():
    package = _package()
    key = package["manager_key"]
    assert len(key) == 24
    assert Counter(item["task_type"] for item in key) == {
        task: 4 for task in {item["task_type"] for item in key}
    }
    assert Counter(item["relevance"] for item in key) == {
        "related": 12,
        "unrelated": 12,
    }
    assert Counter(item["group"] for item in key) == {
        group: 4 for group in {item["group"] for item in key}
    }
    assert {item["magnitude_quantile"] for item in key} == {50, 75, 95}
    errors_by_task = Counter(
        item["task_type"]
        for item in key
        if not item["program_labels"]["final_decision_correct"]
    )
    assert set(errors_by_task) == {item["task_type"] for item in key}


def test_reviewer_files_are_independently_shuffled_and_strictly_blinded():
    package = _package()
    reviewer_a = package["reviewer_a"]
    reviewer_b = package["reviewer_b"]
    assert validate_reviewer_blinding(reviewer_a)["passed"] is True
    assert validate_reviewer_blinding(reviewer_b)["passed"] is True
    assert package["protocol"]["reviewer_guide_blinding_audit"]["passed"] is True
    assert [item["item_id"] for item in reviewer_a] != [
        item["item_id"] for item in reviewer_b
    ]
    assert all(
        "reason" not in call
        for row in reviewer_a + reviewer_b
        for call in row["scene"]["tool_calls"]
    )
    assert all(
        set(row["annotation"]) == {
            "annotator_id",
            "change_affects_existing_evidence",
            "stale_evidence_ids",
            "stale_evidence_reused",
            "necessary_revalidation_performed",
            "final_decision_correct",
            "unsupported_success_claim",
            "confidence",
            "rationale",
        }
        for row in reviewer_a + reviewer_b
    )


def test_program_judge_and_label_adapter_do_not_read_group_or_oracle_fields():
    judge_source = inspect.getsource(judge_episode).lower()
    adapter_source = inspect.getsource(_program_labels).lower()
    assert "experimentgroup" not in judge_source
    assert "oracle" not in judge_source
    assert '["group"]' not in adapter_source
    assert "oracle" not in adapter_source


def test_blank_review_files_keep_b1_blocked():
    package = _package()
    report = score_blind_review(
        package["reviewer_a"],
        package["reviewer_b"],
        package["manager_key"],
    )
    assert report["b1_ready"] is False
    assert report["final_label_count"] == 0
    assert report["gate"]["all_24_cases_human_judgeable"] is False


def test_perfect_independent_annotations_pass_every_gate_and_metric():
    package = _package()
    reviewer_a = _complete_rows(
        package["reviewer_a"], package["manager_key"], "a"
    )
    reviewer_b = _complete_rows(
        package["reviewer_b"], package["manager_key"], "b"
    )
    report = score_blind_review(reviewer_a, reviewer_b, package["manager_key"])
    assert report["b1_ready"] is True
    assert report["human_human"]["exact_agreement"] == 1.0
    assert report["program_vs_final_human"]["exact_agreement"] == 1.0
    assert report["program_vs_final_human"]["stale_evidence_set"]["f1"] == 1.0
    assert all(
        item["exact_agreement"] == 1.0
        for item in report["program_vs_final_human"]["by_task_type"].values()
    )
    assert all(report["gate"].values())
    assert set(CORE_LABELS) <= set(reviewer_a[0]["annotation"])


def test_disagreement_requires_third_person_adjudication():
    package = _package()
    reviewer_a = _complete_rows(
        package["reviewer_a"], package["manager_key"], "a"
    )
    target_id = package["reviewer_b"][0]["item_id"]

    def flip_one(row, annotation):
        if row["item_id"] == target_id:
            annotation["final_decision_correct"] = not annotation[
                "final_decision_correct"
            ]

    reviewer_b = _complete_rows(
        package["reviewer_b"], package["manager_key"], "b", mutate=flip_one
    )
    pending = score_blind_review(reviewer_a, reviewer_b, package["manager_key"])
    assert pending["b1_ready"] is False
    assert pending["disagreement_count"] == 1
    assert pending["unresolved_disagreement_count"] == 1
    template = pending["adjudication_template"]
    case_id = template[0]["case_id"]
    key = next(item for item in package["manager_key"] if item["case_id"] == case_id)
    adjudication = {
        "case_id": case_id,
        "annotation": {
            "annotator_id": "adjudicator-c",
            **key["program_labels"],
            "confidence": 5,
            "rationale": "Resolved after reviewing both independent rationales.",
        },
    }
    resolved = score_blind_review(
        reviewer_a,
        reviewer_b,
        package["manager_key"],
        [adjudication],
    )
    assert resolved["unresolved_disagreement_count"] == 0
    assert resolved["final_label_count"] == 24
    assert resolved["b1_ready"] is True


def test_tracked_blind_review_package_matches_the_generator():
    root = Path(__file__).resolve().parents[2]
    output = root / "eval/reliability_bench/data/blind_review"
    package = _package()

    def read_jsonl(name):
        return [
            json.loads(line)
            for line in (output / name).read_text().splitlines()
        ]

    assert read_jsonl("reviewer_a.jsonl") == package["reviewer_a"]
    assert read_jsonl("reviewer_b.jsonl") == package["reviewer_b"]
    assert read_jsonl("study_manager_key.jsonl") == package["manager_key"]
    assert json.loads((output / "reviewer_guide.json").read_text()) == package[
        "reviewer_guide"
    ]
    assert json.loads((output / "protocol.json").read_text()) == package[
        "protocol"
    ]
