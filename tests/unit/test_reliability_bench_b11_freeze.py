"""Tests for freezing B1.1 human-taxonomy artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eval.reliability_bench.b11_audit import CODEBOOK_ZH, empty_annotation
from eval.reliability_bench.b11_freeze import (
    build_consensus_labels,
    freeze_human_artifacts,
    validate_completed_reconciliation,
)


def _annotation(outcome="correct", stage=None, tags=None):
    value = empty_annotation()
    value.update(
        {
            "annotator_id": "reviewer",
            "trajectory_outcome": outcome,
            "primary_failure_stage": stage,
            "secondary_error_tags": tags or [],
            "confidence": 0.9,
            "rationale": "依据可见轨迹判断。",
        }
    )
    return value


def _fixtures():
    manager = []
    reviewer_a = []
    reviewer_b = []
    for index in range(24):
        case_id = f"case-{index:02d}"
        item_a = f"item-a-{index:02d}"
        item_b = f"item-b-{index:02d}"
        manager.append(
            {
                "case_id": case_id,
                "reviewer_a_item_id": item_a,
                "reviewer_b_item_id": item_b,
                "task_type": "backend_selection",
            }
        )
        reviewer_a.append(
            {"item_id": item_a, "scene": {"trace_complete": True}, "annotation": _annotation()}
        )
        reviewer_b.append(
            {"item_id": item_b, "scene": {"trace_complete": True}, "annotation": _annotation()}
        )
    reviewer_b[0]["annotation"] = _annotation(
        "incorrect", "Evidence", ["stale_evidence_reuse"]
    )
    from eval.reliability_bench.b11_freeze import _discussion_id

    original = [
        {
            "discussion_id": _discussion_id("case-00"),
            "scene": {"trace_complete": True},
            "reviewer_a_annotation": reviewer_a[0]["annotation"],
            "reviewer_b_annotation": reviewer_b[0]["annotation"],
            "resolution": {
                "resolver_ids": [],
                "taxonomy_change_needed": None,
                "agreed_outcome": None,
                "agreed_primary_failure_stage": None,
                "agreed_secondary_error_tags": [],
                "codebook_change_notes": "",
                "rationale": "",
            },
        }
    ]
    completed = copy.deepcopy(original)
    completed[0]["resolution"] = {
        "resolver_ids": ["resolver-1"],
        "taxonomy_change_needed": False,
        "agreed_outcome": "correct",
        "agreed_primary_failure_stage": None,
        "agreed_secondary_error_tags": [],
        "codebook_change_notes": "",
        "rationale": "最终动作与最新证据一致。",
    }
    return manager, reviewer_a, reviewer_b, original, completed


def _write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows, *, bom=False):
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding=encoding,
    )


def test_completed_reconciliation_preserves_blind_payload_and_accepts_bom(tmp_path):
    _, _, _, original, completed = _fixtures()
    result = validate_completed_reconciliation(original, completed)
    assert result["passed"] is True
    assert result["taxonomy_change_needed_count"] == 0

    changed = copy.deepcopy(completed)
    changed[0]["scene"] = {"trace_complete": False}
    result = validate_completed_reconciliation(original, changed)
    assert result["passed"] is False
    assert "immutable_payload_modified" in next(iter(result["errors"].values()))


def test_consensus_requires_exact_reconciliation_set():
    manager, reviewer_a, reviewer_b, _, completed = _fixtures()
    consensus = build_consensus_labels(
        reviewer_a, reviewer_b, manager, completed
    )
    assert len(consensus) == 24
    assert consensus[0]["consensus_source"] == "independent_adjudication"
    assert consensus[0]["trajectory_outcome"] == "correct"
    with pytest.raises(ValueError, match="unresolved disagreement"):
        build_consensus_labels(reviewer_a, reviewer_b, manager, [])


def test_freeze_writes_raw_reconciliation_and_blocks_generation(tmp_path):
    manager, reviewer_a, reviewer_b, original, completed = _fixtures()
    package = tmp_path / "package"
    score = tmp_path / "score"
    returned = tmp_path / "returned"
    package.mkdir()
    score.mkdir()
    returned.mkdir()
    _write_jsonl(package / "reviewer_a.jsonl", reviewer_a)
    _write_jsonl(package / "reviewer_b.jsonl", reviewer_b)
    _write_jsonl(package / "study_manager_key.jsonl", manager)
    _write_jsonl(score / "blind_reconciliation.jsonl", original)
    _write_json(score / "proposed_codebook_v0.2_zh.json", CODEBOOK_ZH)
    reviewer_a_path = returned / "reviewer_a.jsonl"
    reviewer_b_path = returned / "reviewer_b.jsonl"
    reconciliation_path = returned / "reconciliation.jsonl"
    _write_jsonl(reviewer_a_path, reviewer_a)
    _write_jsonl(reviewer_b_path, reviewer_b)
    _write_jsonl(reconciliation_path, completed, bom=True)

    output = tmp_path / "frozen"
    manifest = freeze_human_artifacts(
        package_dir=package,
        score_dir=score,
        reviewer_a_completed=reviewer_a_path,
        reviewer_b_completed=reviewer_b_path,
        reconciliation_completed=reconciliation_path,
        output_dir=output,
        git_commit="deadbeef",
        generated_at="2026-08-20T00:00:00+00:00",
    )
    assert manifest["taxonomy_frozen"] is True
    assert manifest["method_validation_generation_allowed"] is False
    assert (output / "reconciliation_completed_raw.jsonl").read_bytes() == (
        reconciliation_path.read_bytes()
    )
    report = json.loads((output / "human_freeze_report.json").read_text())
    assert report["consensus_case_count"] == 24
    assert report["judge_v1_formal_scoring_allowed"] is False
    frozen_codebook = json.loads(
        (output / "frozen_codebook_v0.2_zh.json").read_text()
    )
    assert frozen_codebook["codebook_version"] == "0.2"
    assert frozen_codebook["status"] == "frozen"
