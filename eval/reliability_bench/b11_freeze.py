"""Freeze independently reviewed B1.1 human-taxonomy artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .b11_audit import (
    AUDIT_VERSION,
    PRIMARY_FAILURE_STAGES,
    SECONDARY_ERROR_TAGS,
    validate_reviewer_blinding,
)


FREEZE_VERSION = "reliabilitybench-q/b1.1-human-taxonomy-freeze-1.0"
EXPECTED_CASE_COUNT = 24

RESOLUTION_FIELDS = {
    "resolver_ids",
    "taxonomy_change_needed",
    "agreed_outcome",
    "agreed_primary_failure_stage",
    "agreed_secondary_error_tags",
    "codebook_change_notes",
    "rationale",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL while accepting (and preserving separately) an optional BOM."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _discussion_id(case_id: str) -> str:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
    return f"discussion-{digest}"


def _annotation_labels(annotation: dict[str, Any]) -> tuple[Any, Any, tuple[str, ...]]:
    return (
        annotation["trajectory_outcome"],
        annotation["primary_failure_stage"],
        tuple(sorted(annotation["secondary_error_tags"])),
    )


def _resolution_labels(resolution: dict[str, Any]) -> tuple[Any, Any, tuple[str, ...]]:
    return (
        resolution["agreed_outcome"],
        resolution["agreed_primary_failure_stage"],
        tuple(sorted(resolution["agreed_secondary_error_tags"])),
    )


def _validate_resolution(resolution: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(resolution, dict) or set(resolution) != RESOLUTION_FIELDS:
        return ["resolution_fields"]

    resolver_ids = resolution["resolver_ids"]
    if (
        not isinstance(resolver_ids, list)
        or not resolver_ids
        or len(resolver_ids) != len(set(resolver_ids))
        or any(not isinstance(item, str) or not item.strip() for item in resolver_ids)
    ):
        errors.append("resolver_ids")

    taxonomy_change = resolution["taxonomy_change_needed"]
    if not isinstance(taxonomy_change, bool):
        errors.append("taxonomy_change_needed")

    outcome = resolution["agreed_outcome"]
    if outcome not in {"correct", "incorrect", "uncertain"}:
        errors.append("agreed_outcome")
    stage = resolution["agreed_primary_failure_stage"]
    if outcome == "incorrect" and stage not in PRIMARY_FAILURE_STAGES:
        errors.append("agreed_primary_failure_stage")
    if outcome == "correct" and stage is not None:
        errors.append("agreed_primary_failure_stage")
    if outcome == "uncertain" and stage not in {None, *PRIMARY_FAILURE_STAGES}:
        errors.append("agreed_primary_failure_stage")

    tags = resolution["agreed_secondary_error_tags"]
    if (
        not isinstance(tags, list)
        or len(tags) != len(set(tags))
        or any(tag not in SECONDARY_ERROR_TAGS for tag in tags)
    ):
        errors.append("agreed_secondary_error_tags")
    if not isinstance(resolution["codebook_change_notes"], str):
        errors.append("codebook_change_notes")
    if taxonomy_change is True and not str(
        resolution["codebook_change_notes"]
    ).strip():
        errors.append("missing_codebook_change_notes")
    if (
        not isinstance(resolution["rationale"], str)
        or not resolution["rationale"].strip()
    ):
        errors.append("rationale")
    return sorted(set(errors))


def validate_completed_reconciliation(
    original: list[dict[str, Any]], completed: list[dict[str, Any]]
) -> dict[str, Any]:
    original_by_id = {row["discussion_id"]: row for row in original}
    completed_ids = [row.get("discussion_id") for row in completed]
    errors: dict[str, list[str]] = {}
    for row in completed:
        discussion_id = row.get("discussion_id", "unknown")
        row_errors: list[str] = []
        source = original_by_id.get(discussion_id)
        if source is None:
            row_errors.append("unknown_discussion_id")
        elif {
            key: value for key, value in row.items() if key != "resolution"
        } != {
            key: value for key, value in source.items() if key != "resolution"
        }:
            row_errors.append("immutable_payload_modified")
        row_errors.extend(_validate_resolution(row.get("resolution")))
        if row_errors:
            errors[discussion_id] = sorted(set(row_errors))

    blinding = validate_reviewer_blinding(completed)
    id_set_equal = set(completed_ids) == set(original_by_id)
    ids_unique = len(completed_ids) == len(set(completed_ids))
    row_count_equal = len(completed) == len(original)
    passed = row_count_equal and ids_unique and id_set_equal and not errors and blinding[
        "passed"
    ]
    return {
        "passed": passed,
        "expected_row_count": len(original),
        "actual_row_count": len(completed),
        "row_count_equal": row_count_equal,
        "discussion_ids_unique": ids_unique,
        "discussion_id_set_equal": id_set_equal,
        "error_count": len(errors),
        "errors": errors,
        "blinding": blinding,
        "taxonomy_change_needed_count": sum(
            row.get("resolution", {}).get("taxonomy_change_needed") is True
            for row in completed
        ),
    }


def build_consensus_labels(
    reviewer_a: list[dict[str, Any]],
    reviewer_b: list[dict[str, Any]],
    manager_key: list[dict[str, Any]],
    reconciliation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(manager_key) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"human taxonomy freeze requires {EXPECTED_CASE_COUNT} cases, "
            f"got {len(manager_key)}"
        )
    by_a = {row["item_id"]: row for row in reviewer_a}
    by_b = {row["item_id"]: row for row in reviewer_b}
    resolved = {row["discussion_id"]: row["resolution"] for row in reconciliation}
    disagreement_ids: set[str] = set()
    consensus: list[dict[str, Any]] = []
    for manager in manager_key:
        annotation_a = by_a[manager["reviewer_a_item_id"]]["annotation"]
        annotation_b = by_b[manager["reviewer_b_item_id"]]["annotation"]
        labels_a = _annotation_labels(annotation_a)
        labels_b = _annotation_labels(annotation_b)
        discussion_id = _discussion_id(manager["case_id"])
        if labels_a == labels_b:
            labels = labels_a
            source = "independent_exact_agreement"
        else:
            disagreement_ids.add(discussion_id)
            if discussion_id not in resolved:
                raise ValueError(f"unresolved disagreement: {discussion_id}")
            labels = _resolution_labels(resolved[discussion_id])
            source = "independent_adjudication"
        consensus.append(
            {
                "case_id": manager["case_id"],
                "task_type": manager["task_type"],
                "consensus_source": source,
                "trajectory_outcome": labels[0],
                "primary_failure_stage": labels[1],
                "secondary_error_tags": list(labels[2]),
            }
        )
    if set(resolved) != disagreement_ids:
        raise ValueError("reconciliation set does not exactly match label disagreements")
    return consensus


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(_canonical_json(row) for row in rows) + "\n", encoding="utf-8"
    )


def freeze_human_artifacts(
    *,
    package_dir: Path,
    score_dir: Path,
    reviewer_a_completed: Path,
    reviewer_b_completed: Path,
    reconciliation_completed: Path,
    output_dir: Path,
    git_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    paths = {
        "reviewer_a_original": package_dir / "reviewer_a.jsonl",
        "reviewer_b_original": package_dir / "reviewer_b.jsonl",
        "study_manager_key": package_dir / "study_manager_key.jsonl",
        "reviewer_a_completed": reviewer_a_completed,
        "reviewer_b_completed": reviewer_b_completed,
        "blind_reconciliation_original": score_dir / "blind_reconciliation.jsonl",
        "codebook_v0.2_draft": score_dir / "proposed_codebook_v0.2_zh.json",
        "reconciliation_completed_raw": reconciliation_completed,
    }
    original_reconciliation = read_jsonl(paths["blind_reconciliation_original"])
    completed_reconciliation = read_jsonl(reconciliation_completed)
    reconciliation_validation = validate_completed_reconciliation(
        original_reconciliation, completed_reconciliation
    )
    if not reconciliation_validation["passed"]:
        raise ValueError("completed reconciliation failed validation")
    if reconciliation_validation["taxonomy_change_needed_count"]:
        raise ValueError("taxonomy changes remain unresolved; codebook cannot be frozen")

    reviewer_a = read_jsonl(reviewer_a_completed)
    reviewer_b = read_jsonl(reviewer_b_completed)
    manager_key = read_jsonl(paths["study_manager_key"])
    consensus = build_consensus_labels(
        reviewer_a, reviewer_b, manager_key, completed_reconciliation
    )
    codebook = json.loads(
        paths["codebook_v0.2_draft"].read_text(encoding="utf-8-sig")
    )
    if codebook.get("codebook_version") != "0.2-draft":
        raise ValueError("expected codebook v0.2-draft")
    frozen_codebook = copy.deepcopy(codebook)
    frozen_codebook["codebook_version"] = "0.2"
    frozen_codebook["status"] = "frozen"
    frozen_codebook["freeze_policy"] = (
        "任何语义修改必须升级codebook版本；不得原地修改本冻结文件。"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    codebook_path = output_dir / "frozen_codebook_v0.2_zh.json"
    consensus_path = output_dir / "consensus_labels_v0.2.jsonl"
    raw_reconciliation_path = output_dir / "reconciliation_completed_raw.jsonl"
    report_path = output_dir / "human_freeze_report.json"
    _write_json(codebook_path, frozen_codebook)
    _write_jsonl(consensus_path, consensus)
    shutil.copyfile(reconciliation_completed, raw_reconciliation_path)
    report = {
        "freeze_version": FREEZE_VERSION,
        "source_audit_version": AUDIT_VERSION,
        "status": "human_taxonomy_frozen",
        "taxonomy_frozen": True,
        "codebook_version": "0.2",
        "consensus_case_count": len(consensus),
        "adjudicated_case_count": sum(
            row["consensus_source"] == "independent_adjudication"
            for row in consensus
        ),
        "consensus_outcomes": dict(
            sorted(Counter(row["trajectory_outcome"] for row in consensus).items())
        ),
        "reconciliation_validation": reconciliation_validation,
        "judge_v1_status": "development-only / invalidated judge v1",
        "judge_v1_formal_scoring_allowed": False,
        "method_validation_generation_allowed": False,
        "method_validation_generation_blocker": "generator_schema_not_frozen",
        "method_validation_unseal_allowed": False,
        "method_validation_unseal_blockers": [
            "judge_v2_not_independently_audited_and_frozen",
            "action_guard_not_frozen",
        ],
    }
    _write_json(report_path, report)

    output_hashes = {
        path.name: _sha256_file(path)
        for path in (codebook_path, consensus_path, raw_reconciliation_path, report_path)
    }
    manifest = {
        "freeze_version": FREEZE_VERSION,
        "status": "human_taxonomy_frozen",
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "input_files_sha256": {
            name: _sha256_file(path) for name, path in sorted(paths.items())
        },
        "output_files_sha256": output_hashes,
        "taxonomy_frozen": True,
        "judge_v1_status": "development-only / invalidated judge v1",
        "method_validation_generation_allowed": False,
        "method_validation_unseal_allowed": False,
    }
    _write_json(output_dir / "freeze_manifest.json", manifest)
    return manifest


__all__ = [
    "EXPECTED_CASE_COUNT",
    "FREEZE_VERSION",
    "build_consensus_labels",
    "freeze_human_artifacts",
    "read_jsonl",
    "validate_completed_reconciliation",
]
