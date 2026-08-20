#!/usr/bin/env python3
"""Freeze human consensus before unsealing and scoring judge-v2 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reliability_bench.judge_audit_finalize_v2 import (  # noqa: E402
    FINALIZATION_VERSION,
    build_human_consensus,
    decrypt_prediction_rows,
    score_judge_predictions,
    validate_stage2_return,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(_canonical_json(dict(row)) for row in rows) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _require_file(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--coordination-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--stage1-freeze-manifest", type=Path, required=True)
    parser.add_argument("--stage2-return", type=Path, required=True)
    parser.add_argument("--stage2-return-manifest", type=Path, required=True)
    parser.add_argument("--prediction-key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("formal finalization requires a clean tracked worktree")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    audit_root = args.audit_root.resolve()
    coordination_root = args.coordination_root.resolve()
    stage2_root = args.stage2_root.resolve()
    output_root = args.out.resolve()
    if output_root.exists():
        raise RuntimeError("formal output already exists; use a new directory")

    config_path = _require_file(ROOT / "eval/reliability_bench/data/judge_audit_v2_config.json")
    stage2_original_path = _require_file(
        stage2_root
        / "stage2_for_third_annotator/第三标注者_阶段2最终裁决_待填写.jsonl"
    )
    codebook_path = _require_file(
        stage2_root / "stage2_for_third_annotator/冻结标注规范_v0.2.json"
    )
    adjudication_key_path = _require_file(
        stage2_root / "study_manager_only/adjudication_manager_key.jsonl"
    )
    coordination_key_path = _require_file(
        coordination_root / "study_manager_only/coordination_manager_key.jsonl"
    )
    reviewer_a_path = _require_file(
        coordination_root / "study_manager_only/returned_inputs/reviewer_a_returned.jsonl"
    )
    reviewer_b_path = _require_file(
        coordination_root / "study_manager_only/returned_inputs/reviewer_b_returned.jsonl"
    )
    audit_key_path = _require_file(
        audit_root / "blind_review/study_manager_only/audit_manager_key.jsonl"
    )
    prediction_archive_path = audit_root / "sealed_judge_predictions/judge_predictions.jsonl.aesgcm"
    prediction_commitment_path = audit_root / "sealed_judge_predictions/prediction_commitment_manifest.json"
    stage2_return_path = _require_file(args.stage2_return)
    stage2_return_manifest_path = _require_file(args.stage2_return_manifest)
    stage1_freeze_manifest_path = _require_file(args.stage1_freeze_manifest)
    prediction_key_path = args.prediction_key.resolve()

    original_rows = _load_jsonl(stage2_original_path)
    returned_rows = _load_jsonl(stage2_return_path)
    codebook = _load_json(codebook_path)
    validation = validate_stage2_return(
        original_rows=original_rows,
        returned_rows=returned_rows,
        frozen_codebook=codebook,
    )
    if not validation["passed"]:
        raise RuntimeError(f"stage2 return cannot be frozen: {validation}")

    consensus_rows = build_human_consensus(
        stage2_returned=returned_rows,
        adjudication_manager_key=_load_jsonl(adjudication_key_path),
        coordination_manager_key=_load_jsonl(coordination_key_path),
        audit_manager_key=_load_jsonl(audit_key_path),
        reviewer_a=_load_jsonl(reviewer_a_path),
        reviewer_b=_load_jsonl(reviewer_b_path),
    )
    consensus_distribution = {}
    for row in consensus_rows:
        label = row["trajectory_outcome"]
        consensus_distribution[label] = consensus_distribution.get(label, 0) + 1

    output_root.mkdir(parents=True, mode=0o700)
    manager_root = output_root / "study_manager_only"
    returned_root = manager_root / "returned_inputs"
    unsealed_root = manager_root / "unsealed_after_consensus_freeze"
    public_root = output_root / "aggregate_report"
    for directory in (manager_root, returned_root, public_root):
        directory.mkdir(mode=0o700)

    frozen_stage2_return = returned_root / "third_annotator_stage2_return.jsonl"
    frozen_stage2_manifest = returned_root / "third_annotator_returned_manifest.json"
    shutil.copyfile(stage2_return_path, frozen_stage2_return)
    shutil.copyfile(stage2_return_manifest_path, frozen_stage2_manifest)
    for path in (frozen_stage2_return, frozen_stage2_manifest):
        os.chmod(path, 0o600)
    if _sha256(frozen_stage2_return) != _sha256(stage2_return_path):
        raise RuntimeError("frozen stage2 return hash mismatch")

    consensus_path = manager_root / "human_consensus_24.jsonl"
    _write_jsonl(consensus_path, consensus_rows)
    consensus_sha256 = _sha256(consensus_path)
    stage1_freeze_manifest = _load_json(stage1_freeze_manifest_path)
    frozen_at = datetime.now(timezone.utc).isoformat()
    consensus_manifest = {
        "finalization_version": FINALIZATION_VERSION,
        "status": "human_consensus_validated_and_frozen_before_prediction_unseal",
        "frozen_at": frozen_at,
        "git_commit": git_commit,
        "consensus_count": len(consensus_rows),
        "consensus_distribution": dict(sorted(consensus_distribution.items())),
        "consensus_source_counts": {
            "stage2_independent_adjudication": sum(
                row["consensus_source"] == "stage2_independent_adjudication"
                for row in consensus_rows
            ),
            "independent_a_b_exact_agreement": sum(
                row["consensus_source"] == "independent_a_b_exact_agreement"
                for row in consensus_rows
            ),
        },
        "human_consensus_sha256": consensus_sha256,
        "stage2_original_sha256": _sha256(stage2_original_path),
        "stage2_return_sha256": _sha256(stage2_return_path),
        "stage2_return_manifest_sha256": _sha256(stage2_return_manifest_path),
        "reviewer_a_return_sha256": _sha256(reviewer_a_path),
        "reviewer_b_return_sha256": _sha256(reviewer_b_path),
        "stage1_frozen_return_sha256": stage1_freeze_manifest["stage1_return_sha256"],
        "codebook_sha256": _sha256(codebook_path),
        "stage2_validation": validation,
        "codebook_systematic_ambiguity_count": sum(
            row["codebook_systematic_ambiguity"] for row in consensus_rows
        ),
        "judge_predictions_accessed": False,
        "consensus_frozen_before_prediction_unseal": True,
        "method_validation_archive_accessed": False,
    }
    consensus_manifest_path = manager_root / "human_consensus_freeze_manifest.json"
    _write_json(consensus_manifest_path, consensus_manifest)

    # Prediction artifacts are not resolved or opened until the consensus and
    # its freeze manifest have both been durably written above.
    prediction_archive_path = _require_file(prediction_archive_path)
    prediction_commitment_path = _require_file(prediction_commitment_path)
    prediction_key_path = _require_file(prediction_key_path)
    commitment = _load_json(prediction_commitment_path)
    config = _load_json(config_path)
    if commitment["judge_candidate_commit"] != config["judge_candidate_commit"]:
        raise RuntimeError("prediction candidate commit differs from preregistration")
    if commitment["audit_config_sha256"] != _sha256(config_path):
        raise RuntimeError("prediction commitment config hash mismatch")
    prediction_rows, plaintext = decrypt_prediction_rows(
        archive=prediction_archive_path.read_bytes(),
        key_encoded=prediction_key_path.read_bytes(),
        commitment=commitment,
    )
    unsealed_root.mkdir(mode=0o700)
    plaintext_path = unsealed_root / "judge_predictions.jsonl"
    plaintext_path.write_bytes(plaintext)
    os.chmod(plaintext_path, 0o600)
    if _sha256(plaintext_path) != commitment["plaintext_predictions_sha256"]:
        raise RuntimeError("persisted prediction plaintext hash mismatch")
    unsealed_at = datetime.now(timezone.utc).isoformat()
    if unsealed_at <= frozen_at:
        raise RuntimeError("prediction unseal timestamp must follow consensus freeze")
    unseal_manifest = {
        "finalization_version": FINALIZATION_VERSION,
        "status": "judge_predictions_unsealed_after_human_consensus_freeze",
        "unsealed_at": unsealed_at,
        "human_consensus_frozen_at": frozen_at,
        "human_consensus_sha256": consensus_sha256,
        "human_consensus_freeze_manifest_sha256": _sha256(consensus_manifest_path),
        "prediction_commitment_manifest_sha256": _sha256(prediction_commitment_path),
        "encrypted_predictions_sha256": _sha256(prediction_archive_path),
        "plaintext_predictions_sha256": _sha256(plaintext_path),
        "prediction_count": len(prediction_rows),
        "consensus_frozen_before_prediction_unseal": True,
        "method_validation_archive_accessed": False,
    }
    unseal_manifest_path = unsealed_root / "prediction_unseal_manifest.json"
    _write_json(unseal_manifest_path, unseal_manifest)

    report = score_judge_predictions(
        consensus_rows=consensus_rows,
        prediction_rows=prediction_rows,
        config=config,
    )
    report.update(
        {
            "git_commit": git_commit,
            "judge_candidate_commit": commitment["judge_candidate_commit"],
            "audit_config_sha256": _sha256(config_path),
            "human_consensus_sha256": consensus_sha256,
            "human_consensus_freeze_manifest_sha256": _sha256(consensus_manifest_path),
            "prediction_unseal_manifest_sha256": _sha256(unseal_manifest_path),
            "consensus_frozen_before_prediction_unseal": True,
            "method_validation_archive_accessed": False,
            "method_validation_unseal_allowed": False,
        }
    )
    report_path = manager_root / "formal_judge_vs_human_consensus_report.json"
    _write_json(report_path, report)
    public_summary = {
        key: report[key]
        for key in (
            "finalization_version",
            "status",
            "judge_freeze_allowed",
            "human_consensus_distribution",
            "class_balance_passed",
            "outcome",
            "critical_labels",
            "diagnostics",
            "prediction_count",
            "consensus_count",
            "judge_candidate_commit",
            "human_consensus_sha256",
            "human_consensus_freeze_manifest_sha256",
            "prediction_unseal_manifest_sha256",
            "consensus_frozen_before_prediction_unseal",
            "method_validation_archive_accessed",
            "method_validation_unseal_allowed",
        )
    }
    public_summary_path = public_root / "judge_v2_formal_audit_aggregate.json"
    _write_json(public_summary_path, public_summary)
    aggregate_manifest = {
        "status": "formal_judge_audit_aggregate_only",
        "aggregate_report_sha256": _sha256(public_summary_path),
        "contains_per_case_predictions": False,
        "contains_controller_group_names": False,
        "method_validation_archive_accessed": False,
    }
    _write_json(public_root / "aggregate_manifest.json", aggregate_manifest)
    print(json.dumps(public_summary, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
