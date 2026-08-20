#!/usr/bin/env python3
"""Build stage-1-only third-annotator coordination materials on the server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reliability_bench.judge_audit_package_v2 import (  # noqa: E402
    build_stage1_coordination_package,
)


def _load_jsonl(path: Path) -> list[dict]:
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


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    audit_root = args.audit_root.resolve()
    reviewer_a_path = args.reviewer_a.resolve()
    reviewer_b_path = args.reviewer_b.resolve()
    output_dir = args.out.resolve()
    if output_dir.exists():
        raise RuntimeError("coordination output already exists; use a new directory")
    prohibited = ("sealed_judge_predictions", "judge_predictions.jsonl.aesgcm")
    if any(token in str(path) for path in (reviewer_a_path, reviewer_b_path) for token in prohibited):
        raise RuntimeError("judge prediction paths are prohibited")

    blind_root = audit_root / "blind_review"
    original_a_path = blind_root / "给标注者A/标注者A_待标注.jsonl"
    original_b_path = blind_root / "给标注者B/标注者B_待标注.jsonl"
    manager_key_path = blind_root / "study_manager_only/audit_manager_key.jsonl"
    codebook_path = blind_root / "给标注者A/冻结标注规范_v0.2.json"
    reviewer_a_sha = _sha256(reviewer_a_path)
    reviewer_b_sha = _sha256(reviewer_b_path)
    result = build_stage1_coordination_package(
        reviewer_a=_load_jsonl(reviewer_a_path),
        reviewer_b=_load_jsonl(reviewer_b_path),
        original_a=_load_jsonl(original_a_path),
        original_b=_load_jsonl(original_b_path),
        manager_key=_load_jsonl(manager_key_path),
        output_dir=output_dir,
        frozen_codebook=json.loads(codebook_path.read_text(encoding="utf-8")),
        reviewer_a_source_sha256=reviewer_a_sha,
        reviewer_b_source_sha256=reviewer_b_sha,
    )

    manager_inputs = output_dir / "study_manager_only/returned_inputs"
    manager_inputs.mkdir(mode=0o700)
    saved_a = manager_inputs / "reviewer_a_returned.jsonl"
    saved_b = manager_inputs / "reviewer_b_returned.jsonl"
    shutil.copyfile(reviewer_a_path, saved_a)
    shutil.copyfile(reviewer_b_path, saved_b)
    os.chmod(saved_a, 0o600)
    os.chmod(saved_b, 0o600)
    if _sha256(saved_a) != reviewer_a_sha or _sha256(saved_b) != reviewer_b_sha:
        raise RuntimeError("returned annotation copy hash mismatch")
    print(
        json.dumps(
            {
                "status": result["status"],
                "disagreement_count": result["disagreement_count"],
                "outcome_disagreement_count": result["outcome_disagreement_count"],
                "stage1_manifest_sha256": result["stage1_manifest_sha256"],
                "stage2_generated": result["stage2_generated"],
                "judge_predictions_accessed": result["judge_predictions_accessed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
