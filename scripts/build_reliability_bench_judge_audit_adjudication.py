#!/usr/bin/env python3
"""Freeze the valid stage-1 return, then build stage-2 adjudication materials."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reliability_bench.judge_audit_package_v2 import (  # noqa: E402
    build_stage2_adjudication_package,
    validate_stage1_coordination_return,
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


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordination-root", type=Path, required=True)
    parser.add_argument("--stage1-return", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    coordination_root = args.coordination_root.resolve()
    stage1_return_path = args.stage1_return.resolve()
    output_dir = args.out.resolve()
    if output_dir.exists():
        raise RuntimeError("stage2 output already exists; use a new directory")
    if subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("stage2 generation requires a clean tracked worktree")

    stage1_public = coordination_root / "stage1_for_third_annotator"
    manager_root = coordination_root / "study_manager_only"
    original_path = stage1_public / "第三标注者_阶段1独立标注_待填写.jsonl"
    codebook_path = stage1_public / "冻结标注规范_v0.2.json"
    manager_key_path = manager_root / "coordination_manager_key.jsonl"
    reviewer_a_path = manager_root / "returned_inputs/reviewer_a_returned.jsonl"
    reviewer_b_path = manager_root / "returned_inputs/reviewer_b_returned.jsonl"
    original_rows = _load_jsonl(original_path)
    returned_rows = _load_jsonl(stage1_return_path)
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    validation = validate_stage1_coordination_return(
        original_rows=original_rows,
        returned_rows=returned_rows,
        frozen_codebook=codebook,
    )
    if not validation["passed"]:
        raise RuntimeError(f"stage1 return cannot be frozen: {validation}")
    returned_sha256 = _sha256(stage1_return_path)
    freeze_dir = manager_root / f"stage1_return_frozen_{returned_sha256[:12]}"
    if freeze_dir.exists():
        raise RuntimeError("this stage1 return has already been frozen")
    freeze_dir.mkdir(mode=0o700)
    frozen_return_path = freeze_dir / "third_annotator_stage1_return.jsonl"
    shutil.copyfile(stage1_return_path, frozen_return_path)
    os.chmod(frozen_return_path, 0o600)
    if _sha256(frozen_return_path) != returned_sha256:
        raise RuntimeError("frozen stage1 return hash mismatch")
    freeze_manifest = {
        "status": "stage1_return_validated_and_frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "stage1_return_sha256": returned_sha256,
        "row_count": validation["row_count"],
        "validation_error_count": validation["error_count"],
        "stage2_generation_allowed": True,
        "judge_predictions_accessed": False,
        "judge_unseal_allowed": False,
    }
    freeze_manifest_path = freeze_dir / "stage1_return_freeze_manifest.json"
    _write_json(freeze_manifest_path, freeze_manifest)

    result = build_stage2_adjudication_package(
        stage1_original=original_rows,
        stage1_returned=returned_rows,
        coordination_manager_key=_load_jsonl(manager_key_path),
        reviewer_a=_load_jsonl(reviewer_a_path),
        reviewer_b=_load_jsonl(reviewer_b_path),
        output_dir=output_dir,
        frozen_codebook=codebook,
        frozen_stage1_return_sha256=returned_sha256,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "item_count": result["item_count"],
                "frozen_stage1_return_sha256": returned_sha256,
                "public_manifest_sha256": result["public_manifest_sha256"],
                "judge_predictions_accessed": result["judge_predictions_accessed"],
                "judge_unseal_allowed": result["judge_unseal_allowed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
