#!/usr/bin/env python3
"""Validate and compare two completed B1.1 calibration reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reliability_bench.b11_audit import (  # noqa: E402
    score_calibration_reviews,
    write_calibration_score,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = score_calibration_reviews(
        _read_jsonl(args.reviewer_a),
        _read_jsonl(args.reviewer_b),
        _read_jsonl(args.package_dir / "reviewer_a.jsonl"),
        _read_jsonl(args.package_dir / "reviewer_b.jsonl"),
        _read_jsonl(args.package_dir / "study_manager_key.jsonl"),
    )
    report["source_files_sha256"] = {
        "reviewer_a_completed": _sha256(args.reviewer_a),
        "reviewer_b_completed": _sha256(args.reviewer_b),
        "reviewer_a_original": _sha256(args.package_dir / "reviewer_a.jsonl"),
        "reviewer_b_original": _sha256(args.package_dir / "reviewer_b.jsonl"),
        "study_manager_key": _sha256(args.package_dir / "study_manager_key.jsonl"),
    }
    write_calibration_score(report, args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "disagreement_count": report.get("disagreement_count"),
                "agreement": report.get("agreement"),
                "output_dir": str(args.out.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] != "invalid_returned_annotations" else 2


if __name__ == "__main__":
    raise SystemExit(main())
