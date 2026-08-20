#!/usr/bin/env python3
"""Freeze B1.1 human taxonomy artifacts after independent adjudication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reliability_bench.b11_freeze import freeze_human_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--generated-at")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_human_artifacts(
        package_dir=args.package_dir,
        score_dir=args.score_dir,
        reviewer_a_completed=args.reviewer_a,
        reviewer_b_completed=args.reviewer_b,
        reconciliation_completed=args.reconciliation,
        output_dir=args.out,
        git_commit=args.git_commit,
        generated_at=args.generated_at,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
