#!/usr/bin/env python3
"""Build the blinded B1.1 failure-taxonomy calibration package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reliability_bench.b11_audit import write_calibration_package  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = write_calibration_package(_read_jsonl(args.traces), args.out)
    print(
        json.dumps(
            {
                "output_dir": str(args.out.resolve()),
                "status": manifest["status"],
                "codebook_frozen": manifest["codebook_frozen"],
                "validation_set_generation_allowed": manifest[
                    "validation_set_generation_allowed"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
