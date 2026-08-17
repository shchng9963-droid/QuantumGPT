"""Run the development-only deterministic ReliabilityBench-Q B0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.reliability_bench.b0 import write_b0_outputs
from eval.reliability_bench.generator import generate_stage_a_episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/reliability_bench/data"),
    )
    args = parser.parse_args()
    report = write_b0_outputs(generate_stage_a_episodes(), args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
