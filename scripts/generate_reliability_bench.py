"""Generate the locked Stage-A ReliabilityBench-Q Synthetic dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.reliability_bench.generator import write_stage_a_dataset
from eval.reliability_bench.mechanisms import write_stage_a_mechanism_audit
from eval.reliability_bench.generator import generate_stage_a_episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/reliability_bench/data/stage_a_synthetic_v1.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/reliability_bench/data/stage_a_manifest.json"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "eval/reliability_bench/data/stage_a_mechanism_audit.json"
        ),
    )
    args = parser.parse_args()
    manifest = write_stage_a_dataset(args.output, args.manifest)
    audit = write_stage_a_mechanism_audit(
        generate_stage_a_episodes(), args.audit
    )
    print(json.dumps({"manifest": manifest, "audit": audit}, indent=2, sort_keys=True))
    return 0 if audit["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
