from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.reliability_bench.guard_repair_stress_v1_2 import run_controlled_repair_stress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/reliability_bench/data/action_guard_repair_stress_v1_2_summary.json"),
    )
    args = parser.parse_args()
    report = run_controlled_repair_stress()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(args.output), "case_count": report["case_count"], "passed": report["all_reason_codes_covered"] and report["all_repairs_allowed"]}, ensure_ascii=False))
    return 0 if report["all_reason_codes_covered"] and report["all_repairs_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
