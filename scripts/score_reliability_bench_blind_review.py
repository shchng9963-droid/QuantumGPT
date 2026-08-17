"""Score two frozen reviewer files and optional third-person adjudications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.reliability_bench.human_audit import load_jsonl, score_blind_review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--manager-key", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adjudication-template", type=Path)
    args = parser.parse_args()
    report = score_blind_review(
        load_jsonl(args.reviewer_a),
        load_jsonl(args.reviewer_b),
        load_jsonl(args.manager_key),
        load_jsonl(args.adjudications) if args.adjudications else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.adjudication_template:
        rows = report["adjudication_template"]
        args.adjudication_template.parent.mkdir(parents=True, exist_ok=True)
        args.adjudication_template.write_text(
            "\n".join(
                json.dumps(item, sort_keys=True, ensure_ascii=False)
                for item in rows
            )
            + ("\n" if rows else ""),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["b1_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
