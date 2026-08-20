#!/usr/bin/env python3
"""Score frozen B1 development traces with trace-aware judge v2.

This script is development-only. It must never read or unseal the held-out
method-validation archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reliability_bench.schema import episode_from_dict
from eval.reliability_bench.trace_judge_v2 import JUDGE_V2_VERSION, judge_trace_v2


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_development_input(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if "sealed-method-validation" in lowered or path.suffix == ".aesgcm":
        raise ValueError(f"sealed method-validation input is prohibited: {path}")


def _cohen_kappa(confusion: Counter) -> float:
    total = sum(confusion.values())
    if not total:
        return 0.0
    observed = (confusion[(True, True)] + confusion[(False, False)]) / total
    human_true = (confusion[(True, True)] + confusion[(True, False)]) / total
    judge_true = (confusion[(True, True)] + confusion[(False, True)]) / total
    expected = human_true * judge_true + (1 - human_true) * (1 - judge_true)
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--manager-key", type=Path, required=True)
    parser.add_argument("--human-consensus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.episodes,
        args.traces,
        args.manager_key,
        args.human_consensus,
    ):
        _assert_development_input(path)

    episodes = {row["episode_id"]: episode_from_dict(row) for row in _jsonl(args.episodes)}
    traces = {row["trace_id"]: row for row in _jsonl(args.traces)}
    manager = _jsonl(args.manager_key)
    human = {row["case_id"]: row for row in _jsonl(args.human_consensus)}
    rows = []
    confusion = Counter()
    for link in manager:
        trace = traces[link["trace_id"]]
        episode = episodes[link["episode_id"]]
        result = judge_trace_v2(trace, episode)
        human_correct = human[link["case_id"]]["trajectory_outcome"] == "correct"
        judge_correct = result.correct is True
        confusion[(human_correct, judge_correct)] += 1
        rows.append(
            {
                "case_id": link["case_id"],
                "trace_id": link["trace_id"],
                "episode_id": link["episode_id"],
                "human_correct": human_correct,
                "human_primary_failure_stage": human[link["case_id"]]["primary_failure_stage"],
                "human_secondary_error_tags": human[link["case_id"]]["secondary_error_tags"],
                "judge_v2": result.to_dict(),
            }
        )
    agreement = (confusion[(True, True)] + confusion[(False, False)]) / len(rows)
    primary_stage_agreement = sum(
        row["human_primary_failure_stage"] == row["judge_v2"]["primary_failure_stage"]
        for row in rows
    ) / len(rows)
    secondary_exact_agreement = sum(
        set(row["human_secondary_error_tags"])
        == set(row["judge_v2"]["secondary_error_tags"])
        for row in rows
    ) / len(rows)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "development_scores.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = {
        "judge_version": JUDGE_V2_VERSION,
        "status": "development-only_not_independent_evidence",
        "case_count": len(rows),
        "agreement": agreement,
        "cohen_kappa": _cohen_kappa(confusion),
        "primary_failure_stage_exact_agreement": primary_stage_agreement,
        "secondary_error_tags_exact_agreement": secondary_exact_agreement,
        "confusion": {
            "human_correct_judge_correct": confusion[(True, True)],
            "human_correct_judge_incorrect": confusion[(True, False)],
            "human_incorrect_judge_correct": confusion[(False, True)],
            "human_incorrect_judge_incorrect": confusion[(False, False)],
        },
        "method_validation_archive_accessed": False,
        "formal_freeze_allowed": False,
        "input_sha256": {
            "episodes": _sha256(args.episodes),
            "traces": _sha256(args.traces),
            "manager_key": _sha256(args.manager_key),
            "human_consensus": _sha256(args.human_consensus),
        },
    }
    (args.out / "development_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
