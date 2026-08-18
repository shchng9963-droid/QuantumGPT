#!/usr/bin/env python3
"""Run the frozen ReliabilityBench-Q B1 preflight or formal smoke pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.reliability_bench.b1 import (  # noqa: E402
    DeepSeekCompletionClient,
    append_jsonl,
    build_b1_report,
    build_manifest,
    build_schedule,
    load_b1_config,
    run_b1_trace,
    select_frozen_episodes,
    write_json,
)
from eval.reliability_bench.generator import generate_stage_a_episodes  # noqa: E402
from eval.reliability_bench.groups import ExperimentGroup  # noqa: E402


DEFAULT_CONFIG = ROOT / "eval/reliability_bench/data/b1_config.json"


def _load_env_key(path: Path, key: str) -> None:
    if os.environ.get(key) or not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            os.environ[key] = value.strip().strip('"').strip("'")
            return


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_preflight(path: Path, manifest: dict) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("mode") != "preflight" or not report.get("acceptance_passed"):
        raise RuntimeError("formal B1 requires a passing preflight report")
    previous = report["manifest"]
    for key in ("config_sha256", "benchmark_sha256"):
        if previous.get(key) != manifest.get(key):
            raise RuntimeError(f"preflight {key} does not match formal run")
    if previous.get("git", {}).get("commit") != manifest.get("git", {}).get("commit"):
        raise RuntimeError("preflight and formal B1 must use the same Git commit")
    if previous.get("frozen_file_sha256") != manifest.get("frozen_file_sha256"):
        raise RuntimeError("preflight and formal B1 frozen files do not match")


def _validate_resume_manifest(path: Path, manifest: dict) -> None:
    previous = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "mode",
        "config_sha256",
        "benchmark_sha256",
        "frozen_file_sha256",
    ):
        if previous.get(key) != manifest.get(key):
            raise RuntimeError(f"resume manifest mismatch: {key}")
    if previous.get("git", {}).get("commit") != manifest.get("git", {}).get("commit"):
        raise RuntimeError("resume Git commit does not match original run")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    out_dir = args.out.resolve()
    traces_path = out_dir / "traces.jsonl"
    report_path = out_dir / "report.json"
    preflight = args.mode == "preflight"
    config = load_b1_config(config_path)
    episodes = select_frozen_episodes(generate_stage_a_episodes(), config)
    episodes_by_id = {item.episode_id: item for item in episodes}
    schedule = build_schedule(config, preflight=preflight)
    manifest = build_manifest(ROOT, config_path, episodes, preflight=preflight)
    if manifest["git"]["tracked_worktree_dirty"]:
        raise RuntimeError("B1 requires a clean tracked Git worktree")
    if not preflight:
        if args.preflight_report is None:
            raise RuntimeError("--preflight-report is required for formal B1")
        _validate_preflight(args.preflight_report.resolve(), manifest)

    if traces_path.exists() and not args.resume:
        raise RuntimeError(
            f"trace file already exists: {traces_path}; use a new directory or --resume"
        )
    if args.resume and (out_dir / "manifest.json").exists():
        _validate_resume_manifest(out_dir / "manifest.json", manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "schedule.json", schedule)
    write_json(out_dir / "config.lock.json", config)

    _load_env_key(ROOT / ".env", "DEEPSEEK_API_KEY")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not available")
    client = DeepSeekCompletionClient(api_key, config)

    traces = _read_jsonl(traces_path) if args.resume else []
    completed = {item["trace_id"] for item in traces}
    total_cost = sum(float(item["llm"]["cost"]["estimated_cost"]) for item in traces)
    for item in schedule:
        episode = episodes_by_id[item["episode_id"]]
        group = ExperimentGroup(item["controller_group"])
        expected_trace_id = (
            "b1-"
            + __import__("hashlib")
            .sha256(
                (
                    f"{manifest['config_sha256']}|{item['schedule_index']}|"
                    f"{episode.episode_id}|{group.value}"
                ).encode("utf-8")
            )
            .hexdigest()[:20]
        )
        if expected_trace_id in completed:
            continue
        if total_cost >= float(config["run_cost_cap_usd"]):
            raise RuntimeError("frozen B1 run cost cap reached before next call")
        trace = run_b1_trace(episode, group, client, config, item, manifest)
        append_jsonl(traces_path, trace)
        traces.append(trace)
        completed.add(trace["trace_id"])
        total_cost += float(trace["llm"]["cost"]["estimated_cost"])
        print(
            json.dumps(
                {
                    "progress": f"{len(traces)}/{len(schedule)}",
                    "episode_id": episode.episode_id,
                    "hidden_group_id": item["hidden_group_id"],
                    "group": group.value,
                    "complete": trace["trace_complete"],
                    "scoreable": not trace["unscorable"],
                    "tool_calls": trace["budget_usage"]["tool_calls"],
                    "tokens": trace["llm"]["usage"].get("total_tokens", 0),
                    "cost_usd": trace["llm"]["cost"]["estimated_cost"],
                    "api_error_attempts": len(trace["llm"]["api_errors"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    report = build_b1_report(traces, manifest, preflight=preflight)
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "run_count": report["run_count"],
                "acceptance_passed": report["acceptance_passed"],
                "estimated_cost_usd": round(total_cost, 8),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["acceptance_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
