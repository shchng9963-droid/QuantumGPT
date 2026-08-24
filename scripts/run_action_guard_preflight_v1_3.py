#!/usr/bin/env python3
"""Run the fresh 6 x 2 x 6 ActionGuard package v1.3 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.reliability_bench.action_guard_preflight_v1_3 import (  # noqa: E402
    PREFLIGHT_VERSION,
    SHARED_AGENT_SYSTEM_PROMPT_SHA256,
    DeepSeekCompletionClient,
    build_schedule,
    canonical_json,
    generate_development_preflight_episodes,
    run_preflight_trace,
)
from eval.reliability_bench.public_runtime_v3_1 import SHARED_RUNTIME_CONFIG, StudyArm  # noqa: E402
from eval.reliability_bench.schema import episode_from_dict  # noqa: E402


DEFAULT_CONFIG = ROOT / "eval/reliability_bench/data/action_guard_preflight_config_v1_3.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_env_key(path: Path, key: str) -> None:
    if os.environ.get(key) or not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            if name.strip() == key:
                os.environ[key] = value.strip().strip('"').strip("'")
                return


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["protocol_version"] != PREFLIGHT_VERSION:
        raise RuntimeError("preflight version/config mismatch")
    shared = {
        "max_tokens": SHARED_RUNTIME_CONFIG.max_tokens,
        "max_turns": SHARED_RUNTIME_CONFIG.max_turns,
        "max_tool_calls": SHARED_RUNTIME_CONFIG.max_tool_calls,
        "max_cost_units": SHARED_RUNTIME_CONFIG.max_cost_units,
        "request_timeout_seconds": SHARED_RUNTIME_CONFIG.request_timeout_seconds,
        "format_max_attempts": SHARED_RUNTIME_CONFIG.format_max_attempts,
    }
    for key, expected in shared.items():
        if config[key] != expected:
            raise RuntimeError(f"preflight {key} differs from frozen shared runtime")
    if config["retry"] != {
        "max_attempts": SHARED_RUNTIME_CONFIG.api_max_attempts,
        "fixed_backoff_seconds": SHARED_RUNTIME_CONFIG.api_fixed_backoff_seconds,
    }:
        raise RuntimeError("preflight retry policy differs from frozen shared runtime")
    config["config_sha256"] = sha256_file(config_path)
    episodes = generate_development_preflight_episodes(int(config["development_episode_seed"]))
    schedule = build_schedule(episodes, int(config["schedule_seed"]))
    out = args.out.resolve()
    traces_path = out / "traces.jsonl"
    if traces_path.exists() and not args.resume:
        raise RuntimeError("output already exists; use a new directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.lock.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "schedule.json").write_text(
        json.dumps(schedule, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(canonical_json(episode.to_dict()) + "\n")
    manifest = {
        "preflight_version": PREFLIGHT_VERSION,
        "development_only": True,
        "formal_claims_allowed": False,
        "config_sha256": config["config_sha256"],
        "shared_system_prompt_sha256": SHARED_AGENT_SYSTEM_PROMPT_SHA256,
        "episode_sha256": sha256_file(out / "episodes.jsonl"),
        "schedule_sha256": sha256_file(out / "schedule.json"),
        "run_count": 72,
        "method_validation_v2_1_access": {
            "status": "not_accessed",
            "evidence_type": "custodian_declaration_only",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _load_env_key(ROOT / ".env", "DEEPSEEK_API_KEY")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is unavailable")
    client = DeepSeekCompletionClient(api_key, config)
    existing = []
    if args.resume and traces_path.exists():
        existing = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines() if line]
    completed = {item["schedule"]["schedule_index"] for item in existing}
    total_cost = sum(float(item["llm"]["estimated_cost_usd"]) for item in existing)
    by_episode = {item.episode_id: item for item in episodes}
    for item in schedule:
        if item["schedule_index"] in completed:
            continue
        if total_cost >= float(config["run_cost_cap_usd"]):
            raise RuntimeError("development preflight cost cap reached")
        trace = run_preflight_trace(
            episode=by_episode[item["episode_id"]],
            arm=StudyArm(item["arm"]),
            schedule_item=item,
            client=client,
            config=config,
        )
        _append_jsonl(traces_path, trace)
        total_cost += float(trace["llm"]["estimated_cost_usd"])
        print(
            canonical_json(
                {
                    "progress": f"{item['schedule_index'] + 1}/72",
                    "task_type": item["task_type"],
                    "arm": item["arm"],
                    "complete": trace["trace_complete"],
                    "turns": trace["budget_usage"]["turns"],
                    "guard_events": len(trace["guard_events"]),
                    "cost_usd": trace["llm"]["estimated_cost_usd"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
