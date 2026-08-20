#!/usr/bin/env python3
"""Run the preregistered 24-trace independent audit for judge v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reliability_bench.b1 import (  # noqa: E402
    DeepSeekCompletionClient,
    append_jsonl,
    benchmark_sha256,
    run_b1_trace,
)
from eval.reliability_bench.generator import generate_stage_a_episodes  # noqa: E402
from eval.reliability_bench.groups import ExperimentGroup  # noqa: E402
from eval.reliability_bench.judge_audit_package_v2 import (  # noqa: E402
    build_blinded_review_package,
    seal_judge_predictions,
)
from eval.reliability_bench.judge_audit_v2 import (  # noqa: E402
    AUDIT_EPISODE_COUNT,
    FROZEN_AUDIT_CONFIG,
    audit_deduplication,
    build_audit_schedule,
    generate_judge_audit_episodes,
    validate_audit_config,
)


DEFAULT_CONFIG = ROOT / "eval/reliability_bench/data/judge_audit_v2_config.json"
JUDGE_SOURCE = ROOT / "eval/reliability_bench/trace_judge_v2.py"


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "\n".join(_canonical_json(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _load_env_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "DEEPSEEK_API_KEY":
                return value.strip().strip('"').strip("'")
    raise RuntimeError("DEEPSEEK_API_KEY is not available")


def _verify_judge_candidate(config: dict) -> dict:
    commit = config["judge_candidate_commit"]
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT
    )
    candidate = subprocess.check_output(
        ["git", "show", f"{commit}:eval/reliability_bench/trace_judge_v2.py"],
        cwd=ROOT,
    )
    candidate_hash = hashlib.sha256(candidate).hexdigest()
    current_hash = _sha256_file(JUDGE_SOURCE)
    expected = config["judge_source_sha256"]
    if candidate_hash != expected or current_hash != expected:
        raise RuntimeError("judge candidate source differs from preregistered e2ceb18")
    return {
        "candidate_commit": commit,
        "candidate_source_sha256": candidate_hash,
        "current_source_sha256": current_hash,
        "unchanged_since_candidate_commit": True,
    }


def _expected_trace_id(manifest: dict, item: dict, group: ExperimentGroup) -> str:
    value = (
        f"{manifest['config_sha256']}|{item['schedule_index']}|"
        f"{item['episode_id']}|{group.value}"
    )
    return "b1-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def main() -> int:
    # All pre-unblinding artifacts are study-manager material.  A restrictive
    # process umask keeps episodes, traces, manifests, and keys private even if
    # the host account has a permissive default umask.
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction-key-out", type=Path, required=True)
    parser.add_argument("--frozen-codebook", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    out = args.out.resolve()
    key_path = args.prediction_key_out.resolve()
    codebook_path = args.frozen_codebook.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_audit_config(config)
    if any(
        "sealed-method-validation" in str(path).lower() or path.suffix == ".aesgcm"
        for path in (config_path, out, codebook_path)
    ):
        raise RuntimeError("method-validation archive paths are prohibited")
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("independent audit requires a clean tracked worktree")
    judge_lock = _verify_judge_candidate(config)
    episodes = generate_judge_audit_episodes(FROZEN_AUDIT_CONFIG)
    episodes_by_id = {episode.episode_id: episode for episode in episodes}
    dedup = audit_deduplication(episodes, generate_stage_a_episodes())
    if not dedup["passed"]:
        raise RuntimeError(f"audit episode de-duplication failed: {dedup}")
    schedule = build_audit_schedule(episodes)
    config_sha256 = _sha256_file(config_path)
    git_commit = _git("rev-parse", "HEAD")
    manifest = {
        "audit_version": config["protocol_version"],
        "status": "protocol_frozen_before_api_calls",
        "mode": "independent_judge_audit",
        "config": config,
        "config_sha256": config_sha256,
        "benchmark_sha256": benchmark_sha256(episodes),
        "git": {
            "commit": git_commit,
            "branch": _git("branch", "--show-current"),
            "tracked_worktree_dirty": False,
        },
        "judge_candidate": judge_lock,
        "generator_source_sha256": _sha256_file(
            ROOT / "eval/reliability_bench/judge_audit_v2.py"
        ),
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "package_source_sha256": _sha256_file(
            ROOT / "eval/reliability_bench/judge_audit_package_v2.py"
        ),
        "deduplication": dedup,
        "method_validation_archive_accessed": False,
        "human_annotations_seen": False,
    }
    traces_path = out / "traces.jsonl"
    if out.exists() and not args.resume:
        raise RuntimeError("audit output already exists; use a new directory or --resume")
    if args.resume:
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        for field in ("config_sha256", "benchmark_sha256", "judge_candidate"):
            if previous[field] != manifest[field]:
                raise RuntimeError(f"resume manifest mismatch: {field}")
    else:
        out.mkdir(parents=True, exist_ok=False)
        _write_json(out / "manifest.json", manifest)
        _write_json(out / "config.lock.json", config)
        _write_json(out / "deduplication_report.json", dedup)
        _write_json(out / "schedule.json", schedule)
        _write_jsonl(out / "audit_episodes.jsonl", [episode.to_dict() for episode in episodes])

    traces = _read_jsonl(traces_path) if args.resume else []
    completed = {trace["trace_id"] for trace in traces}
    total_cost = sum(float(trace["llm"]["cost"]["estimated_cost"]) for trace in traces)
    client = DeepSeekCompletionClient(_load_env_key(), config)
    for item in schedule:
        episode = episodes_by_id[item["episode_id"]]
        group = ExperimentGroup(item["controller_group"])
        trace_id = _expected_trace_id(manifest, item, group)
        if trace_id in completed:
            continue
        if total_cost >= float(config["run_cost_cap_usd"]):
            raise RuntimeError("preregistered audit cost cap reached")
        trace = run_b1_trace(episode, group, client, config, item, manifest)
        append_jsonl(traces_path, trace)
        traces.append(trace)
        completed.add(trace["trace_id"])
        total_cost += float(trace["llm"]["cost"]["estimated_cost"])
        print(
            json.dumps(
                {
                    "progress": f"{len(traces)}/{AUDIT_EPISODE_COUNT}",
                    "opaque_episode": hashlib.sha256(
                        episode.episode_id.encode("utf-8")
                    ).hexdigest()[:12],
                    "hidden_group_id": item["hidden_group_id"],
                    "complete": trace["trace_complete"],
                    "api_error_attempts": len(trace["llm"]["api_errors"]),
                    "estimated_cost_usd": trace["llm"]["cost"]["estimated_cost"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    returned_models = sorted(
        {
            model
            for trace in traces
            for model in trace["llm"].get("returned_model_ids", [])
        }
    )
    technical_gate = {
        "trace_count_24": len(traces) == AUDIT_EPISODE_COUNT,
        "trace_ids_unique": len({trace["trace_id"] for trace in traces}) == AUDIT_EPISODE_COUNT,
        "all_traces_complete": all(trace["trace_complete"] and not trace["unscorable"] for trace in traces),
        "returned_model_ids": returned_models,
        "returned_model_exactly_frozen": returned_models == [config["requested_model"]],
        "all_within_budget": all(
            trace["budget_usage"]["turns"] <= config["max_turns"]
            and trace["budget_usage"]["tool_calls"] <= config["max_tool_calls"]
            and trace["budget_usage"]["cost_units"] <= config["max_cost_units"]
            and trace["budget_usage"]["completion_tokens"] <= config["max_tokens"]
            for trace in traces
        ),
    }
    technical_gate["passed"] = all(
        value for key, value in technical_gate.items() if key != "returned_model_ids"
    )
    _write_json(out / "technical_gate.json", technical_gate)
    if not technical_gate["passed"]:
        _write_json(
            out / "audit_status.json",
            {
                "status": "technical_gate_failed_development_only",
                "human_annotation_allowed": False,
                "prediction_commitment_created": False,
                "technical_gate": technical_gate,
            },
        )
        return 2

    prediction_manifest = seal_judge_predictions(
        traces=traces,
        episodes=episodes_by_id,
        output_dir=out / "sealed_judge_predictions",
        key_path=key_path,
        judge_candidate_commit=config["judge_candidate_commit"],
        judge_source_path=JUDGE_SOURCE,
        audit_config_path=config_path,
    )
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    package_manifest = build_blinded_review_package(
        traces=traces,
        output_dir=out / "blind_review",
        frozen_codebook=codebook,
        prediction_commitment=prediction_manifest,
        audit_config_sha256=config_sha256,
    )
    final = {
        "status": "independent_blind_annotation_pending",
        "trace_count": len(traces),
        "technical_gate": technical_gate,
        "estimated_cost_usd": round(total_cost, 8),
        "prediction_commitment_sha256": _sha256_file(
            out / "sealed_judge_predictions/prediction_commitment_manifest.json"
        ),
        "blind_package_manifest_sha256": _sha256_file(
            out / "blind_review/study_manager_only/package_manifest.json"
        ),
        "human_annotations_seen": False,
        "judge_freeze_allowed": False,
        "method_validation_archive_accessed": False,
        "package_status": package_manifest["status"],
    }
    _write_json(out / "audit_status.json", final)
    print(json.dumps(final, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
