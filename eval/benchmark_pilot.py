#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "eval" / "run_orchestrator_b10.py"
DEFAULT_ORACLE_JSONL = "eval/results/v3_oracle_full/oracle_results.jsonl"
DEFAULT_SYSTEMS = ["QuantumGPT-Orchestrator", "QuantumGPT-Orch-NoVerifier"]
DEFAULT_PROFILES = ["mild_gradual"]
DEFAULT_CIRCUITS = ["QAOA-6", "qft_4", "VQE_SU2-4", "GraphState-5", "vqe_4"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_MAX_TOOL_CALLS = 6
DEFAULT_MAX_SECONDS = 60
DEFAULT_MAX_USD = 2.0
DEFAULT_SOFT_WARN_USD = 1.5
DEFAULT_MAX_TASKS = None


def _csv(items: Sequence[object] | None) -> str | None:
    if items is None:
        return None
    return ",".join(str(item) for item in items)


def load_env_file(path: str | Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        loaded[key] = os.environ.get(key, value)
    return loaded


def build_pilot_command(
    *,
    oracle_jsonl: str = DEFAULT_ORACLE_JSONL,
    out_dir: str | Path = ROOT / "eval" / "results" / "v3_orchestrator_pilot",
    systems: Sequence[str] = DEFAULT_SYSTEMS,
    profiles: Sequence[str] = DEFAULT_PROFILES,
    circuits: Sequence[str] = DEFAULT_CIRCUITS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    max_usd: float = DEFAULT_MAX_USD,
    soft_warn_usd: float = DEFAULT_SOFT_WARN_USD,
    max_tasks: int | None = DEFAULT_MAX_TASKS,
    provider: str = "deepseek",
    include_infeasible: bool = False,
    use_stub_planner: bool = False,
    verbose: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--oracle-jsonl",
        str(oracle_jsonl),
        "--out-dir",
        str(out_dir),
        "--systems",
        _csv(systems) or "",
        "--profiles",
        _csv(profiles) or "",
        "--circuits",
        _csv(circuits) or "",
        "--seeds",
        _csv(seeds) or "",
        "--provider",
        provider,
        "--max-tool-calls",
        str(max_tool_calls),
        "--max-seconds",
        str(max_seconds),
        "--max-usd",
        str(max_usd),
        "--soft-warn-usd",
        str(soft_warn_usd),
    ]
    if max_tasks is not None:
        cmd.extend(["--max-tasks", str(max_tasks)])
    if include_infeasible:
        cmd.append("--include-infeasible")
    if use_stub_planner:
        cmd.append("--use-stub-planner")
    if verbose:
        cmd.append("--verbose")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch a calibrated benchmark pilot for orchestrator B10.")
    parser.add_argument("--oracle-jsonl", default=DEFAULT_ORACLE_JSONL)
    parser.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "v3_orchestrator_pilot"))
    parser.add_argument("--systems", default=_csv(DEFAULT_SYSTEMS))
    parser.add_argument("--profiles", default=_csv(DEFAULT_PROFILES))
    parser.add_argument("--circuits", default=_csv(DEFAULT_CIRCUITS))
    parser.add_argument("--seeds", default=_csv(DEFAULT_SEEDS))
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    parser.add_argument("--soft-warn-usd", type=float, default=DEFAULT_SOFT_WARN_USD)
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--include-infeasible", action="store_true")
    parser.add_argument("--use-stub-planner", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    cmd = build_pilot_command(
        oracle_jsonl=args.oracle_jsonl,
        out_dir=args.out_dir,
        systems=(args.systems or "").split(",") if args.systems else [],
        profiles=(args.profiles or "").split(",") if args.profiles else [],
        circuits=(args.circuits or "").split(",") if args.circuits else [],
        seeds=[int(x) for x in (args.seeds or "").split(",") if x],
        max_tool_calls=args.max_tool_calls,
        max_seconds=args.max_seconds,
        max_usd=args.max_usd,
        soft_warn_usd=args.soft_warn_usd,
        max_tasks=args.max_tasks,
        provider=args.provider,
        include_infeasible=args.include_infeasible,
        use_stub_planner=args.use_stub_planner,
        verbose=args.verbose,
    )
    print(" ".join(cmd))
    if args.print_only:
        return 0
    child_env = os.environ.copy()
    child_env.update(load_env_file(ROOT / ".env"))
    return subprocess.call(cmd, cwd=ROOT, env=child_env)


if __name__ == "__main__":
    raise SystemExit(main())
