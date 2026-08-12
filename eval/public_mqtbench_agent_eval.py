#!/usr/bin/env python3
"""Real-LLM public MQTBench agent evaluation against deterministic oracle bounds."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.synthetic_drift import STABLE, SyntheticDriftBackend
from eval.drift_experiment import (
    _extract_fidelities,
    _save_trace,
    _trace_tool_names,
    _usage_metadata_from_traces,
    make_real_agent,
)
from eval.paper_grade.result_schema import PaperGradeResultRow, write_result_jsonl
from eval.public_mqtbench_oracle import PROFILE_REGISTRY


@dataclass
class PublicAgentRunResult:
    system: str
    task_id: str
    circuit: str
    backend: str
    drift_profile: str
    seed: int
    target_fidelity: float
    oracle_best: float
    oracle_feasible: bool
    oracle_best_action: str
    pre_fidelities: list[float]
    post_fidelities: list[float]
    provider: str
    model: str
    tool_calls: list[str]
    trace_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_fidelities(self) -> list[float]:
        return self.pre_fidelities + self.post_fidelities

    @property
    def first_post_fidelity(self) -> float | None:
        return self.post_fidelities[0] if self.post_fidelities else None

    @property
    def best_post_fidelity(self) -> float | None:
        return max(self.post_fidelities) if self.post_fidelities else None

    @property
    def final_fidelity(self) -> float | None:
        return self.post_fidelities[-1] if self.post_fidelities else None

    @property
    def post_drift_improvement(self) -> float | None:
        if self.first_post_fidelity is None or self.best_post_fidelity is None:
            return None
        return self.best_post_fidelity - self.first_post_fidelity

    @property
    def target_success(self) -> bool:
        return bool(self.best_post_fidelity is not None and self.best_post_fidelity >= self.target_fidelity)

    @property
    def oracle_gap(self) -> float | None:
        if self.best_post_fidelity is None:
            return None
        return self.oracle_best - self.best_post_fidelity


def load_oracle_rows(
    path: str | Path,
    *,
    profiles: set[str] | None = None,
    circuits: set[str] | None = None,
    only_feasible: bool = True,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if profiles is not None and row.get("profile") not in profiles:
            continue
        if circuits is not None and row.get("circuit") not in circuits:
            continue
        if only_feasible and not row.get("oracle_feasible", False):
            continue
        rows.append(row)
        if max_tasks is not None and len(rows) >= max_tasks:
            break
    return rows


def build_agent_prompt(oracle_row: dict[str, Any]) -> str:
    circuit = oracle_row["circuit"]
    target = float(oracle_row.get("target_fidelity", 0.85))
    oracle_best = float(oracle_row.get("best_post_drift_score", 0.0))
    best_action = oracle_row.get("best_action", "unknown")
    profile = oracle_row.get("profile", "unknown")
    return (
        f"Run public benchmark circuit {circuit} on the current backend under the {profile} drift setting. "
        f"Target fidelity is {target:.2f}. The deterministic oracle best observed post-drift fidelity for this task "
        f"is {oracle_best:.4f} using action {best_action}; use this only as a benchmark target, not as a tool result. "
        "First inspect current backend health and drift. Then choose tools to run, transpile, predict, or mitigate the circuit. "
        "Keep track of the best observed fidelity; if a later action is worse, report the best observed value rather than only the final one. "
        "Give concrete fidelity numbers and say whether the target was met."
    )


def _pre_prompt(oracle_row: dict[str, Any]) -> str:
    circuit = oracle_row["circuit"]
    return (
        f"Run public benchmark circuit {circuit} on the current backend as a pre-drift reference. "
        "Report the observed fidelity number."
    )


def _profile_and_time(profile_name: str):
    if profile_name not in PROFILE_REGISTRY:
        raise ValueError(f"Unknown oracle profile {profile_name!r}; available={sorted(PROFILE_REGISTRY)}")
    return PROFILE_REGISTRY[profile_name]


def run_public_agent_task(
    oracle_row: dict[str, Any],
    *,
    system: str = "QuantumGPT-Full",
    provider: str = "auto",
    model: str | None = None,
    seed: int | None = None,
    out_dir: str | Path = "eval/results/public_mqtbench_agent_deepseek",
    verbose: bool = False,
    max_turns: int = 8,
    max_tool_calls: int = 16,
) -> PublicAgentRunResult:
    """Run one real agent pre/post-drift public benchmark task."""

    use_drift_aware = system != "QuantumGPT-No-Drift"
    circuit = str(oracle_row["circuit"])
    backend_name = str(oracle_row.get("backend", "FakeBrisbane"))
    profile_name = str(oracle_row.get("profile", "moderate"))
    task_seed = int(seed if seed is not None else oracle_row.get("seed", 0))
    target = float(oracle_row.get("target_fidelity", 0.85))
    profile, post_time = _profile_and_time(profile_name)

    backend = SyntheticDriftBackend(backend_name, STABLE)
    backend.set_time(0.0)
    backend.profile = STABLE

    agent = make_real_agent(
        backend,
        provider=provider,
        model=model,
        use_drift_aware=use_drift_aware,
        target_fidelity=target,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        verbose=verbose,
    )

    t0 = time.time()
    pre_trace = agent.run(_pre_prompt(oracle_row))

    backend.profile = profile
    backend.set_time(post_time)
    post_trace = agent.run(build_agent_prompt(oracle_row))
    elapsed = time.time() - t0

    result_dir = Path(out_dir) / system.replace(" ", "_").replace("/", "_") / profile_name / circuit / f"seed{task_seed}"
    trace_files = [
        _save_trace(pre_trace, result_dir, "pre_drift_trace"),
        _save_trace(post_trace, result_dir, "post_drift_trace"),
    ]

    metadata = {
        "execution_mode": "real",
        "use_mock": False,
        "seed": task_seed,
        "max_turns": max_turns,
        "max_tool_calls": max_tool_calls,
        "pre_fidelity_count": len(_extract_fidelities(pre_trace)),
        "post_fidelity_count": len(_extract_fidelities(post_trace)),
        "pre_final_answer_chars": len(pre_trace.final_answer or ""),
        "post_final_answer_chars": len(post_trace.final_answer or ""),
        "elapsed_seconds": elapsed,
        "oracle_row": oracle_row,
        **_usage_metadata_from_traces([pre_trace, post_trace]),
    }

    return PublicAgentRunResult(
        system=system,
        task_id=str(oracle_row.get("task_id", f"{backend_name}:{profile_name}:{circuit}:seed{task_seed}")),
        circuit=circuit,
        backend=backend_name,
        drift_profile=profile_name,
        seed=task_seed,
        target_fidelity=target,
        oracle_best=float(oracle_row.get("best_post_drift_score", 0.0)),
        oracle_feasible=bool(oracle_row.get("oracle_feasible", False)),
        oracle_best_action=str(oracle_row.get("best_action", "unknown")),
        pre_fidelities=_extract_fidelities(pre_trace),
        post_fidelities=_extract_fidelities(post_trace),
        provider=agent.provider,
        model=agent.model,
        tool_calls=_trace_tool_names(pre_trace) + _trace_tool_names(post_trace),
        trace_files=trace_files,
        metadata=metadata,
    )


DEFAULT_STATIC_ACTION = "transpile_o1_s1024"


def run_static_pipeline_task(
    oracle_row: dict[str, Any],
    *,
    action: str = DEFAULT_STATIC_ACTION,
) -> PublicAgentRunResult:
    """Non-adaptive baseline: apply one fixed deterministic action.

    Reads the chosen action's post-drift fidelity directly from the
    pre-computed oracle JSONL (no LLM, no oracle selection, no drift
    monitor). The result is wrapped in PublicAgentRunResult so it shares
    the paper-grade JSONL schema with the agent rows.
    """

    actions = {a["action"]: a for a in oracle_row.get("actions", [])}
    if action not in actions:
        raise KeyError(
            f"Static-Pipeline action {action!r} not found in oracle row "
            f"actions={sorted(actions)!r}"
        )
    chosen = actions[action]
    fid = float(chosen["fidelity"])
    seed = int(oracle_row.get("seed", 0))
    target = float(oracle_row.get("target_fidelity", 0.85))
    return PublicAgentRunResult(
        system="Static-Pipeline",
        task_id=str(oracle_row.get("task_id", f"{oracle_row.get('backend','')}:"
                                    f"{oracle_row.get('profile','')}:"
                                    f"{oracle_row.get('circuit','')}:seed{seed}")),
        circuit=str(oracle_row.get("circuit", "")),
        backend=str(oracle_row.get("backend", "FakeBrisbane")),
        drift_profile=str(oracle_row.get("profile", "moderate")),
        seed=seed,
        target_fidelity=target,
        oracle_best=float(oracle_row.get("best_post_drift_score", 0.0)),
        oracle_feasible=bool(oracle_row.get("oracle_feasible", False)),
        oracle_best_action=str(oracle_row.get("best_action", "unknown")),
        pre_fidelities=[float(oracle_row.get("pre_fidelity", 0.0))],
        post_fidelities=[fid],
        provider="none",
        model=f"static-{action}",
        tool_calls=[action],
        trace_files=[],
        metadata={
            "execution_mode": "simulator",
            "use_mock": False,
            "seed": seed,
            "max_turns": 0,
            "max_tool_calls": 1,
            "pre_fidelity_count": 1,
            "post_fidelity_count": 1,
            "pre_final_answer_chars": 0,
            "post_final_answer_chars": 0,
            "elapsed_seconds": float(oracle_row.get("elapsed_seconds", 0.0)),
            "oracle_row": oracle_row,
            "static_action": action,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "api_latency_ms": 0.0,
            "drift_alert_count": 0,
            "replan_triggered": False,
            "trace_diagnostics": [],
            "trace_cost_summaries": [],
        },
    )


def paper_grade_row_from_public_result(result: PublicAgentRunResult) -> PaperGradeResultRow:
    best_post = result.best_post_fidelity
    final = result.final_fidelity
    prompt_tokens = int(result.metadata.get("prompt_tokens", 0))
    completion_tokens = int(result.metadata.get("completion_tokens", 0))
    return PaperGradeResultRow(
        run_id=f"public_mqtbench:{result.system}:{result.task_id}:seed{result.seed}",
        system=result.system,
        task=f"Public MQTBench recovery: {result.task_id}",
        circuit=result.circuit,
        drift_profile=result.drift_profile,
        seed=result.seed,
        execution_mode="real" if result.provider not in {"none", "static"} else "simulator",
        provider=result.provider,
        model=result.model,
        use_mock=bool(result.metadata.get("use_mock", False)),
        trace_file=result.trace_files[-1] if result.trace_files else f"{result.system}:no-trace",
        target_fidelity=result.target_fidelity,
        final_score=final,
        best_score=best_post,
        target_success=result.target_success,
        tool_calls=len(result.tool_calls),
        max_tool_calls=int(result.metadata.get("max_tool_calls", len(result.tool_calls))),
        elapsed_seconds=float(result.metadata.get("elapsed_seconds", 0.0)),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_cost_usd=float(result.metadata.get("total_cost_usd", 0.0)),
        diagnostics={
            "task_id": result.task_id,
            "backend": result.backend,
            "tool_calls": result.tool_calls,
            "trace_files": result.trace_files,
            "oracle_feasible": result.oracle_feasible,
            "oracle_best_action": result.oracle_best_action,
            "drift_alert_count": int(result.metadata.get("drift_alert_count", 0)),
            "replan_triggered": bool(result.metadata.get("replan_triggered", False)),
            **result.metadata,
        },
        metrics={
            "pre_fidelities": result.pre_fidelities,
            "post_fidelities": result.post_fidelities,
            "first_post_drift_score": result.first_post_fidelity,
            "best_post_drift_score": best_post,
            "post_drift_improvement": result.post_drift_improvement,
            "post_drift_target_success": result.target_success,
            "oracle_best": result.oracle_best,
            "oracle_gap": result.oracle_gap,
            "oracle_feasible": result.oracle_feasible,
        },
        failure_reason=result.metadata.get("failure_reason"),
    )


def write_public_agent_results(results: list[PublicAgentRunResult], path: str | Path) -> None:
    write_result_jsonl(path, [paper_grade_row_from_public_result(r) for r in results])


def _parse_csv(value: str | None) -> set[str] | None:
    if value is None or value == "":
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-jsonl", required=True)
    parser.add_argument("--out-dir", default="eval/results/public_mqtbench_agent_deepseek")
    parser.add_argument("--systems", default="QuantumGPT-Full,QuantumGPT-No-Drift",
                        help="Comma-separated systems. May include "
                             "QuantumGPT-Full, QuantumGPT-No-Drift, Static-Pipeline, "
                             "Oracle-Adaptive, ChatLLM-NoTools.")
    parser.add_argument("--static-action", default=DEFAULT_STATIC_ACTION,
                        help="Action used by Static-Pipeline rows.")
    parser.add_argument("--chat-fallback-action", default=DEFAULT_STATIC_ACTION,
                        help="Action ChatLLM-NoTools falls back to when LLM "
                             "output is unparseable or names an unknown action.")
    parser.add_argument("--profiles", default="moderate")
    parser.add_argument("--circuits", default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--include-infeasible", action="store_true")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    rows = load_oracle_rows(
        args.oracle_jsonl,
        profiles=_parse_csv(args.profiles),
        circuits=_parse_csv(args.circuits),
        only_feasible=not args.include_infeasible,
        max_tasks=args.max_tasks,
    )
    systems = sorted(_parse_csv(args.systems) or {"QuantumGPT-Full"})
    seeds = [int(s) for s in sorted(_parse_csv(args.seeds) or {"0"})]

    results: list[PublicAgentRunResult] = []
    chat_client = None  # lazily built if ChatLLM-NoTools is requested
    for row in rows:
        for system in systems:
            for seed in seeds:
                print(
                    f"[agent-eval] system={system} seed={seed} task={row.get('task_id')} circuit={row.get('circuit')} profile={row.get('profile')}",
                    flush=True,
                )
                if system == "Static-Pipeline":
                    static_row = dict(row)
                    static_row["seed"] = seed
                    result = run_static_pipeline_task(static_row, action=args.static_action)
                elif system == "Oracle-Adaptive":
                    from eval.baselines.oracle_adaptive import run_oracle_adaptive_task
                    oa_row = dict(row)
                    oa_row["seed"] = seed
                    result = run_oracle_adaptive_task(oa_row)
                elif system == "ChatLLM-NoTools":
                    from eval.baselines.chat_llm_notools import (
                        build_deepseek_client,
                        run_chat_llm_no_tools_task,
                    )
                    if chat_client is None:
                        chat_client = build_deepseek_client()
                    chat_row = dict(row)
                    chat_row["seed"] = seed
                    result = run_chat_llm_no_tools_task(
                        chat_row,
                        client=chat_client,
                        model=args.model or "deepseek-chat",
                        provider="deepseek",
                        fallback_action=args.chat_fallback_action,
                    )
                else:
                    result = run_public_agent_task(
                        row,
                        system=system,
                        provider=args.provider,
                        model=args.model,
                        seed=seed,
                        out_dir=out_dir,
                        verbose=args.verbose,
                        max_turns=args.max_turns,
                        max_tool_calls=args.max_tool_calls,
                    )
                results.append(result)
                write_public_agent_results(results, out_dir / "result_summary.jsonl")

    summary = {
        "n_rows": len(results),
        "n_success": sum(1 for r in results if r.target_success),
        "success_rate": (sum(1 for r in results if r.target_success) / len(results)) if results else 0.0,
        "mean_best_post_drift_score": (sum((r.best_post_fidelity or 0.0) for r in results) / len(results)) if results else 0.0,
        "mean_oracle_gap": (sum((r.oracle_gap or 0.0) for r in results) / len(results)) if results else 0.0,
        "total_cost_usd": round(sum(float(r.metadata.get("total_cost_usd", 0.0)) for r in results), 6),
    }
    (out_dir / "compact_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
