"""QC-Agent-Bench Runner — evaluates systems on the 60-task benchmark.

Usage:
    python benchmark/runner.py --systems all --tiers 1,2,3 --output results.json
    python benchmark/runner.py --systems react_drift --tiers 1 --fast
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.tasks import TASKS, BenchmarkTask, get_tier
from backends.fake_adapter import FakeBackendAdapter
from backends.synthetic_drift import SyntheticDriftBackend, STABLE, LINEAR_DECAY, SUDDEN_DEGRADATION, DIURNAL_CYCLE
from agent.react import ReActAgent, AgentTrace
from agent.baselines import StaticPipeline, LLMSingleShot, ReActNoBudget


# ═══════════════════════════════════════════════════════
# Evaluation Result
# ═══════════════════════════════════════════════════════

@dataclass
class EvalResult:
    task_id: str
    system: str
    success: bool
    fidelity: float | None
    tool_calls: int
    wall_time_s: float
    tools_used: list[str]
    error: str | None = None
    requested_provider: str | None = None
    resolved_provider: str | None = None
    model: str | None = None
    final_answer_length: int | None = None
    no_final_answer: bool = False
    invalid_tool_call_count: int = 0
    malformed_json_count: int = 0
    repeated_tool_count: int = 0
    max_turns_exceeded: bool = False
    hallucinated_tool_names: list[str] | None = None
    trace: dict[str, Any] | None = None
    # F5: token and cost tracking
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


# ═══════════════════════════════════════════════════════
# System Factory
# ═══════════════════════════════════════════════════════

DRIFT_PROFILES = {
    "linear_decay": LINEAR_DECAY,
    "sudden": SUDDEN_DEGRADATION,
    "diurnal": DIURNAL_CYCLE,
    None: STABLE,
}


def make_backend(task: BenchmarkTask):
    """Create appropriate backend for a task."""
    profile = DRIFT_PROFILES.get(task.drift_profile, STABLE)
    backend = SyntheticDriftBackend(task.backend, profile)
    if task.drift_time is not None:
        backend.set_time(task.drift_time)
    return backend


def make_systems(
    backend,
    systems_list: list[str],
    provider: str = "mock",
    model: str | None = None,
    base_url: str | None = None,
    verbose: bool = True,
) -> dict:
    """Create requested systems."""
    available = {}

    def add_react_system(name: str, **kwargs: Any) -> None:
        system = ReActAgent(backend, **{**react_kwargs, **kwargs})
        if provider not in ("mock", "auto") and getattr(system, "provider", None) == "mock":
            raise RuntimeError(
                f"System {name} requested provider '{provider}' but fell back to mock. "
                "Check API key, SDK dependency, and base URL before running real LLM evaluation."
            )
        available[name] = system

    react_kwargs: dict[str, Any] = {
        "provider": provider,
        "use_memory": False,
        "use_drift_aware": False,
        "verbose": verbose,
    }
    if model:
        react_kwargs["model"] = model
    if base_url:
        react_kwargs["base_url"] = base_url

    if "static" in systems_list or "all" in systems_list:
        available["static"] = StaticPipeline(backend)

    if "single_shot" in systems_list or "all" in systems_list:
        available["single_shot"] = LLMSingleShot(backend)

    if "react_no_budget" in systems_list or "all" in systems_list:
        available["react_no_budget"] = ReActNoBudget(backend)

    if "react_full" in systems_list or "all" in systems_list:
        add_react_system("react_full")

    if "react_memory" in systems_list or "all" in systems_list:
        add_react_system("react_memory", use_memory=True, use_drift_aware=False)

    if "react_drift" in systems_list or "all" in systems_list:
        add_react_system("react_drift", use_memory=True, use_drift_aware=True)
    return available


def select_tasks(
    tiers: list[int],
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
) -> list[BenchmarkTask]:
    """Select benchmark tasks by tier, optional explicit IDs, and optional limit."""
    tasks = [t for t in TASKS if t.tier in tiers]

    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.id in wanted]

    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    return tasks



# ═══════════════════════════════════════════════════════
# Success Evaluation
# ═══════════════════════════════════════════════════════

def evaluate_success(task: BenchmarkTask, trace: AgentTrace) -> bool:
    """Check if a trace satisfies the task's success criteria."""
    criteria = task.success_criteria

    # Check must_call
    if "must_call" in criteria:
        tools_used = {s.action for s in trace.steps if s.action}
        for tool in criteria["must_call"]:
            if tool not in tools_used:
                return False

    # Check min_tool_calls
    if "min_tool_calls" in criteria:
        if trace.num_tool_calls < criteria["min_tool_calls"]:
            return False

    # Check min_circuits_run
    if "min_circuits_run" in criteria:
        n_runs = sum(1 for s in trace.steps if s.action == "run_circuit")
        if n_runs < criteria["min_circuits_run"]:
            return False

    # Check fidelity_above
    if "fidelity_above" in criteria:
        fids = [s.fidelity_observed for s in trace.steps if s.fidelity_observed]
        if not fids or max(fids) < criteria["fidelity_above"]:
            return False

    # Check must_detect_drift
    if criteria.get("must_detect_drift"):
        # Check if any step mentions drift
        for s in trace.steps:
            if s.observation and "drift" in s.observation.lower():
                try:
                    obs = json.loads(s.observation)
                    if obs.get("drift_score", 0) > 0.1 or obs.get("drift_detected"):
                        break
                except (json.JSONDecodeError, AttributeError):
                    if "drift" in str(s.observation).lower():
                        break
        else:
            return False

    return True


# ═══════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════

def run_benchmark(
    tiers: list[int] | None = None,
    systems_list: list[str] | None = None,
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
    provider: str = "mock",
    model: str | None = None,
    base_url: str | None = None,
    verbose: bool = False,
    save_traces: bool = False,
) -> list[EvalResult]:
    """Run the full benchmark."""
    if tiers is None:
        tiers = [1, 2, 3]
    if systems_list is None:
        systems_list = ["all"]
    tasks = select_tasks(tiers=tiers, task_ids=task_ids, max_tasks=max_tasks)
    results = []

    total_systems = 6 if "all" in systems_list else len(systems_list)
    total = len(tasks) * total_systems
    done = 0
    t0 = time.time()

    for task in tasks:
        backend = make_backend(task)
        systems = make_systems(
            backend,
            systems_list,
            provider=provider,
            model=model,
            base_url=base_url,
            verbose=verbose,
        )

        for sys_name, system in systems.items():
            done += 1
            if verbose:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f"  [{done}/{total}] {task.id} × {sys_name} "
                      f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

            try:
                t_start = time.time()
                trace = system.run(task.prompt)
                wall_time = time.time() - t_start

                success = evaluate_success(task, trace)
                fids = [s.fidelity_observed for s in trace.steps if s.fidelity_observed]
                fidelity = max(fids) if fids else None
                tools_used = [s.action for s in trace.steps if s.action]

                diagnostics = getattr(trace, "diagnostics", None)
                trace_payload = None
                if save_traces and hasattr(trace, "to_dict"):
                    trace_payload = trace.to_dict(include_observations=True)

                results.append(EvalResult(
                    task_id=task.id, system=sys_name,
                    success=success, fidelity=fidelity,
                    tool_calls=trace.num_tool_calls,
                    wall_time_s=round(wall_time, 3),
                    tools_used=tools_used,
                    requested_provider=getattr(diagnostics, "requested_provider", provider) if diagnostics else provider,
                    resolved_provider=getattr(diagnostics, "resolved_provider", None) if diagnostics else None,
                    model=getattr(diagnostics, "model", model) if diagnostics else model,
                    final_answer_length=getattr(diagnostics, "final_answer_length", None) if diagnostics else None,
                    no_final_answer=getattr(diagnostics, "no_final_answer", False) if diagnostics else False,
                    invalid_tool_call_count=getattr(diagnostics, "invalid_tool_call_count", 0) if diagnostics else 0,
                    malformed_json_count=getattr(diagnostics, "malformed_json_count", 0) if diagnostics else 0,
                    repeated_tool_count=getattr(diagnostics, "repeated_tool_count", 0) if diagnostics else 0,
                    max_turns_exceeded=getattr(diagnostics, "max_turns_exceeded", False) if diagnostics else False,
                    hallucinated_tool_names=getattr(diagnostics, "hallucinated_tool_names", []) if diagnostics else [],
                    trace=trace_payload,
                    total_tokens=getattr(trace, "total_tokens", 0),
                    prompt_tokens=getattr(trace, "prompt_tokens_total", 0),
                    completion_tokens=getattr(trace, "completion_tokens_total", 0),
                    cost_usd=getattr(trace, "cost_summary", {}).get("total_cost_usd", 0.0) if hasattr(trace, "cost_summary") else 0.0,
                ))
            except Exception as e:
                results.append(EvalResult(
                    task_id=task.id, system=sys_name,
                    success=False, fidelity=None,
                    tool_calls=0, wall_time_s=0,
                    tools_used=[], error=str(e),
                ))

    return results


# ═══════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════

def generate_report(results: list[EvalResult]) -> str:
    """Generate a text report from results."""
    LABELS = {
        "static": "Static", "single_shot": "LLM-1Shot",
        "react_no_budget": "ReAct", "react_full": "+Budget",
        "react_memory": "+Memory", "react_drift": "Full(Ours)",
    }

    lines = ["", "=" * 70, "  QC-Agent-Bench Results", "=" * 70]

    # Aggregate by system
    sys_agg = defaultdict(lambda: {"succ": 0, "n": 0, "fids": [], "calls": [], "times": []})
    for r in results:
        a = sys_agg[r.system]
        a["n"] += 1
        if r.success: a["succ"] += 1
        if r.fidelity: a["fids"].append(r.fidelity)
        a["calls"].append(r.tool_calls)
        a["times"].append(r.wall_time_s)

    # Overall table
    lines.append("\n  Overall Performance:")
    lines.append(f"  {'System':<12} {'Success%':>8} {'AvgFid':>8} {'AvgCalls':>9} {'AvgTime':>8}")
    lines.append("  " + "-" * 50)

    systems_order = ["static", "single_shot", "react_no_budget", "react_full", "react_memory", "react_drift"]
    for s in systems_order:
        if s not in sys_agg:
            continue
        a = sys_agg[s]
        rate = a["succ"] / a["n"] * 100 if a["n"] else 0
        avg_fid = sum(a["fids"]) / len(a["fids"]) if a["fids"] else 0
        avg_calls = sum(a["calls"]) / len(a["calls"]) if a["calls"] else 0
        avg_time = sum(a["times"]) / len(a["times"]) if a["times"] else 0
        label = LABELS.get(s, s)
        lines.append(f"  {label:<12} {rate:>7.1f}% {avg_fid:>8.4f} {avg_calls:>9.1f} {avg_time:>7.2f}s")

    # Per-tier breakdown
    for tier in [1, 2, 3]:
        tier_results = [r for r in results if r.task_id.startswith(f"T{tier}")]
        if not tier_results:
            continue
        tier_names = {1: "Static", 2: "Drift", 3: "Failure"}
        lines.append(f"\n  Tier {tier} ({tier_names[tier]}):")
        lines.append(f"  {'System':<12} {'Success%':>8} {'AvgFid':>8}")
        lines.append("  " + "-" * 30)

        for s in systems_order:
            sr = [r for r in tier_results if r.system == s]
            if not sr:
                continue
            rate = sum(1 for r in sr if r.success) / len(sr) * 100
            fids = [r.fidelity for r in sr if r.fidelity]
            avg_fid = sum(fids) / len(fids) if fids else 0
            label = LABELS.get(s, s)
            lines.append(f"  {label:<12} {rate:>7.1f}% {avg_fid:>8.4f}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="QC-Agent-Bench Runner")
    parser.add_argument("--systems", default="all", help="Comma-separated systems or 'all'")
    parser.add_argument("--tiers", default="1,2,3", help="Comma-separated tiers")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task IDs, e.g. T1-01,T1-06")
    parser.add_argument("--max-tasks", type=int, default=None, help="Maximum number of selected tasks to run")
    parser.add_argument("--provider", default="mock", help="Planner provider: mock, deepseek, openai, anthropic, or auto")
    parser.add_argument("--model", default=None, help="Model name, e.g. deepseek-chat")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--output", default="benchmark/results.json", help="Output file")
    parser.add_argument("--save-traces", action="store_true", help="Include full per-run traces in the JSON output")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    tiers = [int(t) for t in args.tiers.split(",") if t.strip()]
    task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()] if args.task_ids else None
    selected_tasks = select_tasks(tiers=tiers, task_ids=task_ids, max_tasks=args.max_tasks)

    print(f"QC-Agent-Bench: {len(selected_tasks)} tasks × {len(systems)} systems")
    print(f"Tiers: {tiers}, Systems: {systems}, Provider: {args.provider}")
    if task_ids:
        print(f"Task IDs: {task_ids}")

    results = run_benchmark(
        tiers=tiers,
        systems_list=systems,
        task_ids=task_ids,
        max_tasks=args.max_tasks,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        verbose=args.verbose,
        save_traces=args.save_traces,
    )

    # Report
    report = generate_report(results)
    print(report)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
