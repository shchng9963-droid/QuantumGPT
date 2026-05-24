"""End-to-end evaluation: 7 tasks × 4 systems.

Produces a performance table for the paper.
"""

from __future__ import annotations

import json
import time
import sys
import traceback
from dataclasses import dataclass, field

from backends.fake_adapter import FakeBackendAdapter
from agent.baselines import get_all_systems


# ═══════════════════════════════════════════════════════
# 7 evaluation tasks
# ═══════════════════════════════════════════════════════

EVAL_TASKS = [
    {
        "id": "T1",
        "name": "Health Check + GHZ-5",
        "prompt": "Check the backend health and run the GHZ-5 circuit. Report the fidelity.",
        "success_criteria": lambda trace: any(
            s.action == "run_circuit" and s.fidelity_observed is not None
            for s in trace.steps
        ),
        "extract_metric": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
    {
        "id": "T2",
        "name": "Multi-Circuit Benchmark",
        "prompt": "Run ghz_5, qft_4, and bv_5 circuits and compare their fidelities.",
        "success_criteria": lambda trace: sum(
            1 for s in trace.steps if s.action == "run_circuit"
        ) >= 2,
        "extract_metric": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
    {
        "id": "T3",
        "name": "Diagnose + Mitigate",
        "prompt": "Run the GHZ-5 circuit. If fidelity is below 0.95, diagnose the issue and try error mitigation.",
        "success_criteria": lambda trace: any(
            s.action in ("apply_mitigation", "diagnose_and_suggest")
            for s in trace.steps
        ),
        "extract_metric": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
    {
        "id": "T4",
        "name": "Fidelity Prediction",
        "prompt": "Predict the fidelity of the ghz_5 circuit, then run it to verify the prediction.",
        "success_criteria": lambda trace: (
            any(s.action == "predict_fidelity" for s in trace.steps) or
            any(s.action == "run_circuit" for s in trace.steps)
        ),
        "extract_metric": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
    {
        "id": "T5",
        "name": "Transpile + Run",
        "prompt": "Transpile the qft_4 circuit with optimization level 2, then run it.",
        "success_criteria": lambda trace: (
            any(s.action == "transpile_circuit" for s in trace.steps) or
            any(s.action == "run_circuit" for s in trace.steps)
        ),
        "extract_metric": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
    {
        "id": "T6",
        "name": "Rabi Experiment",
        "prompt": "Run a Rabi oscillation experiment at 5 GHz with a gaussian pulse, fit the data, and report the pi-pulse amplitude.",
        "success_criteria": lambda trace: any(
            s.action == "rabi_experiment" for s in trace.steps
        ),
        "extract_metric": lambda trace: next(
            (
                _safe_json(s.observation).get("pi_amplitude_mhz")
                for s in trace.steps if s.action in ("fit_rabi", "rabi_experiment") and s.observation
            ),
            None,
        ),
    },
    {
        "id": "T7",
        "name": "Full Pipeline",
        "prompt": (
            "Check the backend health, run the GHZ-5 circuit, "
            "if fidelity is below 0.95 apply error mitigation, "
            "then diagnose any remaining issues."
        ),
        "success_criteria": lambda trace: (
            any(s.action == "get_backend_health" for s in trace.steps) and
            any(s.action == "run_circuit" for s in trace.steps)
        ),
        "extract_metric": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
]

def _safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════

@dataclass
class EvalResult:
    task_id: str
    task_name: str
    system: str
    success: bool
    metric: float | None
    num_tool_calls: int
    elapsed_seconds: float
    error: str | None = None


def run_evaluation(
    n_seeds: int = 3,
    verbose: bool = False,
    target_fidelity: float = 0.85,
) -> list[EvalResult]:
    """Run all tasks × all systems × n_seeds."""
    results: list[EvalResult] = []

    for seed in range(n_seeds):
        be = FakeBackendAdapter("FakeBrisbane")
        systems = get_all_systems(be, verbose=verbose, target_fidelity=target_fidelity)

        for task in EVAL_TASKS:
            for sys_name, system in systems.items():
                try:
                    trace = system.run(task["prompt"])
                    success = task["success_criteria"](trace)
                    metric = task["extract_metric"](trace)
                    results.append(EvalResult(
                        task_id=task["id"],
                        task_name=task["name"],
                        system=sys_name,
                        success=success,
                        metric=round(float(metric), 4) if metric is not None else None,
                        num_tool_calls=trace.num_tool_calls,
                        elapsed_seconds=round(trace.elapsed_seconds, 2),
                    ))
                except Exception as e:
                    results.append(EvalResult(
                        task_id=task["id"],
                        task_name=task["name"],
                        system=sys_name,
                        success=False,
                        metric=None,
                        num_tool_calls=0,
                        elapsed_seconds=0.0,
                        error=str(e),
                    ))

                # Progress
                sys.stdout.write(".")
                sys.stdout.flush()

    print()
    return results


def print_table(results: list[EvalResult]):
    """Print a readable performance comparison table."""
    systems = ["static", "single_shot", "react_no_budget", "react_full"]
    tasks = list(dict.fromkeys(r.task_id for r in results))

    # Aggregate over seeds
    from collections import defaultdict
    agg = defaultdict(lambda: {"successes": 0, "count": 0, "metrics": [], "calls": [], "times": []})

    for r in results:
        key = (r.task_id, r.system)
        a = agg[key]
        a["count"] += 1
        if r.success:
            a["successes"] += 1
        if r.metric is not None:
            a["metrics"].append(r.metric)
        a["calls"].append(r.num_tool_calls)
        a["times"].append(r.elapsed_seconds)

    # Header
    hdr = f"{'Task':<8} {'Name':<22}"
    for s in systems:
        hdr += f" | {s:>16}"
    print("\n" + "=" * len(hdr))
    print("QUANTUMGPT EVALUATION: 7 Tasks × 4 Systems")
    print("=" * len(hdr))

    # Success rate
    print(f"\n--- Success Rate ---")
    print(hdr)
    print("-" * len(hdr))
    for tid in tasks:
        tname = next(r.task_name for r in results if r.task_id == tid)
        row = f"{tid:<8} {tname:<22}"
        for s in systems:
            a = agg[(tid, s)]
            rate = a["successes"] / a["count"] if a["count"] > 0 else 0
            row += f" | {rate:>15.0%}"
        print(row)

    # Average metric (fidelity / pi-amp)
    print(f"\n--- Primary Metric (avg) ---")
    print(hdr)
    print("-" * len(hdr))
    for tid in tasks:
        tname = next(r.task_name for r in results if r.task_id == tid)
        row = f"{tid:<8} {tname:<22}"
        for s in systems:
            a = agg[(tid, s)]
            if a["metrics"]:
                avg = sum(a["metrics"]) / len(a["metrics"])
                row += f" | {avg:>15.4f}"
            else:
                row += f" | {'N/A':>15}"
        print(row)

    # Tool calls
    print(f"\n--- Tool Calls (avg) ---")
    print(hdr)
    print("-" * len(hdr))
    for tid in tasks:
        tname = next(r.task_name for r in results if r.task_id == tid)
        row = f"{tid:<8} {tname:<22}"
        for s in systems:
            a = agg[(tid, s)]
            avg_calls = sum(a["calls"]) / len(a["calls"]) if a["calls"] else 0
            row += f" | {avg_calls:>15.1f}"
        print(row)

    # Elapsed time
    print(f"\n--- Elapsed Time (avg, sec) ---")
    print(hdr)
    print("-" * len(hdr))
    for tid in tasks:
        tname = next(r.task_name for r in results if r.task_id == tid)
        row = f"{tid:<8} {tname:<22}"
        for s in systems:
            a = agg[(tid, s)]
            avg_t = sum(a["times"]) / len(a["times"]) if a["times"] else 0
            row += f" | {avg_t:>15.1f}"
        print(row)

    # System-level aggregates
    print(f"\n--- System Averages ---")
    print(f"{'System':<20} {'Success%':>10} {'Metric':>10} {'Calls':>8} {'Time(s)':>10}")
    print("-" * 62)
    for s in systems:
        all_success = sum(agg[(t, s)]["successes"] for t in tasks)
        all_count = sum(agg[(t, s)]["count"] for t in tasks)
        all_metrics = [m for t in tasks for m in agg[(t, s)]["metrics"]]
        all_calls = [c for t in tasks for c in agg[(t, s)]["calls"]]
        all_times = [t_ for t in tasks for t_ in agg[(t, s)]["times"]]

        sr = all_success / all_count if all_count > 0 else 0
        avg_m = sum(all_metrics) / len(all_metrics) if all_metrics else 0
        avg_c = sum(all_calls) / len(all_calls) if all_calls else 0
        avg_t = sum(all_times) / len(all_times) if all_times else 0

        print(f"{s:<20} {sr:>9.0%} {avg_m:>10.4f} {avg_c:>8.1f} {avg_t:>10.1f}")


def export_json(results: list[EvalResult], path: str = "eval_results.json"):
    import json
    data = [
        {
            "task_id": r.task_id,
            "task_name": r.task_name,
            "system": r.system,
            "success": r.success,
            "metric": r.metric,
            "num_tool_calls": r.num_tool_calls,
            "elapsed_seconds": r.elapsed_seconds,
            "error": r.error,
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    print("Running QuantumGPT evaluation: 7 tasks × 4 systems × 3 seeds")
    results = run_evaluation(n_seeds=3, verbose=False)
    print_table(results)
    export_json(results, "eval_results.json")
