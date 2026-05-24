"""6-System × 7-Task Full Comparison — Paper Table 1 data.

Systems:
  1. static          — Fixed pipeline (health → run → report)
  2. single_shot     — One LLM call to plan, then execute
  3. react_no_budget — ReAct loop, no fidelity gating
  4. react_full      — ReAct + fidelity budget
  5. react_memory    — ReAct + budget + ExperimentRecord memory
  6. react_drift     — ReAct + budget + memory + drift-aware replanning

Tasks: T1–T7 from eval/run_eval.py

Output: JSON results + formatted table for paper.

Usage:
    python eval/full_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backends.fake_adapter import FakeBackendAdapter
from backends.synthetic_drift import SyntheticDriftBackend, STABLE, SUDDEN_DEGRADATION
from agent.baselines import StaticPipeline, LLMSingleShot, ReActNoBudget
from agent.react import ReActAgent


# ═══════════════════════════════════════════════════════
# 7 evaluation tasks (same as run_eval.py)
# ═══════════════════════════════════════════════════════

def _safe_json(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


EVAL_TASKS = [
    {
        "id": "T1",
        "name": "Health + GHZ-5",
        "prompt": "Check the backend health and run the GHZ-5 circuit. Report the fidelity.",
        "success_criteria": lambda trace: any(
            s.action == "run_circuit" and s.fidelity_observed is not None
            for s in trace.steps
        ),
        "extract_fidelity": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
    {
        "id": "T2",
        "name": "Multi-Circuit",
        "prompt": "Run ghz_5, qft_4, and bv_5 circuits and compare their fidelities.",
        "success_criteria": lambda trace: sum(
            1 for s in trace.steps if s.action == "run_circuit"
        ) >= 2,
        "extract_fidelity": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
    {
        "id": "T3",
        "name": "Diagnose+Mitigate",
        "prompt": "Run the GHZ-5 circuit. If fidelity is below 0.95, diagnose the issue and try error mitigation.",
        "success_criteria": lambda trace: any(
            s.action in ("apply_mitigation", "diagnose_and_suggest")
            for s in trace.steps
        ),
        "extract_fidelity": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
    {
        "id": "T4",
        "name": "Fidelity Predict",
        "prompt": "Predict the fidelity of the ghz_5 circuit, then run it to verify the prediction.",
        "success_criteria": lambda trace: (
            any(s.action == "predict_fidelity" for s in trace.steps) or
            any(s.action == "run_circuit" for s in trace.steps)
        ),
        "extract_fidelity": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
    {
        "id": "T5",
        "name": "Transpile+Run",
        "prompt": "Transpile the qft_4 circuit with optimization level 2, then run it.",
        "success_criteria": lambda trace: (
            any(s.action == "transpile_circuit" for s in trace.steps) or
            any(s.action == "run_circuit" for s in trace.steps)
        ),
        "extract_fidelity": lambda trace: max(
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
        "extract_fidelity": lambda trace: next(
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
        "extract_fidelity": lambda trace: max(
            (s.fidelity_observed for s in trace.steps if s.fidelity_observed is not None),
            default=None
        ),
    },
]


# ═══════════════════════════════════════════════════════
# System factory
# ═══════════════════════════════════════════════════════

def get_6_systems(backend, verbose=False, target_fidelity=0.85):
    """Return all 6 systems for comparison."""
    return {
        "static": StaticPipeline(backend, verbose=verbose),
        "single_shot": LLMSingleShot(backend, verbose=verbose),
        "react_no_budget": ReActNoBudget(backend, verbose=verbose),
        "react_full": ReActAgent(
            backend, provider="mock", verbose=verbose,
            target_fidelity=target_fidelity,
            use_memory=False, use_drift_aware=False,
        ),
        "react_memory": ReActAgent(
            backend, provider="mock", verbose=verbose,
            target_fidelity=target_fidelity,
            use_memory=True, use_drift_aware=False,
        ),
        "react_drift": ReActAgent(
            backend, provider="mock", verbose=verbose,
            target_fidelity=target_fidelity,
            use_memory=True, use_drift_aware=True,
        ),
    }


# ═══════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════

@dataclass
class EvalResult:
    task_id: str
    task_name: str
    system: str
    success: bool
    fidelity: float | None
    num_tool_calls: int
    elapsed_seconds: float
    error: str | None = None


def run_full_comparison(n_seeds: int = 3, verbose: bool = False) -> list[EvalResult]:
    """Run 6 systems × 7 tasks × n_seeds."""
    results = []
    total = 6 * 7 * n_seeds
    count = 0
    t0 = time.time()

    for seed in range(n_seeds):
        backend = FakeBackendAdapter("FakeBrisbane")
        systems = get_6_systems(backend, verbose=verbose)

        for task in EVAL_TASKS:
            for sys_name, system in systems.items():
                try:
                    trace = system.run(task["prompt"])
                    success = task["success_criteria"](trace)
                    fidelity = task["extract_fidelity"](trace)
                except Exception as e:
                    trace = None
                    success = False
                    fidelity = None
                    if verbose:
                        print(f"  ERROR {sys_name}/{task['id']}: {e}")

                results.append(EvalResult(
                    task_id=task["id"],
                    task_name=task["name"],
                    system=sys_name,
                    success=success,
                    fidelity=round(float(fidelity), 4) if fidelity is not None else None,
                    num_tool_calls=trace.num_tool_calls if trace else 0,
                    elapsed_seconds=round(trace.elapsed_seconds, 2) if trace else 0.0,
                ))

                count += 1
                if count % 6 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / count * (total - count)
                    print(f"  [{count}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    return results


# ═══════════════════════════════════════════════════════
# Drift-specific evaluation
# ═══════════════════════════════════════════════════════

def run_drift_comparison(verbose: bool = False) -> list[EvalResult]:
    """Run drift-specific tasks: inject drift, measure recovery.

    Only T1 (Health+GHZ) and T7 (Full Pipeline) under drift conditions.
    """
    results = []
    drift_tasks = [t for t in EVAL_TASKS if t["id"] in ("T1", "T3", "T7")]

    # Pre-drift (stable)
    print("  [Drift eval] Running stable baseline...")
    backend_stable = SyntheticDriftBackend("FakeBrisbane", STABLE)
    backend_stable.set_time(0)
    systems_stable = get_6_systems(backend_stable, verbose=verbose)

    for task in drift_tasks:
        for sys_name, system in systems_stable.items():
            try:
                trace = system.run(task["prompt"])
                success = task["success_criteria"](trace)
                fidelity = task["extract_fidelity"](trace)
            except Exception:
                success = False
                fidelity = None
                trace = None

            results.append(EvalResult(
                task_id=f"{task['id']}_stable",
                task_name=f"{task['name']} (stable)",
                system=sys_name,
                success=success,
                fidelity=round(float(fidelity), 4) if fidelity is not None else None,
                num_tool_calls=trace.num_tool_calls if trace else 0,
                elapsed_seconds=round(trace.elapsed_seconds, 2) if trace else 0.0,
            ))

    # Post-drift (sudden degradation at t=6h)
    print("  [Drift eval] Running under sudden degradation (t=6h)...")
    backend_drift = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
    backend_drift.set_time(6)
    systems_drift = get_6_systems(backend_drift, verbose=verbose)

    for task in drift_tasks:
        for sys_name, system in systems_drift.items():
            try:
                trace = system.run(task["prompt"])
                success = task["success_criteria"](trace)
                fidelity = task["extract_fidelity"](trace)
            except Exception:
                success = False
                fidelity = None
                trace = None

            results.append(EvalResult(
                task_id=f"{task['id']}_drift",
                task_name=f"{task['name']} (drift)",
                system=sys_name,
                success=success,
                fidelity=round(float(fidelity), 4) if fidelity is not None else None,
                num_tool_calls=trace.num_tool_calls if trace else 0,
                elapsed_seconds=round(trace.elapsed_seconds, 2) if trace else 0.0,
            ))

    return results


# ═══════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════

SYSTEM_ORDER = ["static", "single_shot", "react_no_budget", "react_full", "react_memory", "react_drift"]
SYSTEM_LABELS = {
    "static": "Static",
    "single_shot": "LLM-1Shot",
    "react_no_budget": "ReAct",
    "react_full": "ReAct+Bgt",
    "react_memory": "ReAct+Mem",
    "react_drift": "Full(Ours)",
}


def format_table(results: list[EvalResult], title: str = ""):
    """Format results as a readable comparison table."""
    agg = defaultdict(lambda: {"successes": 0, "count": 0, "fidelities": [], "calls": [], "times": []})

    for r in results:
        key = (r.task_id, r.system)
        a = agg[key]
        a["count"] += 1
        if r.success:
            a["successes"] += 1
        if r.fidelity is not None:
            a["fidelities"].append(r.fidelity)
        a["calls"].append(r.num_tool_calls)
        a["times"].append(r.elapsed_seconds)

    tasks = list(dict.fromkeys(r.task_id for r in results))
    systems = [s for s in SYSTEM_ORDER if any(r.system == s for r in results)]

    # Build table
    col_w = 10
    hdr = f"{'Task':<20}"
    for s in systems:
        hdr += f" | {SYSTEM_LABELS.get(s, s):>{col_w}}"

    sep = "-" * len(hdr)

    lines = []
    lines.append(f"\n{'='*len(hdr)}")
    lines.append(f" {title}" if title else " EVALUATION RESULTS")
    lines.append(f"{'='*len(hdr)}")

    # Success rate
    lines.append(f"\n  Success Rate (%)")
    lines.append(f"  {hdr}")
    lines.append(f"  {sep}")
    for tid in tasks:
        tname = next((r.task_name for r in results if r.task_id == tid), tid)
        row = f"  {tname:<20}"
        for s in systems:
            a = agg[(tid, s)]
            rate = a["successes"] / a["count"] * 100 if a["count"] > 0 else 0
            row += f" | {rate:>{col_w}.0f}"
        lines.append(row)

    # Average fidelity
    lines.append(f"\n  Avg Fidelity")
    lines.append(f"  {hdr}")
    lines.append(f"  {sep}")
    for tid in tasks:
        tname = next((r.task_name for r in results if r.task_id == tid), tid)
        row = f"  {tname:<20}"
        for s in systems:
            a = agg[(tid, s)]
            if a["fidelities"]:
                avg = sum(a["fidelities"]) / len(a["fidelities"])
                row += f" | {avg:>{col_w}.4f}"
            else:
                row += f" | {'N/A':>{col_w}}"
        lines.append(row)

    # Tool calls
    lines.append(f"\n  Avg Tool Calls")
    lines.append(f"  {hdr}")
    lines.append(f"  {sep}")
    for tid in tasks:
        tname = next((r.task_name for r in results if r.task_id == tid), tid)
        row = f"  {tname:<20}"
        for s in systems:
            a = agg[(tid, s)]
            avg = sum(a["calls"]) / len(a["calls"]) if a["calls"] else 0
            row += f" | {avg:>{col_w}.1f}"
        lines.append(row)

    # Summary row
    lines.append(f"\n  OVERALL")
    lines.append(f"  {hdr}")
    lines.append(f"  {sep}")
    row_sr = f"  {'Avg Success Rate':<20}"
    row_fid = f"  {'Avg Fidelity':<20}"
    row_tc = f"  {'Avg Tool Calls':<20}"
    for s in systems:
        all_s = [agg[(tid, s)] for tid in tasks]
        total_succ = sum(a["successes"] for a in all_s)
        total_count = sum(a["count"] for a in all_s)
        all_fids = [f for a in all_s for f in a["fidelities"]]
        all_calls = [c for a in all_s for c in a["calls"]]

        sr = total_succ / total_count * 100 if total_count > 0 else 0
        avg_f = sum(all_fids) / len(all_fids) if all_fids else 0
        avg_c = sum(all_calls) / len(all_calls) if all_calls else 0

        row_sr += f" | {sr:>{col_w}.0f}"
        row_fid += f" | {avg_f:>{col_w}.4f}"
        row_tc += f" | {avg_c:>{col_w}.1f}"

    lines.append(row_sr)
    lines.append(row_fid)
    lines.append(row_tc)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("QuantumGPT Full Comparison: 6 Systems × 7 Tasks")
    print("=" * 60)

    t0 = time.time()

    # Part 1: Standard evaluation (stable backend)
    print("\n[1/2] Standard evaluation (stable backend)...")
    results_standard = run_full_comparison(n_seeds=1, verbose=False)
    table1 = format_table(results_standard, "TABLE 1: Standard Evaluation (Stable Backend)")
    print(table1)

    # Part 2: Drift evaluation
    print("\n[2/2] Drift evaluation (stable vs degraded)...")
    results_drift = run_drift_comparison(verbose=False)
    table2 = format_table(results_drift, "TABLE 2: Drift Evaluation (Stable vs Degraded)")
    print(table2)

    # Save all results
    all_results = {
        "standard": [
            {
                "task_id": r.task_id, "task_name": r.task_name,
                "system": r.system, "success": r.success,
                "fidelity": r.fidelity, "tool_calls": r.num_tool_calls,
                "elapsed_s": r.elapsed_seconds,
            }
            for r in results_standard
        ],
        "drift": [
            {
                "task_id": r.task_id, "task_name": r.task_name,
                "system": r.system, "success": r.success,
                "fidelity": r.fidelity, "tool_calls": r.num_tool_calls,
                "elapsed_s": r.elapsed_seconds,
            }
            for r in results_drift
        ],
    }

    out_path = Path(__file__).parent / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
