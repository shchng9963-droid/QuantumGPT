"""Fast 6-system comparison using mock executor (no actual simulation).

This produces the same structure as full_comparison.py but uses cached/mock
results for speed. Suitable for generating Paper Table 1 structure.

The mock executor returns realistic but pre-computed results based on
the circuit name and backend state, avoiding the 0.5s/run simulation overhead.
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
from agent.react import ReActAgent, AgentTrace, TraceStep
from agent.baselines import StaticPipeline, LLMSingleShot, ReActNoBudget
from tools.quantum_tools import ToolExecutor


# ═══════════════════════════════════════════════════════
# Fast Mock Executor — returns pre-computed results
# ═══════════════════════════════════════════════════════

class FastMockExecutor(ToolExecutor):
    """Override execute to return fast mock results without simulation."""

    # Pre-computed fidelities for each circuit on FakeBrisbane (stable)
    FIDELITY_TABLE = {
        "ghz_3": 0.941, "ghz_5": 0.912, "qft_4": 0.876,
        "bv_5": 0.903, "vqe_4": 0.889, "qaoa_4": 0.867,
    }

    def __init__(self, backend, drift_factor: float = 1.0):
        super().__init__(backend)
        self.drift_factor = drift_factor  # 1.0 = stable, <1.0 = degraded

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Fast mock execution — no actual simulation."""
        if tool_name == "get_backend_health":
            h = self.backend.get_health()
            return json.dumps({
                "backend": h.name, "num_qubits": h.num_qubits,
                "avg_1q_error": round(h.avg_1q_error, 6),
                "avg_2q_error": round(h.avg_2q_error, 6),
                "avg_readout_error": round(h.avg_readout_error, 6),
                "avg_t1_us": round(h.avg_t1_us, 1),
                "avg_t2_us": round(h.avg_t2_us, 1),
                "drift_score": round(h.drift_score, 4) if h.drift_score else 0.0,
            })

        elif tool_name == "run_circuit":
            name = tool_input.get("circuit_name", "ghz_5")
            base_fid = self.FIDELITY_TABLE.get(name, 0.88)
            fid = base_fid * self.drift_factor
            # Add small noise
            import random
            fid += random.gauss(0, 0.01)
            fid = max(0.3, min(1.0, fid))
            return json.dumps({
                "circuit": name, "shots": 4096,
                "fidelity": round(fid, 4),
                "transpiled_depth": 17,
                "top_counts": {"00000": 1900, "11111": 1800, "00001": 100},
            })

        elif tool_name == "predict_fidelity":
            name = tool_input.get("circuit_name", "ghz_5")
            base_fid = self.FIDELITY_TABLE.get(name, 0.88)
            pred = base_fid * self.drift_factor
            return json.dumps({
                "circuit": name, "backend": self.backend.name,
                "predicted_fidelity": round(pred, 4),
                "method": "xgboost", "confidence": "high",
            })

        elif tool_name == "diagnose_and_suggest":
            h = self.backend.get_health()
            severity = "nominal" if self.drift_factor > 0.9 else "degraded"
            return json.dumps({
                "severity": severity,
                "drift_score": round(h.drift_score or 0, 4),
                "suggestions": ["recalibrate"] if severity == "degraded" else [],
            })

        elif tool_name == "apply_mitigation":
            name = tool_input.get("circuit_name", "ghz_5")
            base_fid = self.FIDELITY_TABLE.get(name, 0.88)
            # Mitigation improves by ~5%
            fid = min(1.0, base_fid * self.drift_factor + 0.05)
            return json.dumps({
                "circuit": name, "method": "zne",
                "mitigated_fidelity": round(fid, 4),
                "improvement": 0.05,
            })

        elif tool_name == "transpile_circuit":
            return json.dumps({
                "circuit": tool_input.get("circuit_name", "ghz_5"),
                "optimization_level": tool_input.get("optimization_level", 1),
                "transpiled_depth": 15, "n_2q_gates": 4,
            })

        elif tool_name == "rabi_experiment":
            return json.dumps({
                "amplitudes": list(range(30)),
                "populations": [0.5] * 30,
                "pi_amplitude_mhz": 5.03,
            })

        elif tool_name == "fit_rabi":
            return json.dumps({
                "pi_amplitude_mhz": 5.03,
                "rabi_freq_mhz": 25.1,
                "r_squared": 0.999,
            })

        elif tool_name == "list_benchmarks":
            return json.dumps({"circuits": list(self.FIDELITY_TABLE.keys())})

        elif tool_name == "get_qubit_properties":
            return json.dumps({"qubits": [
                {"qubit": q, "t1_us": 220, "t2_us": 140, "readout_error": 0.02}
                for q in tool_input.get("qubits", [0, 1, 2, 3, 4])
            ]})

        elif tool_name == "get_coupling_map":
            return json.dumps({"edges": [[0,1],[1,2],[2,3],[3,4]], "num_qubits": 127})

        elif tool_name == "retrieve_past_experiments":
            return json.dumps({"experiments": [], "count": 0})

        elif tool_name == "check_drift_since":
            ds = self.backend.get_health().drift_score or 0
            return json.dumps({"drift_detected": ds > 0.3, "drift_score": round(ds, 4)})

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ═══════════════════════════════════════════════════════
# Monkey-patch systems to use FastMockExecutor
# ═══════════════════════════════════════════════════════

def get_6_systems_fast(backend, drift_factor=1.0, verbose=False):
    """Create 6 systems with fast mock executor."""
    mock_exec = FastMockExecutor(backend, drift_factor=drift_factor)

    # Static pipeline
    static = StaticPipeline(backend, verbose=verbose)
    static.executor = mock_exec

    # LLM single shot
    single = LLMSingleShot(backend, verbose=verbose)
    single.executor = mock_exec

    # ReAct no budget
    react_nb = ReActNoBudget(backend, verbose=verbose)
    react_nb._agent.executor = mock_exec

    # ReAct full
    react_full = ReActAgent(backend, provider="mock", verbose=verbose,
                            use_memory=False, use_drift_aware=False)
    react_full.executor = mock_exec

    # ReAct + memory
    react_mem = ReActAgent(backend, provider="mock", verbose=verbose,
                           use_memory=True, use_drift_aware=False)
    react_mem.executor = mock_exec

    # ReAct + drift-aware
    react_drift = ReActAgent(backend, provider="mock", verbose=verbose,
                             use_memory=True, use_drift_aware=True)
    react_drift.executor = mock_exec

    return {
        "static": static,
        "single_shot": single,
        "react_no_budget": react_nb,
        "react_full": react_full,
        "react_memory": react_mem,
        "react_drift": react_drift,
    }


# ═══════════════════════════════════════════════════════
# Tasks (same as full_comparison.py)
# ═══════════════════════════════════════════════════════

def _safe_json(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}

EVAL_TASKS = [
    {"id": "T1", "name": "Health + GHZ-5",
     "prompt": "Check the backend health and run the GHZ-5 circuit. Report the fidelity.",
     "success": lambda t: any(s.action == "run_circuit" and s.fidelity_observed for s in t.steps),
     "metric": lambda t: max((s.fidelity_observed for s in t.steps if s.fidelity_observed), default=None)},
    {"id": "T2", "name": "Multi-Circuit",
     "prompt": "Run ghz_5, qft_4, and bv_5 circuits and compare their fidelities.",
     "success": lambda t: sum(1 for s in t.steps if s.action == "run_circuit") >= 2,
     "metric": lambda t: max((s.fidelity_observed for s in t.steps if s.fidelity_observed), default=None)},
    {"id": "T3", "name": "Diagnose+Mitigate",
     "prompt": "Run the GHZ-5 circuit. If fidelity is below 0.95, diagnose the issue and try error mitigation.",
     "success": lambda t: any(s.action in ("apply_mitigation", "diagnose_and_suggest") for s in t.steps),
     "metric": lambda t: max((s.fidelity_observed for s in t.steps if s.fidelity_observed), default=None)},
    {"id": "T4", "name": "Fidelity Predict",
     "prompt": "Predict the fidelity of the ghz_5 circuit, then run it to verify.",
     "success": lambda t: any(s.action in ("predict_fidelity", "run_circuit") for s in t.steps),
     "metric": lambda t: max((s.fidelity_observed for s in t.steps if s.fidelity_observed), default=None)},
    {"id": "T5", "name": "Transpile+Run",
     "prompt": "Transpile the qft_4 circuit with optimization level 2, then run it.",
     "success": lambda t: any(s.action in ("transpile_circuit", "run_circuit") for s in t.steps),
     "metric": lambda t: max((s.fidelity_observed for s in t.steps if s.fidelity_observed), default=None)},
    {"id": "T6", "name": "Rabi Experiment",
     "prompt": "Run a Rabi oscillation experiment at 5 GHz, fit the data, report pi-pulse amplitude.",
     "success": lambda t: any(s.action == "rabi_experiment" for s in t.steps),
     "metric": lambda t: next((_safe_json(s.observation).get("pi_amplitude_mhz")
                               for s in t.steps if s.action in ("fit_rabi", "rabi_experiment") and s.observation), None)},
    {"id": "T7", "name": "Full Pipeline",
     "prompt": "Check health, run GHZ-5, if fidelity < 0.95 apply mitigation, then diagnose.",
     "success": lambda t: (any(s.action == "get_backend_health" for s in t.steps) and
                           any(s.action == "run_circuit" for s in t.steps)),
     "metric": lambda t: max((s.fidelity_observed for s in t.steps if s.fidelity_observed), default=None)},
]


# ═══════════════════════════════════════════════════════
# Run evaluation
# ═══════════════════════════════════════════════════════

@dataclass
class Result:
    task_id: str
    task_name: str
    system: str
    success: bool
    fidelity: float | None
    tool_calls: int
    elapsed_s: float


def run_eval(n_seeds=3, drift_factor=1.0, verbose=False) -> list[Result]:
    results = []
    for seed in range(n_seeds):
        backend = FakeBackendAdapter("FakeBrisbane")
        systems = get_6_systems_fast(backend, drift_factor=drift_factor, verbose=verbose)

        for task in EVAL_TASKS:
            for sys_name, system in systems.items():
                try:
                    trace = system.run(task["prompt"])
                    success = task["success"](trace)
                    metric = task["metric"](trace)
                except Exception as e:
                    trace = None
                    success = False
                    metric = None

                results.append(Result(
                    task_id=task["id"], task_name=task["name"],
                    system=sys_name, success=success,
                    fidelity=round(float(metric), 4) if metric else None,
                    tool_calls=trace.num_tool_calls if trace else 0,
                    elapsed_s=round(trace.elapsed_seconds, 3) if trace else 0,
                ))
    return results


def format_table(results: list[Result], title: str = ""):
    """Format as paper-ready table."""
    SYSTEMS = ["static", "single_shot", "react_no_budget", "react_full", "react_memory", "react_drift"]
    LABELS = {"static": "Static", "single_shot": "LLM-1Shot", "react_no_budget": "ReAct",
              "react_full": "+Budget", "react_memory": "+Memory", "react_drift": "Full(Ours)"}

    agg = defaultdict(lambda: {"succ": 0, "n": 0, "fids": [], "calls": []})
    for r in results:
        a = agg[(r.task_id, r.system)]
        a["n"] += 1
        if r.success: a["succ"] += 1
        if r.fidelity: a["fids"].append(r.fidelity)
        a["calls"].append(r.tool_calls)

    tasks = list(dict.fromkeys(r.task_id for r in results))
    systems = [s for s in SYSTEMS if any(r.system == s for r in results)]

    lines = [f"\n{'='*80}", f"  {title}", f"{'='*80}"]

    # Header
    hdr = f"  {'Task':<18}"
    for s in systems:
        hdr += f" {LABELS.get(s,s):>9}"
    lines.append(f"\n  Success Rate (%)")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for tid in tasks:
        tname = next(r.task_name for r in results if r.task_id == tid)
        row = f"  {tname:<18}"
        for s in systems:
            a = agg[(tid, s)]
            rate = a["succ"] / a["n"] * 100 if a["n"] else 0
            row += f" {rate:>9.0f}"
        lines.append(row)

    # Overall
    lines.append("  " + "-" * (len(hdr) - 2))
    row = f"  {'OVERALL':<18}"
    for s in systems:
        total_s = sum(agg[(tid, s)]["succ"] for tid in tasks)
        total_n = sum(agg[(tid, s)]["n"] for tid in tasks)
        rate = total_s / total_n * 100 if total_n else 0
        row += f" {rate:>9.0f}"
    lines.append(row)

    # Fidelity
    lines.append(f"\n  Avg Fidelity")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for tid in tasks:
        tname = next(r.task_name for r in results if r.task_id == tid)
        row = f"  {tname:<18}"
        for s in systems:
            a = agg[(tid, s)]
            if a["fids"]:
                row += f" {sum(a['fids'])/len(a['fids']):>9.4f}"
            else:
                row += f" {'N/A':>9}"
        lines.append(row)

    # Tool calls
    lines.append(f"\n  Avg Tool Calls")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for tid in tasks:
        tname = next(r.task_name for r in results if r.task_id == tid)
        row = f"  {tname:<18}"
        for s in systems:
            a = agg[(tid, s)]
            avg = sum(a["calls"]) / len(a["calls"]) if a["calls"] else 0
            row += f" {avg:>9.1f}"
        lines.append(row)

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("QuantumGPT Fast Comparison: 6 Systems x 7 Tasks")
    print("=" * 60)

    t0 = time.time()

    # Standard (stable)
    print("\n[1/2] Standard evaluation (stable)...")
    results_stable = run_eval(n_seeds=3, drift_factor=1.0)
    print(format_table(results_stable, "TABLE 1: Standard Evaluation (Stable Backend)"))

    # Drift (degraded)
    print("\n[2/2] Drift evaluation (degraded, factor=0.75)...")
    results_drift = run_eval(n_seeds=3, drift_factor=0.75)
    print(format_table(results_drift, "TABLE 2: Under Drift (75% degradation)"))

    # Save
    out = {
        "stable": [{"task": r.task_id, "system": r.system, "success": r.success,
                    "fidelity": r.fidelity, "calls": r.tool_calls} for r in results_stable],
        "drift": [{"task": r.task_id, "system": r.system, "success": r.success,
                   "fidelity": r.fidelity, "calls": r.tool_calls} for r in results_drift],
    }
    out_path = Path(__file__).parent / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results: {out_path}")
