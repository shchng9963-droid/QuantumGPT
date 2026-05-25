#!/usr/bin/env python3
"""QC-Agent-Bench v2.4: 20-task evaluation across 4 tiers.

Tier 1 (8 tasks): Static backend — perception, circuit, transpilation
Tier 2 (5 tasks): Drift detection and recalibration
Tier 3 (3 tasks): Injected failures / diagnosis
Tier 4 (4 tasks): Device-level pulse experiments (Rabi, Ramsey, T1, DRAG+RB)

Usage:
  python eval/bench_v24.py                # mock only (no API key needed)
  python eval/bench_v24.py --llm deepseek # mock + deepseek comparison
  python eval/bench_v24.py --llm openai   # mock + openai comparison
"""

import sys
import os
import json
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from backends.fake_adapter import FakeBackendAdapter
from agent.react import ReActAgent


# ═══════════════════════════════════════════════════════
# 20 benchmark tasks
# ═══════════════════════════════════════════════════════

TASKS = [
    # ── Tier 1: Static backend (8 tasks) ──
    {"tier": 1, "id": "T1.1", "name": "Backend health",
     "prompt": "Check the health of the backend and report the average gate errors, T1 time, and number of qubits.",
     "expected_tools": ["get_backend_health"],
     "pass_criterion": "Reports avg T1 and gate error"},

    {"tier": 1, "id": "T1.2", "name": "Qubit properties",
     "prompt": "What are the T1, T2, and readout error for qubits 0, 1, and 2?",
     "expected_tools": ["get_qubit_properties"],
     "pass_criterion": "Reports 3 qubits with T1/T2/readout"},

    {"tier": 1, "id": "T1.3", "name": "Coupling map",
     "prompt": "Show the coupling map of this backend. Which qubits are connected to qubit 0?",
     "expected_tools": ["get_coupling_map"],
     "pass_criterion": "Lists qubit-0 neighbors"},

    {"tier": 1, "id": "T1.4", "name": "Run GHZ circuit",
     "prompt": "Run a 5-qubit GHZ circuit and report the fidelity.",
     "expected_tools": ["run_circuit"],
     "pass_criterion": "Reports a fidelity value"},

    {"tier": 1, "id": "T1.5", "name": "Transpile + run",
     "prompt": "Transpile a 3-qubit QFT circuit for this backend and run it. Report gate count and fidelity.",
     "expected_tools": ["transpile_circuit", "run_circuit"],
     "pass_criterion": "Reports gate count and fidelity"},

    {"tier": 1, "id": "T1.6", "name": "Fidelity prediction",
     "prompt": "Predict the fidelity of a 4-qubit random circuit before running it.",
     "expected_tools": ["predict_fidelity"],
     "pass_criterion": "Returns a predicted fidelity"},

    {"tier": 1, "id": "T1.7", "name": "ZNE mitigation",
     "prompt": "Run a 3-qubit circuit with ZNE error mitigation and compare raw vs mitigated fidelity.",
     "expected_tools": ["run_circuit", "apply_mitigation"],
     "pass_criterion": "Reports both raw and mitigated fidelity"},

    {"tier": 1, "id": "T1.8", "name": "Backend comparison",
     "prompt": "Compare the current backend's qubit properties with a reference. Which qubit has the best T1?",
     "expected_tools": ["get_qubit_properties"],
     "pass_criterion": "Identifies best-T1 qubit"},

    # ── Tier 2: Drift detection (5 tasks) ──
    {"tier": 2, "id": "T2.1", "name": "Drift detection",
     "prompt": "Check if there is any calibration drift on the backend. Report the drift score.",
     "expected_tools": ["detect_drift"],
     "pass_criterion": "Reports drift score"},

    {"tier": 2, "id": "T2.2", "name": "Calibration age",
     "prompt": "How old is the current calibration? Should we recalibrate?",
     "expected_tools": ["get_calibration_age"],
     "pass_criterion": "Reports age and recommendation"},

    {"tier": 2, "id": "T2.3", "name": "Drift + recalibrate",
     "prompt": "Detect drift. If drift is significant, run a Rabi experiment to recalibrate the pi-pulse.",
     "expected_tools": ["detect_drift", "rabi_experiment"],
     "pass_criterion": "Conditionally runs Rabi"},

    {"tier": 2, "id": "T2.4", "name": "Multi-qubit drift",
     "prompt": "Check properties for qubits 0-4 and flag any qubit with T1 below 100 μs.",
     "expected_tools": ["get_qubit_properties"],
     "pass_criterion": "Flags low-T1 qubits"},

    {"tier": 2, "id": "T2.5", "name": "Drift + diagnosis",
     "prompt": "Detect drift on the backend. If drift is found, diagnose the cause and suggest actions.",
     "expected_tools": ["detect_drift", "diagnose_and_suggest"],
     "pass_criterion": "Provides diagnosis and actions"},

    # ── Tier 3: Failure diagnosis (3 tasks) ──
    {"tier": 3, "id": "T3.1", "name": "Diagnose qubit",
     "prompt": "Diagnose qubit 0 and suggest improvements if any parameters are suboptimal.",
     "expected_tools": ["diagnose_and_suggest"],
     "pass_criterion": "Provides concrete suggestions"},

    {"tier": 3, "id": "T3.2", "name": "Low fidelity recovery",
     "prompt": "Run a circuit. If fidelity is below 0.9, apply ZNE mitigation. If still below 0.85, diagnose the issue.",
     "expected_tools": ["run_circuit", "apply_mitigation", "diagnose_and_suggest"],
     "pass_criterion": "Follows conditional recovery logic"},

    {"tier": 3, "id": "T3.3", "name": "Full diagnosis pipeline",
     "prompt": "Check backend health. If any metric is concerning, diagnose it and recommend which experiments to run.",
     "expected_tools": ["get_backend_health", "diagnose_and_suggest"],
     "pass_criterion": "Provides experiment recommendations"},

    # ── Tier 4: Device-level experiments (4 tasks) ──
    {"tier": 4, "id": "T4.1", "name": "Rabi calibration",
     "prompt": "Run a Rabi experiment to find the pi-pulse amplitude for qubit 0. Fit the data and report pi and pi/2 amplitudes.",
     "expected_tools": ["rabi_experiment", "fit_rabi"],
     "pass_criterion": "Reports pi-pulse amplitude with fit"},

    {"tier": 4, "id": "T4.2", "name": "T2* measurement",
     "prompt": "Measure the T2* dephasing time of qubit 0 using a Ramsey experiment with 2 MHz artificial detuning.",
     "expected_tools": ["ramsey_experiment", "fit_ramsey"],
     "pass_criterion": "Reports T2* value"},

    {"tier": 4, "id": "T4.3", "name": "T1 measurement",
     "prompt": "Measure the T1 relaxation time of qubit 0. Report the fitted T1 value.",
     "expected_tools": ["t1_experiment", "fit_t1"],
     "pass_criterion": "Reports T1 value"},

    {"tier": 4, "id": "T4.4", "name": "Full tune-up",
     "prompt": "Perform a full tune-up: Rabi → DRAG calibration → Randomized Benchmarking. Report pi-pulse, DRAG alpha, and EPC.",
     "expected_tools": ["rabi_experiment", "fit_rabi", "drag_calibration", "randomized_benchmarking"],
     "pass_criterion": "Reports pi-amp, DRAG alpha, and EPC"},
]


# ═══════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════

def run_task(task: dict, provider: str, backend) -> dict:
    """Run one task and return result dict."""
    max_calls = 15 if task["tier"] >= 3 else 12
    agent = ReActAgent(
        backend=backend,
        provider=provider,
        model="deepseek-chat" if provider == "deepseek" else None,
        verbose=False,
        max_turns=max_calls + 5,
        target_fidelity=0.0,
        max_tool_calls=max_calls,
        max_seconds=180.0,
        use_memory=False,
    )
    try:
        trace = agent.run(task["prompt"])
        tools_used = [s.action for s in trace.steps if s.action]
        expected = set(task["expected_tools"])
        tools_hit = expected.intersection(set(tools_used))

        return {
            "task_id": task["id"],
            "tier": task["tier"],
            "name": task["name"],
            "provider": provider,
            "success": not trace.final_answer.startswith("Error"),
            "tool_calls": trace.num_tool_calls,
            "tokens": trace.total_tokens,
            "elapsed_s": round(trace.elapsed_seconds, 2),
            "tools_used": tools_used,
            "expected_tools_hit": len(tools_hit),
            "expected_tools_total": len(expected),
            "answer_len": len(trace.final_answer),
            "error": None,
        }
    except Exception as e:
        return {
            "task_id": task["id"],
            "tier": task["tier"],
            "name": task["name"],
            "provider": provider,
            "success": False,
            "tool_calls": 0,
            "tokens": 0,
            "elapsed_s": 0,
            "tools_used": [],
            "expected_tools_hit": 0,
            "expected_tools_total": len(task["expected_tools"]),
            "answer_len": 0,
            "error": str(e)[:200],
        }


def print_results(results: list, providers: list):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("QC-Agent-Bench v2.4 — 20-Task Evaluation")
    print("=" * 100)

    header = f"{'ID':>5} | {'Tier':>4} | {'Task':>25} | {'Prov':>8} | {'OK':>2} | {'Tools':>5} | {'Hit':>5} | {'Tok':>6} | {'Time':>6}"
    print(header)
    print("-" * 100)

    for task in TASKS:
        for prov in providers:
            r = next((r for r in results if r["task_id"] == task["id"] and r["provider"] == prov), None)
            if r is None:
                continue
            ok = "✓" if r["success"] else "✗"
            hit = f"{r['expected_tools_hit']}/{r['expected_tools_total']}"
            print(f"{r['task_id']:>5} | {r['tier']:>4} | {r['name']:>25} | {r['provider']:>8} | {ok:>2} | {r['tool_calls']:>5} | {hit:>5} | {r['tokens']:>6} | {r['elapsed_s']:>5.1f}s")

    # Per-tier summary
    print("\n" + "=" * 80)
    print("PER-TIER SUMMARY")
    print("=" * 80)

    for prov in providers:
        prov_results = [r for r in results if r["provider"] == prov]
        print(f"\n  Provider: {prov}")
        print(f"  {'Tier':>6} | {'Pass':>6} | {'Total':>5} | {'Rate':>6} | {'Avg Tools':>9} | {'Avg Time':>8}")
        print(f"  " + "-" * 60)

        for tier in [1, 2, 3, 4]:
            tier_r = [r for r in prov_results if r["tier"] == tier]
            if not tier_r:
                continue
            n_pass = sum(1 for r in tier_r if r["success"])
            n_total = len(tier_r)
            avg_tools = sum(r["tool_calls"] for r in tier_r) / n_total
            avg_time = sum(r["elapsed_s"] for r in tier_r) / n_total
            rate = f"{n_pass}/{n_total}"
            print(f"  {tier:>6} | {rate:>6} | {n_total:>5} | {n_pass/n_total*100:>5.0f}% | {avg_tools:>9.1f} | {avg_time:>7.1f}s")

        total_pass = sum(1 for r in prov_results if r["success"])
        total_n = len(prov_results)
        print(f"  {'ALL':>6} | {total_pass}/{total_n}  |   {total_n} | {total_pass/total_n*100:>5.0f}% |")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QC-Agent-Bench v2.4")
    parser.add_argument("--llm", default=None, help="LLM provider for comparison (deepseek, openai)")
    parser.add_argument("--tier", type=int, default=None, help="Only run specific tier (1-4)")
    parser.add_argument("--save", default=None, help="Output JSON path")
    args = parser.parse_args()

    backend = FakeBackendAdapter("FakeBrisbane")
    providers = ["mock"]
    if args.llm:
        providers.append(args.llm)

    tasks = TASKS
    if args.tier:
        tasks = [t for t in TASKS if t["tier"] == args.tier]

    results = []
    total = len(tasks) * len(providers)
    done = 0

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"[{task['id']}] {task['name']} (Tier {task['tier']})")
        print(f"  {task['prompt'][:80]}...")
        print(f"{'='*70}")

        for prov in providers:
            done += 1
            print(f"  [{done}/{total}] Running {prov}...", end="", flush=True)
            r = run_task(task, prov, backend)
            results.append(r)
            ok_mark = "✓" if r["success"] else "✗"
            print(f" {ok_mark} ({r['elapsed_s']}s, {r['tool_calls']} tools)")

    print_results(results, providers)

    # Save
    out_path = args.save or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_v24_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
