"""Day 8 Demo: CalibrationAdvisor — multi-step autonomous advisory cycle.

Runs 3 scenarios:
  1. STABLE backend — advisor says "all good"
  2. LINEAR_DECAY — advisor detects gradual degradation
  3. SUDDEN_DEGRADATION — advisor catches the abrupt change

Each scenario: monitor → diagnose → act, with DuckDB persistence.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from advisor.advisor import CalibrationAdvisor
from backends.synthetic_drift import (
    SyntheticDriftBackend,
    LINEAR_DECAY,
    SUDDEN_DEGRADATION,
    STABLE,
    DIURNAL_CYCLE,
)

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DEMO_DIR, "advisor.duckdb")

# Clean previous run
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)


def run_scenario(name, profile, time_points, probes):
    print(f"\n{'='*70}")
    print(f"  Scenario: {name}")
    print(f"  Time points: {time_points}")
    print(f"  Probe circuits: {probes}")
    print(f"{'='*70}\n")

    backend = SyntheticDriftBackend("FakeBrisbane", profile)
    advisor = CalibrationAdvisor(backend, db_path=DB_PATH, qubit_sample_size=15)

    t0 = time.time()
    report = advisor.run_advisory_cycle(
        time_points=time_points,
        probe_circuits=probes,
        shots=4096,
    )
    elapsed = time.time() - t0

    print(report.summary)
    print(f"  [Elapsed: {elapsed:.1f}s, "
          f"{report.num_snapshots} snapshots, "
          f"{report.num_probes} probes]\n")

    return report


def main():
    print("=" * 70)
    print("  QuantumGPT Day 8: CalibrationAdvisor Demo")
    print("  Multi-step autonomous advisory: monitor → diagnose → act")
    print("=" * 70)

    results = {}

    # Scenario 1: Stable
    results["stable"] = run_scenario(
        "STABLE — healthy device, no drift",
        STABLE,
        time_points=[0, 4, 8, 12],
        probes=["ghz_5", "qft_4"],
    )

    # Scenario 2: Linear Decay
    results["linear"] = run_scenario(
        "LINEAR_DECAY — gradual 24-hour degradation",
        LINEAR_DECAY,
        time_points=[0, 4, 8, 12, 16, 20, 24],
        probes=["ghz_5"],
    )

    # Scenario 3: Sudden Degradation
    results["sudden"] = run_scenario(
        "SUDDEN_DEGRADATION — abrupt change at t=5h",
        SUDDEN_DEGRADATION,
        time_points=[0, 2, 4, 5, 5.5, 6, 8],
        probes=["ghz_5"],
    )

    # Scenario 4: Diurnal Cycle
    results["diurnal"] = run_scenario(
        "DIURNAL_CYCLE — 24h oscillation pattern",
        DIURNAL_CYCLE,
        time_points=[0, 3, 6, 9, 12, 15, 18, 21, 24],
        probes=["ghz_5"],
    )

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Severity':<12} {'Trend':<18} {'Actions':<8} {'Top Action'}")
    print("-" * 90)
    for name, report in results.items():
        top_action = report.actions[0].action.value if report.actions else "N/A"
        print(f"{name:<25} {report.severity.value:<12} "
              f"{report.temporal_analysis.trend.value:<18} "
              f"{len(report.actions):<8} {top_action}")

    # Save results
    results_path = os.path.join(DEMO_DIR, "results.json")
    serializable = {}
    for name, report in results.items():
        serializable[name] = {
            "severity": report.severity.value,
            "trend": report.temporal_analysis.trend.value,
            "t1_slope": report.temporal_analysis.t1_slope,
            "drift_slope": report.temporal_analysis.drift_slope,
            "changepoints": report.temporal_analysis.changepoints,
            "num_actions": len(report.actions),
            "actions": [
                {"type": a.action.value, "priority": a.priority, "reason": a.reason}
                for a in report.actions
            ],
            "health_timeline": report.health_timeline,
            "probes": [
                {"time": pr.time_hours, "circuit": pr.circuit, "fidelity": pr.fidelity}
                for pr in report.probe_results
            ],
            "worst_qubits": report.temporal_analysis.worst_qubits,
            "best_qubits": report.temporal_analysis.best_qubits,
        }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"DuckDB at {DB_PATH}")


if __name__ == "__main__":
    main()
