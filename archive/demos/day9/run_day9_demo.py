"""Day 9 Demo: StreamingAdvisor — real-time telemetry + alert engine.

Runs 3 scenarios with different drift profiles, demonstrates:
  1. Batch mode: step through time, show alerts as they fire
  2. Streaming mode: real-time background thread, time-accelerated
  3. Integration: streaming → advisor handoff

Alert types demonstrated:
  - DRIFT_SPIKE: sudden jump detection
  - DRIFT_HIGH: threshold crossing
  - T1_DEGRADING: trend-based degradation
  - T1_LOW: absolute threshold
  - ERROR_HIGH: 2Q error threshold
  - RECOVERY: device comes back to normal
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from advisor.streaming import StreamingAdvisor, AlertConfig, AlertLevel, AlertType
from advisor.advisor import CalibrationAdvisor
from backends.synthetic_drift import (
    SyntheticDriftBackend,
    DriftProfile,
    LINEAR_DECAY,
    SUDDEN_DEGRADATION,
    STABLE,
    DIURNAL_CYCLE,
)

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DEMO_DIR, "streaming.duckdb")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)


def print_alert(alert):
    """Callback for live alert printing."""
    icons = {
        AlertLevel.INFO: "ℹ️ ",
        AlertLevel.WARNING: "⚠️ ",
        AlertLevel.CRITICAL: "🚨",
    }
    icon = icons.get(alert.level, "  ")
    print(f"    {icon} [{alert.level.value:8s}] t={alert.sim_time_hours:5.1f}h "
          f"| {alert.alert_type.value:15s} | {alert.message}")


def scenario_1_batch():
    """Batch mode: step through sudden degradation, print alerts live."""
    print(f"\n{'='*75}")
    print("  Scenario 1: BATCH MODE — Sudden Degradation at t=5h")
    print(f"{'='*75}\n")

    backend = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
    config = AlertConfig(alert_cooldown_hours=0.3)
    sa = StreamingAdvisor(backend, alert_config=config, db_path=DB_PATH)
    sa.on_alert(print_alert)

    print("  Stepping through t=0..10h in 0.5h increments:")
    print(f"  {'─'*70}")

    time_points = [t * 0.5 for t in range(21)]  # 0, 0.5, ..., 10
    t0 = time.time()
    alerts = sa.run_batch(time_points)
    elapsed = time.time() - t0

    print(f"  {'─'*70}")
    summary = sa.get_summary()
    print(f"\n  Summary: {summary['total_snapshots']} snapshots, "
          f"{summary['total_alerts']} alerts "
          f"({summary['alerts_by_level']['critical']} critical, "
          f"{summary['alerts_by_level']['warning']} warning, "
          f"{summary['alerts_by_level']['info']} info) "
          f"in {elapsed:.2f}s")
    print(f"  T1 slope: {summary['t1_slope']:+.2f} μs/h")
    print(f"  Drift slope: {summary['drift_slope']:+.4f}/h")

    return sa, alerts


def scenario_2_streaming():
    """Streaming mode: real-time background thread."""
    print(f"\n{'='*75}")
    print("  Scenario 2: STREAMING MODE — Linear Decay (24h in 3s)")
    print(f"{'='*75}\n")

    backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
    config = AlertConfig(
        drift_warning=0.15,
        alert_cooldown_hours=2.0,
    )
    sa = StreamingAdvisor(backend, alert_config=config, db_path=DB_PATH)
    sa.on_alert(print_alert)

    # 3 real seconds, each tick = 0.1s, acceleration = 28800x
    # → 0.1s real = 0.8h sim, 3s = 24h simulated
    print("  Starting stream: 3 seconds real-time = 24 simulated hours")
    print(f"  {'─'*70}")

    t0 = time.time()
    sa.start(interval_seconds=0.1, time_acceleration=28800, duration_seconds=3.0)
    elapsed = time.time() - t0

    print(f"  {'─'*70}")
    summary = sa.get_summary()
    print(f"\n  Summary: {summary['total_snapshots']} snapshots, "
          f"{summary['total_alerts']} alerts in {elapsed:.2f}s")

    if summary['latest_health']:
        lh = summary['latest_health']
        print(f"  Final state: T1={lh.get('avg_t1_us', 'N/A'):.1f}μs, "
              f"drift={lh.get('drift_score', 'N/A')}")

    return sa


def scenario_3_recovery():
    """Degradation followed by recovery — catch the recovery event."""
    print(f"\n{'='*75}")
    print("  Scenario 3: RECOVERY — Degradation then recovery")
    print(f"{'='*75}\n")

    # Custom profile: T1 drops at t=3-6, then recovers
    profile = DriftProfile(
        t1_drift=lambda t: 0.3 if 3 <= t <= 6 else 1.0,
        gate_error_drift=lambda t: 5.0 if 3 <= t <= 6 else 1.0,
        drift_score_fn=lambda t: 1.0 if 3 <= t <= 6 else 0.0,
    )
    backend = SyntheticDriftBackend("FakeBrisbane", profile)
    config = AlertConfig(alert_cooldown_hours=0.5)
    sa = StreamingAdvisor(backend, alert_config=config, db_path=DB_PATH)
    sa.on_alert(print_alert)

    print("  Stepping through: healthy → degraded (t=3-6h) → recovered")
    print(f"  {'─'*70}")

    time_points = [t * 0.5 for t in range(21)]  # 0..10h
    alerts = sa.run_batch(time_points)

    print(f"  {'─'*70}")
    recovery_alerts = [a for a in alerts if a.alert_type == AlertType.RECOVERY]
    print(f"\n  Recovery alerts: {len(recovery_alerts)}")
    for ra in recovery_alerts:
        print(f"    Recovered at t={ra.sim_time_hours:.1f}h: {ra.message}")

    return sa, alerts


def scenario_4_integration():
    """Streaming → CalibrationAdvisor handoff."""
    print(f"\n{'='*75}")
    print("  Scenario 4: INTEGRATION — Streaming → Advisor handoff")
    print(f"{'='*75}\n")

    backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
    db_path = os.path.join(DEMO_DIR, "integrated.duckdb")
    if os.path.exists(db_path):
        os.remove(db_path)

    # Phase 1: Streaming monitor (0-12h)
    print("  Phase 1: Streaming monitor (0-12h)...")
    sa = StreamingAdvisor(backend, db_path=db_path,
                          alert_config=AlertConfig(alert_cooldown_hours=1.0))
    sa.on_alert(print_alert)
    sa.run_batch([t for t in range(13)])  # 0..12h
    print(f"    → {sa.alert_count} alerts from streaming phase")

    # Phase 2: Full advisory cycle (12-24h)
    print("\n  Phase 2: CalibrationAdvisor cycle (12-24h)...")
    advisor = CalibrationAdvisor(backend, db_path=db_path, qubit_sample_size=10)
    report = advisor.run_advisory_cycle(
        time_points=[12, 16, 20, 24],
        probe_circuits=["ghz_5"],
    )
    print(f"    → Severity: {report.severity.value}")
    print(f"    → Actions: {len(report.actions)}")
    for a in report.actions:
        print(f"       [{a.action.value}] P{a.priority} — {a.reason[:80]}")

    # Verify shared DB
    from data.store import DataStore
    db = DataStore(db_path)
    counts = db.table_counts()
    print(f"\n  Shared DuckDB: {counts}")

    return sa, report


def main():
    print("=" * 75)
    print("  QuantumGPT Day 9: StreamingAdvisor Demo")
    print("  Real-time telemetry stream + threshold-based alert engine")
    print("=" * 75)

    all_results = {}

    # Scenario 1: Batch
    sa1, alerts1 = scenario_1_batch()
    all_results["sudden_batch"] = {
        "snapshots": sa1.snapshot_count,
        "alerts": [{"time": a.sim_time_hours, "level": a.level.value,
                     "type": a.alert_type.value, "message": a.message}
                   for a in alerts1],
        "summary": sa1.get_summary(),
    }

    # Scenario 2: Streaming
    sa2 = scenario_2_streaming()
    all_results["linear_stream"] = {
        "snapshots": sa2.snapshot_count,
        "alerts": [{"time": a.sim_time_hours, "level": a.level.value,
                     "type": a.alert_type.value, "message": a.message}
                   for a in sa2.get_alert_history()],
        "summary": sa2.get_summary(),
    }

    # Scenario 3: Recovery
    sa3, alerts3 = scenario_3_recovery()
    all_results["recovery"] = {
        "snapshots": sa3.snapshot_count,
        "alerts": [{"time": a.sim_time_hours, "level": a.level.value,
                     "type": a.alert_type.value, "message": a.message}
                   for a in alerts3],
    }

    # Scenario 4: Integration
    sa4, report4 = scenario_4_integration()
    all_results["integration"] = {
        "streaming_alerts": sa4.alert_count,
        "advisor_severity": report4.severity.value,
        "advisor_actions": len(report4.actions),
    }

    # Save results
    results_path = os.path.join(DEMO_DIR, "results.json")
    # Make summary serializable
    for k, v in all_results.items():
        if "summary" in v and v["summary"]:
            s = v["summary"]
            if "latest_health" in s and s["latest_health"]:
                # Convert any non-serializable values
                for hk, hv in list(s["latest_health"].items()):
                    if hv is not None and not isinstance(hv, (str, int, float, bool)):
                        s["latest_health"][hk] = str(hv)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*75}")
    print("  SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Scenario':<25} {'Snapshots':<12} {'Alerts':<10} {'Key Finding'}")
    print(f"  {'─'*75}")

    total_snaps = sa1.snapshot_count + sa2.snapshot_count + sa3.snapshot_count
    total_alerts = sa1.alert_count + sa2.alert_count + sa3.alert_count

    print(f"  {'Sudden (batch)':<25} {sa1.snapshot_count:<12} {sa1.alert_count:<10} "
          f"Drift spike detected at t=5h")
    print(f"  {'Linear (stream)':<25} {sa2.snapshot_count:<12} {sa2.alert_count:<10} "
          f"Gradual drift tracked over 24h")
    print(f"  {'Recovery':<25} {sa3.snapshot_count:<12} {sa3.alert_count:<10} "
          f"Recovery alert at t≈6.5h")
    print(f"  {'Integration':<25} {'—':<12} {sa4.alert_count:<10} "
          f"Streaming → Advisor handoff")
    print(f"  {'─'*75}")
    print(f"  Total: {total_snaps} snapshots, {total_alerts} alerts")

    print(f"\n  Results: {results_path}")
    print(f"  DuckDB:  {DB_PATH}")


if __name__ == "__main__":
    main()
