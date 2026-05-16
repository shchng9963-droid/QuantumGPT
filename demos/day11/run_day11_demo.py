#!/usr/bin/env python3
"""Day 11 Demo: Backend Validation + Cross-Backend Comparison.

Demonstrates all 3 shadow backends + pulse-level dynamics side by side,
validates the test suite, and generates comparison data.

Scenarios:
1. Backend health snapshot comparison (Fake vs Replay vs Synthetic)
2. Drift profiles: stable, linear, sudden across SyntheticDriftBackend
3. PropertiesStream batch replay — 24h telemetry
4. Cross-backend circuit execution (GHZ-5 fidelity)
5. Pulse-level vs circuit-level: Rabi calibration → gate fidelity
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backends import FakeBackendAdapter, SyntheticDriftBackend
from backends.replay_backend import ReplayBackend
from backends.properties_stream import PropertiesStream
from backends.synthetic_drift import STABLE, LINEAR_DECAY, SUDDEN_DEGRADATION, DriftProfile
from dynamics.rabi import RabiConfig, simulate_rabi, sweep_rabi, estimate_pi_pulse


def scenario_1_health_comparison():
    """Compare health metrics across all 3 backend types."""
    print("\n=== Scenario 1: Backend Health Comparison ===")

    results = {}

    # FakeBackendAdapter
    t0 = time.time()
    fake = FakeBackendAdapter("FakeBrisbane")
    h_fake = fake.get_health()
    results["FakeBackendAdapter"] = {
        "type": "FakeBackendAdapter",
        "n_qubits": h_fake.num_qubits,
        "avg_t1_us": round(h_fake.avg_t1_us, 1),
        "avg_2q_error": round(h_fake.avg_2q_error, 5),
        "drift_score": h_fake.drift_score or 0.0,
    }
    print(f"  FakeBackendAdapter:   T1={h_fake.avg_t1_us:.1f}μs, "
          f"2Q_err={h_fake.avg_2q_error:.5f}, drift={h_fake.drift_score or 0:.3f}")

    # ReplayBackend at t=0, t=12, t=23
    replay = ReplayBackend.from_synthetic_history("FakeBrisbane", duration_hours=24, snapshot_interval_hours=2, seed=42)
    replay_data = []
    for t_h in [0, 12, 23]:
        replay.set_time(t_h)
        h = replay.get_health()
        entry = {
            "time_h": t_h,
            "avg_t1_us": round(h.avg_t1_us, 1),
            "avg_2q_error": round(h.avg_2q_error, 5),
            "drift_score": round(h.drift_score or 0, 3),
        }
        replay_data.append(entry)
        print(f"  ReplayBackend(t={t_h}h): T1={h.avg_t1_us:.1f}μs, "
              f"2Q_err={h.avg_2q_error:.5f}, drift={h.drift_score or 0:.3f}")
    results["ReplayBackend"] = replay_data

    # SyntheticDriftBackend with 3 profiles
    synth_data = {}
    for profile_name, profile in [("stable", STABLE), ("linear", LINEAR_DECAY), ("sudden", SUDDEN_DEGRADATION)]:
        backend = SyntheticDriftBackend("FakeBrisbane", profile)
        entries = []
        for t_h in [0, 6, 12, 18, 24]:
            backend.set_time(t_h)
            h = backend.get_health()
            entries.append({
                "time_h": t_h,
                "avg_t1_us": round(h.avg_t1_us, 1),
                "drift_score": round(h.drift_score or 0, 3),
            })
        synth_data[profile_name] = entries
        print(f"  SyntheticDrift({profile_name}): "
              f"t=0h drift={entries[0]['drift_score']:.3f} → "
              f"t=24h drift={entries[-1]['drift_score']:.3f}")
    results["SyntheticDriftBackend"] = synth_data

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")
    results["elapsed_s"] = elapsed
    return results


def scenario_2_drift_profiles():
    """Detailed drift profile evolution over 48 hours."""
    print("\n=== Scenario 2: Drift Profile Evolution ===")

    profiles = {
        "stable": STABLE,
        "linear_decay": LINEAR_DECAY,
        "sudden_degradation": SUDDEN_DEGRADATION,
        "custom_recovery": DriftProfile(
            t1_drift=lambda t: 0.3 if 3 <= t <= 8 else 1.0,
            drift_score_fn=lambda t: 0.8 if 3 <= t <= 8 else 0.0,
        ),
    }

    t0 = time.time()
    times = np.linspace(0, 24, 49).tolist()  # every 0.5h
    results = {"times": times}

    for name, profile in profiles.items():
        backend = SyntheticDriftBackend("FakeBrisbane", profile)
        t1_series = []
        drift_series = []
        for t in times:
            backend.set_time(t)
            h = backend.get_health()
            t1_series.append(round(h.avg_t1_us, 1))
            drift_series.append(round(h.drift_score or 0, 3))
        results[name] = {"t1_us": t1_series, "drift_score": drift_series}
        print(f"  {name}: T1 range [{min(t1_series):.0f}, {max(t1_series):.0f}] μs, "
              f"drift range [{min(drift_series):.3f}, {max(drift_series):.3f}]")

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")
    results["elapsed_s"] = elapsed
    return results


def scenario_3_telemetry_stream():
    """24-hour PropertiesStream replay."""
    print("\n=== Scenario 3: PropertiesStream Telemetry (24h) ===")

    backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
    from backends.properties_stream import PropertiesStream

    t0 = time.time()
    snapshots = PropertiesStream.replay_all(backend, hours=24, step_hours=1)
    elapsed = time.time() - t0

    tele = {
        "n_snapshots": len(snapshots),
        "timestamps": [round(s["sim_time_hours"], 1) for s in snapshots],
        "drift_scores": [round(s.get("drift_score") or 0, 3) for s in snapshots],
        "avg_t1": [round(s.get("avg_t1_us", 0), 1) for s in snapshots],
        "elapsed_s": elapsed,
    }

    print(f"  Snapshots: {len(snapshots)}, interval: 1h")
    print(f"  Drift: {tele['drift_scores'][0]:.3f} → {tele['drift_scores'][-1]:.3f}")
    print(f"  T1: {tele['avg_t1'][0]:.0f}μs → {tele['avg_t1'][-1]:.0f}μs")
    print(f"  Elapsed: {elapsed:.2f}s")

    return tele


def scenario_4_circuit_execution():
    """GHZ-5 circuit fidelity across backends."""
    print("\n=== Scenario 4: GHZ-5 Circuit Fidelity ===")
    from qiskit import QuantumCircuit

    # Build GHZ-5
    ghz = QuantumCircuit(5, 5)
    ghz.h(0)
    for i in range(4):
        ghz.cx(i, i + 1)
    ghz.measure(range(5), range(5))

    results = {}

    # FakeBackendAdapter
    fake = FakeBackendAdapter("FakeBrisbane")
    t0 = time.time()
    r = fake.run(ghz, shots=8192)
    results["FakeBackendAdapter"] = {
        "fidelity": round(r.fidelity, 4),
        "top_counts": dict(sorted(r.counts.items(), key=lambda x: -x[1])[:5]),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"  FakeBackendAdapter: fidelity={r.fidelity:.4f}")

    # ReplayBackend at t=0 and t=20
    replay = ReplayBackend.from_synthetic_history("FakeBrisbane", duration_hours=24, snapshot_interval_hours=2, seed=42)
    for t_h in [0, 20]:
        replay.set_time(t_h)
        t0 = time.time()
        r = replay.run(ghz, shots=8192)
        results[f"ReplayBackend_t{t_h}h"] = {
            "fidelity": round(r.fidelity, 4),
            "elapsed_s": round(time.time() - t0, 2),
        }
        print(f"  ReplayBackend(t={t_h}h): fidelity={r.fidelity:.4f}")

    # SyntheticDrift (stable vs degraded)
    for pname, profile in [("stable", STABLE), ("linear_24h", LINEAR_DECAY)]:
        backend = SyntheticDriftBackend("FakeBrisbane", profile)
        backend.set_time(24.0)
        t0 = time.time()
        r = backend.run(ghz, shots=8192)
        results[f"Synthetic_{pname}"] = {
            "fidelity": round(r.fidelity, 4),
            "elapsed_s": round(time.time() - t0, 2),
        }
        print(f"  Synthetic({pname}, t=24h): fidelity={r.fidelity:.4f}")

    return results


def scenario_5_pulse_vs_circuit():
    """Bridge pulse-level calibration and circuit-level execution."""
    print("\n=== Scenario 5: Pulse-Level π Calibration ===")

    # Calibrate π-pulse for a 100ns Gaussian drive
    cfg = RabiConfig(
        pulse_duration_ns=100,
        pulse_shape="gaussian",
        sigma_ns=25,
        dt_ns=0.5,
        amp_range=(0, 0.12),
        n_amps=41,
    )

    t0 = time.time()
    cal = estimate_pi_pulse(cfg, refine=True)
    elapsed = time.time() - t0

    # Also do a quick detuning sensitivity analysis
    detunings = np.linspace(-0.02, 0.02, 11)
    sensitivity = []
    for det in detunings:
        cfg_det = RabiConfig(
            pulse_duration_ns=100,
            pulse_shape="gaussian",
            sigma_ns=25,
            dt_ns=0.5,
            drive_freq_ghz=5.0 + det,
        )
        r = simulate_rabi(cal["pi_amp"], cfg_det)
        sensitivity.append({"detuning_mhz": round(det * 1e3, 1), "p1": round(r.final_p1, 6)})

    results = {
        "pi_amp_ghz": cal["pi_amp"],
        "pi_amp_mhz": round(cal["pi_amp"] * 1e3, 2),
        "pi_fidelity": cal["pi_fidelity"],
        "half_pi_amp_mhz": round(cal["half_pi_amp"] * 1e3, 2),
        "sensitivity": sensitivity,
        "elapsed_s": elapsed,
    }

    print(f"  π-amp: {results['pi_amp_mhz']:.2f} MHz")
    print(f"  π fidelity: {results['pi_fidelity']:.8f}")
    print(f"  ±20 MHz detuning: P(|1⟩) range "
          f"[{min(s['p1'] for s in sensitivity):.4f}, "
          f"{max(s['p1'] for s in sensitivity):.4f}]")
    print(f"  Elapsed: {elapsed:.2f}s")

    return results


def main():
    print("=" * 60)
    print("Day 11: Backend Validation & Cross-Backend Comparison")
    print("=" * 60)

    results = {}
    results["s1"] = scenario_1_health_comparison()
    results["s2"] = scenario_2_drift_profiles()
    results["s3"] = scenario_3_telemetry_stream()
    results["s4"] = scenario_4_circuit_execution()
    results["s5"] = scenario_5_pulse_vs_circuit()

    out_dir = Path(__file__).parent
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    total = sum(v.get("elapsed_s", 0) for v in results.values() if isinstance(v, dict))
    print(f"\n{'=' * 60}")
    print(f"Results saved. Total elapsed: {total:.1f}s")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
