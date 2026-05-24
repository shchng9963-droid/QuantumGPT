#!/usr/bin/env python3
"""Day 10 Demo: Rabi oscillation simulation via qiskit-dynamics.

Scenarios:
1. Single square pulse — time evolution of P(|1⟩)
2. Amplitude sweep (Rabi curve) — square pulse
3. Gaussian pulse comparison
4. Detuned drive — Rabi chevron
5. Pi-pulse calibration with refinement
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dynamics.rabi import (
    RabiConfig,
    simulate_rabi,
    sweep_rabi,
    estimate_pi_pulse,
)


def scenario_1_time_evolution():
    """Single square pulse: watch P(|1⟩) oscillate in time."""
    print("\n=== Scenario 1: Time Evolution (Square Pulse) ===")
    cfg = RabiConfig(
        pulse_duration_ns=200,
        pulse_shape="square",
        dt_ns=0.2,
    )
    amp = 0.02  # GHz
    t0 = time.time()
    result = simulate_rabi(amp, cfg)
    elapsed = time.time() - t0

    # Analytic check: P(|1⟩) = sin²(Ω·t/2)
    analytic = np.sin(amp * result.times_ns / 2) ** 2
    max_err = np.max(np.abs(result.populations - analytic))

    print(f"  Amplitude: {amp} GHz, Duration: {cfg.pulse_duration_ns} ns")
    print(f"  Time points: {len(result.times_ns)}")
    print(f"  Final P(|1⟩): {result.final_p1:.6f}")
    print(f"  Max error vs analytic: {max_err:.2e}")
    print(f"  Elapsed: {elapsed:.2f}s")

    return {
        "scenario": "time_evolution_square",
        "amplitude_ghz": amp,
        "duration_ns": cfg.pulse_duration_ns,
        "n_time_points": len(result.times_ns),
        "final_p1": result.final_p1,
        "max_analytic_error": float(max_err),
        "elapsed_s": elapsed,
        "times": result.times_ns.tolist(),
        "populations": result.populations.tolist(),
    }


def scenario_2_rabi_sweep_square():
    """Amplitude sweep with square pulse — classic Rabi oscillation."""
    print("\n=== Scenario 2: Rabi Amplitude Sweep (Square) ===")
    cfg = RabiConfig(
        pulse_duration_ns=100,
        pulse_shape="square",
        dt_ns=0.5,
        amp_range=(0, 0.08),
        n_amps=81,
    )
    t0 = time.time()
    sweep = sweep_rabi(cfg)
    elapsed = time.time() - t0

    # Should see oscillations: P(|1⟩) = sin²(Ω·T/2)
    pi_amp = sweep.pi_amplitude
    analytic_pi = np.pi / cfg.pulse_duration_ns

    print(f"  Sweep: {cfg.n_amps} amplitudes from {cfg.amp_range[0]} to {cfg.amp_range[1]} GHz")
    print(f"  Found π-amp: {pi_amp:.6f} GHz (analytic: {analytic_pi:.6f})")
    print(f"  Max P(|1⟩): {np.max(sweep.final_populations):.6f}")
    print(f"  Elapsed: {elapsed:.2f}s")

    return {
        "scenario": "rabi_sweep_square",
        "n_amps": cfg.n_amps,
        "pi_amp_found": pi_amp,
        "pi_amp_analytic": float(analytic_pi),
        "max_p1": float(np.max(sweep.final_populations)),
        "elapsed_s": elapsed,
        "amplitudes": sweep.amplitudes.tolist(),
        "final_populations": sweep.final_populations.tolist(),
    }


def scenario_3_gaussian_sweep():
    """Gaussian pulse sweep — smoother envelope, different Rabi curve."""
    print("\n=== Scenario 3: Rabi Sweep (Gaussian Pulse) ===")
    cfg = RabiConfig(
        pulse_duration_ns=100,
        pulse_shape="gaussian",
        sigma_ns=25,
        dt_ns=0.5,
        amp_range=(0, 0.12),
        n_amps=61,
    )
    t0 = time.time()
    sweep = sweep_rabi(cfg)
    elapsed = time.time() - t0

    pi_amp = sweep.pi_amplitude
    print(f"  Gaussian (σ={cfg.sigma_ns}ns), Duration={cfg.pulse_duration_ns}ns")
    print(f"  Found π-amp: {pi_amp:.6f} GHz")
    print(f"  Max P(|1⟩): {np.max(sweep.final_populations):.6f}")
    print(f"  Elapsed: {elapsed:.2f}s")

    return {
        "scenario": "rabi_sweep_gaussian",
        "sigma_ns": cfg.sigma_ns,
        "pi_amp_found": pi_amp,
        "max_p1": float(np.max(sweep.final_populations)),
        "elapsed_s": elapsed,
        "amplitudes": sweep.amplitudes.tolist(),
        "final_populations": sweep.final_populations.tolist(),
    }


def scenario_4_detuned_chevron():
    """Detuned Rabi: sweep amplitude at several detunings → chevron."""
    print("\n=== Scenario 4: Detuned Rabi (Chevron) ===")
    detunings = np.linspace(-0.05, 0.05, 21)  # GHz
    amps = np.linspace(0, 0.08, 41)

    t0 = time.time()
    chevron = np.zeros((len(detunings), len(amps)))
    for i, det in enumerate(detunings):
        for j, amp in enumerate(amps):
            cfg = RabiConfig(
                pulse_duration_ns=100,
                pulse_shape="square",
                dt_ns=1.0,  # coarser for speed
                drive_freq_ghz=5.0 + det,
            )
            r = simulate_rabi(amp, cfg)
            chevron[i, j] = r.final_p1
    elapsed = time.time() - t0

    print(f"  Grid: {len(detunings)} detunings × {len(amps)} amplitudes = {len(detunings)*len(amps)} sims")
    print(f"  Max P(|1⟩): {np.max(chevron):.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "scenario": "detuned_chevron",
        "n_detunings": len(detunings),
        "n_amps": len(amps),
        "max_p1": float(np.max(chevron)),
        "elapsed_s": elapsed,
        "detunings": detunings.tolist(),
        "amplitudes": amps.tolist(),
        "chevron": chevron.tolist(),
    }


def scenario_5_pi_calibration():
    """Find π-pulse with coarse+fine sweep."""
    print("\n=== Scenario 5: π-Pulse Calibration ===")
    cfg = RabiConfig(
        pulse_duration_ns=100,
        pulse_shape="gaussian",
        sigma_ns=25,
        dt_ns=0.5,
        amp_range=(0, 0.12),
        n_amps=31,
    )
    t0 = time.time()
    result = estimate_pi_pulse(cfg, refine=True)
    elapsed = time.time() - t0

    print(f"  π-pulse amplitude: {result['pi_amp']:.6f} GHz")
    print(f"  π-pulse fidelity: {result['pi_fidelity']:.8f}")
    print(f"  π/2-pulse amplitude: {result['half_pi_amp']:.6f} GHz")
    print(f"  Coarse sweep: {cfg.n_amps} pts, Fine sweep: 51 pts")
    print(f"  Elapsed: {elapsed:.2f}s")

    return {
        "scenario": "pi_calibration",
        "pi_amp": result["pi_amp"],
        "pi_fidelity": result["pi_fidelity"],
        "half_pi_amp": result["half_pi_amp"],
        "elapsed_s": elapsed,
        "coarse_amps": result["coarse_sweep"].amplitudes.tolist(),
        "coarse_pops": result["coarse_sweep"].final_populations.tolist(),
        "fine_amps": result["fine_sweep"].amplitudes.tolist() if result["fine_sweep"] else None,
        "fine_pops": result["fine_sweep"].final_populations.tolist() if result["fine_sweep"] else None,
    }


def main():
    print("=" * 60)
    print("Day 10: Pulse-Level Rabi Simulation via qiskit-dynamics")
    print("=" * 60)

    results = {}
    results["s1"] = scenario_1_time_evolution()
    results["s2"] = scenario_2_rabi_sweep_square()
    results["s3"] = scenario_3_gaussian_sweep()
    results["s4"] = scenario_4_detuned_chevron()
    results["s5"] = scenario_5_pi_calibration()

    # Save results
    out_dir = Path(__file__).parent
    out_path = out_dir / "results.json"

    # Convert for serialization (strip large arrays from chevron for json size)
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()}

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")
    total_time = sum(v.get("elapsed_s", 0) for v in results.values())
    print(f"Total elapsed: {total_time:.1f}s")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
