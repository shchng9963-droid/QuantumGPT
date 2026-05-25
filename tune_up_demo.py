#!/usr/bin/env python3
"""tune_up_demo.py — Full qubit tune-up sequence demo.

Demonstrates the QuantumGPT agent workflow for characterizing a transmon qubit:
    1. Rabi oscillation → find pi-pulse amplitude
    2. Ramsey fringes   → measure T2* and detuning
    3. T1 relaxation    → measure energy relaxation time

Each step feeds its results into the next (pi-amp from Rabi is used by Ramsey/T1).
This is the kind of autonomous workflow the agent would execute when prompted:
    "Characterize qubit 0: find the pi-pulse, measure T2* and T1."

Usage:
    python tune_up_demo.py [--save-dir results/tune_up]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends.dynamics_lab_adapter import DynamicsLabAdapter
from experiments.rabi_experiment import RabiExperiment
from experiments.ramsey_experiment import RamseyExperiment
from experiments.t1_experiment import T1Experiment


def banner(msg: str) -> None:
    """Print a section banner."""
    width = 60
    print()
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def run_tune_up(
    qubit_freq_ghz: float = 5.0,
    t1_us: float = 200.0,
    t2_us: float = 150.0,
    save_dir: str | None = None,
) -> dict:
    """Execute the full tune-up sequence and return summary."""

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ── Backend ──────────────────────────────────────────
    banner("Initializing DynamicsLabAdapter")
    backend = DynamicsLabAdapter(
        num_qubits=1,
        qubit_freq_ghz=qubit_freq_ghz,
        t1_us=t1_us,
        t2_us=t2_us,
        readout_error=0.01,
    )
    print(f"  Backend: {backend.name}")
    print(f"  Qubit freq: {qubit_freq_ghz} GHz")
    print(f"  T1 (configured): {t1_us} us")
    print(f"  T2 (configured): {t2_us} us")

    results = {}

    # ── Step 1: Rabi Oscillation ─────────────────────────
    banner("Step 1: Rabi Oscillation")
    t0 = time.time()
    rabi_exp = RabiExperiment(
        qubit=0,
        amp_min=0.0,
        amp_max=0.08,
        n_points=40,
        pulse_duration_ns=100.0,
        shots=1024,
    )
    rabi_result = rabi_exp.run(backend)
    rabi_result = rabi_exp.analyze(rabi_result)
    rabi_time = time.time() - t0

    pi_amp = rabi_result.fit.parameters.get("pi_amplitude", 0.03) if rabi_result.fit else 0.03
    half_pi_amp = pi_amp / 2

    print(f"  Points: {rabi_exp.n_points}")
    print(f"  Duration: {rabi_time:.2f}s")
    if rabi_result.fit:
        print(f"  Pi amplitude: {pi_amp:.6f} GHz ({pi_amp*1e3:.2f} MHz)")
        print(f"  Pi/2 amplitude: {half_pi_amp:.6f} GHz")
        print(f"  R²: {rabi_result.fit.r_squared:.4f}")
    else:
        print("  WARNING: Fit failed, using default pi_amp = 0.03")

    results["rabi"] = {
        "pi_amplitude_ghz": float(pi_amp),
        "half_pi_amplitude_ghz": float(half_pi_amp),
        "r_squared": rabi_result.fit.r_squared if rabi_result.fit else 0.0,
        "elapsed_s": rabi_time,
    }

    # Update calibration store
    backend.update_calibration(
        __import__("backends.lab_backend", fromlist=["CalibrationUpdate"]).CalibrationUpdate(
            qubit_index=0,
            parameter="pi_amplitude",
            value=pi_amp,
            unit="GHz",
            source="tune_up_demo/rabi",
        )
    )

    # ── Step 2: Ramsey Fringes ───────────────────────────
    banner("Step 2: Ramsey Fringes (T2*)")
    t0 = time.time()
    ramsey_exp = RamseyExperiment(
        qubit=0,
        delay_max_ns=5000.0,
        n_points=50,
        pi_half_amplitude=half_pi_amp,
        artificial_detuning_mhz=2.0,
        shots=1024,
    )
    ramsey_result = ramsey_exp.run(backend)
    ramsey_result = ramsey_exp.analyze(ramsey_result)
    ramsey_time = time.time() - t0

    print(f"  Points: {ramsey_exp.n_points}")
    print(f"  Max delay: {ramsey_exp.delay_max_ns/1000:.1f} us")
    print(f"  Artificial detuning: {ramsey_exp.artificial_detuning_mhz} MHz")
    print(f"  Duration: {ramsey_time:.2f}s")
    if ramsey_result.fit:
        t2star_us = ramsey_result.fit.parameters.get("T2_star_us", 0)
        det_mhz = ramsey_result.fit.parameters.get("detuning_mhz", 0)
        print(f"  T2* = {t2star_us:.2f} us")
        print(f"  Detuning = {det_mhz:.3f} MHz")
        print(f"  R² = {ramsey_result.fit.r_squared:.4f}")
        print(f"  Model: {ramsey_result.fit.model}")
    else:
        print("  WARNING: Fit failed")
        t2star_us = 0.0
        det_mhz = 0.0

    results["ramsey"] = {
        "T2_star_us": float(t2star_us),
        "detuning_mhz": float(det_mhz),
        "r_squared": ramsey_result.fit.r_squared if ramsey_result.fit else 0.0,
        "elapsed_s": ramsey_time,
    }

    # ── Step 3: T1 Relaxation ────────────────────────────
    banner("Step 3: T1 Relaxation")
    t0 = time.time()
    t1_exp = T1Experiment(
        qubit=0,
        delay_max_ns=600_000.0,  # 600 us = 3x T1
        n_points=50,
        pi_amplitude=pi_amp,
        shots=1024,
    )
    t1_result = t1_exp.run(backend)
    t1_result = t1_exp.analyze(t1_result)
    t1_time = time.time() - t0

    print(f"  Points: {t1_exp.n_points}")
    print(f"  Max delay: {t1_exp.delay_max_ns/1000:.1f} us")
    print(f"  Pi amplitude used: {pi_amp:.6f} GHz")
    print(f"  Duration: {t1_time:.2f}s")
    if t1_result.fit:
        t1_meas = t1_result.fit.parameters.get("T1_us", 0)
        print(f"  T1 = {t1_meas:.1f} us (configured: {t1_us} us)")
        print(f"  R² = {t1_result.fit.r_squared:.4f}")
        print(f"  Model: {t1_result.fit.model}")
    else:
        print("  WARNING: Fit failed")
        t1_meas = 0.0

    results["t1"] = {
        "T1_us": float(t1_meas),
        "T1_configured_us": t1_us,
        "r_squared": t1_result.fit.r_squared if t1_result.fit else 0.0,
        "elapsed_s": t1_time,
    }

    # ── Summary ──────────────────────────────────────────
    banner("Tune-Up Summary")
    total_time = results["rabi"]["elapsed_s"] + results["ramsey"]["elapsed_s"] + results["t1"]["elapsed_s"]
    print(f"  Total wall time: {total_time:.2f}s")
    print()
    print(f"  Pi amplitude:  {pi_amp*1e3:.2f} MHz")
    print(f"  T2*:           {t2star_us:.2f} us")
    print(f"  T1:            {t1_meas:.1f} us")
    print()
    print(f"  All fits R² > 0.5: {'YES' if all(r.get('r_squared', 0) > 0.5 for r in results.values()) else 'NO'}")

    results["total_elapsed_s"] = total_time
    results["backend"] = {
        "name": backend.name,
        "qubit_freq_ghz": qubit_freq_ghz,
        "configured_t1_us": t1_us,
        "configured_t2_us": t2_us,
    }

    # ── Save ─────────────────────────────────────────────
    if save_dir:
        # JSON summary
        summary_path = os.path.join(save_dir, "tune_up_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Summary saved: {summary_path}")

        # Plots
        try:
            import matplotlib
            matplotlib.use("Agg")

            rabi_fig_path = os.path.join(save_dir, "01_rabi.png")
            rabi_exp.visualize(rabi_result, save_path=rabi_fig_path)
            print(f"  Rabi plot: {rabi_fig_path}")

            ramsey_fig_path = os.path.join(save_dir, "02_ramsey.png")
            ramsey_exp.visualize(ramsey_result, save_path=ramsey_fig_path)
            print(f"  Ramsey plot: {ramsey_fig_path}")

            t1_fig_path = os.path.join(save_dir, "03_t1.png")
            t1_exp.visualize(t1_result, save_path=t1_fig_path)
            print(f"  T1 plot: {t1_fig_path}")
        except ImportError:
            print("  (matplotlib not available, skipping plots)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Qubit tune-up demonstration")
    parser.add_argument("--save-dir", default="experiments/results/tune_up",
                        help="Directory to save results and plots")
    parser.add_argument("--qubit-freq", type=float, default=5.0,
                        help="Qubit frequency in GHz (default: 5.0)")
    parser.add_argument("--t1", type=float, default=200.0,
                        help="T1 in microseconds (default: 200)")
    parser.add_argument("--t2", type=float, default=150.0,
                        help="T2 in microseconds (default: 150)")
    args = parser.parse_args()

    results = run_tune_up(
        qubit_freq_ghz=args.qubit_freq,
        t1_us=args.t1,
        t2_us=args.t2,
        save_dir=args.save_dir,
    )

    # Exit code: 0 if all fits good, 1 otherwise
    all_good = all(
        results.get(k, {}).get("r_squared", 0) > 0.5
        for k in ["rabi", "ramsey", "t1"]
    )
    sys.exit(0 if all_good else 1)


if __name__ == "__main__":
    main()
