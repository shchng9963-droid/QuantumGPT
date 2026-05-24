#!/usr/bin/env python3
"""Day 10 figures — clean Nature-like style for Rabi oscillation results."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT_DIR = Path(__file__).parent

# Style: Nature-like (clean, muted, white bg)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "font.size": 10,
    "font.family": "sans-serif",
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 9,
})

COLORS = ["#2166ac", "#d6604d", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def load_results():
    with open(OUT_DIR / "results.json") as f:
        return json.load(f)


def fig1_time_evolution(data):
    """P(|1⟩) vs time for a driven qubit — shows Rabi oscillations."""
    s = data["s1"]
    t = np.array(s["times"])
    p = np.array(s["populations"])
    amp = s["amplitude_ghz"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, p, color=COLORS[0], linewidth=1.2, label=f"Simulation (Ω={amp} GHz)")

    # Analytic overlay
    analytic = np.sin(amp * t / 2) ** 2
    ax.plot(t, analytic, "--", color=COLORS[1], linewidth=0.9, alpha=0.7, label="Analytic sin²(Ωt/2)")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("P(|1⟩)")
    ax.set_title("Rabi Oscillation — Time Evolution (Square Pulse)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")

    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"rabi_time_evolution.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ rabi_time_evolution.pdf/png")


def fig2_rabi_curve_comparison(data):
    """Square vs Gaussian Rabi curves side by side."""
    s2 = data["s2"]
    s3 = data["s3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Square
    ax1.plot(np.array(s2["amplitudes"]) * 1e3, s2["final_populations"],
             "o-", color=COLORS[0], markersize=3, linewidth=1.0, label="Square")

    # Analytic for square: sin²(Ω·T/2), T=100ns
    a_sq = np.array(s2["amplitudes"])
    analytic_sq = np.sin(a_sq * 100 / 2) ** 2
    ax1.plot(a_sq * 1e3, analytic_sq, "--", color=COLORS[1], linewidth=0.8, alpha=0.6,
             label="sin²(Ω·T/2)")

    ax1.axvline(s2["pi_amp_analytic"] * 1e3, color="#999999", linestyle=":", linewidth=0.8,
                label=f"π-amp (analytic)")
    ax1.set_xlabel("Amplitude (MHz)")
    ax1.set_ylabel("P(|1⟩)")
    ax1.set_title("Square Pulse (T=100 ns)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(-0.05, 1.1)

    # Gaussian
    ax2.plot(np.array(s3["amplitudes"]) * 1e3, s3["final_populations"],
             "o-", color=COLORS[2], markersize=3, linewidth=1.0, label="Gaussian (σ=25ns)")
    ax2.axvline(s3["pi_amp_found"] * 1e3, color="#999999", linestyle=":", linewidth=0.8,
                label=f"π-amp ≈ {s3['pi_amp_found']*1e3:.1f} MHz")
    ax2.set_xlabel("Amplitude (MHz)")
    ax2.set_title("Gaussian Pulse (σ=25 ns, T=100 ns)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(-0.05, 1.1)

    fig.suptitle("Rabi Oscillation Curves — Amplitude Sweep", fontsize=12, y=1.02)
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"rabi_curves.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ rabi_curves.pdf/png")


def fig3_chevron(data):
    """Rabi chevron — P(|1⟩) as function of amplitude and detuning."""
    s = data["s4"]
    detunings = np.array(s["detunings"]) * 1e3  # → MHz
    amps = np.array(s["amplitudes"]) * 1e3       # → MHz
    chevron = np.array(s["chevron"])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(amps, detunings, chevron, cmap="RdBu_r", shading="auto",
                       vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="P(|1⟩)")
    ax.set_xlabel("Drive Amplitude (MHz)")
    ax.set_ylabel("Detuning Δ (MHz)")
    ax.set_title("Rabi Chevron — Detuned Qubit Drive", fontsize=11)

    # Mark the on-resonance π-amp
    pi_amp_approx = s["amplitudes"][np.argmax(chevron[len(detunings)//2])] * 1e3
    ax.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.plot(pi_amp_approx, 0, "w*", markersize=10, markeredgecolor="black", markeredgewidth=0.5)

    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"rabi_chevron.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ rabi_chevron.pdf/png")


def fig4_pi_calibration(data):
    """Coarse + fine sweep for π-pulse calibration."""
    s = data["s5"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Coarse
    ax1.plot(np.array(s["coarse_amps"]) * 1e3, s["coarse_pops"],
             "o-", color=COLORS[0], markersize=4, linewidth=1.0)
    ax1.axvline(s["pi_amp"] * 1e3, color=COLORS[1], linestyle="--", linewidth=0.8,
                label=f"π-amp = {s['pi_amp']*1e3:.2f} MHz")
    ax1.set_xlabel("Amplitude (MHz)")
    ax1.set_ylabel("P(|1⟩)")
    ax1.set_title("Coarse Sweep (31 pts)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.1)

    # Fine
    if s["fine_amps"]:
        ax2.plot(np.array(s["fine_amps"]) * 1e3, s["fine_pops"],
                 "o-", color=COLORS[2], markersize=3, linewidth=1.0)
        ax2.axvline(s["pi_amp"] * 1e3, color=COLORS[1], linestyle="--", linewidth=0.8,
                    label=f"π-amp = {s['pi_amp']*1e3:.2f} MHz\nFidelity = {s['pi_fidelity']:.6f}")
        ax2.set_xlabel("Amplitude (MHz)")
        ax2.set_title("Fine Sweep (51 pts, zoomed)", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.set_ylim(-0.05, 1.1)

    fig.suptitle("π-Pulse Calibration — Gaussian Pulse", fontsize=12, y=1.02)
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pi_calibration.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ pi_calibration.pdf/png")


def fig5_pulse_shapes(data):
    """Visualize the two pulse shapes + their Bloch-sphere trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Panel 1: Pulse envelopes
    t = np.linspace(0, 100, 500)
    amp = 0.05
    square = np.full_like(t, amp)
    gaussian = amp * np.exp(-0.5 * ((t - 50) / 25) ** 2)

    axes[0].plot(t, square * 1e3, color=COLORS[0], linewidth=1.2, label="Square")
    axes[0].plot(t, gaussian * 1e3, color=COLORS[2], linewidth=1.2, label="Gaussian (σ=25ns)")
    axes[0].fill_between(t, 0, gaussian * 1e3, color=COLORS[2], alpha=0.15)
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Ω(t) (MHz)")
    axes[0].set_title("Drive Envelopes", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, amp * 1e3 * 1.15)

    # Panel 2: Bloch sphere projection (XZ plane) for square pulse π-rotation
    from dynamics.rabi import simulate_rabi, RabiConfig
    cfg = RabiConfig(pulse_duration_ns=100, pulse_shape="square", dt_ns=0.2)
    pi_amp = np.pi / 100
    r = simulate_rabi(pi_amp, cfg)

    # Bloch coordinates from statevector: |ψ⟩ = c0|0⟩ + c1|1⟩
    c0 = r.statevectors[:, 0]
    c1 = r.statevectors[:, 1]
    bx = 2 * np.real(np.conj(c0) * c1)
    by = 2 * np.imag(np.conj(c0) * c1)
    bz = np.abs(c0) ** 2 - np.abs(c1) ** 2

    # Bloch circle
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[1].plot(np.sin(theta), np.cos(theta), color="#dddddd", linewidth=0.8)
    axes[1].plot([-1.1, 1.1], [0, 0], color="#eeeeee", linewidth=0.5)
    axes[1].plot([0, 0], [-1.1, 1.1], color="#eeeeee", linewidth=0.5)

    # Trajectory
    colors_traj = plt.cm.viridis(np.linspace(0, 1, len(bx)))
    for i in range(len(bx) - 1):
        axes[1].plot(bx[i:i+2], bz[i:i+2], color=colors_traj[i], linewidth=1.5)

    axes[1].plot(bx[0], bz[0], "o", color=COLORS[0], markersize=8, label="|0⟩")
    axes[1].plot(bx[-1], bz[-1], "s", color=COLORS[1], markersize=8, label="|1⟩")
    axes[1].set_xlabel("⟨X⟩")
    axes[1].set_ylabel("⟨Z⟩")
    axes[1].set_title("Bloch XZ (π-pulse)", fontsize=10)
    axes[1].set_xlim(-1.3, 1.3)
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=8, loc="upper right")

    # Panel 3: Gaussian π-pulse Bloch trajectory
    cfg_g = RabiConfig(pulse_duration_ns=100, pulse_shape="gaussian", sigma_ns=25, dt_ns=0.2)
    # Use the calibrated Gaussian π-amp from scenario 5
    s5 = data["s5"]
    r_g = simulate_rabi(s5["pi_amp"], cfg_g)

    c0g = r_g.statevectors[:, 0]
    c1g = r_g.statevectors[:, 1]
    bxg = 2 * np.real(np.conj(c0g) * c1g)
    bzg = np.abs(c0g) ** 2 - np.abs(c1g) ** 2

    axes[2].plot(np.sin(theta), np.cos(theta), color="#dddddd", linewidth=0.8)
    axes[2].plot([-1.1, 1.1], [0, 0], color="#eeeeee", linewidth=0.5)
    axes[2].plot([0, 0], [-1.1, 1.1], color="#eeeeee", linewidth=0.5)

    colors_g = plt.cm.magma(np.linspace(0, 1, len(bxg)))
    for i in range(len(bxg) - 1):
        axes[2].plot(bxg[i:i+2], bzg[i:i+2], color=colors_g[i], linewidth=1.5)

    axes[2].plot(bxg[0], bzg[0], "o", color=COLORS[0], markersize=8, label="|0⟩")
    axes[2].plot(bxg[-1], bzg[-1], "s", color=COLORS[1], markersize=8, label="|1⟩")
    axes[2].set_xlabel("⟨X⟩")
    axes[2].set_ylabel("⟨Z⟩")
    axes[2].set_title("Bloch XZ (Gaussian π)", fontsize=10)
    axes[2].set_xlim(-1.3, 1.3)
    axes[2].set_ylim(-1.3, 1.3)
    axes[2].set_aspect("equal")
    axes[2].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pulse_shapes_bloch.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ pulse_shapes_bloch.pdf/png")


def main():
    print("Generating Day 10 figures...")
    data = load_results()
    fig1_time_evolution(data)
    fig2_rabi_curve_comparison(data)
    fig3_chevron(data)
    fig4_pi_calibration(data)
    fig5_pulse_shapes(data)
    print("Done!")


if __name__ == "__main__":
    main()
