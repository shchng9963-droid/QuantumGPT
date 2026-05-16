#!/usr/bin/env python3
"""Day 11 figures — Backend comparison and validation visualizations."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT_DIR = Path(__file__).parent

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


def fig1_drift_profiles(data):
    """4 drift profiles: T1 and drift score over 24h."""
    s = data["s2"]
    times = s["times"]
    profiles = ["stable", "linear_decay", "sudden_degradation", "custom_recovery"]
    labels = ["Stable", "Linear Decay", "Sudden Degradation", "Custom Recovery"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for i, (pname, label) in enumerate(zip(profiles, labels)):
        ax1.plot(times, s[pname]["t1_us"], color=COLORS[i], linewidth=1.3, label=label)
        ax2.plot(times, s[pname]["drift_score"], color=COLORS[i], linewidth=1.3, label=label)

    ax1.set_ylabel("Avg T1 (μs)")
    ax1.set_title("Backend Drift Profiles — 24h Evolution", fontsize=11)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_ylim(0, 260)

    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Drift Score")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"drift_profiles.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ drift_profiles.pdf/png")


def fig2_telemetry_stream(data):
    """PropertiesStream 24h telemetry: T1 + drift over time."""
    s = data["s3"]
    times = s["timestamps"]
    t1 = s["avg_t1"]
    drift = s["drift_scores"]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color1 = COLORS[0]
    color2 = COLORS[1]

    ax1.plot(times, t1, "o-", color=color1, markersize=4, linewidth=1.0, label="Avg T1")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Avg T1 (μs)", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 260)

    ax2 = ax1.twinx()
    ax2.plot(times, drift, "s-", color=color2, markersize=4, linewidth=1.0, label="Drift Score")
    ax2.set_ylabel("Drift Score", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.05, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    ax1.set_title("PropertiesStream Telemetry — Linear Decay (24h)", fontsize=11)
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"telemetry_stream.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ telemetry_stream.pdf/png")


def fig3_circuit_fidelity(data):
    """GHZ-5 fidelity comparison across backends."""
    s = data["s4"]
    backends = list(s.keys())
    fidelities = [s[b]["fidelity"] for b in backends]

    # Clean up names
    clean_names = []
    for b in backends:
        b = b.replace("_", "\n").replace("Backend", "")
        clean_names.append(b)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(len(backends)), fidelities, color=[COLORS[i % len(COLORS)] for i in range(len(backends))],
                   edgecolor="#333333", linewidth=0.5)

    ax.set_yticks(range(len(backends)))
    ax.set_yticklabels(clean_names, fontsize=9)
    ax.set_xlabel("GHZ-5 Fidelity")
    ax.set_title("Circuit Fidelity Across Backend Types", fontsize=11)
    ax.set_xlim(0.8, 1.0)
    ax.axvline(1.0, color="#cccccc", linestyle="--", linewidth=0.5)

    for i, (bar, fid) in enumerate(zip(bars, fidelities)):
        ax.text(fid + 0.002, i, f"{fid:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"circuit_fidelity.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ circuit_fidelity.pdf/png")


def fig4_detuning_sensitivity(data):
    """π-pulse sensitivity to detuning."""
    s = data["s5"]
    sens = s["sensitivity"]
    det = [x["detuning_mhz"] for x in sens]
    p1 = [x["p1"] for x in sens]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(det, p1, "o-", color=COLORS[0], markersize=5, linewidth=1.2)
    ax.axhline(1.0, color="#cccccc", linestyle="--", linewidth=0.5)
    ax.axvline(0.0, color="#cccccc", linestyle=":", linewidth=0.5)

    ax.fill_between(det, p1, alpha=0.1, color=COLORS[0])

    ax.set_xlabel("Drive Detuning (MHz)")
    ax.set_ylabel("P(|1⟩) at π-pulse")
    ax.set_title(f"π-Pulse Detuning Sensitivity (amp={s['pi_amp_mhz']:.1f} MHz)", fontsize=11)
    ax.set_ylim(0.7, 1.05)

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"detuning_sensitivity.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ detuning_sensitivity.pdf/png")


def fig5_architecture(data):
    """Backend architecture overview — text-based diagram as figure."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(5, 9.5, "QuantumGPT Backend Architecture", fontsize=14,
            ha="center", va="center", fontweight="bold")

    # Boxes
    boxes = [
        # (x, y, w, h, label, color)
        (0.5, 7.5, 3, 1.2, "FakeBackendAdapter\n(Static IBM Snapshot)", COLORS[0]),
        (3.8, 7.5, 3, 1.2, "ReplayBackend\n(Time-Travel Drift)", COLORS[2]),
        (7.1, 7.5, 2.5, 1.2, "SyntheticDrift\n(Parametric Profiles)", COLORS[3]),
        (1.5, 5.5, 3.5, 1.0, "PropertiesStream\n(Telemetry Layer)", COLORS[4]),
        (5.5, 5.5, 3.5, 1.0, "CalibrationAdvisor\n+ StreamingAdvisor", COLORS[1]),
        (0.5, 3.5, 4, 1.0, "dynamics/rabi.py\n(Pulse-Level: qiskit-dynamics)", COLORS[0]),
        (5.5, 3.5, 4, 1.0, "DuckDB Store\n(Persistent Telemetry)", COLORS[5]),
        (2.5, 1.5, 5, 1.0, "QuantumAgent + CLI\n(User Interface)", "#666666"),
    ]

    for x, y, w, h, label, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")

    # Arrows (simplified)
    arrow_kw = dict(arrowstyle="->", color="#666666", lw=1.2)
    from matplotlib.patches import FancyArrowPatch
    arrows = [
        ((2, 7.5), (2.5, 6.5)),    # Fake → Stream
        ((5.3, 7.5), (4, 6.5)),    # Replay → Stream
        ((8.3, 7.5), (7.5, 6.5)),  # Synthetic → Advisor
        ((4, 6.0), (5.5, 6.0)),    # Stream → Advisor
        ((7.2, 5.5), (7.2, 4.5)),  # Advisor → DuckDB
        ((3.2, 5.5), (3.2, 4.5)),  # Stream → DuckDB
        ((5, 3.5), (5, 2.5)),      # DuckDB → Agent
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))

    # Stats box
    stats = (
        "Test Suite: 96 pass / 1 skip\n"
        "Backends: 3 shadow + 1 pulse\n"
        f"Dependencies: qiskit 1.3 + dynamics 0.6\n"
    )
    ax.text(0.3, 0.5, stats, fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#cccccc"))

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"architecture.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ architecture.pdf/png")


def main():
    print("Generating Day 11 figures...")
    data = load_results()
    fig1_drift_profiles(data)
    fig2_telemetry_stream(data)
    fig3_circuit_fidelity(data)
    fig4_detuning_sensitivity(data)
    fig5_architecture(data)
    print("Done!")


if __name__ == "__main__":
    main()
