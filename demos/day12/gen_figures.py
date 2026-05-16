"""Day 12 Demo — MVP CLI fidelity comparison across circuits and backends.

Generates:
  1. fidelity_comparison.png — bar chart: fidelity per circuit on FakeBrisbane
  2. backend_comparison.png — bar chart: GHZ-5 fidelity across 4 backends
"""

import json
import subprocess
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CIRCUITS = ["ghz", "qft", "bv", "vqe", "qaoa"]
BACKENDS = ["FakeBrisbane", "FakeKyiv", "FakeSherbrooke", "FakeTorino"]
SHOTS = 4096
OUT_DIR = "demos/day12"


def run_qgpt(circuit: str, backend: str = "FakeBrisbane", shots: int = SHOTS) -> dict:
    """Run qgpt simulate and parse JSON output."""
    cmd = [sys.executable, "-m", "cli", "simulate", circuit,
           "-b", backend, "-s", str(shots), "-j"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  WARN: {circuit}@{backend} failed: {result.stderr[:200]}")
        return {"fidelity": None, "circuit": circuit}
    return json.loads(result.stdout)


def plot_circuit_comparison():
    """Bar chart: fidelity per circuit on FakeBrisbane."""
    print("=== Circuit Comparison (FakeBrisbane) ===")
    results = []
    for circ in CIRCUITS:
        print(f"  Running {circ}...")
        r = run_qgpt(circ)
        results.append(r)
        fid = r.get("fidelity")
        print(f"    {circ}: fidelity={fid}")

    names = [r.get("circuit", c) for r, c in zip(results, CIRCUITS)]
    fids = [r.get("fidelity", 0) or 0 for r in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2ecc71" if f >= 0.9 else "#f39c12" if f >= 0.7 else "#e74c3c"
              for f in fids]
    bars = ax.bar(names, fids, color=colors, edgecolor="white", linewidth=0.5)

    for bar, fid in zip(bars, fids):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{fid:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fidelity", fontsize=12)
    ax.set_title("Circuit Fidelity on FakeBrisbane (4096 shots)", fontsize=13, fontweight="bold")
    ax.axhline(y=0.9, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(names)-0.5, 0.905, "0.9 threshold", fontsize=8, color="#888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = f"{OUT_DIR}/fidelity_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return results


def plot_backend_comparison():
    """Bar chart: GHZ-5 fidelity across all 4 backends."""
    print("\n=== Backend Comparison (GHZ-5) ===")
    results = []
    for be in BACKENDS:
        print(f"  Running ghz on {be}...")
        r = run_qgpt("ghz", backend=be)
        results.append(r)
        fid = r.get("fidelity")
        print(f"    {be}: fidelity={fid}")

    fids = [r.get("fidelity", 0) or 0 for r in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#3498db", "#9b59b6", "#1abc9c", "#e67e22"]
    bars = ax.bar(BACKENDS, fids, color=colors, edgecolor="white", linewidth=0.5)

    for bar, fid in zip(bars, fids):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{fid:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fidelity", fontsize=12)
    ax.set_title("GHZ-5 Fidelity Across Backends (4096 shots)", fontsize=13, fontweight="bold")
    ax.axhline(y=0.9, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = f"{OUT_DIR}/backend_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return results


if __name__ == "__main__":
    c_results = plot_circuit_comparison()
    b_results = plot_backend_comparison()
    print("\nDone! Check demos/day12/ for outputs.")
