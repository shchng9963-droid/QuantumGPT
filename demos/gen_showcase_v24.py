"""Generate v2.4 showcase figure.

Creates a 3-panel summary showcasing v2.4 capabilities:
  Panel A: DRAG calibration leakage curve
  Panel B: Randomized Benchmarking decay
  Panel C: Tune-up sequence timeline (6 steps)

Uses data from the latest tuneup_demo run.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "output", "tuneup", "tuneup_trace.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUT_PATH = os.path.join(OUT_DIR, "v24_showcase.png")


def load_trace():
    if not os.path.exists(TRACE_PATH):
        print(f"Trace not found at {TRACE_PATH}")
        print("Run: python demos/tuneup_demo.py first")
        sys.exit(1)
    with open(TRACE_PATH) as f:
        return json.load(f)


def main():
    trace = load_trace()
    steps = trace["steps"]
    cal = trace["calibration"]

    fig = plt.figure(figsize=(16, 5.5))
    fig.suptitle("QuantumGPT v2.4 — End-to-End Tune-up Showcase",
                 fontsize=15, fontweight="bold", y=0.98)

    # ── Panel A: DRAG leakage ──
    ax1 = fig.add_subplot(131)
    drag_step = steps[3]  # step 4 = DRAG
    drag_out = drag_step["output"]
    if "coarse_sweep" in drag_out:
        alphas = drag_out["coarse_sweep"]["alphas"]
        leakages = drag_out["coarse_sweep"]["leakages"]
        ax1.semilogy(alphas, leakages, "o-", color="#2c7fb8", markersize=4, linewidth=1.2)
        opt_a = cal["drag_alpha"]
        ax1.axvline(opt_a, color="#e7298a", linestyle="--", linewidth=1,
                    label=f"α* = {opt_a:.3f}")
        ax1.legend(fontsize=9)
    ax1.set_xlabel("DRAG α", fontsize=11)
    ax1.set_ylabel("Leakage P(|2⟩)", fontsize=11)
    ax1.set_title("(a) DRAG Calibration", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # ── Panel B: RB decay ──
    ax2 = fig.add_subplot(132)
    rb_step = steps[4]  # step 5 = RB
    rb_out = rb_step["output"]
    if "sequence_lengths" in rb_out:
        lengths = rb_out["sequence_lengths"]
        survivals = rb_out["survival_probabilities"]
        ax2.plot(lengths, survivals, "o-", color="#d95f02", markersize=5, linewidth=1.2)
        epc = cal["error_per_clifford"]
        ax2.text(0.95, 0.05, f"EPC = {epc:.2e}", transform=ax2.transAxes,
                 fontsize=10, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax2.set_xlabel("Clifford sequence length", fontsize=11)
    ax2.set_ylabel("Survival probability", fontsize=11)
    ax2.set_title("(b) Randomized Benchmarking", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linewidth=0.5)

    # ── Panel C: Tune-up pipeline timeline ──
    ax3 = fig.add_subplot(133)
    step_names = [s["name"] for s in steps]
    step_times = [s["elapsed_s"] for s in steps]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]

    y_pos = np.arange(len(step_names))
    bars = ax3.barh(y_pos, step_times, color=colors[:len(steps)], edgecolor="white", height=0.6)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(step_names, fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel("Time (s)", fontsize=11)
    ax3.set_title("(c) Tune-up Pipeline", fontsize=12, fontweight="bold")

    # Add time labels
    for bar, t in zip(bars, step_times):
        ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{t:.2f}s", va="center", fontsize=8)

    # Result badge
    status = "PASS ✓" if trace["success"] else "ITERATE"
    total = trace["total_time_s"]
    ax3.text(0.95, 0.95, f"{status}\n{total:.1f}s total",
             transform=ax3.transAxes, fontsize=10, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#d4edda" if trace["success"] else "#f8d7da",
                       alpha=0.9))

    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(labelsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PATH.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Showcase saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
