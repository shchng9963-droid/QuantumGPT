"""Day 8 Figures: CalibrationAdvisor visualizations.

Generates 5 figures:
  1. advisory_timeline — health metrics + actions over time (multi-scenario)
  2. fidelity_probes — fidelity degradation across scenarios
  3. action_matrix — heatmap of action types by scenario
  4. changepoint_detection — drift score with changepoint markers
  5. qubit_ranking — worst vs best qubit T1 comparison
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# Style: clean Nature-like
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
})

COLORS = {
    "stable": "#4CAF50",
    "linear": "#FF9800",
    "sudden": "#F44336",
    "diurnal": "#2196F3",
}

SEVERITY_COLORS = {
    "nominal": "#4CAF50",
    "warning": "#FF9800",
    "critical": "#F44336",
}

ACTION_COLORS = {
    "OK": "#4CAF50",
    "WAIT": "#2196F3",
    "APPLY_MITIGATION": "#FF9800",
    "REMAP_LAYOUT": "#9C27B0",
    "EXCLUDE_QUBITS": "#795548",
    "RECALIBRATE": "#F44336",
}


def load_results():
    with open(os.path.join(DEMO_DIR, "results.json")) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(DEMO_DIR, f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}.pdf/png")


def fig1_advisory_timeline(data):
    """Health metrics over time for each scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Advisory Timeline: Health Metrics Over Time", fontsize=13, fontweight="bold")

    for ax, (name, color) in zip(axes.flat, COLORS.items()):
        d = data[name]
        times = [h["time_hours"] for h in d["health_timeline"]]
        t1s = [h["avg_t1_us"] for h in d["health_timeline"]]
        drifts = [h["drift_score"] or 0 for h in d["health_timeline"]]
        errors = [h["avg_2q_error"] for h in d["health_timeline"]]

        ax2 = ax.twinx()

        l1, = ax.plot(times, t1s, "o-", color=color, linewidth=1.5, markersize=4, label="T1 (us)")
        l2, = ax2.plot(times, drifts, "s--", color="#F44336", linewidth=1.2,
                       markersize=3, alpha=0.8, label="Drift score")

        # Mark changepoints
        for cp in d.get("changepoints", []):
            ax.axvline(cp, color="#F44336", linestyle=":", alpha=0.5, linewidth=1)

        ax.set_title(f"{name.upper()} — {d['severity']}", fontsize=10)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("T1 (μs)", color=color)
        ax2.set_ylabel("Drift score", color="#F44336")
        ax2.set_ylim(-0.05, 1.1)

        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=8)

    fig.tight_layout()
    save(fig, "advisory_timeline")


def fig2_fidelity_probes(data):
    """Fidelity degradation across scenarios."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Probe Circuit Fidelity Over Time", fontsize=13, fontweight="bold")

    for name, color in COLORS.items():
        probes = data[name]["probes"]
        if not probes:
            continue
        # Filter to ghz_5 for comparability
        ghz = [p for p in probes if p["circuit"] == "ghz_5"]
        if ghz:
            times = [p["time"] for p in ghz]
            fids = [p["fidelity"] for p in ghz]
            ax.plot(times, fids, "o-", color=color, linewidth=1.5,
                    markersize=5, label=f"{name}")

    ax.axhline(0.85, color="#FF9800", linestyle="--", alpha=0.5, label="Warning threshold")
    ax.axhline(0.70, color="#F44336", linestyle="--", alpha=0.5, label="Critical threshold")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.55, 1.0)
    ax.legend(fontsize=9)
    ax.set_title("GHZ-5 probe circuit", fontsize=10)

    fig.tight_layout()
    save(fig, "fidelity_probes")


def fig3_action_matrix(data):
    """Heatmap: action types by scenario."""
    scenarios = list(data.keys())
    all_actions = ["OK", "WAIT", "APPLY_MITIGATION", "REMAP_LAYOUT",
                   "EXCLUDE_QUBITS", "RECALIBRATE"]

    matrix = np.zeros((len(scenarios), len(all_actions)))
    for i, name in enumerate(scenarios):
        for action in data[name]["actions"]:
            j = all_actions.index(action["type"])
            matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(3, matrix.max()))

    ax.set_xticks(range(len(all_actions)))
    ax.set_xticklabels(all_actions, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.upper() for s in scenarios], fontsize=10)

    # Annotate cells
    for i in range(len(scenarios)):
        for j in range(len(all_actions)):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if val >= 2 else "black")

    fig.colorbar(im, ax=ax, label="Count", shrink=0.8)
    ax.set_title("Advisory Action Matrix by Scenario", fontsize=13, fontweight="bold")

    fig.tight_layout()
    save(fig, "action_matrix")


def fig4_changepoint_detection(data):
    """Drift score with changepoint markers."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle("Changepoint Detection in Drift Score", fontsize=13, fontweight="bold")

    for ax, name in zip(axes, ["linear", "sudden", "diurnal"]):
        d = data[name]
        times = [h["time_hours"] for h in d["health_timeline"]]
        drifts = [h["drift_score"] or 0 for h in d["health_timeline"]]

        ax.plot(times, drifts, "o-", color=COLORS[name], linewidth=1.5, markersize=5)
        ax.fill_between(times, drifts, alpha=0.15, color=COLORS[name])

        for cp in d.get("changepoints", []):
            ax.axvline(cp, color="#F44336", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.annotate(f"CP", (cp, max(drifts) * 0.95),
                        fontsize=7, color="#F44336", ha="center")

        ax.axhspan(0.5, 1.1, alpha=0.08, color="#F44336")
        ax.axhspan(0.2, 0.5, alpha=0.08, color="#FF9800")

        ax.set_title(name.upper(), fontsize=10)
        ax.set_xlabel("Time (hours)")
        if ax == axes[0]:
            ax.set_ylabel("Drift Score")
        ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()
    save(fig, "changepoint_detection")


def fig5_qubit_ranking(data):
    """Worst vs best qubit comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Qubit Ranking: Worst vs Best (by T1/T2/readout composite)",
                 fontsize=13, fontweight="bold")

    for ax, name in zip(axes, ["stable", "sudden"]):
        d = data[name]
        worst = d.get("worst_qubits", [])[:5]
        best = d.get("best_qubits", [])[:5]

        # Get T1 values from health timeline qubit data
        # We'll use the last health snapshot's avg T1 as reference
        last_health = d["health_timeline"][-1]
        avg_t1 = last_health["avg_t1_us"]

        x = np.arange(5)
        width = 0.35

        # We don't have individual qubit T1 in results.json, so show ranking
        ax.bar(x - width/2, range(5, 0, -1), width, color="#F44336", alpha=0.7,
               label="Worst qubits")
        ax.bar(x + width/2, range(1, 6), width, color="#4CAF50", alpha=0.7,
               label="Best qubits")

        worst_labels = [f"Q{q}" for q in worst] if worst else [f"Q?" for _ in range(5)]
        best_labels = [f"Q{q}" for q in best] if best else [f"Q?" for _ in range(5)]

        # Annotate
        for i, (wl, bl) in enumerate(zip(worst_labels, best_labels)):
            ax.text(i - width/2, 5.2 - i, wl, ha="center", fontsize=8, color="#F44336")
            ax.text(i + width/2, 0.8 + i, bl, ha="center", fontsize=8, color="#2E7D32")

        ax.set_title(f"{name.upper()} (avg T1={avg_t1:.1f}μs)", fontsize=10)
        ax.set_ylabel("Quality Rank")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Rank {i+1}" for i in range(5)], fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 6.5)

    fig.tight_layout()
    save(fig, "qubit_ranking")


def fig6_architecture(data):
    """Advisory chain architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("CalibrationAdvisor: Monitor → Diagnose → Act",
                 fontsize=14, fontweight="bold")

    box_style = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.3", linewidth=1.2)
    phase_style = dict(fontsize=12, fontweight="bold", ha="center", va="center")
    detail_style = dict(fontsize=8, ha="center", va="top", color="0.4")

    # Phase 1: Monitor
    ax.add_patch(plt.Rectangle((0.5, 3.5), 3, 2, fill=True,
                                facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5, zorder=2))
    ax.text(2, 5.0, "1. MONITOR", **phase_style, color="#1565C0")
    ax.text(2, 4.4, "Health snapshots\nProbe circuits\nQubit properties\n→ DuckDB", **detail_style)

    # Phase 2: Diagnose
    ax.add_patch(plt.Rectangle((4.5, 3.5), 3, 2, fill=True,
                                facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.5, zorder=2))
    ax.text(6, 5.0, "2. DIAGNOSE", **phase_style, color="#E65100")
    ax.text(6, 4.4, "Trend analysis (LR)\nChangepoint detection\nQubit ranking\nFidelity trend", **detail_style)

    # Phase 3: Act
    ax.add_patch(plt.Rectangle((8.5, 3.5), 3, 2, fill=True,
                                facecolor="#FFEBEE", edgecolor="#C62828", linewidth=1.5, zorder=2))
    ax.text(10, 5.0, "3. ACT", **phase_style, color="#C62828")
    ax.text(10, 4.4, "Rule engine\nAction generation\nPriority ranking\nAdvisory report", **detail_style)

    # Arrows
    arrow_props = dict(arrowstyle="-|>", color="0.3", linewidth=1.5)
    ax.annotate("", xy=(4.4, 4.5), xytext=(3.6, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(8.4, 4.5), xytext=(7.6, 4.5), arrowprops=arrow_props)

    # Backend box at bottom
    ax.add_patch(plt.Rectangle((1.5, 0.8), 3, 1.5, fill=True,
                                facecolor="#F3E5F5", edgecolor="#6A1B9A", linewidth=1.2, zorder=2))
    ax.text(3, 2.0, "ShadowBackend", fontsize=10, fontweight="bold",
            ha="center", va="center", color="#6A1B9A")
    ax.text(3, 1.3, "Fake / Replay / Synthetic", fontsize=8,
            ha="center", va="center", color="0.5")

    # DuckDB box at bottom right
    ax.add_patch(plt.Rectangle((5.5, 0.8), 2.5, 1.5, fill=True,
                                facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.2, zorder=2))
    ax.text(6.75, 2.0, "DuckDB", fontsize=10, fontweight="bold",
            ha="center", va="center", color="#2E7D32")
    ax.text(6.75, 1.3, "health / probes / events", fontsize=8,
            ha="center", va="center", color="0.5")

    # Actions box at bottom right
    ax.add_patch(plt.Rectangle((8.8, 0.8), 2.5, 1.5, fill=True,
                                facecolor="#FFF8E1", edgecolor="#F57F17", linewidth=1.2, zorder=2))
    ax.text(10.05, 2.0, "Actions", fontsize=10, fontweight="bold",
            ha="center", va="center", color="#F57F17")
    actions_text = "RECALIBRATE\nEXCLUDE\nREMAP\nMITIGATE\nWAIT / OK"
    ax.text(10.05, 1.3, actions_text, fontsize=7, ha="center", va="center", color="0.5")

    # Vertical arrows
    ax.annotate("", xy=(2, 3.4), xytext=(3, 2.4), arrowprops=dict(arrowstyle="-|>", color="#6A1B9A", linewidth=1))
    ax.annotate("", xy=(6, 3.4), xytext=(6.75, 2.4), arrowprops=dict(arrowstyle="-|>", color="#2E7D32", linewidth=1))
    ax.annotate("", xy=(10, 3.4), xytext=(10.05, 2.4), arrowprops=dict(arrowstyle="-|>", color="#F57F17", linewidth=1))

    save(fig, "advisor_architecture")


def main():
    print("Generating Day 8 figures...")
    data = load_results()

    fig1_advisory_timeline(data)
    fig2_fidelity_probes(data)
    fig3_action_matrix(data)
    fig4_changepoint_detection(data)
    fig5_qubit_ranking(data)
    fig6_architecture(data)

    print("Done! All figures in demos/day8/")


if __name__ == "__main__":
    main()
