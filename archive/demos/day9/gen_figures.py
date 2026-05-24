"""Day 9 Figures: StreamingAdvisor visualizations.

Generates 5 figures:
  1. alert_timeline — alerts over simulated time (sudden degradation scenario)
  2. streaming_linear — live health metrics from streaming mode
  3. recovery_cycle — degradation → recovery with alert markers
  4. alert_type_breakdown — alert type distribution across scenarios
  5. streaming_architecture — architecture diagram
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

LEVEL_COLORS = {"info": "#4CAF50", "warning": "#FF9800", "critical": "#F44336"}
TYPE_COLORS = {
    "drift_high": "#F44336",
    "drift_spike": "#D32F2F",
    "t1_degrading": "#FF9800",
    "t1_low": "#E65100",
    "error_high": "#9C27B0",
    "recovery": "#4CAF50",
    "fidelity_drop": "#2196F3",
    "changepoint": "#795548",
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


def fig1_alert_timeline(data):
    """Alert timeline for sudden degradation scenario."""
    alerts = data["sudden_batch"]["alerts"]
    if not alerts:
        print("  Skipping alert_timeline — no alerts")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Alert Timeline: Sudden Degradation at t=5h",
                 fontsize=13, fontweight="bold")

    # Group by type for y-axis
    types = list(dict.fromkeys(a["type"] for a in alerts))
    type_to_y = {t: i for i, t in enumerate(types)}

    for a in alerts:
        y = type_to_y[a["type"]]
        color = LEVEL_COLORS.get(a["level"], "#999")
        marker = "^" if a["level"] == "critical" else "o" if a["level"] == "warning" else "s"
        ax.scatter(a["time"], y, c=color, marker=marker, s=60, zorder=3, alpha=0.8)

    ax.axvline(5.0, color="#F44336", linestyle="--", alpha=0.5, linewidth=1.5,
               label="Degradation onset (t=5h)")

    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=9)
    ax.set_xlabel("Simulated Time (hours)")
    ax.set_ylabel("Alert Type")
    ax.set_xlim(-0.5, 11)

    # Legend for levels
    handles = [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#F44336",
                    markersize=8, label="Critical"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800",
                    markersize=8, label="Warning"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50",
                    markersize=8, label="Info"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    fig.tight_layout()
    save(fig, "alert_timeline")


def fig2_streaming_linear(data):
    """Streaming mode: alerts over time for linear decay."""
    alerts = data["linear_stream"]["alerts"]
    if not alerts:
        print("  Skipping streaming_linear — no alerts")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Streaming Monitor: Linear Decay (24h simulated in 3s)",
                 fontsize=13, fontweight="bold")

    # Top: alerts by level over time
    for level, color in LEVEL_COLORS.items():
        level_alerts = [a for a in alerts if a["level"] == level]
        if level_alerts:
            times = [a["time"] for a in level_alerts]
            ax1.scatter(times, [level] * len(times), c=color, s=50, zorder=3,
                       alpha=0.8, label=f"{level} ({len(level_alerts)})")

    ax1.set_ylabel("Alert Level")
    ax1.legend(loc="upper left", fontsize=9)

    # Bottom: alert density (cumulative)
    all_times = sorted(a["time"] for a in alerts)
    ax2.plot(all_times, range(1, len(all_times) + 1), "o-",
             color="#1565C0", linewidth=1.5, markersize=3)
    ax2.fill_between(all_times, range(1, len(all_times) + 1),
                     alpha=0.1, color="#1565C0")
    ax2.set_xlabel("Simulated Time (hours)")
    ax2.set_ylabel("Cumulative Alerts")

    fig.tight_layout()
    save(fig, "streaming_linear")


def fig3_recovery_cycle(data):
    """Recovery detection: degradation → recovery with alert markers."""
    alerts = data["recovery"]["alerts"]
    if not alerts:
        print("  Skipping recovery_cycle — no alerts")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Recovery Detection: Degradation at t=3-6h, Recovery at t≈6.5h",
                 fontsize=13, fontweight="bold")

    # Background zones
    ax.axvspan(3, 6, alpha=0.1, color="#F44336", label="Degradation window")
    ax.axvspan(6, 7, alpha=0.1, color="#4CAF50", label="Recovery window")

    # Plot alerts
    for a in alerts:
        color = LEVEL_COLORS.get(a["level"], "#999")
        marker = "^" if a["level"] == "critical" else "o" if a["level"] == "warning" else "D"
        size = 100 if a["type"] == "recovery" else 40
        ax.scatter(a["time"], 0.5, c=color, marker=marker, s=size, zorder=3, alpha=0.7)

    # Annotate recovery
    recovery = [a for a in alerts if a["type"] == "recovery"]
    for r in recovery:
        ax.annotate(f"RECOVERY\nt={r['time']:.1f}h",
                    xy=(r["time"], 0.5), xytext=(r["time"] + 0.5, 0.8),
                    fontsize=9, fontweight="bold", color="#2E7D32",
                    arrowprops=dict(arrowstyle="->", color="#2E7D32"))

    ax.set_xlabel("Simulated Time (hours)")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)

    # Alert count annotation
    critical = sum(1 for a in alerts if a["level"] == "critical")
    warning = sum(1 for a in alerts if a["level"] == "warning")
    info = sum(1 for a in alerts if a["level"] == "info")
    ax.text(0.02, 0.95, f"Alerts: {critical} critical, {warning} warning, {info} info",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    save(fig, "recovery_cycle")


def fig4_alert_breakdown(data):
    """Alert type breakdown across scenarios."""
    scenarios = {
        "Sudden\n(batch)": data["sudden_batch"]["alerts"],
        "Linear\n(stream)": data["linear_stream"]["alerts"],
        "Recovery": data["recovery"]["alerts"],
    }

    all_types = sorted(set(a["type"] for alerts in scenarios.values() for a in alerts))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Alert Type Distribution by Scenario",
                 fontsize=13, fontweight="bold")

    for ax, (name, alerts) in zip(axes, scenarios.items()):
        type_counts = {}
        for a in alerts:
            type_counts[a["type"]] = type_counts.get(a["type"], 0) + 1

        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors = [TYPE_COLORS.get(t, "#999") for t in types]

            bars = ax.barh(types, counts, color=colors, alpha=0.8)
            for bar, count in zip(bars, counts):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        str(count), va="center", fontsize=9)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Count")
        ax.set_xlim(0, max(counts) * 1.3 if type_counts else 5)

    fig.tight_layout()
    save(fig, "alert_breakdown")


def fig5_architecture(data):
    """Streaming advisory architecture."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("StreamingAdvisor Architecture",
                 fontsize=14, fontweight="bold")

    box_kw = dict(linewidth=1.2, zorder=2)

    # Backend
    ax.add_patch(plt.Rectangle((0.3, 2.5), 2.2, 1.5, fill=True,
                                facecolor="#F3E5F5", edgecolor="#6A1B9A", **box_kw))
    ax.text(1.4, 3.5, "ShadowBackend", fontsize=9, fontweight="bold",
            ha="center", color="#6A1B9A")
    ax.text(1.4, 2.9, "set_time(t)\nget_properties()", fontsize=7,
            ha="center", color="0.5")

    # PropertiesStream
    ax.add_patch(plt.Rectangle((3.3, 2.5), 2.2, 1.5, fill=True,
                                facecolor="#E3F2FD", edgecolor="#1565C0", **box_kw))
    ax.text(4.4, 3.5, "PropertiesStream", fontsize=9, fontweight="bold",
            ha="center", color="#1565C0")
    ax.text(4.4, 2.9, "background thread\ntime acceleration\nobserver pattern", fontsize=7,
            ha="center", color="0.5")

    # HealthTracker
    ax.add_patch(plt.Rectangle((6.3, 2.5), 2.2, 1.5, fill=True,
                                facecolor="#FFF3E0", edgecolor="#E65100", **box_kw))
    ax.text(7.4, 3.5, "HealthTracker", fontsize=9, fontweight="bold",
            ha="center", color="#E65100")
    ax.text(7.4, 2.9, "sliding window\ntrend analysis\njump detection", fontsize=7,
            ha="center", color="0.5")

    # Alert Engine
    ax.add_patch(plt.Rectangle((9.3, 2.5), 2.2, 1.5, fill=True,
                                facecolor="#FFEBEE", edgecolor="#C62828", **box_kw))
    ax.text(10.4, 3.5, "Alert Engine", fontsize=9, fontweight="bold",
            ha="center", color="#C62828")
    ax.text(10.4, 2.9, "thresholds\ncooldown\ncallbacks", fontsize=7,
            ha="center", color="0.5")

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color="0.3", linewidth=1.5)
    ax.annotate("", xy=(3.2, 3.25), xytext=(2.6, 3.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.2, 3.25), xytext=(5.6, 3.25), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.2, 3.25), xytext=(8.6, 3.25), arrowprops=arrow_kw)

    # DuckDB (bottom)
    ax.add_patch(plt.Rectangle((3.3, 0.5), 2.2, 1.2, fill=True,
                                facecolor="#E8F5E9", edgecolor="#2E7D32", **box_kw))
    ax.text(4.4, 1.3, "DuckDB", fontsize=9, fontweight="bold",
            ha="center", color="#2E7D32")
    ax.text(4.4, 0.8, "health_snapshots\ndrift_events", fontsize=7,
            ha="center", color="0.5")

    # Callbacks (bottom right)
    ax.add_patch(plt.Rectangle((9.3, 0.5), 2.2, 1.2, fill=True,
                                facecolor="#FFF8E1", edgecolor="#F57F17", **box_kw))
    ax.text(10.4, 1.3, "Callbacks", fontsize=9, fontweight="bold",
            ha="center", color="#F57F17")
    ax.text(10.4, 0.8, "print / log / slack\nadvisor handoff", fontsize=7,
            ha="center", color="0.5")

    # Vertical arrows
    ax.annotate("", xy=(4.4, 2.4), xytext=(4.4, 1.8), arrowprops=dict(arrowstyle="-|>", color="#2E7D32", linewidth=1))
    ax.annotate("", xy=(10.4, 2.4), xytext=(10.4, 1.8), arrowprops=dict(arrowstyle="-|>", color="#F57F17", linewidth=1))

    # Alert types (top)
    alert_types = ["DRIFT_SPIKE", "DRIFT_HIGH", "T1_LOW", "T1_DEGRADING",
                   "ERROR_HIGH", "RECOVERY"]
    for i, at in enumerate(alert_types):
        x = 1.5 + i * 1.9
        color = "#F44336" if "DRIFT" in at else "#FF9800" if "T1" in at else "#9C27B0" if "ERROR" in at else "#4CAF50"
        ax.text(x, 5.2, at, fontsize=7, ha="center", color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, linewidth=0.8))

    ax.text(6.5, 5.7, "Alert Types", fontsize=10, ha="center", fontweight="bold")

    save(fig, "streaming_architecture")


def main():
    print("Generating Day 9 figures...")
    data = load_results()

    fig1_alert_timeline(data)
    fig2_streaming_linear(data)
    fig3_recovery_cycle(data)
    fig4_alert_breakdown(data)
    fig5_architecture(data)

    print("Done! All figures in demos/day9/")


if __name__ == "__main__":
    main()
