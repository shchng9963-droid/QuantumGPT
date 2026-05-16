"""Generate Phase 0 showcase figure — single composite image for presentations."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

OUT = Path("PHASE0_SHOWCASE.png")
DEMOS = Path("demos")

# Pick the most visually impressive figures
PANELS = [
    ("demos/day11/architecture.png",    "System Architecture"),
    ("demos/day10/rabi_chevron.png",    "Rabi Chevron (Pulse-Level)"),
    ("demos/day12/fidelity_comparison.png", "Circuit Fidelity Benchmark"),
    ("demos/day8/changepoint_detection.png", "Drift Changepoint Detection"),
    ("demos/day9/streaming_architecture.png", "Streaming Telemetry"),
    ("demos/day13/experiment_fidelity_scatter.png", "ExperimentRecord Query"),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("QuantumGPT — Phase 0 Showcase\n"
             "Device-Aware Closed-Loop Agent for NISQ Workflows",
             fontsize=18, fontweight="bold", y=0.98)

for ax, (path, title) in zip(axes.flat, PANELS):
    try:
        img = mpimg.imread(path)
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, f"[{path}]\n{e}", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.axis("off")

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(str(OUT), dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
