"""Generate Phase 0 showcase composite figure.

Assembles key visuals from 13 days into a single high-impact figure
suitable for showing to advisors/leadership.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

OUT = "demos/phase0_showcase.png"

# Curated selection: 6 panels that tell the story
PANELS = [
    ("demos/day11/architecture.png",           "System Architecture"),
    ("demos/day12/fidelity_comparison.png",    "Circuit Fidelity Benchmark"),
    ("demos/day10/rabi_curves.png",            "Pulse-Level: Rabi Oscillation"),
    ("demos/day8/changepoint_detection.png",   "Drift Changepoint Detection"),
    ("demos/day6/tool_call_flow.png",          "Agent Tool-Call Flow"),
    ("demos/day13/experiment_fidelity_scatter.png", "ExperimentRecord Analytics"),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("QuantumGPT · Phase 0 Showcase (2 weeks, 1 person)",
             fontsize=18, fontweight="bold", y=0.98)

for ax, (path, title) in zip(axes.flat, PANELS):
    if os.path.exists(path):
        img = mpimg.imread(path)
        ax.imshow(img)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.axis("off")

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
