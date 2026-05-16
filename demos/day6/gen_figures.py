"""Day 6 visualizations — agent loop architecture + tool call flow."""
import sys, os, json
sys.path.insert(0, "/home/wangshuchang/quantumgpt")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "/home/wangshuchang/quantumgpt/demos/day6"
os.makedirs(OUT, exist_ok=True)

# Load results
with open(os.path.join(OUT, "agent_runs.json")) as f:
    results = json.load(f)

# ═══════════════════════════════════════════════════
# Fig 1: Agent Architecture Diagram
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")
fig.patch.set_facecolor("white")

# Boxes
box_style = dict(boxstyle="round,pad=0.4", facecolor="#E8F0FE", edgecolor="#4285F4", linewidth=1.5)
tool_style = dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#FB8C00", linewidth=1.5)
result_style = dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#43A047", linewidth=1.5)

# User
ax.text(1.5, 6, "User Prompt", fontsize=12, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FCE4EC", edgecolor="#E91E63", linewidth=1.5))

# LLM
ax.text(5, 6, "LLM\n(Claude / RulePlanner)", fontsize=11, fontweight="bold", ha="center", va="center",
        bbox=box_style)

# While loop box
loop_rect = FancyBboxPatch((2.5, 1.8), 5.5, 3.5, boxstyle="round,pad=0.2",
                            facecolor="#FAFAFA", edgecolor="#757575", linewidth=2, linestyle="--")
ax.add_patch(loop_rect)
ax.text(5.25, 5.0, "while loop (max 15 turns)", fontsize=9, ha="center",
        fontstyle="italic", color="#757575")

# Tool Executor
ax.text(5, 3.5, "Tool Executor", fontsize=11, fontweight="bold", ha="center", va="center",
        bbox=tool_style)

# Tools (5 boxes)
tools = ["get_backend\n_health", "get_qubit\n_properties", "run_circuit", "list\n_benchmarks", "diagnose\n_& suggest"]
for i, t in enumerate(tools):
    x = 1.5 + i * 1.8
    ax.text(x, 2.2, t, fontsize=7, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF8E1", edgecolor="#FFA000", linewidth=1))

# Backend
ax.text(8.5, 3.5, "Shadow\nBackend", fontsize=10, fontweight="bold", ha="center", va="center",
        bbox=result_style)

# Final answer
ax.text(5, 0.7, "Final Answer", fontsize=12, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#43A047", linewidth=1.5))

# Arrows
arrow_kw = dict(arrowstyle="-|>", color="#333", linewidth=1.5, mutation_scale=15)
ax.annotate("", xy=(3.5, 6), xytext=(2.5, 6), arrowprops=arrow_kw)
ax.annotate("", xy=(5, 4.6), xytext=(5, 5.5), arrowprops=arrow_kw)
ax.annotate("", xy=(5, 3.9), xytext=(5, 4.5),
            arrowprops=dict(arrowstyle="-|>", color="#FB8C00", linewidth=1.5, mutation_scale=15))

# Tool executor -> Backend
ax.annotate("", xy=(7.5, 3.5), xytext=(6.2, 3.5),
            arrowprops=dict(arrowstyle="<->", color="#43A047", linewidth=1.5, mutation_scale=15))

# Back loop
ax.annotate("", xy=(6.5, 5.7), xytext=(6.5, 4.0),
            arrowprops=dict(arrowstyle="-|>", color="#4285F4", linewidth=1.2, mutation_scale=12,
                           connectionstyle="arc3,rad=-0.3"))
ax.text(7.3, 4.8, "tool results", fontsize=8, color="#4285F4", fontstyle="italic")

# Final arrow
ax.annotate("", xy=(5, 1.1), xytext=(5, 1.7),
            arrowprops=dict(arrowstyle="-|>", color="#43A047", linewidth=1.5, mutation_scale=15))
ax.text(4.3, 1.4, "no more\ntool calls", fontsize=7, color="#43A047", fontstyle="italic")

ax.text(5, -0.05, "QuantumGPT Agent Architecture", fontsize=14, fontweight="bold", ha="center")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "agent_architecture.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "agent_architecture.pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("Saved agent_architecture.png/pdf")


# ═══════════════════════════════════════════════════
# Fig 2: Tool Call Flow per Task (timeline)
# ═══════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(10, 7), gridspec_kw={"hspace": 0.4})

tool_colors = {
    "get_backend_health": "#4285F4",
    "get_qubit_properties": "#34A853",
    "run_circuit": "#EA4335",
    "list_benchmarks": "#FBBC05",
    "diagnose_and_suggest": "#9C27B0",
}

for ax_i, (ax, r) in enumerate(zip(axes, results)):
    calls = r["tool_calls"]
    task = r["task"]
    y = 0.5
    for i, c in enumerate(calls):
        color = tool_colors.get(c["tool"], "#999")
        ax.barh(y, 1, left=i, height=0.6, color=color, edgecolor="white", linewidth=1)
        # Tool name (short)
        short = c["tool"].replace("get_backend_", "").replace("get_qubit_", "qubit_").replace("_and_suggest", "")
        ax.text(i + 0.5, y, short, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    ax.set_xlim(-0.3, max(len(calls), 5) + 0.3)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Step", fontsize=9)
    ax.set_title(f"Task {ax_i+1}: {task}", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

# Legend
legend_patches = [mpatches.Patch(color=v, label=k.replace("_", " ")) for k, v in tool_colors.items()]
fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=8, frameon=False)

fig.suptitle("Agent Tool Call Sequences", fontsize=14, fontweight="bold", y=1.01)
fig.savefig(os.path.join(OUT, "tool_call_flow.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "tool_call_flow.pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("Saved tool_call_flow.png/pdf")


# ═══════════════════════════════════════════════════
# Fig 3: Fidelity comparison
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4))

# Extract fidelities from results
fidelities = []
labels = []
colors_bar = []
for r in results:
    for c in r["tool_calls"]:
        if c["tool"] == "run_circuit":
            # Need to re-extract fidelity from the answer
            pass

# Parse from final answers directly
import re
for r in results:
    answer = r["final_answer"]
    # Find all "Fidelity: X.XXXX" patterns
    fidels = re.findall(r"Fidelity:\s*([\d.]+)", answer)
    circs = re.findall(r"Circuit:\s*(\S+)", answer)
    for circ, fid in zip(circs, fidels):
        labels.append(f"{circ}\n({r['backend'][:20]})")
        fidelities.append(float(fid))

bar_colors = ["#4285F4", "#34A853", "#EA4335", "#FBBC05"]
bars = ax.bar(range(len(fidelities)), fidelities,
              color=bar_colors[:len(fidelities)], edgecolor="white", linewidth=1.5, width=0.6)

for bar, fid in zip(bars, fidelities):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{fid:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Fidelity", fontsize=11)
ax.set_ylim(0, 1.1)
ax.axhline(y=0.9, color="#999", linestyle="--", linewidth=0.8, label="0.9 threshold")
ax.axhline(y=0.7, color="#E91E63", linestyle="--", linewidth=0.8, label="0.7 critical")
ax.legend(fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Circuit Fidelities Across Agent Tasks", fontsize=13, fontweight="bold")

fig.savefig(os.path.join(OUT, "fidelity_comparison.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "fidelity_comparison.pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fidelity_comparison.png/pdf")


# ═══════════════════════════════════════════════════
# Fig 4: Healthy vs Degraded backend comparison
# ═══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Parse health data from results
import re

# Task 1 health
t1_answer = results[0]["final_answer"]
# Task 2 health
t2_answer = results[1]["final_answer"]

metrics = ["T1 (μs)", "T2 (μs)", "2Q Error (%)", "Readout Err (%)"]

# Parse values
def extract_val(text, pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else 0

healthy_vals = [
    extract_val(t1_answer, r"T1:\s*([\d.]+)"),
    extract_val(t1_answer, r"T2:\s*([\d.]+)"),
    extract_val(t1_answer, r"2Q error:\s*([\d.]+)") * 100,
    extract_val(t1_answer, r"Readout error:\s*([\d.]+)") * 100,
]
degraded_vals = [
    extract_val(t2_answer, r"T1:\s*([\d.]+)"),
    extract_val(t2_answer, r"T2:\s*([\d.]+)"),
    extract_val(t2_answer, r"2Q error:\s*([\d.]+)") * 100,
    extract_val(t2_answer, r"Readout error:\s*([\d.]+)") * 100,
]

# T1/T2 comparison
x = np.arange(2)
width = 0.35
axes[0].bar(x - width/2, [healthy_vals[0], healthy_vals[1]], width, label="Healthy", color="#4285F4")
axes[0].bar(x + width/2, [degraded_vals[0], degraded_vals[1]], width, label="Degraded", color="#EA4335")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["T1", "T2"], fontsize=11)
axes[0].set_ylabel("μs", fontsize=11)
axes[0].set_title("Coherence Times", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Error comparison
axes[1].bar(x - width/2, [healthy_vals[2], healthy_vals[3]], width, label="Healthy", color="#4285F4")
axes[1].bar(x + width/2, [degraded_vals[2], degraded_vals[3]], width, label="Degraded", color="#EA4335")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["2Q Gate", "Readout"], fontsize=11)
axes[1].set_ylabel("%", fontsize=11)
axes[1].set_title("Error Rates", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

fig.suptitle("Healthy vs Degraded Backend (Agent-Collected Data)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "healthy_vs_degraded.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "healthy_vs_degraded.pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("Saved healthy_vs_degraded.png/pdf")

print("\nAll Day 6 visualizations generated!")
