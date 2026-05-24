"""Day 7 visualizations — DuckDB query results + fidelity trends."""
import sys, os, json
sys.path.insert(0, "/home/wangshuchang/quantumgpt")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.store import DataStore

OUT = "/home/wangshuchang/quantumgpt/demos/day7"
DB_PATH = os.path.join(OUT, "quantumgpt.duckdb")
db = DataStore(DB_PATH)

# ═══════════════════════════════════════════════════
# Fig 1: Fidelity by circuit + backend
# ═══════════════════════════════════════════════════
runs = db.query("""
    SELECT circuit_name, fidelity, transpiled_depth, backend
    FROM circuit_runs ORDER BY ts
""")

fig, ax = plt.subplots(figsize=(10, 5))
labels = [f"{r['circuit_name']}\n({r['backend'][:15]}...)" for r in runs]
fids = [r['fidelity'] for r in runs]
depths = [r['transpiled_depth'] for r in runs]
colors = ['#4285F4' if 'Synthetic' not in r['backend'] else '#EA4335' for r in runs]

bars = ax.bar(range(len(fids)), fids, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
for bar, fid, depth in zip(bars, fids, depths):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{fid:.3f}\n(d={depth})", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7, rotation=0)
ax.set_ylabel("Fidelity", fontsize=11)
ax.set_ylim(0, 1.15)
ax.axhline(y=0.9, color="#999", linestyle="--", linewidth=0.8, label="0.9 threshold")
ax.axhline(y=0.7, color="#E91E63", linestyle="--", linewidth=0.8, label="0.7 critical")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(color='#4285F4', label='Healthy backend'),
    mpatches.Patch(color='#EA4335', label='Degraded backend'),
    plt.Line2D([0],[0], color='#999', linestyle='--', label='0.9 threshold'),
    plt.Line2D([0],[0], color='#E91E63', linestyle='--', label='0.7 critical'),
], fontsize=8, loc='lower right')

ax.set_title("Circuit Fidelities from DuckDB (5 Agent Runs)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fidelity_by_circuit.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "fidelity_by_circuit.pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fidelity_by_circuit")

# ═══════════════════════════════════════════════════
# Fig 2: Health metrics over time (healthy vs degraded)
# ═══════════════════════════════════════════════════
health = db.query("SELECT * FROM health_snapshots ORDER BY ts")

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

metrics = [
    ("avg_t1_us", "Avg T1 (μs)", axes[0,0]),
    ("avg_t2_us", "Avg T2 (μs)", axes[0,1]),
    ("avg_2q_error", "Avg 2Q Error", axes[1,0]),
    ("drift_score", "Drift Score", axes[1,1]),
]

x = range(len(health))
backends = [h['backend'][:20] for h in health]

for metric_key, label, ax in metrics:
    vals = [h[metric_key] if h[metric_key] is not None else 0 for h in health]
    colors_m = ['#4285F4' if 'Synthetic' not in h['backend'] else '#EA4335' for h in health]
    ax.bar(x, vals, color=colors_m, edgecolor='white', width=0.6)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Snap {i+1}" for i in x], fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(label, fontsize=10, fontweight="bold")

fig.suptitle("Backend Health Snapshots from DuckDB", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "health_over_time.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "health_over_time.pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("Saved health_over_time")

# ═══════════════════════════════════════════════════
# Fig 3: Agent session stats
# ═══════════════════════════════════════════════════
sessions = db.query("""
    SELECT substr(user_prompt, 1, 30) as prompt,
           num_tool_calls, total_tokens, elapsed_seconds
    FROM agent_sessions ORDER BY ts_start
""")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

prompts = [s['prompt']+'...' for s in sessions]
x = range(len(sessions))

# Tool calls
axes[0].barh(x, [s['num_tool_calls'] for s in sessions], color='#4285F4')
axes[0].set_yticks(list(x))
axes[0].set_yticklabels(prompts, fontsize=7)
axes[0].set_xlabel("Tool Calls")
axes[0].set_title("Tool Calls per Session", fontweight="bold")

# Tokens
axes[1].barh(x, [s['total_tokens'] for s in sessions], color='#34A853')
axes[1].set_yticks(list(x))
axes[1].set_yticklabels(prompts, fontsize=7)
axes[1].set_xlabel("Tokens")
axes[1].set_title("Token Usage per Session", fontweight="bold")

# Time
axes[2].barh(x, [s['elapsed_seconds'] for s in sessions], color='#FBBC05')
axes[2].set_yticks(list(x))
axes[2].set_yticklabels(prompts, fontsize=7)
axes[2].set_xlabel("Seconds")
axes[2].set_title("Wall Time per Session", fontweight="bold")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Agent Session Statistics from DuckDB", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "session_stats.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "session_stats.pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("Saved session_stats")

# ═══════════════════════════════════════════════════
# Fig 4: Drift events severity timeline
# ═══════════════════════════════════════════════════
drift_events = db.query("SELECT * FROM drift_events ORDER BY ts")

fig, ax = plt.subplots(figsize=(8, 4))
sev_colors = {"nominal": "#34A853", "warning": "#FBBC05", "critical": "#EA4335"}
x = range(len(drift_events))
colors_d = [sev_colors.get(d['severity'], '#999') for d in drift_events]
drift_vals = [d['drift_score'] if d['drift_score'] else 0 for d in drift_events]

bars = ax.bar(x, drift_vals, color=colors_d, edgecolor='white', width=0.5)
for bar, d in zip(bars, drift_events):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            d['severity'], ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(list(x))
ax.set_xticklabels([d['backend'][:25] for d in drift_events], fontsize=8)
ax.set_ylabel("Drift Score")
ax.set_ylim(0, 1.3)
ax.axhline(y=0.3, color="#FBBC05", linestyle="--", linewidth=0.8, label="Warning (0.3)")
ax.axhline(y=0.6, color="#EA4335", linestyle="--", linewidth=0.8, label="Critical (0.6)")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Drift Events from Agent Diagnosis (DuckDB)", fontsize=13, fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "drift_events.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "drift_events.pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("Saved drift_events")

# ═══════════════════════════════════════════════════
# Fig 5: Data pipeline architecture
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis("off")
fig.patch.set_facecolor("white")

box = dict(boxstyle="round,pad=0.4", facecolor="#E8F0FE", edgecolor="#4285F4", linewidth=1.5)
tool_box = dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#FB8C00", linewidth=1.5)
db_box = dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#43A047", linewidth=1.5)
wb_box = dict(boxstyle="round,pad=0.4", facecolor="#F3E5F5", edgecolor="#9C27B0", linewidth=1.5)

ax.text(1.5, 5, "User\nPrompt", fontsize=10, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FCE4EC", edgecolor="#E91E63", linewidth=1.5))
ax.text(4, 5, "QuantumGPT\nAgent Loop", fontsize=10, fontweight="bold", ha="center", va="center", bbox=box)
ax.text(7, 5, "DeepSeek\nV4-Flash", fontsize=10, ha="center", va="center", bbox=box)

ax.text(4, 3, "Instrumented\nToolExecutor", fontsize=10, fontweight="bold", ha="center", va="center", bbox=tool_box)
ax.text(7, 3, "Shadow\nBackend", fontsize=10, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0F7FA", edgecolor="#00ACC1", linewidth=1.5))

ax.text(2, 1, "DuckDB\n(5 tables, 65 rows)", fontsize=10, fontweight="bold", ha="center", va="center", bbox=db_box)
ax.text(6, 1, "W&B\n(offline run)", fontsize=10, fontweight="bold", ha="center", va="center", bbox=wb_box)
ax.text(9.5, 1, "Query API\n(SQL helpers)", fontsize=9, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F9A825", linewidth=1.5))

arrow = dict(arrowstyle="-|>", color="#333", linewidth=1.5, mutation_scale=12)
ax.annotate("", xy=(3, 5), xytext=(2.3, 5), arrowprops=arrow)
ax.annotate("", xy=(6, 5), xytext=(5.2, 5), arrowprops=arrow)
ax.annotate("", xy=(4, 4.5), xytext=(4, 3.5), arrowprops=arrow)
ax.annotate("", xy=(6, 3), xytext=(5.2, 3), arrowprops=arrow)
ax.annotate("", xy=(2, 2.5), xytext=(3.2, 2.7),
            arrowprops=dict(arrowstyle="-|>", color="#43A047", linewidth=1.5, mutation_scale=12))
ax.annotate("", xy=(6, 2.5), xytext=(4.8, 2.7),
            arrowprops=dict(arrowstyle="-|>", color="#9C27B0", linewidth=1.5, mutation_scale=12))
ax.annotate("", xy=(8.5, 1), xytext=(3.5, 1),
            arrowprops=dict(arrowstyle="-|>", color="#F9A825", linewidth=1.2, mutation_scale=10))

ax.text(2.5, 2.7, "auto-persist", fontsize=7, color="#43A047", fontstyle="italic")
ax.text(5.3, 2.7, "auto-log", fontsize=7, color="#9C27B0", fontstyle="italic")
ax.text(5.5, 0.6, "SQL queries", fontsize=7, color="#F9A825", fontstyle="italic")

ax.text(5.5, -0.1, "Day 7: Data Pipeline Architecture", fontsize=14, fontweight="bold", ha="center")

fig.savefig(os.path.join(OUT, "data_pipeline.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(OUT, "data_pipeline.pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("Saved data_pipeline")

db.close()
print("\nAll Day 7 visualizations generated!")
