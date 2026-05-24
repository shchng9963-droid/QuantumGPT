"""Day 2 可视化: 校准漂移时间线 + ReplayBackend 演示."""
import sys
sys.path.insert(0, "/home/wangshuchang/quantumgpt")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backends.calibration_data import extract_snapshot_from_fake_backend, generate_drift_series
from backends.replay import ReplayBackend

# ── Style: Nature-like ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
})
colors = ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214"]

# ═══════ Figure 1: T1 / T2 drift for 5 qubits over 168 hours ═══════
print("Generating calibration drift data...")
base_snap = extract_snapshot_from_fake_backend("FakeBrisbane")
series = generate_drift_series(base_snap, num_snapshots=168, seed=42)

timestamps_h = [(s.timestamp - series[0].timestamp) / 3600 for s in series]
qubits_show = [0, 5, 10, 15, 20]

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# T1
for i, q in enumerate(qubits_show):
    t1_vals = [s.t1_us[q] for s in series]
    axes[0].plot(timestamps_h, t1_vals, color=colors[i], linewidth=0.9,
                 label=f"Q{q}", alpha=0.85)
axes[0].set_ylabel("T1 (μs)")
axes[0].set_title("Qubit Coherence Drift — FakeBrisbane (168 hours)", fontsize=12, fontweight="bold")
axes[0].legend(ncol=5, fontsize=8, loc="upper right")

# T2
for i, q in enumerate(qubits_show):
    t2_vals = [s.t2_us[q] for s in series]
    axes[1].plot(timestamps_h, t2_vals, color=colors[i], linewidth=0.9,
                 label=f"Q{q}", alpha=0.85)
axes[1].set_ylabel("T2 (μs)")
axes[1].set_xlabel("Time (hours)")
axes[1].legend(ncol=5, fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("/home/wangshuchang/quantumgpt/demos/day2/t1_t2_drift_timeline.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/home/wangshuchang/quantumgpt/demos/day2/t1_t2_drift_timeline.png",
            bbox_inches="tight", dpi=150)
print("  -> t1_t2_drift_timeline.pdf/png saved")

# ═══════ Figure 2: Gate error + readout error drift ═══════
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

for i, q in enumerate(qubits_show):
    ge_vals = [s.gate_error_1q[q] for s in series]
    axes[0].plot(timestamps_h, ge_vals, color=colors[i], linewidth=0.9,
                 label=f"Q{q}", alpha=0.85)
axes[0].set_ylabel("1Q Gate Error")
axes[0].set_title("Gate & Readout Error Drift — FakeBrisbane (168 hours)", fontsize=12, fontweight="bold")
axes[0].legend(ncol=5, fontsize=8, loc="upper right")
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))

for i, q in enumerate(qubits_show):
    re_vals = [s.readout_error[q] for s in series]
    axes[1].plot(timestamps_h, re_vals, color=colors[i], linewidth=0.9,
                 label=f"Q{q}", alpha=0.85)
axes[1].set_ylabel("Readout Error")
axes[1].set_xlabel("Time (hours)")
axes[1].legend(ncol=5, fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("/home/wangshuchang/quantumgpt/demos/day2/gate_readout_drift.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/home/wangshuchang/quantumgpt/demos/day2/gate_readout_drift.png",
            bbox_inches="tight", dpi=150)
print("  -> gate_readout_drift.pdf/png saved")

# ═══════ Figure 3: ReplayBackend health over time ═══════
print("Running ReplayBackend health scan...")
backend = ReplayBackend(series, speed=1e9)

hours_scan = list(range(0, 169, 1))
t1_avg = []
drift_scores = []

for h_idx in hours_scan:
    idx = min(h_idx, len(series) - 1)
    backend.seek(idx)
    health = backend.get_health()
    t1_avg.append(health.avg_t1_us)
    # Compute a simple drift score
    base_t1 = np.mean(series[0].t1_us)
    cur_t1 = health.avg_t1_us
    drift_scores.append(abs(cur_t1 - base_t1) / base_t1)

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(hours_scan, t1_avg, color="#2166ac", linewidth=1.2, label="Avg T1")
ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("Average T1 (μs)", color="#2166ac")
ax1.tick_params(axis='y', labelcolor="#2166ac")

ax2 = ax1.twinx()
ax2.fill_between(hours_scan, drift_scores, alpha=0.15, color="#b2182b")
ax2.plot(hours_scan, drift_scores, color="#b2182b", linewidth=0.8, alpha=0.6, label="Drift Score")
ax2.set_ylabel("Drift Score (relative)", color="#b2182b")
ax2.tick_params(axis='y', labelcolor="#b2182b")

ax1.set_title("ReplayBackend — Average T1 & Drift Score Over Time", fontsize=12, fontweight="bold")
fig.tight_layout()
plt.savefig("/home/wangshuchang/quantumgpt/demos/day2/replay_health_timeline.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/home/wangshuchang/quantumgpt/demos/day2/replay_health_timeline.png",
            bbox_inches="tight", dpi=150)
print("  -> replay_health_timeline.pdf/png saved")

print("\nDay 2 visualizations complete!")
