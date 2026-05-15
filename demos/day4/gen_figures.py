"""Day 4 可视化: DriftProfile 对比 + ReplayBackend 时间线 + 测试结果."""
import sys
sys.path.insert(0, "/home/wangshuchang/quantumgpt")

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit

from backends.replay_backend import ReplayBackend
from backends.synthetic_drift import (
    SyntheticDriftBackend, DriftProfile,
    STABLE, LINEAR_DECAY, SUDDEN_DEGRADATION, DIURNAL_CYCLE,
)
from backends.properties_stream import PropertiesStream

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ═══════ Figure 1: 4 DriftProfile presets — T1 + drift_score ═══════
print("Generating DriftProfile comparison...")
profiles = {
    "STABLE": STABLE,
    "LINEAR_DECAY": LINEAR_DECAY,
    "SUDDEN_DEGRADATION": SUDDEN_DEGRADATION,
    "DIURNAL_CYCLE": DIURNAL_CYCLE,
}
colors = {"STABLE": "#1b7837", "LINEAR_DECAY": "#2166ac",
          "SUDDEN_DEGRADATION": "#b2182b", "DIURNAL_CYCLE": "#762a83"}

hours = np.linspace(0, 24, 200)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for name, prof in profiles.items():
    t1_factors = [prof.t1_drift(t) for t in hours]
    axes[0].plot(hours, t1_factors, color=colors[name], linewidth=1.5,
                 label=name, alpha=0.9)

axes[0].set_ylabel("T1 Multiplicative Factor")
axes[0].set_title("DriftProfile Presets — Parameter Evolution (24 hours)",
                   fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

for name, prof in profiles.items():
    if prof.drift_score_fn:
        scores = [prof.drift_score_fn(t) for t in hours]
    else:
        scores = [abs(1.0 - prof.t1_drift(t)) for t in hours]
    axes[1].plot(hours, scores, color=colors[name], linewidth=1.5,
                 label=name, alpha=0.9)

axes[1].set_ylabel("Drift Score [0, 1]")
axes[1].set_xlabel("Time (hours)")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("/home/wangshuchang/quantumgpt/demos/day4/drift_profiles_comparison.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/home/wangshuchang/quantumgpt/demos/day4/drift_profiles_comparison.png",
            bbox_inches="tight", dpi=150)
print("  -> drift_profiles_comparison.pdf/png saved")

# ═══════ Figure 2: Fidelity degradation under each profile ═══════
print("Running fidelity under each DriftProfile...")

# GHZ-5 circuit
qc = QuantumCircuit(5)
qc.h(0)
for i in range(4):
    qc.cx(i, i + 1)
qc.measure_all()

time_points = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]

fig, ax = plt.subplots(figsize=(10, 5))

for name, prof in profiles.items():
    backend = SyntheticDriftBackend("FakeBrisbane", prof)
    fidelities = []
    for t in time_points:
        backend.set_time(t)
        result = backend.run(qc, shots=4096)
        fidelities.append(result.fidelity if result.fidelity else 0)
    ax.plot(time_points, fidelities, 'o-', color=colors[name], linewidth=1.5,
            markersize=5, label=name, alpha=0.9)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("GHZ-5 Fidelity")
ax.set_title("Fidelity Degradation Under Different Drift Profiles",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("/home/wangshuchang/quantumgpt/demos/day4/fidelity_degradation.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/home/wangshuchang/quantumgpt/demos/day4/fidelity_degradation.png",
            bbox_inches="tight", dpi=150)
print("  -> fidelity_degradation.pdf/png saved")

# ═══════ Figure 3: replay_all batch stepping ═══════
print("Running replay_all batch stepping...")

replay_bk = ReplayBackend.from_synthetic_history(
    "FakeBrisbane", duration_hours=48, snapshot_interval_hours=1, seed=42
)
snapshots = PropertiesStream.replay_all(replay_bk, hours=48, step_hours=2)

sim_hours = [s["stream_sim_time_hours"] for s in snapshots]
avg_t1 = [s["avg_t1_us"] for s in snapshots]
drift_s = [s["drift_score"] for s in snapshots]
avg_1q = [s["avg_1q_error"] for s in snapshots]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(sim_hours, avg_t1, color="#2166ac", linewidth=1.2)
axes[0].set_ylabel("Avg T1 (μs)")
axes[0].set_title("ReplayBackend — replay_all() Batch Stepping (48h, step=2h)",
                   fontsize=12, fontweight="bold")

axes[1].fill_between(sim_hours, drift_s, alpha=0.2, color="#b2182b")
axes[1].plot(sim_hours, drift_s, color="#b2182b", linewidth=1.0)
axes[1].set_ylabel("Drift Score")

axes[2].plot(sim_hours, avg_1q, color="#1b7837", linewidth=1.0)
axes[2].set_ylabel("Avg 1Q Error")
axes[2].set_xlabel("Simulated Time (hours)")

plt.tight_layout()
plt.savefig("/home/wangshuchang/quantumgpt/demos/day4/replay_all_timeline.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/home/wangshuchang/quantumgpt/demos/day4/replay_all_timeline.png",
            bbox_inches="tight", dpi=150)
print("  -> replay_all_timeline.pdf/png saved")

# ═══════ Figure 4: Profile hot-swap demo ═══════
print("Running profile hot-swap demo...")

backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
time_line = list(range(0, 25))
fid_hotswap = []

for t in time_line:
    if t == 10:
        backend.profile = SUDDEN_DEGRADATION  # hot swap at t=10
    backend.set_time(t)
    result = backend.run(qc, shots=4096)
    fid_hotswap.append(result.fidelity if result.fidelity else 0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_line, fid_hotswap, 'o-', color="#2166ac", linewidth=1.5, markersize=4)
ax.axvline(x=10, color="#b2182b", linestyle='--', linewidth=1.5, alpha=0.7,
           label="Hot-swap: STABLE → SUDDEN_DEGRADATION")
ax.axvline(x=15, color="#b2182b", linestyle=':', linewidth=1.0, alpha=0.5,
           label="SUDDEN_DEGRADATION spike at t=5h (relative)")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("GHZ-5 Fidelity")
ax.set_title("Profile Hot-Swap Demo", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("/home/wangshuchang/quantumgpt/demos/day4/hotswap_demo.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/home/wangshuchang/quantumgpt/demos/day4/hotswap_demo.png",
            bbox_inches="tight", dpi=150)
print("  -> hotswap_demo.pdf/png saved")

print("\nDay 4 visualizations complete!")
