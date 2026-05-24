"""Day 3 可视化: 基准电路 fidelity 交叉验证 + 电路结构图."""
import sys
sys.path.insert(0, "/home/wangshuchang/quantumgpt")

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backends.fake_adapter import FakeBackendAdapter
from backends.replay_backend import ReplayBackend
from backends.synthetic_drift import SyntheticDriftBackend, LINEAR_DECAY, STABLE
from bench.circuits import get_benchmark, list_benchmarks

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

# ═══════ Run cross-validation ═══════
print("Setting up backends...")
backends = {
    "FakeAdapter": FakeBackendAdapter("FakeBrisbane"),
    "Replay(t=0)": ReplayBackend.from_synthetic_history("FakeBrisbane", 48, 2, seed=42),
    "SynthDrift(t=0)": SyntheticDriftBackend("FakeBrisbane", STABLE),
}
backends["Replay(t=0)"].set_time(0)
backends["SynthDrift(t=0)"].set_time(0)

benchmarks = list_benchmarks()
print(f"Benchmarks: {list(benchmarks.keys())}")

shots = 4096
results = {}  # (circuit, backend) -> fidelity

print("\nRunning cross-validation (5 circuits x 3 backends)...")
print(f"{'Circuit':<12} {'Backend':<20} {'Fidelity':>10} {'Depth':>7} {'Time':>6}")
print("=" * 60)

for circ_name in benchmarks:
    circuit, ideal = get_benchmark(circ_name)
    for bk_name, backend in backends.items():
        t0 = time.time()
        result = backend.run(circuit, shots=shots)
        elapsed = time.time() - t0
        fid = result.fidelity
        depth = result.metadata.get("transpiled_depth", "?")
        fid_str = f"{fid:.4f}" if fid is not None else "N/A"
        print(f"{circ_name:<12} {bk_name:<20} {fid_str:>10} {depth:>7} {elapsed:>5.1f}s")
        results[(circ_name, bk_name)] = fid
    print("-" * 60)

# ═══════ Figure 1: Grouped bar chart of fidelities ═══════
circ_names = list(benchmarks.keys())
bk_names = list(backends.keys())
colors_bar = ["#2166ac", "#b2182b", "#1b7837"]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(circ_names))
width = 0.25

for i, bk in enumerate(bk_names):
    fids = [results.get((c, bk), 0) or 0 for c in circ_names]
    bars = ax.bar(x + i * width, fids, width, label=bk, color=colors_bar[i], alpha=0.85)
    for bar, f in zip(bars, fids):
        if f > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{f:.3f}", ha='center', va='bottom', fontsize=7)

ax.set_xlabel("Benchmark Circuit")
ax.set_ylabel("Fidelity")
ax.set_title("Cross-Backend Fidelity Comparison (t=0, 4096 shots)", fontsize=12, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels([benchmarks[c] for c in circ_names], rotation=15, ha="right", fontsize=8)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", fontsize=9)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("/home/wangshuchang/quantumgpt/demos/day3/fidelity_comparison.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/home/wangshuchang/quantumgpt/demos/day3/fidelity_comparison.png",
            bbox_inches="tight", dpi=150)
print("\n  -> fidelity_comparison.pdf/png saved")

# ═══════ Figure 2: Circuit depth and gate count ═══════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

circ_info = {}
for circ_name in circ_names:
    circuit, _ = get_benchmark(circ_name)
    ops = circuit.count_ops()
    n_cx = ops.get("cx", 0)
    n_1q = sum(v for k, v in ops.items() if k not in ("cx", "measure", "barrier"))
    circ_info[circ_name] = {
        "depth": circuit.depth(),
        "cx_count": n_cx,
        "1q_count": n_1q,
        "total_gates": n_cx + n_1q,
        "qubits": circuit.num_qubits,
    }

# Depth
depths = [circ_info[c]["depth"] for c in circ_names]
axes[0].barh(range(len(circ_names)), depths, color="#2166ac", alpha=0.8)
axes[0].set_yticks(range(len(circ_names)))
axes[0].set_yticklabels([benchmarks[c] for c in circ_names], fontsize=8)
axes[0].set_xlabel("Circuit Depth")
axes[0].set_title("Circuit Depth", fontsize=11, fontweight="bold")
for i, d in enumerate(depths):
    axes[0].text(d + 0.3, i, str(d), va='center', fontsize=8)

# Gate count breakdown
cx_counts = [circ_info[c]["cx_count"] for c in circ_names]
q1_counts = [circ_info[c]["1q_count"] for c in circ_names]
y = range(len(circ_names))
axes[1].barh(y, q1_counts, color="#1b7837", alpha=0.8, label="1Q gates")
axes[1].barh(y, cx_counts, left=q1_counts, color="#b2182b", alpha=0.8, label="CX gates")
axes[1].set_yticks(range(len(circ_names)))
axes[1].set_yticklabels([benchmarks[c] for c in circ_names], fontsize=8)
axes[1].set_xlabel("Gate Count")
axes[1].set_title("Gate Composition", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("/home/wangshuchang/quantumgpt/demos/day3/circuit_structure.pdf",
            bbox_inches="tight", dpi=150)
plt.savefig("/home/wangshuchang/quantumgpt/demos/day3/circuit_structure.png",
            bbox_inches="tight", dpi=150)
print("  -> circuit_structure.pdf/png saved")

print("\nDay 3 visualizations complete!")
