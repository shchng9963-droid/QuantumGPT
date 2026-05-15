"""Day 5: MQTBench integration — 5 gate-level benchmarks × 3 backends.

Picks 5 representative circuits from MQTBench at NATIVEGATES level
(IBM Eagle gate set: ECR, RZ, SX, X), runs them on all 3 shadow backends,
and generates comparison visualizations.
"""
import sys, os, time, json
sys.path.insert(0, "/home/wangshuchang/quantumgpt")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mqt.bench import get_benchmark, BenchmarkLevel
from mqt.bench.targets import get_target_for_gateset

from backends.fake_adapter import FakeBackendAdapter
from backends.replay_backend import ReplayBackend
from backends.synthetic_drift import SyntheticDriftBackend, STABLE, LINEAR_DECAY

OUT = "/home/wangshuchang/quantumgpt/demos/day5"
os.makedirs(OUT, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.linewidth": 0.8, "axes.grid": True,
    "grid.alpha": 0.2, "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# ═══════ 1. Generate MQTBench circuits ═══════
print("=" * 60)
print("Generating MQTBench circuits (IBM Eagle native gates)...")
print("=" * 60)

target5 = get_target_for_gateset("ibm_eagle", num_qubits=5)
target4 = get_target_for_gateset("ibm_eagle", num_qubits=4)

CIRCUITS = {
    "GHZ-5":       ("ghz",          5, target5, "Entanglement benchmark"),
    "DJ-5":        ("dj",           5, target5, "Deutsch-Jozsa oracle"),
    "GraphState-5":("graphstate",   5, target5, "Graph state preparation"),
    "QFTent-5":    ("qftentangled", 5, target5, "QFT + entanglement"),
    "VQE_SU2-4":   ("vqe_su2",      4, target4, "Variational SU(2) ansatz"),
}

circuits = {}
circuit_info = {}
for label, (name, size, target, desc) in CIRCUITS.items():
    circ = get_benchmark(name, level=BenchmarkLevel.NATIVEGATES,
                         circuit_size=size, target=target)
    circuits[label] = circ
    ops = dict(circ.count_ops())
    ecr = ops.get("ecr", 0)
    total_gates = sum(v for k, v in ops.items() if k not in ("measure", "barrier"))
    info = {
        "mqt_name": name, "qubits": circ.num_qubits,
        "depth": circ.depth(), "ecr_count": ecr,
        "total_gates": total_gates, "description": desc,
        "ops": {k: v for k, v in ops.items() if k != "barrier"},
    }
    circuit_info[label] = info
    print(f"  {label:16s}  q={info['qubits']}  depth={info['depth']:>3}  "
          f"ECR={ecr:>2}  gates={total_gates:>3}  ({desc})")

# ═══════ 2. Set up backends ═══════
print("\nSetting up 3 shadow backends...")
backends = {
    "FakeAdapter": FakeBackendAdapter("FakeBrisbane"),
    "Replay(t=0)": ReplayBackend.from_synthetic_history(
        "FakeBrisbane", 48, 2, seed=42),
    "SynthDrift(stable)": SyntheticDriftBackend("FakeBrisbane", STABLE),
}
backends["Replay(t=0)"].set_time(0)
backends["SynthDrift(stable)"].set_time(0)

# ═══════ 3. Run cross-validation ═══════
shots = 8192
results = {}  # (circuit_label, backend_name) -> dict

print(f"\nRunning {len(circuits)} circuits x {len(backends)} backends "
      f"({shots} shots each)...")
print(f"{'Circuit':<16} {'Backend':<22} {'Fidelity':>9} {'TrDepth':>8} {'Time':>6}")
print("-" * 65)

for clabel, circ in circuits.items():
    for bname, backend in backends.items():
        t0 = time.time()
        result = backend.run(circ, shots=shots)
        elapsed = time.time() - t0
        fid = result.fidelity
        tdepth = result.metadata.get("transpiled_depth", "?")
        fid_str = f"{fid:.4f}" if fid is not None else "N/A"
        print(f"{clabel:<16} {bname:<22} {fid_str:>9} {tdepth:>8} {elapsed:>5.1f}s")
        results[(clabel, bname)] = {
            "fidelity": fid,
            "transpiled_depth": tdepth,
            "elapsed": elapsed,
            "counts_top3": dict(sorted(result.counts.items(),
                                       key=lambda x: -x[1])[:3]),
        }
    print()

# Save raw results
with open(os.path.join(OUT, "benchmark_results.json"), "w") as f:
    json.dump({
        "circuits": circuit_info,
        "results": {f"{k[0]}|{k[1]}": v for k, v in results.items()},
        "shots": shots,
    }, f, indent=2, default=str)
print(f"Raw results -> {OUT}/benchmark_results.json")

# ═══════ 4. Figures ═══════
clabels = list(circuits.keys())
bnames = list(backends.keys())
colors_bar = ["#2166ac", "#b2182b", "#1b7837"]

# --- Fig 1: Fidelity grouped bar chart ---
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(clabels))
width = 0.24

for i, bk in enumerate(bnames):
    fids = [results[(c, bk)]["fidelity"] or 0 for c in clabels]
    bars = ax.bar(x + i * width, fids, width, label=bk,
                  color=colors_bar[i], alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, f in zip(bars, fids):
        if f > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{f:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xlabel("MQTBench Circuit")
ax.set_ylabel("Fidelity")
ax.set_title("MQTBench × 3 Shadow Backends — Baseline Fidelity (IBM Eagle native gates)",
             fontsize=11, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(clabels, fontsize=9)
ax.set_ylim(0, 1.12)
ax.legend(loc="upper right", fontsize=9)
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT, f"mqtbench_fidelity.{ext}"),
                bbox_inches="tight", dpi=150)
print("  -> mqtbench_fidelity.pdf/png")
plt.close()

# --- Fig 2: Circuit complexity (depth + ECR count) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

depths = [circuit_info[c]["depth"] for c in clabels]
ecrs = [circuit_info[c]["ecr_count"] for c in clabels]
totals = [circuit_info[c]["total_gates"] for c in clabels]

# Depth
bars = axes[0].barh(range(len(clabels)), depths, color="#2166ac", alpha=0.8)
axes[0].set_yticks(range(len(clabels)))
axes[0].set_yticklabels(clabels, fontsize=9)
axes[0].set_xlabel("Circuit Depth (native gates)")
axes[0].set_title("Pre-Transpile Depth", fontsize=11, fontweight="bold")
for i, d in enumerate(depths):
    axes[0].text(d + 0.5, i, str(d), va="center", fontsize=8)

# Gate composition
y = range(len(clabels))
non_ecr = [t - e for t, e in zip(totals, ecrs)]
axes[1].barh(y, non_ecr, color="#1b7837", alpha=0.8, label="1Q gates")
axes[1].barh(y, ecrs, left=non_ecr, color="#b2182b", alpha=0.8, label="ECR (2Q)")
axes[1].set_yticks(range(len(clabels)))
axes[1].set_yticklabels(clabels, fontsize=9)
axes[1].set_xlabel("Gate Count")
axes[1].set_title("Gate Composition (IBM Eagle)", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=8)
for i, (ne, e) in enumerate(zip(non_ecr, ecrs)):
    axes[1].text(ne + e + 1, i, f"{ne+e}", va="center", fontsize=8)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT, f"mqtbench_complexity.{ext}"),
                bbox_inches="tight", dpi=150)
print("  -> mqtbench_complexity.pdf/png")
plt.close()

# --- Fig 3: Fidelity vs circuit complexity scatter ---
fig, ax = plt.subplots(figsize=(8, 5))

for i, bk in enumerate(bnames):
    fids = [results[(c, bk)]["fidelity"] or 0 for c in clabels]
    ax.scatter(ecrs, fids, s=80, color=colors_bar[i], label=bk,
               alpha=0.85, edgecolors="white", linewidth=0.5, zorder=3)

# Connect same circuit across backends
for j, c in enumerate(clabels):
    fids_c = [results[(c, bk)]["fidelity"] or 0 for bk in bnames]
    ax.plot([ecrs[j]] * len(bnames), fids_c, color="gray",
            linewidth=0.5, alpha=0.3, zorder=1)
    ax.annotate(c, (ecrs[j], max(fids_c) + 0.015), fontsize=7,
                ha="center", color="gray")

ax.set_xlabel("ECR (2-Qubit) Gate Count")
ax.set_ylabel("Fidelity")
ax.set_title("Fidelity vs 2Q Gate Count", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0.5, 1.08)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT, f"fidelity_vs_ecr.{ext}"),
                bbox_inches="tight", dpi=150)
print("  -> fidelity_vs_ecr.pdf/png")
plt.close()

# --- Fig 4: Transpiled depth comparison ---
fig, ax = plt.subplots(figsize=(11, 5))

for i, bk in enumerate(bnames):
    tdepths = []
    for c in clabels:
        td = results[(c, bk)]["transpiled_depth"]
        tdepths.append(td if isinstance(td, (int, float)) else 0)
    ax.bar(x + i * width, tdepths, width, label=bk,
           color=colors_bar[i], alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar_x, td in zip(x + i * width, tdepths):
        ax.text(bar_x, td + 1, str(td), ha="center", va="bottom", fontsize=7)

ax.set_xlabel("MQTBench Circuit")
ax.set_ylabel("Transpiled Depth")
ax.set_title("Transpiled Circuit Depth on Each Backend",
             fontsize=11, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(clabels, fontsize=9)
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT, f"transpiled_depth.{ext}"),
                bbox_inches="tight", dpi=150)
print("  -> transpiled_depth.pdf/png")
plt.close()

# --- Fig 5: Fidelity degradation over time (LINEAR_DECAY) for MQTBench circuits ---
print("\nRunning fidelity degradation under LINEAR_DECAY...")
backend_ld = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
time_points = [0, 4, 8, 12, 16, 20, 24]

fig, ax = plt.subplots(figsize=(10, 5))
colors_circ = ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214"]

for ci, (clabel, circ) in enumerate(circuits.items()):
    fids_t = []
    for t in time_points:
        backend_ld.set_time(t)
        r = backend_ld.run(circ, shots=shots)
        fids_t.append(r.fidelity if r.fidelity else 0)
    ax.plot(time_points, fids_t, "o-", color=colors_circ[ci],
            linewidth=1.5, markersize=5, label=clabel, alpha=0.9)

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Fidelity")
ax.set_title("MQTBench Fidelity Under LINEAR_DECAY Drift (24h)",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(os.path.join(OUT, f"mqtbench_linear_decay.{ext}"),
                bbox_inches="tight", dpi=150)
print("  -> mqtbench_linear_decay.pdf/png")
plt.close()

print("\n" + "=" * 60)
print("Day 5 complete!")
print(f"All outputs in {OUT}/")
print("=" * 60)
