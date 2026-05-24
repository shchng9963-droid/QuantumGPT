"""Day 13 Demo — ExperimentRecord end-to-end: write, query, visualize.

Runs multiple circuits via InstrumentedExecutor (auto-creates ExperimentRecords),
then queries them and generates analysis figures.

Generates:
  1. experiment_fidelity_scatter.png — fidelity by circuit, colored by outcome
  2. experiment_stats_table.png      — rendered stats table
"""

import json
import os
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "demos/day13"

# Circuits to benchmark
CIRCUITS = ["ghz_5", "qft_4", "bv_5", "vqe_4", "qaoa_4"]
BACKENDS = ["FakeBrisbane", "FakeKyiv"]
SHOTS = 2048  # smaller for speed


def run_demo():
    """Run experiments and generate figures."""
    from backends.fake_adapter import FakeBackendAdapter
    from data.store import DataStore
    from data.instrumented import InstrumentedExecutor
    from data.experiment_record import ExperimentRecord

    # Use temp DB so demo is reproducible
    db_path = os.path.join(OUT_DIR, "demo.duckdb")
    if os.path.exists(db_path):
        os.remove(db_path)

    db = DataStore(db_path)

    print("=" * 60)
    print("Day 13: ExperimentRecord End-to-End Demo")
    print("=" * 60)

    # Phase 1: Run experiments (auto-creates ExperimentRecords)
    print("\n--- Phase 1: Running experiments ---")
    for backend_name in BACKENDS:
        be = FakeBackendAdapter(backend_name)
        executor = InstrumentedExecutor(be, db=db)

        # Get health first (caches for backend_snapshot)
        executor.execute("get_backend_health", {})

        for circ in CIRCUITS:
            print(f"  {circ} on {backend_name}...", end=" ", flush=True)
            t0 = time.time()
            result_str = executor.execute("run_circuit", {
                "circuit_name": circ, "shots": SHOTS
            })
            result = json.loads(result_str)
            elapsed = time.time() - t0
            fid = result.get("fidelity", "?")
            print(f"fidelity={fid} ({elapsed:.1f}s)")

    # Also run a diagnose to generate a diagnosis record
    be = FakeBackendAdapter("FakeBrisbane")
    executor = InstrumentedExecutor(be, db=db)
    executor.execute("get_backend_health", {})
    executor.execute("diagnose_and_suggest", {})
    print("  + diagnosis record")

    # Phase 2: Query and display
    print("\n--- Phase 2: Querying ExperimentRecords ---")
    exp_store = db.experiments

    total = exp_store.count()
    print(f"  Total records: {total}")

    stats = exp_store.stats()
    print(f"  Avg fidelity: {stats.get('avg_fidelity', 0):.4f}")
    print(f"  Successes: {stats.get('successes', 0)}/{stats.get('total_experiments', 0)}")

    by_circuit = exp_store.stats_by_circuit()
    print("\n  Per-circuit stats:")
    for row in by_circuit:
        print(f"    {row['circuit_name']}: avg={row['avg_fidelity']:.4f}, "
              f"runs={row['runs']}, success={row['successes']}")

    # Search examples
    high_fid = exp_store.search(min_fidelity=0.95)
    print(f"\n  High-fidelity runs (>=0.95): {len(high_fid)}")

    brisbane_runs = exp_store.by_backend("FakeBrisbane")
    print(f"  FakeBrisbane runs: {len(brisbane_runs)}")

    # Get by ID roundtrip
    recent = exp_store.recent(limit=1)
    if recent:
        r = recent[0]
        fetched = exp_store.get(r.id)
        assert fetched is not None
        print(f"\n  Roundtrip test: ID={r.id[:10]}.. backend={fetched.backend} OK")

    # Phase 3: Generate figures
    print("\n--- Phase 3: Generating figures ---")
    all_recs = exp_store.search(
        experiment_type="circuit_benchmark", limit=100)

    _plot_fidelity_scatter(all_recs)
    _plot_stats_summary(by_circuit, stats)

    db.close()
    print(f"\nDone! Records in {db_path}, figures in {OUT_DIR}/")


def _plot_fidelity_scatter(records):
    """Scatter plot: fidelity per circuit, grouped by backend."""
    fig, ax = plt.subplots(figsize=(10, 5))

    backends = sorted(set(r.backend for r in records))
    circuits = sorted(set(r.circuit_name for r in records if r.circuit_name))
    colors = {"FakeBrisbane": "#3498db", "FakeKyiv": "#9b59b6",
              "FakeSherbrooke": "#1abc9c", "FakeTorino": "#e67e22"}

    width = 0.35
    x = np.arange(len(circuits))

    for i, be in enumerate(backends):
        fids = []
        for circ in circuits:
            matching = [r.fidelity for r in records
                       if r.circuit_name == circ and r.backend == be
                       and r.fidelity is not None]
            fids.append(matching[0] if matching else 0)

        offset = (i - len(backends)/2 + 0.5) * width
        bars = ax.bar(x + offset, fids, width * 0.9,
                      label=be, color=colors.get(be, "#666"),
                      edgecolor="white", linewidth=0.5)

        for bar, fid in zip(bars, fids):
            if fid > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{fid:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(circuits, fontsize=10)
    ax.set_ylabel("Fidelity", fontsize=12)
    ax.set_title("ExperimentRecord: Fidelity by Circuit & Backend",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.axhline(y=0.9, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = f"{OUT_DIR}/experiment_fidelity_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_stats_summary(by_circuit, overall_stats):
    """Rendered summary stats as a figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-circuit bar chart of avg fidelity
    names = [r["circuit_name"] for r in by_circuit]
    avgs = [r["avg_fidelity"] for r in by_circuit]
    mins = [r["min_fidelity"] for r in by_circuit]
    maxs = [r["max_fidelity"] for r in by_circuit]

    x = np.arange(len(names))
    ax1.bar(x, avgs, color="#2ecc71", alpha=0.8, label="Avg", edgecolor="white")
    ax1.scatter(x, mins, color="#e74c3c", zorder=5, s=30, label="Min")
    ax1.scatter(x, maxs, color="#3498db", zorder=5, s=30, label="Max")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9, rotation=15)
    ax1.set_ylabel("Fidelity")
    ax1.set_title("Avg/Min/Max Fidelity by Circuit", fontweight="bold")
    ax1.set_ylim(0, 1.08)
    ax1.axhline(y=0.9, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: text summary
    ax2.axis("off")
    total = overall_stats.get("total_experiments", 0)
    succ = overall_stats.get("successes", 0)
    avg_f = overall_stats.get("avg_fidelity", 0)
    text = (
        f"ExperimentRecord Summary\n"
        f"{'─' * 30}\n"
        f"Total experiments: {total}\n"
        f"Unique backends:   {overall_stats.get('unique_backends', 0)}\n"
        f"Unique circuits:   {overall_stats.get('unique_circuits', 0)}\n"
        f"Success rate:      {succ}/{total} ({100*succ/total:.0f}%)\n"
        f"Avg fidelity:      {avg_f:.4f}\n"
        f"Min fidelity:      {overall_stats.get('min_fidelity', 0):.4f}\n"
        f"Max fidelity:      {overall_stats.get('max_fidelity', 0):.4f}\n"
    )
    ax2.text(0.1, 0.5, text, fontsize=12, fontfamily="monospace",
             verticalalignment="center", transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0"))

    fig.tight_layout()
    path = f"{OUT_DIR}/experiment_stats_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    run_demo()
