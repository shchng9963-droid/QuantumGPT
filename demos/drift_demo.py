"""Drift-aware agent demo — shows the DriftMonitor + replanning chain:

  stable device  →  sudden degradation  →  agent detects drift
  →  replan  →  higher opt level / more shots  →  improved fidelity

Usage:
    python demos/drift_demo.py
    # or from showcase runner:
    from demos.drift_demo import run_drift_demo
    run_drift_demo(save_dir='showcase/drift_demo_output')
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.synthetic_drift import (
    SyntheticDriftBackend,
    SUDDEN_DEGRADATION,
)
from agent.react import ReActAgent


def run_drift_demo(save_dir: str = "demos/output/drift_v21") -> dict:
    """Run the drift-aware agent demo and return a summary dict."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Use SUDDEN_DEGRADATION: clean until t=5h, then catastrophic
    backend = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)

    prompt = "Run GHZ-5 circuit. If fidelity degrades, diagnose and retry with higher optimization."

    # ── Phase 1: clean device (t=0) ──────────────────────────────────
    print("\n┌─ Phase 1: Clean device (t=0h) ─────────────────────────────┐")
    backend.set_time(0)
    agent = ReActAgent(backend, provider="mock", target_fidelity=0.90, verbose=True)
    trace1 = agent.run(prompt)
    fid1 = trace1.best_fidelity
    print(f"│  Best fidelity: {fid1:.4f}")
    print("└────────────────────────────────────────────────────────────┘\n")

    # ── Phase 2: post-degradation (t=6h) ─────────────────────────────
    print("┌─ Phase 2: After sudden degradation (t=6h) ─────────────────┐")
    backend.set_time(6)
    agent2 = ReActAgent(backend, provider="mock", target_fidelity=0.90, verbose=True)
    trace2 = agent2.run(prompt)
    fid2 = trace2.best_fidelity
    print(f"│  Best fidelity: {fid2:.4f}")
    print("└────────────────────────────────────────────────────────────┘\n")

    # ── Phase 3: memory-informed recovery (agent remembers t=6h was bad) ──
    print("┌─ Phase 3: Memory-informed recovery (t=6h, with memory) ────┐")
    memory_ctx = (
        f"Prior run of GHZ-5 on FakeBrisbane at t=6h yielded fidelity={fid2:.4f}. "
        "Device drift detected: T1 dropped significantly."
    )
    agent3 = ReActAgent(backend, provider="mock", target_fidelity=0.90, verbose=True)
    trace3 = agent3.run(prompt, memory_context=memory_ctx)
    fid3 = trace3.best_fidelity
    print(f"│  Best fidelity: {fid3:.4f}")
    print("└────────────────────────────────────────────────────────────┘\n")

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "phase1_clean_fidelity": round(fid1, 4),
        "phase2_degraded_fidelity": round(fid2, 4),
        "phase3_memory_recovery_fidelity": round(fid3, 4),
        "drift_impact": round(fid1 - fid2, 4),
        "memory_recovery_gain": round(fid3 - fid2, 4),
    }

    summary_path = out / "drift_demo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save traces
    for i, tr in enumerate([trace1, trace2, trace3], 1):
        trace_path = out / f"drift_trace_phase{i}.json"
        with open(trace_path, "w") as f:
            json.dump(tr.to_dict(include_observations=True), f, indent=2, default=str)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Drift Demo Summary")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Phase 1 (clean):        fidelity = {fid1:.4f}")
    print(f"  Phase 2 (degraded):     fidelity = {fid2:.4f}  (Δ vs clean = {fid2-fid1:+.4f})")
    print(f"  Phase 3 (memory+diag):  fidelity = {fid3:.4f}  (Δ vs degraded = {fid3-fid2:+.4f})")
    print(f"\n  Outputs: {out}/")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return summary


if __name__ == "__main__":
    run_drift_demo()
