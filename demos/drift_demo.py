"""Drift end-to-end demo: stable run → inject drift → drift-aware vs naive.

Showcases the QuantumGPT differentiator: when device parameters drift, the
drift-aware agent invalidates stale results, replans, and re-runs the
affected circuit. A naive ReAct agent (drift monitor disabled) just keeps
going on degraded hardware.

Pipeline (mock provider, no API key needed):

  Phase A — Stable run on SyntheticDriftBackend(STABLE) at t=0h.
            Agent runs ghz_5, fidelity ≈ baseline.
  Phase B — Inject SUDDEN_DEGRADATION at t=6h. Same agent, same prompt.
            Drift-aware path detects drift, replans, re-runs ghz_5.
  Phase C — Ablation: drift-naive agent on the same degraded backend.
            No drift detection, no replan.

Outputs (mirrored to demos/output/drift/ and showcase/drift_demo_output/):

  drift_timeline.{png,pdf}    — three-panel figure: T1, gate error, fidelity
  drift_report.md             — human-readable narrative + numbers
  drift_trace_stable.json     — Phase A trace
  drift_trace_aware.json      — Phase B trace (drift-aware)
  drift_trace_naive.json      — Phase C trace (drift-naive)
  drift_summary.json          — machine-readable comparison
  talk_track.md               — ~30s spoken pitch
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from typing import Any, Optional

import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.react import ReActAgent, AgentTrace
from backends.synthetic_drift import (
    SyntheticDriftBackend, STABLE, SUDDEN_DEGRADATION,
)


# ──────────────────────���────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────

PROMPT = (
    "Run the ghz_5 circuit and report the fidelity. "
    "If hardware drift is detected, recover and re-run before answering."
)
DRIFT_TIME_HOURS = 6.0


def _extract_run_circuit_results(trace: AgentTrace) -> list[dict]:
    """Pull every successful run_circuit observation out of a trace."""
    results = []
    for step in trace.steps:
        if step.action != "run_circuit":
            continue
        obs = step.observation
        if not obs:
            continue
        try:
            r = json.loads(obs)
        except Exception:
            continue
        if isinstance(r, dict) and not r.get("error"):
            results.append({
                "step_num": step.step_num,
                "circuit": r.get("circuit"),
                "fidelity": r.get("fidelity"),
                "shots": r.get("shots"),
            })
    return results


def _detected_drift_steps(trace: AgentTrace) -> list[dict]:
    """Find steps whose thought text mentions drift detection."""
    out = []
    for step in trace.steps:
        thought = step.thought or ""
        if "DRIFT" in thought.upper() or "drift" in thought.lower():
            out.append({
                "step_num": step.step_num,
                "thought": thought,
                "action": step.action,
            })
    return out


def _backend_snapshot(backend: SyntheticDriftBackend) -> dict[str, Any]:
    """Capture current calibration snapshot (after drift) for plotting."""
    snap = backend._get_current_snapshot()
    return {
        "backend": backend.name,
        "time_hours": backend._current_time,
        "drift_score": backend._compute_drift_score(),
        "avg_t1_us": float(np.mean(snap.t1_us)),
        "avg_t2_us": float(np.mean(snap.t2_us)),
        "avg_1q_error": float(np.mean(snap.gate_error_1q)),
        "avg_readout_error": float(np.mean(snap.readout_error)),
        "t1_us": [float(x) for x in snap.t1_us],
        "gate_error_1q": [float(x) for x in snap.gate_error_1q],
    }


def _trace_summary(trace: AgentTrace, label: str) -> dict[str, Any]:
    runs = _extract_run_circuit_results(trace)
    return {
        "label": label,
        "tool_calls": trace.num_tool_calls,
        "elapsed_seconds": round(trace.elapsed_seconds, 3),
        "best_fidelity": (
            trace.budget_summary.get("best_fidelity")
            if trace.budget_summary else None
        ),
        "first_fidelity": runs[0]["fidelity"] if runs else None,
        "last_fidelity": runs[-1]["fidelity"] if runs else None,
        "n_circuit_runs": len(runs),
        "drift_alerts": _detected_drift_steps(trace),
        "tools_used": [s.action for s in trace.steps if s.action],
    }


# ───────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────

def _plot_timeline(snap_stable: dict, snap_drift: dict,
                   summary_stable: dict, summary_aware: dict, summary_naive: dict,
                   save_dir: str) -> str:
    """Three-panel figure: T1 per qubit, gate error per qubit, fidelity bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel 1 — T1 per qubit (first 8 qubits)
    n = min(8, len(snap_stable["t1_us"]))
    qubits = np.arange(n)
    width = 0.4
    axes[0].bar(qubits - width/2, snap_stable["t1_us"][:n], width,
                color="#2c7fb8", label=f"Stable (t=0h)")
    axes[0].bar(qubits + width/2, snap_drift["t1_us"][:n], width,
                color="#d95f02", label=f"Drift (t={DRIFT_TIME_HOURS:g}h)")
    axes[0].set_xlabel("Qubit index")
    axes[0].set_ylabel("T1 (μs)")
    axes[0].set_title("T1 collapse after drift")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, linewidth=0.5)
    axes[0].set_xticks(qubits)

    # Panel 2 — 1Q gate error per qubit
    axes[1].bar(qubits - width/2, np.array(snap_stable["gate_error_1q"][:n])*1e3, width,
                color="#2c7fb8", label="Stable")
    axes[1].bar(qubits + width/2, np.array(snap_drift["gate_error_1q"][:n])*1e3, width,
                color="#d95f02", label="Drift")
    axes[1].set_xlabel("Qubit index")
    axes[1].set_ylabel("1Q gate error (×10⁻³)")
    axes[1].set_title("1Q gate error inflation")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, linewidth=0.5)
    axes[1].set_xticks(qubits)

    # Panel 3 — fidelity bars across the three runs
    labels = ["Stable\n(no drift)", "Drift-aware\n(replan)", "Drift-naive\n(no replan)"]
    fids = [
        summary_stable["best_fidelity"] or 0.0,
        summary_aware["best_fidelity"] or 0.0,
        summary_naive["best_fidelity"] or 0.0,
    ]
    colors = ["#2c7fb8", "#1b9e77", "#d95f02"]
    bars = axes[2].bar(labels, fids, color=colors, width=0.55)
    for b, f in zip(bars, fids):
        axes[2].text(b.get_x() + b.get_width()/2, f + 0.01, f"{f:.3f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[2].set_ylabel("Best fidelity (ghz_5)")
    axes[2].set_ylim(0, max(1.0, max(fids) * 1.15))
    axes[2].set_title("Drift-aware vs drift-naive recovery")
    axes[2].grid(True, alpha=0.3, linewidth=0.5, axis="y")

    fig.suptitle(
        "QuantumGPT Drift-Aware Demo — SyntheticDriftBackend(FakeBrisbane), SUDDEN_DEGRADATION",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    for ext in ("pdf", "png"):
        out = os.path.join(save_dir, f"drift_timeline.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return os.path.join(save_dir, "drift_timeline.png")


# ───────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────

def _write_report(snap_stable, snap_drift, sum_stable, sum_aware, sum_naive,
                  save_dir: str) -> str:
    def f(x: Optional[float], digits: int = 4) -> str:
        return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "N/A"

    aware_drop = (sum_stable["best_fidelity"] or 0) - (sum_aware["best_fidelity"] or 0)
    naive_drop = (sum_stable["best_fidelity"] or 0) - (sum_naive["best_fidelity"] or 0)
    recovery_gain = (sum_aware["best_fidelity"] or 0) - (sum_naive["best_fidelity"] or 0)

    lines = [
        "# QuantumGPT Drift-Aware Demo Report",
        "",
        "Goal: show that when synthetic hardware drift is injected mid-experiment,",
        "the drift-aware agent detects the change, invalidates stale results,",
        "and re-runs the affected circuit — while a drift-naive baseline does not.",
        "",
        "## Setup",
        "",
        f"- Backend: SyntheticDriftBackend(FakeBrisbane)",
        f"- Drift profile: SUDDEN_DEGRADATION applied at t={DRIFT_TIME_HOURS:g}h",
        f"- Provider: mock (deterministic rule planner) — no API key required",
        f"- Prompt: \"{PROMPT}\"",
        "",
        "## Backend state",
        "",
        f"| Metric | Stable (t=0h) | Drift (t={DRIFT_TIME_HOURS:g}h) | Δ |",
        "|---|---|---|---|",
        f"| Avg T1 (μs) | {f(snap_stable['avg_t1_us'], 1)} | {f(snap_drift['avg_t1_us'], 1)} | "
        f"{f((snap_drift['avg_t1_us'] or 0) - (snap_stable['avg_t1_us'] or 0), 1)} |",
        f"| Avg 1Q error | {f(snap_stable['avg_1q_error'])} | {f(snap_drift['avg_1q_error'])} | "
        f"{f((snap_drift['avg_1q_error'] or 0) - (snap_stable['avg_1q_error'] or 0))} |",
        f"| Avg readout error | {f(snap_stable['avg_readout_error'])} | {f(snap_drift['avg_readout_error'])} | "
        f"{f((snap_drift['avg_readout_error'] or 0) - (snap_stable['avg_readout_error'] or 0))} |",
        f"| Drift score | {f(snap_stable['drift_score'])} | {f(snap_drift['drift_score'])} | — |",
        "",
        "## Agent runs",
        "",
        "| Run | Best fidelity | Tool calls | Drift alerts | Run_circuit count |",
        "|---|---|---|---|---|",
        f"| A — Stable | {f(sum_stable['best_fidelity'])} | {sum_stable['tool_calls']} | "
        f"{len(sum_stable['drift_alerts'])} | {sum_stable['n_circuit_runs']} |",
        f"| B — Drift + drift-aware | {f(sum_aware['best_fidelity'])} | {sum_aware['tool_calls']} | "
        f"{len(sum_aware['drift_alerts'])} | {sum_aware['n_circuit_runs']} |",
        f"| C — Drift + drift-naive | {f(sum_naive['best_fidelity'])} | {sum_naive['tool_calls']} | "
        f"{len(sum_naive['drift_alerts'])} | {sum_naive['n_circuit_runs']} |",
        "",
        "## Headline numbers",
        "",
        f"- Drift hit on the naive baseline: **Δ fidelity = {f(naive_drop)}** (Phase A → Phase C).",
        f"- Residual hit after drift-aware recovery: **Δ fidelity = {f(aware_drop)}**.",
        f"- Drift-aware recovery gain over the naive baseline: **+{f(recovery_gain)}**.",
        f"- Drift-aware agent issued **{len(sum_aware['drift_alerts'])}** drift alert(s) "
        f"and re-ran the circuit **{sum_aware['n_circuit_runs']}** time(s).",
        "",
        "## Drift-aware trace highlights",
    ]

    if sum_aware["drift_alerts"]:
        for alert in sum_aware["drift_alerts"]:
            lines.append(f"- step {alert['step_num']}: {alert['thought']}")
    else:
        lines.append("- (no drift alert recorded — check drift_threshold)")

    lines += [
        "",
        "## Files",
        "",
        "- `drift_timeline.png` / `drift_timeline.pdf` — three-panel figure",
        "- `drift_trace_stable.json` — Phase A trace",
        "- `drift_trace_aware.json` — Phase B trace (drift-aware)",
        "- `drift_trace_naive.json` — Phase C trace (drift-naive)",
        "- `drift_summary.json` — machine-readable summary",
        "- `talk_track.md` — 30s spoken pitch",
    ]
    path = os.path.join(save_dir, "drift_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def _write_talk_track(sum_stable, sum_aware, sum_naive, save_dir: str) -> str:
    fid_stable = sum_stable["best_fidelity"] or 0.0
    fid_aware = sum_aware["best_fidelity"] or 0.0
    fid_naive = sum_naive["best_fidelity"] or 0.0
    text = (
        "# 30s Demo Talk Track — Drift-Aware Recovery\n\n"
        f"On a stable simulated FakeBrisbane the agent runs ghz_5 and gets "
        f"fidelity ≈ {fid_stable:.3f}. We then inject sudden hardware drift — T1 collapses "
        f"and gate errors spike — at t={DRIFT_TIME_HOURS:g}h.\n\n"
        f"The drift-aware agent picks this up from the calibration snapshot, "
        f"invalidates the stale fidelity result, replans, and re-runs the circuit. "
        f"Final fidelity: {fid_aware:.3f}.\n\n"
        f"For comparison, the same prompt against a drift-naive ReAct agent "
        f"reports {fid_naive:.3f} — it never notices the device changed and keeps the stale answer.\n\n"
        f"The 'recovery gap' (drift-aware − drift-naive = +{fid_aware - fid_naive:.3f}) "
        "is the value-add we sell: continuously valid results on a non-stationary device.\n"
    )
    path = os.path.join(save_dir, "talk_track.md")
    with open(path, "w") as fh:
        fh.write(text)
    return path


# ───────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────

def _make_agent(backend, drift_aware: bool) -> ReActAgent:
    return ReActAgent(
        backend,
        provider="mock",
        verbose=False,
        target_fidelity=0.9,
        max_tool_calls=12,
        use_memory=False,             # keep memory off for a clean demo
        use_drift_aware=drift_aware,
        drift_threshold=0.15,
    )


def run_drift_demo(save_dirs: Optional[list[str]] = None) -> dict:
    if save_dirs is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dirs = [
            os.path.join(root, "demos", "output", "drift"),
            os.path.join(root, "showcase", "drift_demo_output"),
        ]
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)
    primary = save_dirs[0]

    print(f"\n{'#'*70}")
    print(f"# QUANTUMGPT DRIFT DEMO")
    print(f"# Prompt: {PROMPT}")
    print(f"{'#'*70}\n")

    # ── Phase A: stable run ──────────────────────────────
    backend = SyntheticDriftBackend("FakeBrisbane", profile=STABLE)
    backend.set_time(0.0)
    snap_stable = _backend_snapshot(backend)
    print(f"[Phase A] Stable backend at t=0h, drift_score={snap_stable['drift_score']:.3f}")
    agent_a = _make_agent(backend, drift_aware=True)
    trace_a = agent_a.run(PROMPT)
    sum_a = _trace_summary(trace_a, "stable")
    print(f"  → best_fidelity={sum_a['best_fidelity']}, tool_calls={sum_a['tool_calls']}")

    # ── Phase B: inject drift, drift-aware agent ────────
    backend_b = SyntheticDriftBackend("FakeBrisbane", profile=STABLE)
    backend_b.set_time(0.0)
    agent_b = _make_agent(backend_b, drift_aware=True)
    # Initialize agent on the stable snapshot so the monitor has a clean baseline.
    # Then inject drift before run.
    backend_b.profile = SUDDEN_DEGRADATION
    backend_b.set_time(DRIFT_TIME_HOURS)
    snap_drift = _backend_snapshot(backend_b)
    print(f"[Phase B] Drift injected at t={DRIFT_TIME_HOURS:g}h, "
          f"drift_score={snap_drift['drift_score']:.3f}, "
          f"avg_T1={snap_drift['avg_t1_us']:.1f}μs (was {snap_stable['avg_t1_us']:.1f})")
    trace_b = agent_b.run(PROMPT)
    sum_b = _trace_summary(trace_b, "drift_aware")
    print(f"  → best_fidelity={sum_b['best_fidelity']}, tool_calls={sum_b['tool_calls']}, "
          f"drift_alerts={len(sum_b['drift_alerts'])}")

    # ── Phase C: same drift, drift-NAIVE agent ──────────
    backend_c = SyntheticDriftBackend("FakeBrisbane", profile=SUDDEN_DEGRADATION)
    backend_c.set_time(DRIFT_TIME_HOURS)
    agent_c = _make_agent(backend_c, drift_aware=False)
    print(f"[Phase C] Drift-naive baseline on the same degraded backend")
    trace_c = agent_c.run(PROMPT)
    sum_c = _trace_summary(trace_c, "drift_naive")
    print(f"  → best_fidelity={sum_c['best_fidelity']}, tool_calls={sum_c['tool_calls']}")

    # ── Render artifacts (write to primary, then copy) ──
    fig_path = _plot_timeline(snap_stable, snap_drift, sum_a, sum_b, sum_c, primary)
    report_path = _write_report(snap_stable, snap_drift, sum_a, sum_b, sum_c, primary)
    talk_path = _write_talk_track(sum_a, sum_b, sum_c, primary)

    # Persist traces
    trace_paths: dict[str, str] = {}
    for label, tr in (("stable", trace_a), ("aware", trace_b), ("naive", trace_c)):
        p = os.path.join(primary, f"drift_trace_{label}.json")
        with open(p, "w") as fh:
            json.dump(tr.to_dict(include_observations=True), fh, indent=2, default=str)
        trace_paths[label] = p

    summary = {
        "prompt": PROMPT,
        "drift_time_hours": DRIFT_TIME_HOURS,
        "backend_stable": snap_stable,
        "backend_drift": snap_drift,
        "phase_a_stable": sum_a,
        "phase_b_drift_aware": sum_b,
        "phase_c_drift_naive": sum_c,
        "fidelity_drop_naive": (sum_a["best_fidelity"] or 0) - (sum_c["best_fidelity"] or 0),
        "fidelity_drop_after_recovery": (sum_a["best_fidelity"] or 0) - (sum_b["best_fidelity"] or 0),
        "recovery_gain": (sum_b["best_fidelity"] or 0) - (sum_c["best_fidelity"] or 0),
    }
    sum_path = os.path.join(primary, "drift_summary.json")
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Mirror to secondary save dirs
    import shutil
    primary_files = [report_path, talk_path, sum_path] + list(trace_paths.values())
    primary_files += [
        os.path.join(primary, "drift_timeline.png"),
        os.path.join(primary, "drift_timeline.pdf"),
    ]
    for d in save_dirs[1:]:
        for src in primary_files:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(d, os.path.basename(src)))

    print(f"\n{'='*60}")
    print("Demo complete. Artifacts:")
    print(f"  Figure : {fig_path}")
    print(f"  Report : {report_path}")
    print(f"  Talk   : {talk_path}")
    print(f"  Summary: {sum_path}")
    for d in save_dirs[1:]:
        print(f"  Mirror : {d}")
    print(f"{'='*60}")
    return summary


if __name__ == "__main__":
    run_drift_demo()
