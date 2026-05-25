"""End-to-end Tune-up Sequence Demo.

Demonstrates QuantumGPT's full calibration workflow:
  1. Check backend health
  2. Rabi experiment → extract π-pulse amplitude
  3. Ramsey experiment → measure T2* and refine frequency
  4. DRAG calibration → suppress leakage to |2⟩
  5. Randomized Benchmarking → verify gate quality (EPC)
  6. Active Learning → suggest next experiment

This is the Plan v2 Appendix C standard tune-up sequence,
executed by the agent autonomously.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.react import ReActAgent
from agent.reporting import TraceReporter
from backends.fake_adapter import FakeBackendAdapter


# ═══════════════════════════════════════════════════════
# Scripted tune-up (deterministic, no LLM API needed)
# ═══════════════════════════════════════════════════════

@dataclass
class TuneupStep:
    """A single step in the tune-up trace."""
    step_num: int
    name: str
    tool: str
    input_params: dict
    output: dict
    decision: str
    elapsed_s: float


@dataclass
class TuneupResult:
    """Full tune-up result."""
    steps: list[TuneupStep]
    calibration: dict
    total_time_s: float
    success: bool
    summary: str


def run_tuneup_demo(
    qubit: int = 0,
    save_dir: str | None = None,
    verbose: bool = True,
) -> TuneupResult:
    """Run the full tune-up sequence using direct tool calls.

    This demonstrates the ideal agent behavior without requiring an LLM API key.
    The sequence mirrors what the QuantumGPT agent would do autonomously.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "tuneup")
    os.makedirs(save_dir, exist_ok=True)

    from tools.quantum_tools import ToolExecutor
    from backends.fake_adapter import FakeBackendAdapter

    be = FakeBackendAdapter("FakeBrisbane")
    executor = ToolExecutor(be)

    steps: list[TuneupStep] = []
    calibration: dict[str, Any] = {"qubit": qubit}
    t_start = time.time()

    def log(msg: str):
        if verbose:
            print(msg)

    # ─────────────────────────────────────────────────
    # STEP 1: Check backend health
    # ─────────────────────────────────────────────────
    log("\n" + "═" * 70)
    log("STEP 1 · Backend Health Check")
    log("═" * 70)

    t0 = time.time()
    health_raw = executor.execute("get_backend_health", {})
    health = json.loads(health_raw)
    dt = time.time() - t0

    log(f"  Backend: {health.get('backend_name', 'FakeBrisbane')}")
    log(f"  Avg T1: {health.get('avg_t1_us', '?')} μs")
    log(f"  Avg gate error: {health.get('avg_gate_error', '?')}")
    log(f"  Drift score: {health.get('drift_score', 'N/A')}")

    decision = "Backend healthy. Proceed with tune-up."
    steps.append(TuneupStep(1, "Health Check", "get_backend_health", {}, health, decision, dt))

    # ─────────────────────────────────────────────────
    # STEP 2: Rabi experiment → π-pulse amplitude
    # ─────────────────────────────────────────────────
    log("\n" + "═" * 70)
    log("STEP 2 · Rabi Experiment")
    log("═" * 70)

    t0 = time.time()
    rabi_params = {
        "qubit_freq_ghz": 5.0,
        "amp_min": 0.0,
        "amp_max": 0.01,
        "n_points": 30,
        "pulse_duration_ns": 100,
        "pulse_shape": "gaussian",
    }
    rabi_raw = executor.execute("rabi_experiment", rabi_params)
    rabi_data = json.loads(rabi_raw)
    dt = time.time() - t0

    # Fit the Rabi data
    fit_params = {
        "amplitudes": rabi_data["amplitudes"],
        "populations": rabi_data["populations"],
    }
    fit_raw = executor.execute("fit_rabi", fit_params)
    fit_data = json.loads(fit_raw)

    pi_amp = fit_data.get("pi_amplitude_ghz", 0.005)
    calibration["pi_amp_ghz"] = pi_amp
    calibration["pi_amp_mhz"] = pi_amp * 1000

    log(f"  π-pulse amplitude: {pi_amp*1000:.3f} MHz ({pi_amp:.6f} GHz)")
    log(f"  Fit R²: {fit_data.get('r_squared', 'N/A')}")

    decision = f"π-amplitude locked at {pi_amp*1000:.3f} MHz."
    steps.append(TuneupStep(2, "Rabi Experiment", "rabi_experiment + fit_rabi",
                            rabi_params, {"rabi": rabi_data, "fit": fit_data}, decision, dt))

    # ─────────────────────────────────────────────────
    # STEP 3: Ramsey experiment → T2* and frequency
    # ─────────────────────────────────────────────────
    log("\n" + "═" * 70)
    log("STEP 3 · Ramsey Experiment")
    log("═" * 70)

    t0 = time.time()
    ramsey_params = {
        "qubit": qubit,
        "delay_max_ns": 5000,
        "n_points": 50,
        "artificial_detuning_mhz": 2.0,
        "shots": 1024,
    }
    ramsey_raw = executor.execute("ramsey_experiment", ramsey_params)
    ramsey_data = json.loads(ramsey_raw)
    dt = time.time() - t0

    # Fit Ramsey
    ramsey_fit_params = {
        "delays_ns": ramsey_data["delays_ns"],
        "populations": ramsey_data["populations"],
        "artificial_detuning_mhz": 2.0,
    }
    ramsey_fit_raw = executor.execute("fit_ramsey", ramsey_fit_params)
    ramsey_fit = json.loads(ramsey_fit_raw)

    t2_star = ramsey_fit.get("T2_star_us", ramsey_fit.get("T2_us"))
    detuning = ramsey_fit.get("detuning_mhz", ramsey_fit.get("frequency_mhz"))
    calibration["T2_star_us"] = t2_star
    calibration["detuning_mhz"] = detuning

    log(f"  T2* = {t2_star} μs")
    log(f"  Detuning = {detuning} MHz")

    decision = f"T2* = {t2_star} μs. Frequency offset noted."
    steps.append(TuneupStep(3, "Ramsey Experiment", "ramsey_experiment + fit_ramsey",
                            ramsey_params, {"ramsey": ramsey_data, "fit": ramsey_fit}, decision, dt))

    # ─────────────────────────────────────────────────
    # STEP 4: DRAG calibration → suppress |2⟩ leakage
    # ─────────────────────────────────────────────────
    log("\n" + "═" * 70)
    log("STEP 4 · DRAG Calibration")
    log("═" * 70)

    t0 = time.time()
    drag_params = {
        "pi_amp_ghz": pi_amp,
        "anharmonicity_ghz": -0.3,
        "pulse_duration_ns": 100,
        "pulse_shape": "gaussian",
        "alpha_min": -2.0,
        "alpha_max": 2.0,
        "n_points": 21,
        "refine": True,
    }
    drag_raw = executor.execute("drag_calibration", drag_params)
    drag_data = json.loads(drag_raw)
    dt = time.time() - t0

    optimal_alpha = drag_data.get("optimal_alpha", 0.0)
    min_leakage = drag_data.get("min_leakage", 0.0)
    calibration["drag_alpha"] = optimal_alpha
    calibration["leakage"] = min_leakage

    log(f"  Optimal DRAG α = {optimal_alpha:.4f}")
    log(f"  Min leakage = {min_leakage:.2e}")
    log(f"  Gate fidelity = {drag_data.get('gate_fidelity', 'N/A')}")

    decision = f"DRAG α locked at {optimal_alpha:.4f}. Leakage suppressed to {min_leakage:.2e}."
    steps.append(TuneupStep(4, "DRAG Calibration", "drag_calibration",
                            drag_params, drag_data, decision, dt))

    # ─────────────────────────────────────────────────
    # STEP 5: Randomized Benchmarking → EPC
    # ─────────────────────────────────────────────────
    log("\n" + "═" * 70)
    log("STEP 5 · Randomized Benchmarking")
    log("═" * 70)

    t0 = time.time()
    rb_params = {
        "qubit": qubit,
        "sequence_lengths": [1, 2, 4, 8, 16, 32, 64, 128],
        "n_sequences": 20,
        "shots": 1024,
        "seed": 42,
    }
    rb_raw = executor.execute("randomized_benchmarking", rb_params)
    rb_data = json.loads(rb_raw)
    dt = time.time() - t0

    epc = rb_data.get("error_per_clifford", 0.0)
    gate_fidelity = rb_data.get("gate_fidelity", 0.0)
    calibration["error_per_clifford"] = epc
    calibration["gate_fidelity_rb"] = gate_fidelity

    log(f"  Error per Clifford = {epc:.2e}")
    log(f"  Gate fidelity = {gate_fidelity:.4f}")
    log(f"  Fit R² = {rb_data.get('fit_r_squared', 'N/A')}")

    # Decision: is gate quality sufficient?
    target_epc = 1e-3
    if epc < target_epc:
        decision = f"EPC = {epc:.2e} < {target_epc:.0e} target. Gate quality verified ✓"
        calibration["rb_pass"] = True
    else:
        decision = f"EPC = {epc:.2e} ≥ {target_epc:.0e} target. May need recalibration."
        calibration["rb_pass"] = False
    steps.append(TuneupStep(5, "Randomized Benchmarking", "randomized_benchmarking",
                            rb_params, rb_data, decision, dt))

    log(f"  Decision: {decision}")

    # ─────────────────────────────────────────────────
    # STEP 6: Active Learning suggestion
    # ─────────────────────────────────────────────────
    log("\n" + "═" * 70)
    log("STEP 6 · Active Learning — Next Experiment Suggestion")
    log("═" * 70)

    t0 = time.time()
    # Feed the calibration observations to the BO designer
    al_params = {
        "parameter_bounds": {
            "drag_alpha": [-2.0, 2.0],
            "pi_amp_ghz": [0.001, 0.01],
        },
        "observations": [
            {"parameters": {"drag_alpha": 0.0, "pi_amp_ghz": pi_amp}, "objective": 0.95},
            {"parameters": {"drag_alpha": optimal_alpha, "pi_amp_ghz": pi_amp}, "objective": gate_fidelity},
            {"parameters": {"drag_alpha": -1.0, "pi_amp_ghz": pi_amp * 0.8}, "objective": 0.85},
            {"parameters": {"drag_alpha": 1.5, "pi_amp_ghz": pi_amp * 1.2}, "objective": 0.88},
        ],
        "objective_name": "gate_fidelity",
        "seed": 42,
    }
    al_raw = executor.execute("next_best_experiment", al_params)
    al_data = json.loads(al_raw)
    dt = time.time() - t0

    suggested = al_data.get("suggested_parameters", {})
    log(f"  Suggested next: {suggested}")
    log(f"  Predicted fidelity: {al_data.get('predicted_objective', '?')}")
    log(f"  Rationale: {al_data.get('rationale', '')[:100]}...")

    decision = f"BO suggests: {suggested}. Agent can continue iterating."
    steps.append(TuneupStep(6, "Active Learning", "next_best_experiment",
                            al_params, al_data, decision, dt))

    # ═══════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════
    total_time = time.time() - t_start

    log("\n" + "═" * 70)
    log("TUNE-UP COMPLETE")
    log("═" * 70)
    log(f"  Total time: {total_time:.1f}s")
    log(f"  Calibration result:")
    for k, v in calibration.items():
        if isinstance(v, float):
            log(f"    {k}: {v:.6f}")
        else:
            log(f"    {k}: {v}")

    success = calibration.get("rb_pass", False)
    summary = (
        f"Qubit Q{qubit} tune-up {'PASSED' if success else 'NEEDS ITERATION'}. "
        f"π-amp={calibration['pi_amp_mhz']:.3f} MHz, "
        f"DRAG α={calibration['drag_alpha']:.4f}, "
        f"EPC={calibration['error_per_clifford']:.2e}, "
        f"T2*={calibration['T2_star_us']} μs. "
        f"Total time: {total_time:.1f}s ({len(steps)} steps, 8 tool calls)."
    )
    log(f"\n  {summary}")

    result = TuneupResult(
        steps=steps,
        calibration=calibration,
        total_time_s=total_time,
        success=success,
        summary=summary,
    )

    # Save outputs
    _save_outputs(result, save_dir)

    return result


def _save_outputs(result: TuneupResult, save_dir: str):
    """Save trace, report, and figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. JSON trace
    trace_path = os.path.join(save_dir, "tuneup_trace.json")
    trace_data = {
        "summary": result.summary,
        "success": result.success,
        "total_time_s": result.total_time_s,
        "calibration": result.calibration,
        "steps": [
            {
                "step": s.step_num,
                "name": s.name,
                "tool": s.tool,
                "input": s.input_params,
                "output": s.output,
                "decision": s.decision,
                "elapsed_s": round(s.elapsed_s, 3),
            }
            for s in result.steps
        ],
    }
    with open(trace_path, "w") as f:
        json.dump(trace_data, f, indent=2, default=str)
    print(f"\n  Trace saved: {trace_path}")

    # 2. Markdown report
    report_path = os.path.join(save_dir, "tuneup_report.md")
    report = _generate_report(result)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    # 3. Summary figure — RB decay curve + DRAG leakage
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: DRAG leakage vs α
    drag_step = result.steps[3]  # DRAG step
    if "coarse_sweep" in drag_step.output:
        alphas = drag_step.output["coarse_sweep"]["alphas"]
        leakages = drag_step.output["coarse_sweep"]["leakages"]
        axes[0].semilogy(alphas, leakages, "o-", color="#2c7fb8", markersize=4)
        axes[0].axvline(result.calibration["drag_alpha"], color="#e7298a",
                       linestyle="--", label=f"α* = {result.calibration['drag_alpha']:.3f}")
        axes[0].set_xlabel("DRAG α")
        axes[0].set_ylabel("Leakage P(|2⟩)")
        axes[0].set_title("DRAG Calibration", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Panel B: RB survival curve
    rb_step = result.steps[4]  # RB step
    if "sequence_lengths" in rb_step.output:
        lengths = rb_step.output["sequence_lengths"]
        survivals = rb_step.output["survival_probabilities"]
        axes[1].plot(lengths, survivals, "o-", color="#d95f02", markersize=5)
        axes[1].set_xlabel("Clifford sequence length")
        axes[1].set_ylabel("Survival probability")
        axes[1].set_title(
            f"Randomized Benchmarking (EPC = {result.calibration['error_per_clifford']:.2e})",
            fontweight="bold",
        )
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0.9, 1.01)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig_path = os.path.join(save_dir, "tuneup_results.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "tuneup_results.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


def _generate_report(result: TuneupResult) -> str:
    """Generate markdown report."""
    cal = result.calibration
    lines = [
        "# QuantumGPT Tune-up Report",
        "",
        f"**Status:** {'✅ PASS' if result.success else '⚠️ NEEDS ITERATION'}",
        f"**Qubit:** Q{cal['qubit']}",
        f"**Total time:** {result.total_time_s:.1f}s ({len(result.steps)} steps)",
        "",
        "## Calibration Results",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| π-pulse amplitude | {cal['pi_amp_mhz']:.3f} MHz |",
        f"| DRAG α | {cal['drag_alpha']:.4f} |",
        f"| T2* | {cal['T2_star_us']} μs |",
        f"| Detuning | {cal['detuning_mhz']} MHz |",
        f"| Error per Clifford | {cal['error_per_clifford']:.2e} |",
        f"| Gate fidelity (RB) | {cal['gate_fidelity_rb']:.4f} |",
        f"| Leakage | {cal['leakage']:.2e} |",
        "",
        "## Tune-up Sequence",
        "",
    ]

    for step in result.steps:
        lines.extend([
            f"### Step {step.step_num}: {step.name}",
            f"- **Tool:** `{step.tool}`",
            f"- **Decision:** {step.decision}",
            f"- **Time:** {step.elapsed_s:.2f}s",
            "",
        ])

    lines.extend([
        "## Figures",
        "",
        "![Tune-up Results](tuneup_results.png)",
        "",
        "---",
        f"*Generated by QuantumGPT v2.4 tune-up demo*",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# Agent-driven mode (requires LLM API key)
# ═══════════════════════════════════════════════════════

def run_tuneup_agent(
    qubit: int = 0,
    provider: str = "mock",
    save_dir: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run the tune-up sequence via the ReAct agent.

    This uses the LLM to decide the sequence autonomously.
    Falls back to mock mode if no API key is available.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "tuneup_agent")
    os.makedirs(save_dir, exist_ok=True)

    be = FakeBackendAdapter("FakeBrisbane")
    agent = ReActAgent(
        be,
        provider=provider,
        verbose=verbose,
        target_fidelity=0.0,
        max_tool_calls=15,
        use_memory=False,
    )

    prompt = (
        f"Perform a complete single-qubit tune-up sequence on qubit Q{qubit}. "
        "Steps: (1) check backend health, (2) run Rabi experiment to find π-pulse amplitude, "
        "(3) run Ramsey to measure T2* and detuning, (4) run DRAG calibration to suppress leakage, "
        "(5) run Randomized Benchmarking to verify gate quality. "
        "Report the final calibration parameters."
    )

    trace = agent.run(prompt)
    bundle = TraceReporter(trace).write_bundle(save_dir, prefix="tuneup")

    return {
        "trace_path": str(bundle["trace"]),
        "report_path": str(bundle["report"]),
        "talk_track_path": str(bundle["talk_track"]),
        "steps": trace.num_tool_calls,
        "elapsed": trace.elapsed_seconds,
    }


# ═══════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuantumGPT Tune-up Demo")
    parser.add_argument("--mode", choices=["scripted", "agent"], default="scripted",
                       help="scripted (no API key) or agent (requires LLM)")
    parser.add_argument("--qubit", type=int, default=0)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.mode == "scripted":
        result = run_tuneup_demo(
            qubit=args.qubit,
            save_dir=args.save_dir,
            verbose=not args.quiet,
        )
        print(f"\n{'='*60}")
        print(f"  {result.summary}")
        print(f"{'='*60}")
    else:
        result = run_tuneup_agent(
            qubit=args.qubit,
            provider=args.provider,
            save_dir=args.save_dir,
            verbose=not args.quiet,
        )
        print(f"\nAgent trace saved: {result['trace_path']}")
