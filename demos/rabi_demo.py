"""Rabi end-to-end demo: prompt → agent → pulse simulation → fit → report.

Demonstrates the full QuantumGPT workflow for a Rabi oscillation experiment.
Generates a publication-quality figure and a structured report.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.react import ReActAgent
from agent.reporting import TraceReporter
from backends.fake_adapter import FakeBackendAdapter


def run_rabi_demo(
    freq_ghz: float = 5.0,
    pulse_shape: str = "gaussian",
    duration_ns: float = 100.0,
    save_dir: str | None = None,
) -> dict:
    """Run the Rabi demo end-to-end."""

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(save_dir, exist_ok=True)

    # Build the prompt
    prompt = (
        f"Run a Rabi oscillation experiment on a qubit at {freq_ghz} GHz "
        f"using a {pulse_shape} pulse of {duration_ns:.0f}ns duration. "
        f"Fit the oscillation data to extract the π-pulse amplitude. "
        f"Report the π and π/2 pulse amplitudes."
    )

    print(f"\n{'#'*70}")
    print(f"# QUANTUMGPT RABI DEMO")
    print(f"# Prompt: {prompt}")
    print(f"{'#'*70}\n")

    # Create backend and agent
    be = FakeBackendAdapter("FakeBrisbane")
    agent = ReActAgent(
        be,
        provider="mock",
        verbose=True,
        target_fidelity=0.0,  # No fidelity target for characterization tasks
        max_tool_calls=10,
        use_memory=False,
    )

    # Run agent
    trace = agent.run(prompt)

    # Extract data from trace
    rabi_data = None
    fit_data = None
    for step in trace.steps:
        if step.action == "rabi_experiment" and step.observation:
            rabi_data = json.loads(step.observation)
        elif step.action == "fit_rabi" and step.observation:
            fit_data = json.loads(step.observation)

    if rabi_data is None:
        print("ERROR: No Rabi data in trace!")
        return {}

    # Generate figure
    fig_path = _plot_rabi(rabi_data, fit_data, save_dir, pulse_shape, freq_ghz, duration_ns)

    # Save human-readable report, raw trace JSON, and short demo talk track.
    bundle = TraceReporter(trace).write_bundle(save_dir, prefix="rabi")
    report_path = str(bundle["report"])
    trace_path = str(bundle["trace"])
    talk_track_path = str(bundle["talk_track"])
    print(f"\nReport saved to: {report_path}")
    print(f"Trace saved to: {trace_path}")
    print(f"Talk track saved to: {talk_track_path}")

    return {
        "fig_path": fig_path,
        "report_path": report_path,
        "trace_path": trace_path,
        "talk_track_path": talk_track_path,
        "pi_amplitude_mhz": fit_data.get("pi_amplitude_mhz") if fit_data else rabi_data.get("pi_amplitude_mhz"),
        "r_squared": fit_data.get("r_squared") if fit_data else None,
    }


def _plot_rabi(rabi_data, fit_data, save_dir, pulse_shape, freq_ghz, duration_ns):
    """Generate a publication-quality Rabi oscillation figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amps = np.array(rabi_data["amplitudes"])
    pops = np.array(rabi_data["populations"])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    # Data points
    ax.plot(amps * 1e3, pops, "o", color="#2c7fb8", markersize=5, label="Simulated data", zorder=3)

    # Fit curve
    if fit_data and fit_data.get("fit_successful", False):
        a_pi = fit_data["pi_amplitude_ghz"]
        A = fit_data["amplitude"]
        offset = fit_data["offset"]

        amps_fine = np.linspace(amps[0], amps[-1], 200)
        pops_fit = A * np.sin(np.pi * amps_fine / (2 * a_pi)) ** 2 + offset
        ax.plot(amps_fine * 1e3, pops_fit, "-", color="#d95f02", linewidth=1.5,
                label=f"Fit (R²={fit_data['r_squared']:.4f})", zorder=2)

        # Mark π-pulse
        ax.axvline(a_pi * 1e3, color="#e7298a", linestyle="--", linewidth=1,
                   label=f"π-pulse = {a_pi*1e3:.2f} MHz", zorder=1)
        ax.axvline(a_pi * 1e3 / 2, color="#66a61e", linestyle=":", linewidth=1,
                   label=f"π/2-pulse = {a_pi*1e3/2:.2f} MHz", zorder=1)

    ax.set_xlabel("Drive amplitude (MHz)", fontsize=12)
    ax.set_ylabel("P(|1⟩)", fontsize=12)
    ax.set_title(
        f"Rabi Oscillation — {pulse_shape} pulse, {freq_ghz} GHz, {duration_ns:.0f} ns",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=10)

    # Clean style
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig.tight_layout()

    # Save
    for ext in ["pdf", "png"]:
        path = os.path.join(save_dir, f"rabi_oscillation.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig_path = os.path.join(save_dir, "rabi_oscillation.png")
    print(f"Figure saved to: {fig_path}")
    return fig_path


def _generate_report(trace, rabi_data, fit_data, freq_ghz, pulse_shape, duration_ns):
    """Generate a markdown report."""
    lines = [
        "# Rabi Oscillation Experiment Report",
        "",
        "## Experiment Parameters",
        f"- **Qubit frequency:** {freq_ghz} GHz",
        f"- **Pulse shape:** {pulse_shape}",
        f"- **Pulse duration:** {duration_ns:.0f} ns",
        f"- **Amplitude sweep:** {rabi_data['amplitudes'][0]*1e3:.2f} – {rabi_data['amplitudes'][-1]*1e3:.2f} MHz",
        f"- **Number of points:** {rabi_data['n_points']}",
        "",
        "## Results",
    ]

    if fit_data and fit_data.get("fit_successful", False):
        lines.extend([
            f"- **π-pulse amplitude:** {fit_data['pi_amplitude_mhz']:.2f} MHz ({fit_data['pi_amplitude_ghz']:.6f} GHz)",
            f"- **π/2-pulse amplitude:** {fit_data['half_pi_amplitude_ghz']*1e3:.2f} MHz ({fit_data['half_pi_amplitude_ghz']:.6f} GHz)",
            f"- **Fit R²:** {fit_data['r_squared']:.4f}",
            f"- **Oscillation amplitude:** {fit_data['amplitude']:.4f}",
            f"- **Offset:** {fit_data['offset']:.4f}",
            f"- **π uncertainty:** {fit_data.get('pi_uncertainty_ghz', 'N/A')} GHz",
        ])
    else:
        lines.extend([
            f"- **π-pulse amplitude (from data):** {rabi_data['pi_amplitude_mhz']:.2f} MHz",
            f"- **Max P(|1⟩):** {rabi_data['max_population']:.4f}",
        ])

    lines.extend([
        "",
        "## Agent Trace",
        f"- **Model:** {trace.model}",
        f"- **Backend:** {trace.backend}",
        f"- **Tool calls:** {trace.num_tool_calls}",
        f"- **Elapsed:** {trace.elapsed_seconds:.1f}s",
        "",
        "### Steps:",
    ])

    for s in trace.steps:
        if s.thought:
            lines.append(f"1. **Thought:** {s.thought}")
        if s.action:
            lines.append(f"   **Action:** `{s.action}`")
        if s.fidelity_observed is not None:
            lines.append(f"   **Fidelity:** {s.fidelity_observed}")

    lines.extend([
        "",
        "## Figure",
        "![Rabi Oscillation](rabi_oscillation.png)",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_rabi_demo()
    print(f"\n{'='*60}")
    print(f"Demo complete!")
    print(f"  π-amplitude: {result['pi_amplitude_mhz']} MHz")
    print(f"  R²: {result['r_squared']}")
    print(f"  Figure: {result['fig_path']}")
    print(f"  Report: {result['report_path']}")
    print(f"{'='*60}")
