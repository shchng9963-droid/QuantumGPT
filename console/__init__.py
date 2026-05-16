"""Operator Console — terminal dashboard for quantum hardware monitoring.

Displays real-time (or replayed) drift telemetry in a Rich Live terminal UI.
Supports:
  - Live streaming from PropertiesStream
  - Batch replay of historical data (SyntheticDrift or ReplayBackend)
  - Drift changepoint visualization
  - One-week drift history playback

Usage:
    # Replay SUDDEN_DEGRADATION over 168h (1 week):
    python -m console.dashboard --profile sudden --hours 168 --speed 100

    # Replay stable baseline:
    python -m console.dashboard --profile stable --hours 24

    # Custom: linear decay over 48h at 50x speed:
    python -m console.dashboard --profile linear --hours 48 --speed 50
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from backends.synthetic_drift import (
    SyntheticDriftBackend,
    DriftProfile,
    STABLE,
    LINEAR_DECAY,
    SUDDEN_DEGRADATION,
)
from backends.properties_stream import PropertiesStream
from detection.drift_detector import DriftDetector, extract_features, standardize


PROFILES = {
    "stable": STABLE,
    "linear": LINEAR_DECAY,
    "sudden": SUDDEN_DEGRADATION,
    "periodic": DriftProfile(
        t1_drift=lambda t: 1.0 - 0.15 * np.sin(2 * np.pi * t / 24),
        gate_error_drift=lambda t: 1.0 + 0.1 * np.sin(2 * np.pi * t / 24),
        drift_score_fn=lambda t: float(abs(np.sin(2 * np.pi * t / 24)) * 0.3),
    ),
    "week_realistic": DriftProfile(
        # Realistic 1-week pattern: daily cycles + gradual degradation + sudden shift at day 5
        t1_drift=lambda t: (
            max(0.3, 1.0 - 0.005 * t) *  # gradual degradation
            (1.0 - 0.1 * np.sin(2 * np.pi * t / 24)) *  # daily cycle
            (0.4 if t >= 120 else 1.0)  # sudden shift at hour 120 (day 5)
        ),
        gate_error_drift=lambda t: (
            min(5.0, 1.0 + 0.003 * t) *
            (1.0 + 0.05 * np.sin(2 * np.pi * t / 24)) *
            (3.0 if t >= 120 else 1.0)
        ),
        readout_drift=lambda t: (
            (1.0 + 0.002 * t) *
            (2.0 if t >= 120 else 1.0)
        ),
    ),
}


def make_sparkline(values: list[float], width: int = 40) -> str:
    """Create a Unicode sparkline from values."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1.0

    # Downsample if needed
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    return "".join(blocks[min(8, int((v - mn) / rng * 8))] for v in sampled)


def severity_color(value: float, thresholds: tuple = (0.3, 0.6)) -> str:
    """Return a color name based on severity."""
    if value < thresholds[0]:
        return "green"
    elif value < thresholds[1]:
        return "yellow"
    return "red"


def build_dashboard(
    snapshots: list[dict],
    changepoints: list[dict],
    current_idx: int,
    total: int,
    backend_name: str,
    elapsed_real: float,
) -> Layout:
    """Build the Rich Layout for the dashboard."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )

    # Header
    snap = snapshots[current_idx] if current_idx < len(snapshots) else snapshots[-1]
    sim_time = snap.get("stream_sim_time_hours", 0)
    header_text = Text(f" QuantumGPT Operator Console — {backend_name}", style="bold white on blue")
    header_text.append(f"  |  t = {sim_time:.1f}h  |  {current_idx + 1}/{total} snapshots", style="white on blue")
    layout["header"].update(Panel(header_text, style="blue"))

    # Left: metrics table + sparklines
    left_layout = Layout()
    left_layout.split_column(
        Layout(name="metrics", ratio=1),
        Layout(name="sparklines", ratio=1),
    )

    # Current metrics table
    metrics_table = Table(title="Current Metrics", box=box.SIMPLE_HEAVY, expand=True)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_column("Status", justify="center")

    if "qubit_t1_us" in snap:
        t1 = np.mean(snap["qubit_t1_us"])
        t2 = np.mean(snap["qubit_t2_us"])
        re = np.mean(snap["qubit_readout_error"])
        ge1q = np.mean(snap["qubit_gate_error_1q"])
        ge2q = np.mean(list(snap.get("gate_error_2q", {}).values())) if snap.get("gate_error_2q") else 0
    else:
        t1 = snap.get("avg_t1_us", 0)
        t2 = snap.get("avg_t2_us", 0)
        re = snap.get("avg_readout_error", 0)
        ge1q = snap.get("avg_1q_error", 0)
        ge2q = snap.get("avg_2q_error", 0)

    drift = snap.get("drift_score", 0)

    rows = [
        ("Avg T1", f"{t1:.1f} μs", "green" if t1 > 150 else ("yellow" if t1 > 80 else "red")),
        ("Avg T2", f"{t2:.1f} μs", "green" if t2 > 100 else ("yellow" if t2 > 50 else "red")),
        ("Readout Err", f"{re:.4f}", "green" if re < 0.03 else ("yellow" if re < 0.06 else "red")),
        ("1Q Gate Err", f"{ge1q:.5f}", "green" if ge1q < 0.001 else ("yellow" if ge1q < 0.005 else "red")),
        ("2Q Gate Err", f"{ge2q:.5f}", "green" if ge2q < 0.01 else ("yellow" if ge2q < 0.03 else "red")),
        ("Drift Score", f"{drift:.3f}", severity_color(drift)),
    ]

    for name, val, color in rows:
        metrics_table.add_row(name, val, Text("●", style=color))

    left_layout["metrics"].update(Panel(metrics_table))

    # Sparklines
    history = snapshots[:current_idx + 1]
    if history:
        raw, times = extract_features(history)
        t1_history = raw[:, 0].tolist()
        t2_history = raw[:, 1].tolist()
        re_history = raw[:, 2].tolist()

        spark_text = Text()
        spark_text.append("T1  ", style="cyan")
        spark_text.append(make_sparkline(t1_history, 50))
        spark_text.append(f"  {t1_history[-1]:.0f}μs\n")
        spark_text.append("T2  ", style="cyan")
        spark_text.append(make_sparkline(t2_history, 50))
        spark_text.append(f"  {t2_history[-1]:.0f}μs\n")
        spark_text.append("RdE ", style="cyan")
        spark_text.append(make_sparkline(re_history, 50))
        spark_text.append(f"  {re_history[-1]:.4f}\n")

        # Mark changepoints in timeline
        if changepoints:
            cp_times = [cp["time_hours"] for cp in changepoints]
            spark_text.append("\n⚡ Changepoints: ", style="bold red")
            spark_text.append(", ".join(f"{t:.1f}h" for t in cp_times))

        left_layout["sparklines"].update(Panel(spark_text, title="History"))
    else:
        left_layout["sparklines"].update(Panel("Collecting data..."))

    layout["left"].update(left_layout)

    # Right: changepoints + recommendations
    cp_table = Table(title="Drift Events", box=box.SIMPLE, expand=True)
    cp_table.add_column("Time", style="cyan")
    cp_table.add_column("Sev", justify="center")
    cp_table.add_column("Dir", style="dim")

    for cp in changepoints[-8:]:  # last 8
        sev = cp.get("severity", 0)
        color = severity_color(sev)
        cp_table.add_row(
            f"{cp['time_hours']:.1f}h",
            Text(f"{sev:.2f}", style=color),
            cp.get("direction", "?"),
        )

    if not changepoints:
        cp_table.add_row("—", "—", "No drift detected")

    layout["right"].update(Panel(cp_table))

    # Footer
    pct = (current_idx + 1) / total * 100
    bar_width = 40
    filled = int(pct / 100 * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    footer_text = Text(f" [{bar}] {pct:.0f}%  |  Real: {elapsed_real:.1f}s", style="dim")
    layout["footer"].update(Panel(footer_text))

    return layout


def run_dashboard(
    profile_name: str = "sudden",
    hours: float = 168,
    step_hours: float = 1.0,
    speed: float = 50,
):
    """Run the operator console dashboard."""
    profile = PROFILES.get(profile_name, STABLE)
    backend = SyntheticDriftBackend(profile=profile)

    console = Console()
    console.print(f"[bold]Generating {hours}h of telemetry (step={step_hours}h)...[/bold]")

    # Generate all snapshots upfront
    snapshots = PropertiesStream.replay_all(backend, hours=hours, step_hours=step_hours)
    total = len(snapshots)
    console.print(f"[green]{total} snapshots ready.[/green]")

    # Run drift detection
    console.print("[bold]Running drift detection...[/bold]")
    detector = DriftDetector(method="pelt", pelt_penalty=3.0)
    report = detector.detect(snapshots)
    changepoints = [
        {
            "time_hours": cp.time_hours,
            "severity": cp.severity,
            "direction": cp.direction,
            "features_affected": cp.features_affected,
        }
        for cp in report.changepoints
    ]
    console.print(f"[green]Detection complete: {len(changepoints)} changepoints found.[/green]")
    console.print("[dim]Starting replay... (Ctrl+C to stop)[/dim]\n")

    # Real-time delay between frames
    delay = step_hours * 3600 / speed  # real seconds per step

    t0 = time.time()
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        for i in range(total):
            elapsed = time.time() - t0
            dashboard = build_dashboard(
                snapshots, changepoints, i, total, backend.name, elapsed
            )
            live.update(dashboard)

            # Sleep with interruptibility
            try:
                time.sleep(max(0.05, delay))
            except KeyboardInterrupt:
                break

    console.print("\n[bold green]Replay complete.[/bold green]")
    console.print(f"Drift report:\n{report.summary()}")


def main():
    parser = argparse.ArgumentParser(description="QuantumGPT Operator Console")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="week_realistic",
                        help="Drift profile (default: week_realistic)")
    parser.add_argument("--hours", type=float, default=168, help="Hours to simulate (default: 168 = 1 week)")
    parser.add_argument("--step", type=float, default=1.0, help="Step size in hours (default: 1.0)")
    parser.add_argument("--speed", type=float, default=100, help="Replay speed multiplier (default: 100)")
    args = parser.parse_args()

    run_dashboard(
        profile_name=args.profile,
        hours=args.hours,
        step_hours=args.step,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
