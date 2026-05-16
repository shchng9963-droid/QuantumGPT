"""QuantumGPT MVP CLI — one-liner quantum simulation & diagnostics.

Usage:
    qgpt simulate ghz --shots 8192 --backend FakeBrisbane
    qgpt health --backend FakeBrisbane
    qgpt list
    qgpt agent "Run a GHZ-5 and diagnose" --backend FakeBrisbane
"""

import json
import os
import sys
import time

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# ── Backend factory ──────────────────────────────────────────────

BACKEND_CHOICES = ["FakeBrisbane", "FakeKyiv", "FakeSherbrooke", "FakeTorino"]

# Map short circuit names → (builder_func, description)
CIRCUIT_ALIASES = {
    "ghz":   ("ghz_5",  5),
    "ghz3":  ("ghz_3",  3),
    "ghz5":  ("ghz_5",  5),
    "qft":   ("qft_4",  4),
    "qft4":  ("qft_4",  4),
    "bv":    ("bv_5",   5),
    "bv5":   ("bv_5",   5),
    "vqe":   ("vqe_4",  4),
    "vqe4":  ("vqe_4",  4),
    "qaoa":  ("qaoa_4", 4),
    "qaoa4": ("qaoa_4", 4),
}


def _make_backend(name: str):
    """Instantiate a ShadowBackend by name."""
    from backends.fake_adapter import FakeBackendAdapter
    return FakeBackendAdapter(name)


def _fidelity_color(f: float) -> str:
    if f >= 0.90:
        return "green"
    elif f >= 0.70:
        return "yellow"
    else:
        return "red"


# ── Main group ───────────────────────────────────────────────────

@click.group()
@click.version_option("0.1.0", prog_name="qgpt")
def main():
    """QuantumGPT — AI-powered quantum computing assistant."""
    pass


# ── simulate ─────────────────────────────────────────────────────

@main.command()
@click.argument("circuit")
@click.option("--shots", "-s", default=4096, show_default=True,
              help="Number of measurement shots.")
@click.option("--backend", "-b", default="FakeBrisbane", show_default=True,
              type=click.Choice(BACKEND_CHOICES, case_sensitive=True),
              help="Shadow backend to use.")
@click.option("--json-output", "-j", is_flag=True,
              help="Output raw JSON instead of rich table.")
def simulate(circuit, shots, backend, json_output):
    """Run a quantum circuit on a shadow backend.

    CIRCUIT can be a short alias (ghz, qft, bv, vqe, qaoa) or a full
    benchmark name (ghz_5, qft_4, bv_5, vqe_4, qaoa_4, DJ-5, etc.).
    """
    t0 = time.time()

    # Resolve alias
    circuit_lower = circuit.lower().replace("-", "").replace("_", "")
    if circuit_lower in CIRCUIT_ALIASES:
        circuit_name, _ = CIRCUIT_ALIASES[circuit_lower]
    else:
        circuit_name = circuit  # pass through as-is

    if not json_output:
        console.print(f"\n[bold cyan]qgpt simulate[/] {circuit_name} "
                      f"--shots {shots} --backend {backend}\n")

    # Build backend + executor
    be = _make_backend(backend)
    from tools.quantum_tools import ToolExecutor
    executor = ToolExecutor(be)

    result_str = executor.execute("run_circuit", {
        "circuit_name": circuit_name,
        "shots": shots,
    })
    result = json.loads(result_str)
    elapsed = time.time() - t0

    if json_output:
        result["elapsed_seconds"] = round(elapsed, 2)
        click.echo(json.dumps(result, indent=2))
        return

    # Rich output
    fid = result.get("fidelity")
    fid_str = f"{fid:.4f}" if fid is not None else "N/A"
    fid_color = _fidelity_color(fid) if fid else "white"

    # Header panel
    header = Text()
    header.append("Circuit: ", style="bold")
    header.append(f"{result['circuit']}\n")
    header.append("Backend: ", style="bold")
    header.append(f"{backend} ({be.num_qubits}q)\n")
    header.append("Shots:   ", style="bold")
    header.append(f"{result['shots']}\n")
    header.append("Fidelity: ", style="bold")
    header.append(fid_str, style=f"bold {fid_color}")
    header.append(f"\nDepth:   ", style="bold")
    header.append(f"{result.get('transpiled_depth', 'N/A')}")
    header.append(f"\nTime:    ", style="bold")
    header.append(f"{elapsed:.2f}s")

    console.print(Panel(header, title="[bold]Simulation Result[/]",
                        border_style="cyan"))

    # Counts table
    top_counts = result.get("top_counts", {})
    if top_counts:
        table = Table(title="Top Measurement Outcomes", show_lines=False)
        table.add_column("Bitstring", style="cyan", justify="center")
        table.add_column("Count", justify="right")
        table.add_column("Probability", justify="right")
        total = sum(top_counts.values())
        for bs, cnt in sorted(top_counts.items(), key=lambda x: -x[1]):
            prob = cnt / total
            table.add_row(bs, str(cnt), f"{prob:.4f}")
        console.print(table)

    console.print()


# ── health ───────────────────────────────────────────────────────

@main.command()
@click.option("--backend", "-b", default="FakeBrisbane", show_default=True,
              type=click.Choice(BACKEND_CHOICES, case_sensitive=True),
              help="Shadow backend to query.")
@click.option("--json-output", "-j", is_flag=True,
              help="Output raw JSON.")
def health(backend, json_output):
    """Show backend health: T1/T2, error rates, drift."""
    be = _make_backend(backend)
    from tools.quantum_tools import ToolExecutor
    executor = ToolExecutor(be)

    result_str = executor.execute("get_backend_health", {})
    result = json.loads(result_str)

    if json_output:
        click.echo(json.dumps(result, indent=2))
        return

    console.print(f"\n[bold cyan]Backend Health: {result['backend']}[/]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Qubits", str(result["num_qubits"]))
    table.add_row("Avg T1", f"{result['avg_t1_us']} μs")
    table.add_row("Avg T2", f"{result['avg_t2_us']} μs")

    # Color-code errors
    e1 = result["avg_1q_error"]
    e2 = result["avg_2q_error"]
    er = result["avg_readout_error"]
    table.add_row("Avg 1Q error", f"[{'green' if e1 < 0.001 else 'yellow' if e1 < 0.01 else 'red'}]{e1:.6f}[/]")
    table.add_row("Avg 2Q error", f"[{'green' if e2 < 0.01 else 'yellow' if e2 < 0.05 else 'red'}]{e2:.6f}[/]")
    table.add_row("Readout error", f"[{'green' if er < 0.01 else 'yellow' if er < 0.05 else 'red'}]{er:.6f}[/]")

    drift = result.get("drift_score")
    if drift is not None:
        drift_color = "green" if drift < 0.2 else "yellow" if drift < 0.5 else "red"
        table.add_row("Drift score", f"[{drift_color}]{drift:.4f}[/]")

    table.add_row("Cal age", f"{result['calibration_age_minutes']:.1f} min")

    console.print(table)
    console.print()


# ── list ─────────────────────────────────────────────────────────

@main.command("list")
@click.option("--json-output", "-j", is_flag=True, help="Output raw JSON.")
def list_circuits(json_output):
    """List available benchmark circuits."""
    from bench.circuits import list_benchmarks
    from bench.mqtbench import get_mqtbench_circuits

    hand = list_benchmarks()

    try:
        mqt_raw = get_mqtbench_circuits()
        mqt = {k: v for k, v in mqt_raw.items()}
    except Exception:
        mqt = {}

    if json_output:
        out = {"hand_written": hand}
        if mqt:
            out["mqtbench"] = {k: {"description": v.description,
                                    "qubits": v.num_qubits}
                               for k, v in mqt.items()}
        click.echo(json.dumps(out, indent=2))
        return

    console.print("\n[bold cyan]Available Benchmark Circuits[/]\n")

    table = Table(title="Hand-Written", show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Alias", style="dim")
    for name, desc in hand.items():
        # find alias
        alias = next((a for a, (n, _) in CIRCUIT_ALIASES.items()
                      if n == name and len(a) <= 5), "")
        table.add_row(name, desc, alias)
    console.print(table)

    if mqt:
        console.print()
        table2 = Table(title="MQTBench", show_lines=False)
        table2.add_column("Name", style="cyan")
        table2.add_column("Qubits", justify="right")
        table2.add_column("Depth", justify="right")
        table2.add_column("ECR gates", justify="right")
        table2.add_column("Description")
        for label, mc in mqt.items():
            table2.add_row(label, str(mc.num_qubits),
                          str(mc.depth), str(mc.ecr_count),
                          mc.description)
        console.print(table2)

    console.print()


# ── agent ────────────────────────────────────────────────────────

@main.command()
@click.argument("prompt")
@click.option("--backend", "-b", default="FakeBrisbane", show_default=True,
              type=click.Choice(BACKEND_CHOICES, case_sensitive=True))
@click.option("--model", "-m", default="rule-planner-v1", show_default=True,
              help="LLM model (or 'rule-planner-v1' for offline mock).")
@click.option("--provider", "-p", default="auto",
              type=click.Choice(["auto", "openai", "anthropic", "mock"]),
              help="API provider.")
@click.option("--json-output", "-j", is_flag=True)
def agent(prompt, backend, model, provider, json_output):
    """Run the QuantumGPT agent with a natural-language prompt.

    Examples:
        qgpt agent "Check health and run GHZ-5"
        qgpt agent "Diagnose why fidelity is low" --model claude-sonnet-4-20250514
    """
    from agent.loop import QuantumAgent

    be = _make_backend(backend)

    use_mock = (model == "rule-planner-v1" or provider == "mock")
    ag = QuantumAgent(
        backend=be,
        model=model,
        provider=provider,
        verbose=not json_output,
        use_mock=use_mock,
    )
    result = ag.run(prompt)

    if json_output:
        click.echo(json.dumps({
            "final_answer": result.final_answer,
            "tool_calls": result.tool_calls_made,
            "total_tokens": result.total_tokens,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "model": result.model,
        }, indent=2))
    elif not ag.verbose:
        console.print(result.final_answer)


# ── diagnose ─────────────────────────────────────────────────────

@main.command()
@click.option("--backend", "-b", default="FakeBrisbane", show_default=True,
              type=click.Choice(BACKEND_CHOICES, case_sensitive=True))
@click.option("--json-output", "-j", is_flag=True)
def diagnose(backend, json_output):
    """Run diagnostics on the backend and get suggestions."""
    be = _make_backend(backend)
    from tools.quantum_tools import ToolExecutor
    executor = ToolExecutor(be)

    result_str = executor.execute("diagnose_and_suggest", {})
    result = json.loads(result_str)

    if json_output:
        click.echo(json.dumps(result, indent=2))
        return

    sev = result["severity"]
    sev_color = {"nominal": "green", "warning": "yellow", "critical": "red"}.get(sev, "white")

    console.print(f"\n[bold cyan]Diagnostics: {backend}[/]")
    console.print(f"Severity: [bold {sev_color}]{sev.upper()}[/]\n")

    if result.get("drift_score") is not None:
        console.print(f"  Drift score:  {result['drift_score']:.4f}")
    console.print(f"  Avg 1Q error: {result['avg_1q_error']:.6f}")
    console.print(f"  Avg 2Q error: {result['avg_2q_error']:.6f}")
    console.print()

    for s in result.get("suggestions", []):
        console.print(f"  [yellow]>[/] {s}")
    console.print()


# ── records ──────────────────────────────────────────────────────

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "quantumgpt.duckdb")


@main.command()
@click.option("--circuit", "-c", default=None, help="Filter by circuit name.")
@click.option("--backend", "-b", default=None, help="Filter by backend.")
@click.option("--limit", "-n", default=20, show_default=True,
              help="Max records to show.")
@click.option("--stats", "-s", is_flag=True, help="Show aggregate stats.")
@click.option("--json-output", "-j", is_flag=True)
@click.option("--db", default=DEFAULT_DB_PATH, help="Path to DuckDB file.",
              show_default=True)
def records(circuit, backend, limit, stats, json_output, db):
    """Browse experiment records (structured experiment memory).

    Examples:
        qgpt records                    # recent experiments
        qgpt records -c ghz_5           # filter by circuit
        qgpt records -b FakeBrisbane    # filter by backend
        qgpt records --stats            # aggregate statistics
    """
    import os as _os
    if not _os.path.exists(db):
        console.print(f"[yellow]No database found at {db}[/]")
        console.print("Run some experiments first (e.g., qgpt simulate ghz)")
        return

    from data.store import DataStore
    store = DataStore(db)
    exp_store = store.experiments

    if stats:
        _show_stats(exp_store, json_output)
        store.close()
        return

    # Fetch records
    if circuit:
        recs = exp_store.by_circuit(circuit, limit=limit)
    elif backend:
        recs = exp_store.by_backend(backend, limit=limit)
    else:
        recs = exp_store.recent(limit=limit)

    if json_output:
        click.echo(json.dumps([r.to_dict() for r in recs], indent=2,
                               default=str))
        store.close()
        return

    if not recs:
        console.print("[yellow]No experiment records found.[/]")
        console.print("Run experiments with instrumented executor or agent mode.")
        store.close()
        return

    console.print(f"\n[bold cyan]Experiment Records[/] ({len(recs)} shown)\n")

    table = Table(show_lines=False)
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Type", style="cyan")
    table.add_column("Backend")
    table.add_column("Circuit")
    table.add_column("Fidelity", justify="right")
    table.add_column("Outcome")
    table.add_column("Summary", max_width=50)

    for r in recs:
        fid_str = f"{r.fidelity:.4f}" if r.fidelity is not None else "-"
        fid_color = _fidelity_color(r.fidelity) if r.fidelity else "white"
        outcome_color = {"success": "green", "partial": "yellow",
                         "failure": "red", "error": "red"}.get(r.outcome, "white")
        table.add_row(
            r.id[:10],
            r.experiment_type.replace("_", " "),
            r.backend,
            r.circuit_name or "-",
            f"[{fid_color}]{fid_str}[/]",
            f"[{outcome_color}]{r.outcome}[/]",
            (r.summary[:48] + "..") if len(r.summary) > 50 else r.summary,
        )

    console.print(table)
    console.print()
    store.close()


def _show_stats(exp_store, json_output):
    """Show aggregate statistics."""
    overall = exp_store.stats()
    by_circuit = exp_store.stats_by_circuit()
    by_backend = exp_store.stats_by_backend()

    if json_output:
        click.echo(json.dumps({
            "overall": overall,
            "by_circuit": by_circuit,
            "by_backend": by_backend,
        }, indent=2, default=str))
        return

    console.print("\n[bold cyan]Experiment Statistics[/]\n")

    # Overall
    table = Table(title="Overall", show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total experiments", str(overall.get("total_experiments", 0)))
    table.add_row("Unique backends", str(overall.get("unique_backends", 0)))
    table.add_row("Unique circuits", str(overall.get("unique_circuits", 0)))
    s = overall.get("successes", 0)
    t = overall.get("total_experiments", 0)
    rate = f"{s}/{t} ({100*s/t:.0f}%)" if t else "0/0"
    table.add_row("Success rate", rate)
    avg_f = overall.get("avg_fidelity")
    table.add_row("Avg fidelity", f"{avg_f:.4f}" if avg_f else "-")
    table.add_row("Min fidelity", f"{overall.get('min_fidelity', 0):.4f}"
                  if overall.get("min_fidelity") else "-")
    table.add_row("Max fidelity", f"{overall.get('max_fidelity', 0):.4f}"
                  if overall.get("max_fidelity") else "-")
    console.print(table)

    # By circuit
    if by_circuit:
        console.print()
        ct = Table(title="By Circuit", show_lines=False)
        ct.add_column("Circuit", style="cyan")
        ct.add_column("Runs", justify="right")
        ct.add_column("Avg Fidelity", justify="right")
        ct.add_column("Min", justify="right")
        ct.add_column("Max", justify="right")
        ct.add_column("Success", justify="right")
        for row in by_circuit:
            ct.add_row(
                row["circuit_name"] or "-",
                str(row["runs"]),
                f"{row['avg_fidelity']:.4f}" if row["avg_fidelity"] else "-",
                f"{row['min_fidelity']:.4f}" if row["min_fidelity"] else "-",
                f"{row['max_fidelity']:.4f}" if row["max_fidelity"] else "-",
                str(row["successes"]),
            )
        console.print(ct)

    # By backend
    if by_backend:
        console.print()
        bt = Table(title="By Backend", show_lines=False)
        bt.add_column("Backend", style="cyan")
        bt.add_column("Runs", justify="right")
        bt.add_column("Avg Fidelity", justify="right")
        bt.add_column("Success", justify="right")
        bt.add_column("Circuits", justify="right")
        for row in by_backend:
            bt.add_row(
                row["backend"],
                str(row["runs"]),
                f"{row['avg_fidelity']:.4f}" if row["avg_fidelity"] else "-",
                str(row["successes"]),
                str(row["unique_circuits"]),
            )
        console.print(bt)

    console.print()


if __name__ == "__main__":
    main()
