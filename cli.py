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
    from agent.factory import build_react_agent

    be = _make_backend(backend)
    ag = build_react_agent(
        be,
        model=model,
        provider=provider,
        verbose=not json_output,
        use_memory=False,
    )
    trace = ag.run(prompt)

    if json_output:
        click.echo(json.dumps({
            "final_answer": trace.final_answer,
            "tool_calls": trace.tool_calls_made,
            "total_tokens": trace.total_tokens,
            "elapsed_seconds": round(trace.elapsed_seconds, 2),
            "model": model,
        }, indent=2))
    elif not ag.verbose:
        console.print(trace.final_answer)


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


def _build_orchestrator_task(
    *,
    task_id,
    circuit_name,
    backend,
    target,
    shots,
    max_tool_calls,
    max_seconds,
    drift_profile,
    provider,
    model,
    system,
):
    from agent.orchestrator.state import make_task

    task = make_task(
        user_prompt=(
            f"Run circuit {circuit_name} on {backend}. "
            f"Target fidelity {target}. Apply error mitigation if needed."
        ),
        task_type="benchmark",
        circuit=circuit_name,
        profile=drift_profile,
        target_fidelity=target,
        backend_id=backend,
        max_seconds=float(max_seconds),
        max_tool_calls=max_tool_calls,
        task_id=task_id,
        meta={
            "provider": provider,
            "model": model,
            "shots": shots,
            "system": system,
        },
    )
    task["provider"] = provider
    task["model"] = model
    task["shots"] = shots
    return task


def _run_orchestrator_graph(*, task, task_id, system, save_artifact, append_trace):
    from agent.orchestrator.graph import build_graph
    from agent.orchestrator.state import new_orchestrator_state

    graph = build_graph(use_stub_verifier=(system == "QuantumGPT-Orch-NoVerifier"))
    pre_state = dict(new_orchestrator_state(task))
    save_artifact(task_id, "pre_state", pre_state)

    post_state = dict(graph.invoke(pre_state))
    save_artifact(task_id, "post_state", post_state)

    for entry in post_state.get("history") or []:
        append_trace(task_id, entry)

    verification = dict(post_state.get("verification") or {})
    fidelity_check = dict(verification.get("fidelity_check") or {})
    budget_used = dict(post_state.get("budget_used") or {})
    traces = list(post_state.get("executor_traces") or [])
    final_answer = post_state.get("final_answer") or ""
    if not final_answer and traces:
        final_answer = traces[-1].get("final_answer", "") or ""

    best_score = post_state.get("last_fidelity")
    if best_score is None:
        best_score = fidelity_check.get("recomputed")
    if best_score is None:
        best_score = fidelity_check.get("best_in_trace")

    return {
        "best_score": best_score,
        "final_answer": final_answer,
        "tool_calls": int(budget_used.get("tool_calls") or 0),
        "elapsed_seconds": round(float(budget_used.get("walltime_seconds") or 0.0), 3),
        "total_cost_usd": float(budget_used.get("cost_usd") or 0.0),
        "verifier_satisfied": bool(verification.get("is_satisfied")),
        "verifier_hallucinated": bool(verification.get("hallucinated")),
        "termination_reason": post_state.get("termination_reason") or "",
        "plan_revision": int(post_state.get("plan_revision") or 0),
        "verifier_reason": verification.get("reason") or "",
    }


# ── run / runs / show / rerun ────────────────────────────────────
# Stable task_id-addressed task records under ~/.quantumgpt/runs/<id>/

@main.command()
@click.argument("circuit")
@click.option("--backend", "-b", default="FakeBrisbane", show_default=True,
              type=click.Choice(BACKEND_CHOICES, case_sensitive=True))
@click.option("--target", "-t", default=0.85, show_default=True, type=float,
              help="Target fidelity.")
@click.option("--shots", "-s", default=4096, show_default=True, type=int)
@click.option("--max-tool-calls", default=12, show_default=True, type=int)
@click.option("--max-seconds", default=60, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--drift-profile", default="stable", show_default=True,
              type=click.Choice(["stable", "mild_gradual", "multi_shock",
                                 "severe_sudden"], case_sensitive=False))
@click.option("--system", default="QuantumGPT-Full", show_default=True,
              help="Agent variant: QuantumGPT-Full / QuantumGPT-Orchestrator / "
                   "ChatLLM-NoTools / Static-Pipeline.")
@click.option("--model", "-m", default="rule-planner-v1", show_default=True,
              help="LLM model (or 'rule-planner-v1' for offline mock).")
@click.option("--provider", "-p", default="auto",
              type=click.Choice(["auto", "openai", "anthropic", "deepseek",
                                 "mock"]))
@click.option("--json-output", "-j", is_flag=True)
def run(circuit, backend, target, shots, max_tool_calls, max_seconds, seed,
        drift_profile, system, model, provider, json_output):
    """Run a full agent task, persist to ~/.quantumgpt/runs/<task_id>/.

    Examples:
        qgpt run ghz_5 --target 0.85 --backend FakeBrisbane
        qgpt run qft_4 --backend FakeKyiv --max-seconds 90 --seed 7
        qgpt run ghz --system QuantumGPT-Orchestrator --model deepseek-chat

    The task_id is a stable hash of (circuit, backend, target, budget, seed,
    code_commit). Re-running with identical args reuses the same task_id —
    use `qgpt rerun <task_id>` to replay a previously saved spec.
    """
    from agent.run_persistence import (
        TaskSpec, compute_task_id, save_task_spec, save_result,
        append_trace, save_artifact, run_dir,
    )

    # Resolve circuit alias
    circuit_lower = circuit.lower().replace("-", "").replace("_", "")
    if circuit_lower in CIRCUIT_ALIASES:
        circuit_name, _ = CIRCUIT_ALIASES[circuit_lower]
    else:
        circuit_name = circuit

    spec = TaskSpec(
        circuit=circuit_name,
        backend=backend,
        target_fidelity=target,
        shots=shots,
        max_tool_calls=max_tool_calls,
        max_seconds=max_seconds,
        seed=seed,
        drift_profile=drift_profile,
        system=system,
        model=model,
        provider=provider,
    )
    task_id = compute_task_id(spec)
    save_task_spec(spec, task_id=task_id)
    d = run_dir(task_id)

    if not json_output:
        console.print(f"\n[bold cyan]qgpt run[/] {circuit_name}")
        console.print(f"  task_id : [bold]{task_id}[/]")
        console.print(f"  spec    : target={target}  budget={max_tool_calls} calls / {max_seconds}s")
        console.print(f"  system  : {system}  model={model}\n")

    t0 = time.time()
    try:
        if system == "QuantumGPT-Full":
            be = _make_backend(backend)
            from agent.factory import build_react_agent
            ag = build_react_agent(
                be,
                model=model,
                provider=provider,
                verbose=not json_output,
                target_fidelity=target,
                max_tool_calls=max_tool_calls,
                max_seconds=max_seconds,
                use_memory=False,
            )
            prompt = (f"Run circuit {circuit_name} on {backend}. "
                      f"Target fidelity {target}. Apply error mitigation if needed.")
            trace = ag.run(prompt)
            best_score = trace.best_fidelity
            final_answer = trace.final_answer
            tool_calls = trace.num_tool_calls
            elapsed = round(trace.elapsed_seconds, 3)
            cost_usd = float(trace.cost_summary.get("total_cost_usd") or 0.0)
            verifier_satisfied = (best_score is not None and best_score >= target)
            verifier_hallucinated = False
            termination_reason = ""
            plan_revision = 0
            verifier_reason = ""
        elif system in ("QuantumGPT-Orchestrator", "QuantumGPT-Orch-NoVerifier"):
            orch_task = _build_orchestrator_task(
                task_id=task_id,
                circuit_name=circuit_name,
                backend=backend,
                target=target,
                shots=shots,
                max_tool_calls=max_tool_calls,
                max_seconds=max_seconds,
                drift_profile=drift_profile,
                provider=provider,
                model=model,
                system=system,
            )
            orch_result = _run_orchestrator_graph(
                task=orch_task,
                task_id=task_id,
                system=system,
                save_artifact=save_artifact,
                append_trace=append_trace,
            )
            best_score = orch_result["best_score"]
            final_answer = orch_result["final_answer"]
            tool_calls = orch_result["tool_calls"]
            elapsed = orch_result["elapsed_seconds"] or round(time.time() - t0, 3)
            cost_usd = orch_result["total_cost_usd"]
            verifier_satisfied = orch_result["verifier_satisfied"]
            verifier_hallucinated = orch_result["verifier_hallucinated"]
            termination_reason = orch_result["termination_reason"]
            plan_revision = orch_result["plan_revision"]
            verifier_reason = orch_result["verifier_reason"]
        else:
            # Other systems require a richer task harness — just simulate for now
            from tools.quantum_tools import ToolExecutor
            be = _make_backend(backend)
            ex = ToolExecutor(be)
            sim_str = ex.execute("run_circuit", {
                "circuit_name": circuit_name, "shots": shots,
            })
            sim = json.loads(sim_str)
            best_score = sim.get("fidelity")
            final_answer = (f"[{system}] simple simulate fallback. "
                            f"Fidelity = {best_score:.4f} on {backend}.")
            tool_calls = 1
            elapsed = round(time.time() - t0, 3)
            cost_usd = 0.0
            verifier_satisfied = (best_score is not None and best_score >= target)
            verifier_hallucinated = False
            termination_reason = ""
            plan_revision = 0
            verifier_reason = ""

        result = {
            "task_id": task_id,
            "system": system,
            "circuit": circuit_name,
            "backend": backend,
            "target_fidelity": target,
            "best_score": best_score,
            "verifier_satisfied": verifier_satisfied,
            "verifier_hallucinated": verifier_hallucinated,
            "termination_reason": termination_reason,
            "plan_revision": plan_revision,
            "verifier_reason": verifier_reason,
            "final_answer": final_answer[:1000] if isinstance(final_answer, str) else str(final_answer)[:1000],
            "tool_calls": tool_calls,
            "elapsed_seconds": elapsed,
            "total_cost_usd": cost_usd,
        }
        save_result(task_id, result, status="completed")
    except Exception as exc:
        save_result(task_id, {
            "task_id": task_id,
            "error": str(exc),
            "elapsed_seconds": round(time.time() - t0, 3),
        }, status="failed")
        if json_output:
            click.echo(json.dumps({"task_id": task_id, "status": "failed",
                                   "error": str(exc)}, indent=2))
        else:
            console.print(f"[bold red]✗ task failed:[/] {exc}")
            console.print(f"  artifacts: {d}")
        return

    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        fid_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        fid_color = _fidelity_color(best_score) if best_score is not None else "white"
        sat_icon = "✓" if verifier_satisfied else "✗"
        sat_color = "green" if verifier_satisfied else "yellow"
        console.print()
        panel = Panel.fit(
            f"task_id  : [bold]{task_id}[/]\n"
            f"fidelity : [{fid_color}]{fid_str}[/]  (target {target:.3f})\n"
            f"satisfy  : [{sat_color}]{sat_icon} {verifier_satisfied}[/]\n"
            f"calls    : {tool_calls}  elapsed: {elapsed}s  cost: ${cost_usd:.4f}\n"
            f"artifacts: {d}",
            title="Run Result", border_style="cyan",
        )
        console.print(panel)
        console.print()


@main.command()
@click.option("--backend", "-b", default=None, help="Filter by backend.")
@click.option("--circuit", "-c", default=None, help="Filter by circuit.")
@click.option("--status", "-s", default=None,
              type=click.Choice(["running", "completed", "failed"]))
@click.option("--since", default=None,
              help="ISO date filter (e.g. 2026-05-01). Earlier runs hidden.")
@click.option("--limit", "-n", default=20, show_default=True, type=int)
@click.option("--json-output", "-j", is_flag=True)
def runs(backend, circuit, status, since, limit, json_output):
    """List task runs from ~/.quantumgpt/runs/, newest first.

    Examples:
        qgpt runs
        qgpt runs --backend FakeBrisbane --circuit ghz_5
        qgpt runs --since 2026-05-01 --status completed
    """
    from agent.run_persistence import list_runs

    since_epoch = None
    if since:
        try:
            since_epoch = time.mktime(time.strptime(since, "%Y-%m-%d"))
        except ValueError:
            click.echo(f"Bad --since format (expect YYYY-MM-DD): {since}",
                       err=True)
            sys.exit(1)

    rows = list_runs(backend=backend, circuit=circuit,
                     since_epoch=since_epoch, status=status, limit=limit)

    if json_output:
        click.echo(json.dumps(rows, indent=2, default=str))
        return

    if not rows:
        console.print("\n[dim]No runs found.[/]\n")
        return

    table = Table(title=f"QuantumGPT Runs ({len(rows)} shown)", show_lines=False)
    table.add_column("task_id", style="cyan", no_wrap=True)
    table.add_column("when", style="dim", no_wrap=True)
    table.add_column("status")
    table.add_column("system", style="dim")
    table.add_column("circuit", style="cyan")
    table.add_column("backend", style="dim")
    table.add_column("fid", justify="right")
    table.add_column("target", justify="right", style="dim")
    table.add_column("✓?", justify="center")
    table.add_column("reason", style="dim")
    table.add_column("rev", justify="right", style="dim")

    for r in rows:
        spec = r.get("spec") or {}
        summary = r.get("summary") or {}
        st = r.get("status", "?")
        st_color = {"completed": "green", "failed": "red",
                    "running": "yellow"}.get(st, "white")
        when = r.get("started_at_iso", "")[:19]
        if when:
            when = when.replace("T", " ")
        fid = summary.get("best_score")
        fid_str = f"{fid:.4f}" if isinstance(fid, (int, float)) else "—"
        fid_color = _fidelity_color(fid) if isinstance(fid, (int, float)) else "white"
        target = spec.get("target_fidelity")
        target_str = f"{target:.2f}" if isinstance(target, (int, float)) else ""
        sat = summary.get("verifier_satisfied")
        sat_str = "[green]✓[/]" if sat else ("[yellow]✗[/]" if sat is False else "[dim]—[/]")
        reason = summary.get("termination_reason") or ""
        plan_revision = summary.get("plan_revision")
        rev_str = str(plan_revision) if plan_revision is not None else ""
        table.add_row(
            r["task_id"],
            when,
            f"[{st_color}]{st}[/]",
            spec.get("system", "")[:18],
            spec.get("circuit", "")[:14],
            spec.get("backend", "")[:14],
            f"[{fid_color}]{fid_str}[/]",
            target_str,
            sat_str,
            reason[:14],
            rev_str,
        )

    console.print()
    console.print(table)
    console.print()


@main.command()
@click.argument("task_id")
@click.option("--json-output", "-j", is_flag=True)
@click.option("--show-trace", is_flag=True, help="Print all trace entries.")
def show(task_id, json_output, show_trace):
    """Show detailed info for a task_id.

    Example:
        qgpt show abc123def456
        qgpt show abc123 --show-trace
    """
    from agent.run_persistence import load_run
    try:
        run = load_run(task_id)
    except FileNotFoundError as exc:
        click.echo(f"Not found: {exc}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(json.dumps(run, indent=2, default=str))
        return

    meta = run.get("metadata") or {}
    spec = run.get("spec") or {}
    result = run.get("result") or {}
    trace = run.get("trace") or []

    console.print(f"\n[bold cyan]Task[/] [bold]{task_id}[/]\n")

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column("Field", style="dim")
    info.add_column("Value")
    info.add_row("status", meta.get("status", "?"))
    info.add_row("started", meta.get("started_at_iso", "")[:19].replace("T", " "))
    if "walltime_seconds" in meta:
        info.add_row("walltime", f"{meta['walltime_seconds']:.1f}s")
    info.add_row("commit", meta.get("code_commit", "—"))
    info.add_row("hostname", meta.get("hostname", "—"))
    info.add_row("circuit", spec.get("circuit", ""))
    info.add_row("backend", spec.get("backend", ""))
    info.add_row("system", spec.get("system", ""))
    info.add_row("target", str(spec.get("target_fidelity", "")))
    info.add_row("budget", f"{spec.get('max_tool_calls', '?')} calls / "
                          f"{spec.get('max_seconds', '?')}s")
    info.add_row("artifacts", str(run.get("run_dir", "")))
    available_artifacts = [
        name for name in ("pre_state", "post_state", "oracle_row", "trace")
        if run.get(name)
    ]
    if available_artifacts:
        info.add_row("available artifacts", ", ".join(available_artifacts))
    console.print(info)

    if result:
        console.print()
        fid = result.get("best_score")
        sat = result.get("verifier_satisfied")
        hall = result.get("verifier_hallucinated")
        sat_color = "green" if sat else "yellow"
        hall_color = "red" if hall else "dim"
        body_lines = []
        if fid is not None:
            body_lines.append(f"best fidelity     : {fid:.4f}")
        body_lines.append(f"satisfied         : [{sat_color}]{sat}[/]")
        body_lines.append(f"hallucinated      : [{hall_color}]{hall}[/]")
        if result.get("termination_reason"):
            body_lines.append(f"termination reason: {result['termination_reason']}")
        if result.get("verifier_reason"):
            body_lines.append(f"verifier reason   : {result['verifier_reason']}")
        if "plan_revision" in result:
            body_lines.append(f"plan revision     : {result['plan_revision']}")
        if "tool_calls" in result:
            body_lines.append(f"tool_calls        : {result['tool_calls']}")
        if "total_cost_usd" in result:
            body_lines.append(f"cost              : ${result['total_cost_usd']:.4f}")
        if result.get("error"):
            body_lines.append(f"[red]error[/]             : {result['error']}")
        if result.get("final_answer"):
            fa = result["final_answer"]
            body_lines.append(f"\n[dim]final_answer (truncated):[/]\n{fa[:500]}")
        console.print(Panel("\n".join(body_lines), title="Result",
                            border_style="cyan"))

    if show_trace and trace:
        console.print(f"\n[bold cyan]Trace[/] ({len(trace)} entries)\n")
        for i, e in enumerate(trace):
            console.print(f"  [dim]{i:3d}[/]  {json.dumps(e, default=str)[:120]}")
        console.print()


@main.command()
@click.argument("task_id")
@click.option("--json-output", "-j", is_flag=True)
def rerun(task_id, json_output):
    """Re-run a previously saved task by task_id (uses saved spec)."""
    from agent.run_persistence import load_spec, run_dir
    try:
        spec = load_spec(task_id)
    except FileNotFoundError as exc:
        click.echo(f"Not found: {exc}", err=True)
        sys.exit(1)

    if not json_output:
        console.print(f"\n[bold cyan]qgpt rerun[/] {task_id}")
        console.print(f"  reusing spec from: {run_dir(task_id)}\n")

    # Delegate back to `run` programmatically by invoking click's callback.
    ctx = click.get_current_context()
    ctx.invoke(
        run,
        circuit=spec.circuit,
        backend=spec.backend,
        target=spec.target_fidelity,
        shots=spec.shots,
        max_tool_calls=spec.max_tool_calls,
        max_seconds=spec.max_seconds,
        seed=spec.seed,
        drift_profile=spec.drift_profile,
        system=spec.system,
        model=spec.model,
        provider=spec.provider,
        json_output=json_output,
    )


@main.command(name="run-dir")
@click.argument("task_id")
def run_dir_cmd(task_id):
    """Print the absolute path to ~/.quantumgpt/runs/<task_id>/ (for scripting)."""
    from agent.run_persistence import run_dir as _rd
    click.echo(str(_rd(task_id)))


# ── records ──────────────────────────────────────────────────────


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
