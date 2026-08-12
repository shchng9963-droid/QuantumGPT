#!/usr/bin/env python3
"""B10 prod runner — paired evaluation of the LangGraph orchestrator.

Plan v2.5 §3.2 B10 deliverable.

Output schema is exactly the Sprint A `PublicAgentRunResult` so the
result_summary.jsonl can be merged with `eval/results/v3_main_merged/` for
direct paired stats (Wilcoxon, bootstrap CI, Pareto).

Two systems, both run with REAL DeepSeek:
  - QuantumGPT-Orchestrator         : full graph (planner+exec+verifier)
  - QuantumGPT-Orch-NoVerifier      : verifier replaced with passthrough stub

Per-task structure mirrors Sprint A:
  pre-drift run  → backend(STABLE, t=0)
  drift event    → backend.profile = profile, backend.set_time(post_time)
  post-drift run → graph.invoke(task) on the now-drifted backend

Each row gets flushed to result_summary.jsonl immediately after completion,
so the runner is restart-safe — re-running with the same out-dir simply
appends new rows (caller is responsible for de-duping if interrupted mid-row).

Quick start (smoke):
    set -a; source .env; set +a
    .venv/bin/python eval/run_orchestrator_b10.py \\
        --oracle-jsonl eval/results/v3_oracle_full/result_summary.jsonl \\
        --systems QuantumGPT-Orchestrator \\
        --circuits ghz_5,qaoa_4 --seeds 0 --profiles severe_sudden \\
        --out-dir eval/results/v3_orchestrator_smoke

Full B10 run (450 rows, ~$6-8, ~2h):
    .venv/bin/python eval/run_orchestrator_b10.py \\
        --oracle-jsonl eval/results/v3_oracle_full/result_summary.jsonl \\
        --systems QuantumGPT-Orchestrator,QuantumGPT-Orch-NoVerifier \\
        --seeds 0,1,2,3,4 \\
        --profiles severe_sudden,mild_gradual,multi_shock \\
        --out-dir eval/results/v3_orchestrator
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.orchestrator.executor_node import (
    default_agent_factory,
    make_executor_node,
)
from agent.orchestrator.graph import build_graph
from agent.orchestrator.state import make_task, new_orchestrator_state
from agent.run_persistence import (
    TaskSpec,
    append_trace,
    compute_task_id,
    run_dir,
    save_artifact,
    save_result,
    save_task_spec,
)
from backends.synthetic_drift import STABLE, SyntheticDriftBackend

# Reuse Sprint A primitives so the result_summary.jsonl schema is identical.
from eval.public_mqtbench_agent_eval import (
    PublicAgentRunResult,
    _parse_csv,
    _profile_and_time,
    build_agent_prompt,
    load_oracle_rows,
    write_public_agent_results,
)
from eval.public_mqtbench_oracle import PROFILE_REGISTRY


# ── ExecutorNode plumbing ─────────────────────────────────────────────────


def _backend_factory_from_instance(backend_instance):
    """Adapt a pre-built backend into an ExecutorNode backend_factory."""
    def _factory(_task: dict[str, Any]):
        return backend_instance
    return _factory


# ── single-task runner ────────────────────────────────────────────────────


def run_orchestrator_task(
    oracle_row: dict[str, Any],
    *,
    system: str,
    seed: int,
    out_dir: str | Path,
    provider: str = "deepseek",
    max_tool_calls: int = 12,
    max_seconds: float = 90.0,
    verbose: bool = False,
    use_stub_planner: bool = False,
) -> PublicAgentRunResult:
    """Run one paired (pre, post-drift) orchestrator task.

    The graph is invoked twice on the same backend instance so drift state
    propagates between invocations exactly as in the flat-ReAct pipeline.
    """
    circuit = str(oracle_row["circuit"])
    backend_name = str(oracle_row.get("backend", "FakeBrisbane"))
    profile_name = str(oracle_row.get("profile", "moderate"))
    task_seed = int(seed)
    target = float(oracle_row.get("target_fidelity", 0.85))
    profile, post_time = _profile_and_time(profile_name)

    backend = SyntheticDriftBackend(backend_name, STABLE)
    backend.set_time(0.0)
    backend.profile = STABLE

    use_verifier_stub = (system == "QuantumGPT-Orch-NoVerifier")
    executor_fn = make_executor_node(
        backend_factory=_backend_factory_from_instance(backend),
        agent_factory=default_agent_factory,
    )
    app = build_graph(
        executor_fn=executor_fn,
        use_stub_planner=use_stub_planner,
        use_stub_verifier=use_verifier_stub,
    )

    pre_prompt = (
        f"Run public benchmark circuit {circuit} on the current backend "
        "as a pre-drift reference. Report the observed fidelity number."
    )
    post_prompt = build_agent_prompt(oracle_row)

    def _task(prompt: str, task_type: str) -> dict[str, Any]:
        t = make_task(
            prompt,
            task_type=task_type,
            circuit=circuit,
            target_fidelity=target,
            backend_id=backend_name,
            max_usd=0.50,
            max_seconds=max_seconds,
            max_tool_calls=max_tool_calls,
        )
        t["provider"] = provider
        return t

    t0 = time.time()
    pre_state = new_orchestrator_state(_task(pre_prompt, "benchmark"))
    pre_out = app.invoke(pre_state)

    backend.profile = profile
    backend.set_time(post_time)

    post_state = new_orchestrator_state(_task(post_prompt, "benchmark"))
    post_out = app.invoke(post_state)
    elapsed = time.time() - t0

    pre_fids = _extract_fidelities_from_state(pre_out)
    post_fids = _extract_fidelities_from_state(post_out)
    pre_calls = _tool_calls_from_state(pre_out)
    post_calls = _tool_calls_from_state(post_out)

    # cost = sum across both invocations (planner + executor + verifier)
    cost_usd = (
        _cost_from_state(pre_out) + _cost_from_state(post_out)
    )
    tokens_in, tokens_out = _tokens_from_states([pre_out, post_out])

    # Save full graph traces for forensic value.
    result_dir = (Path(out_dir)
                  / system.replace(" ", "_").replace("/", "_")
                  / profile_name / circuit / f"seed{task_seed}")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "pre_graph_state.json").write_text(
        json.dumps(_dump_state_compact(pre_out), indent=2)
    )
    (result_dir / "post_graph_state.json").write_text(
        json.dumps(_dump_state_compact(post_out), indent=2)
    )
    trace_files = [
        str(result_dir / "pre_graph_state.json"),
        str(result_dir / "post_graph_state.json"),
    ]

    # Verifier-specific bookkeeping for reliability eval downstream.
    post_verification = post_out.get("verification") or {}

    # ── task_id persistence (P0-2) ────────────────────────────────────
    # Mirror the legacy <out_dir>/<system>/<profile>/<circuit>/seed{N}/ layout
    # AND additionally write a stable task_id-addressed copy under
    # ~/.quantumgpt/runs/<task_id>/. The legacy layout is preserved so that
    # the 1575 historical rows merge directly with the new ones.
    spec = TaskSpec(
        circuit=circuit,
        backend=backend_name,
        target_fidelity=target,
        shots=int(oracle_row.get("shots", 4096)),
        max_tool_calls=max_tool_calls,
        max_seconds=int(max_seconds),
        seed=task_seed,
        drift_profile=profile_name,
        system=system,
        model="deepseek-chat",
        provider=provider,
    )
    task_id = compute_task_id(spec)
    save_task_spec(spec, task_id=task_id)
    save_artifact(task_id, "pre_graph_state", _dump_state_compact(pre_out))
    save_artifact(task_id, "post_graph_state", _dump_state_compact(post_out))
    save_artifact(task_id, "oracle_row", oracle_row)

    metadata = {
        "execution_mode": "real",
        "use_mock": False,
        "seed": task_seed,
        "max_turns": max_tool_calls + 5,
        "max_tool_calls": max_tool_calls,
        "pre_fidelity_count": len(pre_fids),
        "post_fidelity_count": len(post_fids),
        "elapsed_seconds": elapsed,
        "oracle_row": oracle_row,
        "total_cost_usd": cost_usd,
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "verifier_satisfied": bool(post_verification.get("is_satisfied", False)),
        "verifier_hallucinated": bool(post_verification.get("hallucinated", False)),
        "verifier_reason": (post_verification.get("reason") or "")[:300],
        "plan_revision": int(post_out.get("plan_revision") or 0),
        "termination_reason": post_out.get("termination_reason", ""),
        "graph_done": bool(post_out.get("done", False)),
        "stub_verifier": use_verifier_stub,
        "task_id": task_id,
    }

    best_post = max(post_fids) if post_fids else None
    save_result(task_id, {
        "task_id": task_id,
        "system": system,
        "circuit": circuit,
        "backend": backend_name,
        "drift_profile": profile_name,
        "target_fidelity": target,
        "best_score": best_post,
        "best_pre_score": max(pre_fids) if pre_fids else None,
        "verifier_satisfied": metadata["verifier_satisfied"],
        "verifier_hallucinated": metadata["verifier_hallucinated"],
        "tool_calls": len(pre_calls) + len(post_calls),
        "elapsed_seconds": round(elapsed, 3),
        "total_cost_usd": cost_usd,
        "oracle_best": float(oracle_row.get("best_post_drift_score", 0.0)),
        "oracle_gap": (float(oracle_row.get("best_post_drift_score", 0.0))
                       - best_post) if best_post is not None else None,
        "termination_reason": post_out.get("termination_reason", ""),
        "trace_files": trace_files,
        "task_id_run_dir": str(run_dir(task_id)),
    }, status="completed")

    return PublicAgentRunResult(
        system=system,
        # Keep the legacy oracle-style task_id so result_summary.jsonl rows
        # still join with the historical 1575 paired rows. The stable
        # task_id (sha256 hash) is recorded in metadata['task_id'] and the
        # on-disk artifacts live under ~/.quantumgpt/runs/<task_id>/.
        task_id=str(oracle_row.get(
            "task_id",
            f"{backend_name}:{profile_name}:{circuit}:seed{task_seed}",
        )),
        circuit=circuit,
        backend=backend_name,
        drift_profile=profile_name,
        seed=task_seed,
        target_fidelity=target,
        oracle_best=float(oracle_row.get("best_post_drift_score", 0.0)),
        oracle_feasible=bool(oracle_row.get("oracle_feasible", False)),
        oracle_best_action=str(oracle_row.get("best_action", "unknown")),
        pre_fidelities=pre_fids,
        post_fidelities=post_fids,
        provider=provider,
        model="deepseek-chat",
        tool_calls=pre_calls + post_calls,
        trace_files=trace_files + [str(run_dir(task_id))],
        metadata=metadata,
    )


# ── trace helpers (orchestrator-state shape) ──────────────────────────────


def _extract_fidelities_from_state(state: dict[str, Any]) -> list[float]:
    """Pull every fidelity reading from every executor trace step."""
    fids: list[float] = []
    for trace in (state.get("executor_traces") or []):
        for step in (trace.get("steps") or []):
            f = step.get("fidelity_observed")
            if f is None:
                # also check best_fidelity-like fields
                f = step.get("fidelity")
            if isinstance(f, (int, float)) and f > 0:
                fids.append(float(f))
        # fallback: trace-level best_fidelity is the canonical end-of-run value
        bf = trace.get("best_fidelity")
        if isinstance(bf, (int, float)) and bf > 0 and not fids:
            fids.append(float(bf))
    # Also surface last_fidelity if executor_traces somehow missed it.
    lf = state.get("last_fidelity")
    if isinstance(lf, (int, float)) and lf > 0 and not fids:
        fids.append(float(lf))
    return fids


def _tool_calls_from_state(state: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for trace in (state.get("executor_traces") or []):
        for step in (trace.get("steps") or []):
            tool = step.get("action") or ""
            if tool:
                names.append(tool)
    return names


def _cost_from_state(state: dict[str, Any]) -> float:
    total = 0.0
    for trace in (state.get("executor_traces") or []):
        cs = trace.get("cost_summary") or {}
        total += float(cs.get("total_cost_usd") or 0.0)
    return total


def _tokens_from_states(states: list[dict[str, Any]]) -> tuple[int, int]:
    pin = 0
    pout = 0
    for st in states:
        for trace in (st.get("executor_traces") or []):
            cs = trace.get("cost_summary") or {}
            pin += int(cs.get("prompt_tokens") or 0)
            pout += int(cs.get("completion_tokens") or 0)
    return pin, pout


def _dump_state_compact(state: dict[str, Any]) -> dict[str, Any]:
    """Trim the state for on-disk storage — drop redundant raw fields."""
    keep = {}
    for k in ("done", "termination_reason", "plan", "plan_revision",
              "verification", "last_fidelity", "last_action",
              "budget_used", "history"):
        if k in state:
            keep[k] = state[k]
    # Trim traces — keep schema but drop verbose observations.
    traces = state.get("executor_traces") or []
    keep["executor_traces"] = [
        {
            "final_answer": (t.get("final_answer") or "")[:500],
            "best_fidelity": t.get("best_fidelity"),
            "num_tool_calls": t.get("num_tool_calls"),
            "elapsed_seconds": t.get("elapsed_seconds"),
            "cost_summary": t.get("cost_summary"),
            "steps": [
                {
                    "step_num": s.get("step_num"),
                    "action": s.get("action"),
                    "fidelity_observed": s.get("fidelity_observed"),
                }
                for s in (t.get("steps") or [])
                if s.get("action")
            ],
        }
        for t in traces
    ]
    return keep


# ── runner main ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-jsonl", required=True,
                        help="Pre-computed oracle JSONL (e.g. v3_oracle_full)")
    parser.add_argument(
        "--out-dir",
        default="eval/results/v3_orchestrator",
    )
    parser.add_argument(
        "--systems",
        default="QuantumGPT-Orchestrator,QuantumGPT-Orch-NoVerifier",
    )
    parser.add_argument("--profiles",
                        default="severe_sudden,mild_gradual,multi_shock")
    parser.add_argument("--circuits", default=None,
                        help="Comma-separated circuit names; default = all in oracle JSONL")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--include-infeasible", action="store_true")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--max-usd", type=float, default=15.0,
                        help="Soft cost cap; aborts cleanly when exceeded.")
    parser.add_argument("--soft-warn-usd", type=float, default=10.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-stub-planner", action="store_true",
                        help="Use offline rule-based planner (no LLM calls; "
                             "for smoke tests, not for production paired stats).")
    args = parser.parse_args(argv)

    if not args.use_stub_planner and not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY missing; run `set -a; source .env; set +a` first.",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "result_summary.jsonl"

    profiles = _parse_csv(args.profiles)
    circuits = _parse_csv(args.circuits)
    rows = load_oracle_rows(
        args.oracle_jsonl,
        profiles=profiles,
        circuits=circuits,
        only_feasible=not args.include_infeasible,
        max_tasks=args.max_tasks,
    )
    systems = sorted(_parse_csv(args.systems) or {"QuantumGPT-Orchestrator"})
    seeds = [int(s) for s in sorted(_parse_csv(args.seeds) or {"0"})]

    print(f"[orch-b10] {len(rows)} oracle rows × {len(systems)} systems × "
          f"{len(seeds)} seeds = {len(rows)*len(systems)*len(seeds)} tasks")
    print(f"[orch-b10] systems: {systems}")
    print(f"[orch-b10] seeds: {seeds}")
    print(f"[orch-b10] cost cap: ${args.max_usd:.2f} (warn at ${args.soft_warn_usd:.2f})")
    print(f"[orch-b10] writing to {summary_path}")

    results: list[PublicAgentRunResult] = []
    cumulative_cost = 0.0
    warned = False
    aborted = False
    n_total = len(rows) * len(systems) * len(seeds)
    n_done = 0
    t_start = time.time()

    for row in rows:
        for system in systems:
            for seed in seeds:
                n_done += 1
                if cumulative_cost >= args.max_usd:
                    print(f"[orch-b10] cost cap ${args.max_usd:.2f} hit "
                          f"(${cumulative_cost:.4f}); ABORT after {n_done-1}/{n_total}",
                          flush=True)
                    aborted = True
                    break
                if not warned and cumulative_cost >= args.soft_warn_usd:
                    print(f"[orch-b10] WARN: cumulative cost ${cumulative_cost:.4f} "
                          f"≥ soft warn ${args.soft_warn_usd:.2f}", flush=True)
                    warned = True

                t_task = time.time()
                try:
                    result = run_orchestrator_task(
                        row,
                        system=system,
                        seed=seed,
                        out_dir=out_dir,
                        provider=args.provider,
                        max_tool_calls=args.max_tool_calls,
                        max_seconds=args.max_seconds,
                        verbose=args.verbose,
                        use_stub_planner=args.use_stub_planner,
                    )
                except Exception as exc:
                    print(f"[orch-b10] FAIL system={system} seed={seed} "
                          f"task={row.get('task_id')}: {exc}", flush=True)
                    if args.verbose:
                        traceback.print_exc()
                    continue

                results.append(result)
                cumulative_cost += float(result.metadata.get("total_cost_usd") or 0.0)
                write_public_agent_results(results, summary_path)

                bf = result.best_post_fidelity
                bf_str = f"{bf:.4f}" if bf is not None else "n/a"
                tid = result.metadata.get("task_id", "")
                print(
                    f"[orch-b10] {n_done:>4d}/{n_total} "
                    f"system={system} seed={seed} "
                    f"profile={row.get('profile')} circuit={row.get('circuit')} "
                    f"task_id={tid} "
                    f"best_post={bf_str} cost=${result.metadata.get('total_cost_usd', 0):.4f} "
                    f"elapsed={time.time()-t_task:.1f}s "
                    f"cum_cost=${cumulative_cost:.4f}",
                    flush=True,
                )
            if aborted:
                break
        if aborted:
            break

    # ── final summary ─────────────────────────────────────────────────────
    summary = {
        "n_rows": len(results),
        "n_total_planned": n_total,
        "n_success": sum(1 for r in results if r.target_success),
        "success_rate": (
            sum(1 for r in results if r.target_success) / len(results)
            if results else 0.0
        ),
        "mean_best_post_drift_score": (
            sum((r.best_post_fidelity or 0.0) for r in results) / len(results)
            if results else 0.0
        ),
        "mean_oracle_gap": (
            sum((r.oracle_gap or 0.0) for r in results) / len(results)
            if results else 0.0
        ),
        "total_cost_usd": round(cumulative_cost, 6),
        "aborted_due_to_cap": aborted,
        "wallclock_seconds": round(time.time() - t_start, 1),
        "max_usd_cap": args.max_usd,
        "systems": systems,
        "profiles": sorted(profiles or set()),
        "seeds": seeds,
        # P0-2: list every stable task_id for reproducibility / paper appendix
        "task_ids": [r.metadata.get("task_id") for r in results
                     if r.metadata.get("task_id")],
    }
    (out_dir / "compact_summary.json").write_text(json.dumps(summary, indent=2))
    # Also write a plain task_ids.txt for grep / qgpt show <id> workflows.
    if summary["task_ids"]:
        (out_dir / "task_ids.txt").write_text(
            "\n".join(summary["task_ids"]) + "\n"
        )
    print()
    print(json.dumps(summary, indent=2))
    return 0 if not aborted else 1


if __name__ == "__main__":
    raise SystemExit(main())
