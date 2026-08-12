"""Drift-aware replanning experiment using real LLM agents and AerSimulator.

This module intentionally has no synthetic executor. Agentic systems are run
through ReActAgent with a real LLM provider and ToolExecutor. Tool execution
therefore flows into SyntheticDriftBackend.run(), which builds a Qiskit
AerSimulator noise model from the current drifted calibration snapshot.

Comparison systems:
  1. QuantumGPT-Full: real LLM ReActAgent with drift monitoring enabled.
  2. QuantumGPT-No-Drift: same real LLM ReActAgent without drift monitoring.
  3. Static-Pipeline: fixed non-agentic ToolExecutor sequence on the same
     SyntheticDriftBackend/AerSimulator physics path.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.react import AgentTrace, ReActAgent
from backends.synthetic_drift import STABLE, SUDDEN_DEGRADATION, SyntheticDriftBackend
from eval.paper_grade.result_schema import (
    PaperGradeResultRow,
    assert_paper_grade_execution,
    write_result_jsonl,
)
from tools.quantum_tools import ToolExecutor


DEFAULT_TASK = (
    "Run ghz_5 on the current backend. First inspect current backend health and "
    "drift. If fidelity is below 0.85 or drift is detected, use the available "
    "tools to diagnose and adapt. Report concrete fidelity numbers."
)


@dataclass
class DriftExperimentResult:
    """Result of one real drift experiment run."""

    system: str
    task: str
    drift_profile: str
    fidelities: list[float]
    drift_injected_at: int
    recovery_step: int
    final_fidelity: float
    total_steps: int
    replan_triggered: bool
    replan_at_step: int
    provider: str = ""
    model: str = ""
    tool_calls: list[str] = field(default_factory=list)
    trace_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def recovered(self) -> bool:
        return self.recovery_step > 0

    @property
    def recovery_latency(self) -> int:
        if self.recovery_step > 0:
            return self.recovery_step - self.drift_injected_at
        return -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_real_llm_provider(provider: str = "auto", api_key: str | None = None) -> tuple[str, str]:
    """Resolve a concrete real LLM provider and API key.

    provider="auto" chooses an installed real provider key. It never returns
    "mock" and never permits ReActAgent to fall back to the rule planner.
    """

    provider = provider.lower()
    if provider == "auto":
        candidates = [
            ("deepseek", os.environ.get("DEEPSEEK_API_KEY")),
            ("openai", os.environ.get("OPENAI_API_KEY")),
            ("anthropic", os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        ]
        for name, key in candidates:
            if key:
                return name, key
        raise RuntimeError(
            "A real LLM API key is required for drift experiments "
            "(DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN); "
            "no rule-planner fallback is allowed."
        )

    return provider, require_real_llm_provider(provider, api_key=api_key)


def require_real_llm_provider(provider: str = "deepseek", api_key: str | None = None) -> str:
    """Return an API key for a real LLM provider or raise."""

    provider = provider.lower()
    env_by_provider = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    if provider not in env_by_provider:
        raise ValueError(f"Unsupported real LLM provider: {provider}")
    key = api_key or os.environ.get(env_by_provider[provider])
    if not key and provider == "anthropic":
        key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not key:
        raise RuntimeError(
            f"{env_by_provider[provider]} is required for real LLM drift experiments; "
            "no rule-planner fallback is allowed."
        )
    return key


def make_real_agent(
    backend: SyntheticDriftBackend,
    *,
    provider: str = "auto",
    model: str | None = None,
    use_drift_aware: bool = True,
    target_fidelity: float = 0.85,
    max_turns: int = 12,
    max_tool_calls: int = 24,
    verbose: bool = True,
    api_key: str | None = None,
) -> ReActAgent:
    """Construct a ReActAgent that is guaranteed to use a real LLM."""

    provider, key = resolve_real_llm_provider(provider, api_key=api_key)
    assert_paper_grade_execution(execution_mode="real", provider=provider, use_mock=False)
    if model is None:
        model = "deepseek-chat" if provider == "deepseek" else None
    kwargs: dict[str, Any] = {
        "backend": backend,
        "provider": provider,
        "api_key": key,
        "use_mock": False,
        "use_drift_aware": use_drift_aware,
        "target_fidelity": target_fidelity,
        "max_turns": max_turns,
        "max_tool_calls": max_tool_calls,
        "max_seconds": 240.0,
        "verbose": verbose,
        "use_memory": False,
    }
    if model:
        kwargs["model"] = model
    agent = ReActAgent(**kwargs)
    if agent.use_mock or agent.provider == "mock":
        raise RuntimeError("ReActAgent resolved to rule-planner mode; aborting real experiment.")
    return agent


def _extract_fidelities(trace: AgentTrace) -> list[float]:
    return [float(s.fidelity_observed) for s in trace.steps if s.fidelity_observed is not None]


def _trace_tool_names(trace: AgentTrace) -> list[str]:
    return [s.action for s in trace.steps if s.action]


def _usage_metadata_from_traces(traces: list[AgentTrace]) -> dict[str, Any]:
    """Aggregate token, latency, cost, and diagnostics from saved agent traces."""

    prompt_tokens = sum(int(getattr(t, "prompt_tokens_total", 0) or 0) for t in traces)
    completion_tokens = sum(int(getattr(t, "completion_tokens_total", 0) or 0) for t in traces)
    total_tokens = sum(int(getattr(t, "total_tokens", 0) or 0) for t in traces)
    elapsed_seconds = sum(float(getattr(t, "elapsed_seconds", 0.0) or 0.0) for t in traces)
    total_cost_usd = sum(float(t.cost_summary.get("total_cost_usd", 0.0)) for t in traces)
    api_latency_ms = sum(
        float(getattr(step, "api_latency_ms", 0.0) or 0.0)
        for t in traces
        for step in getattr(t, "steps", [])
    )
    drift_alert_count = sum(
        int(getattr(t.diagnostics, "drift_alert_count", 0) or 0) for t in traces
    )
    replan_triggered = any(
        bool(getattr(t.diagnostics, "replan_triggered", False)) for t in traces
    )
    drift_alert_steps_per_trace = [
        list(getattr(t.diagnostics, "drift_alert_steps", []) or []) for t in traces
    ]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost_usd, 6),
        "elapsed_seconds": elapsed_seconds,
        "api_latency_ms": round(api_latency_ms, 3),
        "trace_diagnostics": [t.diagnostics.to_dict() for t in traces],
        "trace_cost_summaries": [t.cost_summary for t in traces],
        "drift_alert_count": drift_alert_count,
        "replan_triggered": replan_triggered,
        "drift_alert_steps_per_trace": drift_alert_steps_per_trace,
    }


def _save_trace(trace: AgentTrace, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(trace.to_dict(include_observations=True), indent=2, default=str))
    return str(path)


def _detect_replan_from_trace(trace: AgentTrace) -> tuple[bool, int]:
    for i, step in enumerate(trace.steps, start=1):
        state = step.state_summary or {}
        drift = state.get("drift_state") if isinstance(state, dict) else None
        if isinstance(drift, dict) and drift.get("is_drifting"):
            return True, i
        thought = (step.thought or "").lower()
        if "drift" in thought and any(w in thought for w in ["re-run", "replan", "invalid", "updated"]):
            return True, i
    return False, 0


def run_llm_drift_experiment(
    backend: SyntheticDriftBackend,
    task: str = DEFAULT_TASK,
    drift_at_step: int = 3,
    use_drift_aware: bool = True,
    target_fidelity: float = 0.85,
    provider: str = "auto",
    model: str | None = None,
    verbose: bool = True,
    out_dir: str | Path = "eval/results/drift_real",
    max_turns: int = 12,
    max_tool_calls: int = 24,
) -> DriftExperimentResult:
    """Run a real LLM-driven pre/post drift experiment.

    The experiment uses one persistent agent so its drift monitor baseline is
    initialized on the stable backend before drift is injected. We run a stable
    pre-drift task, inject SUDDEN_DEGRADATION, then ask the same agent to run
    again while explicitly requiring it to inspect current drift before trusting
    earlier results. Tool calls remain LLM-selected.
    """

    backend.set_time(0)
    backend.profile = STABLE
    agent = make_real_agent(
        backend,
        provider=provider,
        model=model,
        use_drift_aware=use_drift_aware,
        target_fidelity=target_fidelity,
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        verbose=verbose,
    )

    pre_prompt = task + " This is the pre-drift reference run."
    pre_trace = agent.run(pre_prompt)

    backend.set_time(6)
    backend.profile = SUDDEN_DEGRADATION

    post_prompt = (
        task
        + " Drift may have occurred since the reference run. Before accepting old "
        + "results, inspect current backend health/drift and rerun or mitigate if needed."
    )
    post_trace = agent.run(post_prompt)

    pre_fids = _extract_fidelities(pre_trace)
    post_fids = _extract_fidelities(post_trace)
    fidelities = pre_fids + post_fids
    final_fidelity = fidelities[-1] if fidelities else 0.0

    trace_dir = Path(out_dir) / ("full" if use_drift_aware else "no_drift")
    trace_files = [
        _save_trace(pre_trace, trace_dir, "pre_drift_trace"),
        _save_trace(post_trace, trace_dir, "post_drift_trace"),
    ]

    trace_replan, trace_replan_step = _detect_replan_from_trace(post_trace)
    planner_replans = getattr(getattr(agent, "planner", None), "replan_count", 0)
    replan_triggered = bool(use_drift_aware and (trace_replan or planner_replans > 0))
    replan_at_step = drift_at_step if replan_triggered else 0

    recovery_step = 0
    post_start_step = len(pre_fids) + 1
    for i, fid in enumerate(post_fids, start=post_start_step):
        if fid >= target_fidelity:
            recovery_step = i
            break

    system = "QuantumGPT-Full" if use_drift_aware else "QuantumGPT-No-Drift"
    return DriftExperimentResult(
        system=system,
        task=task,
        drift_profile="SUDDEN_DEGRADATION",
        fidelities=fidelities,
        drift_injected_at=drift_at_step,
        recovery_step=recovery_step,
        final_fidelity=final_fidelity,
        total_steps=pre_trace.num_tool_calls + post_trace.num_tool_calls,
        replan_triggered=replan_triggered,
        replan_at_step=replan_at_step,
        provider=agent.provider,
        model=agent.model,
        tool_calls=_trace_tool_names(pre_trace) + _trace_tool_names(post_trace),
        trace_files=trace_files,
        metadata={
            "post_trace_replan_step": trace_replan_step,
            "planner_replan_count": planner_replans,
            "pre_fidelity_count": len(pre_fids),
            "post_fidelity_count": len(post_fids),
            "pre_final_answer_chars": len(pre_trace.final_answer or ""),
            "post_final_answer_chars": len(post_trace.final_answer or ""),
            **_usage_metadata_from_traces([pre_trace, post_trace]),
        },
    )


def run_drift_aware_experiment(
    backend: SyntheticDriftBackend,
    task: str = DEFAULT_TASK,
    drift_at_step: int = 3,
    max_steps: int = 12,
    use_drift_aware: bool = True,
    target_fidelity: float = 0.85,
    provider: str = "auto",
    model: str | None = None,
    verbose: bool = True,
) -> DriftExperimentResult:
    """Backward-compatible wrapper; now always uses a real LLM agent."""

    return run_llm_drift_experiment(
        backend=backend,
        task=task,
        drift_at_step=drift_at_step,
        use_drift_aware=use_drift_aware,
        target_fidelity=target_fidelity,
        provider=provider,
        model=model,
        verbose=verbose,
    )


def run_static_pipeline_experiment(
    backend: SyntheticDriftBackend,
    drift_at_step: int = 3,
    max_steps: int = 6,
) -> DriftExperimentResult:
    """Static fixed sequence using real ToolExecutor/AerSimulator physics."""

    backend.set_time(0)
    backend.profile = STABLE
    executor = ToolExecutor(backend)
    fidelities: list[float] = []

    json.loads(executor.execute("get_backend_health", {}))
    for i in range(max_steps - 1):
        if i == drift_at_step - 1:
            backend.set_time(6)
            backend.profile = SUDDEN_DEGRADATION
        result = json.loads(executor.execute("run_circuit", {"circuit_name": "ghz_5", "shots": 1024}))
        fid = result.get("fidelity")
        if fid is not None:
            fidelities.append(float(fid))

    return DriftExperimentResult(
        system="Static-Pipeline",
        task="Run GHZ-5",
        drift_profile="SUDDEN_DEGRADATION",
        fidelities=fidelities,
        drift_injected_at=drift_at_step,
        recovery_step=0,
        final_fidelity=fidelities[-1] if fidelities else 0.0,
        total_steps=max_steps,
        replan_triggered=False,
        replan_at_step=0,
        provider="none",
        model="static-tool-executor",
        tool_calls=[c["tool"] for c in executor.call_log],
        metadata={"executor": "ToolExecutor", "physics": "SyntheticDriftBackend/AerSimulator"},
    )


def _post_drift_metrics(
    result: DriftExperimentResult,
    *,
    target_fidelity: float = 0.85,
) -> dict[str, Any]:
    """Summarize post-drift recovery separately from final-only fidelity."""

    if "pre_fidelity_count" in result.metadata:
        start_idx = max(int(result.metadata.get("pre_fidelity_count", 0)), 0)
    else:
        start_idx = max(int(result.drift_injected_at) - 1, 0)
    post = result.fidelities[start_idx:]
    first_post = post[0] if post else None
    best_post = max(post) if post else None
    improvement = None
    if first_post is not None and best_post is not None:
        improvement = best_post - first_post
    return {
        "first_post_drift_score": first_post,
        "best_post_drift_score": best_post,
        "post_drift_improvement": improvement,
        "post_drift_target_success": bool(best_post is not None and best_post >= target_fidelity),
    }


def _paper_grade_row_from_result(
    result: DriftExperimentResult,
    *,
    run_id_prefix: str = "drift",
    target_fidelity: float = 0.85,
) -> PaperGradeResultRow:
    """Convert a drift result into the authoritative paper-grade JSONL schema."""

    use_mock = bool(result.metadata.get("use_mock", False))
    execution_mode = result.metadata.get("execution_mode")
    if execution_mode is None:
        execution_mode = "real" if result.provider not in {"none", "static"} else "simulator"
    trace_file = result.trace_files[-1] if result.trace_files else f"{result.system.lower()}:no-trace"
    seed = int(result.metadata.get("seed", 0))
    best_score = max(result.fidelities) if result.fidelities else None
    prompt_tokens = int(result.metadata.get("prompt_tokens", 0))
    completion_tokens = int(result.metadata.get("completion_tokens", 0))
    return PaperGradeResultRow(
        run_id=f"{run_id_prefix}:{result.system}:{result.drift_profile}:seed{seed}",
        system=result.system,
        task=result.task,
        circuit=str(result.metadata.get("circuit", "ghz_5")),
        drift_profile=result.drift_profile,
        seed=seed,
        execution_mode=str(execution_mode),
        provider=result.provider,
        model=result.model,
        use_mock=use_mock,
        trace_file=trace_file,
        target_fidelity=target_fidelity,
        final_score=result.final_fidelity,
        best_score=best_score,
        target_success=result.final_fidelity >= target_fidelity,
        tool_calls=len(result.tool_calls),
        max_tool_calls=int(result.metadata.get("max_tool_calls", result.total_steps)),
        elapsed_seconds=float(result.metadata.get("elapsed_seconds", 0.0)),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_cost_usd=float(result.metadata.get("total_cost_usd", 0.0)),
        diagnostics={
            "tool_calls": result.tool_calls,
            "trace_files": result.trace_files,
            "drift_injected_at": result.drift_injected_at,
            "recovery_step": result.recovery_step,
            "replan_triggered": result.replan_triggered,
            "replan_at_step": result.replan_at_step,
            **result.metadata,
        },
        metrics={
            "fidelities": result.fidelities,
            "recovered": result.recovered,
            "recovery_latency": result.recovery_latency,
            **_post_drift_metrics(result, target_fidelity=target_fidelity),
        },
        failure_reason=result.metadata.get("failure_reason"),
    )


def write_paper_grade_results(
    results: list[DriftExperimentResult],
    path: str | Path,
    *,
    run_id_prefix: str = "drift",
    target_fidelity: float = 0.85,
) -> None:
    """Persist drift results through the paper-grade schema/no-mock guard."""

    rows = [
        _paper_grade_row_from_result(
            r,
            run_id_prefix=run_id_prefix,
            target_fidelity=target_fidelity,
        )
        for r in results
    ]
    write_result_jsonl(path, rows)


def _print_table(results: list[DriftExperimentResult]) -> None:
    print("\n" + "-" * 92)
    print(f"{'System':<24} {'Provider':<10} {'Pre F':<8} {'Final F':<8} {'Replan':<10} {'Tools'}")
    print("-" * 92)
    for r in results:
        pre_f = r.fidelities[0] if r.fidelities else 0.0
        replan = f"step {r.replan_at_step}" if r.replan_triggered else "No"
        print(
            f"{r.system:<24} {r.provider:<10} {pre_f:<8.4f} {r.final_fidelity:<8.4f} "
            f"{replan:<10} {len(r.tool_calls)}"
        )
    print("-" * 92)


def run_full_comparison(
    provider: str = "auto",
    model: str | None = None,
    include_llm_baseline: bool = True,
    verbose: bool = True,
) -> list[DriftExperimentResult]:
    """Run the full comparison. Requires a real LLM API key."""

    print("\n" + "=" * 76)
    print("P3-T2: Real LLM + AerSimulator Drift-Aware Replanning Experiment")
    print("=" * 76)
    print("Backend: SyntheticDrift(FakeBrisbane)")
    print("Drift: SUDDEN_DEGRADATION at simulated t=6h")
    print("Agentic runs: real ReActAgent, use_mock=False")
    print()

    results: list[DriftExperimentResult] = []

    print("Running: QuantumGPT Full (real LLM, drift-aware)...")
    results.append(
        run_llm_drift_experiment(
            SyntheticDriftBackend("FakeBrisbane"),
            use_drift_aware=True,
            provider=provider,
            model=model,
            verbose=verbose,
        )
    )

    if include_llm_baseline:
        print("Running: QuantumGPT No-Drift (real LLM, no drift monitor)...")
        results.append(
            run_llm_drift_experiment(
                SyntheticDriftBackend("FakeBrisbane"),
                use_drift_aware=False,
                provider=provider,
                model=model,
                verbose=verbose,
            )
        )

    print("Running: Static Pipeline (real ToolExecutor/AerSimulator)...")
    results.append(run_static_pipeline_experiment(SyntheticDriftBackend("FakeBrisbane")))

    _print_table(results)
    result_path = Path("eval/results/drift_real/result_summary.jsonl")
    write_paper_grade_results(results, result_path, run_id_prefix="sudden_drift")
    print(f"\nPaper-grade result JSONL: {result_path}")
    print("\nTrace JSON files:")
    for r in results:
        for p in r.trace_files:
            print(f"  {r.system}: {p}")
    return results


if __name__ == "__main__":
    run_full_comparison()
