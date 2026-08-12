"""Enhanced moderate-drift experiment with real LLM agents and real ZNE.

No mock executor and no hard-coded fidelity boost are used here. The adaptive
agent calls real tools; mitigation is ToolExecutor.apply_mitigation(), which
runs the repository's ZNE implementation against SyntheticDriftBackend and
Qiskit AerSimulator.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.synthetic_drift import DriftProfile, STABLE, SyntheticDriftBackend
from eval.drift_experiment import (
    DriftExperimentResult,
    _print_table,
    _save_trace,
    _trace_tool_names,
    _usage_metadata_from_traces,
    make_real_agent,
    write_paper_grade_results,
)
from tools.quantum_tools import ToolExecutor


MODERATE_DRIFT = DriftProfile(
    t1_drift=lambda t: 0.6 if t >= 5 else 1.0,
    gate_error_drift=lambda t: 2.5 if t >= 5 else 1.0,
    readout_drift=lambda t: 2.0 if t >= 5 else 1.0,
    drift_score_fn=lambda t: 0.6 if t >= 5 else 0.0,
)


ENHANCED_TASK = (
    "Run ghz_5. Inspect backend health and drift first. If the current fidelity "
    "is below 0.85, diagnose the cause and apply ZNE mitigation if appropriate. "
    "Report unmitigated and mitigated fidelity numbers from tool outputs."
)


def _extract_fids_from_trace_dict(trace_dict: dict[str, Any]) -> list[float]:
    fids: list[float] = []
    for step in trace_dict.get("steps", []):
        fid = step.get("fidelity_observed")
        if fid is not None:
            fids.append(float(fid))
    return fids


def run_adaptive_experiment(
    backend: SyntheticDriftBackend,
    use_drift_aware: bool = True,
    use_mitigation: bool = True,
    drift_at_step: int = 3,
    max_steps: int = 12,
    target_fidelity: float = 0.85,
    provider: str = "auto",
    model: str | None = None,
    verbose: bool = True,
    out_dir: str | Path = "eval/results/drift_real_enhanced",
) -> DriftExperimentResult:
    """Run moderate-drift adaptation with a real LLM ReActAgent.

    `use_mitigation=False` is an ablation expressed in the prompt: the LLM is
    asked to diagnose/re-run but not call mitigation. We still do not use any
    mock or hard-coded fidelity adjustment.
    """

    backend.set_time(0)
    backend.profile = STABLE
    agent = make_real_agent(
        backend,
        provider=provider,
        model=model,
        use_drift_aware=use_drift_aware,
        target_fidelity=target_fidelity,
        max_turns=9,
        max_tool_calls=14,
        verbose=verbose,
    )

    pre_trace = agent.run(ENHANCED_TASK + " This is the stable pre-drift reference run.")

    backend.set_time(6)
    backend.profile = MODERATE_DRIFT

    if use_mitigation:
        post_prompt = (
            ENHANCED_TASK
            + " Moderate drift may now be present. Use real mitigation tools if the data justify it."
        )
    else:
        post_prompt = (
            "Run ghz_5 after possible moderate drift. Inspect current backend health, diagnose, "
            "and rerun the circuit, but do not call apply_mitigation; this is a no-mitigation ablation."
        )
    post_trace = agent.run(post_prompt)

    pre_fids = [float(s.fidelity_observed) for s in pre_trace.steps if s.fidelity_observed is not None]
    post_fids = [float(s.fidelity_observed) for s in post_trace.steps if s.fidelity_observed is not None]
    fidelities = pre_fids + post_fids

    trace_dir = Path(out_dir) / (
        "full" if use_drift_aware and use_mitigation else "no_mit" if use_drift_aware else "baseline"
    )
    trace_files = [
        _save_trace(pre_trace, trace_dir, "pre_drift_trace"),
        _save_trace(post_trace, trace_dir, "post_drift_trace"),
    ]

    tool_calls = _trace_tool_names(pre_trace) + _trace_tool_names(post_trace)
    planner_replans = getattr(getattr(agent, "planner", None), "replan_count", 0)
    replan_triggered = bool(use_drift_aware and planner_replans > 0)
    replan_at_step = drift_at_step if replan_triggered else 0

    recovery_step = 0
    for i, fid in enumerate(fidelities, start=1):
        if i >= drift_at_step and fid >= target_fidelity:
            recovery_step = i
            break

    system_name = "QuantumGPT-Full" if use_drift_aware else "ReAct-Baseline"
    if use_drift_aware and not use_mitigation:
        system_name = "QuantumGPT-NoMit"

    return DriftExperimentResult(
        system=system_name,
        task="Run GHZ-5 under MODERATE_DRIFT",
        drift_profile="MODERATE_DRIFT",
        fidelities=fidelities,
        drift_injected_at=drift_at_step,
        recovery_step=recovery_step,
        final_fidelity=fidelities[-1] if fidelities else 0.0,
        total_steps=pre_trace.num_tool_calls + post_trace.num_tool_calls,
        replan_triggered=replan_triggered,
        replan_at_step=replan_at_step,
        provider=agent.provider,
        model=agent.model,
        tool_calls=tool_calls,
        trace_files=trace_files,
        metadata={
            "planner_replan_count": planner_replans,
            "mitigation_tool_used": "apply_mitigation" in tool_calls,
            "requested_mitigation": use_mitigation,
            **_usage_metadata_from_traces([pre_trace, post_trace]),
        },
    )


def run_static_moderate(
    backend: SyntheticDriftBackend,
    drift_at_step: int = 3,
    max_steps: int = 6,
) -> DriftExperimentResult:
    """Static moderate-drift pipeline with real ToolExecutor/AerSimulator."""

    backend.set_time(0)
    backend.profile = STABLE
    executor = ToolExecutor(backend)
    fidelities: list[float] = []

    json.loads(executor.execute("get_backend_health", {}))
    for i in range(max_steps - 1):
        if i == drift_at_step - 1:
            backend.set_time(6)
            backend.profile = MODERATE_DRIFT
        result = json.loads(executor.execute("run_circuit", {"circuit_name": "ghz_5", "shots": 1024}))
        if result.get("fidelity") is not None:
            fidelities.append(float(result["fidelity"]))

    return DriftExperimentResult(
        system="Static-Pipeline",
        task="Run GHZ-5 under MODERATE_DRIFT",
        drift_profile="MODERATE_DRIFT",
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


def main(
    provider: str = "auto",
    model: str | None = None,
    include_llm_baseline: bool = True,
    verbose: bool = True,
) -> list[DriftExperimentResult]:
    print("\n" + "=" * 76)
    print("P3-T2: Enhanced Moderate Drift Experiment (real LLM + real ZNE)")
    print("=" * 76)
    print("Backend: SyntheticDrift(FakeBrisbane)")
    print("Drift: MODERATE_DRIFT at simulated t=6h")
    print("Mitigation: real ZNE via ToolExecutor.apply_mitigation")
    print()

    results: list[DriftExperimentResult] = []

    print("Running: QuantumGPT Full (drift-aware + real mitigation)...")
    results.append(
        run_adaptive_experiment(
            SyntheticDriftBackend("FakeBrisbane"),
            use_drift_aware=True,
            use_mitigation=True,
            provider=provider,
            model=model,
            verbose=verbose,
        )
    )

    print("Running: QuantumGPT No-Mitigation...")
    results.append(
        run_adaptive_experiment(
            SyntheticDriftBackend("FakeBrisbane"),
            use_drift_aware=True,
            use_mitigation=False,
            provider=provider,
            model=model,
            verbose=verbose,
        )
    )

    if include_llm_baseline:
        print("Running: ReAct Baseline (real LLM, no drift monitor)...")
        results.append(
            run_adaptive_experiment(
                SyntheticDriftBackend("FakeBrisbane"),
                use_drift_aware=False,
                use_mitigation=True,
                provider=provider,
                model=model,
                verbose=verbose,
            )
        )

    print("Running: Static Pipeline...")
    results.append(run_static_moderate(SyntheticDriftBackend("FakeBrisbane")))

    _print_table(results)
    result_path = Path("eval/results/drift_real_enhanced/result_summary.jsonl")
    write_paper_grade_results(results, result_path, run_id_prefix="moderate_drift")
    print(f"\nPaper-grade result JSONL: {result_path}")
    print("\nTrace JSON files:")
    for r in results:
        for p in r.trace_files:
            print(f"  {r.system}: {p}")
    return results


if __name__ == "__main__":
    main()
