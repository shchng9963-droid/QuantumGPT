#!/usr/bin/env python3
"""Public MQTBench deterministic oracle/search benchmark.

This module is intentionally no-LLM. It enumerates simple deterministic actions
against public circuits, Qiskit fake backends, and controlled drift profiles to
estimate whether a target fidelity is reachable before spending API budget on
agent runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit import transpile as qiskit_transpile
from qiskit_aer import AerSimulator

from backends.synthetic_drift import (
    LINEAR_DECAY,
    MILD_GRADUAL,
    MULTI_SHOCK,
    STABLE,
    SUDDEN_DEGRADATION,
    DriftProfile,
    SyntheticDriftBackend,
)
from bench.circuits import get_benchmark, list_benchmarks
from bench.mqtbench import EXTENDED_MQTBENCH_SELECTION, get_mqtbench_circuits
from mitigation import run_zne


@dataclass(frozen=True)
class OracleAction:
    name: str
    kind: str
    shots: int
    opt_level: int | None = None
    noise_aware: bool = False


@dataclass
class OracleActionResult:
    action: str
    fidelity: float
    shots: int
    depth: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OracleTaskRow:
    task_id: str
    circuit: str
    circuit_family: str
    backend: str
    profile: str
    seed: int
    target_fidelity: float
    pre_fidelity: float
    first_post_drift_score: float
    best_post_drift_score: float
    final_score: float
    post_drift_improvement: float
    target_success: bool
    oracle_feasible: bool
    best_action: str
    execution_mode: str = "deterministic_oracle"
    use_mock: bool = False
    elapsed_seconds: float = 0.0
    actions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILE_REGISTRY: dict[str, tuple[DriftProfile, float]] = {
    "stable": (STABLE, 0.0),
    "moderate": (LINEAR_DECAY, 5.0),
    "linear_decay": (LINEAR_DECAY, 5.0),
    "severe_sudden": (SUDDEN_DEGRADATION, 5.0),
    "sudden": (SUDDEN_DEGRADATION, 5.0),
    # v2.5 Sprint A additions
    "mild_gradual": (MILD_GRADUAL, 5.0),
    "multi_shock": (MULTI_SHOCK, 8.0),
}


def build_action_grid(
    *,
    opt_levels: Iterable[int] = (0, 1, 2, 3),
    shots: Iterable[int] = (4096,),
    include_mitigation: bool = True,
) -> list[OracleAction]:
    """Return deterministic action candidates for oracle search."""

    actions: list[OracleAction] = []
    for s in shots:
        actions.append(OracleAction(name=f"raw_s{s}", kind="raw", shots=int(s)))
        for opt in opt_levels:
            actions.append(
                OracleAction(
                    name=f"transpile_o{int(opt)}_s{s}",
                    kind="transpile",
                    shots=int(s),
                    opt_level=int(opt),
                    noise_aware=False,
                )
            )
        if include_mitigation:
            actions.append(OracleAction(name=f"zne_s{s}", kind="zne", shots=int(s)))
    return actions


def resolve_circuit(name: str):
    hand = list_benchmarks()
    if name in hand:
        circ, _ = get_benchmark(name)
        return circ, "hand_written"
    # First try the default 5; fall back to the extended candidate pool so v2.5
    # Sprint A circuits (QAOA-6, RealAmpRandom-5, WState-5, HHL-3, Adder-6) are reachable.
    for selection in (None, EXTENDED_MQTBENCH_SELECTION):
        for mc in get_mqtbench_circuits(selection=selection):
            if name == mc.label:
                return mc.circuit, "mqtbench"
    raise ValueError(f"Unknown circuit {name!r}; use hand-written names or MQTBench labels.")


def evaluate_action(backend: SyntheticDriftBackend, circuit_name: str, action: OracleAction) -> OracleActionResult:
    """Evaluate one deterministic action and return observed fidelity."""

    circuit, family = resolve_circuit(circuit_name)
    if action.kind == "raw":
        result = backend.run(circuit, shots=action.shots)
        return OracleActionResult(
            action=action.name,
            fidelity=float(result.fidelity or 0.0),
            shots=action.shots,
            depth=result.metadata.get("transpiled_depth"),
            metadata={"kind": action.kind, "circuit_family": family, **result.metadata},
        )

    if action.kind == "transpile":
        target = AerSimulator()
        transpiled = qiskit_transpile(circuit, backend=target, optimization_level=action.opt_level or 0, seed_transpiler=0)
        result = backend.run(transpiled, shots=action.shots, optimization_level=action.opt_level or 0)
        return OracleActionResult(
            action=action.name,
            fidelity=float(result.fidelity or 0.0),
            shots=action.shots,
            depth=result.metadata.get("transpiled_depth"),
            metadata={
                "kind": action.kind,
                "opt_level": action.opt_level,
                "noise_aware": action.noise_aware,
                "circuit_family": family,
                "pretranspiled_depth": transpiled.depth(),
                **result.metadata,
            },
        )

    if action.kind == "zne":
        zne = run_zne(circuit, backend, shots=action.shots)
        fid = float(zne.get("mitigated_fidelity", 0.0) or 0.0)
        return OracleActionResult(
            action=action.name,
            fidelity=fid,
            shots=action.shots,
            depth=None,
            metadata={"kind": action.kind, "circuit_family": family, **zne},
        )

    raise ValueError(f"Unknown action kind: {action.kind}")


def summarize_oracle_task(
    *,
    task_id: str,
    circuit: str,
    circuit_family: str,
    backend: str,
    profile: str,
    seed: int,
    target_fidelity: float,
    pre_fidelity: float,
    post_actions: list[OracleActionResult],
    elapsed_seconds: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> OracleTaskRow:
    if not post_actions:
        raise ValueError("post_actions must not be empty")
    first = post_actions[0]
    best = max(post_actions, key=lambda r: r.fidelity)
    return OracleTaskRow(
        task_id=task_id,
        circuit=circuit,
        circuit_family=circuit_family,
        backend=backend,
        profile=profile,
        seed=seed,
        target_fidelity=float(target_fidelity),
        pre_fidelity=float(pre_fidelity),
        first_post_drift_score=float(first.fidelity),
        best_post_drift_score=float(best.fidelity),
        final_score=float(best.fidelity),
        post_drift_improvement=float(best.fidelity - first.fidelity),
        target_success=bool(best.fidelity >= target_fidelity),
        oracle_feasible=bool(best.fidelity >= target_fidelity),
        best_action=best.action,
        elapsed_seconds=elapsed_seconds,
        actions=[a.to_dict() for a in post_actions],
        metadata=metadata or {},
    )


def run_oracle_task(
    *,
    circuit_name: str,
    backend_name: str,
    profile_name: str,
    seed: int = 0,
    target_fidelity: float = 0.85,
    actions: list[OracleAction] | None = None,
) -> OracleTaskRow:
    t0 = time.time()
    if profile_name not in PROFILE_REGISTRY:
        raise ValueError(f"Unknown profile {profile_name!r}; available={sorted(PROFILE_REGISTRY)}")
    profile, post_time = PROFILE_REGISTRY[profile_name]
    circuit, family = resolve_circuit(circuit_name)
    actions = actions or build_action_grid()

    backend = SyntheticDriftBackend(backend_name, STABLE)
    backend.set_time(0.0)
    pre = backend.run(circuit, shots=actions[0].shots if actions else 4096)

    backend.profile = profile
    backend.set_time(post_time)
    post_results = [evaluate_action(backend, circuit_name, action) for action in actions]
    health = backend.get_health()
    return summarize_oracle_task(
        task_id=f"{backend_name}:{profile_name}:{circuit_name}:seed{seed}",
        circuit=circuit_name,
        circuit_family=family,
        backend=backend_name,
        profile=profile_name,
        seed=seed,
        target_fidelity=target_fidelity,
        pre_fidelity=float(pre.fidelity or 0.0),
        post_actions=post_results,
        elapsed_seconds=time.time() - t0,
        metadata={
            "post_time_hours": post_time,
            "drift_score": health.drift_score,
            "avg_t1_us": health.avg_t1_us,
            "avg_2q_error": health.avg_2q_error,
        },
    )


def make_summary(rows: list[OracleTaskRow]) -> dict[str, Any]:
    n = len(rows)
    feasible = sum(1 for r in rows if r.oracle_feasible)
    profiles: dict[str, dict[str, Any]] = {}
    for profile in sorted({r.profile for r in rows}):
        subset = [r for r in rows if r.profile == profile]
        profiles[profile] = {
            "n_tasks": len(subset),
            "n_oracle_feasible": sum(1 for r in subset if r.oracle_feasible),
            "oracle_feasible_rate": round(sum(1 for r in subset if r.oracle_feasible) / len(subset), 6) if subset else 0.0,
            "mean_best_post_drift_score": round(sum(r.best_post_drift_score for r in subset) / len(subset), 6) if subset else 0.0,
            "mean_post_drift_improvement": round(sum(r.post_drift_improvement for r in subset) / len(subset), 6) if subset else 0.0,
        }
    return {
        "n_tasks": n,
        "n_oracle_feasible": feasible,
        "oracle_feasible_rate": round(feasible / n, 6) if n else 0.0,
        "target_fidelity_values": sorted({r.target_fidelity for r in rows}),
        "profiles": profiles,
        "best_actions": sorted({r.best_action for r in rows}),
        "oracle_feasible_task_ids": [r.task_id for r in rows if r.oracle_feasible],
        "oracle_infeasible_task_ids": [r.task_id for r in rows if not r.oracle_feasible],
    }


def write_oracle_outputs(rows: list[OracleTaskRow], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "oracle_results.jsonl"
    summary_path = out / "oracle_summary.json"
    jsonl_path.write_text("\n".join(json.dumps(r.to_dict(), sort_keys=True) for r in rows) + ("\n" if rows else ""))
    summary_path.write_text(json.dumps(make_summary(rows), indent=2, sort_keys=True))
    return {"jsonl": str(jsonl_path), "summary": str(summary_path)}


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", default="ghz_5,qft_4,bv_5,GHZ-5,DJ-5")
    parser.add_argument("--backends", default="FakeBrisbane")
    parser.add_argument("--profiles", default="stable,moderate,severe_sudden")
    parser.add_argument("--shots", default="2048")
    parser.add_argument("--opt-levels", default="0,1,2,3")
    parser.add_argument("--target-fidelity", type=float, default=0.85)
    parser.add_argument("--no-mitigation", action="store_true")
    parser.add_argument("--out-dir", default="eval/results/public_mqtbench_oracle")
    args = parser.parse_args(argv)

    shots = [int(x) for x in parse_csv(args.shots)]
    opt_levels = [int(x) for x in parse_csv(args.opt_levels)]
    actions = build_action_grid(opt_levels=opt_levels, shots=shots, include_mitigation=not args.no_mitigation)

    rows: list[OracleTaskRow] = []
    for backend in parse_csv(args.backends):
        for profile in parse_csv(args.profiles):
            for circuit in parse_csv(args.circuits):
                print(f"[oracle] backend={backend} profile={profile} circuit={circuit} actions={len(actions)}", flush=True)
                rows.append(
                    run_oracle_task(
                        circuit_name=circuit,
                        backend_name=backend,
                        profile_name=profile,
                        target_fidelity=args.target_fidelity,
                        actions=actions,
                    )
                )
    paths = write_oracle_outputs(rows, args.out_dir)
    print(json.dumps({"outputs": paths, "summary": make_summary(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
