"""P1-6 — memory ablation in the mock loop.

Design
------
For each (circuit, seed) we run the agent N_TRIALS times consecutively on a
shared DuckDB store. The 'memory_on' arm keeps the store and feeds prior
low-fidelity records into the planner; the 'memory_off' arm wipes the store
between trials so every trial starts fresh.

The hypothesis is that, after the first low-fidelity trial, the
memory_on arm should:
  * trigger ``diagnose_and_suggest`` more often than memory_off,
  * reach the target fidelity (0.95) more often on trial >= 2,
  * spend a similar tool budget (so the win is from better planning,
    not from spending more).

Run this from the repo root:

    python experiments/p1_6_memory_ablation.py
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

from agent.react import ReActAgent
from backends.fake_adapter import FakeBackendAdapter


CIRCUITS = ["ghz_5", "qaoa_4", "vqe_4"]
SEEDS = list(range(10))
N_TRIALS = 3
TARGET_FIDELITY = 0.95
MAX_TOOL_CALLS = 12
PROMPT_TPL = "Run {circuit} on FakeBrisbane and report the fidelity."


@dataclass
class TrialResult:
    system: str       # 'memory_on' | 'memory_off'
    circuit: str
    seed: int
    trial: int        # 1..N_TRIALS
    fidelity: float
    success: bool     # fidelity >= TARGET_FIDELITY
    tool_calls: int
    used_diagnose: bool
    used_retrieve_memory: bool
    elapsed_s: float


def _new_db_path() -> str:
    return os.path.join(
        tempfile.gettempdir(), f"qgpt_p16_{uuid.uuid4().hex[:8]}.duckdb"
    )


def _drop_db(path: str) -> None:
    for ext in ("", ".wal"):
        try:
            os.unlink(path + ext)
        except FileNotFoundError:
            pass


def _run_one(
    *,
    system: str,
    circuit: str,
    seed: int,
    trial: int,
    db_path: str,
    use_memory: bool,
) -> TrialResult:
    backend = FakeBackendAdapter("FakeBrisbane")
    agent = ReActAgent(
        backend,
        provider="mock",
        verbose=False,
        target_fidelity=TARGET_FIDELITY,
        max_tool_calls=MAX_TOOL_CALLS,
        use_memory=use_memory,
        db_path=db_path if use_memory else None,
    )
    t0 = time.time()
    trace = agent.run(PROMPT_TPL.format(circuit=circuit))
    elapsed = time.time() - t0

    fidelity = float(trace.budget_summary.get("best_fidelity") or 0.0)
    tool_calls = sum(1 for s in trace.steps if s.action)
    actions = [s.action for s in trace.steps if s.action]
    return TrialResult(
        system=system,
        circuit=circuit,
        seed=seed,
        trial=trial,
        fidelity=fidelity,
        success=fidelity >= TARGET_FIDELITY,
        tool_calls=tool_calls,
        used_diagnose="diagnose_and_suggest" in actions,
        used_retrieve_memory="retrieve_past_experiments" in actions,
        elapsed_s=elapsed,
    )


def run_ablation() -> list[TrialResult]:
    results: list[TrialResult] = []

    for circuit in CIRCUITS:
        for seed in SEEDS:
            # --- memory_off: fresh DB per trial (or no DB at all) ---
            for trial in range(1, N_TRIALS + 1):
                results.append(
                    _run_one(
                        system="memory_off",
                        circuit=circuit,
                        seed=seed,
                        trial=trial,
                        db_path="",
                        use_memory=False,
                    )
                )

            # --- memory_on: shared DB across the trial sequence ---
            db_path = _new_db_path()
            try:
                for trial in range(1, N_TRIALS + 1):
                    results.append(
                        _run_one(
                            system="memory_on",
                            circuit=circuit,
                            seed=seed,
                            trial=trial,
                            db_path=db_path,
                            use_memory=True,
                        )
                    )
            finally:
                _drop_db(db_path)

    return results


def _agg(rs: list[TrialResult]) -> dict:
    if not rs:
        return {"n": 0}
    fids = [r.fidelity for r in rs]
    return {
        "n": len(rs),
        "fidelity_mean": round(statistics.mean(fids), 4),
        "fidelity_std": round(statistics.pstdev(fids), 4) if len(fids) > 1 else 0.0,
        "success_rate": round(sum(r.success for r in rs) / len(rs), 4),
        "tool_calls_mean": round(statistics.mean(r.tool_calls for r in rs), 3),
        "diagnose_rate": round(sum(r.used_diagnose for r in rs) / len(rs), 4),
    }


def summarise(results: list[TrialResult]) -> dict:
    summary: dict = {
        "config": {
            "circuits": CIRCUITS,
            "seeds": SEEDS,
            "n_trials": N_TRIALS,
            "target_fidelity": TARGET_FIDELITY,
            "max_tool_calls": MAX_TOOL_CALLS,
            "n_runs_total": len(results),
        },
        "by_system": {},
        "by_system_trial": {},
        "by_system_circuit_trial": {},
    }

    for system in ("memory_off", "memory_on"):
        rows = [r for r in results if r.system == system]
        summary["by_system"][system] = _agg(rows)
        for trial in range(1, N_TRIALS + 1):
            tag = f"{system}/trial{trial}"
            summary["by_system_trial"][tag] = _agg(
                [r for r in rows if r.trial == trial]
            )
            for circuit in CIRCUITS:
                key = f"{system}/{circuit}/trial{trial}"
                summary["by_system_circuit_trial"][key] = _agg(
                    [r for r in rows if r.trial == trial and r.circuit == circuit]
                )
    return summary


def main() -> None:
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_ablation()

    raw_path = out_dir / "p1_6_memory_ablation_raw.json"
    summary_path = out_dir / "p1_6_memory_ablation_summary.json"

    raw_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False)
    )
    summary = summarise(results)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")

    print("\n=== by_system ===")
    for k, v in summary["by_system"].items():
        print(f"  {k}: {v}")
    print("\n=== by_system_trial ===")
    for k, v in summary["by_system_trial"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
