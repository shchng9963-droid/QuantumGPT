# P1-6 — Memory Ablation, Real Data

**Date:** 2026-05-24
**Branch:** `feat/v2.1-mainline` (commit pending)
**Scope:** Honest end-to-end ablation of the memory subsystem in the mock loop.
Goal was *"真正出数据"* — produce a result that actually distinguishes memory_on
from memory_off on a non-trivial metric.

## TL;DR

| arm | n | fidelity (mean ± σ) | success @0.95 | tool_calls | diagnose rate |
|---|---|---|---|---|---|
| memory_off | 90 | 0.9577 ± 0.0216 | 0.611 | 2.52 | **0.000** |
| memory_on  | 90 | 0.9587 ± 0.0217 | 0.600 | 3.26 | **0.667** |

Per-trial breakdown (3 circuits × 10 seeds × 3 sequential trials):

```
memory_off/trial1  fid=0.9573  succ=0.667  calls=2.47  diag=0.00
memory_off/trial2  fid=0.9579  succ=0.600  calls=2.53  diag=0.00
memory_off/trial3  fid=0.9579  succ=0.567  calls=2.57  diag=0.00
memory_on /trial1  fid=0.9587  succ=0.600  calls=2.60  diag=0.00   ← empty memory
memory_on /trial2  fid=0.9584  succ=0.567  calls=3.60  diag=1.00   ← memory now seeded
memory_on /trial3  fid=0.9589  succ=0.633  calls=3.57  diag=1.00
```

**The headline:** memory **does** now steer the planner (diagnose-call rate jumps
from 0% → 100% on every trial after a sub-target trial appears in the store),
but in the current mock loop that planning change does **not** translate into
better fidelity or success rate.

## What changed in this round

`agent/react.py::_low_fidelity_circuit_from_memory` was a latent dead branch.
Its regex required `circuit=<name>` and `fidelity=<x>` in the same line, but
`AgentMemory.get_context_summary` actually emits

```
[MEMORY] Relevant past experiments:
  1. Prior ghz_5 run had low fidelity (fidelity=0.62, tools=2, outcome=partial)
```

so the planner never matched, and `diagnose_and_suggest` never fired from
memory in any prior P0/P1 experiment. The fix:

* keep the legacy `circuit=<name>` path (covered by tests),
* add a fallback that scans the bullet line for any known circuit name
  (`ghz_*`, `qaoa_*`, `vqe_*`, …) and pairs it with the `fidelity=` value.

Sanity-check after the fix:

```
> seed memory with ghz_5 fidelity=0.62, then run the same prompt
tools used: get_backend_health, diagnose_and_suggest, run_circuit
best fid: 0.9331
```

Memory → planner → tool selection now actually fires. All 26 memory-related
tests still pass.

## Experimental design

`experiments/p1_6_memory_ablation.py`

* Backend: `FakeBackendAdapter("FakeBrisbane")`
* Provider: `mock`
* Circuits: `ghz_5`, `qaoa_4`, `vqe_4`
* Seeds: 0..9 (10)
* Trials: 3 sequential per (circuit, seed)
* Target fidelity: 0.95
* Tool budget: 12 calls
* Two arms:
  * **memory_off** — `use_memory=False`, every trial fresh.
  * **memory_on**  — shared DuckDB store across the 3 trials in a sequence;
    new DB per (circuit, seed) sequence so seeds stay independent.
* 180 runs total (~85 min wall time, single-threaded mock + Aer noise sim).

Outputs:
* `experiments/results/p1_6_memory_ablation_raw.json` (180 rows)
* `experiments/results/p1_6_memory_ablation_summary.json` (aggregates)

## What this tells us about the paper claim

1. **The wiring claim is now defensible.** The memory subsystem demonstrably
   changes the agent's plan: 0/90 vs 60/90 invocations of
   `diagnose_and_suggest`. Trial 1 of memory_on matches memory_off exactly
   (0% diagnose) because the store is empty — clean negative control.
2. **The outcome claim is not.** Memory_on does not improve fidelity
   (Δ ≈ +0.001, well inside one σ ≈ 0.022) or success rate (-0.011).
   It costs ~+0.7 tool calls per run.
3. **Why?** The mock backend draws fresh i.i.d. noise per trial, and
   `diagnose_and_suggest` just emits an advisory string — it does not actually
   change the circuit, transpilation, or backend selection before the next
   `run_circuit`. So the planner *thinks* about the issue but the run that
   follows is statistically identical.

## Implications / next moves

The honest framing for the paper is **"memory steers planning; closing the
loop to outcomes is future work"** — overclaiming an outcome win would not
survive review. To get an actual outcome win, one of these is needed:

* **(a) Make `diagnose_and_suggest` actionable.** Have it return a concrete
  override (e.g. boost shots, switch optimization_level, pick a different
  layout) that the next `run_circuit` consumes. This is a small wiring
  change in `tools/quantum_tools.py`.
* **(b) Drift scenario.** Use a backend with state — calibration drifts
  over trials — so prior knowledge of "this backend is degrading" can route
  to a healthier alternative. The `react_drift` arm already exists; this
  P1-6 setup deliberately holds drift fixed to isolate the memory effect.
* **(c) Real-provider arm.** With a non-mock LLM the policy can read the
  injected memory and choose richer interventions; the mock planner is
  rule-based and only knows the diagnose detour.

I would prioritize (a): it's a 1-day change and turns memory from
"observable" into "causal" without needing new infrastructure.

## Files touched

* `agent/react.py` — fixed `_low_fidelity_circuit_from_memory` parser
  (handles both legacy `circuit=` format and the bullet format actually
  produced by `AgentMemory.get_context_summary`).
* `experiments/__init__.py` — created (so `python -m experiments.…` works).
* `experiments/p1_6_memory_ablation.py` — the harness.
* `experiments/results/p1_6_memory_ablation_{raw,summary}.json` — data.

## Verification

* Targeted unit tests pass: `tests/test_memory_integration.py`,
  `tests/unit/test_agent_state.py`,
  `tests/unit/test_agent_state_integration.py`,
  `tests/unit/test_agent_trace_diagnostics.py` → 26 passed.
* Sanity probe: seeding a low-fidelity ghz_5 record now triggers
  `diagnose_and_suggest` before `run_circuit` (verified manually).
* End-to-end: 180-run sweep completed cleanly; per-trial breakdown matches
  the expected pattern (trial 1 of memory_on identical to memory_off,
  trial ≥ 2 shows 100% diagnose rate).

## Caveats

* Mock-only. No real-provider numbers in this round.
* Sample size n=30 per (system, trial) cell — fidelity differences below
  ~0.01 are not separable from noise; the success-rate flip
  (memory_off trial1=0.667 vs memory_on trial1=0.600) is **not**
  statistically significant.
* The diagnose tool is currently advisory-only; this report explicitly
  flags that as the bottleneck rather than burying it.
