# DeepSeek vs Mock — Tier1 React-Drift Smoke
Date: 2026-05-24
Status: P0-3 closed. Real-provider path is live.

## Setup

- Provider: DeepSeek (deepseek-chat) via OpenAI-compatible API at https://api.deepseek.com
- Key: stored in `.env` (gitignored), not committed
- Tasks: tier1 first 3 — T1-01 Basic Health Check, T1-02 Qubit Selection, T1-03 Backend Comparison
- System: react_drift (DriftAwareRulePlanner wrapping ReActRulePlanner; in real-provider mode the LLM drives tool calls)
- Trace export: `--save-traces`, full observations included

Output files:

- `/home/wangshuchang/quantumgpt/reports/baseline_week1/deepseek_react_drift_tier1_smoke.json`
- `/home/wangshuchang/quantumgpt/reports/baseline_week1/mock_react_drift_tier1_smoke3.json`

## Headline numbers

| System    | Tasks | Success% | AvgFid | AvgCalls | AvgTime |
|-----------|-------|----------|--------|----------|---------|
| Mock      | 3     | 100%     | 0.9132 | 2.7      | 0.4 s   |
| DeepSeek  | 3     | 100%     | 0.9097 | 4.3      | 21.8 s  |

Both succeeded, fidelity is essentially equal. Difference is in *behaviour*, not score.

## Behavioural diff (the actually interesting result)

### T1-01 Basic Health Check
- Mock:     `get_backend_health, get_qubit_properties` (2 calls, generic stop)
- DeepSeek: `get_backend_health` only (1 call, stops as soon as the question is answered)

### T1-02 Qubit Selection
- Mock:     `get_backend_health, get_qubit_properties, run_circuit` (3 calls; runs a circuit despite the prompt only asking to select qubits)
- DeepSeek: `get_backend_health, list_benchmarks, get_qubit_properties×2, get_coupling_map` (5 calls; gathers connectivity + scans more qubits, computes ranking in the final answer)

### T1-03 Backend Comparison
- Mock:     `get_backend_health, get_qubit_properties, run_circuit` (3 calls; only one backend)
- DeepSeek: `compare_backends, get_backend_health, list_benchmarks, get_coupling_map, get_qubit_properties, run_circuit, retrieve_past_experiments` (7 calls; actually compares two backends, calls memory)

## What this proves

1. The real-LLM path (`_run_openai` w/ DeepSeek base_url) works end-to-end with tool calls, traces, fidelity recording, and trace export. No silent mock fallback.
2. The mock planner is rigid: it always does `health → properties → run`, regardless of whether the prompt asks for it. This is exactly what the AGENT_MATURITY_ROADMAP T2 refactor (planner abstraction + LLMPlanner) is meant to fix.
3. DeepSeek is more selective for narrow questions (T1-01) and more thorough for open-ended ones (T1-02, T1-03). It also actually uses `compare_backends` and `retrieve_past_experiments`, which the rule planner never selects.
4. Cost of being more thorough: 4.3 vs 2.7 tool calls, ~50× wall time (21.8 s vs 0.4 s). Acceptable for paper experiments; not acceptable for high-throughput benchmarking.

## What is NOT proven

- Single seed, 3 tasks. No statistical claim possible.
- DeepSeek `fidelity` field is null on T1-01 and T1-02 (the runner reads the last-circuit fidelity; when no circuit is run, this is null but the task is still graded as success). This is a runner bookkeeping quirk, not an LLM problem.
- Drift, mitigation, memory-with-low-fidelity-history, safety-block paths are not exercised by tier1 first-3.

## Cost

- DeepSeek tokens for 3 tasks: not yet logged structurally — runner does not capture `usage`. This is a known gap; add `total_tokens` aggregation to runner before scaling up.

## Next experiments unlocked by this milestone

- Run the same 3 tasks ×3 seeds → estimate variance.
- Run tier2 (drift) subset with DeepSeek → first datum on Contribution 2 (drift-aware replanning) under a real LLM.
- Run with `use_memory=True/False` on a tier1 + tier2 mix → first datum on Contribution 3 (memory ablation).
- Compare deepseek-chat vs deepseek-reasoner on T1-02-style ambiguous tasks.

## Decision

P0-3 is closed. Real-provider eval is live. The next blocker is no longer "we cannot call an LLM"; it is "the rule planner and LLM planner are tangled in 1500 lines of `react.py`". This is exactly the P1-4 refactor and motivates delegating it.
