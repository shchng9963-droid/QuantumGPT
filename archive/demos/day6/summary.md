# Day 6: Agent Loop — "Prompt → Tools → Result" End-to-End

## Goal
Build the simplest possible agent loop: a while-loop that takes a natural language
prompt, calls quantum backend tools, and produces a structured analysis report.

## What Was Built

### 1. Tool Definitions (`tools/quantum_tools.py`)
5 tools with Anthropic-compatible JSON schema:

| Tool | Purpose |
|------|---------|
| `get_backend_health` | Quick device health summary (T1, T2, errors, drift) |
| `get_qubit_properties` | Detailed per-qubit T1/T2/gate errors |
| `run_circuit` | Run a benchmark circuit, return fidelity + counts |
| `list_benchmarks` | Show all available circuits (hand-written + MQTBench) |
| `diagnose_and_suggest` | Analyze health data, output severity + action items |

### 2. Tool Executor (`tools/quantum_tools.py :: ToolExecutor`)
- Dispatches tool calls to real backend functions
- Handles both hand-written circuits and MQTBench circuits
- Logs all calls with timing for observability
- Returns JSON strings ready for LLM consumption

### 3. Agent Loop (`agent/loop.py :: QuantumAgent`)
Three provider backends:
- **OpenAI-compatible** (DeepSeek, GPT, etc.): via `_run_openai` with function calling
- **Anthropic**: via `_run_api` with native tool_use
- **Mock**: Rule-based `RulePlanner` for offline testing

Successfully tested with **DeepSeek V4-Flash** (`deepseek-chat`) via official API.

Architecture:
```
User prompt
  → LLM (DeepSeek / Claude / RulePlanner)
  → tool_use / function_call blocks
  → ToolExecutor runs them against ShadowBackend
  → results fed back
  → repeat until final text answer (or max 15 turns)
```

### 4. End-to-End Demo (3 tasks, all via DeepSeek API)

**Task 1: Health Check + GHZ-5**
- 4 tool calls: health + qubit_props → run_circuit → diagnose
- GHZ-5 fidelity: **0.9336** on FakeBrisbane
- Identified qubit 3 as best T1 (393.6 μs)
- DeepSeek produced structured markdown report with tables
- 7,315 tokens, 12.6s

**Task 2: Diagnose Degraded Backend**
- 6 tool calls: health + benchmarks → qubit_props → ghz_5 + qft_4 → diagnose
- Drift score: **1.0** (critical), GHZ fidelity dropped to **0.6919**
- DeepSeek independently ran *two* circuits to compare, identified qubit 5 (50% readout error)
- Produced 5 actionable recommendations with severity ratings
- 11,580 tokens, 19.7s

**Task 3: Compare Circuits**
- 6 tool calls: list + health → qubit_props + GHZ-5 + DJ-5 → diagnose
- GHZ-5 fidelity: **0.9272** (depth 14) vs DJ-5: **0.9172** (depth 30)
- DeepSeek correctly attributed the 1% gap to 2x depth and 56% more gates
- Calculated expected noise floors from gate error rates
- 8,770 tokens, 20.2s

## Visualizations
- `agent_architecture.png/pdf` — Architecture diagram
- `tool_call_flow.png/pdf` — Per-task tool call sequences
- `fidelity_comparison.png/pdf` — Fidelity across all 4 circuit runs
- `healthy_vs_degraded.png/pdf` — T1/T2 and error rates comparison

## Key Observations
1. Agent loop works end-to-end with real LLM (DeepSeek V4-Flash) + 5 quantum tools
2. DeepSeek autonomously decides tool call order, runs extra circuits for comparison, produces structured reports
3. Total: 16 tool calls across 3 tasks, 27,665 tokens, ~53s wall time
4. Degraded backend correctly triggers critical diagnosis with actionable recommendations
5. Circuit depth correlates strongly with fidelity drop (GHZ depth 14 > DJ depth 30)
6. Three provider backends: OpenAI-compatible (tested), Anthropic (ready), Mock (fallback)

## Files Changed
```
agent/__init__.py          (new)
agent/loop.py              (new — 580 lines, supports OpenAI/Anthropic/Mock)
tools/__init__.py          (new)
tools/quantum_tools.py     (new — 250 lines)
demos/day6/run_agent_demo.py  (auto-fallback: API → mock)
demos/day6/gen_figures.py
demos/day6/agent_runs.json
demos/day6/*.png, *.pdf    (4 figure pairs)
```

## Next: Day 7
- Hook up W&B experiment tracking
- DuckDB for structured query over calibration/fidelity history
- Dashboard for drift monitoring over time
