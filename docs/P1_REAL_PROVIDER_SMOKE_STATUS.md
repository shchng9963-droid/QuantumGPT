# P1 Real-Provider Smoke Status

Date: 2026-05-23

## Goal

Start P1 by checking whether QuantumGPT can run a tiny API-backed benchmark smoke with saved traces, and verify that the runner does not silently fall back to the mock planner when a real provider is requested.

## Environment check

Command:

```bash
cd /home/wangshuchang/quantumgpt
python - <<'PY'
import importlib.util, os
keys = ['DEEPSEEK_API_KEY','OPENAI_API_KEY','ANTHROPIC_API_KEY']
print('sdk_openai', importlib.util.find_spec('openai') is not None)
print('sdk_anthropic', importlib.util.find_spec('anthropic') is not None)
for k in keys:
    v = os.environ.get(k)
    print(k, 'set' if v else 'missing')
PY
```

Observed:

- `sdk_openai`: True
- `sdk_anthropic`: True
- `DEEPSEEK_API_KEY`: missing
- `OPENAI_API_KEY`: missing
- `ANTHROPIC_API_KEY`: missing

Conclusion: SDK dependencies are installed, but no real-provider API key is available in this shell. A real API-backed smoke cannot run yet.

## Fallback guard check

Command:

```bash
cd /home/wangshuchang/quantumgpt
python benchmark/runner.py \
  --systems react_drift \
  --tiers 1 \
  --max-tasks 1 \
  --provider openai \
  --model gpt-4o-mini \
  --save-traces \
  --output reports/baseline_week1/openai_missing_key_guard_smoke.json \
  --verbose
```

Observed result:

```text
RuntimeError: System react_drift requested provider 'openai' but fell back to mock. Check API key, SDK dependency, and base URL before running real LLM evaluation.
```

Conclusion: the benchmark runner fails loudly when a real provider is requested but resolves to mock because credentials/config are missing. This protects benchmark results from silent mock contamination.

## Next command after credentials are available

After exporting an API key, run a tiny saved-trace smoke first:

```bash
cd /home/wangshuchang/quantumgpt
export OPENAI_API_KEY=...  # or configure DEEPSEEK_API_KEY / ANTHROPIC_API_KEY
python benchmark/runner.py \
  --systems react_drift \
  --tiers 1 \
  --max-tasks 1 \
  --provider openai \
  --model gpt-4o-mini \
  --save-traces \
  --output reports/baseline_week1/openai_react_drift_tier1_tiny_smoke.json \
  --verbose
```

Acceptance criteria:

1. process exits 0;
2. output JSON exists;
3. result row has `requested_provider=openai`;
4. result row has `resolved_provider != mock`;
5. trace exists because `--save-traces` was set;
6. trace has diagnostics, steps, state_summary, and artifacts;
7. manually inspect trace for malformed tool calls or max-turn exhaustion.
