"""Day 6 end-to-end demo: run the QuantumGPT agent on 3 tasks.

Tries OpenAI-compatible API first, falls back to rule-based mock.
"""
import sys, os, json, time
sys.path.insert(0, "/home/wangshuchang/quantumgpt")

from backends.fake_adapter import FakeBackendAdapter
from backends.synthetic_drift import SyntheticDriftBackend, SUDDEN_DEGRADATION
from agent.loop import QuantumAgent

OUT = "/home/wangshuchang/quantumgpt/demos/day6"
os.makedirs(OUT, exist_ok=True)

# API config — DeepSeek official API
API_KEY = "sk-dc663768963a4ef69b0ce99c5ac01786"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"


def make_agent(backend):
    """Create agent with DeepSeek API."""
    return QuantumAgent(
        backend, model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        provider="openai", verbose=True,
    )


all_results = []

# ═══════ Task 1: Health check + run a circuit ═══════
print("\n" + "=" * 70)
print("TASK 1: Basic health check + circuit run")
print("=" * 70)

backend1 = FakeBackendAdapter("FakeBrisbane")
agent1 = make_agent(backend1)
result1 = agent1.run(
    "Check the backend health. Then run the GHZ-5 circuit and report the fidelity. "
    "Also tell me which qubits 0-4 have the best T1."
)
all_results.append({
    "task": "Health check + GHZ-5 run",
    "backend": "FakeAdapter(FakeBrisbane)",
    "model": result1.model,
    "provider": agent1.provider,
    "final_answer": result1.final_answer,
    "tool_calls": result1.tool_calls_made,
    "tokens": result1.total_tokens,
    "elapsed": result1.elapsed_seconds,
})

# ═══════ Task 2: Diagnose a degraded backend ═══════
print("\n" + "=" * 70)
print("TASK 2: Diagnose a degraded backend")
print("=" * 70)

backend2 = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
backend2.set_time(8)
agent2 = make_agent(backend2)
result2 = agent2.run(
    "The device seems to be performing poorly. Check its health, "
    "run a benchmark circuit to measure fidelity, and diagnose what's wrong. "
    "Suggest what I should do."
)
all_results.append({
    "task": "Diagnose degraded backend",
    "backend": "SyntheticDrift(SUDDEN_DEGRADATION, t=8h)",
    "model": result2.model,
    "provider": agent2.provider,
    "final_answer": result2.final_answer,
    "tool_calls": result2.tool_calls_made,
    "tokens": result2.total_tokens,
    "elapsed": result2.elapsed_seconds,
})

# ═══════ Task 3: Compare circuits ═══════
print("\n" + "=" * 70)
print("TASK 3: Compare circuit performance")
print("=" * 70)

backend3 = FakeBackendAdapter("FakeBrisbane")
agent3 = make_agent(backend3)
result3 = agent3.run(
    "List the available benchmark circuits, then run GHZ-5 and DJ-5 (from MQTBench). "
    "Compare their fidelities and explain why one might be higher than the other."
)
all_results.append({
    "task": "Compare circuit performance",
    "backend": "FakeAdapter(FakeBrisbane)",
    "model": result3.model,
    "provider": agent3.provider,
    "final_answer": result3.final_answer,
    "tool_calls": result3.tool_calls_made,
    "tokens": result3.total_tokens,
    "elapsed": result3.elapsed_seconds,
})

# ═══════ Save results ═══════
with open(os.path.join(OUT, "agent_runs.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for i, r in enumerate(all_results, 1):
    print(f"\nTask {i}: {r['task']}")
    print(f"  Backend:    {r['backend']}")
    print(f"  Model:      {r['model']} ({r['provider']})")
    print(f"  Tool calls: {len(r['tool_calls'])}")
    print(f"  Tokens:     {r['tokens']}")
    print(f"  Time:       {r['elapsed']:.1f}s")
    print(f"  Answer preview: {r['final_answer'][:200]}...")

print(f"\nResults saved to {OUT}/agent_runs.json")
