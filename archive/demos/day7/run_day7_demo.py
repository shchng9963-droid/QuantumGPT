"""Day 7 demo: Agent runs with DuckDB persistence + W&B logging.

Runs 5 agent tasks across healthy and degraded backends,
persists everything to DuckDB, logs to W&B, then queries the DB.
"""
import sys, os, json, time
sys.path.insert(0, "/home/wangshuchang/quantumgpt")

from backends.fake_adapter import FakeBackendAdapter
from backends.synthetic_drift import SyntheticDriftBackend, SUDDEN_DEGRADATION, LINEAR_DECAY
from agent.loop import QuantumAgent
from data.store import DataStore

OUT = "/home/wangshuchang/quantumgpt/demos/day7"
DB_PATH = os.path.join(OUT, "quantumgpt.duckdb")
os.makedirs(OUT, exist_ok=True)

# API config
API_KEY = "sk-dc663768963a4ef69b0ce99c5ac01786"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# ─── W&B setup (offline mode if no key) ─────────────
try:
    import wandb
    # Use offline mode — doesn't need login
    os.environ["WANDB_MODE"] = "offline"
    wandb_run = wandb.init(
        project="quantumgpt",
        name="day7-duckdb-demo",
        dir=OUT,
        config={
            "model": MODEL,
            "provider": "deepseek",
            "backends": ["FakeBrisbane", "SyntheticDrift"],
        },
    )
    print(f"W&B initialized (offline mode) -> {OUT}/wandb/")
except Exception as e:
    print(f"W&B not available: {e}")
    wandb_run = None


def make_agent(backend, wandb_run=None):
    return QuantumAgent(
        backend, model=MODEL, api_key=API_KEY, base_url=BASE_URL,
        provider="openai", verbose=True,
        db_path=DB_PATH, wandb_run=wandb_run,
    )


# ═══════════════════════════════════════════════════
# Run 5 tasks to populate DuckDB
# ═══════════════════════════════════════════════════

tasks = [
    {
        "name": "Healthy health check",
        "backend": FakeBackendAdapter("FakeBrisbane"),
        "prompt": "Check the backend health and report a summary.",
    },
    {
        "name": "Run GHZ-5 on healthy backend",
        "backend": FakeBackendAdapter("FakeBrisbane"),
        "prompt": "Run the GHZ-5 circuit and report fidelity. Also check qubits 0-4.",
    },
    {
        "name": "Run DJ-5 + QFT-4 on healthy",
        "backend": FakeBackendAdapter("FakeBrisbane"),
        "prompt": "Run DJ-5 and qft_4 circuits. Report both fidelities.",
    },
    {
        "name": "Degraded backend (t=8h)",
        "backend": None,  # set below
        "prompt": "Check health, run GHZ-5, and diagnose what's wrong. Suggest fixes.",
    },
    {
        "name": "Slow drift (t=12h)",
        "backend": None,  # set below
        "prompt": "Check health and run a benchmark. Is the device still usable?",
    },
]

# Set up drift backends
degraded = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
degraded.set_time(8)
tasks[3]["backend"] = degraded

slow = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
slow.set_time(12)
tasks[4]["backend"] = slow

results = []
for i, task in enumerate(tasks):
    print(f"\n{'='*70}")
    print(f"TASK {i+1}/{len(tasks)}: {task['name']}")
    print(f"{'='*70}")

    agent = make_agent(task["backend"], wandb_run)
    result = agent.run(task["prompt"])

    results.append({
        "task": task["name"],
        "model": result.model,
        "provider": agent.provider,
        "tool_calls": len(result.tool_calls_made),
        "tokens": result.total_tokens,
        "elapsed": result.elapsed_seconds,
        "answer_preview": result.final_answer[:300],
    })

# ═══════════════════════════════════════════════════
# Query DuckDB to verify data
# ═══════════════════════════════════════════════════

print(f"\n{'='*70}")
print("DuckDB DATA VERIFICATION")
print(f"{'='*70}")

db = DataStore(DB_PATH)

print("\n--- Table Counts ---")
counts = db.table_counts()
for table, count in counts.items():
    print(f"  {table}: {count} rows")

print("\n--- Recent Circuit Runs ---")
runs = db.recent_runs(limit=10)
for r in runs:
    print(f"  {r['circuit_name']:10s} fidelity={r['fidelity']:.4f}  "
          f"depth={r['transpiled_depth']}  backend={r['backend']}")

print("\n--- Health History ---")
health = db.health_history(limit=10)
for h in health:
    print(f"  {h['backend']:30s}  T1={h['avg_t1_us']:.1f}μs  "
          f"drift={h['drift_score']}  2q_err={h['avg_2q_error']:.4f}")

print("\n--- Fidelity Trend (all circuits) ---")
trend = db.fidelity_trend(limit=10)
for t in trend:
    print(f"  {t['circuit_name']:10s} fidelity={t['fidelity']:.4f}  "
          f"depth={t['transpiled_depth']}  [{t['backend']}]")

print("\n--- Best Qubits (FakeBrisbane) ---")
best = db.best_qubits("FakeBrisbane", top_n=5)
for q in best:
    print(f"  Q{q['qubit']}: T1={q['t1_us']:.1f}μs  T2={q['t2_us']:.1f}μs  "
          f"readout_err={q['readout_error']:.4f}")

print("\n--- Drift Events ---")
drift_events = db.query("SELECT * FROM drift_events ORDER BY ts DESC LIMIT 5")
for d in drift_events:
    print(f"  severity={d['severity']:10s}  drift={d['drift_score']}  "
          f"backend={d['backend']}")

print("\n--- Agent Sessions ---")
sessions = db.session_summary()
for s in sessions:
    print(f"  model={s['model']:20s}  tools={s['num_tool_calls']}  "
          f"tokens={s['total_tokens']}  {s['elapsed_seconds']:.1f}s  "
          f"prompt: {s['prompt_preview'][:50]}...")

# Save summary
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump({"tasks": results, "db_counts": counts}, f, indent=2, default=str)

if wandb_run:
    # Log summary table
    wandb_run.log({"db_counts": counts})
    wandb_run.finish()
    print(f"\nW&B run saved to {OUT}/wandb/")

print(f"\nDuckDB saved to {DB_PATH}")
print(f"Results saved to {OUT}/results.json")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
total_tools = sum(r["tool_calls"] for r in results)
total_tokens = sum(r["tokens"] for r in results)
total_time = sum(r["elapsed"] for r in results)
print(f"  Tasks:      {len(results)}")
print(f"  Tool calls: {total_tools}")
print(f"  Tokens:     {total_tokens}")
print(f"  Time:       {total_time:.1f}s")
print(f"  DuckDB rows: {sum(counts.values())}")

db.close()
