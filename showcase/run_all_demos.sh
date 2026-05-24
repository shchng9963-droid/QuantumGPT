#!/usr/bin/env bash
# QuantumGPT v2.1 Showcase Runner
# Runs all demos and collects outputs into showcase/
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         QuantumGPT v2.1 Showcase Runner                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Backend health check (single tool) ──────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 1: Backend Health Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
from tools.quantum_tools import ToolExecutor
from backends.fake_adapter import FakeBackendAdapter
import json
ex = ToolExecutor(FakeBackendAdapter('FakeBrisbane'))
r = json.loads(ex.execute('get_backend_health', {}))
print(f'  Backend: {r[\"backend\"]} ({r[\"num_qubits\"]} qubits)')
print(f'  T1={r[\"avg_t1_us\"]}μs  T2={r[\"avg_t2_us\"]}μs')
print(f'  1Q err={r[\"avg_1q_error\"]}  2Q err={r[\"avg_2q_error\"]}')
print(f'  Readout err={r[\"avg_readout_error\"]}')
"
echo ""

# ── 2. ReAct Agent (GHZ-5 + mitigation) ───────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 2: ReAct Agent (GHZ-5 + Error Mitigation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
from agent.react import ReActAgent
from backends.fake_adapter import FakeBackendAdapter
be = FakeBackendAdapter('FakeBrisbane')
agent = ReActAgent(be, provider='mock', target_fidelity=0.95, verbose=True)
trace = agent.run('Run GHZ-5 circuit. If fidelity is below 0.95, apply error mitigation.')
"
echo ""

# ── 3. Rabi tune-up demo ──────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 3: Rabi Oscillation (prompt → sim → fit → report)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 - <<'PY'
from demos.rabi_demo import run_rabi_demo
run_rabi_demo(save_dir='showcase/physics_demo_output')
PY
echo ""

# ── 4. Drift-aware demo ──────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 4: Drift-Aware Agent (detect drift → recheck → rerun)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 - <<'PY'
from demos.drift_demo import run_drift_demo
run_drift_demo(save_dir='showcase/drift_demo_output')
PY
echo ""

# ── 5. Memory-aware demo ─────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 5: Memory-Aware Agent (recall → diagnose → override)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 - <<'PY'
from demos.memory_demo import run_memory_demo
run_memory_demo(save_dir='showcase/memory_demo_output')
PY
echo ""

# ── 6. Eval (optional, quick) ────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 6: System Evaluation (1-seed quick run)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
from eval.run_eval import run_evaluation, print_table
results = run_evaluation(n_seeds=1, verbose=False)
print_table(results)
"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All demos completed! Outputs in showcase/                  ║"
echo "║                                                             ║"
echo "║  showcase/physics_demo_output/  ← Rabi                     ║"
echo "║  showcase/drift_demo_output/    ← Drift-aware              ║"
echo "║  showcase/memory_demo_output/   ← Memory-aware             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
