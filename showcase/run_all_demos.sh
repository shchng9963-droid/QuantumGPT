#!/usr/bin/env bash
# QuantumGPT Phase 1-2 展示脚本
# 运行所有可展示的 demo 并收集输出
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         QuantumGPT Phase 1-2 Showcase Runner                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 1. 单工具演示
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 1: 后端健康检查 (单工具)"
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

# 2. ReAct Agent 演示
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 2: ReAct Agent (GHZ-5 + 缓解)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
from agent.react import ReActAgent
from backends.fake_adapter import FakeBackendAdapter
be = FakeBackendAdapter('FakeBrisbane')
agent = ReActAgent(be, provider='mock', target_fidelity=0.95, verbose=True)
trace = agent.run('Run GHZ-5 circuit. If fidelity is below 0.95, apply error mitigation.')
"
echo ""

# 3. Rabi Demo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 3: Rabi 端到端 (prompt → sim → fit → report)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 - <<'PY'
from demos.rabi_demo import run_rabi_demo
run_rabi_demo(save_dir='showcase/physics_demo_output')
PY
echo ""

# 4. 评测
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Demo 4: 7×4 系统评测 (简化版, 1 seed)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
from eval.run_eval import run_evaluation, print_table
results = run_evaluation(n_seeds=1, verbose=False)
print_table(results)
"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All demos completed! Check showcase/ for outputs.          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
