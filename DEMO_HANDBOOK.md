# QuantumGPT — 完整演示手册

> 所有命令均在 `/home/wangshuchang/quantumgpt` 目录下执行。

---

## 一、CLI 工具 (`qgpt`)

### 1.1 查看帮助

```bash
qgpt --help
```

**预期效果**：显示 6 个子命令：`simulate`, `health`, `list`, `diagnose`, `agent`, `records`。

---

### 1.2 查看后端健康状态

```bash
qgpt health
qgpt health -b FakeSherbrooke
qgpt health -b FakeBrisbane --json
```

**预期效果**：以表格展示 T1/T2、1Q/2Q 错误率、读出错误率、漂移分数。JSON 模式输出原始数据。

---

### 1.3 列出可用基准电路

```bash
qgpt list
```

**预期效果**：分两组列出：手写电路（ghz_5, qft_4, bv_5, vqe_4, qaoa_4）和 MQTBench 电路（DJ-5, GHZ-5, GraphState-5, QFTent-5, VQE_SU2-4），含描述、量子比特数、深度。

---

### 1.4 运行量子电路仿真

```bash
qgpt simulate ghz --shots 4096
qgpt simulate qft -b FakeSherbrooke -s 2048
qgpt simulate DJ-5 --json
```

**预期效果**：输出电路名称、fidelity（如 0.92）、转译深度、top-5 测量结果计数。

---

### 1.5 诊断与建议

```bash
qgpt diagnose
qgpt diagnose -b FakeSherbrooke
```

**预期效果**：分析后端健康状态，给出分级建议（nominal / warning / critical），建议包括是否需要重校准、错误缓解策略。

---

### 1.6 AI Agent 对话

```bash
qgpt agent "Check health and run GHZ-5 circuit"
qgpt agent "Diagnose why fidelity is low" --model rule-planner-v1
```

**预期效果**：Agent 自动调用工具链（health → simulate → diagnose），展示推理过程和最终建议。`rule-planner-v1` 为离线规则引擎，无需 API key。

---

### 1.7 实验记录查询

```bash
qgpt records
qgpt records -c ghz_5
qgpt records --stats
```

**预期效果**：从 DuckDB 读取历史实验记录，可按电路/后端过滤，`--stats` 显示聚合统计（总次数、平均 fidelity、成功率）。

---

## 二、感知工具（9 个 Agent 工具）

> 以下通过 Python 演示，也可通过 `qgpt agent` 间接调用。

### 2.1 后端健康 — get_backend_health

```bash
python3 -c "
from backends.fake_adapter import FakeBackendAdapter
from tools.quantum_tools import ToolExecutor
import json

ex = ToolExecutor(FakeBackendAdapter('FakeBrisbane'))
print(json.loads(ex.execute('get_backend_health', {})))
"
```

**预期效果**：返回 `avg_t1_us`, `avg_2q_error`, `drift_score` 等 8 个字段。

---

### 2.2 量子比特属性 — get_qubit_properties

```bash
python3 -c "
from backends.fake_adapter import FakeBackendAdapter
from tools.quantum_tools import ToolExecutor
import json

ex = ToolExecutor(FakeBackendAdapter('FakeBrisbane'))
r = json.loads(ex.execute('get_qubit_properties', {'qubits': [0, 1, 2, 3, 4]}))
for q in r['qubits']:
    print(f'  Q{q[\"qubit\"]}: T1={q[\"t1_us\"]}μs T2={q[\"t2_us\"]}μs RE={q[\"readout_error\"]}')
"
```

**预期效果**：显示每个量子比特的 T1、T2、读出错误率、门错误率。

---

### 2.3 耦合图 — get_coupling_map

```bash
python3 -c "
from backends.fake_adapter import FakeBackendAdapter
from tools.quantum_tools import ToolExecutor
import json

ex = ToolExecutor(FakeBackendAdapter('FakeBrisbane'))
r = json.loads(ex.execute('get_coupling_map', {}))
print(f'Backend: {r[\"backend\"]}, Qubits: {r[\"num_qubits\"]}, Edges: {r[\"num_edges\"]}, Avg degree: {r[\"avg_degree\"]}')
"
```

**预期效果**：FakeBrisbane 显示 127 量子比特、144 条边、平均度 2.27。

---

### 2.4 漂移检测 — detect_drift

```bash
python3 -c "
from backends.synthetic_drift import SyntheticDriftBackend, SUDDEN_DEGRADATION
from tools.quantum_tools import ToolExecutor
import json

be = SyntheticDriftBackend(profile=SUDDEN_DEGRADATION)
be.set_time(12.0)
ex = ToolExecutor(be)
r = json.loads(ex.execute('detect_drift', {'window_hours': 24, 'step_hours': 1.0}))
print(f'Algorithm: {r[\"method\"]}')
print(f'Changepoints: {len(r[\"changepoints\"])}')
for cp in r['changepoints']:
    print(f'  t={cp[\"time_hours\"]}h severity={cp[\"severity\"]} {cp[\"direction\"]}')
print(f'Recalibrate: {r[\"recalibrate_recommended\"]}')
"
```

**预期效果**：PELT 检测到 1 个 changepoint（~5h，severity ~0.67，degradation），建议重校准。

---

### 2.5 校准年龄 — get_calibration_age

```bash
python3 -c "
from backends.fake_adapter import FakeBackendAdapter
from tools.quantum_tools import ToolExecutor
import json

ex = ToolExecutor(FakeBackendAdapter('FakeBrisbane'))
r = json.loads(ex.execute('get_calibration_age', {}))
print(f'Age: {r[\"calibration_age_minutes\"]}min, Stale: {r[\"is_stale\"]}')
print(f'Recommendation: {r[\"recommendation\"]}')
"
```

**预期效果**：返回校准年龄、是否过期（>60min）、建议文本。

---

### 2.6 后端比较 — compare_backends

```bash
python3 -c "
from backends.fake_adapter import FakeBackendAdapter
from tools.quantum_tools import ToolExecutor
import json

ex = ToolExecutor(FakeBackendAdapter('FakeBrisbane'))
r = json.loads(ex.execute('compare_backends', {
    'candidates': ['FakeBrisbane', 'FakeSherbrooke', 'FakeKyoto'],
    'circuit_name': 'ghz_5'
}))
for e in r['ranking']:
    fid = e.get('fidelity', 'N/A')
    print(f'  #{e[\"rank\"]} {e[\"backend\"]}: score={e[\"composite_score\"]}, fidelity={fid}, T1={e[\"avg_t1_us\"]}μs')
print(f'Recommended: {r[\"recommended\"]}')
"
```

**预期效果**：3 个后端按综合评分排序，含 fidelity（实际跑电路），推荐最优后端。

---

## 三、漂移检测评测

### 3.1 快速评测

```bash
python -m detection.evaluate --quick
```

**预期效果**：6 个场景、8 个标注点，PELT F1 ~0.84，约 6 秒。

---

### 3.2 完整评测

```bash
python -m detection.evaluate --output detection/results_new.json
```

**预期效果**：80 场景、115 标注 CP，约 8 分钟。结果：

| 方法 | Precision | Recall | F1 |
|------|-----------|--------|----|
| PELT | 0.907 | 0.930 | **0.918** |
| BOCPD | 0.202 | 0.678 | 0.311 |
| Ensemble | 0.907 | 0.930 | 0.918 |

详细报告见 `detection/ACCURACY_REPORT.md`。

---

## 四、Operator Console（终端仪表盘）

### 4.1 一周历史漂移回放

```bash
python -m console.dashboard --profile week_realistic --hours 168 --step 1.0 --speed 200
```

**预期效果**：Rich 终端全屏 dashboard，实时显示：
- 当前指标（T1/T2/错误率/漂移分数）+ 颜色状态灯
- 历史 sparkline 曲线
- 变化点列表（约 3 个 CP：渐变退化 + day5 突变）
- 进度条（168h 回放）

按 `Ctrl+C` 退出后显示完整 DriftReport。

---

### 4.2 不同漂移模式

```bash
# 稳定
python -m console.dashboard --profile stable --hours 24 --speed 500

# 突变
python -m console.dashboard --profile sudden --hours 24 --speed 500

# 线性衰退
python -m console.dashboard --profile linear --hours 48 --speed 300

# 周期性
python -m console.dashboard --profile periodic --hours 48 --speed 300
```

---

## 五、Rabi 脉冲仿真（Qiskit Dynamics）

### 5.1 Jupyter Notebook

```bash
jupyter notebook notebooks/rabi_tutorial.ipynb
```

**预期效果**：5 节交互式 notebook：
1. 单 Rabi 振荡（P(|1⟩) 输出）
2. 方波扫描（50 点 Rabi 曲线 + π脉冲标记）
3. 方波 vs 高斯对比
4. π脉冲精确估计（粗扫 + 细扫）
5. 脉宽依赖（50/100/200/400 ns 对比）

生成 4 张图：`rabi_square.png`, `rabi_comparison.png`, `rabi_pi_estimation.png`, `rabi_durations.png`

---

### 5.2 命令行快速验证

```bash
python3 -c "
from dynamics.rabi import RabiConfig, sweep_rabi, estimate_pi_pulse

cfg = RabiConfig(qubit_freq_ghz=5.0, amp_range=(0.0, 0.08), n_amps=30,
                 pulse_duration_ns=100, dt_ns=0.5, pulse_shape='square')
s = sweep_rabi(cfg)
print(f'Rabi sweep: {len(s.amplitudes)} points')
print(f'P(|1>) range: [{s.final_populations.min():.3f}, {s.final_populations.max():.3f}]')
print(f'Pi-pulse: {s.pi_amplitude*1e3:.2f} MHz')
"
```

**预期效果**：30 点扫描，P(|1⟩) 0~0.99，π脉冲 ~29 MHz。

---

## 六、测试套件

### 6.1 全部测试

```bash
python -m pytest tests/unit/ -v --tb=short
```

**预期效果**：156 passed, 0 failed（约 4-5 分钟）。

---

### 6.2 分模块测试

```bash
# 后端
python -m pytest tests/unit/test_backends.py -v --tb=short

# 漂移检测
python -m pytest tests/unit/test_drift_detector.py -v --tb=short

# 感知工具
python -m pytest tests/unit/test_perception_tools.py -v --tb=short

# Rabi 仿真
python -m pytest tests/unit/test_rabi.py -v --tb=short

# CLI
python -m pytest tests/unit/test_cli.py -v --tb=short
```

---

## 七、端到端 Pipeline 验证

```bash
python3 -c "
import time, json
t0 = time.time()
from backends.fake_adapter import FakeBackendAdapter
from data.store import DataStore
from data.instrumented import InstrumentedExecutor

db = DataStore('/tmp/e2e_test.duckdb')
be = FakeBackendAdapter('FakeBrisbane')
ex = InstrumentedExecutor(be, db=db)

h = json.loads(ex.execute('get_backend_health', {}))
print(f'[1] Health: T1={h[\"avg_t1_us\"]}μs')

r = json.loads(ex.execute('run_circuit', {'circuit_name': 'ghz_5', 'shots': 4096}))
print(f'[2] GHZ-5 fidelity = {r[\"fidelity\"]}')

elapsed = time.time() - t0
runs = db.query('SELECT circuit_name, fidelity FROM circuit_runs')
exps = db.experiments.count()
print(f'[3] DuckDB: {len(runs)} circuit_runs, {exps} experiment_records')
print(f'[4] Time: {elapsed:.1f}s')
print(f'Pipeline: OK')
db.close()
"
```

**预期效果**：~13 秒端到端，数据自动落入 DuckDB。

---

## 八、项目结构

```
quantumgpt/
├── backends/            # 3 种影子后端 (Fake, Replay, SyntheticDrift)
├── bench/               # 基准电路 (手写 + MQTBench)
├── console/             # Rich 终端 dashboard
├── data/                # DuckDB 存储 + ExperimentRecord
├── detection/           # 漂移检测 (PELT + BOCPD)
├── dynamics/            # Qiskit Dynamics Rabi 仿真
├── tools/               # 9 个 Agent 工具
├── agent/               # Agent loop
├── advisor/             # Advisory 系统
├── tests/unit/          # 156 个测试
├── notebooks/           # Rabi tutorial
├── demos/day1-13/       # 每日 demo (48+ 张图)
├── cli.py               # qgpt CLI
├── BACKENDS.md          # 后端文档
└── QuantumGPT_Plan_merged.md  # 12 周计划
```
