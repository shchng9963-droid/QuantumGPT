import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qiskit.circuit import QuantumCircuit


def test_action_grid_covers_raw_transpile_shots_and_mitigation():
    from eval.public_mqtbench_oracle import build_action_grid

    actions = build_action_grid(opt_levels=[0, 1, 3], shots=[1024, 4096], include_mitigation=True)
    names = {a.name for a in actions}

    assert "raw_s1024" in names
    assert "transpile_o0_s1024" in names
    assert "transpile_o3_s4096" in names
    assert "zne_s4096" in names
    assert all(a.shots in {1024, 4096} for a in actions)


def test_summarize_oracle_row_uses_first_post_and_best_post_only():
    from eval.public_mqtbench_oracle import OracleActionResult, summarize_oracle_task

    actions = [
        OracleActionResult(action="raw_s1024", fidelity=0.62, shots=1024, depth=10, metadata={}),
        OracleActionResult(action="transpile_o3_s1024", fidelity=0.81, shots=1024, depth=7, metadata={}),
        OracleActionResult(action="zne_s1024", fidelity=0.79, shots=1024, depth=None, metadata={}),
    ]

    row = summarize_oracle_task(
        task_id="FakeBrisbane:moderate:ghz_3:seed0",
        circuit="ghz_3",
        circuit_family="hand_written",
        backend="FakeBrisbane",
        profile="moderate",
        seed=0,
        target_fidelity=0.8,
        pre_fidelity=0.93,
        post_actions=actions,
    )

    assert row.first_post_drift_score == pytest.approx(0.62)
    assert row.best_post_drift_score == pytest.approx(0.81)
    assert row.post_drift_improvement == pytest.approx(0.19)
    assert row.target_success is True
    assert row.oracle_feasible is True
    assert row.best_action == "transpile_o3_s1024"
    dumped = row.to_dict()
    assert dumped["execution_mode"] == "deterministic_oracle"
    assert dumped["use_mock"] is False


def test_write_outputs_jsonl_and_feasibility_summary(tmp_path):
    from eval.public_mqtbench_oracle import OracleActionResult, summarize_oracle_task, write_oracle_outputs

    feasible = summarize_oracle_task(
        task_id="t1",
        circuit="ghz_3",
        circuit_family="hand_written",
        backend="FakeBrisbane",
        profile="moderate",
        seed=0,
        target_fidelity=0.8,
        pre_fidelity=0.95,
        post_actions=[OracleActionResult("raw", 0.7, 1024, 5, {}), OracleActionResult("best", 0.82, 1024, 5, {})],
    )
    infeasible = summarize_oracle_task(
        task_id="t2",
        circuit="ghz_3",
        circuit_family="hand_written",
        backend="FakeBrisbane",
        profile="severe_sudden",
        seed=0,
        target_fidelity=0.8,
        pre_fidelity=0.95,
        post_actions=[OracleActionResult("raw", 0.5, 1024, 5, {}), OracleActionResult("best", 0.6, 1024, 5, {})],
    )

    paths = write_oracle_outputs([feasible, infeasible], tmp_path)

    rows = [json.loads(line) for line in Path(paths["jsonl"]).read_text().splitlines()]
    summary = json.loads(Path(paths["summary"]).read_text())
    assert len(rows) == 2
    assert summary["n_tasks"] == 2
    assert summary["n_oracle_feasible"] == 1
    assert summary["profiles"]["moderate"]["oracle_feasible_rate"] == 1.0
    assert summary["profiles"]["severe_sudden"]["oracle_feasible_rate"] == 0.0


def test_synthetic_drift_backend_honors_optimization_level_in_run(monkeypatch):
    from backends.synthetic_drift import STABLE, SyntheticDriftBackend

    seen = {}

    class DummyResult:
        def get_counts(self):
            return {"0": 10}

    class DummyJob:
        def result(self):
            return DummyResult()

    class DummySim:
        def run(self, circuit, shots):
            return DummyJob()

    def fake_build_simulator(self, snap):
        return DummySim()

    def fake_transpile(circuit, backend=None, **kwargs):
        seen.update(kwargs)
        return circuit

    monkeypatch.setattr(SyntheticDriftBackend, "_build_simulator", fake_build_simulator)
    monkeypatch.setattr("backends.synthetic_drift.transpile", fake_transpile)
    monkeypatch.setattr(SyntheticDriftBackend, "_estimate_fidelity", lambda self, circuit, counts, shots: 1.0)

    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    qc = QuantumCircuit(1)
    qc.measure_all()
    backend.run(qc, shots=10, optimization_level=3, initial_layout=[0])

    assert seen["optimization_level"] == 3
    assert seen["initial_layout"] == [0]
