"""Unit tests for ReplayBackend."""

import sys
import time
sys.path.insert(0, "/home/wangshuchang/quantumgpt")

from qiskit.circuit import QuantumCircuit

from backends.calibration_data import (
    CalibrationSnapshot,
    extract_snapshot_from_fake_backend,
    generate_drift_series,
)
from backends.replay import ReplayBackend


def test_snapshot_extraction():
    """Test that we can extract a valid snapshot from FakeBrisbane."""
    snap = extract_snapshot_from_fake_backend("FakeBrisbane")
    assert snap.num_qubits == 127
    assert len(snap.t1_us) == 127
    assert len(snap.t2_us) == 127
    assert len(snap.readout_error) == 127
    assert len(snap.gate_error_1q) == 127
    assert len(snap.gate_error_2q) > 0
    assert all(t > 0 for t in snap.t1_us)
    assert all(t > 0 for t in snap.t2_us)
    # T2 <= 2*T1 is the physical limit, but IBM FakeBackend data may
    # have a few qubits where T2_echo > 2*T1 — we just check most comply
    violations = sum(1 for q in range(snap.num_qubits) if snap.t2_us[q] > 2 * snap.t1_us[q] + 5.0)
    assert violations < 10  # at most a handful
    print(f"  ✓ Snapshot: {snap.num_qubits}q, avg T1={sum(snap.t1_us)/len(snap.t1_us):.1f} us")


def test_drift_series():
    """Test that drift series generates changing snapshots."""
    snap = extract_snapshot_from_fake_backend("FakeBrisbane")
    series = generate_drift_series(snap, num_snapshots=50, seed=42)
    
    assert len(series) == 50
    # Check timestamps are increasing
    for i in range(len(series) - 1):
        assert series[i].timestamp < series[i+1].timestamp
    
    # Check that values actually drift (not all identical)
    t1_first = series[0].t1_us[0]
    t1_last = series[-1].t1_us[0]
    # With 50 steps of drift, Q0's T1 should differ
    assert t1_first != t1_last
    
    # Physical constraints preserved
    for s in series:
        for q in range(s.num_qubits):
            assert s.t1_us[q] >= 5.0
            assert s.t2_us[q] <= 2 * s.t1_us[q] + 0.01
            assert 0 < s.readout_error[q] < 0.5
    
    print(f"  ✓ Drift series: {len(series)} snapshots, Q0 T1 drifted {t1_first:.1f} → {t1_last:.1f} us")


def test_replay_backend_health():
    """Test ReplayBackend health reporting."""
    snap = extract_snapshot_from_fake_backend("FakeBrisbane")
    series = generate_drift_series(snap, num_snapshots=10, seed=42)
    
    backend = ReplayBackend(series, speed=1e9)  # very fast replay
    
    health = backend.get_health()
    assert health.num_qubits == 127
    assert health.avg_t1_us > 0
    assert health.avg_2q_error > 0
    print(f"  ✓ Health: avg_1q_err={health.avg_1q_error:.6f}, avg_T1={health.avg_t1_us:.1f} us")


def test_replay_backend_run():
    """Test running a circuit on ReplayBackend."""
    snap = extract_snapshot_from_fake_backend("FakeBrisbane")
    series = generate_drift_series(snap, num_snapshots=10, seed=42)
    
    backend = ReplayBackend(series, speed=1.0)
    
    # Build GHZ-3 (small for speed)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    
    result = backend.run(qc, shots=4096)
    
    assert result.shots == 4096
    assert sum(result.counts.values()) == 4096
    assert result.fidelity is not None
    assert 0.0 < result.fidelity <= 1.0
    assert "snapshot_index" in result.metadata
    
    print(f"  ✓ Run GHZ-3: fidelity={result.fidelity:.4f}, depth={result.metadata['transpiled_depth']}")


def test_replay_seek():
    """Test that seeking changes the noise model."""
    snap = extract_snapshot_from_fake_backend("FakeBrisbane")
    # Generate series with more aggressive drift for visible difference
    series = generate_drift_series(snap, num_snapshots=100, drift_rate=0.05, jump_prob=0.1, seed=42)
    
    backend = ReplayBackend(series, speed=1.0)
    
    # Seek to different points and check health changes
    backend.seek(0)
    health_start = backend.get_health()
    
    backend.seek(99)
    health_end = backend.get_health()
    
    # With aggressive drift, values should differ
    assert health_start.avg_t1_us != health_end.avg_t1_us
    print(f"  ✓ Seek: T1 at start={health_start.avg_t1_us:.1f}, at end={health_end.avg_t1_us:.1f}")


if __name__ == "__main__":
    print("Testing CalibrationSnapshot extraction...")
    test_snapshot_extraction()
    
    print("\nTesting drift series generation...")
    test_drift_series()
    
    print("\nTesting ReplayBackend health...")
    test_replay_backend_health()
    
    print("\nTesting ReplayBackend run...")
    test_replay_backend_run()
    
    print("\nTesting ReplayBackend seek...")
    test_replay_seek()
    
    print("\n" + "=" * 50)
    print("All ReplayBackend tests passed! ✓")
