"""Unit tests for all three shadow backends + properties stream."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from qiskit.circuit import QuantumCircuit

from backends.base import BackendHealth, QubitProperties, SimulationResult, ShadowBackend
from backends.fake_adapter import FakeBackendAdapter
from backends.replay_backend import ReplayBackend
from backends.synthetic_drift import (
    SyntheticDriftBackend,
    DriftProfile,
    LINEAR_DECAY,
    SUDDEN_DEGRADATION,
    DIURNAL_CYCLE,
    STABLE,
)
from backends.properties_stream import PropertiesStream


# ===== Fixtures =====

@pytest.fixture
def ghz5():
    """5-qubit GHZ circuit with measurements."""
    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


@pytest.fixture
def bell():
    """2-qubit Bell state circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def fake_backend():
    return FakeBackendAdapter("FakeBrisbane")


@pytest.fixture
def replay_backend():
    return ReplayBackend.from_synthetic_history(
        base_backend="FakeBrisbane",
        duration_hours=48,
        snapshot_interval_hours=2,
        seed=123,
    )


@pytest.fixture
def synthetic_backend():
    return SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)


# ===== FakeBackendAdapter Tests =====

class TestFakeBackendAdapter:
    def test_instantiation(self, fake_backend):
        assert fake_backend.name == "FakeBrisbane"
        assert fake_backend.num_qubits == 127

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown fake backend"):
            FakeBackendAdapter("NonexistentBackend")

    def test_get_health(self, fake_backend):
        h = fake_backend.get_health()
        assert isinstance(h, BackendHealth)
        assert h.num_qubits == 127
        assert 0 < h.avg_1q_error < 0.1
        assert 0 < h.avg_2q_error < 0.2
        assert h.avg_t1_us > 50  # reasonable T1
        assert h.avg_t2_us > 20  # reasonable T2

    def test_get_qubit_properties(self, fake_backend):
        props = fake_backend.get_qubit_properties([0, 1, 2])
        assert len(props) == 3
        for p in props:
            assert isinstance(p, QubitProperties)
            assert p.t1_us > 0
            assert p.t2_us > 0

    def test_get_coupling_map(self, fake_backend):
        cm = fake_backend.get_coupling_map()
        assert len(cm) > 0
        assert all(isinstance(e, tuple) and len(e) == 2 for e in cm)

    def test_run_ghz(self, fake_backend, ghz5):
        result = fake_backend.run(ghz5, shots=4096)
        assert isinstance(result, SimulationResult)
        assert result.shots == 4096
        assert sum(result.counts.values()) == 4096
        assert result.fidelity is not None
        assert 0.5 < result.fidelity < 1.0  # should be decent but not perfect

    def test_run_bell(self, fake_backend, bell):
        result = fake_backend.run(bell, shots=4096)
        # Bell state: should be mostly 00 and 11
        total = sum(result.counts.values())
        dominant = result.counts.get("00", 0) + result.counts.get("11", 0)
        assert dominant / total > 0.8

    def test_properties_snapshot(self, fake_backend):
        snap = fake_backend.get_properties_snapshot()
        assert "backend" in snap
        assert "timestamp" in snap
        assert snap["num_qubits"] == 127


# ===== ReplayBackend Tests =====

class TestReplayBackend:
    def test_creation(self, replay_backend):
        assert "Replay" in replay_backend.name
        assert replay_backend.num_qubits == 20
        assert replay_backend.total_duration == 48.0

    def test_time_progression(self, replay_backend):
        replay_backend.set_time(0)
        h0 = replay_backend.get_health()
        assert h0.drift_score == 0.0

        replay_backend.set_time(24)
        h24 = replay_backend.get_health()
        assert h24.drift_score > 0  # some drift expected

    def test_drift_increases(self, replay_backend):
        scores = []
        for t in [0, 12, 24, 36, 48]:
            replay_backend.set_time(t)
            scores.append(replay_backend.get_health().drift_score)
        # Drift at t=0 should be 0
        assert scores[0] == 0.0
        # Later drift should generally be non-zero
        assert any(s > 0 for s in scores[1:])

    def test_run_at_different_times(self, replay_backend, ghz5):
        replay_backend.set_time(0)
        r0 = replay_backend.run(ghz5, shots=4096)
        assert r0.fidelity is not None
        assert 0.3 < r0.fidelity < 1.0

        replay_backend.set_time(48)
        r48 = replay_backend.run(ghz5, shots=4096)
        assert r48.fidelity is not None

    def test_qubit_properties(self, replay_backend):
        replay_backend.set_time(10)
        props = replay_backend.get_qubit_properties([0, 1, 2])
        assert len(props) == 3
        assert all(p.t1_us > 0 for p in props)


# ===== SyntheticDriftBackend Tests =====

class TestSyntheticDriftBackend:
    def test_stable_profile(self, ghz5):
        backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
        backend.set_time(0)
        h0 = backend.get_health()
        backend.set_time(10)
        h10 = backend.get_health()
        # Stable: T1 should not change
        assert abs(h0.avg_t1_us - h10.avg_t1_us) < 1.0

    def test_linear_decay(self, synthetic_backend):
        synthetic_backend.set_time(0)
        t1_start = synthetic_backend.get_health().avg_t1_us
        synthetic_backend.set_time(10)
        t1_end = synthetic_backend.get_health().avg_t1_us
        # Linear decay: T1 should decrease
        assert t1_end < t1_start

    def test_sudden_degradation(self, ghz5):
        backend = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
        backend.set_time(3)
        r_before = backend.run(ghz5, shots=4096)
        backend.set_time(6)
        r_after = backend.run(ghz5, shots=4096)
        # Fidelity should drop significantly after the spike
        assert r_after.fidelity < r_before.fidelity - 0.05

    def test_custom_profile(self, ghz5):
        # T1 drops to 50% at t=1, everything else stable
        profile = DriftProfile(t1_drift=lambda t: max(0.5, 1.0 - 0.5 * t))
        backend = SyntheticDriftBackend("FakeBrisbane", profile)
        backend.set_time(0)
        t1_0 = backend.get_health().avg_t1_us
        backend.set_time(1)
        t1_1 = backend.get_health().avg_t1_us
        assert t1_1 < t1_0 * 0.7  # should be around 50%

    def test_per_qubit_override(self):
        # Only qubit 0 degrades, others stable
        profile = DriftProfile(
            per_qubit_t1={0: lambda t: 0.3},  # Q0 always at 30%
        )
        backend = SyntheticDriftBackend("FakeBrisbane", profile)
        backend.set_time(5)
        props = backend.get_qubit_properties([0, 1])
        assert props[0].t1_us < props[1].t1_us * 0.5  # Q0 much worse

    def test_profile_hot_swap(self, ghz5):
        backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
        backend.set_time(5)
        r_stable = backend.run(ghz5, shots=4096)

        # Hot-swap to sudden degradation
        backend.profile = SUDDEN_DEGRADATION
        r_degraded = backend.run(ghz5, shots=4096)
        assert r_degraded.fidelity < r_stable.fidelity


# ===== PropertiesStream Tests =====

class TestPropertiesStream:
    def test_replay_all(self, replay_backend):
        snapshots = PropertiesStream.replay_all(replay_backend, hours=48, step_hours=12)
        assert len(snapshots) == 5  # 0, 12, 24, 36, 48
        assert snapshots[0]["stream_sim_time_hours"] == 0.0
        assert snapshots[-1]["stream_sim_time_hours"] == 48.0
        assert all("drift_score" in s for s in snapshots)

    def test_replay_all_synthetic(self):
        backend = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
        snapshots = PropertiesStream.replay_all(backend, hours=10, step_hours=1)
        assert len(snapshots) == 11
        # Before spike: drift=0, after spike: drift=1
        assert snapshots[0]["drift_score"] == 0.0
        assert snapshots[6]["drift_score"] == 1.0

    def test_realtime_stream(self, replay_backend):
        import time
        stream = PropertiesStream(replay_backend, interval_seconds=0.1, time_acceleration=3600)
        stream.start()
        deadline = time.time() + 1.5
        try:
            while stream.buffer_size < 3 and time.time() < deadline:
                time.sleep(0.02)
        finally:
            stream.stop()
        assert stream.buffer_size >= 3  # should have captured several snapshots
        latest = stream.get_latest()
        assert latest is not None
        assert "drift_score" in latest


# ===== Interface Compliance =====

class TestInterfaceCompliance:
    """Verify all backends implement the full ShadowBackend interface."""

    @pytest.mark.parametrize("backend_factory", [
        lambda: FakeBackendAdapter("FakeBrisbane"),
        lambda: ReplayBackend.from_synthetic_history("FakeBrisbane", 24, 4, seed=1),
        lambda: SyntheticDriftBackend("FakeBrisbane", STABLE),
    ])
    def test_all_methods(self, backend_factory, ghz5):
        backend = backend_factory()
        if hasattr(backend, "set_time"):
            backend.set_time(1.0)

        # All required methods
        assert isinstance(backend.name, str)
        assert isinstance(backend.num_qubits, int)
        assert backend.num_qubits > 0

        health = backend.get_health()
        assert isinstance(health, BackendHealth)

        props = backend.get_qubit_properties([0])
        assert isinstance(props, list)
        assert len(props) == 1

        cm = backend.get_coupling_map()
        assert isinstance(cm, list)

        result = backend.run(ghz5, shots=1024)
        assert isinstance(result, SimulationResult)
        assert sum(result.counts.values()) == 1024

        snap = backend.get_properties_snapshot()
        assert isinstance(snap, dict)
        assert "backend" in snap


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
