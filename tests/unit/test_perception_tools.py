"""Tests for the 6 perception tools + existing tools in ToolExecutor."""
import json
import pytest

from backends.fake_adapter import FakeBackendAdapter
from backends.synthetic_drift import SyntheticDriftBackend, DriftProfile, SUDDEN_DEGRADATION
from tools.quantum_tools import ToolExecutor, TOOL_DEFINITIONS


@pytest.fixture
def fake_executor():
    return ToolExecutor(FakeBackendAdapter("FakeBrisbane"))

@pytest.fixture
def drift_executor():
    be = SyntheticDriftBackend(profile=SUDDEN_DEGRADATION)
    be.set_time(12.0)
    return ToolExecutor(be)


class TestToolDefinitions:
    def test_current_tool_surface_is_defined_without_duplicates(self):
        names = [t["name"] for t in TOOL_DEFINITIONS]
        expected = {
            "get_backend_health",
            "get_qubit_properties",
            "get_coupling_map",
            "detect_drift",
            "get_calibration_age",
            "compare_backends",
            "run_circuit",
            "list_benchmarks",
            "transpile_circuit",
            "apply_mitigation",
            "predict_fidelity",
            "rabi_experiment",
            "fit_rabi",
            "ramsey_experiment",
            "fit_ramsey",
            "t1_experiment",
            "fit_t1",
            "diagnose_and_suggest",
        }
        assert len(names) == len(set(names))
        assert set(names) == expected

    def test_all_have_schemas(self):
        for t in TOOL_DEFINITIONS:
            assert "input_schema" in t
            assert "description" in t


class TestGetBackendHealth:
    def test_returns_all_fields(self, fake_executor):
        r = json.loads(fake_executor.execute("get_backend_health", {}))
        for key in ["backend", "num_qubits", "avg_t1_us", "avg_2q_error"]:
            assert key in r


class TestGetQubitProperties:
    def test_returns_qubit_data(self, fake_executor):
        r = json.loads(fake_executor.execute("get_qubit_properties", {"qubits": [0, 1]}))
        assert len(r["qubits"]) == 2
        assert r["qubits"][0]["qubit"] == 0


class TestGetCouplingMap:
    def test_returns_coupling_map(self, fake_executor):
        r = json.loads(fake_executor.execute("get_coupling_map", {}))
        assert "coupling_map" in r
        assert r["num_edges"] > 0
        assert r["avg_degree"] > 0
        # Each edge is [i, j]
        for edge in r["coupling_map"]:
            assert len(edge) == 2


class TestDetectDrift:
    def test_no_drift_on_fake(self, fake_executor):
        """FakeBackendAdapter has no set_time — returns static score."""
        r = json.loads(fake_executor.execute("detect_drift", {}))
        assert r["method"] == "static_health"
        assert r["changepoints"] == []

    def test_detects_sudden_degradation(self, drift_executor):
        r = json.loads(drift_executor.execute("detect_drift", {
            "window_hours": 24, "step_hours": 1.0,
        }))
        assert r["method"] == "pelt"
        assert len(r["changepoints"]) >= 1
        assert r["recalibrate_recommended"] is True
        cp = r["changepoints"][0]
        assert cp["direction"] == "degradation"
        assert cp["severity"] > 0.2


class TestGetCalibrationAge:
    def test_returns_age(self, fake_executor):
        r = json.loads(fake_executor.execute("get_calibration_age", {}))
        assert "calibration_age_minutes" in r
        assert "is_stale" in r
        assert isinstance(r["is_stale"], bool)


class TestCompareBackends:
    def test_ranking_returned(self, fake_executor):
        r = json.loads(fake_executor.execute("compare_backends", {
            "candidates": ["FakeBrisbane", "FakeSherbrooke"],
        }))
        assert "ranking" in r
        assert len(r["ranking"]) >= 2
        assert r["ranking"][0]["rank"] == 1
        assert r["recommended"] is not None

    def test_with_circuit(self, fake_executor):
        r = json.loads(fake_executor.execute("compare_backends", {
            "candidates": ["FakeBrisbane"],
            "circuit_name": "ghz_5",
        }))
        assert "fidelity" in r["ranking"][0]


class TestCallLog:
    def test_log_recorded(self, fake_executor):
        fake_executor.execute("get_backend_health", {})
        fake_executor.execute("get_coupling_map", {})
        assert len(fake_executor.call_log) == 2
        assert fake_executor.call_log[0]["tool"] == "get_backend_health"
        assert fake_executor.call_log[1]["tool"] == "get_coupling_map"

    def test_unknown_tool_logged(self, fake_executor):
        r = json.loads(fake_executor.execute("nonexistent_tool", {}))
        assert "error" in r
        assert fake_executor.call_log[-1]["success"] is False
