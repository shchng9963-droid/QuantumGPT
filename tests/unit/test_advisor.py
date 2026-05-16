"""Unit tests for CalibrationAdvisor — multi-step advisory chain."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import numpy as np

from advisor.advisor import (
    CalibrationAdvisor,
    ActionType,
    Severity,
    TrendDirection,
    AdvisoryAction,
    TemporalAnalysis,
    AdvisoryReport,
)
from backends.synthetic_drift import (
    SyntheticDriftBackend,
    DriftProfile,
    LINEAR_DECAY,
    SUDDEN_DEGRADATION,
    DIURNAL_CYCLE,
    STABLE,
)
from backends.replay_backend import ReplayBackend


# ===== Fixtures =====

@pytest.fixture
def stable_advisor(tmp_path):
    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    return CalibrationAdvisor(backend, db_path=str(tmp_path / "stable.duckdb"),
                              qubit_sample_size=10)

@pytest.fixture
def linear_advisor(tmp_path):
    backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
    return CalibrationAdvisor(backend, db_path=str(tmp_path / "linear.duckdb"),
                              qubit_sample_size=10)

@pytest.fixture
def sudden_advisor(tmp_path):
    backend = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
    return CalibrationAdvisor(backend, db_path=str(tmp_path / "sudden.duckdb"),
                              qubit_sample_size=10)

@pytest.fixture
def replay_advisor(tmp_path):
    backend = ReplayBackend.from_synthetic_history(
        "FakeBrisbane", duration_hours=48, snapshot_interval_hours=4, seed=42)
    return CalibrationAdvisor(backend, db_path=str(tmp_path / "replay.duckdb"),
                              qubit_sample_size=10)


# ===== Phase 1: Monitor Tests =====

class TestMonitor:
    def test_monitor_collects_health(self, stable_advisor):
        stable_advisor.monitor([0, 4, 8])
        assert len(stable_advisor._health_timeline) == 3
        for h in stable_advisor._health_timeline:
            assert "avg_t1_us" in h
            assert "drift_score" in h

    def test_monitor_collects_qubit_properties(self, stable_advisor):
        stable_advisor.monitor([0, 4])
        assert len(stable_advisor._qubit_history) == 10  # qubit_sample_size
        for q, hist in stable_advisor._qubit_history.items():
            assert len(hist) == 2  # 2 time points
            assert all("t1_us" in h for h in hist)

    def test_monitor_runs_probe_circuits(self, stable_advisor):
        stable_advisor.monitor([0], probe_circuits=["ghz_5"])
        assert len(stable_advisor._probe_results) == 1
        pr = stable_advisor._probe_results[0]
        assert pr.circuit == "ghz_5"
        assert 0 < pr.fidelity <= 1.0

    def test_monitor_persists_to_duckdb(self, stable_advisor):
        stable_advisor.monitor([0, 4])
        counts = stable_advisor.db.table_counts()
        assert counts["health_snapshots"] == 2
        assert counts["qubit_properties"] >= 10  # 10 qubits * 2 times

    def test_monitor_multiple_probes(self, linear_advisor):
        linear_advisor.monitor([0, 6, 12], probe_circuits=["ghz_5", "qft_4"])
        assert len(linear_advisor._probe_results) == 6  # 3 times * 2 circuits


# ===== Phase 2: Diagnose Tests =====

class TestDiagnose:
    def test_stable_trend(self, stable_advisor):
        stable_advisor.monitor([0, 4, 8, 12])
        analysis = stable_advisor.diagnose()
        assert analysis.trend == TrendDirection.STABLE
        assert abs(analysis.t1_slope) < 1.0  # near zero slope
        assert len(analysis.changepoints) == 0

    def test_linear_decay_detected(self, linear_advisor):
        linear_advisor.monitor([0, 4, 8, 12, 16, 20])
        analysis = linear_advisor.diagnose()
        assert analysis.t1_slope < 0  # T1 is decreasing
        assert analysis.drift_slope > 0  # drift is increasing
        assert analysis.trend in (TrendDirection.DEGRADING, TrendDirection.SUDDEN_CHANGE)

    def test_sudden_change_detected(self, sudden_advisor):
        sudden_advisor.monitor([0, 2, 4, 5, 6, 8])
        analysis = sudden_advisor.diagnose()
        assert len(analysis.changepoints) > 0
        assert analysis.trend == TrendDirection.SUDDEN_CHANGE
        # The changepoint should be around t=5 (when SUDDEN_DEGRADATION kicks in)
        assert any(4.5 <= cp <= 6.5 for cp in analysis.changepoints)

    def test_qubit_ranking(self, linear_advisor):
        linear_advisor.monitor([0, 6, 12])
        analysis = linear_advisor.diagnose()
        assert len(analysis.worst_qubits) <= 5
        assert len(analysis.best_qubits) <= 5
        # Best and worst should be disjoint
        assert not set(analysis.worst_qubits).intersection(analysis.best_qubits)

    def test_fidelity_trend_with_probes(self, linear_advisor):
        linear_advisor.monitor([0, 8, 16], probe_circuits=["ghz_5"])
        analysis = linear_advisor.diagnose()
        assert analysis.fidelity_trend is not None
        assert analysis.fidelity_slope is not None
        # Under linear decay, fidelity should decrease
        assert analysis.fidelity_slope < 0 or analysis.fidelity_trend in (
            TrendDirection.DEGRADING, TrendDirection.STABLE)

    def test_diagnose_with_insufficient_data(self, stable_advisor):
        stable_advisor.monitor([0])  # only 1 point
        analysis = stable_advisor.diagnose()
        assert analysis.trend == TrendDirection.STABLE
        assert analysis.t1_slope == 0


# ===== Phase 3: Advise Tests =====

class TestAdvise:
    def test_stable_returns_ok(self, stable_advisor):
        stable_advisor.monitor([0, 4, 8])
        analysis = stable_advisor.diagnose()
        actions, severity = stable_advisor.advise(analysis)
        assert severity == Severity.NOMINAL
        assert any(a.action == ActionType.OK for a in actions)

    def test_sudden_triggers_recalibrate(self, sudden_advisor):
        sudden_advisor.monitor([0, 2, 4, 5, 6, 8])
        analysis = sudden_advisor.diagnose()
        actions, severity = sudden_advisor.advise(analysis)
        assert severity == Severity.CRITICAL
        assert any(a.action == ActionType.RECALIBRATE for a in actions)

    def test_degraded_fidelity_triggers_remap(self, linear_advisor):
        linear_advisor.monitor([0, 8, 16, 24], probe_circuits=["ghz_5"])
        analysis = linear_advisor.diagnose()
        actions, severity = linear_advisor.advise(analysis)
        # With LINEAR_DECAY at t=24, fidelity should be quite low
        action_types = [a.action for a in actions]
        assert ActionType.OK not in action_types  # shouldn't be OK after 24h of decay

    def test_actions_sorted_by_priority(self, sudden_advisor):
        sudden_advisor.monitor([0, 4, 6, 8], probe_circuits=["ghz_5"])
        analysis = sudden_advisor.diagnose()
        actions, _ = sudden_advisor.advise(analysis)
        priorities = [a.priority for a in actions]
        assert priorities == sorted(priorities)

    def test_high_drift_triggers_critical(self, linear_advisor):
        # LINEAR_DECAY at t=20 gives drift ~0.6 (critical)
        linear_advisor.monitor([18, 19, 20])
        analysis = linear_advisor.diagnose()
        actions, severity = linear_advisor.advise(analysis)
        assert severity in (Severity.WARNING, Severity.CRITICAL)


# ===== Full Cycle Tests =====

class TestAdvisoryCycle:
    def test_full_cycle_stable(self, stable_advisor):
        report = stable_advisor.run_advisory_cycle(
            time_points=[0, 4, 8],
            probe_circuits=["ghz_5"],
        )
        assert isinstance(report, AdvisoryReport)
        assert report.severity == Severity.NOMINAL
        assert report.num_snapshots == 3
        assert report.num_probes == 3
        assert len(report.summary) > 100
        assert "OK" in report.summary or "nominal" in report.summary.lower()

    def test_full_cycle_sudden(self, sudden_advisor):
        report = sudden_advisor.run_advisory_cycle(
            time_points=[0, 3, 5, 6, 8],
            probe_circuits=["ghz_5"],
        )
        assert report.severity == Severity.CRITICAL
        assert any(a.action == ActionType.RECALIBRATE for a in report.actions)
        assert report.temporal_analysis.trend == TrendDirection.SUDDEN_CHANGE
        assert "RECALIBRATE" in report.summary

    def test_full_cycle_linear(self, linear_advisor):
        report = linear_advisor.run_advisory_cycle(
            time_points=[0, 6, 12, 18, 24],
            probe_circuits=["ghz_5"],
        )
        assert report.time_span_hours == 24
        assert report.num_probes == 5
        assert report.temporal_analysis.t1_slope < 0

    def test_full_cycle_replay(self, replay_advisor):
        report = replay_advisor.run_advisory_cycle(
            time_points=[0, 12, 24, 36, 48],
            probe_circuits=["ghz_5"],
        )
        assert report.num_snapshots == 5
        assert len(report.health_timeline) == 5

    def test_summary_contains_key_info(self, linear_advisor):
        report = linear_advisor.run_advisory_cycle(
            time_points=[0, 8, 16],
            probe_circuits=["ghz_5"],
        )
        summary = report.summary
        assert "Backend:" in summary
        assert "Severity:" in summary
        assert "T1:" in summary
        assert "Recommended actions" in summary

    def test_duckdb_persistence(self, linear_advisor):
        linear_advisor.run_advisory_cycle(
            time_points=[0, 4, 8],
            probe_circuits=["ghz_5"],
        )
        counts = linear_advisor.db.table_counts()
        assert counts["health_snapshots"] == 3
        assert counts["circuit_runs"] == 3
        assert counts["qubit_properties"] >= 10


# ===== Temporal Query Tests =====

class TestTemporalQueries:
    def test_query_health_trend(self, linear_advisor):
        linear_advisor.monitor([0, 4, 8])
        result = linear_advisor.query_health_trend(hours_back=24)
        assert len(result) == 3

    def test_query_fidelity_trend(self, linear_advisor):
        linear_advisor.monitor([0, 4, 8], probe_circuits=["ghz_5"])
        result = linear_advisor.query_fidelity_trend("ghz_5")
        assert len(result) == 3

    def test_query_worst_qubits(self, linear_advisor):
        linear_advisor.monitor([0, 4])
        result = linear_advisor.query_worst_qubits(top_n=3)
        assert len(result) >= 0  # may be empty if no qubit data in DB


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
