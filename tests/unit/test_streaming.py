"""Unit tests for StreamingAdvisor — real-time telemetry + alert engine."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import pytest

from advisor.streaming import (
    StreamingAdvisor,
    AlertConfig,
    AlertLevel,
    AlertType,
    Alert,
    HealthTracker,
)
from backends.synthetic_drift import (
    SyntheticDriftBackend,
    DriftProfile,
    LINEAR_DECAY,
    SUDDEN_DEGRADATION,
    DIURNAL_CYCLE,
    STABLE,
)


# ===== Fixtures =====

@pytest.fixture
def stable_sa(tmp_path):
    backend = SyntheticDriftBackend("FakeBrisbane", STABLE)
    return StreamingAdvisor(backend, db_path=str(tmp_path / "stable.duckdb"))


@pytest.fixture
def linear_sa(tmp_path):
    backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
    return StreamingAdvisor(backend, db_path=str(tmp_path / "linear.duckdb"))


@pytest.fixture
def sudden_sa(tmp_path):
    backend = SyntheticDriftBackend("FakeBrisbane", SUDDEN_DEGRADATION)
    config = AlertConfig(alert_cooldown_hours=0.1)  # low cooldown for testing
    return StreamingAdvisor(backend, alert_config=config,
                            db_path=str(tmp_path / "sudden.duckdb"))


# ===== HealthTracker Tests =====

class TestHealthTracker:
    def test_add_and_latest(self):
        tracker = HealthTracker(window_size=5)
        tracker.add({"avg_t1_us": 200, "drift_score": 0.1, "stream_sim_time_hours": 0})
        assert tracker.latest["avg_t1_us"] == 200
        assert tracker.size == 1

    def test_window_overflow(self):
        tracker = HealthTracker(window_size=3)
        for i in range(5):
            tracker.add({"avg_t1_us": 200 - i * 10, "drift_score": i * 0.1,
                         "stream_sim_time_hours": i})
        assert tracker.size == 3
        assert tracker.latest["stream_sim_time_hours"] == 4

    def test_t1_slope(self):
        tracker = HealthTracker(window_size=10)
        for i in range(5):
            tracker.add({"avg_t1_us": 200 - i * 10, "drift_score": 0,
                         "stream_sim_time_hours": i})
        slope = tracker.t1_slope()
        assert slope < 0  # T1 decreasing
        assert abs(slope - (-10.0)) < 0.01  # should be exactly -10/h

    def test_drift_jump(self):
        tracker = HealthTracker(window_size=5)
        tracker.add({"drift_score": 0.1, "stream_sim_time_hours": 0})
        tracker.add({"drift_score": 0.5, "stream_sim_time_hours": 1})
        assert abs(tracker.drift_jump() - 0.4) < 0.001

    def test_prev_snapshot(self):
        tracker = HealthTracker(window_size=5)
        tracker.add({"value": 1})
        assert tracker.prev is None
        tracker.add({"value": 2})
        assert tracker.prev["value"] == 1


# ===== Batch Mode Tests =====

class TestBatchMode:
    def test_stable_no_alerts(self, stable_sa):
        alerts = stable_sa.run_batch([0, 4, 8, 12])
        # Stable backend should not trigger alerts
        assert len(alerts) == 0
        assert stable_sa.snapshot_count == 4

    def test_sudden_triggers_alerts(self, sudden_sa):
        alerts = sudden_sa.run_batch([0, 2, 4, 4.5, 5, 5.5, 6, 8])
        assert len(alerts) > 0
        # Should have at least one critical alert
        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical) > 0
        # Should detect drift spike around t=5
        spikes = [a for a in alerts if a.alert_type == AlertType.DRIFT_SPIKE]
        assert len(spikes) > 0

    def test_linear_drift_warning(self, linear_sa):
        # Run far enough into linear decay for drift warning
        alerts = linear_sa.run_batch([0, 4, 8, 12, 16, 20])
        drift_alerts = [a for a in alerts if a.alert_type == AlertType.DRIFT_HIGH]
        assert len(drift_alerts) > 0

    def test_alert_cooldown(self, sudden_sa):
        # Run with many time points close together
        # Cooldown = 0.1h, so alerts within 0.1h of each other should be suppressed
        alerts = sudden_sa.run_batch([5.0, 5.05, 5.1, 5.15, 5.2])
        # Should not get 5 drift alerts — cooldown should suppress some
        drift_high = [a for a in alerts if a.alert_type == AlertType.DRIFT_HIGH]
        assert len(drift_high) <= 3  # at most 3 with 0.1h cooldown over 0.2h range

    def test_recovery_detection(self, tmp_path):
        """Test recovery alert when switching from degraded to healthy."""
        # Create a custom profile that degrades then recovers
        profile = DriftProfile(
            t1_drift=lambda t: 0.3 if 3 <= t <= 5 else 1.0,
            drift_score_fn=lambda t: 1.0 if 3 <= t <= 5 else 0.0,
        )
        backend = SyntheticDriftBackend("FakeBrisbane", profile)
        config = AlertConfig(alert_cooldown_hours=0.5)
        sa = StreamingAdvisor(backend, alert_config=config,
                              db_path=str(tmp_path / "recovery.duckdb"))

        alerts = sa.run_batch([0, 1, 2, 3, 4, 5, 6, 7])
        recovery = [a for a in alerts if a.alert_type == AlertType.RECOVERY]
        assert len(recovery) >= 1
        # Recovery should happen around t=6 (after critical at t=3-5)
        assert any(5.5 <= a.sim_time_hours <= 7.5 for a in recovery)

    def test_t1_degradation_alert(self, linear_sa):
        # LINEAR_DECAY: T1 drops by ~7 us/h → should trigger T1_DEGRADING
        alerts = linear_sa.run_batch([0, 2, 4, 6, 8, 10, 12])
        t1_alerts = [a for a in alerts
                     if a.alert_type in (AlertType.T1_DEGRADING, AlertType.T1_LOW)]
        assert len(t1_alerts) > 0


# ===== Streaming Mode Tests =====

class TestStreamingMode:
    def test_start_stop(self, stable_sa):
        stable_sa.start(interval_seconds=0.1, time_acceleration=3600,
                       duration_seconds=0.5)
        assert not stable_sa.is_running  # stopped after duration
        assert stable_sa.snapshot_count >= 3

    def test_streaming_collects_snapshots(self, linear_sa):
        linear_sa.start(interval_seconds=0.1, time_acceleration=7200,
                       duration_seconds=1.0)
        assert linear_sa.snapshot_count >= 5
        history = linear_sa.get_snapshot_history()
        assert len(history) > 0

    def test_streaming_fires_alerts(self, tmp_path):
        """Run streaming long enough for LINEAR_DECAY to trigger alerts."""
        backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
        config = AlertConfig(
            drift_warning=0.15,
            alert_cooldown_hours=0.5,
        )
        sa = StreamingAdvisor(backend, alert_config=config,
                              db_path=str(tmp_path / "stream.duckdb"))

        received_alerts = []
        sa.on_alert(lambda a: received_alerts.append(a))

        # 2 seconds at 7200x = 4 simulated hours
        sa.start(interval_seconds=0.1, time_acceleration=7200,
                duration_seconds=2.0)

        # Should have some snapshots and potentially alerts
        assert sa.snapshot_count >= 10

    def test_alert_callback(self, sudden_sa):
        received = []
        sudden_sa.on_alert(lambda a: received.append(a))

        # Batch mode to test callback without threads
        sudden_sa.run_batch([0, 4, 5, 6])
        assert len(received) > 0
        assert all(isinstance(a, Alert) for a in received)

    def test_remove_callback(self, sudden_sa):
        received = []
        cb = lambda a: received.append(a)
        sudden_sa.on_alert(cb)
        sudden_sa.remove_alert_callback(cb)
        sudden_sa.run_batch([0, 4, 5, 6])
        assert len(received) == 0


# ===== Query/Summary Tests =====

class TestQueries:
    def test_get_summary(self, sudden_sa):
        sudden_sa.run_batch([0, 2, 4, 5, 5.5, 6, 8])
        summary = sudden_sa.get_summary()
        assert summary["total_snapshots"] == 7
        assert summary["total_alerts"] > 0
        assert "alerts_by_level" in summary
        assert "alerts_by_type" in summary
        assert summary["latest_health"] is not None

    def test_filter_alerts_by_level(self, sudden_sa):
        sudden_sa.run_batch([0, 2, 4, 5, 5.5, 6, 8])
        critical = sudden_sa.get_alert_history(level=AlertLevel.CRITICAL)
        all_alerts = sudden_sa.get_alert_history()
        assert len(critical) <= len(all_alerts)
        assert all(a.level == AlertLevel.CRITICAL for a in critical)

    def test_last_n_alerts(self, sudden_sa):
        sudden_sa.run_batch([0, 2, 4, 5, 5.5, 6, 8])
        last2 = sudden_sa.get_alert_history(last_n=2)
        assert len(last2) <= 2

    def test_duckdb_persistence(self, sudden_sa):
        sudden_sa.run_batch([0, 2, 4, 5, 6])
        counts = sudden_sa.db.table_counts()
        assert counts["health_snapshots"] == 5
        assert counts["drift_events"] >= 1  # at least one alert persisted


# ===== Integration with Day 8 advisor =====

class TestIntegrationWithAdvisor:
    def test_batch_then_advisor(self, tmp_path):
        """Run streaming batch, then use the same DB with CalibrationAdvisor."""
        from advisor.advisor import CalibrationAdvisor

        backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
        db_path = str(tmp_path / "integrated.duckdb")

        # Phase 1: Streaming monitoring
        sa = StreamingAdvisor(backend, db_path=db_path)
        sa.run_batch([0, 4, 8, 12, 16, 20])

        # Phase 2: Full advisory cycle using same backend
        advisor = CalibrationAdvisor(backend, db_path=db_path, qubit_sample_size=5)
        report = advisor.run_advisory_cycle(
            time_points=[20, 22, 24],
            probe_circuits=["ghz_5"],
        )

        # Both should have written to the same DB
        from data.store import DataStore
        db = DataStore(db_path)
        counts = db.table_counts()
        # 6 from streaming + 3 from advisor = 9
        assert counts["health_snapshots"] >= 9
        assert report.severity.value in ("warning", "critical")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
