"""Tests for detection.drift_detector module."""
import numpy as np
import pytest

from backends.synthetic_drift import SyntheticDriftBackend, DriftProfile, STABLE, SUDDEN_DEGRADATION
from backends.properties_stream import PropertiesStream
from detection.drift_detector import (
    DriftDetector,
    DriftReport,
    ChangePoint,
    BOCPD,
    extract_features,
    standardize,
    detect_pelt,
    FEATURE_NAMES,
)


# ──── Fixtures ────

@pytest.fixture
def stable_snapshots():
    be = SyntheticDriftBackend(profile=STABLE)
    return PropertiesStream.replay_all(be, hours=24, step_hours=0.5)

@pytest.fixture
def sudden_snapshots():
    profile = DriftProfile(
        t1_drift=lambda t: 0.3 if t >= 10.0 else 1.0,
        gate_error_drift=lambda t: 5.0 if t >= 10.0 else 1.0,
        readout_drift=lambda t: 3.0 if t >= 10.0 else 1.0,
    )
    be = SyntheticDriftBackend(profile=profile)
    return PropertiesStream.replay_all(be, hours=24, step_hours=0.5)

@pytest.fixture
def double_shift_snapshots():
    profile = DriftProfile(
        t1_drift=lambda t: 0.4 if 6 <= t < 16 else (0.9 if t >= 16 else 1.0),
        gate_error_drift=lambda t: 2.5 if 6 <= t < 16 else (1.1 if t >= 16 else 1.0),
    )
    be = SyntheticDriftBackend(profile=profile)
    return PropertiesStream.replay_all(be, hours=24, step_hours=0.5)


# ──── Feature extraction ────

class TestFeatureExtraction:
    def test_shape(self, stable_snapshots):
        raw, times = extract_features(stable_snapshots)
        assert raw.shape == (49, 5)
        assert len(times) == 49

    def test_feature_names(self):
        assert len(FEATURE_NAMES) == 5
        assert "mean_t1" in FEATURE_NAMES

    def test_standardize(self, stable_snapshots):
        raw, _ = extract_features(stable_snapshots)
        sig, mu, std = standardize(raw)
        assert sig.shape == raw.shape
        np.testing.assert_allclose(sig.mean(axis=0), 0, atol=1e-10)


# ──── PELT ────

class TestPELT:
    def test_stable_no_changepoints(self, stable_snapshots):
        raw, _ = extract_features(stable_snapshots)
        sig, _, _ = standardize(raw)
        cps = detect_pelt(sig, pen=3.0)
        assert len(cps) == 0, f"Should find no CPs in stable signal, got {cps}"

    def test_sudden_shift_detected(self, sudden_snapshots):
        raw, times = extract_features(sudden_snapshots)
        sig, _, _ = standardize(raw)
        cps = detect_pelt(sig, pen=3.0)
        assert len(cps) >= 1, "Should detect at least 1 CP for sudden shift"
        # CP should be near idx=20 (t=10h with step=0.5h)
        assert any(abs(cp - 20) <= 3 for cp in cps), f"CP near idx=20 not found, got {cps}"

    def test_double_shift(self, double_shift_snapshots):
        raw, _ = extract_features(double_shift_snapshots)
        sig, _, _ = standardize(raw)
        cps = detect_pelt(sig, pen=3.0)
        assert len(cps) >= 2, f"Should detect 2 CPs, got {len(cps)}: {cps}"


# ──── BOCPD ────

class TestBOCPD:
    def test_reset(self):
        bocpd = BOCPD(hazard_rate=0.1, threshold=0.1)
        for i in range(10):
            bocpd.update(np.array([0.0, 0.0]))
        bocpd.reset()
        assert bocpd.get_changepoints() == []

    def test_sudden_shift(self, sudden_snapshots):
        raw, _ = extract_features(sudden_snapshots)
        sig, _, _ = standardize(raw)
        bocpd = BOCPD(hazard_rate=0.1, threshold=0.1)
        for i in range(len(sig)):
            bocpd.update(sig[i])
        cps = bocpd.get_changepoints()
        # Should detect something near idx=20
        assert len(cps) >= 1, "BOCPD should detect at least 1 CP"


# ──── DriftDetector (high-level) ────

class TestDriftDetector:
    def test_detect_returns_report(self, sudden_snapshots):
        det = DriftDetector(method="pelt")
        report = det.detect(sudden_snapshots)
        assert isinstance(report, DriftReport)
        assert report.n_observations == 49
        assert len(report.feature_names) == 5
        assert report.signal.shape == (49, 5)
        assert len(report.drift_scores) == 49

    def test_stable_no_recalibrate(self, stable_snapshots):
        det = DriftDetector(method="pelt")
        report = det.detect(stable_snapshots)
        assert report.recalibrate_recommended is False
        assert len(report.changepoints) == 0

    def test_sudden_shift_recalibrate(self, sudden_snapshots):
        det = DriftDetector(method="pelt")
        report = det.detect(sudden_snapshots)
        assert report.recalibrate_recommended is True
        assert len(report.changepoints) >= 1
        cp = report.changepoints[0]
        assert isinstance(cp, ChangePoint)
        assert cp.direction == "degradation"
        assert cp.severity > 0.1
        assert len(cp.features_affected) >= 1

    def test_summary_string(self, sudden_snapshots):
        det = DriftDetector(method="pelt")
        report = det.detect(sudden_snapshots)
        s = report.summary()
        assert "DriftReport" in s
        assert "Recalibrate" in s

    def test_ensemble_method(self, sudden_snapshots):
        det = DriftDetector(method="ensemble")
        report = det.detect(sudden_snapshots)
        assert "ensemble" in report.algorithm
        assert len(report.changepoints) >= 1

    def test_double_shift_two_cps(self, double_shift_snapshots):
        det = DriftDetector(method="pelt")
        report = det.detect(double_shift_snapshots)
        assert len(report.changepoints) >= 2, (
            f"Should detect 2+ CPs for double shift, got {len(report.changepoints)}"
        )
        # First should be degradation, second recovery
        directions = [cp.direction for cp in report.changepoints[:2]]
        assert "degradation" in directions
        assert "recovery" in directions

    def test_drift_scores_bounded(self, sudden_snapshots):
        det = DriftDetector(method="pelt")
        report = det.detect(sudden_snapshots)
        assert np.all(report.drift_scores >= 0)
        assert np.all(report.drift_scores <= 1)
