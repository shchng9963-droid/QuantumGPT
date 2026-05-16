"""StreamingAdvisor — real-time telemetry stream + advisory-in-the-loop.

Connects PropertiesStream to CalibrationAdvisor for continuous monitoring.
When the stream emits a snapshot, the advisor evaluates it against a sliding
window of recent history and fires alerts when thresholds are crossed.

Two modes:
  1. Accelerated simulation: time_acceleration > 1, useful for testing
     (e.g. 1 real second = 1 simulated hour)
  2. Real-time: time_acceleration = 1, for production monitoring

Alert system:
  - Configurable thresholds (drift, fidelity, T1)
  - Callback-based: register functions that receive Alert objects
  - Cooldown to prevent alert storms
  - Alert history for post-hoc analysis

Usage:
    from advisor.streaming import StreamingAdvisor, AlertConfig
    from backends.synthetic_drift import SyntheticDriftBackend, LINEAR_DECAY

    backend = SyntheticDriftBackend("FakeBrisbane", LINEAR_DECAY)
    sa = StreamingAdvisor(backend, alert_config=AlertConfig(drift_critical=0.5))
    sa.on_alert(lambda alert: print(f"ALERT: {alert}"))
    sa.start(interval_seconds=0.5, time_acceleration=3600, duration_seconds=10)
    print(sa.get_alert_history())
"""

import json
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from backends.base import ShadowBackend
from backends.properties_stream import PropertiesStream
from data.store import DataStore


# ═══════════════════════════════════════════════════════
# Alert system
# ═══════════════════════════════════════════════════════

class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    DRIFT_HIGH = "drift_high"
    DRIFT_SPIKE = "drift_spike"
    T1_DEGRADING = "t1_degrading"
    T1_LOW = "t1_low"
    FIDELITY_DROP = "fidelity_drop"
    ERROR_HIGH = "error_high"
    CHANGEPOINT = "changepoint"
    RECOVERY = "recovery"


@dataclass
class Alert:
    """A single alert emitted by the streaming advisor."""
    timestamp: str
    sim_time_hours: float
    level: AlertLevel
    alert_type: AlertType
    message: str
    details: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Alert({self.level.value}:{self.alert_type.value} @ t={self.sim_time_hours:.1f}h — {self.message})"


@dataclass
class AlertConfig:
    """Thresholds and cooldowns for the alert system."""
    # Drift thresholds
    drift_warning: float = 0.2
    drift_critical: float = 0.5
    drift_spike_threshold: float = 0.15  # jump between consecutive snapshots

    # T1 thresholds (us)
    t1_warning: float = 100.0  # below this → warning
    t1_critical: float = 50.0  # below this → critical
    t1_degradation_rate: float = -5.0  # us/hour

    # Error thresholds
    error_2q_warning: float = 0.03
    error_2q_critical: float = 0.06

    # Cooldowns (in simulated hours)
    alert_cooldown_hours: float = 1.0  # min time between same alert type

    # Sliding window size for trend analysis
    window_size: int = 5


# ═══════════════════════════════════════════════════════
# Streaming health tracker (sliding window analysis)
# ═══════════════════════════════════════════════════════

class HealthTracker:
    """Maintains a sliding window of health snapshots for real-time analysis."""

    def __init__(self, window_size: int = 10):
        self._window: deque[dict] = deque(maxlen=window_size)
        self._prev_snapshot: Optional[dict] = None

    def add(self, snapshot: dict):
        """Add a snapshot and update the window."""
        self._prev_snapshot = self._window[-1] if self._window else None
        self._window.append(snapshot)

    @property
    def window(self) -> list[dict]:
        return list(self._window)

    @property
    def prev(self) -> Optional[dict]:
        return self._prev_snapshot

    @property
    def latest(self) -> Optional[dict]:
        return self._window[-1] if self._window else None

    @property
    def size(self) -> int:
        return len(self._window)

    def t1_slope(self) -> float:
        """Compute T1 slope over the window (us/hour)."""
        if len(self._window) < 2:
            return 0.0
        times = [s.get("stream_sim_time_hours", 0) for s in self._window]
        t1s = [s.get("avg_t1_us", 0) for s in self._window]
        if np.std(times) == 0:
            return 0.0
        return float(np.polyfit(times, t1s, 1)[0])

    def drift_slope(self) -> float:
        """Compute drift slope over the window."""
        if len(self._window) < 2:
            return 0.0
        times = [s.get("stream_sim_time_hours", 0) for s in self._window]
        drifts = [s.get("drift_score", 0) or 0 for s in self._window]
        if np.std(times) == 0:
            return 0.0
        return float(np.polyfit(times, drifts, 1)[0])

    def drift_jump(self) -> float:
        """Compute drift jump from previous to current snapshot."""
        if self._prev_snapshot is None or not self._window:
            return 0.0
        prev_drift = self._prev_snapshot.get("drift_score", 0) or 0
        curr_drift = self._window[-1].get("drift_score", 0) or 0
        return curr_drift - prev_drift

    def error_2q_slope(self) -> float:
        """Compute 2Q error slope over the window."""
        if len(self._window) < 2:
            return 0.0
        times = [s.get("stream_sim_time_hours", 0) for s in self._window]
        errs = [s.get("avg_2q_error", 0) for s in self._window]
        if np.std(times) == 0:
            return 0.0
        return float(np.polyfit(times, errs, 1)[0])


# ═══════════════════════════════════════════════════════
# StreamingAdvisor
# ═══════════════════════════════════════════════════════

class StreamingAdvisor:
    """Real-time streaming advisor: PropertiesStream + alert engine.

    Connects to a PropertiesStream and evaluates every snapshot against
    configurable thresholds. Fires alerts via registered callbacks.

    Args:
        backend: ShadowBackend instance (must support set_time())
        alert_config: Alert thresholds and cooldowns
        db_path: Optional DuckDB path for persistence
    """

    def __init__(
        self,
        backend: ShadowBackend,
        alert_config: Optional[AlertConfig] = None,
        db_path: Optional[str] = None,
    ):
        self.backend = backend
        self.config = alert_config or AlertConfig()
        self.db = DataStore(db_path) if db_path else None

        self._tracker = HealthTracker(window_size=self.config.window_size)
        self._alert_callbacks: list[Callable[[Alert], None]] = []
        self._alert_history: list[Alert] = []
        self._last_alert_time: dict[str, float] = {}  # alert_type → sim_time
        self._stream: Optional[PropertiesStream] = None
        self._snapshot_count = 0
        self._was_critical = False  # track for recovery alerts

    # ─── Alert registration ───────────────────────────────

    def on_alert(self, callback: Callable[[Alert], None]):
        """Register an alert callback."""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable):
        """Remove an alert callback."""
        self._alert_callbacks = [c for c in self._alert_callbacks if c is not callback]

    # ─── Alert emission ───────────────────────────────────

    def _emit_alert(self, level: AlertLevel, alert_type: AlertType,
                     message: str, sim_time: float, details: dict = None):
        """Create and emit an alert if not in cooldown."""
        # Check cooldown
        last = self._last_alert_time.get(alert_type.value, -999)
        if sim_time - last < self.config.alert_cooldown_hours:
            return  # In cooldown

        alert = Alert(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sim_time_hours=sim_time,
            level=level,
            alert_type=alert_type,
            message=message,
            details=details or {},
        )

        self._alert_history.append(alert)
        self._last_alert_time[alert_type.value] = sim_time

        # Persist to DuckDB as a drift event
        if self.db:
            self.db.log_drift_event({
                "backend": self.backend.name,
                "drift_score": details.get("drift_score") if details else None,
                "severity": level.value,
                "suggestions": [message],
            })

        # Fire callbacks
        for cb in self._alert_callbacks:
            try:
                cb(alert)
            except Exception:
                pass

    # ─── Snapshot evaluation ──────────────────────────────

    def _evaluate_snapshot(self, snapshot: dict):
        """Evaluate a single snapshot against thresholds and trends."""
        self._tracker.add(snapshot)
        self._snapshot_count += 1

        sim_time = snapshot.get("stream_sim_time_hours", 0)
        drift = snapshot.get("drift_score", 0) or 0
        t1 = snapshot.get("avg_t1_us", 999)
        error_2q = snapshot.get("avg_2q_error", 0)

        is_critical = False

        # ── Drift checks ──

        # Drift spike (sudden jump between consecutive snapshots)
        drift_jump = self._tracker.drift_jump()
        if drift_jump > self.config.drift_spike_threshold:
            is_critical = True
            self._emit_alert(
                AlertLevel.CRITICAL,
                AlertType.DRIFT_SPIKE,
                f"Drift spiked by +{drift_jump:.3f} (from {drift - drift_jump:.3f} to {drift:.3f})",
                sim_time,
                {"drift_score": drift, "drift_jump": drift_jump},
            )

        # High drift level
        if drift > self.config.drift_critical:
            is_critical = True
            self._emit_alert(
                AlertLevel.CRITICAL,
                AlertType.DRIFT_HIGH,
                f"Drift score {drift:.3f} exceeds critical threshold ({self.config.drift_critical})",
                sim_time,
                {"drift_score": drift},
            )
        elif drift > self.config.drift_warning:
            self._emit_alert(
                AlertLevel.WARNING,
                AlertType.DRIFT_HIGH,
                f"Drift score {drift:.3f} exceeds warning threshold ({self.config.drift_warning})",
                sim_time,
                {"drift_score": drift},
            )

        # ── T1 checks ──

        if t1 < self.config.t1_critical:
            is_critical = True
            self._emit_alert(
                AlertLevel.CRITICAL,
                AlertType.T1_LOW,
                f"Avg T1 = {t1:.1f} μs is below critical threshold ({self.config.t1_critical} μs)",
                sim_time,
                {"avg_t1_us": t1},
            )
        elif t1 < self.config.t1_warning:
            self._emit_alert(
                AlertLevel.WARNING,
                AlertType.T1_LOW,
                f"Avg T1 = {t1:.1f} μs is below warning threshold ({self.config.t1_warning} μs)",
                sim_time,
                {"avg_t1_us": t1},
            )

        # T1 degradation rate (need ≥2 snapshots)
        if self._tracker.size >= 3:
            t1_rate = self._tracker.t1_slope()
            if t1_rate < self.config.t1_degradation_rate:
                self._emit_alert(
                    AlertLevel.WARNING,
                    AlertType.T1_DEGRADING,
                    f"T1 degrading at {t1_rate:+.2f} μs/h (threshold: {self.config.t1_degradation_rate})",
                    sim_time,
                    {"t1_slope": t1_rate, "avg_t1_us": t1},
                )

        # ── Error checks ──

        if error_2q > self.config.error_2q_critical:
            is_critical = True
            self._emit_alert(
                AlertLevel.CRITICAL,
                AlertType.ERROR_HIGH,
                f"2Q error {error_2q:.5f} exceeds critical threshold ({self.config.error_2q_critical})",
                sim_time,
                {"avg_2q_error": error_2q},
            )
        elif error_2q > self.config.error_2q_warning:
            self._emit_alert(
                AlertLevel.WARNING,
                AlertType.ERROR_HIGH,
                f"2Q error {error_2q:.5f} exceeds warning threshold ({self.config.error_2q_warning})",
                sim_time,
                {"avg_2q_error": error_2q},
            )

        # ── Recovery detection ──
        if self._was_critical and not is_critical:
            self._emit_alert(
                AlertLevel.INFO,
                AlertType.RECOVERY,
                f"Device recovered: drift={drift:.3f}, T1={t1:.1f}μs, 2Q_err={error_2q:.5f}",
                sim_time,
                {"drift_score": drift, "avg_t1_us": t1, "avg_2q_error": error_2q},
            )

        self._was_critical = is_critical

        # Persist snapshot to DuckDB
        if self.db:
            self.db.log_health(snapshot)

    # ─── Start/stop ───────────────────────────────────────

    def start(
        self,
        interval_seconds: float = 1.0,
        time_acceleration: float = 3600.0,
        duration_seconds: Optional[float] = None,
    ) -> "StreamingAdvisor":
        """Start the streaming advisory loop.

        Args:
            interval_seconds: Real-time interval between emissions
            time_acceleration: Simulated time multiplier (3600 = 1s real = 1h sim)
            duration_seconds: If set, run for this many real seconds then stop.
                            If None, run until stop() is called.

        Returns:
            self (for chaining)
        """
        self._stream = PropertiesStream(
            self.backend,
            interval_seconds=interval_seconds,
            time_acceleration=time_acceleration,
        )
        self._stream.on_snapshot(self._evaluate_snapshot)
        self._stream.start()

        if duration_seconds is not None:
            time.sleep(duration_seconds)
            self.stop()

        return self

    def stop(self):
        """Stop the stream."""
        if self._stream is not None:
            self._stream.stop()

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.is_running

    # ─── Batch mode (synchronous, no threading) ───────────

    def run_batch(
        self,
        time_points: list[float],
    ) -> list[Alert]:
        """Evaluate a sequence of time points synchronously.

        Useful for testing or replaying historical data without threads.

        Args:
            time_points: Simulated hours to evaluate

        Returns:
            List of alerts generated during the run
        """
        initial_alert_count = len(self._alert_history)

        for t in time_points:
            if hasattr(self.backend, "set_time"):
                self.backend.set_time(t)
            snapshot = self.backend.get_properties_snapshot()
            snapshot["stream_sim_time_hours"] = t
            self._evaluate_snapshot(snapshot)

        return self._alert_history[initial_alert_count:]

    # ─── Query ────────────────────────────────────────────

    def get_alert_history(self, level: Optional[AlertLevel] = None,
                          last_n: Optional[int] = None) -> list[Alert]:
        """Get alert history, optionally filtered by level."""
        alerts = self._alert_history
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        if last_n is not None:
            alerts = alerts[-last_n:]
        return alerts

    def get_snapshot_history(self, last_n: Optional[int] = None) -> list[dict]:
        """Get the sliding window of recent snapshots."""
        return self._tracker.window if last_n is None else self._tracker.window[-last_n:]

    @property
    def snapshot_count(self) -> int:
        return self._snapshot_count

    @property
    def alert_count(self) -> int:
        return len(self._alert_history)

    def get_summary(self) -> dict:
        """Return a summary of the streaming session."""
        alerts = self._alert_history
        return {
            "backend": self.backend.name,
            "total_snapshots": self._snapshot_count,
            "total_alerts": len(alerts),
            "alerts_by_level": {
                "info": sum(1 for a in alerts if a.level == AlertLevel.INFO),
                "warning": sum(1 for a in alerts if a.level == AlertLevel.WARNING),
                "critical": sum(1 for a in alerts if a.level == AlertLevel.CRITICAL),
            },
            "alerts_by_type": {
                t.value: sum(1 for a in alerts if a.alert_type == t)
                for t in AlertType
                if sum(1 for a in alerts if a.alert_type == t) > 0
            },
            "latest_health": self._tracker.latest,
            "t1_slope": self._tracker.t1_slope(),
            "drift_slope": self._tracker.drift_slope(),
        }
