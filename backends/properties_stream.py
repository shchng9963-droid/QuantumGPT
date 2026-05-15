"""Properties Stream — mock real-time telemetry from shadow backends.

Periodically emits calibration snapshots from a ShadowBackend,
simulating the real-time telemetry feed that a physical QPU provides.

The stream supports:
  - Multiple listeners (observer pattern)
  - Configurable emit interval
  - Background thread (non-blocking)
  - History buffer for replay/analysis
  - Static replay_all() for batch time stepping

Usage:
    from backends import FakeBackendAdapter
    from backends.properties_stream import PropertiesStream

    # Batch mode: step through time and collect snapshots
    snapshots = PropertiesStream.replay_all(backend, hours=48, step_hours=12)

    # Streaming mode: background telemetry
    stream = PropertiesStream(backend, interval_seconds=0.1, time_acceleration=3600)
    stream.start()
    # ... do work ...
    stream.stop()
    history = stream.get_history(last_n=10)
"""

import threading
import time
from collections import deque
from typing import Any, Callable, Optional

from backends.base import ShadowBackend


class PropertiesStream:
    """Streams calibration snapshots from a ShadowBackend at fixed intervals.

    Runs in a background daemon thread. Listeners receive snapshots
    as plain dicts (serializable, suitable for logging/storage).
    """

    def __init__(
        self,
        backend: ShadowBackend,
        interval: float = 4.0,
        history_size: int = 1000,
        *,
        interval_seconds: Optional[float] = None,
        time_acceleration: float = 1.0,
    ):
        """
        Args:
            backend: The shadow backend to poll
            interval: Seconds between emissions (legacy, use interval_seconds)
            history_size: Max snapshots to keep in memory
            interval_seconds: Seconds between emissions (preferred param name)
            time_acceleration: How fast simulated time advances relative to real time.
                              E.g., 3600 means 1 real second = 1 simulated hour.
        """
        self._backend = backend
        self._interval = interval_seconds if interval_seconds is not None else interval
        self._time_acceleration = time_acceleration
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._emit_count = 0
        self._wall_start: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def emit_count(self) -> int:
        return self._emit_count

    @property
    def buffer_size(self) -> int:
        """Number of snapshots currently in the history buffer."""
        return len(self._history)

    def on_snapshot(self, callback: Callable[[dict[str, Any]], None]):
        """Register a listener that receives each snapshot dict.

        Callbacks are invoked in the emitter thread — keep them fast
        or offload heavy work to a queue.
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        """Remove a previously registered listener."""
        self._listeners = [l for l in self._listeners if l is not callback]

    def start(self):
        """Start emitting in a background daemon thread."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._wall_start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        """Stop the stream and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def emit_once(self, sim_time_hours: Optional[float] = None) -> dict[str, Any]:
        """Manually emit a single snapshot (synchronous).

        If sim_time_hours is provided, sets backend time first (for backends
        with set_time method).
        """
        if sim_time_hours is not None and hasattr(self._backend, "set_time"):
            self._backend.set_time(sim_time_hours)

        snapshot = self._backend.get_properties_snapshot()
        snapshot["stream_seq"] = self._emit_count
        if sim_time_hours is not None:
            snapshot["stream_sim_time_hours"] = sim_time_hours
        self._emit_count += 1
        self._history.append(snapshot)

        for listener in self._listeners:
            try:
                listener(snapshot)
            except Exception:
                pass  # don't let one bad listener kill the stream

        return snapshot

    def get_history(self, last_n: Optional[int] = None) -> list[dict[str, Any]]:
        """Return recent snapshots from the history buffer.

        Args:
            last_n: Number of most recent snapshots to return (None = all)
        """
        if last_n is None:
            return list(self._history)
        return list(self._history)[-last_n:]

    def get_latest(self) -> Optional[dict[str, Any]]:
        """Return the most recent snapshot, or None if empty."""
        if self._history:
            return self._history[-1]
        return None

    def _run(self):
        """Background loop: emit snapshots at fixed intervals.

        Advances simulated time on backends that support set_time()
        proportional to wall-clock time * time_acceleration.
        """
        while not self._stop_event.is_set():
            # Compute simulated time if backend supports it
            elapsed_wall = time.time() - self._wall_start
            sim_hours = (elapsed_wall * self._time_acceleration) / 3600.0
            self.emit_once(sim_time_hours=sim_hours)
            self._stop_event.wait(timeout=self._interval)

    # ===== Static batch replay =====

    @staticmethod
    def replay_all(
        backend: ShadowBackend,
        hours: float,
        step_hours: float,
    ) -> list[dict[str, Any]]:
        """Step through simulated time and collect all snapshots.

        Requires the backend to have a set_time(hours) method
        (ReplayBackend, SyntheticDriftBackend).

        Args:
            backend: Backend with set_time() method
            hours: Total simulated duration in hours
            step_hours: Step size in hours

        Returns:
            List of snapshot dicts, one per time step (inclusive of endpoints)
        """
        if not hasattr(backend, "set_time"):
            raise TypeError(
                f"Backend {type(backend).__name__} does not support set_time(); "
                "replay_all requires a time-stepping backend."
            )

        snapshots = []
        t = 0.0
        while t <= hours + 1e-9:  # inclusive endpoint
            backend.set_time(t)
            snap = backend.get_properties_snapshot()
            snap["stream_sim_time_hours"] = round(t, 6)
            snapshots.append(snap)
            t += step_hours

        return snapshots
