"""Properties Stream — mock telemetry that emits calibration snapshots.

Simulates the real-time data feed from a quantum device's calibration
system. The agent's drift detector and perception tools consume this stream.

Two modes:
  - Realtime: emits at wall-clock intervals (for live demo / dashboards)
  - Accelerated: emits a full history instantly (for evaluation / testing)

Usage:
    from backends.properties_stream import PropertiesStream

    stream = PropertiesStream(backend, interval_seconds=4.0)
    stream.start()

    # Consume snapshots
    for snapshot in stream.iter_snapshots():
        print(snapshot["drift_score"])
        if should_stop:
            break
    stream.stop()

    # Or: get all snapshots from a replay at once
    snapshots = PropertiesStream.replay_all(backend, hours=168, step_hours=1.0)
"""

import threading
import time
from collections import deque
from typing import Any, Generator

from backends.base import ShadowBackend


class PropertiesStream:
    """Emits calibration snapshots from a ShadowBackend at regular intervals.

    For ReplayBackend/SyntheticDriftBackend, it advances the backend's
    internal time with each tick, simulating real-time hardware telemetry.

    Thread-safe: can be consumed from a different thread than the emitter.
    """

    def __init__(
        self,
        backend: ShadowBackend,
        interval_seconds: float = 4.0,
        time_acceleration: float = 1.0,
        max_buffer_size: int = 10000,
    ):
        """
        Args:
            backend: The shadow backend to stream from.
            interval_seconds: Wall-clock seconds between emissions.
            time_acceleration: How many simulated hours pass per wall-clock second.
                E.g., 3600 means 1 wall-second = 1 sim-hour.
            max_buffer_size: Max snapshots to buffer before dropping oldest.
        """
        self._backend = backend
        self._interval = interval_seconds
        self._acceleration = time_acceleration
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_buffer_size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sim_time = 0.0  # simulated hours elapsed

    @property
    def backend(self) -> ShadowBackend:
        return self._backend

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def sim_time(self) -> float:
        """Current simulated time in hours."""
        return self._sim_time

    def start(self) -> None:
        """Start emitting snapshots in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._emit_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the emission thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _emit_loop(self) -> None:
        """Background loop: advance time and emit snapshots."""
        while self._running:
            # Advance simulated time
            self._sim_time += self._interval * self._acceleration / 3600.0

            # Set backend time if it supports it
            if hasattr(self._backend, "set_time"):
                self._backend.set_time(self._sim_time)

            # Get snapshot
            snapshot = self._backend.get_properties_snapshot()
            snapshot["stream_sim_time_hours"] = self._sim_time

            with self._lock:
                self._buffer.append(snapshot)

            time.sleep(self._interval)

    def get_latest(self) -> dict[str, Any] | None:
        """Get the most recent snapshot (non-blocking)."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_all(self) -> list[dict[str, Any]]:
        """Get all buffered snapshots and clear the buffer."""
        with self._lock:
            snapshots = list(self._buffer)
            self._buffer.clear()
            return snapshots

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Get the N most recent snapshots (does not clear buffer)."""
        with self._lock:
            return list(self._buffer)[-n:]

    def iter_snapshots(self) -> Generator[dict[str, Any], None, None]:
        """Blocking iterator that yields new snapshots as they arrive."""
        last_size = 0
        while self._running:
            current_size = len(self._buffer)
            if current_size > last_size:
                with self._lock:
                    new_items = list(self._buffer)[last_size:]
                last_size = current_size
                for item in new_items:
                    yield item
            else:
                time.sleep(self._interval / 2)

    @staticmethod
    def replay_all(
        backend: ShadowBackend,
        hours: float,
        step_hours: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Generate all snapshots from a replay/drift backend at once.

        Non-blocking, returns immediately. Useful for evaluation and testing.
        """
        snapshots = []
        t = 0.0
        while t <= hours:
            if hasattr(backend, "set_time"):
                backend.set_time(t)
            snap = backend.get_properties_snapshot()
            snap["stream_sim_time_hours"] = t
            snapshots.append(snap)
            t += step_hours
        return snapshots
