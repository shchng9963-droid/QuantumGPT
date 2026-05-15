"""Properties Stream — mock real-time telemetry from shadow backends.

Periodically emits calibration snapshots from a ShadowBackend,
simulating the real-time telemetry feed that a physical QPU provides.

The stream supports:
  - Multiple listeners (observer pattern)
  - Configurable emit interval
  - Background thread (non-blocking)
  - History buffer for replay/analysis

Usage:
    from backends import FakeBackendAdapter
    from backends.properties_stream import PropertiesStream

    backend = FakeBackendAdapter("FakeBrisbane")
    stream = PropertiesStream(backend, interval=4.0)

    # Register a callback
    stream.on_snapshot(lambda snap: print(f"T1={snap['avg_t1_us']:.1f}"))

    stream.start()
    # ... do work ...
    stream.stop()

    # Get history
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
    ):
        """
        Args:
            backend: The shadow backend to poll
            interval: Seconds between emissions
            history_size: Max snapshots to keep in memory
        """
        self._backend = backend
        self._interval = interval
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._emit_count = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def emit_count(self) -> int:
        return self._emit_count

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
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        """Stop the stream and wait for the thread to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def emit_once(self) -> dict[str, Any]:
        """Manually emit a single snapshot (synchronous). 
        
        Useful for testing or on-demand polling.
        """
        snapshot = self._backend.get_properties_snapshot()
        snapshot["stream_seq"] = self._emit_count
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
        """Background loop: emit snapshots at fixed intervals."""
        while not self._stop_event.is_set():
            self.emit_once()
            self._stop_event.wait(timeout=self._interval)
