from __future__ import annotations

import asyncio
import sys
import threading
import time
import traceback
from typing import Protocol


class MonitorLogger(Protocol):
    def info(self, message: str) -> object:
        ...

    def warning(self, message: str) -> object:
        ...


class EventLoopLagMonitor:
    """Temporarily watch one asyncio loop for multi-second scheduling stalls."""

    def __init__(
        self,
        *,
        logger: MonitorLogger,
        threshold: float = 3.0,
        probe_interval: float = 0.25,
        stack_limit: int = 30,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be greater than zero")

        if probe_interval <= 0:
            raise ValueError("probe_interval must be greater than zero")

        if stack_limit <= 0:
            raise ValueError("stack_limit must be greater than zero")

        self._logger = logger
        self._threshold = float(threshold)
        self._probe_interval = float(probe_interval)
        self._stack_limit = int(stack_limit)
        self._stop_event = threading.Event()
        self._pending_lock = threading.Lock()
        self._pending_probe: threading.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread_id: int | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    async def start(self) -> None:
        """Start the watchdog for the currently running event loop."""
        if self.is_alive:
            return

        if self._stop_event.is_set():
            raise RuntimeError("a closed event-loop monitor cannot be restarted")

        self._loop = asyncio.get_running_loop()
        self._loop_thread_id = threading.get_ident()
        self._thread = threading.Thread(
            target=self._watch,
            name="event-loop-watchdog",
            daemon=True,
        )
        self._thread.start()
        self._safe_log(
            "info",
            "[EVENT_LOOP_MONITOR] started "
            f"threshold={self._threshold:.3f}s "
            f"probe_interval={self._probe_interval:.3f}s",
        )

    async def close(self) -> None:
        """Stop the watchdog without blocking the monitored event loop."""
        self.stop()

        thread = self._thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, max(1.0, self._probe_interval * 2))

    def stop(self) -> None:
        """Signal the watchdog from synchronous shutdown code."""
        self._stop_event.set()
        with self._pending_lock:
            pending = self._pending_probe

        if pending is not None:
            pending.set()

    def _watch(self) -> None:
        loop = self._loop
        if loop is None:
            return

        incident_count = 0
        max_delay = 0.0
        while not self._stop_event.wait(self._probe_interval):
            acknowledged = threading.Event()
            scheduled_at = time.monotonic()
            with self._pending_lock:
                self._pending_probe = acknowledged

            try:
                loop.call_soon_threadsafe(acknowledged.set)

            except RuntimeError:
                break

            if acknowledged.wait(self._threshold):
                self._clear_pending(acknowledged)
                continue

            if self._stop_event.is_set():
                break

            incident_count += 1
            delay_at_detection = time.monotonic() - scheduled_at
            max_delay = max(max_delay, delay_at_detection)
            stack = self._capture_event_loop_stack()
            self._safe_log(
                "warning",
                "[EVENT_LOOP_BLOCKED] "
                f"incident={incident_count} "
                f"threshold={self._threshold:.3f}s "
                f"delay_at_detection={delay_at_detection:.3f}s "
                f"max_delay={max_delay:.3f}s\n"
                "Event loop thread stack at detection:\n"
                f"{stack}",
            )

            while not self._stop_event.is_set():
                if acknowledged.wait(0.1):
                    break

            if self._stop_event.is_set():
                break

            total_delay = time.monotonic() - scheduled_at
            max_delay = max(max_delay, total_delay)
            self._safe_log(
                "warning",
                "[EVENT_LOOP_RECOVERED] "
                f"incident={incident_count} "
                f"total_delay={total_delay:.3f}s "
                f"max_delay={max_delay:.3f}s",
            )
            self._clear_pending(acknowledged)

        with self._pending_lock:
            self._pending_probe = None

    def _capture_event_loop_stack(self) -> str:
        thread_id = self._loop_thread_id
        if thread_id is None:
            return "<event-loop thread id unavailable>"

        frame = sys._current_frames().get(thread_id)
        if frame is None:
            return "<event-loop stack unavailable>"

        return "".join(
            traceback.format_stack(frame, limit=self._stack_limit)
        ).rstrip()

    def _clear_pending(self, acknowledged: threading.Event) -> None:
        with self._pending_lock:
            if self._pending_probe is acknowledged:
                self._pending_probe = None

    def _safe_log(self, level: str, message: str) -> None:
        try:
            getattr(self._logger, level)(message)

        except Exception:
            pass


__all__ = ["EventLoopLagMonitor"]
