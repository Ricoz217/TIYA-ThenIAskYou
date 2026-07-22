"""Singleton command input hub with queue/broadcast delivery modes.

This module is intentionally independent from Rich/prompt_toolkit.
It can run a native ``input()`` loop in a dedicated daemon thread and
provide both sync and async read APIs without blocking caller threads.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Literal

__all__ = ["InputHubConfig", "InputHub", "get_input_hub", "reset_input_hub"]

InputMode = Literal["queue", "broadcast"]


@dataclass(slots=True)
class InputHubConfig:
    """Configuration for :class:`InputHub`."""

    prompt: str = "> "
    mode: InputMode = "queue"
    echo_to_logger: bool = True
    history_limit: int = 2000
    broadcast_buffer_limit: int = 2000


class InputHub:
    """Thread-safe singleton-style command input center.

    Delivery modes:
    - ``queue``: commands are consumed FIFO by any caller.
    - ``broadcast``: each subscriber consumes its own command stream.
    """

    def __init__(self, config: InputHubConfig | None = None) -> None:
        self._config = config or InputHubConfig()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._queue: Deque[str] = deque()
        self._history: Deque[str] = deque(maxlen=max(1, self._config.history_limit))
        self._broadcast: Deque[tuple[int, str]] = deque(maxlen=max(1, self._config.broadcast_buffer_limit))
        self._subscribers: dict[str, int] = {}
        self._sequence = 0
        self._last_error: str | None = None
        self._logger: Any | None = None

    @property
    def mode(self) -> InputMode:
        with self._lock:
            return self._config.mode

    def configure(
        self,
        *,
        prompt: str | None = None,
        mode: InputMode | None = None,
        echo_to_logger: bool | None = None,
        history_limit: int | None = None,
        broadcast_buffer_limit: int | None = None,
    ) -> None:
        """Update runtime configuration."""
        with self._lock:
            if prompt is not None:
                self._config.prompt = prompt
            if mode is not None:
                self._config.mode = mode
            if echo_to_logger is not None:
                self._config.echo_to_logger = bool(echo_to_logger)
            if history_limit is not None and history_limit > 0:
                self._config.history_limit = history_limit
                self._history = deque(self._history, maxlen=history_limit)
            if broadcast_buffer_limit is not None and broadcast_buffer_limit > 0:
                self._config.broadcast_buffer_limit = broadcast_buffer_limit
                self._broadcast = deque(self._broadcast, maxlen=broadcast_buffer_limit)
            self._condition.notify_all()

    def set_logger(self, logger: Any | None) -> None:
        """Set optional logger-like object for input echoing.

        Logger contract:
        - Must expose ``info(str, **kwargs)``.
        """
        with self._lock:
            self._logger = logger

    def start(self) -> None:
        """Start native input reader thread (idempotent)."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._reader_loop, name="input-hub-reader", daemon=True)
            self._thread.start()

    def stop(self, timeout: float | None = 0.2) -> None:
        """Request reader stop and wake waiting readers.

        Note: native ``input()`` cannot be forcefully interrupted on all platforms.
        The thread usually exits after next newline/EOF.
        """
        self._stop_event.set()
        with self._lock:
            self._condition.notify_all()
            thread = self._thread
        if thread and timeout is not None:
            thread.join(timeout=timeout)

    def running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def subscribe(self, subscriber_id: str) -> None:
        """Register or reset a broadcast subscriber cursor to latest."""
        with self._lock:
            self._subscribers[subscriber_id] = self._sequence + 1

    def unsubscribe(self, subscriber_id: str) -> None:
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def publish(self, command: str) -> None:
        """Publish one command into hub storage and wake readers."""
        cmd = str(command)
        with self._lock:
            self._sequence += 1
            seq = self._sequence
            self._queue.append(cmd)
            self._broadcast.append((seq, cmd))
            self._history.append(cmd)
            self._condition.notify_all()

        self._echo(cmd)

    def read(
        self,
        timeout: float | None = None,
        *,
        subscriber_id: str = "default",
    ) -> str | None:
        """Read command using current configured mode."""
        if self.mode == "broadcast":
            return self.read_broadcast(subscriber_id=subscriber_id, timeout=timeout)
        return self.read_queue(timeout=timeout)

    async def read_async(
        self,
        timeout: float | None = None,
        *,
        subscriber_id: str = "default",
    ) -> str | None:
        """Async wrapper for :meth:`read` using ``asyncio.to_thread``."""
        return await asyncio.to_thread(self.read, timeout, subscriber_id=subscriber_id)

    def read_queue(self, timeout: float | None = None) -> str | None:
        """Read FIFO command; returns ``None`` on timeout/stop."""
        deadline = None if timeout is None else (time.monotonic() + max(0.0, timeout))
        with self._condition:
            while not self._queue and not self._stop_event.is_set():
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()
            if self._queue:
                return self._queue.popleft()
            return None

    def read_broadcast(self, *, subscriber_id: str, timeout: float | None = None) -> str | None:
        """Read next command for one broadcast subscriber."""
        deadline = None if timeout is None else (time.monotonic() + max(0.0, timeout))
        with self._condition:
            if subscriber_id not in self._subscribers:
                self._subscribers[subscriber_id] = self._sequence + 1

            while not self._stop_event.is_set():
                next_seq = self._subscribers[subscriber_id]
                # buffer may have already dropped older commands; then jump forward.
                if self._broadcast:
                    first_seq = self._broadcast[0][0]
                    if next_seq < first_seq:
                        next_seq = first_seq
                        self._subscribers[subscriber_id] = next_seq
                    for seq, cmd in self._broadcast:
                        if seq >= next_seq:
                            self._subscribers[subscriber_id] = seq + 1
                            return cmd
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()
            return None

    def get_history(self, limit: int | None = None) -> list[str]:
        """Snapshot input history."""
        with self._lock:
            data = list(self._history)
        if limit is None or limit <= 0:
            return data
        return data[-limit:]

    def clear(self) -> None:
        """Clear queue, broadcast buffer, history, and subscriber cursors."""
        with self._lock:
            self._queue.clear()
            self._broadcast.clear()
            self._history.clear()
            self._subscribers.clear()
            self._condition.notify_all()

    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def _echo(self, command: str) -> None:
        with self._lock:
            logger = self._logger
            should_echo = self._config.echo_to_logger
        if not should_echo or logger is None:
            return
        try:
            logger.info(f"> {command}", show_time=False, show_location=False)
        except Exception:
            # Keep input flow resilient if logger side fails.
            return

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                # Always use readline backend to avoid input() prompt repaint conflicts with Rich Live.
                raw = sys.stdin.readline()
                if raw == "":
                    time.sleep(0.05)
                    continue
                command = raw.rstrip("\r\n")
            except EOFError:
                time.sleep(0.05)
                continue
            except Exception as exc:  # pragma: no cover - defensive runtime path
                with self._lock:
                    self._last_error = str(exc)
                time.sleep(0.1)
                continue
            self.publish(command)


_INSTANCE_LOCK = threading.Lock()
_INSTANCE: InputHub | None = None


def get_input_hub(config: InputHubConfig | None = None) -> InputHub:
    """Return process-global :class:`InputHub` singleton."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = InputHub(config=config)
        elif config is not None:
            _INSTANCE.configure(
                prompt=config.prompt,
                mode=config.mode,
                echo_to_logger=config.echo_to_logger,
                history_limit=config.history_limit,
                broadcast_buffer_limit=config.broadcast_buffer_limit,
            )
        return _INSTANCE


def reset_input_hub() -> None:
    """Stop and drop the process-global :class:`InputHub` singleton."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            _INSTANCE.stop(timeout=0.0)
        _INSTANCE = None
