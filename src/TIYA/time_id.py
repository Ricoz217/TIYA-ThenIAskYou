from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from TIYA.config import TIME_ID_STATE_FILE


_EPOCH_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z
_SEQUENCE_BITS = 12
_SEQUENCE_MASK = (1 << _SEQUENCE_BITS) - 1
_DEFAULT_RESERVATION_SIZE = 4096
_DEFAULT_RESERVATION_DURATION_MS = 60_000
_SAVE_RETRY_COUNT = 3
_SAVE_RETRY_DELAY_SEC = 0.01
_PERSIST_RETRY_BACKOFF_SEC = 1.0

_log = logging.getLogger(__name__)


@dataclass
class _State:
    last_ms: int = 0
    sequence: int = 0
    reserved_through_id: int | None = None


class TimeBasedIdGenerator:
    """Monotonic time-based unique ID generator with persisted state."""

    def __init__(
        self,
        state_file: str | Path | None = None,
        time_ms_fn: Callable[[], int] | None = None,
        reservation_size: int = _DEFAULT_RESERVATION_SIZE,
        reservation_duration_ms: int = _DEFAULT_RESERVATION_DURATION_MS,
    ) -> None:
        if reservation_size <= 0:
            raise ValueError("reservation_size must be greater than zero")
        if reservation_duration_ms < 0:
            raise ValueError("reservation_duration_ms must not be negative")

        self._state_file = Path(state_file) if state_file else TIME_ID_STATE_FILE
        self._time_ms_fn = time_ms_fn or (lambda: time.time_ns() // 1_000_000)
        self._reservation_size = reservation_size
        self._reservation_duration_ms = reservation_duration_ms
        self._lock = threading.Lock()
        self._state = self._load_state()
        self._persist_degraded = False
        self._next_persist_retry_at = 0.0

    def next_id(self) -> int:
        with self._lock:
            now_ms = self._time_ms_fn()
            current_ms = now_ms if now_ms > self._state.last_ms else self._state.last_ms

            if current_ms == self._state.last_ms:
                next_seq = self._state.sequence + 1
                if next_seq > _SEQUENCE_MASK:
                    current_ms += 1
                    next_seq = 0
            else:
                next_seq = 0

            self._state.last_ms = current_ms
            self._state.sequence = next_seq
            new_id = ((current_ms - _EPOCH_MS) << _SEQUENCE_BITS) | next_seq
            reserved_through = self._state.reserved_through_id
            if reserved_through is None or new_id > reserved_through:
                reserved_time_ms = now_ms + self._reservation_duration_ms
                reserved_time_id = (
                    ((reserved_time_ms - _EPOCH_MS) << _SEQUENCE_BITS)
                    | _SEQUENCE_MASK
                )
                reserved_through = max(
                    new_id + self._reservation_size - 1,
                    reserved_time_id,
                )
                if time.monotonic() >= self._next_persist_retry_at:
                    if self._save_state(reserved_through):
                        self._state.reserved_through_id = reserved_through
                        self._next_persist_retry_at = 0.0
                    else:
                        self._next_persist_retry_at = (
                            time.monotonic() + _PERSIST_RETRY_BACKOFF_SEC
                        )
            return new_id

    def _load_state(self) -> _State:
        if not self._state_file.exists():
            return _State()
        try:
            payload = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return _State()

        if "reserved_through_id" in payload:
            reserved_through_id = int(payload["reserved_through_id"])
            return _State(
                last_ms=(reserved_through_id >> _SEQUENCE_BITS) + _EPOCH_MS,
                sequence=reserved_through_id & _SEQUENCE_MASK,
                reserved_through_id=reserved_through_id,
            )

        # Migrate the original exact-position state to a one-item reservation.
        last_ms = int(payload.get("last_ms", 0))
        sequence = int(payload.get("sequence", 0))
        reserved_through_id = None
        if last_ms:
            reserved_through_id = ((last_ms - _EPOCH_MS) << _SEQUENCE_BITS) | sequence
        return _State(
            last_ms=last_ms,
            sequence=sequence,
            reserved_through_id=reserved_through_id,
        )

    def _save_state(self, reserved_through_id: int) -> bool:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._persist_degraded = True
            _log.warning(
                "time_id state directory is unavailable (%s): %s",
                self._state_file.parent,
                exc,
            )
            return False

        reserved_ms = (reserved_through_id >> _SEQUENCE_BITS) + _EPOCH_MS
        reserved_sequence = reserved_through_id & _SEQUENCE_MASK
        payload = {
            "version": 2,
            "reserved_through_id": reserved_through_id,
            # Keep legacy fields at the reserved boundary for safe code rollback.
            "last_ms": reserved_ms,
            "sequence": reserved_sequence,
        }
        payload_text = json.dumps(payload, ensure_ascii=True)
        last_error: OSError | None = None

        for attempt in range(1, _SAVE_RETRY_COUNT + 1):
            temp_file = self._state_file.with_suffix(
                f"{self._state_file.suffix}.{os.getpid()}.{threading.get_ident()}.{attempt}.tmp"
            )
            try:
                temp_file.write_text(payload_text, encoding="utf-8")
                os.replace(temp_file, self._state_file)
                if self._persist_degraded:
                    _log.warning("time_id state persistence recovered: %s", self._state_file)
                    self._persist_degraded = False
                return True
            except OSError as exc:
                last_error = exc
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except OSError:
                    pass
                if attempt < _SAVE_RETRY_COUNT:
                    time.sleep(_SAVE_RETRY_DELAY_SEC)

        self._persist_degraded = True
        _log.warning(
            "time_id state persistence failed after %s attempts (%s): %s",
            _SAVE_RETRY_COUNT,
            self._state_file,
            last_error,
        )
        return False


_GLOBAL_GENERATOR: TimeBasedIdGenerator | None = None
_GLOBAL_LOCK = threading.Lock()


def get_global_time_id_generator(
    state_file: str | Path | None = None,
) -> TimeBasedIdGenerator:
    global _GLOBAL_GENERATOR
    if _GLOBAL_GENERATOR is not None:
        return _GLOBAL_GENERATOR

    with _GLOBAL_LOCK:
        if _GLOBAL_GENERATOR is None:
            _GLOBAL_GENERATOR = TimeBasedIdGenerator(state_file=state_file)
    return _GLOBAL_GENERATOR


def next_time_id() -> int:
    return get_global_time_id_generator().next_id()
