from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .runtime import RelatednessRuntime


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported audit value: {type(value).__name__}")


def _append_records(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        for record in records
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
        stream.write("\n")


class RelatednessAudit:
    """Lossy, non-blocking online audit stream for offline parameter analysis."""

    def __init__(
        self,
        *,
        data_path: str | Path,
        group_id: str,
        bot_id: str,
        runtime: RelatednessRuntime,
        session_id: str | None = None,
        queue_limit: int = 4_096,
        batch_size: int = 64,
        flush_interval: float = 1.0,
    ) -> None:
        if queue_limit <= 0 or batch_size <= 0 or flush_interval <= 0:
            raise ValueError("audit queue, batch and flush values must be positive")
        self.group_id = str(group_id)
        self.bot_id = str(bot_id)
        self.session_id = session_id or (
            time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        self.path = Path(data_path) / f"relatedness_audit_{self.session_id}.jsonl"
        self._runtime = runtime
        self._queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue(queue_limit)
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._sequence = 0
        self._dropped_events = 0
        self._task: asyncio.Task[None] | None = None
        self._stop_requested = asyncio.Event()
        self._write_error: str | None = None
        self._closed = False

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def write_error(self) -> str | None:
        return self._write_error

    def set_bot_id(self, bot_id: str) -> None:
        self.bot_id = str(bot_id)

    async def start(self, metadata: dict[str, Any] | None = None) -> None:
        if self._closed:
            raise RuntimeError("relatedness audit is closed")
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run(),
            name=f"relatedness-audit-{self.group_id}",
        )
        self.emit("session_start", metadata or {})

    def emit(self, event: str, payload: dict[str, Any]) -> bool:
        if self._closed:
            return False
        record = self._record(event, payload)
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._dropped_events += 1
            return False
        return True

    async def close(self) -> None:
        if self._closed:
            return
        if self._task is None:
            self._closed = True
            return
        self.emit("session_end", {"dropped_events": self._dropped_events})
        self._closed = True
        self._stop_requested.set()
        try:
            await self._task
        except Exception as exc:
            self._write_error = f"{type(exc).__name__}: {exc}"

    def _record(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._sequence += 1
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "sequence": self._sequence,
            "observed_at": time.time(),
            "monotonic_ns": time.monotonic_ns(),
            "group_id": self.group_id,
            "bot_id": self.bot_id,
            "event": event,
            "payload": payload,
        }

    async def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        while True:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=self._flush_interval,
                )
            except TimeoutError:
                item = None

            if isinstance(item, dict):
                batch.append(item)
            if batch and (item is None or len(batch) >= self._batch_size):
                await self._flush(batch)
            if self._stop_requested.is_set() and self._queue.empty():
                if batch:
                    await self._flush(batch)
                return

    async def _flush(self, batch: list[dict[str, Any]]) -> None:
        records = tuple(batch)
        batch.clear()
        await self._runtime.run_maintenance(
            f"{self.group_id}:{self.session_id}:audit",
            _append_records,
            self.path,
            records,
        )
