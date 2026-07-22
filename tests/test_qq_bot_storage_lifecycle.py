from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import TIYA.storage_runtime as storage_runtime


def test_storage_runtime_starts_cache_before_importing_builtin_resources(
        monkeypatch,
) -> None:
    calls: list[str] = []

    async def start_cache() -> None:
        calls.append("start-cache")

    async def ensure_resources():
        calls.append("ensure-resources")
        return SimpleNamespace(imported=3, skipped=12, created_indexes=1)

    monkeypatch.setattr(storage_runtime, "start_file_cache", start_cache)
    monkeypatch.setattr(storage_runtime, "ensure_builtin_resources", ensure_resources)

    asyncio.run(storage_runtime.start_storage_runtime(SimpleNamespace(info=Mock())))

    assert calls == ["start-cache", "ensure-resources"]


def test_builtin_resource_failure_does_not_hide_successful_cache_start(
        monkeypatch,
) -> None:
    start_cache = AsyncMock()
    ensure_resources = AsyncMock(side_effect=RuntimeError("broken resource"))
    logger = SimpleNamespace(info=Mock(), error=Mock(), debug=Mock())
    monkeypatch.setattr(storage_runtime, "start_file_cache", start_cache)
    monkeypatch.setattr(storage_runtime, "ensure_builtin_resources", ensure_resources)

    asyncio.run(storage_runtime.start_storage_runtime(logger))

    start_cache.assert_awaited_once_with()
    ensure_resources.assert_awaited_once_with()
    assert "broken resource" in logger.error.call_args.args[0]


def test_storage_runtime_close_delegates_to_file_cache(monkeypatch) -> None:
    close_cache = AsyncMock()
    monkeypatch.setattr(storage_runtime, "close_file_cache", close_cache)

    asyncio.run(storage_runtime.close_storage_runtime())

    close_cache.assert_awaited_once_with()
