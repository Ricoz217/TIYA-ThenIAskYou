from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import TIYA.auto_mapping as auto_mapping_module
from TIYA.auto_mapping import (
    AutoMapping,
    AutoMappingClosedError,
    AutoMappingMigrationError,
    AutoMappingPersistenceError,
)


def _legacy_payload(
    entries: dict[str, tuple[float, object]],
    *,
    updated_at: float | None = None,
) -> dict[str, object]:
    return {
        "update": time.time() if updated_at is None else updated_at,
        "description": "此文件为映射表持久化数据",
        "version": "0.1.0",
        "data": {
            key: {"update": entry_updated_at, "data": value}
            for key, (entry_updated_at, value) in entries.items()
        },
    }


def _read_entries(database_path: Path) -> dict[str, tuple[object, float]]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT key, value_json, updated_at FROM entries ORDER BY key"
        ).fetchall()
    return {
        key: (json.loads(value_json), updated_at)
        for key, value_json, updated_at in rows
    }


def test_new_mapping_persists_to_derived_sqlite_path(tmp_path: Path) -> None:
    legacy_path = tmp_path / "mapping.json"
    mapping = AutoMapping[str](legacy_path, persist_period="UPDATE")

    mapping["name"] = "value"
    mapping.flush_blocking()

    assert not legacy_path.exists()
    assert mapping._save_path == legacy_path
    assert mapping.database_path == tmp_path / "mapping.sqlite3"
    assert _read_entries(mapping.database_path)["name"][0] == "value"
    assert AutoMapping[str](legacy_path)["name"] == "value"


def test_legacy_json_migrates_atomically_and_keeps_backup(tmp_path: Path) -> None:
    now = time.time()
    legacy_path = tmp_path / "mapping.json"
    legacy_path.write_text(
        json.dumps(
            _legacy_payload({"title": (now, ["hash-1", "hash-2"])}),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    mapping = AutoMapping[set[str]](
        legacy_path,
        value_type=set,
        expire_day=30,
    )

    assert mapping["title"] == {"hash-1", "hash-2"}
    assert mapping.database_path.exists()
    assert not legacy_path.exists()
    backups = list(tmp_path.glob("mapping.json.legacy-*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8"))["version"] == "0.1.0"
    with sqlite3.connect(mapping.database_path) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    assert metadata["schema_version"] == "1"
    assert metadata["legacy_source"] == str(legacy_path)


def test_migration_discards_expired_entries(tmp_path: Path) -> None:
    now = time.time()
    legacy_path = tmp_path / "mapping.json"
    legacy_path.write_text(
        json.dumps(
            _legacy_payload(
                {
                    "expired": (now - 3 * 86400, "old"),
                    "active": (now, "new"),
                }
            )
        ),
        encoding="utf-8",
    )

    mapping = AutoMapping[str](legacy_path, expire_day=1)

    assert mapping.mapping() == {"active": "new"}
    assert set(_read_entries(mapping.database_path)) == {"active"}


def test_corrupt_legacy_json_fails_without_replacing_source(tmp_path: Path) -> None:
    legacy_path = tmp_path / "mapping.json"
    legacy_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(AutoMappingMigrationError, match="JSON"):
        AutoMapping(legacy_path)

    assert legacy_path.read_text(encoding="utf-8") == "{broken"
    assert not legacy_path.with_suffix(".sqlite3").exists()


def test_existing_sqlite_takes_precedence_over_legacy_json(tmp_path: Path) -> None:
    legacy_path = tmp_path / "mapping.json"
    mapping = AutoMapping[str](legacy_path, persist_period="UPDATE")
    mapping["source"] = "sqlite"
    mapping.flush_blocking()
    legacy_path.write_text(
        json.dumps(_legacy_payload({"source": (time.time(), "json")})),
        encoding="utf-8",
    )

    loaded = AutoMapping[str](legacy_path)

    assert loaded["source"] == "sqlite"
    assert legacy_path.exists()


def test_getitem_renews_ttl_but_contains_live_can_avoid_touch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1_000_000.0]
    monkeypatch.setattr(auto_mapping_module.time, "time", lambda: clock[0])
    mapping = AutoMapping[str](expire_day=1)
    mapping["key"] = "value"
    original_updated_at = mapping.to_dict()["data"]["key"]["update"]

    clock[0] += 10
    assert mapping.contains_live("key", touch=False) is True
    assert mapping.to_dict()["data"]["key"]["update"] == original_updated_at

    clock[0] += 10
    assert mapping["key"] == "value"
    assert mapping.to_dict()["data"]["key"]["update"] == clock[0]


def test_contains_live_removes_expired_key_without_renewing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [2_000_000.0]
    monkeypatch.setattr(auto_mapping_module.time, "time", lambda: clock[0])
    mapping = AutoMapping[str](expire_day=1)
    mapping["key"] = "value"

    clock[0] += 86401

    assert "key" in mapping
    assert mapping.contains_live("key", touch=False) is False
    assert "key" not in mapping


def test_callable_expire_day_is_resolved_lazily() -> None:
    ready = False

    def resolve_expire_day() -> int:
        if not ready:
            raise AssertionError("TTL callback was evaluated during construction")
        return 30

    mapping = AutoMapping[str](expire_day=resolve_expire_day)
    ready = True
    mapping["key"] = "value"

    assert mapping.contains_live("key", touch=False) is True


def test_bulk_updates_and_delete_upsert_order_persist_final_state(tmp_path: Path) -> None:
    mapping = AutoMapping[str](tmp_path / "mapping.json", persist_period="MINUTE")

    mapping.update_from_dict({"first": "one", "second": "two"})
    mapping.pop("first")
    mapping["first"] = "final"
    mapping["second"] = "changed"
    mapping.remove("second")
    mapping.flush_blocking()

    assert AutoMapping[str](tmp_path / "mapping.json").mapping() == {"first": "final"}


def test_flush_async_works_and_blocking_flush_rejects_event_loop_thread(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        mapping = AutoMapping[str](tmp_path / "mapping.json", persist_period="MINUTE")
        mapping["key"] = "value"

        with pytest.raises(RuntimeError, match="event loop"):
            mapping.flush_blocking()

        await mapping.flush_async()

    asyncio.run(run())
    assert AutoMapping[str](tmp_path / "mapping.json")["key"] == "value"


def test_update_does_not_wait_for_sqlite_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mapping = AutoMapping[str](tmp_path / "mapping.json", persist_period="UPDATE")
    write_started = threading.Event()
    release_write = threading.Event()
    original = mapping._write_batch

    def delayed_write(*args, **kwargs):
        write_started.set()
        assert release_write.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(mapping, "_write_batch", delayed_write)

    started_at = time.perf_counter()
    mapping["key"] = "value"
    elapsed = time.perf_counter() - started_at

    assert elapsed < 0.1
    assert write_started.wait(timeout=1)
    release_write.set()
    mapping.flush_blocking()


def test_async_open_loads_mapping_and_close_rejects_new_writes(tmp_path: Path) -> None:
    legacy_path = tmp_path / "mapping.json"
    initial = AutoMapping[str](legacy_path, persist_period="UPDATE")
    initial["key"] = "value"
    initial.flush_blocking()

    async def run() -> None:
        loaded = await AutoMapping.open(legacy_path)
        assert loaded["key"] == "value"
        await loaded.close_async()
        with pytest.raises(AutoMappingClosedError):
            loaded["other"] = "new"

    asyncio.run(run())


def test_close_flushes_pending_changes(tmp_path: Path) -> None:
    legacy_path = tmp_path / "mapping.json"

    async def run() -> None:
        mapping = AutoMapping[str](legacy_path, persist_period="HOUR")
        mapping["key"] = "value"
        await mapping.close_async()

    asyncio.run(run())
    assert AutoMapping[str](legacy_path)["key"] == "value"


def test_failed_transaction_keeps_dirty_data_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mapping = AutoMapping[str](tmp_path / "mapping.json", persist_period="MINUTE")
    mapping["key"] = "value"
    original = mapping._write_batch
    failures = [OSError("disk unavailable")]

    def fail_once(*args, **kwargs):
        if failures:
            raise failures.pop()
        return original(*args, **kwargs)

    monkeypatch.setattr(mapping, "_write_batch", fail_once)

    with pytest.raises(AutoMappingPersistenceError, match="disk unavailable"):
        mapping.flush_blocking()

    mapping.flush_blocking()
    assert AutoMapping[str](tmp_path / "mapping.json")["key"] == "value"


def test_no_storage_path_keeps_full_in_memory_api() -> None:
    mapping = AutoMapping[int](default={"one": 1})

    mapping.update("two", 2)
    assert mapping.get("missing", 3) == 3
    assert list(mapping.keys()) == ["one", "two"]
    assert list(mapping.values()) == [1, 2]
    assert list(mapping.items()) == [("one", 1), ("two", 2)]
    assert mapping.copy() == {"one": 1, "two": 2}
    assert len(mapping) == 2
    mapping.persist()
    mapping.flush_blocking()


def test_load_dict_is_a_compatible_import_and_persists(tmp_path: Path) -> None:
    mapping = AutoMapping[str](tmp_path / "mapping.json", persist_period="MINUTE")
    payload = _legacy_payload({"imported": (time.time(), "value")})

    mapping.load_dict(payload)
    mapping.flush_blocking()

    assert AutoMapping[str](tmp_path / "mapping.json")["imported"] == "value"
