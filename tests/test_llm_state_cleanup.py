import sqlite3

import pytest

from TIYA.llm.state import StateStore


def test_cleanup_removes_expired_state_and_leases_but_preserves_live_rows(tmp_path, monkeypatch):
    from TIYA.llm import state as module
    monkeypatch.setattr(module.time, "time", lambda: 1000)
    store = StateStore(tmp_path / "state.db", retention_seconds=100)
    store.put("checkpoints", "expired", [], expires=999)
    store.put("checkpoints", "live", [], expires=2000)
    store.claim("old", "owner", seconds=1)
    store.claim("live", "owner", seconds=500)
    store.cleanup(now=1002)
    with store.db.transaction() as db:
        assert [row["key"] for row in db.execute("SELECT key FROM state")] == ["live"]
        assert [row["key"] for row in db.execute("SELECT key FROM leases")] == ["live"]


def test_active_metadata_renews_idle_retention_without_extending_remote_expiry(tmp_path, monkeypatch):
    from TIYA.llm import state as module
    clock = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    store = StateStore(tmp_path / "state.db", retention_seconds=100)
    store.put("uploads", "active", {"file_id": "f1"})
    store.put("uploads", "idle", {"file_id": "f2"})
    store.put("uploads", "remote-expiry", {"file_id": "f3"}, expires=1100)
    clock[0] = 1075
    assert store.get("uploads", "active") == {"file_id": "f1"}
    assert store.get("uploads", "remote-expiry") == {"file_id": "f3"}
    store.cleanup(now=1101)
    assert store.get("uploads", "active", now=1101)
    assert store.get("uploads", "idle", now=1101) is None
    assert store.get("uploads", "remote-expiry", now=1101) is None


def test_cleanup_is_bounded_and_regular_access_runs_maintenance(tmp_path, monkeypatch):
    from TIYA.llm import state as module
    clock = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    store = StateStore(tmp_path / "state.db", retention_seconds=100)
    for index in range(6):
        store.put("media", str(index), {}, expires=1050)

    assert store.cleanup(now=1051, batch_size=2) == 2
    with store.db.transaction() as db:
        assert db.execute("SELECT count(*) FROM state").fetchone()[0] == 4

    clock[0] = 5000
    store.put("media", "new", {})
    with store.db.transaction() as db:
        assert db.execute("SELECT count(*) FROM state").fetchone()[0] == 1


def test_instances_share_one_hourly_cleanup_schedule(tmp_path, monkeypatch):
    from TIYA.llm import state as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    calls = []
    real_cleanup = StateStore._cleanup

    def cleanup(self, db, now, batch_size):
        calls.append(self.db.path)
        return real_cleanup(self, db, now, batch_size)

    monkeypatch.setattr(StateStore, "_cleanup", cleanup)
    first = StateStore(tmp_path / "state.db")
    second = StateStore(tmp_path / "alias" / ".." / "state.db")
    first.put("test", "first", {})
    second.put("test", "second", {})
    clock[0] += 3599
    second.put("test", "third", {})
    assert len(calls) == 1
    clock[0] += 1
    second.put("test", "fourth", {})
    first.get("test", "first")
    assert len(calls) == 2


def test_concurrent_instances_do_not_repeat_cleanup(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from TIYA.llm import state as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)
    calls = []
    real_cleanup = StateStore._cleanup

    def cleanup(self, db, now, batch_size):
        calls.append(self)
        return real_cleanup(self, db, now, batch_size)

    monkeypatch.setattr(StateStore, "_cleanup", cleanup)
    stores = [StateStore(tmp_path / "state.db") for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda store: store.get("test", "missing"), stores))

    assert len(calls) == 1


def test_cleanup_cadence_can_be_daily_and_isolated_per_database(tmp_path, monkeypatch):
    from TIYA.llm import state as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    calls = []
    real_cleanup = StateStore._cleanup

    def cleanup(self, db, now, batch_size):
        calls.append(self.db.path)
        return real_cleanup(self, db, now, batch_size)

    monkeypatch.setattr(StateStore, "_cleanup", cleanup)
    first = StateStore(tmp_path / "first.db", cleanup_interval_seconds=86400)
    other = StateStore(tmp_path / "other.db", cleanup_interval_seconds=86400)
    first.get("test", "missing")
    other.get("test", "missing")
    assert len(calls) == 2
    clock[0] += 86399
    first.get("test", "missing")
    assert len(calls) == 2
    clock[0] += 1
    first.get("test", "missing")
    assert len(calls) == 3


def test_manual_cleanup_updates_shared_schedule_without_double_sweep(tmp_path, monkeypatch):
    from TIYA.llm import state as module

    clock = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    calls = []
    real_cleanup = StateStore._cleanup

    def cleanup(self, db, now, batch_size):
        calls.append(batch_size)
        return real_cleanup(self, db, now, batch_size)

    monkeypatch.setattr(StateStore, "_cleanup", cleanup)
    first = StateStore(tmp_path / "state.db")
    second = StateStore(tmp_path / "state.db")
    assert first.cleanup(batch_size=3) == 0
    second.get("test", "missing")
    assert calls == [3]
    clock[0] += 3600
    second.get("test", "missing")
    assert calls == [3, 500]


def test_full_batch_does_not_trigger_cleanup_on_every_following_access(tmp_path, monkeypatch):
    from TIYA.llm import state as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 1000.0)
    calls = []

    def full_batch(self, db, now, batch_size):
        calls.append(batch_size)
        return batch_size

    monkeypatch.setattr(StateStore, "_cleanup", full_batch)
    first = StateStore(tmp_path / "state.db")
    first.get("test", "missing")
    StateStore(tmp_path / "state.db").get("test", "missing")
    first.get("test", "missing")
    assert calls == [500]


def test_existing_state_database_gets_retention_without_losing_checkpoints(tmp_path, monkeypatch):
    from TIYA.llm import state as module
    monkeypatch.setattr(module.time, "time", lambda: 1000)
    path = tmp_path / "state.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE state(namespace TEXT,key TEXT,value TEXT NOT NULL,expires REAL,PRIMARY KEY(namespace,key))")
        db.execute("INSERT INTO state VALUES('checkpoints','session','[{\"response_id\":\"r1\"}]',NULL)")

    store = StateStore(path, retention_seconds=100)
    assert store.get("checkpoints", "session", now=1050) == [{"response_id": "r1"}]
    assert StateStore(path, retention_seconds=100).get("checkpoints", "session", now=1101)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_retention_is_rejected(tmp_path, seconds):
    with pytest.raises(ValueError):
        StateStore(tmp_path / "state.db", retention_seconds=seconds)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_cleanup_interval_is_rejected(tmp_path, seconds):
    with pytest.raises(ValueError):
        StateStore(tmp_path / "state.db", cleanup_interval_seconds=seconds)


@pytest.mark.parametrize("dialect", ["openai", "deepseek", "anthropic"])
def test_files_upload_defaults_to_remote_ttl_and_can_disable_it(dialect):
    from TIYA.llm.provider.files import FilesProvider
    settings = {"endpoint": "https://test/files", "dialect": dialect}
    ttl_field = "expires_in_seconds" if dialect == "anthropic" else "expires_after[seconds]"
    upload = FilesProvider(settings).upload_request(b"image", "image.png", "image/png")
    assert upload.data[ttl_field] == "86400"
    custom = FilesProvider({**settings, "expires_after_seconds": 7200})
    assert custom.upload_request(b"image", "image.png", "image/png").data[ttl_field] == "7200"
    disabled = FilesProvider({**settings, "expires_after_seconds": None})
    assert ttl_field not in disabled.upload_request(b"image", "image.png", "image/png").data
