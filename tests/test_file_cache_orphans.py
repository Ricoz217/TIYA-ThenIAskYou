from __future__ import annotations

import hashlib

import TIYA.file_cache as file_cache_module
from TIYA.file_cache import FileCache, FileRetention


def _write_orphan(cache_path, payload: bytes) -> tuple[str, object]:
    hash_name = hashlib.blake2b(payload, digest_size=16).hexdigest()
    target = cache_path / hash_name[:2] / hash_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return hash_name, target


def test_cleanup_rebuilds_expiring_metadata_for_orphan_hash_file(
        monkeypatch,
        tmp_path,
) -> None:
    now = 1_000_000
    monkeypatch.setattr(file_cache_module.time, "time", lambda: now)
    cache = FileCache(tmp_path, expire_days=30)
    hash_name, target = _write_orphan(tmp_path, b"orphan-cache-file")

    result = cache.cleanup_once()
    metadata = cache.get_file_metadata(hash_name)

    assert target.is_file()
    assert result.deleted == 0
    assert metadata is not None
    assert metadata.create_time == now
    assert metadata.use_time == now
    assert metadata.size == target.stat().st_size
    assert metadata.retention is FileRetention.EXPIRING


def test_rebuilt_orphan_expires_after_full_retention_period(
        monkeypatch,
        tmp_path,
) -> None:
    clock = [1_000_000]
    monkeypatch.setattr(file_cache_module.time, "time", lambda: clock[0])
    cache = FileCache(tmp_path, expire_days=30)
    hash_name, target = _write_orphan(tmp_path, b"eventually-expired-orphan")

    cache.cleanup_once()
    clock[0] += 31 * 24 * 3600
    result = cache.cleanup_once()

    assert result.deleted == 1
    assert not target.exists()
    assert cache.get_file_metadata(hash_name) is None


def test_cleanup_ignores_files_outside_the_hash_layout(tmp_path) -> None:
    cache = FileCache(tmp_path, expire_days=30)
    hash_name, _ = _write_orphan(tmp_path, b"valid-orphan")
    wrong_bucket = tmp_path / "ff" / hash_name
    malformed = tmp_path / "aa" / "not-a-hash"
    wrong_bucket.parent.mkdir(parents=True, exist_ok=True)
    malformed.parent.mkdir(parents=True, exist_ok=True)
    wrong_bucket.write_bytes(b"wrong-bucket")
    malformed.write_bytes(b"malformed")

    cache.cleanup_once()

    assert cache.get_file_metadata(hash_name) is not None
    assert wrong_bucket.is_file()
    assert malformed.is_file()


def test_rebuild_does_not_restore_index_for_file_removed_before_flush(
        monkeypatch,
        tmp_path,
) -> None:
    cache = FileCache(tmp_path, expire_days=30)
    cache.initialize()
    hash_name, target = _write_orphan(tmp_path, b"removed-during-rebuild")

    class RemoveBeforeFlush:
        def __init__(self, lock) -> None:
            self._lock = lock
            self._entries = 0

        def __enter__(self):
            self._entries += 1
            if self._entries == 2:
                cache.remove_file(hash_name)
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self._lock.release()

    monkeypatch.setattr(cache, "_lock", RemoveBeforeFlush(cache._lock))

    recovered = cache._rebuild_orphan_index(1_000_000)

    assert recovered == 0
    assert not target.exists()
    assert cache.get_file_metadata(hash_name) is None
