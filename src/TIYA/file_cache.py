from __future__ import annotations
"""
file_cache.py
SQLite-backed file cache.

Importing this module does not create files, open SQLite, or start background
work. The public module functions use a lazily created default
:class:`FileCache` instance.
文件缓存系统，重构为 SQLite 方案
"""
__version__ = "0.2.0"
import asyncio
import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any

from TIYA.config import FILE_CACHE_DIR, SETTING_CFG
from TIYA.executor import FILE_CACHE_EXECUTOR
from TIYA.logger import get_logger


_CACHE_PATH = FILE_CACHE_DIR
_DATABASE_NAME = "metadata.sqlite3"
_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
_SCHEMA_VERSION = 1
_DEFAULT_CLEANUP_INTERVAL = 24 * 3600
_DEFAULT_CLEANUP_BATCH_SIZE = 200
_log = get_logger()


class FileRetention(StrEnum):
    """Automatic eviction policy for a cached file."""

    EXPIRING = "expiring"
    PERMANENT = "permanent"


@dataclass(slots=True)
class FileInfo:
    create_time: int = field(default_factory=lambda: int(time.time()))
    use_time: int = field(default_factory=lambda: int(time.time()))
    file_type: str = "file"
    note: str = ""
    size: int = 0
    retention: FileRetention = FileRetention.EXPIRING

    def to_dict(self) -> dict[str, int | str]:
        return {
            "create_time": self.create_time,
            "use_time": self.use_time,
            "file_type": self.file_type,
            "note": self.note,
            "size": self.size,
            "retention": self.retention.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileInfo:
        return cls(
            create_time=int(float(data.get("create_time", time.time()))),
            use_time=int(float(data.get("use_time", time.time()))),
            file_type=str(data.get("file_type", "file")),
            note=str(data.get("note", "")),
            size=int(data.get("size", 0)),
            retention=FileRetention(
                data.get("retention", FileRetention.EXPIRING.value)
            ),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> FileInfo:
        return cls(
            create_time=int(row["create_time"]),
            use_time=int(row["use_time"]),
            file_type=str(row["file_type"]),
            note=str(row["note"]),
            size=int(row["size"]),
            retention=FileRetention(row["retention"]),
        )


@dataclass(frozen=True, slots=True)
class CleanupResult:
    scanned: int
    deleted: int
    missing: int
    failed: int
    released_bytes: int
    started_at: int
    finished_at: int
    aborted: bool = False


ExpireDays = float | int | Callable[[], float | int]


class FileCache:
    """Content-addressed file storage with SQLite metadata."""

    def __init__(
            self,
            cache_path: str | Path,
            *,
            expire_days: ExpireDays | None = None,
            cleanup_interval: float = _DEFAULT_CLEANUP_INTERVAL,
            cleanup_batch_size: int = _DEFAULT_CLEANUP_BATCH_SIZE,
            executor: Executor | None = None,
            logger: Any | None = None,
            database_name: str = _DATABASE_NAME,
    ) -> None:
        if cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be greater than zero")

        if cleanup_batch_size <= 0:
            raise ValueError("cleanup_batch_size must be greater than zero")

        self.cache_path = Path(cache_path)
        self.database_path = self.cache_path / database_name
        self._expire_days = expire_days
        self._cleanup_interval = float(cleanup_interval)
        self._cleanup_batch_size = int(cleanup_batch_size)
        self._executor = executor or FILE_CACHE_EXECUTOR
        self._log = logger or _log

        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._initialized = False
        self._cleanup_task: asyncio.Task[None] | None = None
        self._cleanup_stop: asyncio.Event | None = None
        self._cleanup_abort = threading.Event()
        self._lifecycle_loop: asyncio.AbstractEventLoop | None = None

    @staticmethod
    def _normalize_hash(hash_name: str) -> str:
        if not isinstance(hash_name, str) or not _HASH_PATTERN.fullmatch(hash_name):
            raise ValueError(f"invalid file hash: {hash_name!r}")
        return hash_name.lower()

    def _target_path(self, hash_name: str) -> Path:
        return self.cache_path / hash_name[:2] / hash_name

    def check_file_exists(self, hash_name: str) -> bool:
        """Check whether the physical cache file exists without touching metadata."""
        try:
            hash_name = self._normalize_hash(hash_name)
            return self._target_path(hash_name).is_file()

        except (ValueError, OSError):
            return False

    def _current_expire_days(self) -> float:
        value: float | int
        if self._expire_days is None:
            value = SETTING_CFG.Common.FileCacheExpire

        elif callable(self._expire_days):
            value = self._expire_days()

        else:
            value = self._expire_days

        days = float(value)
        if days < 0:
            raise ValueError("expire_days must not be negative")

        return days

    def _connect_locked(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is not None:
            return connection

        self.cache_path.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.database_path,
            timeout=5,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        try:
            journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                raise RuntimeError(
                    f"failed to enable SQLite WAL mode: {journal_mode!r}"
                )

            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    hash_name TEXT PRIMARY KEY,
                    create_time INTEGER NOT NULL,
                    use_time INTEGER NOT NULL,
                    file_type TEXT NOT NULL,
                    note TEXT NOT NULL,
                    size INTEGER NOT NULL CHECK (size >= 0),
                    retention TEXT NOT NULL
                        CHECK (retention IN ('expiring', 'permanent'))
                );

                CREATE INDEX IF NOT EXISTS idx_files_cleanup
                    ON files (retention, use_time);

                CREATE TABLE IF NOT EXISTS cache_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            connection.execute(
                """
                INSERT INTO cache_state (key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(_SCHEMA_VERSION),),
            )
            connection.commit()

        except BaseException:
            connection.close()
            raise

        self._connection = connection
        return connection

    @staticmethod
    def _set_state_locked(
            connection: sqlite3.Connection,
            key: str,
            value: dict[str, Any] | str,
    ) -> None:
        serialized = value if isinstance(value, str) else json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        connection.execute(
            """
            INSERT INTO cache_state (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, serialized),
        )

    def initialize(self) -> None:
        """Open SQLite and create the current schema."""
        with self._lock:
            if self._initialized:
                return

            self._connect_locked()
            self._initialized = True

    def _rebuild_orphan_index(self, now: int) -> int:
        """Register valid hash files missing from SQLite as expiring entries."""
        with self._lock:
            connection = self._connect_locked()
            indexed = {
                str(row["hash_name"])
                for row in connection.execute("SELECT hash_name FROM files")
            }

        recovered = 0
        pending: list[tuple[str, int, int, str, str, int, str]] = []

        def flush() -> None:
            nonlocal recovered
            if not pending:
                return

            with self._lock:
                connection = self._connect_locked()
                verified = []
                for row in pending:
                    hash_name = row[0]
                    target = self._target_path(hash_name)
                    try:
                        if target.is_symlink() or not target.is_file():
                            continue
                        size = target.stat().st_size

                    except OSError:
                        continue

                    verified.append((*row[:5], int(size), row[6]))

                before = connection.total_changes
                with connection:
                    connection.executemany(
                        """
                        INSERT OR IGNORE INTO files (
                            hash_name, create_time, use_time,
                            file_type, note, size, retention
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        verified,
                    )
                recovered += connection.total_changes - before

            pending.clear()

        try:
            buckets = list(self.cache_path.iterdir())
        except OSError as exc:
            self._log.error(f"扫描文件缓存目录失败: {exc}")
            return 0

        for bucket in buckets:
            bucket_name = bucket.name
            if (
                    len(bucket_name) != 2
                    or bucket_name != bucket_name.lower()
                    or not all(char in "0123456789abcdef" for char in bucket_name)
                    or bucket.is_symlink()
                    or not bucket.is_dir()
            ):
                continue

            try:
                for target in bucket.iterdir():
                    hash_name = target.name
                    if (
                            hash_name in indexed
                            or hash_name != hash_name.lower()
                            or not _HASH_PATTERN.fullmatch(hash_name)
                            or hash_name[:2] != bucket_name
                            or target.is_symlink()
                            or not target.is_file()
                    ):
                        continue

                    try:
                        size = target.stat().st_size
                    except OSError as exc:
                        self._log.warning(f"读取孤儿缓存文件失败 [{target}]: {exc}")
                        continue

                    indexed.add(hash_name)
                    pending.append((
                        hash_name,
                        now,
                        now,
                        "file",
                        "recovered orphan cache file",
                        int(size),
                        FileRetention.EXPIRING.value,
                    ))
                    if len(pending) >= self._cleanup_batch_size:
                        flush()

            except OSError as exc:
                self._log.warning(f"扫描文件缓存分片失败 [{bucket}]: {exc}")

        flush()
        return recovered

    @staticmethod
    def _read_input(file: str | Path | bytes | BytesIO) -> bytes:
        if isinstance(file, str):
            data = base64.b64decode(file)

        elif isinstance(file, Path):
            data = file.read_bytes()

        elif isinstance(file, bytes):
            data = file

        elif hasattr(file, "read") and hasattr(file, "seek"):
            stream = file
            current_pos = stream.tell()
            try:
                stream.seek(0)
                data = stream.read()

            finally:
                stream.seek(current_pos)

        else:
            raise TypeError(f"unsupported file input: {type(file)!r}")

        if not isinstance(data, bytes):
            raise TypeError("file input could not be read as bytes")

        return data

    def _upsert_file_locked(
            self,
            connection: sqlite3.Connection,
            *,
            hash_name: str,
            now: int,
            file_type: str,
            note: str,
            size: int,
            retention: FileRetention,
    ) -> None:
        with connection:
            connection.execute(
                """
                INSERT INTO files (
                    hash_name, create_time, use_time,
                    file_type, note, size, retention
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hash_name) DO UPDATE SET
                    use_time=excluded.use_time,
                    file_type=excluded.file_type,
                    note=excluded.note,
                    size=excluded.size,
                    retention=excluded.retention
                """,
                (
                    hash_name,
                    now,
                    now,
                    file_type,
                    note,
                    size,
                    retention.value,
                ),
            )

    def add_file(
            self,
            file: str | Path | bytes | BytesIO,
            file_type: str = "file",
            note: str = "",
            *,
            retention: FileRetention | str = FileRetention.EXPIRING,
    ) -> str:
        data = self._read_input(file)
        hash_name = hashlib.blake2b(data, digest_size=16).hexdigest()
        retention = FileRetention(retention)
        self.initialize()

        target = self._target_path(hash_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        now = int(time.time())
        size = len(data)

        with self._lock:
            connection = self._connect_locked()
            try:
                target_size = target.stat().st_size

            except FileNotFoundError:
                target_size = None

            if target_size == size and target.is_file():
                self._upsert_file_locked(
                    connection,
                    hash_name=hash_name,
                    now=now,
                    file_type=str(file_type),
                    note=str(note),
                    size=size,
                    retention=retention,
                )
                return hash_name

        temp_path = target.with_name(
            f".{target.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temp_path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())

            with self._lock:
                connection = self._connect_locked()
                try:
                    target_size = target.stat().st_size

                except FileNotFoundError:
                    target_size = None

                if target_size != size or not target.is_file():
                    os.replace(temp_path, target)

                self._upsert_file_locked(
                    connection,
                    hash_name=hash_name,
                    now=now,
                    file_type=str(file_type),
                    note=str(note),
                    size=size,
                    retention=retention,
                )

            return hash_name

        finally:
            try:
                temp_path.unlink(missing_ok=True)

            except OSError:
                self._log.warning(f"无法清理文件缓存临时文件: {temp_path}")

    def get_file_path(self, hash_name: str) -> Path:
        hash_name = self._normalize_hash(hash_name)
        self.initialize()
        target = self._target_path(hash_name)
        now = int(time.time())
        with self._lock:
            connection = self._connect_locked()
            try:
                size = target.stat().st_size

            except FileNotFoundError:
                with connection:
                    connection.execute(
                        "DELETE FROM files WHERE hash_name = ?",
                        (hash_name,),
                    )

                raise FileNotFoundError(target) from None

            if not target.is_file():
                with connection:
                    connection.execute(
                        "DELETE FROM files WHERE hash_name = ?",
                        (hash_name,),
                    )

                raise FileNotFoundError(target)

            with connection:
                cursor = connection.execute(
                    "UPDATE files SET use_time = ?, size = ? WHERE hash_name = ?",
                    (now, int(size), hash_name),
                )
                if cursor.rowcount == 0:
                    connection.execute(
                        """
                        INSERT INTO files (
                            hash_name, create_time, use_time,
                            file_type, note, size, retention
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            hash_name,
                            now,
                            now,
                            "file",
                            "",
                            int(size),
                            FileRetention.EXPIRING.value,
                        ),
                    )

            return target

    def get_file_metadata(self, hash_name: str) -> FileInfo | None:
        hash_name = self._normalize_hash(hash_name)
        self.initialize()
        with self._lock:
            connection = self._connect_locked()
            row = connection.execute(
                """
                SELECT create_time, use_time, file_type, note, size, retention
                FROM files
                WHERE hash_name = ?
                """,
                (hash_name,),
            ).fetchone()

        return None if row is None else FileInfo.from_row(row)

    def renew(self, hash_name: str) -> None:
        hash_name = self._normalize_hash(hash_name)
        self.initialize()
        with self._lock:
            connection = self._connect_locked()
            with connection:
                connection.execute(
                    "UPDATE files SET use_time = ? WHERE hash_name = ?",
                    (int(time.time()), hash_name),
                )

    def remove_file(self, file: str | Path) -> None:
        raw_name = file.name if isinstance(file, Path) else file
        hash_name = self._normalize_hash(raw_name)
        self.initialize()
        target = self._target_path(hash_name)

        with self._lock:
            connection = self._connect_locked()
            try:
                target.unlink()

            except FileNotFoundError:
                pass

            with connection:
                connection.execute(
                    "DELETE FROM files WHERE hash_name = ?",
                    (hash_name,),
                )

    async def add_file_async(
            self,
            file: str | Path | bytes | BytesIO,
            file_type: str = "file",
            note: str = "",
            *,
            retention: FileRetention | str = FileRetention.EXPIRING,
    ) -> str:
        """Run :meth:`add_file` in the file-cache executor."""
        loop = asyncio.get_running_loop()
        call = partial(
            self.add_file,
            file,
            file_type,
            note,
            retention=retention,
        )
        # noinspection PyTypeChecker
        return await loop.run_in_executor(self._executor, call)

    async def get_file_path_async(self, hash_name: str) -> Path:
        """Run :meth:`get_file_path` in the file-cache executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_file_path,
            hash_name,
        )

    async def get_file_metadata_async(self, hash_name: str) -> FileInfo | None:
        """Run :meth:`get_file_metadata` in the file-cache executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_file_metadata,
            hash_name,
        )

    async def renew_async(self, hash_name: str) -> None:
        """Run :meth:`renew` in the file-cache executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.renew, hash_name)

    async def remove_file_async(self, file: str | Path) -> None:
        """Run :meth:`remove_file` in the file-cache executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.remove_file, file)

    def cleanup_once(
            self,
            stop_event: threading.Event | None = None,
    ) -> CleanupResult:
        """Delete one snapshot of expired files in bounded batches."""
        self.initialize()
        started_at = int(time.time())
        recovered = self._rebuild_orphan_index(started_at)
        cutoff = started_at - int(self._current_expire_days() * 24 * 3600)
        scanned = 0
        deleted = 0
        missing = 0
        failed = 0
        released_bytes = 0
        last_rowid = 0
        aborted = False

        while True:
            with self._lock:
                connection = self._connect_locked()
                rows = connection.execute(
                    """
                    SELECT rowid, hash_name
                    FROM files
                    WHERE rowid > ?
                      AND retention = ?
                      AND use_time < ?
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    (
                        last_rowid,
                        FileRetention.EXPIRING.value,
                        cutoff,
                        self._cleanup_batch_size,
                    ),
                ).fetchall()

            if not rows:
                break

            for candidate in rows:
                last_rowid = int(candidate["rowid"])
                hash_name = str(candidate["hash_name"])
                scanned += 1

                with self._lock:
                    connection = self._connect_locked()
                    current = connection.execute(
                        """
                        SELECT use_time, retention, size
                        FROM files
                        WHERE hash_name = ?
                        """,
                        (hash_name,),
                    ).fetchone()
                    if current is None:
                        continue

                    if (
                            current["retention"]
                            != FileRetention.EXPIRING.value
                            or int(current["use_time"]) >= cutoff
                    ):
                        continue

                    target = self._target_path(hash_name)
                    existed = True
                    try:
                        target.unlink()

                    except FileNotFoundError:
                        existed = False
                        missing += 1

                    except OSError as exc:
                        failed += 1
                        self._log.error(
                            f"清理文件缓存失败 [{hash_name}]: {exc}"
                        )
                        continue

                    with connection:
                        connection.execute(
                            "DELETE FROM files WHERE hash_name = ?",
                            (hash_name,),
                        )

                    deleted += 1
                    if existed:
                        released_bytes += int(current["size"])

            if stop_event is not None and stop_event.is_set():
                aborted = True
                break

        finished_at = int(time.time())
        result = CleanupResult(
            scanned=scanned,
            deleted=deleted,
            missing=missing,
            failed=failed,
            released_bytes=released_bytes,
            started_at=started_at,
            finished_at=finished_at,
            aborted=aborted,
        )
        with self._lock:
            connection = self._connect_locked()
            with connection:
                self._set_state_locked(
                    connection,
                    "last_cleanup",
                    {
                        "recovered": recovered,
                        "scanned": result.scanned,
                        "deleted": result.deleted,
                        "missing": result.missing,
                        "failed": result.failed,
                        "released_bytes": result.released_bytes,
                        "started_at": result.started_at,
                        "finished_at": result.finished_at,
                        "aborted": result.aborted,
                    },
                )

        self._log.info(
            "文件缓存清理完成: "
            f"补建[{recovered}]，扫描[{scanned}]，删除[{deleted}]，缺失[{missing}]，"
            f"失败[{failed}]，释放[{released_bytes}] bytes，"
            f"中止[{aborted}]"
        )
        return result

    async def _cleanup_loop(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            try:
                await loop.run_in_executor(
                    self._executor,
                    self.cleanup_once,
                    self._cleanup_abort,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log.error(f"文件缓存自动清理异常: {exc}")

            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=self._cleanup_interval,
                )
            except TimeoutError:
                continue

    async def start(self) -> None:
        """Initialize storage and start exactly one periodic cleanup task."""
        loop = asyncio.get_running_loop()
        with self._lock:
            task = self._cleanup_task
            if task is not None and not task.done():
                if self._lifecycle_loop is not loop:
                    raise RuntimeError("file cache is running on another event loop")
                return

        # noinspection PyTypeChecker
        await loop.run_in_executor(self._executor, self.initialize)
        with self._lock:
            task = self._cleanup_task
            if task is not None and not task.done():
                return
            stop = asyncio.Event()
            self._cleanup_abort.clear()
            self._cleanup_stop = stop
            self._lifecycle_loop = loop
            self._cleanup_task = loop.create_task(
                self._cleanup_loop(stop),
                name="file-cache-cleanup",
            )

    async def close(self) -> None:
        """Stop cleanup, wait for in-flight work, and close SQLite."""
        with self._lock:
            task = self._cleanup_task
            stop = self._cleanup_stop
            lifecycle_loop = self._lifecycle_loop
        if task is not None and not task.done():
            if lifecycle_loop is not asyncio.get_running_loop():
                raise RuntimeError("file cache must close on its lifecycle event loop")
            if stop is not None:
                stop.set()
            self._cleanup_abort.set()
            await task

        with self._lock:
            self._cleanup_task = None
            self._cleanup_stop = None
            self._lifecycle_loop = None
            self._initialized = False
            connection = self._connection
            self._connection = None
            if connection is not None:
                try:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error as exc:
                    self._log.warning(f"文件缓存WAL检查点失败: {exc}")
                finally:
                    connection.close()


_DEFAULT_CACHE: FileCache | None = None
_DEFAULT_CACHE_LOCK = threading.Lock()


def _get_default_cache() -> FileCache:
    global _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        if _DEFAULT_CACHE is None:
            _DEFAULT_CACHE = FileCache(_CACHE_PATH)

        return _DEFAULT_CACHE


def add_file(
        file: str | Path | bytes | BytesIO,
        file_type: str = "file",
        note: str = "",
        *,
        retention: FileRetention | str = FileRetention.EXPIRING,
) -> str:
    """
    将文件存入缓存，并返回哈希文件名
    :param file: 若传入是字符串，必须是base64字符串，而不是文件路径，文件路径只允许Path方式传入
    :param file_type: 文件类型
    :param note: 文件备注
    :param retention: 文件淘汰类型，默认允许按最后访问时间自动淘汰
    :return: 返回的是哈希名字字符串
    """
    return _get_default_cache().add_file(
        file,
        file_type,
        note,
        retention=retention,
    )


async def add_file_async(
        file: str | Path | bytes | BytesIO,
        file_type: str = "file",
        note: str = "",
        *,
        retention: FileRetention | str = FileRetention.EXPIRING,
) -> str:
    """
    在线程池中将文件存入缓存，并返回哈希文件名
    :param file: 若传入是字符串，必须是base64字符串，而不是文件路径，文件路径只允许Path方式传入
    :param file_type: 文件类型
    :param note: 文件备注
    :param retention: 文件淘汰类型，默认允许按最后访问时间自动淘汰
    :return: 返回的是哈希名字字符串
    """
    return await _get_default_cache().add_file_async(
        file,
        file_type,
        note,
        retention=retention,
    )


def get_file_path(hash_name: str) -> Path:
    """
    通过哈希名返回文件
    :param hash_name: 文件哈希名
    :return: 文件路径
    """
    return _get_default_cache().get_file_path(hash_name)


def check_file_exists(hash_name: str) -> bool:
    """
    检查缓存物理文件是否存在，不读取或更新SQLite索引
    :param hash_name: 文件哈希名
    :return: 物理文件是否存在
    """
    return _get_default_cache().check_file_exists(hash_name)


async def get_file_path_async(hash_name: str) -> Path:
    """
    在线程池中通过哈希名返回文件
    :param hash_name: 文件哈希名
    :return: 文件路径
    """
    return await _get_default_cache().get_file_path_async(hash_name)


def get_file_metadata(hash_name: str) -> FileInfo | None:
    """
    通过哈希名返回文件元数据
    :param hash_name: 文件哈希名
    :return: 文件元数据，不存在时返回None
    """
    return _get_default_cache().get_file_metadata(hash_name)


async def get_file_metadata_async(hash_name: str) -> FileInfo | None:
    """
    在线程池中通过哈希名返回文件元数据
    :param hash_name: 文件哈希名
    :return: 文件元数据，不存在时返回None
    """
    return await _get_default_cache().get_file_metadata_async(hash_name)


def renew(hash_name: str) -> None:
    """
    更新文件的最后访问时间，索引不存在时不创建
    :param hash_name: 文件哈希名
    """
    _get_default_cache().renew(hash_name)


async def renew_async(hash_name: str) -> None:
    """
    在线程池中更新文件的最后访问时间，索引不存在时不创建
    :param hash_name: 文件哈希名
    """
    await _get_default_cache().renew_async(hash_name)


def remove_file(file: str | Path) -> None:
    """
    删除缓存文件及其元数据
    :param file: 文件哈希名或缓存文件路径
    """
    _get_default_cache().remove_file(file)


async def remove_file_async(file: str | Path) -> None:
    """
    在线程池中删除缓存文件及其元数据
    :param file: 文件哈希名或缓存文件路径
    """
    await _get_default_cache().remove_file_async(file)


def remove_fire(file: str | Path) -> None:
    """
    删除缓存文件及其元数据，保留该旧名称用于兼容
    :param file: 文件哈希名或缓存文件路径
    """
    remove_file(file)


async def remove_fire_async(file: str | Path) -> None:
    """
    在线程池中删除缓存文件及其元数据，保留该旧名称用于兼容
    :param file: 文件哈希名或缓存文件路径
    """
    await remove_file_async(file)


def cleanup_file_cache() -> CleanupResult:
    """
    立即执行一次过期文件清理
    :return: 本次清理结果
    """
    return _get_default_cache().cleanup_once()


async def start_file_cache() -> None:
    """初始化文件缓存，并启动唯一的自动清理任务。"""
    await _get_default_cache().start()


async def close_file_cache() -> None:
    """停止自动清理任务，并关闭SQLite连接。"""
    with _DEFAULT_CACHE_LOCK:
        cache = _DEFAULT_CACHE
    if cache is not None:
        await cache.close()


__all__ = [
    "CleanupResult",
    "FileCache",
    "FileInfo",
    "FileRetention",
    "add_file",
    "add_file_async",
    "check_file_exists",
    "cleanup_file_cache",
    "close_file_cache",
    "get_file_metadata",
    "get_file_metadata_async",
    "get_file_path",
    "get_file_path_async",
    "remove_file",
    "remove_file_async",
    "remove_fire",
    "remove_fire_async",
    "renew",
    "renew_async",
    "start_file_cache",
]
