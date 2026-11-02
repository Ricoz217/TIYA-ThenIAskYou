"""Transactional communication metadata, separate from business history and accounting."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .config_model import JsonValue
from .storage import Database


class StateStore:
    def __init__(
        self,
        path: str | Path,
        *,
        retention_seconds: float = 30 * 86400,
    ) -> None:
        """保存可丢弃的通信元数据，默认回收闲置 30 天的状态。

        expires 表示远端或能力状态的绝对失效时间，读取不会延长它。
        retention_seconds 只限制本地元数据闲置时间，不控制文件缓存、远端文件或账本。
        普通访问每五分钟触发一批清理，不启动额外后台任务。
        """
        if not math.isfinite(retention_seconds) or retention_seconds <= 0:
            raise ValueError("retention_seconds 必须是有限正数")

        self.retention_seconds = retention_seconds
        self._schema_ready = False
        self._next_cleanup_at = 0.0
        self.db = Database(
            path,
            """
          CREATE TABLE IF NOT EXISTS state(namespace TEXT,key TEXT,value TEXT NOT NULL,expires REAL,
            touched REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(namespace,key));
          CREATE TABLE IF NOT EXISTS leases(key TEXT PRIMARY KEY,owner TEXT NOT NULL,expires REAL NOT NULL);
        """,
        )

    @contextmanager
    def _transaction(self, now: float | None = None) -> Iterator[sqlite3.Connection]:
        now = time.time() if now is None else now
        with self.db.transaction() as db:
            if not self._schema_ready:
                columns = {row["name"] for row in db.execute("PRAGMA table_info(state)")}
                if "touched" not in columns:
                    db.execute("ALTER TABLE state ADD COLUMN touched REAL NOT NULL DEFAULT 0")

                # 旧状态首次打开时从当前时间起计算保留期，不清空已有检查点。
                db.execute("UPDATE state SET touched=? WHERE touched=0", (now,))
                db.execute("CREATE INDEX IF NOT EXISTS state_expires ON state(expires)")
                db.execute("CREATE INDEX IF NOT EXISTS state_touched ON state(touched)")
                db.execute("CREATE INDEX IF NOT EXISTS leases_expires ON leases(expires)")

            if time.monotonic() >= self._next_cleanup_at:
                removed = self._cleanup(db, now, 500)
                self._next_cleanup_at = time.monotonic() + (0 if removed >= 500 else 300)

            yield db

        self._schema_ready = True

    def _cleanup(self, db: sqlite3.Connection, now: float, batch_size: int) -> int:
        expired = db.execute(
            "DELETE FROM state WHERE rowid IN "
            "(SELECT rowid FROM state WHERE expires<=? LIMIT ?)",
            (now, batch_size),
        ).rowcount
        idle = db.execute(
            "DELETE FROM state WHERE rowid IN "
            "(SELECT rowid FROM state WHERE touched<=? LIMIT ?)",
            (now - self.retention_seconds, batch_size - expired),
        ).rowcount
        leases = db.execute(
            "DELETE FROM leases WHERE rowid IN "
            "(SELECT rowid FROM leases WHERE expires<=? LIMIT ?)",
            (now, batch_size),
        ).rowcount
        return expired + idle + leases

    def cleanup(self, *, now: float | None = None, batch_size: int = 500) -> int:
        """同步清理一批：至多 batch_size 条状态及同等数量租约，返回删除总数。

        SQLite 空闲页供后续写入复用；这里不执行阻塞性的 VACUUM。
        """
        if batch_size < 1:
            raise ValueError("batch_size 必须大于零")

        now = time.time() if now is None else now
        # 显式清理不先触发自动批次，保证调用者指定的批次上限。
        self._next_cleanup_at = time.monotonic() + 300
        with self._transaction(now) as db:
            return self._cleanup(db, now, batch_size)

    def get(self, namespace: str, key: str, now: float | None = None) -> JsonValue:
        """同步读取 JSON 状态；不存在或已过期返回 None，now 为 Unix 秒。"""
        now = time.time() if now is None else now
        with self._transaction(now) as db:
            row = db.execute(
                "SELECT value,expires,touched FROM state WHERE namespace=? AND key=?",
                (namespace, key),
            ).fetchone()
            if row is None:
                return None

            expired = row["expires"] is not None and row["expires"] <= now
            idle = row["touched"] <= now - self.retention_seconds
            if expired or idle:
                db.execute("DELETE FROM state WHERE namespace=? AND key=?", (namespace, key))
                return None

            # 活跃引用续期，但不在每次读取时都写更新时间。
            if now - row["touched"] >= min(3600.0, self.retention_seconds / 4):
                db.execute("UPDATE state SET touched=? WHERE namespace=? AND key=?", (now, namespace, key))

            return json.loads(row["value"])

    def put(
        self, namespace: str, key: str, value: JsonValue, expires: float | None = None
    ) -> None:
        """写入 JSON 状态并更新本地保留期；expires 为独立的绝对 Unix 秒。"""
        now = time.time()
        with self._transaction(now) as db:
            db.execute(
                "INSERT INTO state(namespace,key,value,expires,touched) VALUES(?,?,?,?,?) "
                "ON CONFLICT(namespace,key) DO UPDATE SET "
                "value=excluded.value,expires=excluded.expires,touched=excluded.touched",
                (namespace, key, json.dumps(value, ensure_ascii=False), expires, now),
            )

    def delete(self, namespace: str, key: str) -> None:
        """同步删除指定状态；不存在时无操作。"""
        with self._transaction() as db:
            db.execute(
                "DELETE FROM state WHERE namespace=? AND key=?", (namespace, key)
            )

    def claim(self, key: str, owner: str, seconds: float = 120) -> bool:
        """原子获取上传租约，成功返回 True；seconds 为持有秒数。"""
        now = time.time()
        with self._transaction(now) as db:
            db.execute("DELETE FROM leases WHERE key=? AND expires<=?", (key, now))
            cursor = db.execute(
                "INSERT OR IGNORE INTO leases VALUES(?,?,?)",
                (key, owner, now + seconds),
            )
            return cursor.rowcount == 1

    def release(self, key: str, owner: str) -> None:
        """仅释放 owner 持有的上传租约，不影响其他上传者。"""
        with self._transaction() as db:
            db.execute("DELETE FROM leases WHERE key=? AND owner=?", (key, owner))

    def renew_lease(self, key: str, owner: str, seconds: float = 120) -> bool:
        """延长本人租约；已被接管时返回 False，调用方必须停止当次上传。"""
        with self._transaction() as db:
            cursor = db.execute(
                "UPDATE leases SET expires=? WHERE key=? AND owner=?",
                (time.time() + seconds, key, owner),
            )
            return cursor.rowcount == 1
