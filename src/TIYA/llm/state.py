"""Transactional communication metadata, separate from business history and accounting."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config_model import JsonValue
from .storage import Database


class StateStore:
    def __init__(self, path: str | Path) -> None:
        """绑定通信状态 SQLite 路径；首次事务时初始化，不保存业务历史。"""
        self.db = Database(
            path,
            """
          CREATE TABLE IF NOT EXISTS state(namespace TEXT,key TEXT,value TEXT NOT NULL,expires REAL,
            PRIMARY KEY(namespace,key));
          CREATE TABLE IF NOT EXISTS leases(key TEXT PRIMARY KEY,owner TEXT NOT NULL,expires REAL NOT NULL);
        """,
        )

    def get(self, namespace: str, key: str, now: float | None = None) -> JsonValue:
        """同步读取 JSON 状态；不存在或已过期返回 None，now 为 Unix 秒。"""
        with self.db.transaction() as db:
            row = db.execute(
                "SELECT value,expires FROM state WHERE namespace=? AND key=?",
                (namespace, key),
            ).fetchone()
        if (
            row is None
            or row["expires"] is not None
            and row["expires"] <= (time.time() if now is None else now)
        ):
            return None
        return json.loads(row["value"])

    def put(
        self, namespace: str, key: str, value: JsonValue, expires: float | None = None
    ) -> None:
        """同步写入 JSON 状态；expires 为绝对 Unix 秒，None 表示无独立过期时间。"""
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO state VALUES(?,?,?,?) ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value,expires=excluded.expires",
                (namespace, key, json.dumps(value, ensure_ascii=False), expires),
            )

    def delete(self, namespace: str, key: str) -> None:
        """同步删除指定状态；不存在时无操作。"""
        with self.db.transaction() as db:
            db.execute(
                "DELETE FROM state WHERE namespace=? AND key=?", (namespace, key)
            )

    def claim(self, key: str, owner: str, seconds: float = 120) -> bool:
        """原子获取上传租约，成功返回 True；seconds 为持有秒数。"""
        now = time.time()
        with self.db.transaction() as db:
            db.execute("DELETE FROM leases WHERE key=? AND expires<=?", (key, now))
            cursor = db.execute(
                "INSERT OR IGNORE INTO leases VALUES(?,?,?)",
                (key, owner, now + seconds),
            )
            return cursor.rowcount == 1

    def release(self, key: str, owner: str) -> None:
        """仅释放 owner 持有的上传租约，不影响其他上传者。"""
        with self.db.transaction() as db:
            db.execute("DELETE FROM leases WHERE key=? AND owner=?", (key, owner))

    def renew_lease(self, key: str, owner: str, seconds: float = 120) -> bool:
        """延长本人租约；已被接管时返回 False，调用方必须停止当次上传。"""
        with self.db.transaction() as db:
            cursor = db.execute(
                "UPDATE leases SET expires=? WHERE key=? AND owner=?",
                (time.time() + seconds, key, owner),
            )
            return cursor.rowcount == 1
