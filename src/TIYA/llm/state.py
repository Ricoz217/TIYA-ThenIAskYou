"""Transactional communication metadata, separate from business history and accounting."""

from __future__ import annotations

import json
import time

from .storage import Database


class StateStore:
    def __init__(self, path):
        self.db = Database(
            path,
            """
          CREATE TABLE IF NOT EXISTS state(namespace TEXT,key TEXT,value TEXT NOT NULL,expires REAL,
            PRIMARY KEY(namespace,key));
          CREATE TABLE IF NOT EXISTS leases(key TEXT PRIMARY KEY,owner TEXT NOT NULL,expires REAL NOT NULL);
        """,
        )

    def get(self, namespace, key, now=None):
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

    def put(self, namespace, key, value, expires=None):
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO state VALUES(?,?,?,?) ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value,expires=excluded.expires",
                (namespace, key, json.dumps(value, ensure_ascii=False), expires),
            )

    def delete(self, namespace, key):
        with self.db.transaction() as db:
            db.execute(
                "DELETE FROM state WHERE namespace=? AND key=?", (namespace, key)
            )

    def claim(self, key, owner, seconds=120):
        now = time.time()
        with self.db.transaction() as db:
            db.execute("DELETE FROM leases WHERE key=? AND expires<=?", (key, now))
            cursor = db.execute(
                "INSERT OR IGNORE INTO leases VALUES(?,?,?)",
                (key, owner, now + seconds),
            )
            return cursor.rowcount == 1

    def release(self, key, owner):
        with self.db.transaction() as db:
            db.execute("DELETE FROM leases WHERE key=? AND owner=?", (key, owner))

    def renew_lease(self, key, owner, seconds=120):
        with self.db.transaction() as db:
            cursor = db.execute(
                "UPDATE leases SET expires=? WHERE key=? AND owner=?",
                (time.time() + seconds, key, owner),
            )
            return cursor.rowcount == 1
