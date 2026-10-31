"""Small transactional SQLite foundation; no connection is shared across threads."""

from __future__ import annotations

import sqlite3
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path


_INIT_LOCKS = weakref.WeakValueDictionary()
_INIT_LOCKS_GUARD = threading.Lock()


def _initialization_lock(path):
    # Different stores can open the same database concurrently during first use.
    # Serializing WAL/schema setup by path avoids an immediate journal-mode BUSY.
    with _INIT_LOCKS_GUARD:
        key = path.resolve()
        lock = _INIT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _INIT_LOCKS[key] = lock
        return lock


class Database:
    def __init__(self, path: Path, schema: str):
        self.path = Path(path)
        self.schema = schema
        self._ready = False
        self._lock = _initialization_lock(self.path)

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA foreign_keys=ON")
            with self._lock:
                if not self._ready:
                    if connection.execute("PRAGMA user_version").fetchone()[0] > 1:
                        raise RuntimeError(
                            "SQLite schema 版本高于当前通信模块，拒绝降级写入"
                        )

                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute("PRAGMA synchronous=FULL")
                    connection.executescript(self.schema)
                    connection.execute("PRAGMA user_version=1")
                    connection.commit()
                    self._ready = True

            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()

        except BaseException:
            connection.rollback()
            raise

        finally:
            connection.close()
