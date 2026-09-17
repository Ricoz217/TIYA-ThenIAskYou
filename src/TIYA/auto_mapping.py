from __future__ import annotations

from json import JSONDecodeError

"""
auto_mapping.py
将这个自动映射表工具正式独立出来，作为模块。
改用 SQLite 存储，优化完整 json 重构的性能成本
"""
__version__ = "0.1.0"
__schema_version__ = 1

import asyncio
import copy
import heapq
import json
import os
import sqlite3
import threading
import time
import weakref
import traceback

from collections.abc import Callable
from itertools import count
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar
from concurrent.futures import as_completed

from TIYA.executor import AUTOMAPPING_EXECUTOR
from TIYA.logger import get_logger


_V = TypeVar("_V")
PersistPeriod = Literal["UPDATE", "SECOND", "MINUTE", "HOUR", "DAY", "WEEK"]
MutationOperation = Literal["upsert", "delete", "touch"]
_PERIOD_SECONDS: dict[PersistPeriod, float] = {
    "UPDATE": 0.0,
    "SECOND": 1.0,
    "MINUTE": 60.0,
    "HOUR": 3600.0,
    "DAY": 86400.0,
    "WEEK": 604800.0,
}
DAY_TO_SECOND = 86400
_MISSING = object()
_INSTANCES: weakref.WeakSet[AutoMapping[Any]] = weakref.WeakSet()
_INITIATE_LOCK = threading.Lock()
_ACCEPT_NEW_INSTANCE = True
_log = get_logger()


class AutoMappingError(RuntimeError):
    """Base exception for the SQLite AutoMapping implementation."""


class AutoMappingMigrationError(AutoMappingError):
    """Raised when a legacy JSON file cannot be migrated safely."""


class AutoMappingPersistenceError(AutoMappingError):
    """Raised when a requested persistence barrier cannot be completed."""


class AutoMappingClosedError(AutoMappingError):
    """Raised when a mutating operation is attempted after closing."""


@dataclass(slots=True)
class _Entry(Generic[_V]):
    data: _V
    updated_at: float
    generation: int

    def copy(self) -> _Entry[_V]:
        """浅复制"""
        return self.__copy__()

    def __copy__(self) -> _Entry[_V]:
        return _Entry(
            data=self.data,
            updated_at=self.updated_at,
            generation=self.generation
        )


class AutoMapping(Generic[_V]):
    """
    自动管理的映射表
    支持周期持久化、自动清理过期键
    """
    def __init__(
            self,
            save_path: Path | str | os.PathLike[str],
            persist_period: PersistPeriod = "MINUTE",
            expire_day: int | Callable[..., int] = 0,
            default: dict[str, _V] = None,
            value_type: type | None = None
    ):
        """
        自动管理的映射表
        支持周期持久化、自动清理过期键
        :param save_path: 持久化文件路径
        :param persist_period: 持久化周期，UPDATE为每次读取
        :param expire_day: 大于0时自动清理过期项目
        :param value_type: 可选值类型适配，目前支持 set 的首层 JSON 双向转换
        """

        # 延迟导入 TTL 日，让配置先读取
        def _expire_day_helper() -> int:
                return expire_day

        if default is None:
            default = {}

        else:
            default = default.copy()

        # 配置项
        self._save_path = Path(save_path)
        self._expire_day_input: Callable[..., int] = expire_day if callable(expire_day) else _expire_day_helper
        self._persist_period: PersistPeriod = persist_period
        self._default_values: dict[str, _V] = default
        self._value_type = value_type

        if value_type is set:
            self._json_encoder = self._json_encoder_set
            self._json_decoder = self._json_decoder_set

        else:
            self._json_encoder = self._json_encoder_common
            self._json_decoder = self._json_decoder_common

        # 数据容器
        self._mapping: dict[str, _Entry] = {}
        self._dirty_data: dict[str, tuple[_Entry[_V], MutationOperation]] = {}
        self._expiry_heap: list[tuple[float, int, str]] = []  # 最小堆，用于弹出 TTL

        # 锁，我chovy，上锁给我上好了啊！
        self._data_lock = threading.RLock()  # 管内存数据和堆，因为堆也是数据变动
        self._database_lock = threading.RLock()  # 管数据库
        self._drain_lock = threading.Lock()  # 用于确保持久化串行
        self._condition = threading.Condition()  # 自己一把锁，只管队列

        # 标识符
        self._generation_counter = count()
        self._commit_counter = count()
        self._last_persist_time = time.time()

        self._enable = False
        self._draining_expire = False
        self._persist_awake = False

        # SQLite
        self._connection: sqlite3.Connection | None = None

        # LOOP
        self._persist_loop_thread = threading.Thread(target=self._persist_loop_database, daemon=True)

        # 启动！
        with _INITIATE_LOCK:
            if not _ACCEPT_NEW_INSTANCE:
                return

            self._init_future = AUTOMAPPING_EXECUTOR.submit(self._initiate)
            self._init_future.add_done_callback(self._on_initiate_done)

    def _initiate(self):
        """启动流程"""
        json_path = self._resolve_legacy_json_filepath()
        database_path = self._resolve_database_filepath()

        # 判断是否要 migration
        if json_path.is_file() and not database_path.exists():
            self._migration_json_to_sqlite()

        else:
            self._load_data_from_database()

        self._rebuild_heap()

        with _INITIATE_LOCK:
            if not _ACCEPT_NEW_INSTANCE:
                return

            self._enable = True
            self._persist_loop_thread.start()
            _INSTANCES.add(self)

        self._first_persist()

    def _on_initiate_done(self, future):
        try:
            future.result()

        except BaseException as E:
            _log.error(
                f"[AUTOMAPPING] 初始化失败: {self._save_path} {E}\n\n"
                f"{traceback.format_exc()}"
            )

    # =========================================================
    # 基本逻辑，只处理内存态映射表，不做额外处理
    # =========================================================

    @staticmethod
    def _json_encoder_set(value: _V) -> list | _V:
        """用于单独处理集合转换数组的情况，只会转换第一层"""
        if isinstance(value, set):
            return list(value)

        return value

    @staticmethod
    def _json_decoder_set(value: _V) -> set | _V:
        """用于单独处理集合转换数组的情况，只会转换第一层"""
        if isinstance(value, list):
            return set(value)

        return value

    @staticmethod
    def _json_encoder_common(value: _V) -> _V:
        return value

    @staticmethod
    def _json_decoder_common(value: _V) -> _V:
        return value

    def _entry_to_dict(self, entry: _Entry) -> dict:
        return {
            "data": self._json_encoder(entry.data),
            "update_at": entry.updated_at,
            "generation": entry.generation
        }

    def _dict_to_entry(
            self,
            data: _V = _MISSING,
            update_at: float = _MISSING,
            generation: int = _MISSING
    ) -> _Entry[_V]:
        if not (data is not _MISSING and isinstance(update_at, float) and isinstance(generation, int)):
            raise ValueError("所有元数据必须齐全")

        return _Entry(
            data=self._json_decoder(data),
            updated_at=update_at,
            generation=generation
        )

    def to_dict(self) -> dict:
        """生成一个持久化用的，附带元数据的字典。目前这个方法已不再使用"""
        with self._data_lock:
            return {k: self._entry_to_dict(v) for k, v in self._mapping.items()}

    def load_dict(self, data: dict):
        """从持久化字典生成映射表，会覆盖当前映射表，同时不会自动覆盖数据库"""
        if not data or not isinstance(data, dict):
            return

        with self._data_lock:
            self._mapping.clear()
            for k, v in data.items():
                self._mapping[k] = self._dict_to_entry(
                    data=v["data"],
                    update_at=v["update_at"],
                    generation=v["generation"]
                )

            self._rebuild_heap()

    def update(self, data: dict[str, _V]):
        """批量更新映射"""
        with self._data_lock:
            for k, v in data.items():
                if k not in self._mapping:
                    self._mapping[k] = new_entry = self._create_new_generation_entry(data=v)
                    self._dirty_data[k] = (new_entry, "upsert")
                    self._entry_heappush_locked(k, new_entry)

                else:
                    self._mark_metadata_update_locked(k, v)

        self._persist_notify()

    def update_from_dict(self, data: dict[str, _V]):
        """兼容旧接口"""
        return self.update(data)

    def get(self, key: str, default: Any = None) -> _V | None:
        need_persist = False
        with self._data_lock:
            if key not in self._mapping:
                if key in self._default_values:
                    return self._default_values[key]

                return default

            result = self._mapping[key].data
            if key not in self._dirty_data:
                need_persist = True

            self._mark_metadata_update_locked(key)

        if need_persist:
            self._persist_notify()

        return result

    def remove(self, key: str):
        if not isinstance(key, str):
            raise TypeError("AutoMapping 仅支持字符串作为键")

        self._delete_item(key)

    def touch(self, key: str):
        """刷新指定元素的 TTL，KEY 不存在时静默"""
        need_persist = False
        with self._data_lock:
            if not isinstance(key, str):
                raise TypeError("AutoMapping 仅支持字符串作为键")
            if key not in self._dirty_data:
                need_persist = True
            self._touch_locked(key)

        if need_persist:
            self._persist_notify()

    def _touch_locked(self, key: str):
        if key not in self._mapping:
            return

        self._mark_metadata_update_locked(key, self._mapping[key].data)

    def pop(self, key: str, default: Any = None) -> _V | None:
        if not isinstance(key, str):
            raise TypeError("AutoMapping 仅支持字符串作为键")

        with self._data_lock:
            if key not in self._mapping:
                return default

            result = self._mapping[key].data

        self._delete_item(key)
        return result

    def setdefault(self, key: str, default: _V = None) -> _V | None:
        need_persist = False
        if not isinstance(key, str):
            raise TypeError("AutoMapping 仅支持字符串作为键")

        with self._data_lock:
            if key not in self._dirty_data:
                need_persist = True

            if key in self._mapping:
                result = self._mapping[key].data
                self._mark_metadata_update_locked(key)

            else:
                new_entry = _Entry(
                    data=default,
                    updated_at=time.time(),
                    generation=next(self._generation_counter)
                )
                self._mapping[key] = new_entry
                self._dirty_data[key] = (new_entry, "upsert")
                self._entry_heappush_locked(key, new_entry)
                result = default

        if need_persist:
            self._persist_notify()

        return result

    def _delete_item(self, key: str):
        with self._data_lock:
            if key not in self._mapping:
                raise KeyError(f"Key {key} not exist")

            entry= self._mapping[key]
            self._dirty_data[key] = (entry, "delete")
            self._mapping.pop(key, None)

        self._persist_notify()

    def _set_item(self, key: str, value: _V):
        with self._data_lock:
            if key in self._mapping:
                self._mark_metadata_update_locked(key, value)

            else:
                new_entry = _Entry(
                    data=value,
                    updated_at=time.time(),
                    generation=next(self._generation_counter)
                )
                self._mapping[key] = new_entry
                self._dirty_data[key] = (new_entry, "upsert")
                self._entry_heappush_locked(key, new_entry)

        self._persist_notify()

    def _get_item(self, key: str):
        need_persist = False
        with self._data_lock:
            if key not in self._mapping:
                raise KeyError(f"Key {key} not exist")

            value = self._mapping[key].data
            if key not in self._dirty_data:
                need_persist = True

            self._mark_metadata_update_locked(key)

        if need_persist:
            self._persist_notify()

        return value

    # =========================================================
    # 魔法方法
    # =========================================================

    def __contains__(self, item):
        if not isinstance(item, str):
            raise TypeError("AutoMapping 仅支持字符串作为键")

        # 不获取锁
        return item in self._mapping

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError("AutoMapping 仅支持字符串作为键")

        return self._get_item(item)

    def __setitem__(self, key, value: _V):
        if not isinstance(key, str):
            raise TypeError("AutoMapping 仅支持字符串作为键")

        self._set_item(key, value)

    def __delitem__(self, key: str):
        if not isinstance(key, str):
            raise TypeError("AutoMapping 仅支持字符串作为键")

        self._delete_item(key)

    def __len__(self):
        return len(self._mapping)

    def __iter__(self):
        # 会创建一个 KEY 镜像，模拟字典的做法
        with self._data_lock:
            mirror = self._mapping.copy().keys()

        return iter(mirror)

    def __reversed__(self):
        # 会创建一个 KEY 镜像，模拟字典的做法
        with self._data_lock:
            mirror = self._mapping.copy().keys()

        return reversed(mirror)

    # =========================================================
    # 数据转换接口
    # =========================================================

    def deepcopy(self) -> dict[str, _V]:
        """深复制，返回一个字典，破坏引用关系，不再具备功能"""
        with self._data_lock:
            return copy.deepcopy({k: v.data for k, v in self._mapping.items()})

    def keys(self):
        """为镜像"""
        with self._data_lock:
            return tuple(self._mapping.keys())

    def values(self):
        with self._data_lock:
            return tuple(v.data for v in self._mapping.values())

    def items(self):
        with self._data_lock:
            return {k: v.data for k, v in self._mapping.items()}.items()

    def mapping(self) -> dict[str, _V]:
        """返回一个浅复制的 KV 映射表"""
        with self._data_lock:
            return {k: v.data for k, v in self._mapping.items()}

    # =========================================================
    # 元数据管理、TTL逻辑、持久化
    # =========================================================

    def _create_new_generation_entry(
            self,
            *,
            data: _V,
            update_at: float = _MISSING,
    ) -> _Entry[_V]:
        """创建一个新版本的元数据"""
        if not isinstance(update_at, float):
            update_at = time.time()

        return _Entry(
            data=data,
            updated_at=update_at,
            generation=next(self._generation_counter)
        )

    def _mark_metadata_update_locked(self, key: str, value: _V = _MISSING):
        """标记元数据更新，并放入脏数据"""
        if key not in self._mapping:
            raise KeyError("逻辑错误，必须先有数据才能更新")

        # 分两种情况，只有数据更新时才创建副本
        if value is not _MISSING:
            new_entry = self._mapping[key].copy()
            new_entry.updated_at = time.time()
            new_entry.data = value
            self._mapping[key] = new_entry
            self._dirty_data[key] = (new_entry, "upsert")

        else:
            entry = self._mapping[key]
            entry.updated_at = time.time()

            # 这里要小心，不能覆盖掉 UPSERT，无需更新时间，因为是同一个对象，时间会自动变更
            self._dirty_data.setdefault(key, (entry, "touch"))

    def _rebuild_heap(self):
        """重建整个堆，并重置 generation"""
        with self._data_lock:
            self._generation_counter = count()
            expire_second = self._expire_seconds
            self._expiry_heap.clear()
            for k, v in self._mapping.items():
                v.generation = next(self._generation_counter)
                self._expiry_heap.append(
                    (v.updated_at + expire_second if expire_second > 0 else v.updated_at, v.generation, k))

            heapq.heapify(self._expiry_heap)

    def _entry_to_heap_metadata(self, key: str, entry: _Entry[_V]) -> tuple[float, int, str]:
        return entry.updated_at + self._expire_seconds, entry.generation, key

    def _entry_heappush_locked(self, key: str, entry: _Entry[_V]):
        """将一个 entry 推入堆，只有在设置了 TTL 才会执行"""
        if self._expire_seconds <= 0:
            return

        metadata = self._entry_to_heap_metadata(key, entry)
        heapq.heappush(self._expiry_heap, metadata)
        self._keep_heap_health_locked()

    def _entry_heapreplace_locked(self, key: str, entry: _Entry[_V]):
        metadata = self._entry_to_heap_metadata(key, entry)
        heapq.heapreplace(self._expiry_heap, metadata)

    def _drain_expire(self):
        """唤醒一次 TTL 检查，自动将 TTL 清理函数提交到线程池"""
        if not self._enable:
            return

        with self._data_lock:
            if self._draining_expire:
                return

        AUTOMAPPING_EXECUTOR.submit(self._drain_expire_handler)

    def _drain_expire_handler(self):
        with self._data_lock:
            try:
                self._draining_expire = True
                need_persist = False
                if not self._expiry_heap:
                    return

                # 检查堆顶元素
                now = time.time()
                time_start = time.monotonic()  # 记录一个初始时间
                while self._expiry_heap:
                    duration = time.monotonic() - time_start
                    if duration > 0.01:
                        AUTOMAPPING_EXECUTOR.submit(self._drain_expire_handler)
                        return

                    heap_top = self._expiry_heap[0]
                    expire_at = heap_top[0]
                    generation = heap_top[1]
                    key = heap_top[2]

                    # 处理不存在
                    if key not in self._mapping:
                        heapq.heappop(self._expiry_heap)
                        continue

                    # 处理直接退出
                    if expire_at > now:
                        return

                    entry = self._mapping[key]

                    # 检查版本
                    if generation < entry.generation:
                        heapq.heappop(self._expiry_heap)
                        continue

                    # 检查是否真正过期
                    if now < entry.updated_at + self._expire_seconds:
                        # 这里也使用副本替换，以免遗留奇怪的引用问题。但无需数据库同步
                        new_entry = entry.copy()
                        new_entry.generation = next(self._generation_counter)
                        self._mapping[key] = new_entry
                        self._entry_heapreplace_locked(key, new_entry)
                        continue

                    else:
                        self._delete_item(key)
                        heapq.heappop(self._expiry_heap)
                        need_persist = True
                        continue

            finally:
                if need_persist:
                    self.persist()

                self._draining_expire = False

    def _keep_heap_health_locked(self):
        """一个简单的判断，防止堆爆炸"""
        total_metadata = len(self._expiry_heap)
        if total_metadata > 100_000 and total_metadata > 10 * len(self._mapping):
            self._rebuild_heap()

    def set_expire_days(self, days: int | Callable[..., int]):
        """设置新的 TTL"""
        if callable(days):
            self._expire_day_input = days

        else:
            self._expire_day_input = lambda : days

        if self._expire_seconds > 0:
            self._rebuild_heap()

    def flush(self):
        """
        强制持久化一次
        会等到数据库事务提交
        会造成阻塞
        """
        if not self._enable:
            raise AutoMappingPersistenceError("[AUTOMAPPING] Flush 失败，实例已关闭")

        sequence_start = next(self._commit_counter)
        success = self._drain_dirty()
        sequence_now = next(self._commit_counter) - 1
        if not success or sequence_now <= sequence_start:
            raise AutoMappingPersistenceError("[AUTOMAPPING] Flush 失败，未能成功提交事务")

    async def flush_async(self):
        """
        强制持久化一次
        会等到数据库事务提交
        """
        if not self._enable:
            raise AutoMappingPersistenceError("[AUTOMAPPING] Flush 失败，实例已关闭")

        loop = asyncio.get_running_loop()
        sequence_start = next(self._commit_counter)
        #noinspection PyTypeChecker
        future = loop.run_in_executor(AUTOMAPPING_EXECUTOR, self._drain_dirty)
        success = await future
        sequence_now = next(self._commit_counter) - 1
        if not success or sequence_now <= sequence_start:
            raise AutoMappingPersistenceError("[AUTOMAPPING] Flush 失败，未能成功提交事务")

    def persist(self):
        """强制触发一次数据库持久化，不会等待事务提交"""
        if not self._enable:
            return

        AUTOMAPPING_EXECUTOR.submit(self._drain_dirty)

    async def persist_async(self):
        """强制触发一次数据库持久化，不会等待事务提交"""
        return self.persist()

    def _first_persist(self):
        """初始化用一次"""
        if self._drain_dirty():
            with self._database_lock:
                with self._create_or_get_database_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO metadata(key, value) VALUES ('migration_completed', ?)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value
                        """,
                        ("1",)
                    )

        else:
            raise AutoMappingPersistenceError("首次持久化失败")

    # =========================================================
    # SQLite相关
    # =========================================================

    def _create_or_get_database_connection(self) -> sqlite3.Connection:
        """创建 SQLite 连接单例"""
        with self._database_lock:
            if self._connection is not None:
                return self._connection

            connection = sqlite3.connect(
                self._resolve_database_filepath(),
                check_same_thread=False,
                timeout=5.0
            )
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.row_factory = sqlite3.Row
            self._database_initiate_schema(connection)
            self._connection = connection
            return connection

    def _get_database_cursor(self) -> sqlite3.Cursor:
        """完全不会用到"""
        conn = self._create_or_get_database_connection()
        return conn.cursor()

    def _database_initiate_schema(
            self,
            connection: sqlite3.Connection,
    ):
        with connection as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS entries (
                    key TEXT PRIMARY KEY,
                    data_json TEXT,
                    update_at REAL NOT NULL
                );
                """
            )

            metadata = {
                "schema_version": str(__schema_version__),
                "serializer": "json",
                "last_committed_sequence": str(next(self._commit_counter)),
            }

            conn.executemany(
            """
                INSERT INTO metadata(key, value) VALUES (?, ?) 
                ON CONFLICT(key) DO UPDATE SET 
                value = excluded.value
            """,
                metadata.items()
            )
            conn.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('migration_completed', ?)
                ON CONFLICT(key) DO NOTHING
                """,
                ("0",)
            )

    def _persist_loop_database(self):
        """数据库持久化循环"""
        while self._enable:
            try:
                with self._condition:
                    if not self._persist_awake:
                        self._condition.wait()

                    self._persist_awake = False

                # 二次等待
                while self._enable:
                    delay = self._check_persist_delay()
                    if delay > 0:
                        with self._condition:
                            if not self._persist_awake:
                                self._condition.wait(timeout=delay)

                            self._persist_awake = False

                    else:
                        break

                # 不取锁检查，加速
                if self._dirty_data:
                    self._drain_dirty()

                if self._expire_seconds > 0 and self._expiry_heap and self._expiry_heap[0][0] <= time.time():
                    self._drain_expire()  # 唤醒 TTL 检查

            # 保护主循环
            except Exception as E:
                _log.error(f"[AUTOMAPPING] 持久化循环出错: {E}\n\n{traceback.format_exc()}")

    def _check_persist_delay(self) -> float:
        """控制持久化周期"""
        persist_period_seconds = _PERIOD_SECONDS[self._persist_period]
        if persist_period_seconds <= 0:
            return 0.0

        now = time.time()
        pass_time = now - self._last_persist_time
        return persist_period_seconds - pass_time

    def _persist_notify(self):
        """唤醒一次持久化"""
        with self._condition:
            self._condition.notify()
            self._persist_awake = True

    def _drain_dirty(self) -> bool:
        """处理数据变更，返回 True 代表成功"""
        with self._drain_lock:
            if not self._enable:
                return False

            with self._data_lock:
                if not self._dirty_data:
                    next(self._commit_counter)
                    return True

                # 创建一个镜像，然后释放锁
                dirty_data = self._dirty_data.copy()
                self._dirty_data.clear()

            # 创建临时元组
            upsert_data = []
            touch_data = []
            pending_delete = []

            try:
                for k, v in dirty_data.items():
                    if v[1] == "upsert":
                        json_dict = self._entry_to_dict(v[0])
                        upsert_data.append(
                            (
                                k,
                                json.dumps(json_dict["data"], ensure_ascii=False),
                                json_dict["update_at"]
                            )
                        )

                    elif v[1] == "touch":
                        touch_data.append(
                            (v[0].updated_at, k)
                        )

                    elif v[1] == "delete":
                        pending_delete.append(k)

            except Exception as E:
                _log.error(f"[AUTOMAPPING] 序列号写入数据时出错: {E}\n\n{traceback.format_exc()}")
                # 还原修改数据
                with self._data_lock:
                    for k, v in dirty_data.items():
                        self._dirty_data.setdefault(k, v)

                return False

            # 批量修改数据
            try:
                with self._database_lock:
                    with self._create_or_get_database_connection() as conn:
                        conn.executemany(
                            """
                            INSERT INTO entries(key, data_json, update_at)
                            VALUES (?, ?, ?)
                            ON CONFLICT(key) DO UPDATE SET 
                                data_json = excluded.data_json,
                                update_at = excluded.update_at
                            """,
                            upsert_data
                        )
                        conn.executemany(
                            """
                            UPDATE entries SET update_at = ? WHERE key = ?
                            """,
                            touch_data
                        )
                        conn.executemany(
                            """
                            DELETE FROM entries WHERE key = ?
                            """,
                            ((key,) for key in pending_delete)
                        )
                        conn.execute("UPDATE metadata SET value = ? WHERE key = 'last_committed_sequence'",
                                     (str(next(self._commit_counter)),))

                        self._last_persist_time = time.time()
                        return True

            except sqlite3.Error as E:
                _log.error(f"[AUTOMAPPING] 修改数据库时错误: {E}\n\n{traceback.format_exc()}")

                # 还原修改数据
                with self._data_lock:
                    for k, v in dirty_data.items():
                        self._dirty_data.setdefault(k, v)

                return False

    def _load_data_from_database(self):
        """初始化时从数据库加载数据"""
        with self._database_lock:
            if not self._resolve_database_filepath().is_file():
                self._create_or_get_database_connection()
                return

            else:
                with self._create_or_get_database_connection() as conn:
                    # 先判断 Schema 版本和是否迁移完毕
                    schema_version = conn.execute(
                        "SELECT value FROM metadata WHERE key = ?",
                        ("schema_version",)
                    ).fetchone()
                    migration_completed = conn.execute(
                        "SELECT value FROM metadata WHERE key = ?",
                        ("migration_completed",)
                    ).fetchone()

                    if str(__schema_version__) != schema_version["value"]:
                        raise AutoMappingError("[AUTOMAPPING] schema_version 不匹配，无法初始化")

                    # 读取数据
                    entries_data = conn.execute(
                        "SELECT * FROM entries"
                    ).fetchall()

        if migration_completed["value"] != '1' and self._resolve_legacy_json_filepath().is_file():
            self._migration_json_to_sqlite()
            return

        # 处理临时数据副本
        temp_mapping = {}
        bad_jsons = []
        for entry_raw in entries_data:
            key = entry_raw["key"]
            data_json = entry_raw["data_json"]
            update_at = entry_raw["update_at"]
            try:
                data = json.loads(data_json)

            except JSONDecodeError:
                bad_jsons.append(key)
                continue

            entry = _Entry(
                data=self._json_decoder(data),
                updated_at=update_at,
                generation=next(self._generation_counter)
            )
            temp_mapping[key] = entry

        if bad_jsons:
            _log.error(f"[AUTOMAPPING] 读取数据库时出错，无法解析数据 {len(bad_jsons)} 条")

        with self._data_lock:
            """直接原子替换"""
            self._mapping = temp_mapping

    # =========================================================
    # 旧 json 数据迁移
    # =========================================================

    def _migration_json_to_sqlite(self):
        try:
            json_filepath = self._resolve_legacy_json_filepath()
            if not json_filepath.is_file():
                return

            try:
                json_str = json_filepath.read_text(encoding="utf-8")
                load_content: dict = json.loads(json_str)

            except json.JSONDecodeError:
                raise AutoMappingError("[AUTOMAPPING] 迁移 JSON 旧映射时错误: 无法解析 JSON 文件")

            if not isinstance(load_content, dict):
                raise AutoMappingError("[AUTOMAPPING] 迁移 JSON 旧映射时错误: 获取到的数据不是字典")

            legacy_mapping = load_content.get("data", {})
            temp_mapping: dict[str, _Entry[_V]] = {}
            for k, v in legacy_mapping.items():
                if not isinstance(v, dict):
                    continue

                if "data" not in v:
                    continue

                entry_data = v["data"]
                update_at = v.get("update", time.time())
                entry_data_decoded = self._json_decoder(entry_data)

                # 创建 Entry，用临时字典保存
                entry = self._dict_to_entry(entry_data_decoded, update_at, generation=next(self._generation_counter))
                temp_mapping[k] = entry

        # 只要出任何错误都放弃，直接抛出异常结束，不动数据
        except Exception as E:
            raise AutoMappingMigrationError(f"[AUTOMAPPING] 迁移 JSON 旧映射时错误: {E}\n\n{traceback.format_exc()}")

        else:
            # 获取锁，进行一次数据替换
            with self._drain_lock:
                with self._data_lock:
                    self._mapping.clear()
                    self._dirty_data.clear()
                    self._mapping.update(temp_mapping)
                    for k, v in self._mapping.items():
                        self._dirty_data[k] = (v, "upsert")

            # 直接删除数据库文件
            with self._database_lock:
                if self._connection is not None:
                    self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    self._connection.close()
                    self._connection = None

                database_path = self._resolve_database_filepath()
                if database_path.exists():
                    database_path.unlink(missing_ok=True)

    # =========================================================
    # 其他逻辑
    # =========================================================

    @property
    def _expire_seconds(self) -> int:
        return self._expire_day_input() * DAY_TO_SECOND

    def _resolve_database_filepath(self) -> Path:
        stem = self._save_path.stem
        database_filepath = self._save_path.parent / f"{stem}.sqlite3"
        return database_filepath

    def _resolve_legacy_json_filepath(self) -> Path:
        stem = self._save_path.stem
        json_filepath = self._save_path.parent / f"{stem}.json"
        return json_filepath

    async def wait_ready(self) -> bool:
        """仅供调试，生产不使用"""
        if _ACCEPT_NEW_INSTANCE or not self._enable:
            try:
                await asyncio.shield(
                    asyncio.wrap_future(self._init_future)
                )

            except Exception as error:
                raise AutoMappingError(
                    f"AutoMapping 初始化失败: {self._save_path}"
                ) from error

        return self._enable

    def kill(self):
        """关闭、退出"""
        self._drain_dirty()
        self._enable = False
        self._persist_notify()
        self._persist_loop_thread.join()
        with self._data_lock:
            with self._database_lock:
                if self._connection is not None:
                    for _ in range(5):
                        result = self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                        if result["busy"] == 0:
                            break

                        time.sleep(0.1)

                    self._connection.close()
                    self._connection = None


def _shutdown_all_sync():
    global _ACCEPT_NEW_INSTANCE
    errors = []
    with _INITIATE_LOCK:
        _ACCEPT_NEW_INSTANCE = False
        futures = [AUTOMAPPING_EXECUTOR.submit(instance.kill) for instance in _INSTANCES]
        try:
            for future in as_completed(futures, timeout=4.5):
                if future.exception() is not None:
                    errors.append(future.exception())

        except TimeoutError:
            errors.append(TimeoutError)

        if errors:
            _log.error(AutoMappingPersistenceError(f"[AUTOMAPPING] 关闭实例时发生错误: \n\n{errors}"))

async def shutdown_automapping():
    """退出"""
    future = asyncio.to_thread(_shutdown_all_sync)
    await future

__all__ = [
    "AutoMapping",
    "AutoMappingClosedError",
    "AutoMappingError",
    "AutoMappingMigrationError",
    "AutoMappingPersistenceError",
    "shutdown_automapping",
]