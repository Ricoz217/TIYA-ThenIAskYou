from __future__ import annotations

import time

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from hashlib import blake2b
from copy import deepcopy
from TIYA.config import DATA_DIR
from TIYA.logger import get_logger
from .usage import Ledger
import shutil
import threading


_log = get_logger()
_status = _log.status()


@dataclass(slots=True)
class ApiPrice:
    """token单价, currency / 1M tokens"""

    input: float = 0
    output: float = 0
    cached: float = 0
    currency: str = "CNY"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return ApiPrice(
            input=data.get("input", 0),
            output=data.get("output", 0),
            cached=data.get("cached", 0),
            currency=data.get("currency", "CNY"),
        )

    def copy(self):
        return self.__copy__()

    def __bool__(self):
        return bool(self.input or self.output or self.cached)

    def __copy__(self):
        return ApiPrice(self.input, self.output, self.cached, self.currency)


@dataclass(slots=True)
class UsageStorage:
    model: str
    price: ApiPrice | None
    usage: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "input": {"estimate": 0, "verified": 0},
            "output": {"estimate": 0, "verified": 0},
            "cached": {"estimate": 0, "verified": 0},
        }
    )
    last_update: float = field(default_factory=time.time)

    def hash_id(self) -> str:
        data = {
            "model": self.model,
            "price": self.price.to_dict() if self.price is not None else None,
        }
        key = json.dumps(data, ensure_ascii=False).encode()
        h = blake2b(key, digest_size=16)
        return h.hexdigest()

    def to_dict(self):
        return {
            "model": self.model,
            "price": self.price.to_dict() if self.price is not None else None,
            "usage": self.usage,
            "last_update": self.last_update,
        }

    @property
    def input_t(self) -> dict[str, int]:
        return self.usage["input"]

    @property
    def output_t(self) -> dict[str, int]:
        return self.usage["output"]

    @property
    def cached_t(self) -> dict[str, int]:
        return self.usage["cached"]

    @property
    def total_t(self) -> dict[str, int]:
        return {
            "estimate": self.input_t["estimate"] + self.output_t["estimate"],
            "verified": self.input_t["verified"] + self.output_t["verified"],
        }

    # 已弃用
    def __sub__(self, other) -> UsageStorage:
        """Return a delta usage"""
        if not isinstance(other, UsageStorage):
            raise TypeError

        if self.model != other.model:
            raise ValueError("Different model name")

        if self.price != other.price:
            raise ValueError("Different price")

        new_data = {
            "input": {"estimate": 0, "verified": 0},
            "output": {
                "estimate": self.output_t["estimate"],
                "verified": self.output_t["verified"],
            },
            "cached": {"estimate": 0, "verified": 0},
        }

        # input
        if self.input_t["estimate"] and self.input_t["verified"]:
            new_data["input"]["estimate"] = (
                self.input_t["estimate"] - other.input_t["estimate"]
            )
            new_data["input"]["verified"] = (
                self.input_t["verified"] - other.input_t["verified"]
            )

        else:
            if self.input_t["estimate"]:
                new_data["input"]["estimate"] = sum(self.input_t.values())
                new_data["input"]["verified"] = -sum(other.input_t.values())

            elif self.input_t["verified"]:
                new_data["input"]["verified"] = sum(self.input_t.values())
                new_data["input"]["estimate"] = -sum(other.input_t.values())

        # cached
        if self.cached_t["estimate"] and self.cached_t["verified"]:
            new_data["cached"]["estimate"] = (
                self.cached_t["estimate"] - other.cached_t["estimate"]
            )
            new_data["cached"]["verified"] = (
                self.cached_t["verified"] - other.cached_t["verified"]
            )

        else:
            if self.cached_t["estimate"]:
                new_data["cached"]["estimate"] = sum(self.cached_t.values())
                new_data["cached"]["verified"] = -sum(other.cached_t.values())

            elif self.cached_t["verified"]:
                new_data["cached"]["verified"] = sum(self.cached_t.values())
                new_data["cached"]["estimate"] = -sum(other.cached_t.values())

        return UsageStorage(self.model, self.price, new_data)

    def __copy__(self):
        return UsageStorage(
            self.model,
            self.price.copy() if self.price is not None else None,
            deepcopy(self.usage),
            self.last_update,
        )

    # 已弃用
    @staticmethod
    def _redistribute(data: dict):
        """
        再分配token，不能低于0，同时更改可信标记
        :param data:
        :return:
        """
        count = sum(data.values())
        if count < 0:
            raise ValueError("Usage below zero")

        if data["estimate"] < 0:
            data["verified"] = count
            data["estimate"] = 0

        elif data["verified"] < 0:
            data["estimate"] = count
            data["verified"] = 0

    # 已弃用
    def normalize(self):
        self._redistribute(self.input_t)
        self._redistribute(self.output_t)
        self._redistribute(self.cached_t)

    def add(self, other: UsageStorage):
        if not isinstance(other, UsageStorage):
            raise TypeError

        if self.model != other.model:
            raise ValueError("Different model name")

        if self.price != other.price:
            raise ValueError("Different price")

        # input
        self.input_t["estimate"] += other.input_t["estimate"]
        self.input_t["verified"] += other.input_t["verified"]

        # output
        self.output_t["estimate"] += other.output_t["estimate"]
        self.output_t["verified"] += other.output_t["verified"]

        # cached
        self.cached_t["estimate"] += other.cached_t["estimate"]
        self.cached_t["verified"] += other.cached_t["verified"]

        self.last_update = time.time()

    def copy(self):
        return self.__copy__()

    @classmethod
    def from_dict(cls, data: dict):
        price = data.get("price", {})
        return UsageStorage(
            model=data["model"],
            price=ApiPrice.from_dict(price) if price is not None else None,
            usage=data.get(
                "usage",
                {
                    "input": {"estimate": 0, "verified": 0},
                    "output": {"estimate": 0, "verified": 0},
                    "cached": {"estimate": 0, "verified": 0},
                },
            ),
            last_update=data.get("last_update", time.time()),
        )


class LLMUsage:
    """Legacy aggregate/log API backed by a durable request ledger."""

    def __init__(self, data_file: Path, update_callback=None):
        self.data_file = Path(data_file)
        self.callback = update_callback
        self._ledger = None
        self._ledger_lock = threading.Lock()
        self._last_record = {}

    @property
    def ledger(self):
        with self._ledger_lock:
            if self._ledger is None:
                path = self.data_file.with_suffix(".sqlite3")
                opened = Ledger(
                    path, callback=lambda: self.callback() if self.callback else None
                )
                if self.data_file.suffix == ".json" and self.data_file.is_file():
                    data = json.loads(self.data_file.read_text(encoding="utf-8"))
                    backup = self.data_file.with_suffix(".json.bak")
                    if not backup.exists():
                        shutil.copy2(self.data_file, backup)
                    opened.import_aggregate(data, str(self.data_file.resolve()))
                self._ledger = opened
        return self._ledger

    @staticmethod
    def get_key(model, price=None):
        return UsageStorage(model, price).hash_id()

    def update(self, model, update_data, price=None, verified=True):
        call = self.ledger.begin_call("legacy")
        snapshot = None
        if price is not None:
            snapshot = {
                "version": "legacy",
                "rules": {
                    "currency": price.currency,
                    "tiers": [
                        {
                            "rates": {
                                "input_uncached": str(price.input),
                                "cached": str(price.cached),
                                "output": str(price.output),
                            }
                        }
                    ],
                    "units": {
                        "input_uncached": 1000000,
                        "cached": 1000000,
                        "output": 1000000,
                    },
                },
            }
        request = self.ledger.begin_request(
            call,
            {
                "model": model,
                "operation": "legacy_update",
                "legacy_price": price.to_dict() if price is not None else None,
            },
            update_data if not verified else {},
            price_snapshot=snapshot,
        )
        self.ledger.observe_usage(
            request, "reported" if verified else "estimated", update_data, "legacy"
        )
        self.ledger.finish_request(
            request, "completed", remote_state="completed" if verified else "unknown"
        )
        self.ledger.finish_call(call, "completed")
        item = UsageStorage(model, price)
        for key in ("input", "output", "cached"):
            item.usage[key]["verified" if verified else "estimate"] = update_data.get(
                key, 0
            )
        self._last_record[item.hash_id()] = item

    @property
    def _storage(self):
        storage = {}
        offset = 0
        while True:
            rows = self.ledger.query_requests(limit=1000, offset=offset)
            if not rows:
                break
            for row in rows:
                price = row["identity"].get("legacy_price")
                item = UsageStorage(
                    row["identity"].get("model", ""),
                    ApiPrice.from_dict(price) if price is not None else None,
                )
                for key in ("input", "output", "cached"):
                    reported = row["reported"].get(key)
                    item.usage[key][
                        "verified" if reported is not None else "estimate"
                    ] = row["effective"].get(key, 0) or 0
                stamp = datetime.fromtimestamp(row["created"])
                day = (
                    storage.setdefault(str(stamp.year), {})
                    .setdefault(str(stamp.month), {})
                    .setdefault(str(stamp.day), {})
                )
                key = item.hash_id()
                if key in day:
                    day[key].add(item)
                else:
                    day[key] = item
            offset += len(rows)
        with self.ledger.db.transaction() as db:
            imported = db.execute("SELECT data FROM aggregates").fetchall()
        for record in imported:
            for year, months in json.loads(record["data"]).items():
                for month, days in months.items():
                    for day, items in days.items():
                        target = (
                            storage.setdefault(year, {})
                            .setdefault(month, {})
                            .setdefault(day, {})
                        )
                        for key, item in items.items():
                            parsed = UsageStorage.from_dict(item)
                            if key in target:
                                target[key].add(parsed)
                            else:
                                target[key] = parsed
        return storage

    def load(self):
        return self.ledger

    def save(self):
        return None  # Every ledger mutation is already committed.

    def load_dict(self, data):
        from .config import fingerprint

        self.ledger.import_aggregate(data, fingerprint(data))

    def to_dict(self):
        def encode(value):
            if isinstance(value, UsageStorage):
                return value.to_dict()
            return {k: encode(v) for k, v in value.items()}

        return {
            "storage": encode(self._storage),
            "last_record": {k: v.to_dict() for k, v in self._last_record.items()},
        }

    def overview(self, period="MONTH"):
        now = datetime.now()
        starts = {
            "DAY": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "MONTH": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            "YEAR": now.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            ),
        }
        if period not in (*starts, "ALL"):
            raise ValueError(period)
        return self.ledger.summarize(
            starts[period].timestamp() if period != "ALL" else 0
        )

    def query(self, hash_key="", time_window=None):
        if time_window is None:
            time_window = datetime.now().timestamp()
        start, end = (
            time_window
            if isinstance(time_window, tuple)
            else (time_window, time_window)
        )
        start = (
            datetime.fromtimestamp(start)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
        )
        end = (
            datetime.fromtimestamp(end)
            .replace(hour=23, minute=59, second=59, microsecond=999999)
            .timestamp()
        )
        result = {}
        for year, months in self._storage.items():
            for month, days in months.items():
                for day, items in days.items():
                    if (
                        start
                        <= datetime(int(year), int(month), int(day)).timestamp()
                        <= end
                    ):
                        for key, item in items.items():
                            if not hash_key or key == hash_key:
                                result.setdefault(year, {}).setdefault(
                                    month, {}
                                ).setdefault(day, {})[key] = deepcopy(item)
        return result


_GLOBAL_FILE = DATA_DIR / "token_usage" / "usage.json"
_GLOBAL_USAGE = LLMUsage(_GLOBAL_FILE)


def _auto_flush():
    usage = _GLOBAL_USAGE.overview()
    input_t = sum(usage.get("input", {}).values())
    output_t = sum(usage.get("output", {}).values())
    cached_t = sum(usage.get("cached", {}).values())
    cost = {k: v for k, v in usage.get("cost", {}).items() if k in {"CNY", "USD"}}
    output = "本月Token消耗 | "
    output += f"总: [{input_t + output_t:,}]; "
    output += f"输入: [{input_t:,}]; "
    output += f"输出: [{output_t:,}]; "
    output += f"输入(命中缓存): [{cached_t:,}]; "
    reported_input = usage.get("cache_basis", {}).get("input", 0)
    reported_cached = usage.get("cache_basis", {}).get("cached", 0)
    output += f"缓存命中率(已报告): [{reported_cached / reported_input if reported_input else 0:.2%}]"
    estimated = usage.get("input", {}).get("estimate", 0) + usage.get("output", {}).get(
        "estimate", 0
    )
    if estimated:
        output += f"; 含本地估算Token: [{estimated:,}]"
    if usage.get("unknown_cost_requests"):
        output += f"; 费用未知: [{usage['unknown_cost_requests']:,} 请求]"

    if cost:
        output += " | "
        if CNY := cost.get("CNY", 0):
            output += f"CNY: [{CNY:,.2f}]"

        if USD := cost.get("USD", 0):
            output += f"USD: [{USD:,.2f}]"

    _status.update(output)


@dataclass(slots=True)
class UsageNow:
    all_token: int
    input_token: int
    cache_hit_token: int
    output_token: int
    CNY: float
    USD: float


def usage_now() -> UsageNow:
    usage = _GLOBAL_USAGE.overview()
    input_t = sum(usage.get("input", {}).values())
    output_t = sum(usage.get("output", {}).values())
    cached_t = sum(usage.get("cached", {}).values())
    cost = {k: v for k, v in usage.get("cost", {}).items() if k in {"CNY", "USD"}}
    now = UsageNow(
        all_token=input_t + output_t,
        input_token=input_t,
        cache_hit_token=cached_t,
        output_token=output_t,
        CNY=cost.get("CNY", 0.0),
        USD=cost.get("USD", 0.0),
    )
    return now


_GLOBAL_USAGE.callback = _auto_flush
