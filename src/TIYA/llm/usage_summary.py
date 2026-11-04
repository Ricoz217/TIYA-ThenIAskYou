"""状态栏的月度内存视图：增量汇总用量，临时给无单价请求估价，不回写账本。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from .config_model import Params

if TYPE_CHECKING:
    from .pricing import PriceCatalog
    from .usage import Ledger


type Number = int | float


class MeterSummary(TypedDict):
    verified: Number
    estimate: Number


class CacheBasis(TypedDict):
    input: Number
    cached: Number


class MonthlySummary(TypedDict):
    """cost 包含临时估价；temporary_cost 是其中未写入原始请求的部分。"""

    input: MeterSummary
    output: MeterSummary
    cached: MeterSummary
    cost: dict[str, float]
    temporary_cost: dict[str, float]
    unknown_cost_requests: int
    cache_basis: CacheBasis


class _Counts(NamedTuple):
    input: Number = 0
    output: Number = 0
    cached: Number = 0


@dataclass(slots=True)
class _UnpricedRequest:
    ref: str
    usage: Params


@dataclass(slots=True)
class _Contribution:
    reported: _Counts
    estimated: _Counts
    cache_input: Number
    cache_cached: Number
    currency: str | None
    amount: Decimal | None
    unknown_cost: bool


def _empty_summary() -> MonthlySummary:
    return {
        "input": {"verified": 0, "estimate": 0},
        "output": {"verified": 0, "estimate": 0},
        "cached": {"verified": 0, "estimate": 0},
        "cost": {},
        "temporary_cost": {},
        "unknown_cost_requests": 0,
        "cache_basis": {"input": 0, "cached": 0},
    }


# 只传播本进程正式写入接口的失效通知。跨进程部署和外部 SQL 编辑不属于该缓存契约。
_registry_lock = threading.Lock()
_ledger_views: dict[Path, weakref.WeakSet[MonthlyOverview]] = {}
_price_views: dict[Path, weakref.WeakSet[MonthlyOverview]] = {}


def notify_ledger_change(path: Path, request_id: str | None = None) -> None:
    """提交后标记变化的请求；request_id=None 表示旧聚合导入发生变化。"""
    with _registry_lock:
        views = list(_ledger_views.get(path.resolve(), ()))

    for view in views:
        with view._lock:
            if request_id is None:
                view._imports_dirty = True

            elif view._month is not None:
                view._dirty.add(request_id)


def notify_price_change(path: Path) -> None:
    """价格发布成功后刷新相关状态栏；展示异常不能使已经成功的发布报错。"""
    with _registry_lock:
        views = list(_price_views.get(path.resolve(), ()))

    for view in views:
        with view._lock:
            view._prices_dirty = True

        if view.callback:
            try:
                view.callback()

            except Exception:
                pass


class MonthlyOverview:
    """只服务月度展示的可重建视图；原始请求和冻结费用始终由 Ledger 持有。"""

    def __init__(
        self,
        ledger: Ledger,
        prices: PriceCatalog | None = None,
        *,
        callback: Callable[[], None] | None = None,
    ) -> None:
        """绑定借用的账本/价库；同步 IO，由调用方在线程池中执行。

        首次或跨月时读取当月明细，之后仅重读被正式接口修改的请求。
        每条请求只缓存汇总贡献；无单价请求额外保留估价所需的有效用量。
        不修改 requests/observations/charges，重启后可从原始账本重建。
        """
        self.ledger = ledger
        self.prices = prices
        self.callback = callback
        self._lock = threading.RLock()
        self._month: tuple[int, int] | None = None
        self._start = 0.0
        self._end = 0.0
        self._rows: dict[str, _Contribution] = {}
        self._dirty: set[str] = set()
        self._totals = _empty_summary()
        self._imports = _empty_summary()
        self._imports_dirty = True
        self._frozen_cost: dict[str, Decimal] = {}
        self._unpriced: dict[str, _UnpricedRequest] = {}
        self._temporary: dict[str, tuple[str, Decimal]] = {}
        self._temporary_cost: dict[str, Decimal] = {}
        self._plans: dict[str, Params] = {}
        self._prices_dirty = True
        self._next_price_change: float | None = None
        with _registry_lock:
            _ledger_views.setdefault(ledger.db.path.resolve(), weakref.WeakSet()).add(self)
            if prices is not None:
                _price_views.setdefault(prices.db.path.resolve(), weakref.WeakSet()).add(self)

    def summary(self, *, at: float | None = None) -> MonthlySummary:
        """返回 at 所在自然月的独立展示快照，默认当前本地时间。

        原有费用使用冻结结果；只有原本缺少价格快照的请求才按当前价格临时估价。
        cost 包含这部分金额，temporary_cost 单独标出；缺少必要计量仍保持未知。
        """
        now = time.time() if at is None else at
        date = datetime.fromtimestamp(now)
        with self._lock:
            if self._month != (date.year, date.month):
                self._reset_month(date)

            changed: Iterable[str] = self._refresh_requests()
            if self._imports_dirty:
                self._imports = self.ledger._summarize_imports(self._start, self._end - 0.000001)
                self._imports_dirty = False

            prices_due = (
                self._next_price_change is not None
                and now >= self._next_price_change
            )
            if self.prices is not None and (self._prices_dirty or prices_due):
                try:
                    plans, self._next_price_change = self.prices.active_snapshots(at=now)
                    self._plans = {plan["ref"]: plan for plan in plans}
                    self._prices_dirty = False

                except (sqlite3.Error, OSError, ValueError):
                    self._plans = {}
                    self._prices_dirty = True

                self._temporary.clear()
                self._temporary_cost.clear()
                changed = self._unpriced.keys()

            for request_id in changed:
                self._revalue(request_id)

            return self._result()

    def _reset_month(self, date: datetime) -> None:
        start = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if start.month == 12:
            end = start.replace(year=start.year + 1, month=1)

        else:
            end = start.replace(month=start.month + 1)

        self._start = start.timestamp()
        self._end = end.timestamp()
        self._rows.clear()
        self._dirty.clear()
        self._totals = _empty_summary()
        self._frozen_cost.clear()
        self._unpriced.clear()
        self._temporary.clear()
        self._temporary_cost.clear()
        self._imports_dirty = True
        self._prices_dirty = True
        # 读取期间的写入在提交后再发失效通知；重复读取同一请求只替换贡献，不重复累计。
        with self.ledger.db.transaction() as db:
            for row in db.execute(
                "SELECT id,identity,estimated,reported,price_snapshot,charge "
                "FROM requests WHERE created>=? AND created<?",
                (self._start, self._end),
            ):
                self._replace(row)

        self._month = (date.year, date.month)

    def _refresh_requests(self) -> set[str]:
        changed = self._dirty
        if not changed:
            return set()

        identifiers = list(changed)
        with self.ledger.db.transaction() as db:
            for offset in range(0, len(identifiers), 500):
                batch = identifiers[offset:offset + 500]
                slots = ",".join("?" for _ in batch)
                for row in db.execute(
                    "SELECT id,identity,estimated,reported,price_snapshot,charge FROM requests "
                    f"WHERE created>=? AND created<? AND id IN ({slots})",
                    (self._start, self._end, *batch),
                ):
                    self._replace(row)

        self._dirty = set()
        return changed

    def _replace(self, row: sqlite3.Row) -> None:
        request_id = row["id"]
        old = self._rows.get(request_id)
        if old is not None:
            self._apply(old, -1)

        reported = json.loads(row["reported"])
        estimated = json.loads(row["estimated"])
        identity = json.loads(row["identity"])
        charge = json.loads(row["charge"]) if row["charge"] else {}
        unknown = (
            charge.get("amount") is None
            and identity.get("token_usage") != "not_applicable"
        )
        ref = identity.get("provider")
        unpriced = None
        if unknown and row["price_snapshot"] is None and isinstance(ref, str):
            unpriced = _UnpricedRequest(ref, self.ledger._effective(reported, estimated))

        cache_known = (
            reported.get("input") is not None and reported.get("cached") is not None
        )
        reported_counts = []
        estimated_counts = []
        for key in ("input", "output", "cached"):
            actual = reported.get(key)
            reported_counts.append(actual if actual is not None else 0)
            if actual is None:
                estimated_counts.append(estimated.get(key, 0) or 0)

            else:
                estimated_counts.append(0)

        amount = charge.get("amount")
        contribution = _Contribution(
            reported=_Counts(*reported_counts),
            estimated=_Counts(*estimated_counts),
            cache_input=reported["input"] if cache_known else 0,
            cache_cached=reported["cached"] if cache_known else 0,
            currency=charge.get("currency"),
            amount=Decimal(str(amount)) if amount is not None else None,
            unknown_cost=unknown,
        )
        self._rows[request_id] = contribution
        self._apply(contribution, 1)
        self._unpriced.pop(request_id, None)
        if unpriced is not None:
            self._unpriced[request_id] = unpriced

    def _apply(self, item: _Contribution, sign: int) -> None:
        for target, reported, estimated in (
            (self._totals["input"], item.reported.input, item.estimated.input),
            (self._totals["output"], item.reported.output, item.estimated.output),
            (self._totals["cached"], item.reported.cached, item.estimated.cached),
        ):
            target["verified"] += sign * reported
            target["estimate"] += sign * estimated

        self._totals["cache_basis"]["input"] += sign * item.cache_input
        self._totals["cache_basis"]["cached"] += sign * item.cache_cached
        self._totals["unknown_cost_requests"] += sign * item.unknown_cost
        if item.currency is not None and item.amount is not None:
            self._frozen_cost[item.currency] = (
                self._frozen_cost.get(item.currency, Decimal(0)) + sign * item.amount
            )

    def _revalue(self, request_id: str) -> None:
        old = self._temporary.pop(request_id, None)
        if old is not None:
            currency, amount = old
            self._temporary_cost[currency] -= amount

        item = self._unpriced.get(request_id)
        if item is None or self.prices is None:
            return

        charge = self.prices.calculate(self._plans.get(item.ref), item.usage)
        if charge["amount"] is not None:
            currency = charge["currency"]
            amount = charge["amount"]
            self._temporary[request_id] = currency, amount
            self._temporary_cost[currency] = (
                self._temporary_cost.get(currency, Decimal(0)) + amount
            )

    def _result(self) -> MonthlySummary:
        result = _empty_summary()
        for target, current, imported in (
            (result["input"], self._totals["input"], self._imports["input"]),
            (result["output"], self._totals["output"], self._imports["output"]),
            (result["cached"], self._totals["cached"], self._imports["cached"]),
        ):
            target["verified"] = current["verified"] + imported["verified"]
            target["estimate"] = current["estimate"] + imported["estimate"]

        result["cache_basis"] = {
            "input": self._totals["cache_basis"]["input"] + self._imports["cache_basis"]["input"],
            "cached": self._totals["cache_basis"]["cached"] + self._imports["cache_basis"]["cached"],
        }

        currencies = (
            self._frozen_cost.keys()
            | self._temporary_cost.keys()
            | self._imports["cost"].keys()
        )
        result["cost"] = {
            currency: float(
                self._frozen_cost.get(currency, Decimal(0))
                + self._temporary_cost.get(currency, Decimal(0))
                + Decimal(str(self._imports["cost"].get(currency, 0)))
            )
            for currency in currencies
        }
        result["temporary_cost"] = {
            currency: float(amount) for currency, amount in self._temporary_cost.items()
        }
        result["unknown_cost_requests"] = (
            self._totals["unknown_cost_requests"] - len(self._temporary)
        )
        return result
