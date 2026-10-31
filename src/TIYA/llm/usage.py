"""Durable per-attempt ledger. Reported and estimated evidence are independent."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .config_model import JsonValue, Params
from .pricing import PriceCatalog
from .storage import Database


class Ledger:
    def __init__(
        self, path: str | Path, callback: Callable[[], None] | None = None
    ) -> None:
        """同步 SQLite 账本；callback 在写入后通知订阅者，其异常不影响已提交账目。"""
        self.callback = callback
        self.db = Database(
            path,
            """
          CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,preset TEXT,created REAL NOT NULL,status TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS requests(
            id TEXT PRIMARY KEY,call_id TEXT NOT NULL REFERENCES calls(id),created REAL NOT NULL,identity TEXT NOT NULL,
            estimated TEXT NOT NULL,reported TEXT NOT NULL DEFAULT '{}',status TEXT NOT NULL,
            remote_state TEXT NOT NULL DEFAULT 'unknown',diagnostic TEXT NOT NULL DEFAULT '{}',
            price_snapshot TEXT,charge TEXT);
          CREATE INDEX IF NOT EXISTS requests_call ON requests(call_id);
          CREATE INDEX IF NOT EXISTS requests_created ON requests(created,id);
          CREATE INDEX IF NOT EXISTS requests_preset ON requests(json_extract(identity,'$.preset'),created);
          CREATE INDEX IF NOT EXISTS requests_model ON requests(json_extract(identity,'$.model'),created);
          CREATE INDEX IF NOT EXISTS requests_operation ON requests(json_extract(identity,'$.operation'),created);
          CREATE TABLE IF NOT EXISTS observations(
            id TEXT PRIMARY KEY,request_id TEXT NOT NULL,source TEXT NOT NULL,observation_key TEXT NOT NULL,
            values_json TEXT NOT NULL,raw_json TEXT,created REAL NOT NULL,
            UNIQUE(request_id,source,observation_key));
          CREATE TABLE IF NOT EXISTS charges(id TEXT PRIMARY KEY,request_id TEXT,observation_id TEXT,
            data TEXT NOT NULL,created REAL NOT NULL,UNIQUE(request_id,observation_id));
          CREATE TABLE IF NOT EXISTS aggregates(source TEXT PRIMARY KEY,data TEXT NOT NULL);
        """,
        )

    def begin_call(self, preset: str = "", **metadata: JsonValue) -> str:
        """创建逻辑调用并返回 call_id；metadata 保留调用兼容性，目前不持久化。"""
        call_id = uuid4().hex
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO calls VALUES(?,?,?,?)",
                (call_id, preset, time.time(), "pending"),
            )
        return call_id

    def finish_call(self, call_id: str, status: str) -> None:
        """同步设置逻辑调用终态，不删除任何已发送尝试。"""
        with self.db.transaction() as db:
            db.execute("UPDATE calls SET status=? WHERE id=?", (status, call_id))

    def begin_request(
        self,
        call_id: str,
        identity: Params,
        estimated: Params,
        *,
        price_snapshot: Params | None = None,
        request_id: str | None = None,
    ) -> str:
        """发送前同步创建尝试，返回全局唯一 request_id，并保存初始估算证据。
        identity 标识实际预设、连接、模型和操作；estimated 是本地假设值；
        price_snapshot 冻结当时的定价。Files/控制操作也单独建账。
        """
        request_id = request_id or uuid4().hex
        estimate = {"algorithm": "local-v1", "completeness": "lower_bound", **estimated}
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO requests(id,call_id,created,identity,estimated,status,price_snapshot) VALUES(?,?,?,?,?,?,?)",
                (
                    request_id,
                    call_id,
                    time.time(),
                    json.dumps(identity),
                    json.dumps(estimate),
                    "pending",
                    json.dumps(price_snapshot) if price_snapshot is not None else None,
                ),
            )
            db.execute(
                "INSERT INTO observations VALUES(?,?,?,?,?,?,?)",
                (
                    uuid4().hex,
                    request_id,
                    "estimated",
                    "initial",
                    json.dumps(estimate),
                    None,
                    time.time(),
                ),
            )
        return request_id

    def observe_usage(
        self,
        request_id: str,
        source: str,
        values: Params,
        observation_key: str,
        *,
        raw: JsonValue = None,
    ) -> str:
        """追加独立用量证据并返回观察 ID；source 仅为 reported 或 estimated。
        同一 request_id/source/observation_key 幂等；raw 保留原始供应商数据。
        缺失真实值不填成零，effective 按字段优先取 reported，再取 estimated。
        """
        if source not in ("reported", "estimated"):
            raise ValueError("usage source 必须为 reported 或 estimated")
        if any(isinstance(v, (int, float)) and v < 0 for v in values.values()):
            raise ValueError("usage 不能为负")
        observation_id = uuid4().hex
        with self.db.transaction() as db:
            old = db.execute(
                "SELECT id FROM observations WHERE request_id=? AND source=? AND observation_key=?",
                (request_id, source, observation_key),
            ).fetchone()
            if old:
                return old["id"]
            row = db.execute(
                "SELECT * FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(request_id)
            db.execute(
                "INSERT INTO observations VALUES(?,?,?,?,?,?,?)",
                (
                    observation_id,
                    request_id,
                    source,
                    observation_key,
                    json.dumps(values),
                    json.dumps(raw),
                    time.time(),
                ),
            )
            column = "reported" if source == "reported" else "estimated"
            updated = json.loads(row[column])
            updated.update({k: v for k, v in values.items() if v is not None})
            db.execute(
                f"UPDATE requests SET {column}=? WHERE id=?",
                (json.dumps(updated), request_id),
            )
            reported = updated if source == "reported" else json.loads(row["reported"])
            estimated = (
                updated if source == "estimated" else json.loads(row["estimated"])
            )
            effective = self._effective(reported, estimated)
            charge = PriceCatalog.calculate(
                json.loads(row["price_snapshot"]) if row["price_snapshot"] else None,
                effective,
            )
            charge["sources"] = {
                key: "reported" if reported.get(key) is not None else "estimated"
                for key in effective
            }
            encoded = json.dumps(charge, default=str)
            db.execute(
                "INSERT INTO charges VALUES(?,?,?,?,?)",
                (uuid4().hex, request_id, observation_id, encoded, time.time()),
            )
            db.execute("UPDATE requests SET charge=? WHERE id=?", (encoded, request_id))
        self._notify()
        return observation_id

    def finish_request(
        self,
        request_id: str,
        status: str,
        *,
        remote_state: str = "unknown",
        **diagnostic: JsonValue,
    ) -> None:
        """同步结束一次尝试；remote_state 区分确认拒绝、完成与未知，diagnostic 保存诊断。"""
        with self.db.transaction() as db:
            db.execute(
                "UPDATE requests SET status=?,remote_state=?,diagnostic=? WHERE id=?",
                (status, remote_state, json.dumps(diagnostic, default=str), request_id),
            )
        self._notify()

    def _notify(self):
        if self.callback:
            try:
                self.callback()
            except Exception:
                # A log subscriber must never turn a committed charge into a generation retry.
                pass

    @staticmethod
    def _effective(reported, estimated):
        keys = (reported.keys() | estimated.keys()) - {
            "algorithm",
            "completeness",
            "assumptions",
        }
        return {
            k: reported[k] if reported.get(k) is not None else estimated.get(k)
            for k in keys
        }

    @classmethod
    def _decode(cls, row):
        result = dict(row)
        for key in (
            "identity",
            "estimated",
            "reported",
            "diagnostic",
            "price_snapshot",
            "charge",
        ):
            result[key] = json.loads(result[key]) if result[key] is not None else None
        result["effective"] = cls._effective(result["reported"], result["estimated"])
        return result

    def get_request(self, request_id: str) -> Params:
        """同步读取完整尝试，含 estimated/reported/effective；不存在抛 KeyError。"""
        with self.db.transaction() as db:
            row = db.execute(
                "SELECT * FROM requests WHERE id=?", (request_id,)
            ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._decode(row)

    def mirror_request(self, row: Params) -> None:
        """同步镜像已有尝试，不重新分配 call_id/request_id，不增加第二份生成费用。"""
        with self.db.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO calls VALUES(?,?,?,?)",
                (
                    row["call_id"],
                    row["identity"].get("preset", ""),
                    row["created"],
                    "completed",
                ),
            )
            columns = (
                "id",
                "call_id",
                "created",
                "identity",
                "estimated",
                "reported",
                "status",
                "remote_state",
                "diagnostic",
                "price_snapshot",
                "charge",
            )
            encoded = [
                json.dumps(row[k])
                if k
                in (
                    "identity",
                    "estimated",
                    "reported",
                    "diagnostic",
                    "price_snapshot",
                    "charge",
                )
                and row[k] is not None
                else row[k]
                for k in columns
            ]
            db.execute(
                "INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET reported=excluded.reported,status=excluded.status,charge=excluded.charge",
                encoded,
            )
        self._notify()

    def query_requests(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        call_id: str | None = None,
        start: float = 0,
        end: float | None = None,
        preset: str | None = None,
        model: str | None = None,
        operation: str | None = None,
        scope: str | None = None,
        status: str | None = None,
    ) -> list[Params]:
        """同步分页查询，按 created/id 升序；时间范围为 Unix 秒，limit 为 1..1000。
        可按逻辑调用、预设、模型、操作、账户作用域及终态过滤，None 不限制该项。
        """
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("无效分页")
        sql = "SELECT * FROM requests WHERE created>=? AND created<=?"
        args = [start, time.time() if end is None else end]
        if call_id:
            sql += " AND call_id=?"
            args.append(call_id)
        for key, value in (
            ("preset", preset),
            ("model", model),
            ("operation", operation),
            ("scope", scope),
        ):
            if value is not None:
                sql += f" AND json_extract(identity,'$.{key}')=?"
                args.append(value)
        if status is not None:
            sql += " AND status=?"
            args.append(status)
        with self.db.transaction() as db:
            rows = db.execute(
                sql + " ORDER BY created,id LIMIT ? OFFSET ?", (*args, limit, offset)
            ).fetchall()
        return [self._decode(row) for row in rows]

    def query_observations(
        self, request_id: str, *, limit: int = 100, offset: int = 0
    ) -> list[Params]:
        """同步分页读取该次尝试的原始证据，分别保留真实值和估算值。"""
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("无效分页")
        with self.db.transaction() as db:
            rows = db.execute(
                "SELECT * FROM observations WHERE request_id=? ORDER BY created,id LIMIT ? OFFSET ?",
                (request_id, limit, offset),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["values"] = json.loads(item.pop("values_json"))
            item["raw"] = json.loads(item.pop("raw_json") or "null")
            result.append(item)
        return result

    def import_aggregate(self, data: Params, source: str) -> None:
        # Historical aggregates are not invented request records. last_record is deliberately ignored.
        """幂等导入旧累计统计，不虚构逐请求记录，也不重复计入 last_record。"""
        with self.db.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO aggregates VALUES(?,?)",
                (source, json.dumps(data.get("storage", {}))),
            )

    def summarize(self, start: float = 0, end: float | None = None) -> Params:
        """同步汇总 Unix 时间范围内用量，区分真实/估算、币种费用和未知费用请求。"""
        result = {
            k: {"verified": 0, "estimate": 0} for k in ("input", "output", "cached")
        }
        result.update(
            cost={}, unknown_cost_requests=0, cache_basis={"input": 0, "cached": 0}
        )
        with self.db.transaction() as db:
            rows = db.execute(
                "SELECT identity,estimated,reported,charge FROM requests WHERE created>=? AND created<=?",
                (start, time.time() if end is None else end),
            )
            for row in rows:
                reported, estimated = (
                    json.loads(row["reported"]),
                    json.loads(row["estimated"]),
                )
                if (
                    reported.get("input") is not None
                    and reported.get("cached") is not None
                ):
                    result["cache_basis"]["input"] += reported["input"]
                    result["cache_basis"]["cached"] += reported["cached"]
                for key in ("input", "output", "cached"):
                    source = "verified" if reported.get(key) is not None else "estimate"
                    result[key][source] += (
                        reported if source == "verified" else estimated
                    ).get(key, 0) or 0
                charge = json.loads(row["charge"]) if row["charge"] else {}
                if charge.get("amount") is None:
                    if (
                        json.loads(row["identity"]).get("token_usage")
                        != "not_applicable"
                    ):
                        result["unknown_cost_requests"] += 1
                else:
                    currency = charge["currency"]
                    result["cost"][currency] = result["cost"].get(
                        currency, 0.0
                    ) + float(charge["amount"])
            aggregates = db.execute("SELECT data FROM aggregates").fetchall()
        for aggregate in aggregates:
            for year, months in json.loads(aggregate["data"]).items():
                for month, days in months.items():
                    for day, items in days.items():
                        stamp = datetime(int(year), int(month), int(day)).timestamp()
                        if not start <= stamp <= (time.time() if end is None else end):
                            continue
                        for item in items.values():
                            for meter in ("input", "cached"):
                                result["cache_basis"][meter] += (
                                    item.get("usage", {})
                                    .get(meter, {})
                                    .get("verified", 0)
                                )
                            for key in ("input", "output", "cached"):
                                for source in ("verified", "estimate"):
                                    result[key][source] += (
                                        item.get("usage", {})
                                        .get(key, {})
                                        .get(source, 0)
                                    )
                            if item.get("price") is not None:
                                price = item["price"]
                                counts = {
                                    k: sum(item.get("usage", {}).get(k, {}).values())
                                    for k in ("input", "output", "cached")
                                }
                                amount = (
                                    (counts["input"] - counts["cached"])
                                    * price.get("input", 0)
                                    + counts["cached"] * price.get("cached", 0)
                                    + counts["output"] * price.get("output", 0)
                                ) / 1000000
                                currency = price.get("currency", "CNY")
                                result["cost"][currency] = (
                                    result["cost"].get(currency, 0) + amount
                                )
        return result
