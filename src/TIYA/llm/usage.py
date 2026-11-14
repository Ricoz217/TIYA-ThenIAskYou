"""本地用量估算与逐次请求账本；供应商真实用量和本地估算独立保存。"""

from __future__ import annotations

import json
import sqlite3
import time
import tiktoken

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from .config_model import JsonValue, LLMConfig, Params
from .model import (
    BasePrompt,
    Context,
    ContextOverflowError,
    ImagePrompt,
    SystemPrompt,
    TextPrompt,
    ToolCall,
    ToolInput,
    ToolResponse,
)
from .pricing import PriceCatalog
from .storage import Database
from .usage_model import RequestUsageRecord, TokenUsage
from .usage_summary import MonthlySummary, _empty_summary, notify_ledger_change
from .workers import run_sync


class UsageEstimate(TypedDict):
    """本地估算及算法/完整性说明；不能作为供应商实测用量。"""

    input: int
    cached: int
    algorithm: str
    completeness: Literal["lower_bound", "heuristic"]
    assumptions: str


def reported_context_tokens(
    usage: TokenUsage | None,
    route: LLMConfig,
    usage_scope: str | None = None,
) -> int | float | None:
    """读取同模型/连接的真实水位；缺少完整计量或来源不匹配时返回 None。

    调用方先校验 Context 签名。存档没有连接 scope 时仍须匹配 model；
    不读取 effective、估算值或历史累计量，零为有效实测值。
    """
    if usage is None or not usage.samples:
        return None

    latest = usage.samples[-1]
    if latest.model != route.model:
        return None

    if usage_scope is not None and usage_scope != route.scope:
        return None

    # 一轮工具循环也可能包含多个请求；取最后一笔，不能把重复前缀计量累加作水位。
    reported = latest.reported
    total = reported.get("total")
    if isinstance(total, (int, float)):
        return total

    input_tokens = reported.get("input")
    output_tokens = reported.get("output")
    if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
        return input_tokens + output_tokens

    return None


def check_context_window(
    tokens: int | float,
    route: LLMConfig,
    auto_compress_rate: float | None,
) -> None:
    """按配置系数发出压缩信号，不比较裸窗口，不改变历史或自动换候选。"""
    if not route.max_context:
        return

    rate = max(0.3, min(0.9, auto_compress_rate if auto_compress_rate is not None else 0.7))
    if tokens >= rate * route.max_context:
        raise ContextOverflowError(f"候选 {route.ref} 的上下文水位已达到配置阈值")


def estimate(context: Context) -> UsageEstimate:
    """估算上下文 token 下界，复用消息和 Context 上的进程内缓存。

    返回 input/cached 和算法、完整性及假设说明；cached=0 是本地估算，
    不代表供应商真实缓存命中。原生回放元数据不重复计入，图片按固定下界计入。
    不进行网络或图片 IO，不修改业务内容。追加消息仅重算汇总和新增内容；
    显式修改内容后调用 context.refresh_fingerprints() 清除旧估算。
    """
    total = context.estimated_tokens
    if total is None:
        total = sum(_estimate_prompt(prompt) for prompt in context)
        context.estimated_tokens = total

    return {
        "input": total,
        "cached": 0,
        "algorithm": "o200k-image1440-v1",
        "completeness": "lower_bound",
        "assumptions": "images=1440 each; remote output and cached input unknown",
    }


async def estimate_async(context: Context) -> UsageEstimate:
    """已缓存的汇总直接读取；需要汇总/编码时交给 LLM 专用线程池。"""
    # 这里写得有点绕，不过算是能跑
    if context.estimated_tokens is not None:
        return estimate(context)

    return await run_sync(estimate, context)


def _estimate_prompt(prompt: BasePrompt) -> int:
    # 图片计数便宜，且准备/重建派生件可能改变数量，不缓存图片的单条计数。
    if isinstance(prompt, ImagePrompt):
        return 1440 * max(1, len(prompt.image))

    if prompt.estimated_tokens is not None:
        return prompt.estimated_tokens

    tokenizer = tiktoken.get_encoding("o200k_base")
    text = _prompt_text(prompt)
    prompt.estimated_tokens = len(tokenizer.encode(text, disallowed_special=())) + 3
    return prompt.estimated_tokens


def _prompt_text(prompt: BasePrompt) -> str:
    if isinstance(prompt, (TextPrompt, SystemPrompt)):
        text = prompt.text + getattr(prompt, "reasoning_content", "")

    elif isinstance(prompt, ToolCall):
        text = json.dumps(
            [prompt.function_name, prompt.arguments],
            ensure_ascii=False,
            default=str,
        )

    elif isinstance(prompt, ToolResponse):
        text = json.dumps(prompt.response, ensure_ascii=False, default=str)

    elif isinstance(prompt, ToolInput):
        text = json.dumps(
            prompt.tool_payloads["openai"],
            ensure_ascii=False,
            default=str,
        )

    else:
        text = json.dumps(
            {k: v for k, v in vars(prompt).items() if k not in ("native", "id", "time")},
            default=str,
        )

    return text


def assume_usage(context: Context) -> UsageEstimate:
    """账本的低成本假设值：字符数约除以三，图片每张 1440，不运行 tokenizer。

    这不是上下文上限判断，也不保证下界；真实 reported 按字段优先。
    结构化工具内容首次序列化后缓存计数，历史消息不反复序列化。
    由 Runtime 在线程池调用，消息显式刷新时同时清除缓存。
    """
    total = 0
    for prompt in context:
        if isinstance(prompt, ImagePrompt):
            total += 1440 * max(1, len(prompt.image))
            continue

        if prompt._assumed_tokens is None:
            if isinstance(prompt, (TextPrompt, SystemPrompt)):
                length = len(prompt.text) + len(getattr(prompt, "reasoning_content", ""))

            else:
                length = len(_prompt_text(prompt))

            prompt._assumed_tokens = (length + 2) // 3 + 3

        total += prompt._assumed_tokens

    return {
        "input": total,
        "cached": 0,
        "algorithm": "characters-div3-image1440-v1",
        "completeness": "heuristic",
        "assumptions": "characters/3; images=1440 each; not a context limit or measured usage",
    }


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
        notify_ledger_change(self.db.path, request_id)
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
        notify_ledger_change(self.db.path, request_id)
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
    def _effective(reported: Params, estimated: Params) -> Params:
        keys = (reported.keys() | estimated.keys()) - {
            "algorithm",
            "completeness",
            "assumptions",
        }
        return {
            key: reported[key] if reported.get(key) is not None else estimated.get(key)
            for key in keys
        }

    @classmethod
    def _decode(cls, row: sqlite3.Row) -> RequestUsageRecord:
        reported = json.loads(row["reported"])
        estimated = json.loads(row["estimated"])
        return {
            "id": row["id"],
            "call_id": row["call_id"],
            "created": row["created"],
            "identity": json.loads(row["identity"]),
            "estimated": estimated,
            "reported": reported,
            "status": row["status"],
            "remote_state": row["remote_state"],
            "diagnostic": json.loads(row["diagnostic"]),
            "price_snapshot": json.loads(row["price_snapshot"]) if row["price_snapshot"] is not None else None,
            "charge": json.loads(row["charge"]) if row["charge"] is not None else None,
            "effective": cls._effective(reported, estimated),
        }

    def get_request(self, request_id: str) -> RequestUsageRecord:
        """同步读取完整尝试，含 estimated/reported/effective；不存在抛 KeyError。"""
        with self.db.transaction() as db:
            row = db.execute(
                "SELECT * FROM requests WHERE id=?", (request_id,)
            ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._decode(row)

    def mirror_request(self, row: RequestUsageRecord) -> None:
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
                "INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "estimated=excluded.estimated,reported=excluded.reported,status=excluded.status,"
                "remote_state=excluded.remote_state,diagnostic=excluded.diagnostic,charge=excluded.charge",
                encoded,
            )
        notify_ledger_change(self.db.path, row["id"])
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
    ) -> list[RequestUsageRecord]:
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
        notify_ledger_change(self.db.path)

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
        imported = self._summarize_imports(start, end)
        for key in ("input", "output", "cached"):
            for source in ("verified", "estimate"):
                result[key][source] += imported[key][source]

        for meter in ("input", "cached"):
            result["cache_basis"][meter] += imported["cache_basis"][meter]

        for currency, amount in imported["cost"].items():
            result["cost"][currency] = result["cost"].get(currency, 0) + amount

        return result

    def _summarize_imports(self, start: float, end: float | None) -> MonthlySummary:
        """仅汇总旧版导入数据；月度视图在首次加载/导入变化后调用。"""
        result = _empty_summary()
        with self.db.transaction() as db:
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
