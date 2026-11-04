"""Versioned, separately maintained deterministic price catalog."""

from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from TIYA.utils import plain

from .config_model import Params
from .storage import Database
from .usage_summary import notify_price_change


class PriceCatalog:
    def __init__(self, path: str | Path) -> None:
        """绑定独立定价库，所有维护和查询方法都执行同步 SQLite IO。"""
        self.db = Database(
            path,
            """
            CREATE TABLE IF NOT EXISTS plans(
              id TEXT PRIMARY KEY, binding TEXT NOT NULL, identity TEXT NOT NULL,
              rules TEXT NOT NULL, source TEXT NOT NULL, effective REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS plans_binding ON plans(binding,effective);
        """,
        )

    def publish(
        self,
        ref: str,
        rules: Params,
        *,
        source: str,
        effective: float | None = None,
    ) -> str:
        """按配置中的 分类:连接名:模型ID 发布价格版本，返回独立的版本 UUID。

        rules 定义币种、计量单位及阶梯，source 记录核对来源，effective 为生效 Unix 秒。
        不依赖预设名、凭证或账户 scope；不覆盖旧版本，也不修改已冻结的请求价格。
        """
        self._validate_ref(ref)
        self._validate(rules)
        version = uuid4().hex
        with self.db.transaction() as db:
            db.execute(
                "INSERT INTO plans VALUES(?,?,?,?,?,?)",
                (
                    version,
                    ref,
                    json.dumps({"ref": ref}),
                    json.dumps(rules),
                    source,
                    time.time() if effective is None else effective,
                ),
            )
        notify_price_change(self.db.path)
        return version

    @staticmethod
    def _validate_ref(ref: str) -> None:
        if not isinstance(ref, str):
            raise ValueError("价格 ID 必须是 分类:连接名:模型ID")

        parts = ref.split(":", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError("价格 ID 必须是 分类:连接名:模型ID")

    @staticmethod
    def _validate(rules):
        if rules.get("tier_mode", "whole") not in ("whole", "progressive"):
            raise ValueError("tier_mode 必须是 whole 或 progressive")
        previous = 0
        tiers = rules.get("tiers", [])
        if not tiers or not rules.get("currency"):
            raise ValueError("需要 currency 和 tiers")
        for i, tier in enumerate(tiers):
            if not isinstance(tier.get("rates"), dict) or not tier["rates"]:
                raise ValueError("每个价格区间需要明确的非空 rates")
            end = tier.get("up_to")
            if end is None and i != len(tiers) - 1:
                raise ValueError("只有最后一个价格区间允许无上限")

            if end is not None and (
                isinstance(end, bool)
                or not isinstance(end, (int, float))
                or not Decimal(str(end)).is_finite()
            ):
                raise ValueError("价格区间上限必须是有限数值")

            if end is not None and (end <= previous or i == len(tiers) - 1):
                raise ValueError("价格区间必须递增，最后一个区间无上限")
            previous = end or previous
            if any(Decimal(str(rate)) < 0 for rate in tier.get("rates", {}).values()):
                raise ValueError("价格不能为负")
        meters = set().union(*(t.get("rates", {}) for t in tiers))
        if "input" in meters and (
            {"input_uncached", "cached", "input_text", "input_image"} & meters
        ):
            raise ValueError("输入总量不能与子项重复计价")
        if "output" in meters and ({"output_text", "output_image"} & meters):
            raise ValueError("输出总量不能与子项重复计价")
        if rules.get("tier_mode") == "progressive" and meters != {
            rules.get("threshold", "input")
        }:
            raise ValueError(
                "progressive 仅支持按同一计量项逐段计费；混合计量请分开计划"
            )
        if any(Decimal(str(v)) <= 0 for v in rules.get("units", {}).values()):
            raise ValueError("计价单位必须为正")

    def snapshot(
        self, ref: str, at: float | None = None
    ) -> Params | None:
        """按完整配置引用读取 at 时刻生效的价格；不猜连接名或回退到其他模型。"""
        self._validate_ref(ref)
        with self.db.transaction() as db:
            row = db.execute(
                "SELECT * FROM plans WHERE binding=? AND effective<=? ORDER BY effective DESC,rowid DESC LIMIT 1",
                (ref, time.time() if at is None else at),
            ).fetchone()
        if row is None:
            return None
        rules = json.loads(row["rules"])
        self._validate(rules)
        return {
            "version": row["id"],
            "ref": row["binding"],
            "rules": rules,
            "source": row["source"],
        }

    def active_snapshots(self, *, at: float | None = None) -> tuple[list[Params], float | None]:
        """读取各配置引用当前生效的价格及下一次生效时间，供月度展示缓存使用。

        只在价格失效或定时生效边界读取价库，不按历史请求逐条查价。
        返回规则快照，不修改任何请求当时冻结的价格。
        """
        now = time.time() if at is None else at
        with self.db.transaction() as db:
            rows = db.execute(
                "SELECT * FROM (SELECT *, ROW_NUMBER() OVER ("
                "PARTITION BY binding ORDER BY effective DESC,rowid DESC) AS rank "
                "FROM plans WHERE effective<=?) WHERE rank=1",
                (now,),
            ).fetchall()
            next_change = db.execute(
                "SELECT MIN(effective) FROM plans WHERE effective>?", (now,)
            ).fetchone()[0]

        snapshots = []
        for row in rows:
            rules = json.loads(row["rules"])
            self._validate(rules)
            snapshots.append({
                "version": row["id"],
                "ref": row["binding"],
                "rules": rules,
                "source": row["source"],
            })

        return snapshots, next_change

    @staticmethod
    def calculate(snapshot: Params | None, usage: Params) -> Params:
        """纯计算冻结价格对应的费用；缺失价格保留未知，支持整次/逐段阶梯。"""
        unknown = {"amount": None, "currency": None, "version": None}
        if snapshot is None:
            return unknown
        rules = snapshot["rules"]
        result = {
            **unknown,
            "currency": rules["currency"],
            "version": snapshot["version"],
        }
        if all(
            Decimal(str(rate)) == 0
            for tier in rules["tiers"]
            for rate in tier["rates"].values()
        ):
            return {**result, "amount": Decimal(0)}
        values = plain(usage)
        if values.get("input") is not None and values.get("cached") is not None:
            values["input_uncached"] = max(0, values["input"] - values["cached"])
        basis = values.get(rules.get("threshold", "input"))
        if basis is None:
            return result
        total = Decimal(0)
        lower = 0
        for tier in rules["tiers"]:
            upper = tier.get("up_to")
            if rules.get("tier_mode", "whole") == "whole":
                if upper is not None and basis > upper:
                    continue
                quantities = values
            else:
                quantities = {
                    rules.get("threshold", "input"): max(
                        0, min(basis, upper or basis) - lower
                    )
                }
            for meter, rate in tier["rates"].items():
                value = quantities.get(meter)
                if value is None:
                    return result
                total += (
                    Decimal(str(value))
                    * Decimal(str(rate))
                    / Decimal(str(rules.get("units", {}).get(meter, 1)))
                )
            if (
                rules.get("tier_mode", "whole") == "whole"
                or upper is None
                or basis <= upper
            ):
                break
            lower = upper
        return {**result, "amount": total}
