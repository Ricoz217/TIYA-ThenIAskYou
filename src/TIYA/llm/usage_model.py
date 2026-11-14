"""业务用量快照；保存来源和请求关联，不承担持久记账或费用计算。"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from TIYA.time_id import next_time_id

from .config_model import Params

type UsageValues = dict[str, int | float | None]
type UsageSource = Literal["reported", "estimated", "unknown", "derived", "mixed"]

_ESTIMATE_METADATA = {"algorithm", "completeness", "assumptions"}
_TOKEN_FIELDS = {"input", "output", "cached", "total"}


class UsageSampleData(TypedDict):
    model: str
    request_id: str
    call_id: str
    reported: Params
    estimated: Params
    unknown: Params


class TokenUsageData(TypedDict, total=False):
    """旧存档可缺少新字段；to_dict() 总是写出完整 v2 格式。"""

    schema_version: int
    input_t: int
    output_t: int
    cache_hit_t: int
    total_t: int
    model: str
    id: str
    samples: list[UsageSampleData]


class RequestUsageRecord(TypedDict):
    """账本查询返回的一次尝试，TokenUsage 从中取得独立证据与关联 ID。"""

    id: str
    call_id: str
    created: float
    identity: Params
    estimated: Params
    reported: Params
    status: str
    remote_state: str
    diagnostic: Params
    price_snapshot: Params | None
    charge: Params | None
    effective: Params


def _measurements(values: Mapping[str, object]) -> UsageValues:
    result: UsageValues = {}
    for key, value in values.items():
        if key in _ESTIMATE_METADATA:
            continue

        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"用量 {key} 必须是数值或 None")

            if key in _TOKEN_FIELDS and not isinstance(value, int):
                raise TypeError(f"token 用量 {key} 必须是整数")

            if value < 0 or isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"用量 {key} 必须是有限非负数")

        result[key] = value

    return result


def _sum_values(groups: Iterable[Mapping[str, object]]) -> UsageValues:
    result: UsageValues = {}
    for group in groups:
        for key, value in _measurements(group).items():
            if value is None:
                result.setdefault(key, None)

            else:
                result[key] = (result.get(key) or 0) + value

    return result


def _copy_evidence(values: Params) -> Params:
    if not isinstance(values, Mapping):
        raise TypeError("用量证据必须是字典")

    _measurements(values)
    return copy.deepcopy(dict(values))


def select_usage_values(
    reported: Params,
    estimated: Params,
    unknown: Params | None = None,
) -> UsageValues:
    """按字段选取真实值、估算值、未知来源值；零有效，不补造缺失值或 total。"""
    groups = [
        _measurements(reported),
        _measurements(estimated),
        _measurements(unknown or {}),
    ]
    keys = set().union(*groups)
    result: UsageValues = {}
    for key in sorted(keys):
        result[key] = None
        for group in groups:
            value = group.get(key)
            if value is not None:
                result[key] = value
                break

    return result


@dataclass(slots=True)
class UsageSample:
    """一次请求的用量证据；unknown 专门保留旧数据中未注明来源的计量值。

    reported/estimated 保留账本的独立证据，包括估算算法和假设说明。
    request_id/call_id 来自账本；为空表示无法追溯，不补造历史请求 ID。
    """

    model: str = ""
    request_id: str = ""
    call_id: str = ""
    reported: Params = field(default_factory=dict)
    estimated: Params = field(default_factory=dict)
    unknown: Params = field(default_factory=dict)

    def __post_init__(self):
        for identity in (self.model, self.request_id, self.call_id):
            if not isinstance(identity, str):
                raise TypeError("模型和账本 ID 必须是字符串")

        self.reported = _copy_evidence(self.reported)
        self.estimated = _copy_evidence(self.estimated)
        self.unknown = _copy_evidence(self.unknown)

    @property
    def effective(self) -> UsageValues:
        result = select_usage_values(self.reported, self.estimated, self.unknown)
        # total 没有独立返回值时，从已有 input/output 推导；不推导缓存命中量。
        if result.get("total") is None and any(
            result.get(key) is not None for key in ("input", "output")
        ):
            result["total"] = (result.get("input") or 0) + (result.get("output") or 0)

        return result

    def source(self, key: str) -> UsageSource | None:
        if self.reported.get(key) is not None:
            return "reported"

        if self.estimated.get(key) is not None:
            return "estimated"

        if self.unknown.get(key) is not None:
            return "unknown"

        if key == "total" and self.effective.get(key) is not None:
            return "derived"

        return None

    def complete(self, key: str) -> bool:
        if key == "total" and self.source(key) == "derived":
            values = self.effective
            return all(values.get(field) is not None for field in ("input", "output"))

        return self.effective.get(key) is not None

    def to_dict(self) -> UsageSampleData:
        return {
            "model": self.model,
            "request_id": self.request_id,
            "call_id": self.call_id,
            "reported": copy.deepcopy(self.reported),
            "estimated": copy.deepcopy(self.estimated),
            "unknown": copy.deepcopy(self.unknown),
        }


@dataclass(slots=True, init=False)
class TokenUsage:
    """一轮响应或一组已消费响应的用量快照，保留每次请求的来源。

    reported/estimated/unknown 分别汇总对应证据，effective 先在每个 sample 内
    选值再累计。不能拿汇总后的 reported 覆盖整组 estimated，否则会漏掉其他
    请求的估算消耗。incomplete_fields 标明仅有部分计量值，缓存缺失不冒充真实零。

    input_t/output_t/cache_hit_t/total_t 保留旧业务读取，缺失时这些属性返回零；
    model 是最后合入的模型，models 列出全部模型。id 只标识快照，账本身份在
    request_ids/call_ids。费用、失败尝试和完整长期统计以 SQLite 账本为准。
    """

    model: str
    id: str
    samples: list[UsageSample]

    def __init__(
        self,
        input_t: int | None = None,
        output_t: int | None = None,
        cache_hit_t: int | None = None,
        total_t: int | None = None,
        model: str = "",
        id: str | None = None,
        *,
        reported: Params | None = None,
        estimated: Params | None = None,
        unknown: Params | None = None,
        request_id: str = "",
        call_id: str = "",
    ) -> None:
        """旧数字参数按未知来源读取；新调用显式提供 reported/estimated 和账本 ID。"""
        if not isinstance(model, str) or id is not None and not isinstance(id, str):
            raise TypeError("模型和快照 ID 必须是字符串")

        self.model = model
        self.id = str(next_time_id()) if id is None else id
        self.samples = []
        unclassified = _copy_evidence(unknown) if unknown is not None else {}
        for key, value in (
            ("input", input_t),
            ("output", output_t),
            ("cached", cache_hit_t),
            ("total", total_t),
        ):
            if value is not None:
                unclassified[key] = value

        sample = UsageSample(
            model=model,
            request_id=request_id,
            call_id=call_id,
            reported=reported if reported is not None else {},
            estimated=estimated if estimated is not None else {},
            unknown=unclassified,
        )
        if sample.reported or sample.estimated or sample.unknown or request_id or call_id:
            self.samples.append(sample)

    @property
    def reported(self) -> UsageValues:
        return _sum_values(sample.reported for sample in self.samples)

    @property
    def estimated(self) -> UsageValues:
        return _sum_values(sample.estimated for sample in self.samples)

    @property
    def unknown(self) -> UsageValues:
        return _sum_values(sample.unknown for sample in self.samples)

    @property
    def effective(self) -> UsageValues:
        return _sum_values(sample.effective for sample in self.samples)

    @property
    def sources(self) -> dict[str, UsageSource]:
        result: dict[str, UsageSource] = {}
        for key in self.effective:
            sources: set[UsageSource] = set()
            for sample in self.samples:
                source = sample.source(key)
                if source is not None:
                    sources.add(source)

            if sources:
                result[key] = next(iter(sources)) if len(sources) == 1 else "mixed"

        return result

    @property
    def incomplete_fields(self) -> tuple[str, ...]:
        keys = self.effective.keys() | _TOKEN_FIELDS
        return tuple(sorted(
            key for key in keys
            if any(not sample.complete(key) for sample in self.samples)
        ))

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            sample.request_id for sample in self.samples if sample.request_id
        ))

    @property
    def call_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            sample.call_id for sample in self.samples if sample.call_id
        ))

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(sample.model for sample in self.samples if sample.model))

    @property
    def input_t(self) -> int:
        return int(self.effective.get("input") or 0)

    @property
    def output_t(self) -> int:
        return int(self.effective.get("output") or 0)

    @property
    def cache_hit_t(self) -> int:
        return int(self.effective.get("cached") or 0)

    @property
    def total_t(self) -> int:
        return int(self.effective.get("total") or 0)

    def to_dict(self) -> TokenUsageData:
        """导出 v2 快照；旧扁平字段保留供读取，完整证据以 samples 为准。"""
        return {
            "schema_version": 2,
            "input_t": self.input_t,
            "output_t": self.output_t,
            "cache_hit_t": self.cache_hit_t,
            "total_t": self.total_t,
            "model": self.model,
            "id": self.id,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, data: TokenUsageData | dict) -> TokenUsage:
        """读取新快照或旧 Context 的扁平用量；旧来源未知，不伪造真实值和请求 ID。"""
        version = data.get("schema_version")
        if isinstance(version, bool):
            raise ValueError("不支持的用量快照版本")

        if version is None or version == 1:
            return cls(
                input_t=data.get("input_t"),
                output_t=data.get("output_t"),
                cache_hit_t=data.get("cache_hit_t"),
                total_t=data.get("total_t"),
                model=data.get("model", ""),
                id=data.get("id"),
            )

        if version != 2:
            raise ValueError("不支持的用量快照版本")

        samples = data.get("samples")
        if not isinstance(samples, list) or any(
            not isinstance(sample, dict) for sample in samples
        ):
            raise ValueError("用量快照 samples 必须是字典列表")

        result = cls(model=data.get("model", ""), id=data.get("id"))
        result.samples = [
            UsageSample(
                model=sample.get("model", ""),
                request_id=sample.get("request_id", ""),
                call_id=sample.get("call_id", ""),
                reported=sample.get("reported", {}),
                estimated=sample.get("estimated", {}),
                unknown=sample.get("unknown", {}),
            )
            for sample in samples
        ]
        return result

    def clear(self) -> None:
        self.samples.clear()
        self.model = ""

    def copy(self) -> TokenUsage:
        return self.__copy__()

    def __copy__(self) -> TokenUsage:
        """复制外层样本列表，保留内部样本引用；快照 ID 新建，账本关联不变。"""
        result = object.__new__(type(self))
        result.model = self.model
        result.id = str(next_time_id())
        result.samples = self.samples.copy()
        return result

    def __deepcopy__(self, memo: dict[int, object]) -> TokenUsage:
        result = object.__new__(type(self))
        memo[id(self)] = result
        result.model = self.model
        result.id = self.id
        result.samples = copy.deepcopy(self.samples, memo)
        return result

    @classmethod
    def _from_samples(cls, samples: list[UsageSample], model: str = "") -> TokenUsage:
        result = cls(model=model)
        result.samples = copy.deepcopy(samples)
        return result

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """合并不同请求；同 request_id 的右侧快照更新旧值，不重复累计。"""
        if not isinstance(other, TokenUsage):
            raise TypeError("只能合并 TokenUsage")

        combined = copy.deepcopy(self.samples)
        positions = {
            sample.request_id: index
            for index, sample in enumerate(combined)
            if sample.request_id
        }
        for sample in other.samples:
            if sample.request_id and sample.request_id in positions:
                combined[positions[sample.request_id]] = copy.deepcopy(sample)

            else:
                if sample.request_id:
                    positions[sample.request_id] = len(combined)

                combined.append(copy.deepcopy(sample))

        return self._from_samples(combined, other.model or self.model)

    def __sub__(self, other: TokenUsage) -> TokenUsage:
        """移除已知请求子集时保留证据；普通数值差只作为未知来源的比较值。"""
        if not isinstance(other, TokenUsage):
            raise TypeError("只能减去 TokenUsage")

        if not other.samples:
            return self._from_samples(self.samples, self.model)

        removed = set(other.request_ids)
        if all(sample.request_id for sample in other.samples) and removed <= set(self.request_ids):
            remaining = [sample for sample in self.samples if sample.request_id not in removed]
            return self._from_samples(remaining, remaining[-1].model if remaining else self.model)

        left = self.effective
        right = other.effective
        difference: Params = {}
        for key in left.keys() | right.keys():
            a = left.get(key)
            b = right.get(key)
            difference[key] = max(0, a - b) if a is not None and b is not None else None

        return TokenUsage(model=self.model, unknown=difference)

    def __gt__(self, other: TokenUsage) -> bool:
        if not isinstance(other, TokenUsage):
            return NotImplemented

        return self.total_t > other.total_t

    def __lt__(self, other: TokenUsage) -> bool:
        if not isinstance(other, TokenUsage):
            return NotImplemented

        return self.total_t < other.total_t

    def __bool__(self) -> bool:
        return any(value is not None and value > 0 for value in self.effective.values())
