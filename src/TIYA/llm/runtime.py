"""通信调度：准备快照 → 选择候选 → 有界尝试 → 返回待提交结果。

execute 不修改业务 Context。只有 commit/commit_async 校验快照仍有效后才提交；
取消、部分输出或远端执行不明一律停止重发。账本记录每次实际发送，检查点只在
业务提交后持久化；fallback 冷却在安全失败后的候选切换时更新。
"""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, TypedDict, cast

from TIYA.utils import fingerprint

from .config_model import ChatConfig, JsonValue, LLMConfig, Params
from .gateway import Frame, Gateway, HTTPFailure, WireRequest
from .media import MediaError, MediaService, UploadSender
from .model import (
    BasePrompt,
    CallOutcome,
    Context,
    ContextOverflowError,
    ImagePrompt,
    MediaRef,
    Prompts,
    TokenUsage,
)
from .provider import get_provider
from .provider.base import (
    BackgroundStreamDecoder,
    ConversationProvider,
    Failure,
    ProviderFailure,
    Result,
    StreamDecoder,
)
from .provider.common import normalize
from .usage import UsageEstimate, estimate_async
from .workers import run_sync

if TYPE_CHECKING:
    from TIYA.logger import BlockHandle, Logger

    from .pricing import PriceCatalog
    from .state import StateStore
    from .usage import Ledger
    from .usage_compat import LLMUsage

type GatewayFactory = Callable[[LLMConfig], Awaitable[Gateway]]
type SendFile = Callable[[WireRequest], Awaitable[Params]]
type PreparedMedia = dict[str, list[MediaRef]]


class Checkpoint(TypedDict):
    """已提交 response 与当前原始历史前缀的绑定，不只是一个父 ID。"""

    response_id: str
    header: str  # system/tools 的指纹；不同头部禁止续接。
    prefix: list[str]  # 顺序敏感的消息指纹，保留缓存前缀匹配。


@dataclass(slots=True)
class Pending:
    """执行成功但尚未提交的结果；由 Context 所有者决定是否消费。"""

    result: Result
    planned: Context  # 本轮请求使用的独立上下文，包含已准备媒体。
    expected: str  # 提交前必须与业务 Context 指纹一致。
    session: str
    config: ChatConfig
    route: LLMConfig  # 真正成功的候选，不一定是首选。
    route_index: int
    call_id: str
    checkpoint: Checkpoint | None
    fallback_reason: str = ""


@dataclass(slots=True)
class _Call:
    """一次逻辑调用的固定输入和跨候选共享的预算、诊断。"""

    session_id: str
    expected: str
    planned: Context
    config: ChatConfig
    deadline: float  # monotonic 总截止时间，重试不重新计时。
    call_id: str
    route_key: str
    attempts: int = 0
    last_error: BaseException | None = None
    reason: str = ""
    last_request_id: str = ""
    last_remote_state: str = "not_sent"
    last_category: str = "preflight"


@dataclass(slots=True)
class _Candidate:
    """单候选的续接和图片恢复进度；切换候选时不继承这些状态。"""

    route: LLMConfig
    index: int
    provider: ConversationProvider
    gateway: Gateway
    estimation: UsageEstimate
    checkpoint_key: str
    checkpoints: list[Checkpoint]
    header: str
    parent_enabled: bool
    parent: Checkpoint | None
    inline: bool = False
    force: set[str] = field(default_factory=set)
    recovered: set[str] = field(default_factory=set)
    retries: int = 0
    media: PreparedMedia = field(default_factory=dict)


@dataclass(slots=True)
class _Attempt:
    """一次实际生成请求的收发状态，供失败时保留 usage 和停止依据。"""

    request_id: str
    payload: Params
    decoder: StreamDecoder | None = None
    partial: bool = False
    result: Result | None = None
    raw_body: Params | None = None
    background_id: str = ""
    platform_request_id: str = ""


class Runtime:
    """把一轮业务输入转换为待提交的对话结果，并编排恢复与记账。

    state 保存上传/续接/冷却状态，ledger 记录每次真实尝试；prices 和 subscribers
    分别提供可选定价与附加账本。execute() 返回 Pending，由业务持有者决定何时
    commit_async()。业务 Context 和 HTTP 连接的生命周期仍由调用方管理。
    """

    def __init__(
        self,
        state: StateStore,
        ledger: Ledger,
        prices: PriceCatalog | None = None,
        logger: Logger | BlockHandle | None = None,
        subscribers: list[LLMUsage | Ledger] | None = None,
    ) -> None:
        self.state = state
        self.ledger = ledger
        self.prices = prices
        self.logger = logger
        self.media = MediaService(state, on_warning=self._log)
        self.subscribers = subscribers if subscribers is not None else []

    async def _db(self, method: Callable, *args, **kwargs):
        """操作数据库"""
        for attempt in range(3):
            try:
                return await run_sync(method, *args, **kwargs)

            except sqlite3.OperationalError:
                if attempt == 2:
                    raise

                await asyncio.sleep(0.05 * (attempt + 1))

        raise sqlite3.OperationalError

    def _state_get(self, namespace: str, key: str) -> JsonValue:
        try:
            return self.state.get(namespace, key)

        except (sqlite3.Error, OSError) as exc:
            self._log(f"LLM 状态读取失败，将使用完整上下文: {type(exc).__name__}")
            return None

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.warning(message)

    def _state_put(
        self,
        namespace: str,
        key: str,
        value: JsonValue,
        expires: float | None = None,
    ) -> None:
        try:
            self.state.put(namespace, key, value, expires)

        except (sqlite3.Error, OSError) as exc:
            self._log(f"LLM 状态写入失败，当前调用保守恢复: {type(exc).__name__}")

    async def execute(
        self,
        context: Context,
        prompts: Prompts | BasePrompt,
        config: ChatConfig,
        gateway_for_route: GatewayFactory,
        timeout: float,
    ) -> Pending:
        """发送一轮请求，返回待提交结果；失败抛异常并附带 llm_outcome。

        context 是业务持有的原始历史；这里只读并制作副本。prompts 是本轮新增
        输入。config 必须含有序候选和重试策略；gateway_for_route 提供对应连接，
        连接生命周期由调用方管理。timeout 是整次调用的秒数，包含 Files、重试和
        后台轮询。成功也不会提交 Context，调用方须再调用 commit_async。

        主流程只负责候选顺序和全局终态；单候选尝试、记账、恢复分别在下游方法。
        取消/部分输出/远端未知直接抛出，禁止切换候选生成第二份结果。
        """
        if not config.candidates or config.request_policy is None:
            raise ValueError("Runtime 只接受完整 ChatConfig")

        self.media.degraded = False
        expected = context.signature
        planned = copy.deepcopy(context)
        planned.append(copy.deepcopy(prompts))
        planned = normalize(planned)
        deadline = time.monotonic() + timeout
        call_id = await self._db(self.ledger.begin_call, config.preset_name)
        call = _Call(
            session_id=context.session_id,
            expected=expected,
            planned=planned,
            config=config,
            deadline=deadline,
            call_id=call_id,
            route_key=fingerprint([context.session_id, config.revision]),
        )
        cooldown = await run_sync(self._state_get, "cooldown", call.route_key)
        selected = 0
        if cooldown and cooldown["until"] > time.time():
            selected = cooldown.get("index", 0)

        if selected >= len(config.candidates):
            selected = 0

        try:
            for index in range(selected, len(config.candidates)):
                if call.attempts >= config.request_policy.max_attempts:
                    break

                candidate = await self._prepare_candidate(
                    call, index, gateway_for_route
                )
                pending = await self._execute_candidate(call, candidate)
                if pending is not None:
                    return pending

                # 当前候选已安全失败：记录下一候选；末项失败清除 CD，本次就此结束。
                # 下次任务才从首选开始，本次不回绕已尝试的候选。
                await run_sync(self._advance_route, call, index)

            if call.last_error:
                raise call.last_error

            raise RuntimeError("LLM 请求预算耗尽")

        except BaseException as exc:
            exc.llm_outcome = CallOutcome(
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                False,
                call.call_id,
                call.last_request_id,
                call.last_remote_state,
                call.last_category,
            )
            await self._db(self.ledger.finish_call, call.call_id, "stopped")
            raise

        finally:
            await self._mirror_subscribers(call.call_id)

    def _advance_route(self, call: _Call, current_index: int) -> None:
        """安全失败后同步更新下一候选的 CD；不在本次任务内回绕。

        cooldown 的 value 为 {index: 候选下标, until: Unix 到期秒}。
        推进到备用时刷新到期时间；末项失败删除记录，下次任务才从下标 0 开始。
        成功请求不延长 CD。此处包含 SQLite IO，由调用方在线程池执行。
        """
        candidates = call.config.candidates
        assert candidates is not None
        policy = call.config.request_policy
        assert policy is not None
        next_index = current_index + 1
        if next_index == len(candidates):
            next_index = 0

        try:
            if next_index == 0:
                self.state.delete("cooldown", call.route_key)

            else:
                until = time.time() + policy.fallback_cooldown_seconds
                self.state.put(
                    "cooldown",
                    call.route_key,
                    {"index": next_index, "until": until},
                    until,
                )

        except (sqlite3.Error, OSError) as exc:
            self._log(f"LLM 冷却状态更新失败: {type(exc).__name__}")

        if next_index == 0:
            self._log(
                f"LLM 末项候选失败，删除 fallback 冷却；本次失败结束，下次从首选 {candidates[0].ref} 开始; "
                f"原因={call.reason}; call={call.call_id}; request={call.last_request_id}"
            )

        else:
            self._log(
                f"LLM 已进入 fallback: {candidates[current_index].ref} -> {candidates[next_index].ref}; "
                f"原因={call.reason}; call={call.call_id}; request={call.last_request_id}; 冷却至={until}"
            )

    async def _prepare_candidate(
        self,
        call: _Call,
        index: int,
        gateway_for_route: GatewayFactory,
    ) -> _Candidate:
        """检查当前候选窗口/能力并选择检查点；窗口不足直接抛出，不自动换候选。"""
        route = call.config.candidates[index]
        provider = get_provider(route)
        if not callable(getattr(provider, "encode", None)):
            raise ValueError(
                f"{route.provider_type} 不是对话操作，请使用对应的独立客户端"
            )

        parent_enabled = provider.supports_parent and not await run_sync(
            self._state_get,
            "unsupported_parent",
            route.scope,
        )
        if route.model_params.get("background") and not route.provider_options.get(
            "responses", {}
        ).get("supports_background"):
            raise ValueError(
                "后台 Responses 需要显式声明 responses.supports_background"
            )

        estimation = await estimate_async(call.planned)
        if route.max_context and estimation["input"] > route.max_context:
            call.last_category = "context_overflow"
            raise ContextOverflowError(
                f"候选 {route.ref} 的上下文估算下界 {estimation['input']} 超过其窗口 {route.max_context}"
            )

        gateway = await gateway_for_route(route)
        checkpoint_key = fingerprint([call.session_id, route.scope])
        checkpoints = (
            await run_sync(self._state_get, "checkpoints", checkpoint_key) or []
        )
        current_prefix = list(call.planned.message_fingerprints)
        header = call.planned.header_signature
        matching = [
            c
            for c in checkpoints
            if c["header"] == header
            and current_prefix[: len(c["prefix"])] == c["prefix"]
        ]
        parent = (
            max(matching, key=lambda c: len(c["prefix"]), default=None)
            if (
                route.provider_type == "responses_api"
                and route.context_history_mode != "full"
                and parent_enabled
            )
            else None
        )
        return _Candidate(
            route,
            index,
            provider,
            gateway,
            estimation,
            checkpoint_key,
            checkpoints,
            header,
            parent_enabled,
            parent,
        )

    async def _execute_candidate(
        self, call: _Call, candidate: _Candidate
    ) -> Pending | None:
        """在单候选预算内尝试；None 表示安全切换，危险终态直接抛出。"""
        while call.attempts < call.config.request_policy.max_attempts:
            remaining = call.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("LLM 逻辑请求总时限已到")

            # 续接只发送后缀；父 ID 无效时 _recover 会移除 parent，重新完整回放。
            send_context = call.planned
            if candidate.parent:
                suffix = call.planned.message[len(candidate.parent["prefix"]):]
                send_context = Context(suffix, call.planned.tools)
                # 历史前缀包含 system；切掉前缀后仍需构造本轮 instructions。
                system = call.planned.system
                if system is not None:
                    send_context.append(system)

            # 固定发起者和客户端；不能捕获下一次循环可能变化的 candidate/call。
            send_file = UploadSender(
                partial(
                    self._send_shared_file,
                    call_id=call.call_id,
                    route=candidate.route,
                    gateway=candidate.gateway,
                ),
                candidate.gateway,
            )

            try:
                media_changed = await self._prepare_media(
                    send_context,
                    candidate.provider,
                    send_file,
                    candidate.media,
                    candidate.inline,
                    candidate.force,
                )
                if media_changed:
                    call.planned.refresh_fingerprints(content_changed=False)
                    if send_context is not call.planned:
                        send_context.refresh_fingerprints(content_changed=False)

            except (MediaError, OSError, ProviderFailure) as exc:
                call.attempts += 1
                call.last_error = exc
                call.last_category = "media_preflight"
                call.reason = f"media_preflight/{type(exc).__name__}"
                if isinstance(exc, ProviderFailure):
                    failure = exc.failure
                    call.last_category = failure.category
                    if failure.safe and await self._retry_candidate(call, candidate, failure):
                        continue

                    if not failure.fallback_allowed:
                        raise

                return None

            attempt = await self._begin_attempt(call, candidate, send_context)
            try:
                result = await self._send_attempt(call, candidate, attempt, remaining)
                return await self._complete_attempt(call, candidate, attempt, result)

            except BaseException as exc:
                failure = await self._record_failure(call, candidate, attempt, exc)
                if (
                    isinstance(exc, asyncio.CancelledError)
                    or attempt.partial
                    or call.last_remote_state == "completed"
                    or not failure.safe
                ):
                    raise

                call.last_error = exc
                call.reason = (
                    f"{failure.category}/{failure.code} request={attempt.request_id}"
                )
                if await self._recover(call, candidate, failure):
                    if call.attempts < call.config.request_policy.max_attempts:
                        continue

                if failure.fallback_allowed:
                    return None

                raise

        return None

    async def _begin_attempt(
        self,
        call: _Call,
        candidate: _Candidate,
        send_context: Context,
    ) -> _Attempt:
        """编码请求并先落账本估算；每次真正生成都取得独立 request_id。"""
        payload = candidate.provider.encode(send_context, candidate.media)
        if candidate.parent:
            payload["previous_response_id"] = candidate.parent["response_id"]
            payload["store"] = True

        elif (
            candidate.route.provider_type == "responses_api"
            and candidate.parent_enabled
            and candidate.route.context_history_mode != "full"
        ):
            payload["store"] = True

        if any(
            key in candidate.route.model_params
            for key in (
                "messages",
                "input",
                "previous_response_id",
                "tools",
            )
        ):
            raise ValueError("llm_params 不能覆盖 Runtime 管理的上下文或工具字段")

        identity = {
            "preset": call.config.preset_name,
            "provider": candidate.route.ref,
            "scope": candidate.route.scope,
            "model": candidate.route.model,
            "operation": candidate.route.provider_type,
            "config_revision": call.config.revision,
        }
        price = None
        if self.prices:
            try:
                price = await self._db(
                    self.prices.snapshot,
                    {k: identity[k] for k in ("preset", "scope", "model", "operation")},
                )

            except (sqlite3.Error, OSError, ValueError):
                self._log("LLM 定价目录不可读，本次费用保留未知")

        if price is None and candidate.route.legacy_price is not None:
            legacy = candidate.route.legacy_price.to_dict()
            price = {
                "version": "legacy",
                "rules": {
                    "currency": legacy.get("currency", "CNY"),
                    "tiers": [
                        {
                            "rates": {
                                "input_uncached": str(legacy.get("input", 0)),
                                "cached": str(legacy.get("cached", 0)),
                                "output": str(legacy.get("output", 0)),
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

        request_id = await self._db(
            self.ledger.begin_request,
            call.call_id,
            identity,
            candidate.estimation,
            price_snapshot=price,
        )
        call.last_request_id = request_id
        call.last_remote_state = "unknown"
        call.attempts += 1
        candidate.gateway.last_request_id = ""
        return _Attempt(request_id, payload)

    async def _send_attempt(
        self,
        call: _Call,
        candidate: _Candidate,
        attempt: _Attempt,
        remaining: float,
    ) -> Result:
        """一次生成收发完整成功后返回 Result，不提交 Context。

        remaining 是发送前的剩余秒数，call.deadline 继续约束后台轮询总时间。
        attempt 保留已观察到的输出、usage 和任务 ID，异常时供记账和恢复判断；
        其中 result 允许为空，但成功调用方直接取得已确认完成的 Result。
        """
        wire = WireRequest(
            candidate.route.endpoint,
            headers=candidate.provider.headers(),
            json=attempt.payload,
        )
        async with asyncio.timeout(max(0.001, call.deadline - time.monotonic())):
            if attempt.payload.get("stream"):
                attempt.decoder = candidate.provider.stream_decoder()
                async for frame in candidate.gateway.stream(wire, remaining):
                    attempt.platform_request_id = candidate.gateway.last_request_id
                    if isinstance(frame, Frame):
                        if not frame.data.strip():
                            continue

                        if frame.data == "[DONE]":
                            attempt.decoder.complete = True
                            continue

                        body = json.loads(frame.data)

                    else:
                        body = frame

                    attempt.raw_body = body
                    attempt.decoder.feed(body)

                result = attempt.decoder.finish()
                attempt.result = result

            else:
                body = await candidate.gateway.send(wire, remaining)
                attempt.platform_request_id = candidate.gateway.last_request_id
                attempt.raw_body = body
                attempt.background_id = (
                    body.get("id", "") if attempt.payload.get("background") else ""
                )
                if attempt.background_id:
                    while body.get("status") in (
                        "queued",
                        "in_progress",
                    ):
                        body = await self._send_auxiliary_request(
                            call.call_id,
                            candidate.route,
                            candidate.provider.background_operation(
                                attempt.background_id
                            ),
                            candidate.gateway,
                            call.deadline,
                            operation="response_retrieve",
                        )
                        attempt.raw_body = body
                        if body.get("status") in (
                            "queued",
                            "in_progress",
                        ):
                            await asyncio.sleep(0.25)

                result = candidate.provider.decode(body)
                attempt.result = result

            if attempt.decoder and attempt.payload.get("background"):
                attempt.background_id = cast(
                    BackgroundStreamDecoder, attempt.decoder
                ).body.get("id", "")

        attempt.partial = (
            attempt.decoder.partial if attempt.decoder else bool(result.prompts)
        )
        if not result.complete:
            raise RuntimeError("LLM 响应未完成，停止重发")

        return result

    async def _complete_attempt(
        self,
        call: _Call,
        candidate: _Candidate,
        attempt: _Attempt,
        result: Result,
    ) -> Pending:
        """记录用量、缓存返回图片并构造待提交结果；落盘失败也禁止重新生成。"""
        result.request_id = attempt.request_id
        if result.stored is False and candidate.parent_enabled:
            await run_sync(
                self._state_put,
                "unsupported_parent",
                candidate.route.scope,
                {"evidence": "store_false"},
                time.time() + 3600,
            )
            self._log(
                f"LLM 端点未保存 response，后续使用完整历史: {candidate.route.ref}"
            )

        output_estimate = (await estimate_async(Context(result.prompts)))[
            "input"
        ]
        await self._db(
            self.ledger.observe_usage,
            attempt.request_id,
            "estimated",
            {"output": output_estimate},
            "local_output",
        )
        await self._db(
            self.ledger.observe_usage,
            attempt.request_id,
            "reported",
            result.usage,
            "final",
            raw=result.raw_usage,
        )
        for image_prompt, image_wire in result.images:
            cached = await self.media.materialize(image_wire, candidate.gateway)
            image_prompt.resource = cached.image
            image_prompt.hash_name = cached.image.hash_name
            image_prompt.image_input = None

        result.images.clear()
        await self._db(
            self.ledger.finish_request,
            attempt.request_id,
            "completed",
            remote_state="completed",
            response_id=result.response_id,
            platform_request_id=attempt.platform_request_id,
            reported_model=result.reported_model,
        )
        usage_row = await self._db(self.ledger.get_request, attempt.request_id)
        result.prompts.usage = TokenUsage(
            model=candidate.route.model,
            request_id=usage_row["id"],
            call_id=usage_row["call_id"],
            reported=usage_row["reported"],
            estimated=usage_row["estimated"],
        )
        checkpoint = (
            {
                "response_id": result.response_id,
                "header": candidate.header,
                "prefix": list(
                    Context(call.planned.message, result.prompts).message_fingerprints
                ),
            }
            if (
                candidate.parent_enabled
                and result.stored is not False
                and result.response_id
                and candidate.route.context_history_mode != "full"
            )
            else None
        )
        await self._db(self.ledger.finish_call, call.call_id, "completed")
        return Pending(
            result,
            call.planned,
            call.expected,
            call.session_id,
            call.config,
            candidate.route,
            candidate.index,
            call.call_id,
            checkpoint,
            call.reason,
        )

    async def _record_failure(
        self,
        call: _Call,
        candidate: _Candidate,
        attempt: _Attempt,
        exc: BaseException,
    ) -> Failure:
        """保留失败证据，解释 Provider 建议并叠加实际执行状态，不改写原错误对象。"""
        observed_usage: Params = {}
        if attempt.decoder:
            attempt.partial = attempt.decoder.partial
            # Retain known usage even when tool JSON/final framing is damaged.
            try:
                usage_result = candidate.provider.usage_result(attempt.decoder.usage)
                observed_usage = usage_result.usage
                await self._db(
                    self.ledger.observe_usage,
                    attempt.request_id,
                    "reported",
                    usage_result.usage,
                    "partial",
                    raw=attempt.decoder.usage,
                )

            except (ValueError, KeyError):
                pass

        cancelled = isinstance(exc, asyncio.CancelledError)
        if attempt.decoder and attempt.payload.get("background"):
            attempt.background_id = cast(
                BackgroundStreamDecoder, attempt.decoder
            ).body.get("id", "")

        cancel_status = await self._cancel_background(call, candidate, attempt, exc)
        failure = candidate.provider.classify_failure(exc)

        result_complete = bool(attempt.result and attempt.result.complete)
        stream_complete = bool(attempt.decoder and attempt.decoder.complete)
        body = attempt.raw_body if isinstance(attempt.raw_body, dict) else {}
        body_complete = bool(
            body
            and (
                body.get("status") == "completed"
                or any("message" in c for c in body.get("choices", []))
            )
        )
        remote_complete = result_complete or stream_complete or body_complete
        has_observed_usage = any(
            isinstance(value, (int, float)) and value > 0 for value in observed_usage.values()
        )
        if remote_complete:
            failure = replace(failure, remote_state="completed")

        elif cancelled or attempt.partial or attempt.background_id or has_observed_usage:
            # 已创建后台任务、出现输出/消耗或本地取消，不能因错误建议重建生成。
            failure = replace(failure, remote_state="unknown")

        raw_error = body
        if isinstance(exc, HTTPFailure) and isinstance(exc.body, dict):
            raw_error = exc.body

        elif isinstance(exc, ProviderFailure) and exc.failure.raw:
            raw_error = exc.failure.raw

        if raw_error.get("usage"):
            known = candidate.provider.usage_result(raw_error["usage"])
            if not remote_complete and any(isinstance(v, (int, float)) and v > 0 for v in known.usage.values()):
                failure = replace(failure, remote_state="unknown")

            await self._db(
                self.ledger.observe_usage,
                attempt.request_id,
                "reported",
                known.usage,
                "error_usage",
                raw=known.raw_usage,
            )

        if failure.safe:
            await self._db(
                self.ledger.observe_usage,
                attempt.request_id,
                "estimated",
                {
                    "input": 0,
                    "output": 0,
                    "cached": 0,
                    "algorithm": "rejected-before-execution-v1",
                },
                "rejected",
            )

        status = (
            (
                "materialization_failed"
                if isinstance(exc, MediaError)
                else "local_failure"
            )
            if remote_complete
            else (
                "cancelled" if cancelled else "rejected" if failure.safe else "unknown"
            )
        )
        call.last_remote_state = cancel_status or (
            "completed" if remote_complete else failure.remote_state
        )
        call.last_category = failure.category
        await self._db(
            self.ledger.finish_request,
            attempt.request_id,
            status,
            remote_state=cancel_status or "completed"
            if remote_complete
            else cancel_status or failure.remote_state,
            category=failure.category,
            code=failure.code,
            partial=attempt.partial,
            platform_request_id=attempt.platform_request_id
            or candidate.gateway.last_request_id,
            http_status=exc.status
            if isinstance(exc, HTTPFailure)
            else 200
            if isinstance(exc, ProviderFailure)
            else None,
            resource=failure.resource,
            retryable=failure.retryable,
            fallback_allowed=failure.fallback_allowed,
            recovery=failure.recovery,
            resend_safe=failure.safe and not cancelled,
        )
        return failure

    async def _cancel_background(
        self,
        call: _Call,
        candidate: _Candidate,
        attempt: _Attempt,
        exc: BaseException,
    ) -> str | None:
        """仅对已知后台任务 ID 显式取消；3 秒内无法确认时保持远端状态未知。"""
        cancel_status = None
        if attempt.background_id and isinstance(
            exc, (asyncio.CancelledError, TimeoutError)
        ):
            try:
                cancelled_body = await asyncio.wait_for(
                    self._send_auxiliary_request(
                        call.call_id,
                        candidate.route,
                        candidate.provider.background_operation(
                            attempt.background_id, cancel=True
                        ),
                        candidate.gateway,
                        time.monotonic() + 3,
                        operation="response_cancel",
                    ),
                    3,
                )
                if cancelled_body.get("status") in (
                    "cancelled",
                    "completed",
                ):
                    cancel_status = cancelled_body["status"]

            except Exception:
                self._log(f"后台 Responses 取消未确认 request={attempt.request_id}")

        return cancel_status

    async def _recover(
        self,
        call: _Call,
        candidate: _Candidate,
        failure: Failure,
    ) -> bool:
        """执行 Provider 建议的有界恢复/重试，False 表示未恢复，不隐含允许切换。"""
        if failure.recovery == "context_overflow":
            raise ContextOverflowError(
                f"候选 {candidate.route.ref} 返回上下文溢出: {failure.code}"
            ) from call.last_error

        if (
            failure.recovery == "rebuild_parent"
            and candidate.parent
            and "parent" not in candidate.recovered
        ):
            candidate.recovered.add("parent")
            candidate.checkpoints = [
                c
                for c in candidate.checkpoints
                if c["response_id"] != candidate.parent["response_id"]
            ]
            await run_sync(
                self._state_put,
                "checkpoints",
                candidate.checkpoint_key,
                candidate.checkpoints,
            )
            candidate.parent = None
            self._log(
                f"LLM 续接失效，完整回放当前上下文 call={call.call_id} request={call.last_request_id}"
            )
            return True

        if (
            failure.recovery in ("refresh_file", "inline_images", "disable_files")
            and "files" not in candidate.recovered
        ):
            if failure.recovery == "refresh_file" and not failure.resource:
                # 无明确目标不失效任何文件；是否原样重试仍由 Provider 的独立许可决定。
                return await self._retry_candidate(call, candidate, failure)

            candidate.recovered.add("files")
            candidate.inline = failure.recovery in ("inline_images", "disable_files")
            candidate.force = {
                pid
                for pid, refs in candidate.media.items()
                if any(
                    ref.kind == "remote" and ref.value == failure.resource
                    for ref in refs
                )
            }
            if not candidate.inline and candidate.route.files is not None:
                from .provider.files import get_files_provider

                scope = get_files_provider(
                    candidate.route.files.as_provider_settings(),
                    candidate.route.files.provider_type,
                ).scope
                await self._db(
                    self.state.put,
                    "invalid_files",
                    fingerprint([scope, failure.resource]),
                    {"invalid": True},
                )
                if not candidate.force and candidate.parent:
                    candidate.parent = None
                    candidate.media = {}

            if failure.recovery == "disable_files":
                await self._db(
                    self.state.put,
                    "unsupported_files",
                    candidate.route.scope,
                    {"reason": failure.code},
                    time.time() + 3600,
                )

            return True

        return await self._retry_candidate(call, candidate, failure)

    async def _retry_candidate(
        self,
        call: _Call,
        candidate: _Candidate,
        failure: Failure,
    ) -> bool:
        """只执行 Provider 允许的原样重试；统一限制次数、等待和剩余时限。"""
        if call.attempts >= call.config.request_policy.max_attempts:
            return False

        if failure.retryable and candidate.retries < (
            candidate.route.max_retries
            if candidate.route.max_retries is not None
            else call.config.request_policy.max_retries_per_candidate
        ):
            candidate.retries += 1
            delay = min(
                failure.retry_after if failure.retry_after is not None else 0.2,
                max(0.0, call.deadline - time.monotonic()),
            )
            await asyncio.sleep(delay)
            return True

        return False

    async def _mirror_subscribers(self, call_id: str) -> None:
        """将主账本已持久化的尝试镜像到附加账本；不影响生成结果或重发。"""
        for subscriber in tuple(self.subscribers):
            try:
                target = await run_sync(
                    lambda: subscriber.ledger
                    if hasattr(type(subscriber), "ledger")
                    else subscriber
                )
                if target.db.path.resolve() == self.ledger.db.path.resolve():
                    continue

                offset = 0
                while True:
                    rows = await self._db(
                        self.ledger.query_requests,
                        call_id=call_id,
                        limit=1000,
                        offset=offset,
                    )
                    if not rows:
                        break

                    for row in rows:
                        await self._db(target.mirror_request, row)

                    offset += len(rows)

            except Exception as exc:
                self._log(
                    f"附加用量记录器写入失败，主账本已保留 call={call_id}: {type(exc).__name__}"
                )

    async def _prepare_media(
        self,
        send_context: Context,
        provider: ConversationProvider,
        send_file: SendFile,
        media: PreparedMedia,
        inline: bool,
        force: set[str],
    ) -> bool:
        """原地补充/修复 media，返回是否更新；已准备好的传输引用供重试复用。

        inline 只转换尚未内联的图片；force 中成功修复的消息随即移除，避免下一次
        普通重试再次读文件或编码。发送视图与工作上下文共享消息，由调用方刷新指纹。
        """
        changed = False
        for prompt in send_context:
            if not isinstance(prompt, ImagePrompt):
                continue

            existing = media.get(prompt.id)
            needs_prepare = existing is None
            if inline and existing is not None:
                needs_prepare = any(ref.kind != "base64" for ref in existing)

            if prompt.id in force:
                needs_prepare = True

            if not needs_prepare:
                continue

            media[prompt.id] = await self.media.prepare_prompt(
                prompt,
                provider,
                send_file,
                inline=inline,
            )
            force.discard(prompt.id)
            changed = True

        return changed

    async def _send_shared_file(
        self,
        wire: WireRequest,
        *,
        call_id: str,
        route: LLMConfig,
        gateway: Gateway,
    ) -> Params:
        """共享 Files 任务只记一次实际请求，归首次发起的 call_id。

        上传使用自身时限和独立 Gateway 状态，不借用首个等待者的剩余时间，
        也不与模型请求共用 last_request_id；HTTP 客户端仍复用原连接池。
        """
        return await self._send_auxiliary_request(
            call_id,
            route,
            wire,
            Gateway(gateway.client),
            time.monotonic() + self.media.upload_timeout,
        )

    async def _send_auxiliary_request(
        self,
        call_id: str,
        route: LLMConfig,
        wire: WireRequest,
        gateway: Gateway,
        deadline: float,
        operation: str = 'files',
    ) -> Params:
        """执行并单独记账 Files/后台控制操作，不执行生成重试。"""
        from .provider.files import get_files_provider

        file_provider = None
        if operation == "files":
            if route.files is None:
                raise ValueError("未配置 Files 服务")

            file_provider = get_files_provider(
                route.files.as_provider_settings(), route.files.provider_type
            )

        request_id = await self._db(
            self.ledger.begin_request,
            call_id,
            {
                "preset": "",
                "provider": route.files.ref if operation == "files" else route.ref,
                "model": "",
                "operation": operation,
                "token_usage": "not_applicable",
            },
            {"algorithm": "not_applicable"},
        )
        try:
            result = await gateway.send(wire, max(0.001, deadline - time.monotonic()))
            if file_provider:
                file_provider.decode(result)

            else:
                get_provider(route).check_body(result)

        except BaseException as exc:
            provider = file_provider if file_provider is not None else get_provider(route)
            failure = provider.classify_failure(exc)

            await self._db(
                self.ledger.finish_request,
                request_id,
                "rejected" if failure.safe else "unknown",
                remote_state=failure.remote_state,
                code=failure.code,
                http_status=exc.status if isinstance(exc, HTTPFailure) else None,
                platform_request_id=gateway.last_request_id,
                retryable=failure.retryable,
                fallback_allowed=failure.fallback_allowed,
                recovery=failure.recovery,
            )
            if not isinstance(exc, (ProviderFailure, asyncio.CancelledError)):
                raise ProviderFailure(failure) from exc

            raise

        await self._db(
            self.ledger.finish_request,
            request_id,
            "completed",
            remote_state="completed",
            platform_request_id=gateway.last_request_id,
        )
        return result

    def commit(self, context: Context, pending: Pending) -> Prompts:
        """同步提交；包含状态库 IO，仅供同步调用方使用。"""
        self._apply(context, pending)
        self._persist_commit(context.session_id, pending)
        return pending.result.prompts

    async def commit_async(
        self,
        context: Context,
        pending: Pending,
        on_commit: Callable[[], None] | None = None,
    ) -> Prompts:
        """校验并提交 Context，再持久化续接检查点；on_commit 在内存提交后同步调用。"""
        self._apply(context, pending)
        if on_commit:
            on_commit()

        await run_sync(self._persist_commit, context.session_id, pending)
        return pending.result.prompts

    @staticmethod
    def _apply(context: Context, pending: Pending) -> None:
        if (
            context.session_id != pending.session
            or context.signature != pending.expected
        ):
            raise RuntimeError("Context 在请求期间已修改，拒绝提交过期响应")

        # planned 已是本次请求独占的工作上下文，提交时无需再复制整份历史。
        updated = pending.planned
        updated.append(pending.result.prompts)
        context.replace_contents(updated)

    def _persist_commit(self, session_id: str, pending: Pending) -> None:
        """仅为已提交结果保存续接检查点；失败不回滚已消费结果。"""
        try:
            key = fingerprint([session_id, pending.route.scope])
            if pending.checkpoint:
                entries = self.state.get("checkpoints", key) or []
                entries.append(pending.checkpoint)
                self.state.put("checkpoints", key, entries[-64:])

        except (sqlite3.Error, OSError) as exc:
            self._log(f"LLM 结果已提交，但通信状态持久化失败: {type(exc).__name__}")
