"""Deterministic request orchestration; business Context is committed only by its owner."""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import time
from dataclasses import dataclass

import httpx

from .config import fingerprint
from .config_model import ChatConfig, LLMConfig
from .provider.base import Result
from .gateway import WireRequest, HTTPFailure, Frame
from .media import MediaService, MediaError
from .model import (
    Context,
    ImagePrompt,
    SystemPrompt,
    TokenUsage,
    ContextOverflowError,
    CallOutcome,
)
from .provider import get_provider
from .provider.base import Failure, ProviderFailure
from .provider.common import normalize


from .snapshot import stable, prefix, context_signature, estimate


@dataclass
class Pending:
    result: Result
    planned: Context
    expected: str
    session: str
    config: ChatConfig
    route: LLMConfig
    route_index: int
    call_id: str
    checkpoint: dict | None
    fallback_reason: str = ""


class Runtime:
    def __init__(self, state, ledger, prices=None, logger=None, subscribers=None):
        self.state, self.ledger, self.prices, self.logger = (
            state,
            ledger,
            prices,
            logger,
        )
        self.media = MediaService(state, on_warning=self._log)
        self.subscribers = subscribers if subscribers is not None else []

    async def _db(self, method, *args, **kwargs):
        # Retrying local persistence must never repeat a remote model request.
        for attempt in range(3):
            try:
                return await asyncio.to_thread(method, *args, **kwargs)
            except sqlite3.OperationalError:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.05 * (attempt + 1))

    def _state_get(self, namespace, key):
        try:
            return self.state.get(namespace, key)
        except (sqlite3.Error, OSError) as exc:
            self._log(f"LLM 状态读取失败，将使用完整上下文: {type(exc).__name__}")
            return None

    def _log(self, message):
        if self.logger:
            self.logger.warning(message)

    def _state_put(self, namespace, key, value, expires=None):
        try:
            self.state.put(namespace, key, value, expires)
        except (sqlite3.Error, OSError) as exc:
            self._log(f"LLM 状态写入失败，当前调用保守恢复: {type(exc).__name__}")

    async def execute(
        self, context, prompts, config: ChatConfig, gateway_for_route, timeout
    ):
        if not config.candidates or config.request_policy is None:
            raise ValueError("Runtime 只接受完整 ChatConfig")
        self.media.degraded = False
        expected = context_signature(context)
        planned = copy.deepcopy(context)
        planned.append(copy.deepcopy(prompts))
        planned = normalize(planned)
        deadline = time.monotonic() + timeout
        call_id = await self._db(self.ledger.begin_call, config.preset_name)
        route_key = fingerprint([context.session_id, config.revision])
        cooldown = await asyncio.to_thread(self._state_get, "cooldown", route_key)
        selected = (
            cooldown.get("index", 0)
            if cooldown and cooldown["until"] > time.time()
            else 0
        )
        if selected >= len(config.candidates):
            selected = 0
        order = list(range(selected, len(config.candidates)))
        attempts, max_attempts = 0, int(config.request_policy.max_attempts)
        last_error, reason, overflow = None, "", False
        last_request_id = ""
        last_remote_state = "not_sent"
        last_category = "preflight"
        try:
            for index in order:
                route = copy.deepcopy(config.candidates[index])
                provider = get_provider(route)
                if not callable(getattr(provider, "encode", None)):
                    raise ValueError(
                        f"{route.provider_type} 不是对话操作，请使用对应的独立客户端"
                    )
                parent_enabled = (
                    provider.supports_parent
                    and not await asyncio.to_thread(
                        self._state_get, "unsupported_parent", route.scope
                    )
                )
                if route.model_params.get(
                    "background"
                ) and not route.provider_options.get("responses", {}).get(
                    "supports_background"
                ):
                    raise ValueError(
                        "后台 Responses 需要显式声明 responses.supports_background"
                    )
                estimation = await asyncio.to_thread(estimate, planned)
                if route.max_context and estimation["input"] > route.max_context:
                    overflow = True
                    continue
                gateway = await gateway_for_route(route)
                checkpoint_key = fingerprint([context.session_id, route.scope])
                checkpoints = (
                    await asyncio.to_thread(
                        self._state_get, "checkpoints", checkpoint_key
                    )
                    or []
                )
                current_prefix = prefix(planned)
                header = fingerprint(
                    [
                        stable(vars(planned._system_prompt))
                        if planned._system_prompt
                        else None,
                        [stable(vars(t)) for t in planned._tools],
                    ]
                )
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
                inline, force, recovered, retries = False, set(), set(), 0
                media = {}
                while attempts < max_attempts:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("LLM 逻辑请求总时限已到")
                    send_context = copy.deepcopy(planned)
                    if parent:
                        suffix = send_context._messages[len(parent["prefix"]) :]
                        send_context._messages = (
                            [send_context._system_prompt]
                            if send_context._system_prompt
                            else []
                        ) + [p for p in suffix if not isinstance(p, SystemPrompt)]

                    async def send_file(wire):
                        return await self._control(
                            call_id, route, wire, gateway, deadline
                        )

                    try:
                        media = await self._prepare_media(
                            send_context,
                            planned,
                            provider,
                            send_file,
                            media,
                            inline,
                            force,
                        )
                    except (MediaError, OSError, ProviderFailure) as exc:
                        attempts += 1
                        last_error = exc
                        last_category = "media_preflight"
                        reason = f"media_preflight/{type(exc).__name__}"
                        break
                    payload = provider.encode(send_context, media)
                    if parent:
                        payload["previous_response_id"] = parent["response_id"]
                        payload["store"] = True
                    elif (
                        route.provider_type == "responses_api"
                        and parent_enabled
                        and route.context_history_mode != "full"
                    ):
                        payload["store"] = True
                    if any(
                        key in route.model_params
                        for key in (
                            "messages",
                            "input",
                            "previous_response_id",
                            "tools",
                        )
                    ):
                        raise ValueError(
                            "llm_params 不能覆盖 Runtime 管理的上下文或工具字段"
                        )
                    identity = {
                        "preset": config.preset_name,
                        "provider": route.ref,
                        "scope": route.scope,
                        "model": route.model,
                        "operation": route.provider_type,
                        "config_revision": config.revision,
                    }
                    price = None
                    if self.prices:
                        try:
                            price = await self._db(
                                self.prices.snapshot,
                                {
                                    k: identity[k]
                                    for k in ("preset", "scope", "model", "operation")
                                },
                            )
                        except (sqlite3.Error, OSError, ValueError):
                            self._log("LLM 定价目录不可读，本次费用保留未知")
                    if price is None and route.legacy_price is not None:
                        legacy = route.legacy_price.to_dict()
                        price = {
                            "version": "legacy",
                            "rules": {
                                "currency": legacy.get("currency", "CNY"),
                                "tiers": [
                                    {
                                        "rates": {
                                            "input_uncached": str(
                                                legacy.get("input", 0)
                                            ),
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
                        call_id,
                        identity,
                        estimation,
                        price_snapshot=price,
                    )
                    last_request_id = request_id
                    last_remote_state = "unknown"
                    attempts += 1
                    decoder = None
                    partial = False
                    result = None
                    raw_body = None
                    background_id = ""
                    platform_request_id = ""
                    gateway.last_request_id = ""
                    try:
                        wire = WireRequest(
                            route.endpoint, headers=provider.headers(), json=payload
                        )
                        async with asyncio.timeout(
                            max(0.001, deadline - time.monotonic())
                        ):
                            if payload.get("stream"):
                                decoder = provider.stream_decoder()
                                async for frame in gateway.stream(wire, remaining):
                                    platform_request_id = gateway.last_request_id
                                    if isinstance(frame, Frame):
                                        if not frame.data.strip():
                                            continue
                                        if frame.data == "[DONE]":
                                            decoder.complete = True
                                            continue
                                        body = json.loads(frame.data)
                                    else:
                                        body = frame
                                    decoder.feed(body)
                                result = decoder.finish()
                            else:
                                body = await gateway.send(wire, remaining)
                                platform_request_id = gateway.last_request_id
                                raw_body = body
                                background_id = (
                                    body.get("id", "")
                                    if payload.get("background")
                                    else ""
                                )
                                if background_id:
                                    while body.get("status") in (
                                        "queued",
                                        "in_progress",
                                    ):
                                        body = await self._control(
                                            call_id,
                                            route,
                                            provider.background_operation(
                                                background_id
                                            ),
                                            gateway,
                                            deadline,
                                            operation="response_retrieve",
                                        )
                                        raw_body = body
                                        if body.get("status") in (
                                            "queued",
                                            "in_progress",
                                        ):
                                            await asyncio.sleep(0.25)
                                result = provider.decode(body)
                            if decoder and payload.get("background"):
                                background_id = decoder.body.get("id", "")
                        partial = decoder.partial if decoder else bool(result.prompts)
                        if not result.complete:
                            raise RuntimeError("LLM 响应未完成，停止重发")
                        result.request_id = request_id
                        if result.stored is False and parent_enabled:
                            await asyncio.to_thread(
                                self._state_put,
                                "unsupported_parent",
                                route.scope,
                                {"evidence": "store_false"},
                                time.time() + 3600,
                            )
                            self._log(
                                f"LLM 端点未保存 response，后续使用完整历史: {route.ref}"
                            )
                        output_estimate = (
                            await asyncio.to_thread(estimate, Context(result.prompts))
                        )["input"]
                        await self._db(
                            self.ledger.observe_usage,
                            request_id,
                            "estimated",
                            {"output": output_estimate},
                            "local_output",
                        )
                        await self._db(
                            self.ledger.observe_usage,
                            request_id,
                            "reported",
                            result.usage,
                            "final",
                            raw=result.raw_usage,
                        )
                        for image_prompt, image_wire in result.images:
                            cached = await self.media.materialize(image_wire, gateway)
                            image_prompt.resource = cached.image
                            from TIYA.file_cache import get_file_path_async

                            image_prompt.image_input = await get_file_path_async(
                                cached.image.value
                            )
                        result.images.clear()
                        await self._db(
                            self.ledger.finish_request,
                            request_id,
                            "completed",
                            remote_state="completed",
                            response_id=result.response_id,
                            platform_request_id=platform_request_id,
                            reported_model=result.reported_model,
                        )
                        usage_row = await self._db(self.ledger.get_request, request_id)
                        usage = usage_row["effective"]
                        result.prompts.usage = TokenUsage(
                            input_t=usage.get("input", 0) or 0,
                            output_t=usage.get("output", 0) or 0,
                            cache_hit_t=usage.get("cached", 0) or 0,
                            total_t=usage.get(
                                "total",
                                (usage.get("input", 0) or 0)
                                + (usage.get("output", 0) or 0),
                            ),
                            model=route.model,
                        )
                        checkpoint = (
                            {
                                "response_id": result.response_id,
                                "header": header,
                                "prefix": prefix(
                                    Context(planned._messages, result.prompts)
                                ),
                            }
                            if (
                                parent_enabled
                                and result.stored is not False
                                and result.response_id
                                and route.context_history_mode != "full"
                            )
                            else None
                        )
                        await self._db(self.ledger.finish_call, call_id, "completed")
                        return Pending(
                            result,
                            planned,
                            expected,
                            context.session_id,
                            config,
                            route,
                            index,
                            call_id,
                            checkpoint,
                            reason,
                        )
                    except BaseException as exc:
                        if decoder:
                            partial = decoder.partial
                            # Retain known usage even when tool JSON/final framing is damaged.
                            try:
                                usage_result = provider.usage_result(decoder.usage)
                                await self._db(
                                    self.ledger.observe_usage,
                                    request_id,
                                    "reported",
                                    usage_result.usage,
                                    "partial",
                                    raw=decoder.usage,
                                )
                            except (ValueError, KeyError):
                                pass
                        cancelled = isinstance(exc, asyncio.CancelledError)
                        if decoder and payload.get("background"):
                            background_id = decoder.body.get("id", "")
                        cancel_status = None
                        if background_id and isinstance(
                            exc, (asyncio.CancelledError, TimeoutError)
                        ):
                            try:
                                cancelled_body = await asyncio.wait_for(
                                    self._control(
                                        call_id,
                                        route,
                                        provider.background_operation(
                                            background_id, cancel=True
                                        ),
                                        gateway,
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
                                self._log(
                                    f"后台 Responses 取消未确认 request={request_id}"
                                )
                        if isinstance(exc, HTTPFailure):
                            failure = provider.failure(
                                exc.status, exc.body, exc.headers
                            )
                        elif isinstance(exc, ProviderFailure):
                            failure = exc.failure
                        elif isinstance(
                            exc,
                            (
                                httpx.ConnectError,
                                httpx.ConnectTimeout,
                                httpx.PoolTimeout,
                            ),
                        ):
                            failure = Failure("connection", remote_state="rejected")
                        else:
                            failure = Failure(
                                "cancelled" if cancelled else "transport",
                                remote_state="unknown",
                            )
                        if partial and failure.safe:
                            failure.remote_state = "unknown"
                        raw_error = (
                            exc.body
                            if isinstance(exc, HTTPFailure)
                            else exc.failure.raw
                            if isinstance(exc, ProviderFailure)
                            else raw_body or {}
                        )
                        if raw_error.get("usage"):
                            known = provider.usage_result(raw_error["usage"])
                            if any(
                                isinstance(v, (int, float)) and v > 0
                                for v in known.usage.values()
                            ):
                                failure.remote_state = "unknown"
                            await self._db(
                                self.ledger.observe_usage,
                                request_id,
                                "reported",
                                known.usage,
                                "error_usage",
                                raw=known.raw_usage,
                            )
                        if failure.safe:
                            await self._db(
                                self.ledger.observe_usage,
                                request_id,
                                "estimated",
                                {
                                    "input": 0,
                                    "output": 0,
                                    "cached": 0,
                                    "algorithm": "rejected-before-execution-v1",
                                },
                                "rejected",
                            )
                        remote_complete = bool(result and result.complete) or bool(
                            raw_body
                            and (
                                raw_body.get("status") == "completed"
                                or any(
                                    "message" in c for c in raw_body.get("choices", [])
                                )
                            )
                        )
                        status = (
                            (
                                "materialization_failed"
                                if isinstance(exc, MediaError)
                                else "local_failure"
                            )
                            if remote_complete
                            else (
                                "cancelled"
                                if cancelled
                                else "rejected"
                                if failure.safe
                                else "unknown"
                            )
                        )
                        last_remote_state = cancel_status or (
                            "completed" if remote_complete else failure.remote_state
                        )
                        last_category = failure.category
                        await self._db(
                            self.ledger.finish_request,
                            request_id,
                            status,
                            remote_state=cancel_status or "completed"
                            if remote_complete
                            else cancel_status or failure.remote_state,
                            category=failure.category,
                            code=failure.code,
                            partial=partial,
                            platform_request_id=platform_request_id
                            or gateway.last_request_id,
                            http_status=exc.status
                            if isinstance(exc, HTTPFailure)
                            else 200
                            if isinstance(exc, ProviderFailure)
                            else None,
                            resource=failure.resource,
                        )
                        if cancelled or partial or not failure.safe:
                            raise
                        last_error = exc
                        reason = (
                            f"{failure.category}/{failure.code} request={request_id}"
                        )
                        if failure.category == "context_overflow":
                            overflow = True
                            break
                        if (
                            failure.category == "parent_invalid"
                            and parent
                            and "parent" not in recovered
                        ):
                            recovered.add("parent")
                            checkpoints = [
                                c
                                for c in checkpoints
                                if c["response_id"] != parent["response_id"]
                            ]
                            await asyncio.to_thread(
                                self._state_put,
                                "checkpoints",
                                checkpoint_key,
                                checkpoints,
                            )
                            parent = None
                            self._log(
                                f"LLM 续接失效，完整回放当前上下文 call={call_id} request={request_id}"
                            )
                            continue
                        if (
                            failure.category in ("file_invalid", "files_unsupported")
                            and "files" not in recovered
                        ):
                            if (
                                failure.category == "file_invalid"
                                and not failure.resource
                            ):
                                break  # Never invalidate unrelated files without a concrete target.
                            recovered.add("files")
                            inline = failure.category == "files_unsupported"
                            force = {
                                pid
                                for pid, refs in media.items()
                                if any(
                                    ref.kind == "remote"
                                    and ref.value == failure.resource
                                    for ref in refs
                                )
                            }
                            if not inline and route.files is not None:
                                from .provider.files import get_files_provider

                                scope = get_files_provider(
                                    route.files.as_provider_settings(),
                                    route.files.provider_type,
                                ).scope
                                await self._db(
                                    self.state.put,
                                    "invalid_files",
                                    fingerprint([scope, failure.resource]),
                                    {"invalid": True},
                                )
                                if not force and parent:
                                    parent = None
                                    media = {}
                            if inline:
                                await self._db(
                                    self.state.put,
                                    "unsupported_files",
                                    route.scope,
                                    {"reason": failure.code},
                                    time.time() + 3600,
                                )
                            continue
                        if failure.category in (
                            "rate_limit",
                            "connection",
                        ) and retries < (
                            route.max_retries
                            if route.max_retries is not None
                            else config.request_policy.max_retries_per_candidate
                        ):
                            retries += 1
                            delay = min(
                                failure.retry_after or 0.2,
                                max(0, deadline - time.monotonic()),
                            )
                            await asyncio.sleep(delay)
                            continue
                        break
            if overflow:
                raise ContextOverflowError("所有可用组合上下文窗口不足")
            if last_error:
                raise last_error
            raise RuntimeError("LLM 请求预算耗尽")
        except BaseException as exc:
            exc.llm_outcome = CallOutcome(
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                False,
                call_id,
                last_request_id,
                last_remote_state,
                last_category,
            )
            await self._db(self.ledger.finish_call, call_id, "stopped")
            raise
        finally:
            for subscriber in tuple(self.subscribers):
                try:
                    target = await asyncio.to_thread(
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
        self, send_context, planned, provider, send_file, media, inline, force
    ):
        for prompt in send_context:
            if isinstance(prompt, ImagePrompt) and (
                prompt.id not in media or inline or prompt.id in force
            ):
                media[prompt.id] = await self.media.prepare_prompt(
                    prompt,
                    provider,
                    send_file,
                    inline=inline,
                    force=prompt.id in force,
                )
                original = next(p for p in planned._messages if p.id == prompt.id)
                original.image_input = prompt.image_input
                original.image = list(prompt.image)
                original.media_descriptor = prompt.media_descriptor
                original.prepared_input = prompt.prepared_input
        return media

    async def _control(
        self, call_id, route, wire, gateway, deadline, operation="files"
    ):
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
            if isinstance(exc, HTTPFailure):
                classifier = (
                    file_provider.failure
                    if file_provider
                    else get_provider(route).failure
                )
                failure = classifier(exc.status, exc.body, exc.headers)
            elif isinstance(exc, ProviderFailure):
                failure = exc.failure
            else:
                failure = Failure("transport")
            await self._db(
                self.ledger.finish_request,
                request_id,
                "rejected" if failure.safe else "unknown",
                remote_state=failure.remote_state,
                code=failure.code,
                http_status=exc.status if isinstance(exc, HTTPFailure) else None,
                platform_request_id=gateway.last_request_id,
            )
            raise
        await self._db(
            self.ledger.finish_request,
            request_id,
            "completed",
            remote_state="completed",
            platform_request_id=gateway.last_request_id,
        )
        return result

    def commit(self, context, pending):
        self._apply(context, pending)
        self._persist_commit(context.session_id, pending)
        return pending.result.prompts

    async def commit_async(self, context, pending, on_commit=None):
        self._apply(context, pending)
        if on_commit:
            on_commit()
        await asyncio.to_thread(self._persist_commit, context.session_id, pending)
        return pending.result.prompts

    @staticmethod
    def _apply(context, pending):
        if (
            context.session_id != pending.session
            or context_signature(context) != pending.expected
        ):
            raise RuntimeError("Context 在请求期间已修改，拒绝提交过期响应")
        updated = copy.deepcopy(pending.planned)
        if updated._usage_accumulation.model != pending.route.model:
            # Legacy Context exposes one model's aggregate; the ledger retains every model's charges.
            updated._usage_accumulation = TokenUsage(model=pending.route.model)
        updated.append(pending.result.prompts)
        context.__dict__.update(updated.__dict__)

    def _persist_commit(self, session_id, pending):
        try:
            key = fingerprint([session_id, pending.route.scope])
            if pending.checkpoint:
                entries = self.state.get("checkpoints", key) or []
                entries.append(pending.checkpoint)
                self.state.put("checkpoints", key, entries[-64:])
            route_key = fingerprint([session_id, pending.config.revision])
            old = self.state.get("cooldown", route_key)
            if pending.route_index:
                if (
                    not old
                    or old["index"] != pending.route_index
                    or old["until"] <= time.time()
                ):
                    until = time.time() + float(
                        pending.config.request_policy.fallback_cooldown_seconds
                    )
                    self.state.put(
                        "cooldown",
                        route_key,
                        {"index": pending.route_index, "until": until},
                    )
                    self._log(
                        f"LLM 已进入 fallback: {pending.config.candidates[0].ref} -> {pending.route.ref}; "
                        f"原因={pending.fallback_reason}; call={pending.call_id}; request={pending.result.request_id}; 冷却至={until}"
                    )
            elif old:
                self.state.delete("cooldown", route_key)
                self._log(
                    f"LLM 已恢复首选组合 {pending.route.ref}; call={pending.call_id}"
                )
        except (sqlite3.Error, OSError) as exc:
            self._log(f"LLM 结果已提交，但通信状态持久化失败: {type(exc).__name__}")
