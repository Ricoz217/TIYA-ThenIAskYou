"""Media preparation/materialization using the existing file cache and its ordinary TTL."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image

from TIYA.file_cache import add_file_async, get_file_path_async

from .config import fingerprint
from .config_model import Params
from .gateway import Gateway, HTTPFailure, WireRequest
from .model import ImagePrompt, ImageResponseItem, MediaRef
from .provider.base import ConversationProvider, ProviderFailure, classify_error
from .provider.files import get_files_provider
from .state import StateStore
from .workers import run_sync


class MediaError(Exception):
    def __init__(self, stage: str, message: str, *, request_id: str = "") -> None:
        """标记 input/upload/download_decode/validate_cache 等失败阶段及账本请求 ID。"""
        super().__init__(f"{stage}: {message}")
        self.stage, self.request_id = stage, request_id


def inspect_image(content: bytes) -> tuple[str, int, int]:
    """同步验证图片完整性，返回 MIME、宽和高；无效内容抛异常。"""
    with Image.open(io.BytesIO(content)) as image:
        mime = Image.MIME.get(image.format)
        size = image.size
        image.verify()
    if mime is None:
        raise ValueError("未知图片格式")
    return mime, *size


class MediaService:
    def __init__(
        self,
        state: StateStore,
        *,
        lease_seconds: float = 120,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        """借用通信状态库做上传去重；lease_seconds 为租约秒数，不改变原件 TTL。"""
        self.state = state
        self.lease_seconds = lease_seconds
        self.degraded = False
        self.on_warning = on_warning

    async def _meta(self, operation, *args):
        try:
            return await run_sync(getattr(self.state, operation), *args)
        except (sqlite3.Error, OSError) as exc:
            if not self.degraded and self.on_warning:
                self.on_warning(
                    f"LLM 媒体状态库不可用，本次使用内联图片: {type(exc).__name__}"
                )
            self.degraded = True
            return None

    async def content(self, ref: MediaRef, gateway: Gateway | None = None) -> bytes:
        """读取 bytes/path/cache/base64，URL 需 gateway；远端 file_id 不能在此读取。"""
        if ref.kind == "bytes":
            return ref.value
        if ref.kind == "path":
            return await run_sync(Path(ref.value).read_bytes)
        if ref.kind == "cache":
            path = await get_file_path_async(ref.value)
            return await run_sync(path.read_bytes)
        if ref.kind == "base64":
            return base64.b64decode(ref.value, validate=True)
        if ref.kind == "url" and gateway is not None:
            return await gateway.download(ref.value)
        raise MediaError("input", f"无法读取 {ref.kind} 图片")

    async def materialize(self, item: Params, gateway: Gateway) -> ImageResponseItem:
        """立即下载或解码响应图片，验证并写入现有 file_cache 后返回缓存引用。
        不向业务泄露短期 URL；下载/解码和验证/落盘失败由 MediaError.stage 区分。
        """
        try:
            if item.get("b64_json"):
                content = base64.b64decode(item["b64_json"], validate=True)
            elif item.get("url", "").startswith("data:image/"):
                header, separator, payload = item["url"].partition(",")
                if not separator or not header.endswith(";base64"):
                    raise ValueError("不支持的图片 data URL")
                content = base64.b64decode(payload, validate=True)
            elif item.get("url"):
                content = await gateway.download(item["url"])
            else:
                raise ValueError("图片响应缺少 b64_json/url")
        except Exception as exc:
            raise MediaError("download_decode", str(exc)) from exc
        try:
            mime, width, height = await run_sync(inspect_image, content)
            cache_id = await add_file_async(content, "IMAGE")
        except Exception as exc:
            raise MediaError("validate_cache", str(exc)) from exc
        return ImageResponseItem(
            MediaRef("cache", cache_id, mime, width=width, height=height),
            item.get("revised_prompt"),
            {
                k: v
                for k, v in item.items()
                if k not in ("url", "b64_json", "revised_prompt")
            },
        )

    async def upload(
        self,
        ref: MediaRef,
        settings: Params,
        send: Callable[[WireRequest], Awaitable[Params]],
        *,
        force: bool = False,
        timeout: float = 300,
        category: str = "files_api",
    ) -> MediaRef:
        """按文件账户和内容指纹复用上传，返回 remote 引用；send 负责网络和记账。
        force 忽略旧上传映射；timeout 限制等待并发上传的秒数，租约心跳防止重复上传。
        """
        provider = get_files_provider(settings, category)
        content = await self.content(ref)
        digest = hashlib.sha256(content).hexdigest()
        key = fingerprint([provider.scope, digest, ref.mime_type])
        owner = uuid4().hex
        started = time.monotonic()
        while True:
            cached = await self._meta("get", "uploads", key)
            if cached and await self._meta(
                "get", "invalid_files", fingerprint([provider.scope, cached["file_id"]])
            ):
                cached = None
            if cached and not force:
                return MediaRef(
                    "remote",
                    cached["file_id"],
                    ref.mime_type,
                    provider.scope,
                    expires_at=cached.get("expires_at"),
                )
            if await run_sync(self.state.claim, key, owner, self.lease_seconds):
                break
            if time.monotonic() - started > timeout:
                raise MediaError("upload", "等待并发上传超时")
            await asyncio.sleep(0.05)

        async def heartbeat():
            while True:
                await asyncio.sleep(min(30, self.lease_seconds / 3))
                if not await run_sync(
                    self.state.renew_lease, key, owner, self.lease_seconds
                ):
                    raise MediaError("upload", "上传租约已被其他调用接管")

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            cached = await self._meta("get", "uploads", key)
            if cached and await self._meta(
                "get", "invalid_files", fingerprint([provider.scope, cached["file_id"]])
            ):
                cached = None
            if cached and not force:
                return MediaRef(
                    "remote",
                    cached["file_id"],
                    ref.mime_type,
                    provider.scope,
                    expires_at=cached.get("expires_at"),
                )
            body = await send(
                provider.upload_request(
                    content,
                    "image." + (ref.mime_type or "image/png").split("/")[-1],
                    ref.mime_type,
                )
            )
            metadata = provider.decode(body)
            await self._meta(
                "put",
                "uploads",
                key,
                {"file_id": metadata.file_id, "expires_at": metadata.expires_at},
                metadata.expires_at,
            )
            return MediaRef(
                "remote",
                metadata.file_id,
                ref.mime_type,
                provider.scope,
                expires_at=metadata.expires_at,
            )
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            await run_sync(self.state.release, key, owner)

    async def prepare_prompt(
        self,
        prompt: ImagePrompt,
        provider: ConversationProvider,
        send: Callable[[WireRequest], Awaitable[Params]],
        *,
        inline: bool = False,
        force: bool = False,
        _rebuilt: bool = False,
    ) -> list[MediaRef]:
        """准备一条图片输入并返回有序引用，优先复用远端 file_id，按能力降级内联。
        inline 强制内联；force 重建失效上传；_rebuilt 限制本地缓存重建次数。
        会更新当前请求副本的准备信息，不更改原件保留策略或业务历史顺序。
        """
        route = provider.route
        files_settings = route.files.as_provider_settings() if route.files else {}
        descriptor = (
            prompt.media_descriptor
            if prompt.prepared_input == str(prompt.image_input)
            and prompt.media_descriptor
            else fingerprint(
                [
                    hashlib.sha256(prompt.image_input).hexdigest()
                    if isinstance(prompt.image_input, bytes)
                    else str(prompt.image_input),
                    prompt.image,
                ]
            )
        )
        key = fingerprint(
            [route.scope, files_settings, prompt.id, descriptor, prompt.image]
        )
        previous = await self._meta("get", "media", key)
        if previous and previous.get("prepared_input"):
            prompt.image_input = Path(previous["prepared_input"])
            prompt.prepared_input = previous["prepared_input"]
            prompt.media_descriptor = descriptor
        # Reuse valid remote refs without touching a possibly expired local original.
        if previous and not inline and not force and previous.get("remote"):
            expiry = previous.get("expires")
            invalid = False
            for ref in previous["remote"]:
                invalid = invalid or bool(
                    await self._meta(
                        "get",
                        "invalid_files",
                        fingerprint([ref["scope"], ref["value"]]),
                    )
                )
            if not invalid and (expiry is None or expiry > time.time()):
                return [MediaRef(**r) for r in previous["remote"]]
        if previous:
            sources = [MediaRef(**r) for r in previous["sources"]]
            inline = inline or previous.get("mode") == "inline"
        else:
            if prompt.image:
                sources = [MediaRef("cache", value) for value in prompt.image]
            else:
                raw = prompt.image_input
                source = (
                    MediaRef("path", str(raw))
                    if isinstance(raw, Path)
                    else MediaRef("bytes", raw)
                    if isinstance(raw, bytes)
                    else MediaRef("base64", raw)
                )
                data = await self.content(source)
                mime, width, height = await run_sync(inspect_image, data)
                cache_id = await add_file_async(data, "IMAGE")
                sources = [
                    MediaRef("cache", cache_id, mime, width=width, height=height)
                ]
                # Keep an ordinary TTL-backed local reference in Context, not large bytes/Base64.
                prompt.image_input = await get_file_path_async(cache_id)
                prompt.media_descriptor = descriptor
                prompt.prepared_input = str(prompt.image_input)
        prepared = []
        allow_files = files_settings and provider.accepts_image_files() and not inline
        disabled = await self._meta("get", "unsupported_files", route.scope)
        allow_files = allow_files and not disabled
        allow_files = allow_files and not self.degraded
        for index, ref in enumerate(sources):
            if allow_files and previous and previous.get("remote"):
                prior = MediaRef(**previous["remote"][index])
                invalid = await self._meta(
                    "get", "invalid_files", fingerprint([prior.scope, prior.value])
                )
                expires = (
                    prior.expires_at
                    if prior.expires_at is not None
                    else previous.get("expires")
                )
                if (
                    not invalid
                    and not self.degraded
                    and (expires is None or expires > time.time())
                ):
                    prepared.append(prior)
                    continue
            if prompt.image and ref.kind == "cache":
                try:
                    await get_file_path_async(ref.value)
                except FileNotFoundError:
                    if not _rebuilt and await prompt.load_image({}):
                        return await self.prepare_prompt(
                            prompt,
                            provider,
                            send,
                            inline=inline,
                            force=force,
                            _rebuilt=True,
                        )
                    raise MediaError("input", "历史图片派生件已过期，且原件无法重建")
            data = None
            if not ref.mime_type:
                data = await self.content(ref)
                mime, width, height = await run_sync(inspect_image, data)
                ref = MediaRef(ref.kind, ref.value, mime, width=width, height=height)
            if allow_files:
                try:
                    prepared.append(
                        await self.upload(
                            ref,
                            files_settings,
                            send,
                            force=False,  # Concrete invalid IDs are marked above; valid sibling parts stay reusable.
                            category=route.files.provider_type,
                        )
                    )
                    continue
                except HTTPFailure as exc:
                    failure = classify_error(
                        exc.status,
                        exc.body,
                        files_settings.get("error_dialect", "openai"),
                        exc.headers,
                    )
                    # These are upload failures, not model-generation failures. Auth/malformed must stay visible.
                    if failure.category in (
                        "authentication",
                        "invalid_request",
                        "quota",
                    ):
                        raise ProviderFailure(failure) from exc
                    allow_files = False
                except ProviderFailure as exc:
                    if exc.failure.category != "files_unsupported":
                        raise
                    await self._meta(
                        "put",
                        "unsupported_files",
                        route.scope,
                        {"reason": exc.failure.code},
                        time.time() + 3600,
                    )
                    allow_files = False
                except (OSError, TimeoutError, httpx.RequestError, sqlite3.Error):
                    allow_files = False
            data = data if data is not None else await self.content(ref)
            prepared.append(
                MediaRef("base64", base64.b64encode(data).decode(), ref.mime_type)
            )
        remote = all(r.kind == "remote" for r in prepared)
        ttl = (files_settings or {}).get("expires_after_seconds")
        expiries = [ref.expires_at for ref in prepared if ref.expires_at is not None]
        await self._meta(
            "put",
            "media",
            key,
            {
                "sources": [asdict(r) for r in sources],
                "mode": "remote" if remote else "inline",
                "prepared_input": prompt.prepared_input,
                "remote": [asdict(r) for r in prepared] if remote else None,
                "expires": min(expiries)
                if expiries
                else time.time() + ttl
                if ttl is not None
                else None,
            },
        )
        return prepared
