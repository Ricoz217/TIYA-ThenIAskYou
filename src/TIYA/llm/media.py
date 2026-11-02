"""Media preparation/materialization using the existing file cache and its ordinary TTL."""

from __future__ import annotations

import asyncio
import base64
import io
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict
from uuid import uuid4
from weakref import WeakKeyDictionary

import httpx
from PIL import Image

from TIYA.file_cache import add_file_async, get_file_path_async
from TIYA.utils import fingerprint

from .config_model import Params
from .gateway import Gateway, HTTPFailure, WireRequest
from .model import ImagePrompt, ImageResponseItem, MediaRef
from .provider.base import ConversationProvider, ProviderFailure
from .provider.files import FilesProvider, get_files_provider
from .state import StateStore
from .workers import run_sync


@dataclass(slots=True)
class UploadSender:
    """共享上传的发送入口；gateway 的客户端保留到任务结束，send 使用独立时限。

    普通自定义 send 回调仍可直接传入，由其调用方负责借用资源的生命周期。
    """

    send: Callable[[WireRequest], Awaitable[Params]]
    gateway: Gateway

    async def __call__(self, wire: WireRequest) -> Params:
        return await self.send(wire)


@dataclass(slots=True)
class _SharedUpload:
    task: asyncio.Task[MediaRef]
    waiters: int = 0


@dataclass(slots=True)
class _ClientUse:
    uploads: int = 0
    close_requested: bool = False


@dataclass(slots=True)
class _UploadRegistry:
    # 不跨事件循环等待 Task；不同状态库的持久映射与租约也保持隔离。
    tasks: dict[tuple[str, str], _SharedUpload] = field(default_factory=dict)
    clients: dict[httpx.AsyncClient, _ClientUse] = field(default_factory=dict)
    closing: dict[httpx.AsyncClient, asyncio.Task[None]] = field(default_factory=dict)


_UPLOADS: WeakKeyDictionary[asyncio.AbstractEventLoop, _UploadRegistry] = WeakKeyDictionary()


def _upload_registry() -> _UploadRegistry:
    loop = asyncio.get_running_loop()
    registry = _UPLOADS.get(loop)
    if registry is None:
        registry = _UploadRegistry()
        _UPLOADS[loop] = registry

    return registry


class UploadRecord(TypedDict):
    """uploads 命名空间的记录；key 为 Files 账户 scope:hash_name。"""

    file_id: str
    expires_at: float | None
    mime_type: str
    width: int | None
    height: int | None


class MediaError(Exception):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        request_id: str = "",
        hash_name: str = "",
    ) -> None:
        """携带失败阶段、账本 ID 和相关本地缓存 ID；原异常保留在 __cause__。"""
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.request_id = request_id
        self.hash_name = hash_name


def inspect_image(content: bytes) -> tuple[str, int, int]:
    """同步验证图片完整性，返回 MIME、宽和高；无效内容抛异常。"""
    with Image.open(io.BytesIO(content)) as image:
        mime = Image.MIME.get(image.format)
        size = image.size
        image.verify()

    if mime is None:
        raise ValueError("未知图片格式")

    return mime, *size  # type: ignore


class MediaService:
    def __init__(
        self,
        state: StateStore,
        *,
        lease_seconds: float = 120,
        upload_timeout: float = 300,
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        """协调共享上传；upload_timeout 是任务自身的秒数时限，与单个等待者无关。

        lease_seconds 控制跨进程租约，不改变原件 TTL。任务表只保存正在进行的上传。
        """
        self.state = state
        self.lease_seconds = lease_seconds
        self.upload_timeout = upload_timeout
        self.degraded = False
        self.on_warning = on_warning
        self._upload_namespace: str | None = None

    async def _meta(self, operation, *args):
        """通信元数据操作；状态库不可用时发出一次警告，并降级为内联。"""
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
        """按 hash_name 读取本地原件；未归一输入才解释 kind/value。

        本地文件过期、删除（包括路径查询后的删除）及解码失败统一抛 MediaError，
        保留 hash_name 和原异常。远端 file_id 本身不能下载；URL 需 gateway。
        """
        if ref.kind == "url" and not ref.hash_name and isinstance(ref.value, str) and gateway is not None:
            return await gateway.download(ref.value)

        try:
            if ref.hash_name:
                path = await get_file_path_async(ref.hash_name)
                return await run_sync(path.read_bytes)

            if ref.kind == "bytes" and isinstance(ref.value, bytes):
                return ref.value

            if ref.kind == "path" and isinstance(ref.value, (str, Path)):
                return await run_sync(Path(ref.value).read_bytes)

            if ref.kind == "base64" and isinstance(ref.value, (str, bytes)):
                return base64.b64decode(ref.value, validate=True)

        except (OSError, ValueError) as exc:
            detail = "本地文件不存在或已过期" if isinstance(exc, FileNotFoundError) else "读取或解码失败"
            raise MediaError("input", detail, hash_name=ref.hash_name) from exc

        raise MediaError("input", f"无法读取 {ref.kind} 文件", hash_name=ref.hash_name)

    async def materialize(self, item: Params, gateway: Gateway | None) -> ImageResponseItem:
        """立即下载/解码图片，验证并存入 file_cache；仅未知扩展进入 metadata。"""
        try:
            encoded = item.get("b64_json")
            url = item.get("url")
            if isinstance(encoded, str) and encoded:
                content = base64.b64decode(encoded, validate=True)

            elif isinstance(url, str) and url.startswith("data:image/"):
                header, separator, payload = url.partition(",")
                if not separator or not header.endswith(";base64"):
                    raise ValueError("不支持的图片 data URL")

                content = base64.b64decode(payload, validate=True)

            elif isinstance(url, str) and url and gateway is not None:
                content = await gateway.download(url)

            else:
                raise ValueError("图片响应缺少 b64_json/url 或下载 gateway")

        except Exception as exc:
            raise MediaError("download_decode", str(exc)) from exc

        try:
            mime, width, height = await run_sync(inspect_image, content)
            hash_name = await add_file_async(content, "IMAGE")

        except Exception as exc:
            raise MediaError("validate_cache", str(exc)) from exc

        return ImageResponseItem(
            image=MediaRef(hash_name=hash_name, mime_type=mime, width=width, height=height),
            revised_prompt=item.get("revised_prompt"),
            output_format=item.get("output_format"),
            background=item.get("background"),
            quality=item.get("quality"),
            size=item.get("size"),
            image_index=item.get("image_index"),
            partial_image_index=item.get("partial_image_index"),
            created_at=item.get("created_at"),
            event_type=item.get("type"),
            safety_identifier=item.get("safety_identifier"),
            usage=item.get("usage") or {},
            metadata={
                k: v for k, v in item.items()
                if k not in (
                    "url", "b64_json", "revised_prompt", "output_format", "background",
                    "quality", "size", "image_index", "partial_image_index", "created_at",
                    "type", "usage", "safety_identifier",
                )
            },
        )

    async def _cached_upload(
        self,
        provider: FilesProvider,
        key: str,
        ref: MediaRef,
    ) -> MediaRef | None:
        record: UploadRecord | None = await self._meta("get", "uploads", key)
        if not record:
            return None

        invalid = await self._meta(
            "get", "invalid_files", fingerprint([provider.scope, record["file_id"]]),
        )
        if invalid or self.degraded:
            return None

        return MediaRef(
            "remote", record["file_id"], record.get("mime_type") or ref.mime_type,
            scope=provider.scope,
            width=record.get("width"),
            height=record.get("height"),
            expires_at=record.get("expires_at"),
            hash_name=ref.hash_name,
        )

    async def upload(
        self,
        ref: MediaRef,
        settings: Params | FilesProvider,
        send: Callable[[WireRequest], Awaitable[Params]],
        *,
        force: bool = False,
        timeout: float = 300,
        category: str = "files_api",
        content: bytes | None = None,
    ) -> MediaRef:
        """共享同一状态库、Files 账户及 hash_name 的在途上传，返回相同结果或异常。

        timeout 只限制本调用的等待；还有其他等待者时，超时/取消不影响上传。
        全部退出则取消任务并释放租约。任务自身受 upload_timeout 限制，不自动重试。
        force 绕过已完成映射，但仍复用正在进行的上传。命中映射不读取图片文件。
        content 借用本轮已读取的同一原件，避免重复读取；原始输入先进入 file_cache。
        """
        async with asyncio.timeout(timeout):
            return await self._join_upload(ref, settings, send, force, category, content)

    async def _join_upload(
        self,
        ref: MediaRef,
        settings: Params | FilesProvider,
        send: Callable[[WireRequest], Awaitable[Params]],
        force: bool,
        category: str,
        content: bytes | None,
    ) -> MediaRef:
        if isinstance(settings, FilesProvider):
            provider = settings

        else:
            provider = get_files_provider(settings, category)

        if not ref.hash_name:
            content = content if content is not None else await self.content(ref)
            hash_name = await add_file_async(content, "IMAGE")
            ref = MediaRef(hash_name=hash_name, mime_type=ref.mime_type,
                           width=ref.width, height=ref.height)

        key = f"{provider.scope}:{ref.hash_name}"
        if self._upload_namespace is None:
            self._upload_namespace = str(self.state.db.path.resolve())

        registry = _upload_registry()
        task_key = (self._upload_namespace, key)
        while True:
            shared = registry.tasks.get(task_key)
            if shared is not None and shared.task.cancelling():
                # 上一批等待者已全部退出：先等它释放租约，再开始新的调用。
                await asyncio.shield(asyncio.gather(shared.task, return_exceptions=True))
                continue

            if shared is None and not force:
                cached = await self._cached_upload(provider, key, ref)
                # 查库期间其他协程可能已经创建任务；同一事件循环内直接加入。
                shared = registry.tasks.get(task_key)
                if shared is not None:
                    continue

                if cached is not None:
                    return cached

            if shared is None:
                shared = self._start_upload(registry, task_key, ref, provider, send, force, content)

            shared.waiters += 1
            try:
                return await asyncio.shield(shared.task)

            finally:
                shared.waiters -= 1
                if shared.waiters == 0 and not shared.task.done():
                    shared.task.cancel()
                    await asyncio.gather(shared.task, return_exceptions=True)

    def _start_upload(
        self,
        registry: _UploadRegistry,
        task_key: tuple[str, str],
        ref: MediaRef,
        provider: FilesProvider,
        send: Callable[[WireRequest], Awaitable[Params]],
        force: bool,
        content: bytes | None,
    ) -> _SharedUpload:
        async def run() -> MediaRef:
            async with asyncio.timeout(self.upload_timeout):
                return await self._upload_owned(ref, provider, send, task_key[1], force, content)

        # 查表、创建和登记之间没有 await，不需要再增加一把进程内锁。
        client = send.gateway.client if isinstance(send, UploadSender) else None
        shared = _SharedUpload(asyncio.create_task(run()))
        registry.tasks[task_key] = shared
        if client is not None:
            use = registry.clients.get(client)
            if use is None:
                use = _ClientUse()
                registry.clients[client] = use

            use.uploads += 1

        shared.task.add_done_callback(
            lambda task: self._upload_finished(registry, task_key, shared, client)
        )
        return shared

    @staticmethod
    def _upload_finished(
        registry: _UploadRegistry,
        task_key: tuple[str, str],
        shared: _SharedUpload,
        client: httpx.AsyncClient | None,
    ) -> None:
        if registry.tasks.get(task_key) is shared:
            del registry.tasks[task_key]

        if not shared.task.cancelled():
            shared.task.exception()  # 所有等待者均退出时也回收异常，不产生未读取异常告警。

        if client is None:
            return

        use = registry.clients[client]
        use.uploads -= 1
        if use.uploads:
            return

        del registry.clients[client]
        if not use.close_requested:
            return

        closing = asyncio.create_task(client.aclose())
        registry.closing[client] = closing

        def closed(task: asyncio.Task[None]) -> None:
            registry.closing.pop(client, None)
            if not task.cancelled() and (error := task.exception()) is not None:
                task.get_loop().call_exception_handler(
                    {"message": "共享上传结束后关闭 HTTP 客户端失败", "exception": error}
                )

        closing.add_done_callback(closed)

    async def close_clients(self, clients: list[httpx.AsyncClient]) -> None:
        """关闭 Chat 释放的客户端；仍被共享上传使用时登记延后关闭，不阻塞取消者。

        最后一个使用该客户端的上传结束后负责关闭，不创建新的 HTTP 连接池。
        """
        registry = _upload_registry()
        for client in clients:
            use = registry.clients.get(client)
            if use is not None:
                use.close_requested = True
                continue

            closing = registry.closing.get(client)
            if closing is not None:
                await asyncio.shield(closing)

            elif not getattr(client, "is_closed", False):
                await client.aclose()

    async def _upload_owned(
        self,
        ref: MediaRef,
        provider: FilesProvider,
        send: Callable[[WireRequest], Awaitable[Params]],
        key: str,
        force: bool,
        content: bytes | None,
    ) -> MediaRef:
        """共享任务执行一次上传；只有跨进程租约被占用时才轮询持久结果。"""
        owner = uuid4().hex
        while True:
            if await self._claim_upload(key, owner):
                break

            if not force:
                cached = await self._cached_upload(provider, key, ref)
                if cached is not None:
                    return cached

            await asyncio.sleep(0.05)

        async def heartbeat():
            while True:
                await asyncio.sleep(min(30.0, self.lease_seconds / 3))
                if not await run_sync(self.state.renew_lease, key, owner, self.lease_seconds):
                    raise MediaError("upload", "上传租约已被其他调用接管", hash_name=ref.hash_name)

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            # 拿到租约前，另一个调用可能已经完成上传，必须复查。
            if not force:
                cached = await self._cached_upload(provider, key, ref)
                if cached is not None:
                    return cached

            content = content if content is not None else await self.content(ref)
            mime = ref.mime_type
            width = ref.width
            height = ref.height
            if not mime:
                mime, width, height = await run_sync(inspect_image, content)

            wire = provider.upload_request(
                content,
                "image." + mime.split("/")[-1],
                mime,
            )
            body = await send(wire)
            metadata = provider.decode(body)
            record: UploadRecord = {
                "file_id": metadata.file_id,
                "expires_at": metadata.expires_at,
                "mime_type": mime,
                "width": width,
                "height": height,
            }
            await self._meta(
                "put", "uploads", key, record,
                metadata.expires_at,
            )
            return MediaRef(
                "remote", metadata.file_id, mime, scope=provider.scope,
                width=width, height=height, expires_at=metadata.expires_at,
                hash_name=ref.hash_name,
            )

        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            await run_sync(self.state.release, key, owner)

    async def _claim_upload(self, key: str, owner: str) -> bool:
        """SQLite 在线程中执行；取消时等它完成，释放可能刚取得的租约。"""
        claim = asyncio.create_task(run_sync(self.state.claim, key, owner, self.lease_seconds))
        try:
            return await asyncio.shield(claim)

        except asyncio.CancelledError:
            result = await asyncio.gather(claim, return_exceptions=True)
            if result[0] is True:
                await run_sync(self.state.release, key, owner)

            raise

    async def prepare_prompt(
        self,
        prompt: ImagePrompt,
        provider: ConversationProvider,
        send: Callable[[WireRequest], Awaitable[Params]],
        *,
        inline: bool = False,
        _rebuilt: bool = False,
    ) -> list[MediaRef]:
        """归一为缓存 ID，再按能力选择 file_id/内联；保持分片顺序。

        原始输入只读取、验证和入库一次，之后只认 hash_name。旧 image 列表已是
        派生文件 ID，不再次哈希。远端映射有效时允许本地原件已过期；失效后需要
        本地内容才能重传。旧派生件缺失时最多从原件重建一次，否则抛 MediaError。
        inline 强制内联；远端失效按明确的 file_id 标记判断，保留有效兄弟分片映射。
        """
        route = provider.route
        content = None
        if prompt.image:
            sources = [MediaRef(hash_name=value) for value in prompt.image]

        else:
            if not prompt.hash_name:
                raw = prompt.image_input
                if isinstance(raw, Path):
                    source = MediaRef("path", raw)

                elif isinstance(raw, bytes):
                    source = MediaRef("bytes", raw)

                else:
                    source = MediaRef("base64", raw)

                content = await self.content(source)
                try:
                    mime, width, height = await run_sync(inspect_image, content)
                    prompt.hash_name = await add_file_async(content, "IMAGE")

                except (OSError, ValueError) as exc:
                    raise MediaError("input", "验证图片或写入缓存失败") from exc

                prompt.image_input = None
                prompt.resource = MediaRef(hash_name=prompt.hash_name, mime_type=mime,
                                           width=width, height=height)

            resource = prompt.resource
            sources = [resource if resource and resource.hash_name == prompt.hash_name
                       else MediaRef(hash_name=prompt.hash_name)]

        files = None
        if route.files and provider.accepts_image_files() and not inline:
            disabled = await self._meta("get", "unsupported_files", route.scope)
            if not disabled and not self.degraded:
                files = get_files_provider(route.files.as_provider_settings(), route.files.provider_type)

        prepared = []
        try:
            for ref in sources:
                data = content if len(sources) == 1 else None
                if files is not None:
                    try:
                        prepared.append(await self.upload(ref, files, send, content=data))
                        continue

                    except (HTTPFailure, ProviderFailure, OSError, TimeoutError, httpx.RequestError) as exc:
                        failure = files.classify_failure(exc)
                        if failure.recovery not in ("inline_images", "disable_files"):
                            if isinstance(exc, ProviderFailure):
                                raise

                            raise ProviderFailure(failure) from exc

                        if failure.recovery == "disable_files":
                            await self._meta("put", "unsupported_files", route.scope,
                                             {"reason": failure.code}, time.time() + 3600)

                        files = None

                    except sqlite3.Error:
                        files = None

                data = data if data is not None else await self.content(ref)
                mime = ref.mime_type
                if not mime:
                    mime, width, height = await run_sync(inspect_image, data)
                    if ref.hash_name == prompt.hash_name:
                        prompt.resource = MediaRef(hash_name=ref.hash_name, mime_type=mime,
                                                   width=width, height=height)

                prepared.append(MediaRef("base64", base64.b64encode(data).decode(), mime,
                                         hash_name=ref.hash_name))

        except MediaError as exc:
            if prompt.image and not _rebuilt and isinstance(exc.__cause__, FileNotFoundError):
                if await prompt.load_image({}):
                    return await self.prepare_prompt(prompt, provider, send,
                                                     inline=inline, _rebuilt=True)

            raise

        return prepared
