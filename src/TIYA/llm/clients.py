"""Independent Images/Files entry points; neither requires Chat or a conversation."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import mimetypes
import sqlite3
import httpx

from collections.abc import Awaitable, Callable
from io import BytesIO
from PIL import Image
from typing import Any

from TIYA.config import DATA_DIR
from TIYA.utils import fingerprint, parse_proxies_to_httpx
from .config_model import LLMConfig, Params
from .gateway import Frame, Gateway
from .media import MediaError, MediaService, inspect_image
from .model import (
    ImageEditRequest,
    ImageGenerateRequest,
    ImageResponseItem,
    ImagesResponse,
    MediaRef,
)
from .pricing import PriceCatalog
from .provider import get_provider
from .provider.base import ProviderFailure
from .provider.files import FileMetadata, FilesProvider
from .settings import model_config_from_settings
from .state import StateStore
from .usage import Ledger
from .workers import run_sync

type PreviewCallback = Callable[[ImageResponseItem], Awaitable[None] | None]


def _inspect_edit_image(content: bytes, guide: str | None = None) -> tuple[str, int, int]:
    """在线程池内检查编辑输入；尺寸按 EXIF 方向读取，保留原始字节。

    markup 扫描 alpha 确认实际透明像素；普通图片只检查格式和尺寸。
    """
    with Image.open(BytesIO(content)) as image:
        mime = Image.MIME.get(image.format)
        width, height = image.size
        # PNG.getexif() 会尝试解码整图；只读取已有 EXIF，避免让普通 PNG 多解码一次。
        exif = Image.Exif()
        if image.info.get("exif"):
            exif.load(image.info["exif"])

        if exif.get(274) in (5, 6, 7, 8):
            width, height = height, width

        if guide and mime != "image/png":
            raise ValueError(f"{guide} 必须是带透明信息的 PNG")

        if guide == "markup":
            if "A" in image.getbands():
                alpha = image.getchannel("A")

            elif "transparency" in image.info:
                alpha = image.convert("RGBA").getchannel("A")

            else:
                raise ValueError("markup 必须包含实际透明像素")

            if alpha.getextrema()[0] == 255:
                raise ValueError("markup 必须包含实际透明像素")

        else:
            if guide == "mask" and "A" not in image.getbands():
                raise ValueError("mask 必须是带 alpha 通道的 PNG")

            image.verify()

    if mime is None:
        raise ValueError("未知图片格式")

    return mime, width, height


class ServiceClient:
    def __init__(
        self,
        settings: Params,
        *,
        client: httpx.AsyncClient | None = None,
        client_params: dict[str, Any] | None = None,
        ledger: Ledger | None = None,
        state: StateStore | None = None,
        prices: PriceCatalog | None = None,
    ) -> None:
        """复制单个连接节点；可借用 HTTP 客户端和账本，省略时延迟使用默认设施。
        只关闭自行创建的 HTTP 客户端；不关闭共享账本、状态库或 LLM 线程池。
        """
        self.settings = copy.deepcopy(settings)
        self._client, self._own_client = client, client is None
        self.client_params = copy.deepcopy(client_params or {})
        self.ledger = ledger
        self.state = state or StateStore(DATA_DIR / "llm" / "runtime.sqlite3")
        self.prices = prices or PriceCatalog(DATA_DIR / "llm" / "pricing.sqlite3")
        self.media = MediaService(self.state)

    async def _ensure_ledger(self) -> Ledger:
        """返回可用账本，未注入时延迟加载共享账本；初始化失败直接抛出。"""
        ledger = self.ledger
        if ledger is None:
            from .usage_compat import _GLOBAL_USAGE

            ledger = await run_sync(lambda: _GLOBAL_USAGE.ledger)
            self.ledger = ledger

        return ledger

    async def gateway(self) -> Gateway:
        """获取此服务的传输入口，HTTP 连接按需创建，不借用模型连接的认证头。"""
        if self._client is None:
            params = copy.deepcopy(self.client_params)
            proxies = params.pop("proxies", {})
            self._client = httpx.AsyncClient(
                mounts=parse_proxies_to_httpx(proxies), **params
            )
        return Gateway(self._client, download_options=self.client_params)

    async def close(self) -> None:
        """仅关闭本客户端拥有的 HTTP 连接；借用的客户端由调用方管理。"""
        if self._client is not None and self._own_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class FilesClient(ServiceClient):
    async def _operation(self, operation, wire, timeout):
        await self._ensure_ledger()
        call_id = await run_sync(self.ledger.begin_call, "")
        request_id = await run_sync(
            self.ledger.begin_request,
            call_id,
            {
                "operation": "files_" + operation,
                "scope": fingerprint(self.settings),
                "token_usage": "not_applicable",
            },
            {"algorithm": "not_applicable"},
        )
        try:
            gateway = await self.gateway()
            body = await (
                gateway.send_bytes(wire, timeout)
                if operation == "download"
                else gateway.send(wire, timeout)
            )
            if isinstance(body, dict) and (
                body.get("error") or body.get("base_resp", {}).get("status_code")
            ):
                raise ProviderFailure(
                    FilesProvider(self.settings).failure(200, body)
                )
            await run_sync(
                self.ledger.finish_request,
                request_id,
                "completed",
                remote_state="completed",
            )
            return body
        except BaseException:
            await run_sync(self.ledger.finish_request, request_id, "unknown")
            raise
        finally:
            await run_sync(self.ledger.finish_call, call_id, "finished")

    async def upload(
        self, image: MediaRef, *, filename: str = "image", timeout: float = 60
    ) -> FileMetadata:
        """上传文件并返回远端 ID/到期信息；timeout 为网络操作秒数，失败独立记账。"""
        provider = FilesProvider(self.settings)
        gateway = await self.gateway()
        data = await self.media.content(image, gateway)
        mime = image.mime_type
        if mime is None:
            try:
                mime = (await run_sync(inspect_image, data))[0]
            except (ValueError, OSError):
                mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return provider.decode(
            await self._operation(
                "upload", provider.upload_request(data, filename, mime), timeout
            )
        )

    async def retrieve(self, file_id: str, *, timeout: float = 60) -> FileMetadata:
        """查询远端文件元数据，不下载文件内容。"""
        provider = FilesProvider(self.settings)
        return provider.decode(
            await self._operation(
                "retrieve", provider.operation("retrieve", file_id), timeout
            )
        )

    async def list(
        self, *, params: Params | None = None, timeout: float = 60
    ) -> Params:
        """返回原生分页对象；params 按目标 Files API 填写，不自动遍历所有页。"""
        provider = FilesProvider(self.settings)
        return await self._operation(
            "list", provider.operation("list", params=params), timeout
        )

    async def delete(self, file_id: str, *, timeout: float = 60) -> Params:
        """删除远端文件并返回原生响应，不删除本地 file_cache 原件。"""
        provider = FilesProvider(self.settings)
        return await self._operation(
            "delete", provider.operation("delete", file_id), timeout
        )

    async def download(self, file_id: str, *, timeout: float = 60) -> bytes:
        """下载远端文件原始字节；是否可下载取决于服务和具体文件用途。"""
        provider = FilesProvider(self.settings)
        return await self._operation(
            "download", provider.operation("download", file_id), timeout
        )


class ImagesClient(ServiceClient):
    """Direct generation/edits with eager output caching and no generation retries."""

    def __init__(
        self,
        settings: Params,
        *,
        provider_ref: str = "image_generation:direct",
        client: httpx.AsyncClient | None = None,
        client_params: dict[str, object] | None = None,
        ledger: Ledger | None = None,
        state: StateStore | None = None,
        prices: PriceCatalog | None = None,
    ) -> None:
        """settings 为显式连接节点；provider_ref 为分类:连接名，用于选实现及定价身份。"""
        super().__init__(
            settings,
            client=client,
            client_params=client_params,
            ledger=ledger,
            state=state,
            prices=prices,
        )
        parts = provider_ref.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError("图片连接引用必须是 分类:连接名")
        self.provider_ref = provider_ref

    def _candidate(self, model: str) -> LLMConfig:
        """构造本次图片操作的独立连接快照，不读取全局预设。"""
        category, connection = self.provider_ref.split(":")
        candidate = model_config_from_settings(
            self.settings,
            provider_type=category,
            connection_name=connection,
            model=model,
        )
        candidate.client_params = copy.deepcopy(self.client_params)
        candidate.refresh_fingerprints()
        return candidate

    def pricing_identity(
        self, model: str, *, operation: str = "images_generate", preset: str = ""
    ) -> dict[str, str]:
        """取得定价目录所需绑定；生成/编辑分别使用 images_generate/images_edit。"""
        route = self._candidate(model)
        return {
            "preset": preset,
            "scope": route.scope,
            "model": model,
            "operation": operation,
        }

    async def generate(
        self,
        request: ImageGenerateRequest,
        *,
        timeout: float = 300,
        preset: str = "",
        on_preview: PreviewCallback | None = None,
    ) -> ImagesResponse:
        """执行一次独立生图，最终图片全部进入 file_cache 后返回；不自动重发生图。
        timeout 单位秒，preset 仅标识记账来源；on_preview 可同步或异步接收已缓存预览。
        """
        if isinstance(request, ImageEditRequest):
            raise TypeError("编辑请求请调用 edit")
        return await self._run(request, timeout, preset, on_preview)

    async def edit(
        self,
        request: ImageEditRequest,
        *,
        timeout: float = 300,
        preset: str = "",
        on_preview: PreviewCallback | None = None,
    ) -> ImagesResponse:
        """执行一次独立编辑；images 首张为目标、其余为参考，mask/markup 对应首张。
        mask 与 markup 互斥；本地 markup 验证 PNG、实际透明像素及可知尺寸。
        JSON 的远端引用不额外下载，无法在本地读取的内容由服务端验证。
        参数语义同 generate，编辑失败不会降级为文字生图或自动再次生成。
        """
        if not isinstance(request, ImageEditRequest):
            raise TypeError("需要 ImageEditRequest")
        return await self._run(request, timeout, preset, on_preview)

    async def _run(self, request, timeout, preset, on_preview):
        request = copy.deepcopy(request)
        await self._ensure_ledger()
        route = self._candidate(request.model)
        provider = get_provider(route)
        gateway = await self.gateway()
        images = None
        mask = None
        markup = None
        if isinstance(request, ImageEditRequest):
            provider.validate_edit_guides(request)
            if not request.images:
                raise ValueError("编辑请求需要目标图片")

            images = []
            multipart = provider.uses_multipart
            dimensions = []
            guide = request.mask or request.markup
            guide_name = "mask" if request.mask else "markup"
            inputs = [*request.images, *([guide] if guide else [])]
            for index, ref in enumerate(inputs):
                if not multipart and ref.kind in ("remote", "url"):
                    images.append(ref)
                    dimensions.append(
                        (ref.width, ref.height)
                        if ref.width is not None and ref.height is not None
                        else None
                    )
                    continue

                content = await self.media.content(ref, gateway)
                mime, width, height = await run_sync(
                    _inspect_edit_image,
                    content,
                    guide_name if guide and index == len(request.images) else None,
                )
                dimensions.append((width, height))
                images.append(
                    MediaRef("bytes", content, mime, width=width, height=height)
                )

            if guide:
                prepared_guide = images.pop()
                if (
                    dimensions[-1] is not None
                    and dimensions[0] is not None
                    and dimensions[-1] != dimensions[0]
                ):
                    raise ValueError(f"{guide_name} 尺寸必须与第一张图片一致")

                if request.mask:
                    mask = prepared_guide

                else:
                    markup = prepared_guide

            if not multipart:
                file_settings = self.settings.get("files_api")
                files = None
                if file_settings:
                    if not isinstance(file_settings, dict):
                        raise TypeError("files_api 需要连接配置字典")

                    files = FilesClient(
                        file_settings,
                        client=self._client,
                        ledger=self.ledger,
                        state=self.state,
                        prices=self.prices,
                    )

                async def prepare(ref: MediaRef) -> MediaRef:
                    if ref.kind != "bytes":
                        return ref

                    if files is not None:
                        uploaded = await files.upload(ref)
                        return MediaRef("remote", uploaded.file_id, ref.mime_type)

                    import base64

                    return MediaRef(
                        "base64", base64.b64encode(ref.value).decode(), ref.mime_type
                    )

                images = [await prepare(ref) for ref in images]
                mask = await prepare(mask) if mask else None
                markup = await prepare(markup) if markup else None

        wire = provider.encode_operation(request, images, mask, markup)
        operation = (
            "images_edit"
            if isinstance(request, ImageEditRequest)
            else "images_generate"
        )
        identity = {
            "preset": preset,
            "scope": route.scope,
            "model": request.model,
            "operation": operation,
        }
        call_id = await run_sync(self.ledger.begin_call, preset)
        try:
            snapshot = await run_sync(self.prices.snapshot, identity)
        except (sqlite3.Error, OSError, ValueError):
            snapshot = None
        request_id = await run_sync(
            self.ledger.begin_request,
            call_id,
            identity,
            {
                "algorithm": "image-request-v1",
                "images": request.n or 1,
                "completeness": "output_unknown",
            },
            price_snapshot=snapshot,
        )
        body, remote_complete = None, False
        try:
            async with asyncio.timeout(timeout):
                stream_value = (wire.json or wire.data or {}).get("stream")
                if stream_value is True or stream_value == "true":
                    decoder = provider.stream_decoder()
                    body = decoder.body
                    async for frame in gateway.stream(wire, timeout):
                        if isinstance(frame, Frame):
                            if not frame.data.strip():
                                continue
                            if frame.data == "[DONE]":
                                continue
                            event = json.loads(frame.data)
                        else:
                            event = frame
                        preview = decoder.feed(event)
                        body = decoder.body
                        remote_complete = decoder.complete
                        if preview and on_preview:
                            cached_preview = await self.media.materialize(
                                preview, gateway
                            )
                            callback = on_preview(cached_preview)
                            if inspect.isawaitable(callback):
                                await callback
                    body = decoder.finish()
                else:
                    body = provider.normalize_response(
                        await gateway.send(wire, timeout)
                    )
                    remote_complete = True
                await self._usage(request_id, body.get("usage") or {}, provider)
                await run_sync(
                    self.ledger.observe_usage,
                    request_id,
                    "reported",
                    {"images": len(body.get("data") or [])},
                    "final_images",
                )
                result = await self._materialize(body, request_id, gateway)
                await run_sync(
                    self.ledger.finish_request,
                    request_id,
                    "completed",
                    remote_state="completed",
                )
                return result
        except BaseException as exc:
            if body and body.get("usage"):
                await self._usage(request_id, body["usage"], provider)
            await run_sync(
                self.ledger.finish_request,
                request_id,
                "materialization_failed"
                if remote_complete
                else "cancelled"
                if isinstance(exc, asyncio.CancelledError)
                else "unknown",
                remote_state="completed" if remote_complete else "unknown",
            )
            if isinstance(exc, MediaError):
                exc.request_id = request_id
                exc.pending_response = body
            raise
        finally:
            await run_sync(self.ledger.finish_call, call_id, "finished")

    async def _usage(self, request_id, raw, provider):
        values = provider.usage_values(raw)
        await run_sync(
            self.ledger.observe_usage,
            request_id,
            "reported",
            values,
            fingerprint(raw),
            raw=raw,
        )

    async def _materialize(self, body, request_id, gateway):
        if not body.get("data"):
            raise MediaError("response", "缺少最终图片", request_id=request_id)
        items = [await self.media.materialize(item, gateway) for item in body["data"]]
        return ImagesResponse(
            created=body.get("created"),
            data=items,
            usage=body.get("usage") or {},
            request_id=request_id,
            output_format=body.get("output_format"),
            background=body.get("background"),
            quality=body.get("quality"),
            size=body.get("size"),
            safety_identifier=body.get("safety_identifier"),
            metadata={
                k: v for k, v in body.items()
                if k not in ("created", "data", "usage", "output_format", "background",
                             "quality", "size", "safety_identifier")
            },
        )

    async def recover_images(self, error: MediaError) -> ImagesResponse:
        """重试异常保存的图片响应，仅接收/落盘，不重新生成或重复记生成用量。

        使用原 request_id 和同一账本；再次失败时，新 MediaError 继续携带该响应，
        调用方可再次恢复。响应按引用复用，不深复制图片载荷。
        """
        if not isinstance(error, MediaError) or not error.pending_response:
            raise ValueError("该错误不包含可重试的图片响应")

        pending_response = error.pending_response
        ledger = await self._ensure_ledger()
        try:
            result = await self._materialize(
                pending_response, error.request_id, await self.gateway()
            )

        except MediaError as exc:
            exc.request_id = error.request_id
            exc.pending_response = pending_response
            raise

        await run_sync(
            ledger.finish_request,
            error.request_id,
            "completed",
            remote_state="completed",
        )
        return result
