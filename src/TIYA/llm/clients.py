"""Independent Images/Files entry points; neither requires Chat or a conversation."""

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import mimetypes
from io import BytesIO

import httpx
from PIL import Image

from TIYA.config import DATA_DIR
from TIYA.utils import parse_proxies_to_httpx
from .config import fingerprint
from .settings import model_config_from_settings
from .gateway import Gateway, Frame
from .media import MediaService, MediaError, inspect_image
from .model import ImageEditRequest, ImagesResponse, MediaRef
from .pricing import PriceCatalog
from .provider.base import ProviderFailure, classify_error
from .provider.files import FilesProvider
from .provider import get_provider
import inspect
from .state import StateStore


class ServiceClient:
    def __init__(
        self,
        settings,
        *,
        client=None,
        client_params=None,
        ledger=None,
        state=None,
        prices=None,
    ):
        self.settings = copy.deepcopy(settings)
        self._client, self._own_client = client, client is None
        self.client_params = copy.deepcopy(client_params or {})
        self.ledger = ledger
        self.state = state or StateStore(DATA_DIR / "llm" / "runtime.sqlite3")
        self.prices = prices or PriceCatalog(DATA_DIR / "llm" / "pricing.sqlite3")
        self.media = MediaService(self.state)

    async def _ensure_ledger(self):
        if self.ledger is None:
            from .usage_compat import _GLOBAL_USAGE

            self.ledger = await asyncio.to_thread(lambda: _GLOBAL_USAGE.ledger)

    async def gateway(self):
        if self._client is None:
            params = copy.deepcopy(self.client_params)
            proxies = params.pop("proxies", {})
            self._client = httpx.AsyncClient(
                mounts=parse_proxies_to_httpx(proxies), **params
            )
        return Gateway(self._client, download_options=self.client_params)

    async def close(self):
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
        call_id = await asyncio.to_thread(self.ledger.begin_call, "")
        request_id = await asyncio.to_thread(
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
                    classify_error(
                        200, body, self.settings.get("error_dialect", "openai")
                    )
                )
            await asyncio.to_thread(
                self.ledger.finish_request,
                request_id,
                "completed",
                remote_state="completed",
            )
            return body
        except BaseException:
            await asyncio.to_thread(self.ledger.finish_request, request_id, "unknown")
            raise
        finally:
            await asyncio.to_thread(self.ledger.finish_call, call_id, "finished")

    async def upload(self, image: MediaRef, *, filename="image", timeout=60):
        provider = FilesProvider(self.settings)
        gateway = await self.gateway()
        data = await self.media.content(image, gateway)
        mime = image.mime_type
        if mime is None:
            try:
                mime = (await asyncio.to_thread(inspect_image, data))[0]
            except (ValueError, OSError):
                mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return provider.decode(
            await self._operation(
                "upload", provider.upload_request(data, filename, mime), timeout
            )
        )

    async def retrieve(self, file_id, *, timeout=60):
        provider = FilesProvider(self.settings)
        return provider.decode(
            await self._operation(
                "retrieve", provider.operation("retrieve", file_id), timeout
            )
        )

    async def list(self, *, params=None, timeout=60):
        provider = FilesProvider(self.settings)
        return await self._operation(
            "list", provider.operation("list", params=params), timeout
        )

    async def delete(self, file_id, *, timeout=60):
        provider = FilesProvider(self.settings)
        return await self._operation(
            "delete", provider.operation("delete", file_id), timeout
        )

    async def download(self, file_id, *, timeout=60):
        provider = FilesProvider(self.settings)
        return await self._operation(
            "download", provider.operation("download", file_id), timeout
        )


class ImagesClient(ServiceClient):
    """Direct generation/edits with eager output caching and no generation retries."""

    def __init__(self, settings, *, provider_ref="image_generation:direct", **kwargs):
        super().__init__(settings, **kwargs)
        parts = provider_ref.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError("图片连接引用必须是 分类:连接名")
        self.provider_ref = provider_ref

    def _candidate(self, model):
        category, connection = self.provider_ref.split(":")
        candidate = model_config_from_settings(
            self.settings,
            provider_type=category,
            connection_name=connection,
            model=model,
        )
        candidate.client_params = copy.deepcopy(self.client_params)
        return candidate

    def pricing_identity(self, model, *, operation="images_generate", preset=""):
        route = self._candidate(model)
        return {
            "preset": preset,
            "scope": route.scope,
            "model": model,
            "operation": operation,
        }

    async def generate(self, request, *, timeout=300, preset="", on_preview=None):
        if isinstance(request, ImageEditRequest):
            raise TypeError("编辑请求请调用 edit")
        return await self._run(request, timeout, preset, on_preview)

    async def edit(self, request, *, timeout=300, preset="", on_preview=None):
        if not isinstance(request, ImageEditRequest):
            raise TypeError("需要 ImageEditRequest")
        return await self._run(request, timeout, preset, on_preview)

    async def _run(self, request, timeout, preset, on_preview):
        request = copy.deepcopy(request)
        await self._ensure_ledger()
        route = self._candidate(request.model)
        provider = get_provider(route)
        gateway = await self.gateway()
        images, mask = None, None
        if isinstance(request, ImageEditRequest):
            if not request.images:
                raise ValueError("编辑请求需要目标图片")
            images = []
            multipart = provider.uses_multipart
            dimensions = []
            for ref in [*request.images, *([request.mask] if request.mask else [])]:
                if not multipart and ref.kind in ("remote", "url"):
                    images.append(ref)
                    dimensions.append(None)
                    continue
                content = await self.media.content(ref, gateway)
                mime, width, height = await asyncio.to_thread(inspect_image, content)
                dimensions.append((width, height))
                images.append(
                    MediaRef("bytes", content, mime, width=width, height=height)
                )
            if request.mask:
                mask = images.pop()
                if (
                    dimensions[-1] is not None
                    and dimensions[0] is not None
                    and dimensions[-1] != dimensions[0]
                ):
                    raise ValueError("mask 尺寸必须与第一张图片一致")
                if mask.kind == "bytes":
                    with Image.open(BytesIO(mask.value)) as image:
                        if mask.mime_type != "image/png" or "A" not in image.getbands():
                            raise ValueError("mask 必须是带 alpha 通道的 PNG")
            if not multipart:

                async def prepare(ref):
                    if ref.kind != "bytes":
                        return ref
                    if self.settings.get("files_api"):
                        files = FilesClient(
                            self.settings["files_api"],
                            client=self._client,
                            ledger=self.ledger,
                            state=self.state,
                        )
                        uploaded = await files.upload(ref)
                        return MediaRef("remote", uploaded.file_id, ref.mime_type)
                    import base64

                    return MediaRef(
                        "base64", base64.b64encode(ref.value).decode(), ref.mime_type
                    )

                images = [await prepare(ref) for ref in images]
                mask = await prepare(mask) if mask else None
        wire = provider.encode_operation(request, images, mask)
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
        call_id = await asyncio.to_thread(self.ledger.begin_call, preset)
        try:
            snapshot = await asyncio.to_thread(self.prices.snapshot, identity)
        except (sqlite3.Error, OSError, ValueError):
            snapshot = None
        request_id = await asyncio.to_thread(
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
                await asyncio.to_thread(
                    self.ledger.observe_usage,
                    request_id,
                    "reported",
                    {"images": len(body.get("data") or [])},
                    "final_images",
                )
                result = await self._materialize(body, request_id, gateway)
                await asyncio.to_thread(
                    self.ledger.finish_request,
                    request_id,
                    "completed",
                    remote_state="completed",
                )
                return result
        except BaseException as exc:
            if body and body.get("usage"):
                await self._usage(request_id, body["usage"], provider)
            await asyncio.to_thread(
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
            await asyncio.to_thread(self.ledger.finish_call, call_id, "finished")

    async def _usage(self, request_id, raw, provider):
        values = provider.usage_values(raw)
        await asyncio.to_thread(
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
            body.get("created"),
            items,
            body.get("usage") or {},
            request_id,
            {k: v for k, v in body.items() if k not in ("data", "usage")},
        )

    async def recover_images(self, error):
        """Retry download/cache only. Never calls an Images generation endpoint."""
        if not isinstance(error, MediaError) or not getattr(
            error, "pending_response", None
        ):
            raise ValueError("该错误不包含可重试的图片响应")
        result = await self._materialize(
            error.pending_response, error.request_id, await self.gateway()
        )
        await asyncio.to_thread(
            self.ledger.finish_request,
            error.request_id,
            "completed",
            remote_state="completed",
        )
        return result
