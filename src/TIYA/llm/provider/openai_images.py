from __future__ import annotations

import copy
from dataclasses import fields

from TIYA.utils import fingerprint

from ..gateway import WireRequest
from ..model import ImageEditRequest, ImageGenerateRequest, MediaRef
from .base import Provider
from .common import image_url


class Images(Provider):
    @property
    def uses_multipart(self) -> bool:
        return (
            self.route.provider_options.get("images", {}).get(
                "edit_format", "multipart"
            )
            == "multipart"
        )

    def normalize_response(self, body):
        self.check_body(body)
        return body

    def stream_decoder(self):
        return ImagesStream(self)

    @staticmethod
    def usage_values(raw):
        values = {
            target: raw[source]
            for source, target in (
                ("input_tokens", "input"),
                ("output_tokens", "output"),
                ("total_tokens", "total"),
            )
            if raw.get(source) is not None
        }
        for direction in ("input", "output"):
            for source, target in (("text_tokens", "text"), ("image_tokens", "image")):
                value = raw.get(direction + "_tokens_details", {}).get(source)
                if value is not None:
                    values[direction + "_" + target] = value

        return values

    def validate_edit_guides(
        self,
        request: ImageEditRequest,
        mask: MediaRef | None = None,
        markup: MediaRef | None = None,
    ) -> None:
        """校验最终有效的引导参数；客户端在读取/上传图片前调用。"""
        raw_mask = request.params.get("mask", self.route.model_params.get("mask"))
        raw_markup = request.params.get("markup", self.route.model_params.get("markup"))
        has_mask = mask is not None or request.mask is not None or raw_mask is not None
        has_markup = markup is not None or request.markup is not None or raw_markup is not None
        if has_mask and has_markup:
            raise ValueError("mask 与 markup 不能同时提供")

    def encode_operation(
        self,
        request: ImageGenerateRequest,
        images: list[MediaRef] | None = None,
        mask: MediaRef | None = None,
        markup: MediaRef | None = None,
    ) -> WireRequest:
        """编码独立生成/编辑请求；images 顺序不变，mask/markup 只对应首张目标图。

        images、mask 和 markup 为客户端准备后的引用，不负责读取本地文件。
        """
        params = copy.deepcopy(self.route.model_params)
        params.update(request.params)
        for field in fields(ImageGenerateRequest):
            if field.name == "params":
                continue

            value = getattr(request, field.name)
            if value is not None:
                params[field.name] = value

        if not isinstance(request, ImageEditRequest):
            return WireRequest(
                self.route.endpoint,
                headers=self.headers(),
                json=params,
                isolate_headers=True,
            )

        if not images:
            raise ValueError("编辑请求需要至少一张目标图片")

        self.validate_edit_guides(request, mask, markup)
        url = self.route.provider_options.get("edit_endpoint")
        if not url:
            base = self.route.provider_options.get("base_url")
            if not base:
                raise ValueError("编辑需要显式 edit_endpoint 或 base_url")

            url = base.rstrip("/") + "/images/edits"

        if (
            self.route.provider_options.get("images", {}).get(
                "edit_format", "multipart"
            )
            == "json"
        ):
            def reference(ref: MediaRef) -> dict[str, str]:
                if ref.kind == "remote":
                    if not isinstance(ref.value, str):
                        raise ValueError("file_id 必须是字符串")

                    return {"file_id": ref.value}

                return {"image_url": image_url(ref)}

            params["images"] = [reference(ref) for ref in images]
            if mask:
                params["mask"] = reference(mask)

            if markup:
                params["markup"] = reference(markup)

            return WireRequest(
                url, headers=self.headers(), json=params, isolate_headers=True
            )

        files = []
        for index, ref in enumerate(images):
            if ref.kind != "bytes":
                raise ValueError("multipart 图片尚未物化")

            files.append(
                (
                    "image[]",
                    (
                        f"image_{index}.{ref.mime_type.split('/')[-1]}",
                        ref.value,
                        ref.mime_type,
                    ),
                )
            )

        if mask:
            if mask.kind != "bytes":
                raise ValueError("multipart mask 尚未物化")

            params.pop("mask", None)
            files.append(("mask", ("mask.png", mask.value, mask.mime_type)))

        if markup:
            if markup.kind != "bytes":
                raise ValueError("multipart markup 尚未物化")

            params.pop("markup", None)
            files.append(("markup", ("markup.png", markup.value, markup.mime_type)))

        data = {
            k: str(v).lower() if isinstance(v, bool) else str(v)
            for k, v in params.items()
        }
        return WireRequest(
            url, headers=self.headers(), files=files, data=data, isolate_headers=True
        )


class ImagesStream:
    def __init__(self, provider):
        self.provider = provider
        self.body = {"data": [], "usage": {}}
        self.seen = set()
        self.complete = False

    def feed(self, event):
        self.provider.check_body(event)
        self.body["usage"].update(event.get("usage") or {})
        kind = event.get("type", "")
        if kind in ("image_generation.completed", "image_edit.completed"):
            self.complete = True
            key = fingerprint(
                [
                    event.get("image_index", 0),
                    event.get("b64_json"),
                    event.get("url"),
                    event.get("data"),
                ]
            )
            if key not in self.seen:
                self.seen.add(key)
                self.body["data"].extend(event.get("data") or [event])

            if "created_at" in event:
                self.body["created"] = event["created_at"]

            for field in (
                "output_format",
                "background",
                "quality",
                "size",
                "safety_identifier",
            ):
                if field in event:
                    self.body[field] = event[field]

        elif kind in ("image_generation.partial_image", "image_edit.partial_image"):
            return event

        elif "data" in event:
            self.body = self.provider.normalize_response(event)
            self.complete = True

        return None

    def finish(self):
        return self.body
