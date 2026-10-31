"""Composable Files protocol; selected by explicit configuration, never connection name."""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from urllib.parse import quote

from ..config import credential, endpoint, fingerprint
from ..config_model import Params
from ..gateway import WireRequest
from .base import Failure, ProviderFailure, classify_error, merge_headers


@dataclass
class FileMetadata:
    file_id: str
    expires_at: float | None = None
    size: int | None = None
    raw: dict | None = None


class FilesProvider:
    def failure(
        self, status: int, body: Params, headers: dict[str, str] | None = None
    ) -> Failure:
        """按文件服务 dialect 分类错误；不在此尝试模型 fallback。"""
        return classify_error(
            status, body, self.settings.get("error_dialect", self.dialect), headers
        )

    def __init__(self, settings: Params) -> None:
        """读取独立 Files 连接及显式 dialect，不继承模型认证或猜测地址。"""
        self.settings = settings
        self.dialect = settings.get("dialect", "openai")
        if self.dialect not in ("openai", "deepseek", "anthropic"):
            raise ValueError(f"未知 Files dialect: {self.dialect}")
        self.endpoint = endpoint(settings, "files_api")
        self.scope = fingerprint(settings)

    def headers(self) -> dict[str, str]:
        """构造文件服务自己的认证头及需要的版本/beta 头。"""
        if self.dialect == "anthropic":
            headers = {
                "x-api-key": credential(self.settings),
                "anthropic-version": self.settings.get(
                    "anthropic_version", "2023-06-01"
                ),
            }
            if self.settings.get("anthropic_beta"):
                headers["anthropic-beta"] = self.settings["anthropic_beta"]
        else:
            headers = {"Authorization": f"Bearer {credential(self.settings)}"}
        return merge_headers(headers, self.settings.get("headers"))

    def upload_request(self, content: bytes, filename: str, mime: str) -> WireRequest:
        """编码 multipart 上传及供应商过期参数，认证头与模型连接隔离。"""
        data = dict(self.settings.get("upload_params") or {})
        if self.dialect != "anthropic":
            data.setdefault("purpose", self.settings.get("purpose", "user_data"))
            ttl = self.settings.get("expires_after_seconds")
            if ttl is not None:
                data.update(
                    {
                        "expires_after[anchor]": "created_at",
                        "expires_after[seconds]": str(ttl),
                    }
                )
        elif self.settings.get("expires_after_seconds") is not None:
            data["expires_in_seconds"] = str(self.settings["expires_after_seconds"])
        return WireRequest(
            self.endpoint,
            headers=self.headers(),
            data=data,
            files={"file": (filename, content, mime)},
            isolate_headers=True,
        )

    def operation(
        self, operation: str, file_id: str | None = None, params: Params | None = None
    ) -> WireRequest:
        """编码 retrieve/list/delete/download；list 参数透传，其他操作要求 file_id。"""
        if operation == "list":
            return WireRequest(
                self.endpoint,
                "GET",
                self.headers(),
                params=params,
                isolate_headers=True,
            )
        if not file_id:
            raise ValueError("需要 file_id")
        base = self.endpoint.rstrip("/") + "/" + quote(file_id, safe="")
        suffix = (
            self.settings.get("download_suffix", "/content")
            if operation == "download"
            else ""
        )
        return WireRequest(
            base + suffix,
            "DELETE" if operation == "delete" else "GET",
            self.headers(),
            isolate_headers=True,
        )

    def decode(self, body: Params) -> FileMetadata:
        """将各家文件 ID、大小和到期时间转为统一元数据；缺少 ID 则失败。"""
        if body.get("error") or body.get("base_resp", {}).get("status_code"):
            raise ProviderFailure(self.failure(200, body))
        body = body.get("file", body)
        if not isinstance(body.get("id", body.get("file_id")), str):
            raise ValueError("Files 响应缺少 file id")
        expiry = body.get("expires_at")
        if isinstance(expiry, str):
            expiry = datetime.datetime.fromisoformat(
                expiry.replace("Z", "+00:00")
            ).timestamp()
        if expiry is None and self.settings.get("expires_after_seconds") is not None:
            expiry = time.time() + self.settings["expires_after_seconds"]
        return FileMetadata(
            body.get("id", body.get("file_id")),
            expiry,
            body.get("bytes", body.get("size_bytes")),
            body,
        )


REGISTRY = {"files_api": FilesProvider}


def get_files_provider(settings: Params, category: str = "files_api") -> FilesProvider:
    """按显式分类注册表选择 Files 实现，连接名不参与选择。"""
    try:
        factory = REGISTRY[category]
    except KeyError:
        raise ValueError(f"未注册的 Files Provider 分类: {category}") from None
    return factory(settings)
