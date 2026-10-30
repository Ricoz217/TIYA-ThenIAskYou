"""Composable Files protocol; selected by explicit configuration, never connection name."""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from urllib.parse import quote

from ..config import endpoint, credential, fingerprint
from ..gateway import WireRequest
from .base import ProviderFailure, classify_error, merge_headers


@dataclass
class FileMetadata:
    file_id: str
    expires_at: float | None = None
    size: int | None = None
    raw: dict | None = None


class FilesProvider:
    def failure(self, status, body, headers=None):
        return classify_error(
            status, body, self.settings.get("error_dialect", self.dialect), headers
        )

    def __init__(self, settings):
        self.settings = settings
        self.dialect = settings.get("dialect", "openai")
        if self.dialect not in ("openai", "deepseek", "anthropic"):
            raise ValueError(f"未知 Files dialect: {self.dialect}")
        self.endpoint = endpoint(settings, "files_api")
        self.scope = fingerprint(settings)

    def headers(self):
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

    def upload_request(self, content, filename, mime):
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

    def operation(self, operation, file_id=None, params=None):
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

    def decode(self, body):
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


def get_files_provider(settings, category="files_api"):
    try:
        factory = REGISTRY[category]
    except KeyError:
        raise ValueError(f"未注册的 Files Provider 分类: {category}") from None
    return factory(settings)
