"""HTTP transport and SSE framing. No model semantics and no automatic retries."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx

from TIYA.utils import parse_proxies_to_httpx

from .config_model import JsonValue, Params


@dataclass
class Frame:
    event: str
    data: str
    id: str = ""


async def frames(lines: AsyncIterable[str]) -> AsyncIterator[Frame]:
    """按 SSE 空行边界组帧，保留多行 data；不解释模型事件。"""
    event, data, identifier = "message", [], ""
    async for line in lines:
        if line == "":
            if data:
                yield Frame(event, "\n".join(data), identifier)
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if key == "data":
            data.append(value)
        elif key == "event":
            event = value
        elif key == "id":
            identifier = value
    if data:
        yield Frame(event, "\n".join(data), identifier)


@dataclass
class WireRequest:
    """一次已编码请求；isolate_headers 禁止继承借用客户端的默认认证头。"""

    url: str
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    json: JsonValue = None
    data: Params | None = None
    files: (
        dict[str, tuple[str, bytes, str]]
        | list[tuple[str, tuple[str, bytes, str]]]
        | None
    ) = None
    params: Params | None = None
    isolate_headers: bool = False


class HTTPFailure(Exception):
    def __init__(
        self, status: int, body: Params, headers: dict[str, str] | None = None
    ) -> None:
        """保留 HTTP 状态、响应体和请求头，供 Provider 分类；不在此判断可否重试。"""
        super().__init__(f"HTTP {status}")
        self.status, self.body, self.headers = status, body, dict(headers or {})


class Gateway:
    def __init__(
        self,
        client: httpx.AsyncClient,
        download_options: dict[str, object] | None = None,
    ) -> None:
        """借用调用方的 HTTP 客户端；download_options 只保留下载所需网络选项。"""
        self.client = client
        self.last_request_id = ""
        self.download_options = {
            k: v
            for k, v in (download_options or {}).items()
            if k
            in ("proxy", "proxies", "verify", "cert", "http2", "trust_env", "limits")
        }

    def _remember(self, response):
        headers = getattr(response, "headers", {})
        self.last_request_id = str(
            headers.get("x-request-id", headers.get("request-id", ""))
        )[:1024]

    def _kwargs(self, request, timeout):
        result = {"timeout": timeout}
        defaults = getattr(self.client, "headers", {})
        if request.headers and any(
            defaults.get(k) != v for k, v in request.headers.items()
        ):
            result["headers"] = request.headers
        if request.json is not None:
            result["json"] = request.json
        if request.data is not None:
            result["data"] = request.data
        if request.files is not None:
            result["files"] = request.files
        if request.params is not None:
            result["params"] = request.params
        return result

    async def send(self, request: WireRequest, timeout: float) -> Params:
        """发送一次 JSON 请求（timeout 单位秒），HTTP 失败抛 HTTPFailure；不自动重试。"""
        kwargs = self._kwargs(request, timeout)
        if request.isolate_headers:
            prepared = self._isolated_request(request, timeout)
            response = await self.client.send(
                prepared, auth=None, follow_redirects=False
            )
        elif request.method == "POST":
            response = await self.client.post(url=request.url, **kwargs)
        else:
            response = await self.client.request(request.method, request.url, **kwargs)
        self._remember(response)
        if response.status_code == 204:
            return {}
        try:
            body = response.json()
        except ValueError as exc:
            if response.status_code < 300:
                raise ValueError("HTTP 成功响应不是有效 JSON") from exc
            body = {"transport_error": "non_json_response"}
        if response.status_code >= 300:
            raise HTTPFailure(
                response.status_code, body, getattr(response, "headers", {})
            )
        return body

    async def send_bytes(self, request: WireRequest, timeout: float) -> bytes:
        """发送文件下载请求，返回原始字节；HTTP 错误交由调用方处理。"""
        if request.isolate_headers:
            response = await self.client.send(
                self._isolated_request(request, timeout),
                auth=None,
                follow_redirects=False,
            )
        else:
            response = await self.client.request(
                request.method, request.url, **self._kwargs(request, timeout)
            )
        response.raise_for_status()
        return response.content

    @staticmethod
    def _isolated_request(request, timeout):
        return httpx.Request(
            request.method,
            request.url,
            headers=request.headers,
            json=request.json,
            data=request.data,
            files=request.files,
            params=request.params,
            extensions={"timeout": httpx.Timeout(timeout).as_dict()},
        )

    @asynccontextmanager
    async def _response_context(self, request, timeout):
        if request.isolate_headers:
            response = await self.client.send(
                self._isolated_request(request, timeout),
                stream=True,
                auth=None,
                follow_redirects=False,
            )
            try:
                yield response
            finally:
                await response.aclose()
        else:
            async with self.client.stream(
                request.method, request.url, **self._kwargs(request, timeout)
            ) as response:
                yield response

    async def stream(
        self, request: WireRequest, timeout: float
    ) -> AsyncIterator[Frame | Params]:
        """逐帧返回 SSE；服务端退回 application/json 时返回一个完整对象。"""
        async with self._response_context(request, timeout) as response:
            self._remember(response)
            if response.status_code >= 300:
                await response.aread()
                try:
                    body = response.json()
                except ValueError:
                    body = {"transport_error": "non_json_response"}
                raise HTTPFailure(response.status_code, body, response.headers)
            if "application/json" in response.headers.get("content-type", ""):
                await response.aread()
                yield response.json()
                return
            async for frame in frames(response.aiter_lines()):
                yield frame

    async def download(self, url: str, timeout: float = 60) -> bytes:
        # A signed image URL must not receive model credentials/default headers.
        """用独立客户端读取临时图片 URL，不携带模型认证头或 cookies。"""
        options = self.download_options.copy()
        proxies = options.pop("proxies", {})
        if proxies:
            options["mounts"] = parse_proxies_to_httpx(proxies)
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=timeout, **options
        ) as download_client:
            response = await download_client.get(url)
            response.raise_for_status()
            return response.content
