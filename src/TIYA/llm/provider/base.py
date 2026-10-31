from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ..config_model import JsonValue, LLMConfig, Params
from ..gateway import WireRequest
from ..model import Context, ImagePrompt, MediaRef, Prompts


def merge_headers(*mappings: Mapping[str, str] | None) -> dict[str, str]:
    """按大小写不敏感的 HTTP 头名合并，后项覆盖前项。"""
    values = {}
    for mapping in mappings:
        for key, value in (mapping or {}).items():
            values[key.lower()] = (key, value)
    return dict(values.values())


@dataclass
class Failure:
    category: str
    code: str = ""
    resource: str | None = None
    retry_after: float | None = None
    remote_state: str = "unknown"
    raw: dict = field(default_factory=dict)

    @property
    def safe(self) -> bool:
        """仅确认远端拒绝执行才可恢复/重试；普通 5xx 和断流不代表安全。"""
        return self.remote_state == "rejected"


class ProviderFailure(Exception):
    def __init__(self, failure: Failure) -> None:
        """携带已归一化的供应商失败，保留原始证据供账本和恢复使用。"""
        super().__init__(f"{failure.category}: {failure.code}")
        self.failure = failure


def classify_error(
    status: int,
    body: JsonValue,
    dialect: str = "openai",
    headers: Mapping[str, str] | None = None,
) -> Failure:
    """将 HTTP/供应商错误转换为稳定分类；dialect 显式选择码表，不靠域名猜测。
    未知错误默认远端状态 unknown，不能因状态码相似就猜测请求未执行。
    """
    error = body.get("error", {}) if isinstance(body, dict) else {}
    error = error if isinstance(error, dict) else {"code": str(error)}
    code = str(error.get("code") or error.get("type") or "")
    base = body.get("base_resp", {}) if isinstance(body, dict) else {}
    if dialect == "minimax" and base.get("status_code"):
        code = str(base["status_code"])
    category, remote, resource = "unknown", "unknown", None
    quota = {
        "openai": {"insufficient_quota", "billing_hard_limit_reached"},
        "deepseek": set(),
        "glm": {"1113", "1112"},
        "minimax": {"1008", "1009"},
    }
    rate = {"openai": {"rate_limit_exceeded"}, "glm": {"1302"}, "minimax": {"1002"}}
    if code in quota.get(dialect, set()) or status == 402:
        category, remote = "quota", "rejected"
    elif code in rate.get(dialect, set()) or status == 429 and not code:
        category, remote = "rate_limit", "rejected"
    elif status in (401, 403):
        category, remote = "authentication", "rejected"
    elif code in (
        "previous_response_not_found",
        "invalid_previous_response_id",
        "previous_response_id_not_found",
    ):
        category, remote, resource = (
            "parent_invalid",
            "rejected",
            error.get("param", "previous_response_id"),
        )
    elif code in ("file_not_found", "invalid_file_id", "file_expired"):
        category, remote, resource = "file_invalid", "rejected", error.get("file_id")
    elif code in ("unsupported_file_reference", "file_reference_not_supported"):
        category, remote = "files_unsupported", "rejected"
    elif code in ("context_length_exceeded", "context_window_exceeded"):
        category, remote = "context_overflow", "rejected"
    elif status in (400, 404, 405, 422):
        category, remote = "invalid_request", "rejected"
    elif status in (500, 502, 503, 504) or code in ("1305", "overloaded_error"):
        category = (
            "unavailable"  # A 5xx alone does not prove generation did not execute.
        )
    elif code in ("rate_limit_error",):
        category, remote = "rate_limit", "rejected"
    retry_after = None
    try:
        retry_after = float((headers or {}).get("retry-after"))
    except (TypeError, ValueError):
        pass
    return Failure(
        category,
        code,
        resource,
        retry_after,
        remote,
        copy.deepcopy(body) if isinstance(body, dict) else {},
    )


@dataclass
class Result:
    """Provider 解析结果；图片仍是待落盘数据，由 Runtime/MediaService 缓存。"""

    prompts: Prompts = field(default_factory=Prompts)
    usage: Params = field(default_factory=dict)
    raw_usage: Params = field(default_factory=dict)
    native: list[Params] = field(default_factory=list)
    response_id: str = ""
    complete: bool = True
    images: list[tuple[ImagePrompt, Params]] = field(default_factory=list)
    request_id: str = ""
    reported_model: str = ""
    stored: bool | None = None


class StreamDecoder(Protocol):
    """流式对话解码器；partial 表示已观察到输出，之后禁止生成重发。"""

    partial: bool
    complete: bool
    usage: Params

    def feed(self, event: Params) -> None:
        """接收一个 JSON 事件，增量更新内容和 usage。"""
        ...

    def finish(self) -> Result:
        """返回聚合结果；必须保留是否完整的状态。"""
        ...


class BackgroundStreamDecoder(StreamDecoder, Protocol):
    """启用后台 Responses 时的解码器扩展，body 含已创建的远端任务 ID。"""

    body: Params


class ConversationProvider(Protocol):
    """Runtime 消费的对话契约；独立 Images/Files Provider 不属于此契约。"""

    route: LLMConfig

    @property
    def supports_parent(self) -> bool: ...

    def headers(self) -> dict[str, str]: ...

    def encode(self, context: Context, media: dict[str, list[MediaRef]]) -> Params:
        """把稳定上下文及已准备图片引用编码为原生请求体，不进行 IO。"""
        ...

    def decode(self, body: Params) -> Result:
        """解析完整响应；错误应转为 ProviderFailure，不修改业务上下文。"""
        ...

    def stream_decoder(self) -> StreamDecoder: ...

    def usage_result(self, raw: Params) -> Result: ...

    def failure(
        self, status: int, body: Params, headers: dict[str, str] | None = None
    ) -> Failure: ...

    def accepts_image_files(self) -> bool: ...

    def check_body(self, body: Params) -> None: ...

    def background_operation(
        self, response_id: str, cancel: bool = False
    ) -> WireRequest:
        """可选的后台控制扩展，仅显式启用 Responses 后台能力时调用。"""
        ...


class Provider:
    def __init__(self, route: LLMConfig) -> None:
        """接收已解析的单候选配置，不读取全局配置，不管理上下文或网络连接。"""
        self.route = route

    @property
    def supports_parent(self) -> bool:
        # Explicit opt-in: compatibility endpoints may silently ignore store/parent.
        """仅在连接显式声明支持且未设置 store:false 时允许父 response 续接。"""
        return (
            self.route.provider_options.get("responses", {}).get(
                "supports_previous_response_id", False
            )
            is True
            and self.route.model_params.get("store", True) is not False
        )

    def headers(self) -> dict[str, str]:
        """构造此模型连接的认证和覆盖头；Files 服务使用自己的凭证。"""
        headers = {"Authorization": f"Bearer {self.route.token}"}
        return merge_headers(
            headers, self.route.headers, self.route.client_params.get("headers")
        )

    def failure(
        self, status: int, body: Params, headers: Mapping[str, str] | None = None
    ) -> Failure:
        """按配置的 error_dialect 分类响应，不在 Provider 内重试。"""
        return classify_error(
            status,
            body,
            self.route.provider_options.get("error_dialect", "openai"),
            headers,
        )

    def check_body(self, body: Params) -> None:
        """检查 HTTP 200 内嵌业务错误；失败抛 ProviderFailure。"""
        if body.get("error") or body.get("base_resp", {}).get("status_code", 0):
            raise ProviderFailure(self.failure(200, body))
