from __future__ import annotations

import copy
import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx

from ..config_model import JsonValue, LLMConfig, Params
from ..gateway import HTTPFailure, WireRequest
from ..model import Context, ImagePrompt, MediaRef, Prompts


def merge_headers(*mappings: Mapping[str, str] | None) -> dict[str, str]:
    """按大小写不敏感的 HTTP 头名合并，后项覆盖前项。"""
    values = {}
    for mapping in mappings:
        for key, value in (mapping or {}).items():
            values[key.lower()] = (key, value)
    return dict(values.values())


type RecoveryAction = Literal[
    "none",
    "context_overflow",
    "rebuild_parent",
    "refresh_file",
    "inline_images",
    "disable_files",
]


@dataclass
class Failure:
    """Provider 对错误的解释与处理建议，不在此执行网络重发。

    safe 只描述执行状态；retryable 允许同候选原样重试，fallback_allowed
    允许放弃当前候选。recovery 是改变请求内容的专项恢复动作，各自有次数上限。
    Runtime 仍须依据取消、部分输出和真实 usage 否决生成重发。
    """

    category: str  # 诊断分类；Runtime 不再据此猜测重试/切换策略。
    code: str = ""
    resource: str | None = None
    retry_after: float | None = None
    remote_state: str = "unknown"
    raw: Params = field(default_factory=dict)
    retryable: bool = field(default=False, kw_only=True)  # 允许同候选原样重试。
    fallback_allowed: bool = field(default=False, kw_only=True)  # 允许切换模型候选。
    # inline_images 只改传输方式；disable_files 还会缓存 Files 不受支持的证据。
    recovery: RecoveryAction = field(default="none", kw_only=True)

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
    base_value = body.get("base_resp") if isinstance(body, dict) else None
    base = base_value if isinstance(base_value, dict) else {}
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
    raw_retry_after = (headers or {}).get("retry-after")
    if raw_retry_after is not None:
        try:
            retry_after = float(raw_retry_after)
            if not math.isfinite(retry_after) or retry_after < 0:
                retry_after = None

        except (TypeError, ValueError):
            pass

    # 共享默认策略由 Provider 层维护；各 Provider 可覆盖 failure()。
    retryable = category == "rate_limit"
    fallback_allowed = category in (
        "quota", "rate_limit", "authentication", "invalid_request",
        "parent_invalid", "file_invalid", "files_unsupported",
    )
    recovery: RecoveryAction = "none"
    if category == "context_overflow":
        recovery = "context_overflow"

    elif category == "parent_invalid":
        recovery = "rebuild_parent"

    elif category == "file_invalid":
        recovery = "refresh_file"

    elif category == "files_unsupported":
        recovery = "disable_files"

    return Failure(
        category,
        code,
        resource if isinstance(resource, str) else None,
        retry_after,
        remote,
        copy.deepcopy(body) if isinstance(body, dict) else {},
        retryable=retryable,
        fallback_allowed=fallback_allowed,
        recovery=recovery,
    )


def classify_transport_error(error: BaseException) -> Failure:
    """解释传输阶段的默认策略；未发送的连接失败允许重试，读写中断保持未知。"""
    if isinstance(error, asyncio.CancelledError):
        return Failure("cancelled")

    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
        return Failure(
            "connection",
            remote_state="rejected",
            retryable=True,
            fallback_allowed=True,
        )

    return Failure("transport")


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

    def classify_failure(self, error: BaseException) -> Failure:
        """统一解释 HTTP、流式业务错误或传输异常，给出处理建议。"""
        ...

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
        """按 error_dialect 解释响应和处理建议；子类可覆盖，不在此重发。"""
        dialect = self.route.provider_options.get("error_dialect", "openai")
        if not isinstance(dialect, str):
            dialect = "openai"

        return classify_error(
            status,
            body,
            dialect,
            headers,
        )

    def transport_failure(self, error: BaseException) -> Failure:
        """传输异常的默认建议；子类可单独禁止连接重试或候选切换。"""
        return classify_transport_error(error)

    def classify_failure(self, error: BaseException) -> Failure:
        """统一错误入口；已解码的 ProviderFailure 保留原 Provider 的结论。"""
        if isinstance(error, ProviderFailure):
            return error.failure

        if isinstance(error, HTTPFailure):
            return self.failure(error.status, error.body, error.headers)

        return self.transport_failure(error)

    def check_body(self, body: Params) -> None:
        """检查 HTTP 200 内嵌业务错误；失败抛 ProviderFailure。"""
        if body.get("error"):
            raise ProviderFailure(self.failure(200, body))

        base = body.get("base_resp")
        if isinstance(base, dict) and base.get("status_code", 0):
            raise ProviderFailure(self.failure(200, body))
