from __future__ import annotations

import copy
import asyncio
import math
import httpx

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ..config_model import JsonValue, LLMConfig, Params
from ..gateway import HTTPFailure, WireRequest
from ..model import Context, ImagePrompt, MediaRef, Prompts
from .error_codes import RecoveryAction, extract_error, match_error


def merge_headers(*mappings: Mapping[str, str] | None) -> dict[str, str]:
    """按大小写不敏感的 HTTP 头名合并，后项覆盖前项。"""
    values = {}
    for mapping in mappings:
        for key, value in (mapping or {}).items():
            values[key.lower()] = (key, value)

    return dict(values.values())


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
    """按独立码表解释供应商错误；不执行重试，不跨方言猜测业务码。

    精确码表包含独立的处理建议。未知业务码和远端执行状态保持保守，
    Provider 若有特殊响应结构可覆盖 failure()，无需给本函数增加分支。
    """
    code, error = extract_error(body, dialect)
    rule = match_error(status, code, dialect)
    resource = error.get(rule.resource_field, rule.resource_default)
    retry_after = None
    raw_retry_after = (headers or {}).get("retry-after")
    if raw_retry_after is not None:
        try:
            retry_after = float(raw_retry_after)
            if not math.isfinite(retry_after) or retry_after < 0:
                retry_after = None

        except (TypeError, ValueError):
            pass

    return Failure(
        rule.category,
        code,
        resource if isinstance(resource, str) else None,
        retry_after,
        rule.remote_state,
        copy.deepcopy(body) if isinstance(body, dict) else {},
        retryable=rule.retryable,
        fallback_allowed=rule.fallback_allowed,
        recovery=rule.recovery,
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


@dataclass(frozen=True, slots=True)
class StreamText:
    """本次事件新增的可读文本；不包含图片、签名或工具参数等原始载荷。"""

    text: str
    reasoning: bool = False


class StreamDecoder(Protocol):
    """流式对话解码器；partial 表示已观察到输出，之后禁止生成重发。"""

    partial: bool
    complete: bool
    usage: Params

    def feed(self, event: Params) -> list[StreamText] | None:
        """更新内容/usage，并返回本事件新增的日志文本；无文本时为空。"""
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
    error_dialect = "openai"

    def __init__(self, route: LLMConfig) -> None:
        """接收已解析的单候选配置，不读取全局配置，不管理上下文或网络连接。"""
        self.route = route

    @property
    def supports_parent(self) -> bool:
        # Explicit opt-in: compatibility endpoints may silently ignore store/parent.
        """仅在连接显式声明支持且未设置 store:false 时允许父 response 续接。"""
        options = self.route.provider_options.get("responses", {})
        return (
            isinstance(options, dict)
            and options.get("supports_previous_response_id", False) is True
            and self.route.model_params.get("store", True) is not False
        )

    def headers(self) -> dict[str, str]:
        """构造此模型连接的认证和覆盖头；Files 服务使用自己的凭证。"""
        headers = {"Authorization": f"Bearer {self.route.token}"}
        configured_headers = self.route.client_params.get("headers")
        client_headers = None
        if isinstance(configured_headers, dict):
            client_headers = {
                key: value
                for key, value in configured_headers.items()
                if isinstance(value, str)
            }

        return merge_headers(
            headers, self.route.headers, client_headers
        )

    def failure(
        self, status: int, body: Params, headers: Mapping[str, str] | None = None
    ) -> Failure:
        """按 error_dialect 解释响应和处理建议；子类可覆盖，不在此重发。"""
        dialect = self.route.provider_options.get("error_dialect", self.error_dialect)
        if not isinstance(dialect, str):
            dialect = self.error_dialect

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
        dialect = self.route.provider_options.get("error_dialect", self.error_dialect)
        dialect = dialect if isinstance(dialect, str) else self.error_dialect
        code, _ = extract_error(body, dialect)
        if body.get("error") or code:
            raise ProviderFailure(self.failure(200, body))
