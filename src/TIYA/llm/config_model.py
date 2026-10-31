"""Typed Chat configuration. Preset data and live communication state stay outside it."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from TIYA.logger import Logger, BlockHandle
    from .usage_compat import ApiPrice

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
type Params = dict[str, JsonValue]
type HistoryMode = Literal["auto", "full", "previous_response_id"]


@dataclass(slots=True)
class FilesConfig:
    """Independent file service; its credentials never inherit from the model."""

    provider_type: str  # 文件协议注册分类，例如 files_api。
    connection_name: str  # 配置树内的连接名，不用于推断供应商。
    endpoint: str  # 已处理 base_url 后的完整地址。
    token: str = field(repr=False)
    headers: dict[str, str] = field(default_factory=dict)
    client_params: Params = field(default_factory=dict)
    upload_params: Params = field(default_factory=dict)  # 明确的上传表单参数。
    # dialect、远端 TTL 等协议选项，不直接展开进请求体。
    provider_options: Params = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.provider_type}:{self.connection_name}"

    def as_provider_settings(self) -> Params:
        """Serialize only at the Files protocol boundary, including its open options."""
        result = copy.deepcopy(self.provider_options)
        result.update(endpoint=self.endpoint, key=self.token)
        if self.headers:
            result["headers"] = copy.deepcopy(self.headers)

        if self.client_params:
            result["client_params"] = copy.deepcopy(self.client_params)

        if self.upload_params:
            result["upload_params"] = copy.deepcopy(self.upload_params)

        return result


@dataclass(slots=True)
class LLMConfig:
    """One executable candidate. List position, never its name, decides priority."""

    provider_type: str  # 正向选择 Provider 实现，允许注册自定义分类。
    connection_name: str  # 配置树内的连接名；不另建 Provider ID。
    endpoint: str  # 主操作的完整地址；独立生图的编辑地址放在协议选项中。
    token: str = field(repr=False)
    model: str  # 服务端模型 ID，包含冒号时也原样保留。
    # 已合并默认值/覆盖值的原生请求参数。
    model_params: Params = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    # HTTP 客户端选项，不展开到请求体。
    client_params: Params = field(default_factory=dict)
    max_context: int | None = None
    context_history_mode: HistoryMode = "auto"
    # 仅声明关联文件服务，不代表该模型支持图片 file_id。
    files: FilesConfig | None = None
    # 保留开放连接选项，由 Provider 消费。
    provider_options: Params = field(default_factory=dict)
    max_retries: int | None = None  # None uses the enclosing RequestPolicy.
    legacy_price: ApiPrice | None = None  # Only the existing single-model price input.

    @property
    def ref(self) -> str:
        return f"{self.provider_type}:{self.connection_name}:{self.model}"

    @property
    def scope(self) -> str:
        from .config import fingerprint

        # Preserve the existing account/model scope so parent checkpoints remain usable.
        return fingerprint(
            [
                self.provider_type,
                self.connection_name,
                self.model,
                self.endpoint or None,
                self.token,
                self.headers or None,
                {
                    key: self.client_params.get(key)
                    for key in ("headers", "auth", "cookies", "cert")
                },
            ]
        )

    @property
    def revision(self) -> str:
        from .config import fingerprint

        return fingerprint(
            [
                self.ref,
                self.endpoint,
                self.token,
                self.headers,
                self.model_params,
                self.client_params,
                self.max_context,
                self.context_history_mode,
                self.files.as_provider_settings() if self.files else None,
                self.files.ref if self.files else None,
                self.provider_options,
                self.max_retries,
                self.legacy_price.to_dict() if self.legacy_price else None,
            ]
        )


@dataclass(slots=True)
class RequestPolicy:
    """
    没有什么复杂的，就三件事
    1. 总重试次数限制
    2. 每个预设重试限制
    3. fallback 回归主预设的 CD
    """

    max_attempts: int = 6
    max_retries_per_candidate: int = 2
    fallback_cooldown_seconds: float = 900

    def __post_init__(self):
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts 必须是正整数")

        if (
            isinstance(self.max_retries_per_candidate, bool)
            or not isinstance(self.max_retries_per_candidate, int)
            or self.max_retries_per_candidate < 0
        ):
            raise ValueError("max_retries_per_candidate 必须是非负整数")

        if (
            isinstance(self.fallback_cooldown_seconds, bool)
            or not isinstance(self.fallback_cooldown_seconds, (int, float))
            or not math.isfinite(self.fallback_cooldown_seconds)
            or self.fallback_cooldown_seconds < 0
        ):
            raise ValueError("fallback_cooldown_seconds 不能为负数")


@dataclass(slots=True, init=False)
class ChatConfig:
    """Chat input with explicit replacement/update semantics.

    Nonempty candidates replaces the whole model configuration. None supplies only
    session options and/or the historical model_params update. [] is always invalid.
    Effective Chat configuration always has candidates; updates are consumed at entry.
    Flat properties are views of candidates[0], not duplicate storage.
    """

    candidates: list[LLMConfig] | None  # 非空列表替换；None 局部更新；[] 报错。
    preset_name: str | None  # 来源预设名，用于记账和诊断；直接构造时可以没有。
    auto_compress_rate: float | None  # 上游压缩水位比例；完整替换时不继承旧值。
    request_policy: RequestPolicy | None  # 完整配置未指定时由 Chat 安装默认策略。
    system_prompt: str | None
    keep_alive: bool | None
    logger: Logger | BlockHandle | None
    log_off: bool | None
    _model_params_update: Params | None  # Used only while candidates is None.

    def __init__(
        self,
        candidates: list[LLMConfig] | None = None,
        *,
        preset_name: str | None = None,
        auto_compress_rate: float | None = None,
        request_policy: RequestPolicy | None = None,
        system_prompt: str | None = None,
        keep_alive: bool | None = None,
        logger: Logger | BlockHandle | None = None,
        log_off: bool | None = None,
        model_params: Params | None = None,
    ):
        if candidates is not None and (
            not candidates or not all(isinstance(c, LLMConfig) for c in candidates)
        ):
            raise ValueError(
                "candidates 必须是非空 LLMConfig 列表，局部更新请使用 None"
            )
        self.candidates = copy.deepcopy(candidates)
        self.preset_name = preset_name
        self.auto_compress_rate = auto_compress_rate
        self.request_policy = copy.deepcopy(request_policy)
        self.system_prompt, self.keep_alive = system_prompt, keep_alive
        self.logger, self.log_off = logger, log_off
        self._model_params_update = None
        if model_params is not None:
            self.model_params = model_params

    def copy(self) -> ChatConfig:
        """Isolate mutable configuration while keeping the business logger by reference."""
        return ChatConfig(
            self.candidates,
            preset_name=self.preset_name,
            auto_compress_rate=self.auto_compress_rate,
            request_policy=self.request_policy,
            system_prompt=self.system_prompt,
            keep_alive=self.keep_alive,
            logger=self.logger,
            log_off=self.log_off,
            model_params=self._model_params_update if self.candidates is None else None,
        )

    @property
    def primary(self) -> LLMConfig:
        if not self.candidates:
            raise ValueError("局部配置更新没有首选候选")
        return self.candidates[0]

    @property
    def revision(self) -> str:
        from .config import fingerprint

        return fingerprint(
            [
                self.preset_name,
                [c.revision for c in self.candidates or []],
                self.auto_compress_rate,
                vars_policy(self.request_policy or RequestPolicy()),
            ]
        )

    @property
    def model_params(self) -> Params | None:
        return (
            self.primary.model_params
            if self.candidates is not None
            else self._model_params_update
        )

    @model_params.setter
    def model_params(self, value: Params | None):
        if self.candidates is not None:
            self.primary.model_params = copy.deepcopy(value or {})
        else:
            self._model_params_update = copy.deepcopy(value)

    @property
    def model(self) -> str:
        return self.primary.model

    @model.setter
    def model(self, value: str):
        self.primary.model = value

    @property
    def endpoint(self) -> str:
        return self.primary.endpoint

    @endpoint.setter
    def endpoint(self, value: str):
        if value != self.primary.endpoint:
            self.primary.files = (
                None  # A new account must not inherit the old Files binding.
            )
        self.primary.endpoint = value

    @property
    def token(self) -> str:
        return self.primary.token

    @token.setter
    def token(self, value: str):
        if value != self.primary.token:
            self.primary.files = None
        self.primary.token = value

    @property
    def api_provider(self) -> str:
        return self.primary.provider_type

    @api_provider.setter
    def api_provider(self, value: str):
        if value != self.primary.provider_type:
            self.primary.files = None
        self.primary.provider_type = value

    @property
    def max_context(self) -> int | None:
        return self.primary.max_context

    @max_context.setter
    def max_context(self, value: int | None):
        self.primary.max_context = value

    @property
    def client_params(self) -> Params:
        return self.primary.client_params

    @client_params.setter
    def client_params(self, value: Params):
        self.primary.client_params = copy.deepcopy(value)

    @property
    def price(self) -> ApiPrice | None:
        """Compatibility read for old configured prices; new pricing lives in its catalog."""
        return self.primary.legacy_price


def vars_policy(policy: RequestPolicy) -> Params:
    return {
        "max_attempts": policy.max_attempts,
        "max_retries_per_candidate": policy.max_retries_per_candidate,
        "fallback_cooldown_seconds": policy.fallback_cooldown_seconds,
    }
