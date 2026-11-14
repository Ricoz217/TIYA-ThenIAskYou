"""供应商错误码表。连接通过 error_dialect 选择，不按域名或名称推测。

精确业务码优先于 HTTP 通用规则；不同方言的业务码不互相继承。
这里维护原有已识别码，不代表覆盖供应商全部错误。新格式可新增方言；
无法用固定字段/码表表达的协议，由 Provider 覆盖 failure()。
"""

from dataclasses import dataclass, field
from typing import Literal

from ..config_model import JsonValue, Params

type RecoveryAction = Literal[
    "none", "context_overflow", "rebuild_parent", "refresh_file",
    "inline_images", "disable_files",
]


@dataclass(frozen=True, slots=True)
class ErrorRule:
    """一条匹配结果；远端状态、重试、切换和专项恢复分别声明。"""

    category: str
    remote_state: str = "rejected"
    retryable: bool = False
    fallback_allowed: bool = False
    recovery: RecoveryAction = "none"
    resource_field: str = ""
    resource_default: str | None = None


@dataclass(frozen=True, slots=True)
class ErrorDialect:
    """错误字段、业务码及明确的 HTTP 覆盖规则；空/成功码不当作错误。"""

    codes: dict[str, ErrorRule] = field(default_factory=dict)
    error_path: tuple[str, ...] = ("error",)
    code_fields: tuple[str, ...] = ("code", "type")
    success_codes: tuple[str, ...] = ("", "0")
    http_rules: dict[int, ErrorRule] = field(default_factory=dict)


UNKNOWN = ErrorRule("unknown", remote_state="unknown")
QUOTA = ErrorRule("quota", fallback_allowed=True)
RATE_LIMIT = ErrorRule("rate_limit", retryable=True, fallback_allowed=True)
AUTHENTICATION = ErrorRule("authentication", fallback_allowed=True)
# 非法消息/参数应由上层修正或回滚；换模型通常仍会携带同一份非法上下文。
INVALID_REQUEST = ErrorRule("invalid_request")
UNAVAILABLE = ErrorRule("unavailable", remote_state="unknown")
CONTEXT_OVERFLOW = ErrorRule("context_overflow", recovery="context_overflow")
PARENT_INVALID = ErrorRule(
    "parent_invalid", fallback_allowed=True, recovery="rebuild_parent",
    resource_field="param", resource_default="previous_response_id",
)
FILE_INVALID = ErrorRule(
    "file_invalid", fallback_allowed=True, recovery="refresh_file",
    resource_field="file_id",
)
FILES_UNSUPPORTED = ErrorRule(
    "files_unsupported", fallback_allowed=True, recovery="disable_files",
)

OPENAI_CODES = {
    "insufficient_quota": QUOTA,
    "billing_hard_limit_reached": QUOTA,
    "rate_limit_exceeded": RATE_LIMIT,
    "previous_response_not_found": PARENT_INVALID,
    "invalid_previous_response_id": PARENT_INVALID,
    "previous_response_id_not_found": PARENT_INVALID,
    "file_not_found": FILE_INVALID,
    "invalid_file_id": FILE_INVALID,
    "file_expired": FILE_INVALID,
    "unsupported_file_reference": FILES_UNSUPPORTED,
    "file_reference_not_supported": FILES_UNSUPPORTED,
    "context_length_exceeded": CONTEXT_OVERFLOW,
    "context_window_exceeded": CONTEXT_OVERFLOW,
}

DIALECTS: dict[str, ErrorDialect] = {
    "openai": ErrorDialect(OPENAI_CODES),
    # 官方错误表按 HTTP 状态定义语义，429 包含请求速率/并发限制。
    # 其余已由 HTTP_RULES 覆盖，500/503 不据此推断远端尚未执行。
    # https://api-docs.deepseek.com/zh-cn/quick_start/error_codes/
    "deepseek": ErrorDialect(
        {"context_length_exceeded": CONTEXT_OVERFLOW},
        http_rules={429: RATE_LIMIT},
    ),
    "anthropic": ErrorDialect({
        "rate_limit_error": RATE_LIMIT,
        "overloaded_error": UNAVAILABLE,
        "authentication_error": AUTHENTICATION,
        "permission_error": AUTHENTICATION,
        "invalid_request_error": INVALID_REQUEST,
    }),
    "glm": ErrorDialect({
        "1113": QUOTA, "1112": QUOTA,
        "1302": RATE_LIMIT, "1305": UNAVAILABLE,
    }),
    "minimax": ErrorDialect(
        {"1008": QUOTA, "1009": QUOTA, "1002": RATE_LIMIT},
        error_path=("base_resp",), code_fields=("status_code",),
    ),
}

# 仅 HTTP 通用语义放这里。方言未明确覆盖时，带未知业务码的 429 不判定为限流。
HTTP_RULES: dict[int, ErrorRule] = {
    400: INVALID_REQUEST, 401: AUTHENTICATION, 402: QUOTA,
    403: AUTHENTICATION, 404: INVALID_REQUEST, 405: INVALID_REQUEST,
    422: INVALID_REQUEST, 429: RATE_LIMIT,
    500: UNAVAILABLE, 502: UNAVAILABLE, 503: UNAVAILABLE, 504: UNAVAILABLE,
}


def extract_error(body: JsonValue, dialect: str) -> tuple[str, Params]:
    """按显式方言取错误码和错误对象；非对象响应返回空证据。"""
    definition = DIALECTS.get(dialect, ErrorDialect())
    value = body
    for key in definition.error_path:
        value = value.get(key) if isinstance(value, dict) else None

    error = value if isinstance(value, dict) else {}
    for key in definition.code_fields:
        code = str(error.get(key) or "")
        if code not in definition.success_codes:
            return code, error

    if isinstance(value, str) and value:
        return value, error

    return "", error


def match_error(status: int, code: str, dialect: str) -> ErrorRule:
    """依次匹配业务码、方言 HTTP 规则和通用 HTTP 规则；未知情况保持 unknown。"""
    definition = DIALECTS.get(dialect)
    if definition is not None and code in definition.codes:
        return definition.codes[code]

    if definition is not None and status in definition.http_rules:
        return definition.http_rules[status]

    if status == 429 and code:
        return UNKNOWN

    return HTTP_RULES.get(status, UNKNOWN)
