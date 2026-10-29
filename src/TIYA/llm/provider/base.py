from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..model import Prompts
from ..config import credential


def merge_headers(*mappings):
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
    def safe(self):
        return self.remote_state == "rejected"


class ProviderFailure(Exception):
    def __init__(self, failure: Failure):
        super().__init__(f"{failure.category}: {failure.code}")
        self.failure = failure


def classify_error(status, body, dialect="openai", headers=None):
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
    prompts: Prompts = field(default_factory=Prompts)
    usage: dict = field(default_factory=dict)
    raw_usage: dict = field(default_factory=dict)
    native: list = field(default_factory=list)
    response_id: str = ""
    complete: bool = True
    images: list = field(default_factory=list)
    request_id: str = ""
    reported_model: str = ""
    stored: bool | None = None


class Provider:
    def __init__(self, route):
        self.route = route

    @property
    def supports_parent(self):
        # Explicit opt-in: compatibility endpoints may silently ignore store/parent.
        return (
            self.route.settings.get("responses", {}).get(
                "supports_previous_response_id", False
            )
            is True
            and self.route.params.get("store", True) is not False
        )

    def headers(self):
        settings = self.route.settings
        headers = {"Authorization": f"Bearer {credential(settings)}"}
        return merge_headers(
            headers, settings.get("headers"), self.route.client_params.get("headers")
        )

    def failure(self, status, body, headers=None):
        return classify_error(
            status, body, self.route.settings.get("error_dialect", "openai"), headers
        )

    def check_body(self, body):
        if body.get("error") or body.get("base_resp", {}).get("status_code", 0):
            raise ProviderFailure(self.failure(200, body))
