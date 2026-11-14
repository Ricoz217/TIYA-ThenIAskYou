"""DeepSeek 官方 HTTP 错误表与 Runtime 安全重试的回归。"""

import asyncio

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, Prompts, TextPrompt
from TIYA.llm.provider.base import classify_error
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


@pytest.mark.parametrize("code", [None, "invalid_request_error", "rate_limit_exceeded", "vendor_detail"])
@pytest.mark.parametrize("status,category,safe,retryable,fallback", [
    (400, "invalid_request", True, False, False),
    (401, "authentication", True, False, True),
    (402, "quota", True, False, True),
    (422, "invalid_request", True, False, False),
    (429, "rate_limit", True, True, True),
    (500, "unavailable", False, False, False),
    (503, "unavailable", False, False, False),
])
def test_deepseek_documented_http_statuses(
    status,
    category,
    safe,
    retryable,
    fallback,
    code,
):
    failure = classify_error(status, {"error": {"code": code}}, "deepseek", {"retry-after": "0"})
    assert failure.category == category
    assert failure.safe is safe
    assert failure.retryable is retryable
    assert failure.fallback_allowed is fallback
    assert failure.retry_after == 0


def test_deepseek_http_override_does_not_change_other_dialects():
    for dialect in ("openai", "glm", "minimax", "unknown"):
        failure = classify_error(429, {"error": {"code": "vendor_detail"}, "base_resp": {"status_code": 9001}}, dialect)
        assert failure.category == "unknown"
        assert not failure.retryable


def test_deepseek_rate_limit_is_retried_and_each_send_is_accounted(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"code": "invalid_request_error"}}, headers={"retry-after": "0"})

        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        route = candidate("", "chat_completions", "arbitrary", "model", {"endpoint": "https://test/chat", "error_dialect": "deepseek"}, {"stream": False})
        ledger = Ledger(tmp_path / "usage.db")
        runtime = Runtime(StateStore(tmp_path / "state.db"), ledger)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            pending = await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), configuration("test", [route]), gateway, 5)
            assert pending.result.prompts[0].text == "ok"

        assert len(ledger.query_requests()) == 2

    asyncio.run(run())
    assert len(requests) == 2
