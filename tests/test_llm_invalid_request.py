"""非法上下文交给上层回滚，不以切换模型掩盖协议错误。"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.chat import Chat
from TIYA.llm.provider.base import classify_error
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


@pytest.mark.parametrize("dialect", ["openai", "deepseek", "anthropic", "unknown"])
@pytest.mark.parametrize("status", [400, 422])
def test_invalid_format_is_neither_retried_nor_fallen_back(dialect, status):
    failure = classify_error(
        status, {"error": {"type": "invalid_request_error"}}, dialect
    )
    assert failure.category == "invalid_request"
    assert failure.safe
    assert not failure.retryable
    assert not failure.fallback_allowed
    assert failure.recovery == "none"


@pytest.mark.parametrize("status", [400, 422])
def test_chat_preserves_context_route_and_server_error_for_upper_recovery(
    tmp_path,
    monkeypatch,
    status,
):
    calls = []
    errors = []
    body = {"error": {
        "type": "invalid_request_error",
        "message": "Invalid tool_call_id in messages[2]",
        "param": "messages[2].tool_call_id",
    }}
    real_client = httpx.AsyncClient

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(status, json=body)

    def client_factory(**kwargs):
        kwargs.pop("mounts", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    rt = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))
    config = configuration("test", [
        candidate(
            f"chat_completions:{name}:model", "chat_completions", name, "model",
            {"endpoint": f"https://{name}/chat", "error_dialect": "deepseek"},
            {"stream": False},
        )
        for name in ("primary", "backup")
    ])

    async def run():
        logger = SimpleNamespace(
            error=errors.append,
            info=lambda *args: None,
            warning=lambda *args: None,
        )
        chat = Chat(config=config, logger=logger, log_off=True)
        rt.logger = logger
        chat._runtime = rt
        signature = chat._context.signature
        try:
            assert await chat.ask("malformed context", timeout=5) is None
            assert chat.last_outcome.category == "invalid_request"
            assert not chat.last_outcome.committed
            assert chat._context.signature == signature
            assert not chat._context

        finally:
            await chat.close()

    asyncio.run(run())
    assert calls == ["primary"]
    rows = rt.ledger.query_requests()
    assert len(rows) == 1
    assert rows[0]["diagnostic"]["http_status"] == status
    assert rows[0]["diagnostic"]["error"] == body
    assert any(str(status) in text and "Invalid tool_call_id" in text for text in errors)
    with rt.state.db.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM state WHERE namespace='cooldown'").fetchone()[0] == 0
