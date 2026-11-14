"""错误响应落盘，包括恢复成功前的失败和非 JSON 网关错误。"""

import asyncio
import json
from datetime import datetime

import httpx
import pytest

from TIYA.llm.gateway import Gateway, HTTPFailure, WireRequest
from TIYA.logger import Logger, LoggerConfig


@pytest.mark.parametrize("mode", ["json", "stream", "bytes", "download"])
@pytest.mark.parametrize("is_json", [True, False])
def test_http_errors_write_status_request_id_and_original_response(
    tmp_path,
    monkeypatch,
    mode,
    is_json,
):
    import TIYA.llm.gateway as module

    body = {"error": {
        "code": "invalid_request_error", "message": "工具引用不匹配",
        "param": "messages[2].tool_call_id", "detail": {"received": "call-wrong"},
    }}
    raw = json.dumps(body, ensure_ascii=False) if is_json else "<html>upstream rejected call-wrong</html>"
    logger = Logger(LoggerConfig(logs_dir=tmp_path / "logs", use_console=False))
    monkeypatch.setattr(module, "_log", logger, raising=False)
    real_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(
            422, text=raw,
            headers={"x-request-id": "server-request-123", "content-type": "application/json" if is_json else "text/html"},
        )

    async def run():
        async with real_client(transport=httpx.MockTransport(handler)) as client:
            gateway = Gateway(client)
            request = WireRequest(
                "https://test/api", headers={"Authorization": "request-secret"},
                json={"private_context": "request-private-context"},
            )
            if mode == "download":
                def download_client(**kwargs):
                    return real_client(transport=httpx.MockTransport(handler), **kwargs)

                monkeypatch.setattr(httpx, "AsyncClient", download_client)

            with pytest.raises((HTTPFailure, httpx.HTTPStatusError)) as caught:
                if mode == "json":
                    await gateway.send(request, 5)

                elif mode == "stream":
                    async for _ in gateway.stream(request, 5):
                        pass

                elif mode == "bytes":
                    await gateway.send_bytes(request, 5)

                else:
                    await gateway.download("https://test/image", 5)

            if isinstance(caught.value, HTTPFailure) and not is_json:
                assert caught.value.body["response_text"] == raw

    try:
        asyncio.run(run())
        logger.flush()
        text = (tmp_path / "logs" / datetime.now().strftime("%Y-%m-%d") / "debug.log").read_text(encoding="utf-8")
        assert "422" in text
        assert "server-request-123" in text
        assert raw in text
        assert "request-secret" not in text
        assert "request-private-context" not in text

    finally:
        logger.shutdown()


def test_provider_error_body_is_written_even_when_http_request_succeeded(tmp_path, monkeypatch):
    import TIYA.llm.provider.base as module
    from TIYA.llm.config_model import LLMConfig
    from TIYA.llm.provider.openai_chat import ChatCompletions

    body = {"error": {"code": "invalid_request_error", "message": "business error", "param": "messages"}}
    logger = Logger(LoggerConfig(logs_dir=tmp_path / "logs", use_console=False))
    monkeypatch.setattr(module, "_log", logger, raising=False)
    provider = ChatCompletions(LLMConfig("chat_completions", "test", "https://test", "", "m"))
    try:
        with pytest.raises(module.ProviderFailure):
            provider.decode(body)

        logger.flush()
        text = (tmp_path / "logs" / datetime.now().strftime("%Y-%m-%d") / "debug.log").read_text(encoding="utf-8")
        assert "invalid_request_error" in text
        assert json.dumps(body, ensure_ascii=False) in text

    finally:
        logger.shutdown()


def test_retry_success_does_not_erase_the_original_http_error(tmp_path, monkeypatch):
    import TIYA.llm.gateway as module
    from llm_config_fixtures import candidate, configuration
    from TIYA.llm.model import Context, Prompts, TextPrompt
    from TIYA.llm.runtime import Runtime
    from TIYA.llm.state import StateStore
    from TIYA.llm.usage import Ledger

    count = 0
    body = {"error": {"code": "rate_limit_exceeded", "message": "retry later", "limit": 123}}
    logger = Logger(LoggerConfig(logs_dir=tmp_path / "logs", use_console=False))
    monkeypatch.setattr(module, "_log", logger)

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429, json=body, headers={"retry-after": "0"})

        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    async def run():
        runtime = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))
        config = configuration("test", [candidate(
            "chat_completions:test:m", "chat_completions", "test", "m",
            {"endpoint": "https://test/chat"}, {"stream": False},
        )])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            pending = await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5)
            assert pending.result.prompts[0].text == "ok"

    try:
        asyncio.run(run())
        assert count == 2
        logger.flush()
        text = (tmp_path / "logs" / datetime.now().strftime("%Y-%m-%d") / "debug.log").read_text(encoding="utf-8")
        assert "429" in text
        assert '"limit":123' in text
        assert "retry later" in text

    finally:
        logger.shutdown()
