from __future__ import annotations

import asyncio

import TIYA.LLM_connect as llm_module
from TIYA.LLM_connect import AskTask, Chat, ChatConfig, Prompts, TextPrompt


class FakeResponse:
    status_code = 200
    text = "ok"

    @staticmethod
    def json() -> dict:
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    @staticmethod
    def raise_for_status() -> None:
        return None


class FakeAsyncClient:
    instances: list[FakeAsyncClient] = []

    def __init__(self, *, mounts, headers, **kwargs) -> None:
        self.mounts = mounts
        self.headers = dict(headers)
        self.kwargs = kwargs
        self.closed = False
        self.requests: list[dict] = []
        self.instances.append(self)

    async def post(self, *, url, json, timeout):
        self.requests.append({
            "url": url,
            "json": json,
            "timeout": timeout,
            "headers": self.headers.copy(),
        })
        return FakeResponse()

    async def aclose(self) -> None:
        self.closed = True


def _config(endpoint: str, token: str, model: str) -> ChatConfig:
    return ChatConfig(
        endpoint=endpoint,
        token=token,
        model=model,
        api_provider="openai",
        model_params={"stream": False},
        keep_alive=True,
        log_off=True,
    )


def test_keep_alive_client_is_rebuilt_after_switching_preset(monkeypatch) -> None:
    FakeAsyncClient.instances.clear()
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)

    async def run() -> None:
        chat = Chat(config=_config("https://first.test", "first-key", "first"))
        try:
            await chat.post(AskTask(
                Prompts(TextPrompt(role="user", text="first")),
                timeout=1,
            ))
            first_client = FakeAsyncClient.instances[0]

            chat.setting(_config("https://second.test", "second-key", "second"))
            await chat.post(AskTask(
                Prompts(TextPrompt(role="user", text="second")),
                timeout=1,
            ))

            assert len(FakeAsyncClient.instances) == 2
            assert first_client.closed
            assert FakeAsyncClient.instances[1].requests[0]["headers"][
                "Authorization"
            ] == "Bearer second-key"
            assert FakeAsyncClient.instances[1].requests[0]["url"] == (
                "https://second.test"
            )
        finally:
            await chat.close()

    asyncio.run(run())


def test_creating_client_does_not_consume_proxy_configuration(monkeypatch) -> None:
    FakeAsyncClient.instances.clear()
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        llm_module,
        "parse_proxies_to_httpx",
        lambda proxies: {"configured": proxies},
    )
    client_params = {"proxies": {"https://": "proxy.test"}, "verify": False}

    async def run() -> None:
        config = _config("https://first.test", "first-key", "first")
        config.client_params = client_params
        chat = Chat(config=config)
        try:
            await chat.post(AskTask(
                Prompts(TextPrompt(role="user", text="first")),
                timeout=1,
            ))

            assert client_params["proxies"] == {"https://": "proxy.test"}
            assert chat._client_params["proxies"] == {
                "https://": "proxy.test",
            }
        finally:
            await chat.close()

    asyncio.run(run())


def test_switching_only_model_keeps_existing_client_pool(monkeypatch) -> None:
    FakeAsyncClient.instances.clear()
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeAsyncClient)

    async def run() -> None:
        chat = Chat(config=_config("https://same.test", "same-key", "flash"))
        try:
            await chat.post(AskTask(
                Prompts(TextPrompt(role="user", text="first")),
                timeout=1,
            ))

            chat.setting(_config("https://same.test", "same-key", "pro"))
            await chat.post(AskTask(
                Prompts(TextPrompt(role="user", text="second")),
                timeout=1,
            ))

            assert len(FakeAsyncClient.instances) == 1
            assert len(FakeAsyncClient.instances[0].requests) == 2
            assert FakeAsyncClient.instances[0].requests[1]["json"]["model"] == (
                "pro"
            )
        finally:
            await chat.close()

    asyncio.run(run())
