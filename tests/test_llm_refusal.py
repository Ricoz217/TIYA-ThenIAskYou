import asyncio
import json

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.chat import Chat
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, TextPrompt
from TIYA.llm.provider import get_provider
from TIYA.llm.provider.base import ProviderFailure
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


def preset(category, stream):
    return configuration("test", [
        candidate(f"{category}:a:m", category, "a", "m", {"endpoint": "https://test/first"}, {"stream": stream}),
        candidate(f"{category}:b:m", category, "b", "m", {"endpoint": "https://test/backup"}, {"stream": stream}),
    ])


def refusal(category):
    if category == "chat_completions":
        return {"choices": [{"message": {"refusal": "Cannot assist"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3}}

    if category == "responses_api":
        return {"id": "r1", "status": "completed", "output": [
            {"type": "message", "content": [{"type": "refusal", "refusal": "Cannot assist"}]}
        ], "usage": {"input_tokens": 10, "output_tokens": 3}}

    return {"id": "r1", "stop_reason": "refusal", "content": [{"type": "text", "text": "Cannot assist"}],
            "usage": {"input_tokens": 10, "output_tokens": 3}}


class Logger:
    def __init__(self):
        self.messages = []

    def info(self, text, **kwargs):
        self.messages.append(text)
        return self

    def error(self, text):
        self.messages.append(text)

    def warning(self, text):
        self.messages.append(text)

    def write(self, text):
        self.messages.append(text)

    def close(self):
        pass


@pytest.mark.parametrize("category", ["chat_completions", "responses_api", "anthropic_api"])
def test_explicit_refusal_is_not_an_ordinary_prompt(category):
    provider = get_provider(preset(category, False).primary)
    with pytest.raises(ProviderFailure) as caught:
        provider.decode(refusal(category))

    assert caught.value.failure.category == "refusal"
    assert caught.value.failure.remote_state == "completed"
    assert not caught.value.failure.fallback_allowed


@pytest.mark.parametrize("category", ["chat_completions", "responses_api", "anthropic_api"])
@pytest.mark.parametrize("stream", [False, True])
def test_refusal_returns_none_logs_reason_records_usage_and_never_falls_back(
    tmp_path,
    monkeypatch,
    category,
    stream,
):
    calls = []
    logger = Logger()
    body = refusal(category)

    def handler(request):
        calls.append(request)
        if not stream:
            return httpx.Response(200, json=body)

        if category == "chat_completions":
            events = [
                {"choices": [{"index": 0, "delta": {"refusal": "Cannot "}}]},
                {"choices": [{"index": 0, "delta": {"refusal": "assist"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": body["usage"]},
            ]

        elif category == "responses_api":
            events = [
                {"type": "response.refusal.delta", "delta": "Cannot assist"},
                {"type": "response.completed", "response": body},
            ]

        else:
            events = [
                {"type": "message_start", "message": {"id": "r1", "usage": {"input_tokens": 10}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "Cannot assist"}},
                {"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {"output_tokens": 3}},
                {"type": "message_stop"},
            ]

        return httpx.Response(200, text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
                              headers={"content-type": "text/event-stream"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler)))

    async def run():
        chat = Chat(config=preset(category, stream), logger=logger)
        rt = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"), logger=logger)
        chat._runtime = rt
        try:
            assert await chat.ask("recognize", timeout=5) is None
            assert not chat._context
            assert not chat.last_outcome.committed
            assert chat.last_outcome.remote_state == "completed"
            assert chat.last_outcome.category == "refusal"
            row = rt.ledger.query_requests()[0]
            assert row["status"] == "refused"
            assert row["reported"]["input"] == 10
            assert row["reported"]["output"] == 3
            assert len(calls) == 1
            assert "Cannot assist" in "".join(logger.messages)

        finally:
            await chat.close()

    asyncio.run(run())


def test_unmarked_text_is_not_classified_by_keywords():
    provider = get_provider(preset("chat_completions", False).primary)
    result = provider.decode({"choices": [{"message": {"content": "Cannot assist is an example phrase."}}]})
    assert result.prompts[0].text == "Cannot assist is an example phrase."


@pytest.mark.parametrize("category", ["chat_completions", "responses_api"])
def test_refusal_without_a_terminal_event_keeps_remote_state_unknown(tmp_path, category):
    if category == "chat_completions":
        event = {"choices": [{"delta": {"refusal": "Cannot assist"}}]}

    else:
        event = {"type": "response.refusal.delta", "delta": "Cannot assist"}

    async def run():
        rt = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, text=f"data: {json.dumps(event)}\n\n",
                                  headers={"content-type": "text/event-stream"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(ProviderFailure) as caught:
                await rt.execute(Context(), TextPrompt("user", "recognize"), preset(category, True), gateway, 5)

        assert caught.value.llm_outcome.remote_state == "unknown"
        assert rt.ledger.query_requests()[0]["remote_state"] == "unknown"
        assert len(calls) == 1

    asyncio.run(run())
