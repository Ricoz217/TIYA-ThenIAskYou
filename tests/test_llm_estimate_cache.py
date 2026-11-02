import asyncio
import copy

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.model import (
    Context,
    ImagePrompt,
    SystemPrompt,
    TextPrompt,
    ToolCall,
    ToolInput,
    ToolResponse,
)
from TIYA.llm.provider.common import projection_fingerprint
from TIYA.llm.usage import estimate


@pytest.fixture
def encoded_texts(monkeypatch):
    import tiktoken

    texts = []

    class Tokenizer:
        def encode(self, text):
            texts.append(text)
            return list(text)

    tokenizer = Tokenizer()
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: tokenizer)
    return texts


def test_repeated_estimates_and_snapshots_only_encode_new_messages(encoded_texts):
    context = Context(SystemPrompt("system"), TextPrompt("user", "history"))
    first = estimate(context)
    assert estimate(context) == first
    snapshot = copy.deepcopy(context)
    snapshot.append(TextPrompt("user", "new"))
    assert estimate(snapshot)["input"] == first["input"] + len("new") + 3
    assert estimate(context.copy()) == first
    assert encoded_texts == ["system", "history", "new"]


def test_cache_does_not_change_fingerprints_or_archives(encoded_texts):
    prompt = TextPrompt("user", "history")
    context = Context(prompt)
    prompts = context.message
    before = (context.signature, context.header_signature, prompts.signature,
              projection_fingerprint(prompt), context.to_dict())
    estimate(context)
    context.refresh_fingerprints(content_changed=False)
    prompts.refresh_fingerprints(content_changed=False)
    assert (context.signature, context.header_signature, prompts.signature,
            projection_fingerprint(prompt), context.to_dict()) == before
    assert "estimated_tokens" not in vars(prompt)


def test_explicit_content_refresh_invalidates_nested_tool_estimates(encoded_texts):
    def tool(value: str) -> str:
        return value

    text = TextPrompt("user", "before")
    call = ToolCall("tool", "call", {"value": "before"})
    response = ToolResponse("tool", "call", {"result": "before"})
    context = Context(text, ToolInput(tool), call, response)
    estimate(context)
    text.text = "after"
    call.arguments["value"] = "after"
    response.response["result"] = "after"
    context.refresh_fingerprints()
    updated = estimate(context)
    fresh = Context(TextPrompt("user", "after"), ToolInput(tool),
                    ToolCall("tool", "call", {"value": "after"}),
                    ToolResponse("tool", "call", {"result": "after"}))
    assert updated == estimate(fresh)
    assert encoded_texts.count("before") == 1
    assert encoded_texts.count("after") == 2


def test_append_remove_and_rollback_keep_unchanged_message_estimates(encoded_texts):
    context = Context(SystemPrompt("system", create_time="1"),
                      TextPrompt("user", "one", create_time="2"),
                      TextPrompt("assistant", "answer", create_time="3"))
    estimate(context)
    second = TextPrompt("user", "two", create_time="4")
    context.append(second)
    estimate(context)
    context.back2last_round()
    assert estimate(context)["input"] == sum(len(s) + 3 for s in ("system", "one", "answer"))
    context.discard_prompt_ids({context.message[1].id})
    assert estimate(context)["input"] == sum(len(s) + 3 for s in ("system", "answer"))
    context.clear()
    assert estimate(context)["input"] == len("system") + 3
    assert encoded_texts == ["system", "one", "answer", "two"]


def test_estimates_survive_system_copy_and_borrowed_tool_deepcopy(encoded_texts):
    def tool(value: str) -> str:
        return value

    context = Context(SystemPrompt("system"), ToolInput(tool))
    expected = estimate(context)
    assert estimate(copy.deepcopy(context)) == expected
    assert estimate(Context(context.system, copy.deepcopy(context.tools))) == expected
    assert len(encoded_texts) == 2


def test_images_are_counted_without_encoding_payload(encoded_texts):
    image = ImagePrompt("user", b"not-read")
    context = Context(TextPrompt("user", "text"), image)
    assert estimate(context)["input"] == len("text") + 3 + 1440
    image.image = ["first", "second"]
    context.refresh_fingerprints(content_changed=False)
    assert estimate(context)["input"] == len("text") + 3 + 2880
    assert encoded_texts == ["text"]


def test_warm_async_estimate_does_not_schedule_thread_work(encoded_texts, monkeypatch):
    from TIYA.llm import usage

    context = Context(TextPrompt("user", "history"))
    expected = estimate(context)

    async def forbidden(*args, **kwargs):
        pytest.fail("warm estimates must not queue another worker job")

    monkeypatch.setattr(usage, "run_sync", forbidden, raising=False)
    assert asyncio.run(usage.estimate_async(context)) == expected
    assert encoded_texts == ["history"]


def test_chat_threshold_fallback_and_next_round_reuse_estimates(encoded_texts, monkeypatch):
    from TIYA.llm.chat import Chat
    from TIYA.llm.usage_model import TokenUsage

    config = configuration("main", [
        candidate("chat_completions:a:m", "chat_completions", "a", "m",
                  {"endpoint": "https://a.test/chat"}, {"stream": False}, max_context=10000),
        candidate("chat_completions:b:m", "chat_completions", "b", "m",
                  {"endpoint": "https://b.test/chat"}, {"stream": False}, max_context=10000),
    ])
    real_client = httpx.AsyncClient
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "a.test":
            return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})

        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(handler)))

    async def run():
        context = Context(SystemPrompt("system"), TextPrompt("user", "history"))
        chat = Chat(config=config, context=context)
        chat._last_usage = TokenUsage(input_t=6000)
        try:
            assert await chat.ask("question", no_wait=True, timeout=5) is not None
            assert await chat.ask("next", no_wait=True, timeout=5) is not None
            ledger = chat._runtime.ledger
            with ledger.db.transaction() as connection:
                rows = connection.execute("SELECT id, estimated FROM requests").fetchall()
            assert len(rows) == len({row[0] for row in rows}) == 3
            assert all('"input"' in row[1] for row in rows)

        finally:
            await chat.close()

    asyncio.run(run())
    assert encoded_texts.count("system") == encoded_texts.count("history") == 1
    assert encoded_texts.count("question") == encoded_texts.count("next") == 1
    assert encoded_texts.count("answer") == 2
    assert [request.url.host for request in requests] == ["a.test", "b.test", "b.test"]
