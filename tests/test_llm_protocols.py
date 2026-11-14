import asyncio
import json

import httpx
import pytest


def test_sse_multiline_and_comments_and_json_response():
    from TIYA.llm.gateway import frames

    async def run():
        async def lines():
            for line in [
                ":heartbeat",
                "event: response.completed",
                'data: {"a":',
                "data: 1}",
                "",
                "data: [DONE]",
                "",
            ]:
                yield line

        events = [event async for event in frames(lines())]
        assert events[0].event == "response.completed"
        assert json.loads(events[0].data) == {"a": 1}
        assert events[1].data == "[DONE]"

    asyncio.run(run())


def test_success_status_with_html_is_not_a_successful_model_result():
    from TIYA.llm.gateway import Gateway, WireRequest

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, text="<html>bad gateway</html>")
            )
        ) as client:
            with pytest.raises(ValueError):
                await Gateway(client).send(WireRequest("https://test"), 1)

    asyncio.run(run())


def test_image_download_uses_proxy_without_model_credentials(monkeypatch):
    import TIYA.llm.gateway as module

    options = {}

    class DownloadClient:
        def __init__(self, **kwargs):
            options.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            return httpx.Response(
                200, content=b"image", request=httpx.Request("GET", url)
            )

    monkeypatch.setattr(module.httpx, "AsyncClient", DownloadClient)

    async def run():
        gateway = module.Gateway(
            object(),
            download_options={
                "proxy": "http://proxy:8080",
                "headers": {"Authorization": "secret"},
                "cookies": {"token": "secret"},
            },
        )
        assert await gateway.download("https://cdn/image") == b"image"
        assert options["proxy"] == "http://proxy:8080"
        assert "headers" not in options and "cookies" not in options

    asyncio.run(run())


def test_provider_error_dialects_do_not_conflate_quota_and_rate():
    from TIYA.llm.provider.base import classify_error

    assert (
        classify_error(
            429, {"error": {"code": "insufficient_quota"}}, "openai"
        ).category
        == "quota"
    )
    assert classify_error(429, {"error": {"code": "1113"}}, "glm").category == "quota"
    assert (
        classify_error(429, {"error": {"code": "1302"}}, "glm").category == "rate_limit"
    )
    assert (
        classify_error(200, {"base_resp": {"status_code": 1008}}, "minimax").category
        == "quota"
    )
    unknown = classify_error(
        400, {"error": {"message": "some file operation failed"}}, "openai"
    )
    assert unknown.category == "invalid_request"
    assert unknown.resource is None


def test_responses_preserves_function_calls_and_reasoning_native_items():
    from llm_config_fixtures import candidate
    from TIYA.llm.provider import get_provider
    from TIYA.LLM_connect import Context, TextPrompt

    route = candidate(
        "responses_api:any:m", "responses_api", "any", "m", {"endpoint": "https://test"}
    )
    provider = get_provider(route)
    body = {
        "id": "r1",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs",
                "encrypted_content": "opaque",
                "summary": [],
            },
            {
                "type": "function_call",
                "id": "fc",
                "call_id": "c",
                "name": "tool",
                "arguments": '{"x":1}',
            },
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 4,
            "input_tokens_details": {"cached_tokens": 10},
        },
    }
    result = provider.decode(body)
    context = Context(TextPrompt("user", "hi"), result.prompts)
    encoded = provider.encode(context, {})
    assert encoded["input"][-2:] == body["output"]
    assert result.usage["cached"] == 10
    assert result.prompts[0].arguments == {"x": 1}
    restored = Context.from_dict(context.to_dict(), {})
    assert provider.encode(restored, {})["input"][-2:] == body["output"]


def test_broken_tool_json_fails_instead_of_executing_empty_arguments():
    from llm_config_fixtures import candidate
    from TIYA.llm.provider import get_provider

    provider = get_provider(
        candidate("chat_completions:a:m", "chat_completions", "a", "m", {})
    )
    with pytest.raises(ValueError):
        provider.decode(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "c", "function": {"name": "f", "arguments": "{"}}
                            ]
                        }
                    }
                ]
            }
        )


def test_anthropic_staged_usage_is_merged_not_max_selected():
    from llm_config_fixtures import candidate
    from TIYA.llm.provider import get_provider

    provider = get_provider(
        candidate("anthropic_api:a:m", "anthropic_api", "a", "m", {})
    )
    stream = provider.stream_decoder()
    stream.feed(
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 20}},
        }
    )
    stream.feed(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    )
    stream.feed(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ok"},
        }
    )
    stream.feed({"type": "message_delta", "usage": {"output_tokens": 4}})
    stream.feed({"type": "message_stop"})
    result = stream.finish()
    assert result.usage == {"input": 30, "output": 4, "cached": 20}
    assert result.prompts[0].text == "ok"


def test_native_projection_must_not_resurrect_a_deleted_or_edited_message():
    from llm_config_fixtures import candidate
    from TIYA.llm.provider import get_provider
    from TIYA.llm.model import Context

    provider = get_provider(
        candidate(
            "responses_api:x:m", "responses_api", "x", "m", {"endpoint": "https://test"}
        )
    )
    result = provider.decode(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "one"},
                        {"type": "output_text", "text": "two"},
                    ],
                }
            ]
        }
    )
    context = Context(result.prompts)
    context._messages.pop()
    assert "two" not in str(provider.encode(context, {}))
    context._messages[0].text = "changed"
    encoded = provider.encode(context, {})
    assert "one" not in str(encoded)
    assert "changed" in str(encoded)


def test_chat_mixed_content_preserves_text_image_tool_order():
    from llm_config_fixtures import candidate
    from TIYA.llm.provider import get_provider
    from TIYA.llm.model import ImagePrompt, ToolCall

    provider = get_provider(
        candidate(
            "chat_completions:x:m",
            "chat_completions",
            "x",
            "m",
            {"endpoint": "https://test"},
        )
    )
    result = provider.decode(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "before"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://image"},
                            },
                            {"type": "text", "text": "after"},
                        ],
                        "tool_calls": [
                            {"id": "c", "function": {"name": "f", "arguments": "{}"}}
                        ],
                    }
                }
            ]
        }
    )
    assert result.prompts[0].text == "before"
    assert isinstance(result.prompts[1], ImagePrompt)
    assert result.prompts[2].text == "after"
    assert isinstance(result.prompts[3], ToolCall)
    assert result.images[0][1] == {"url": "https://image"}
