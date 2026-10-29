import json

import pytest

from llm_config_fixtures import candidate
from TIYA.llm.model import (
    Context,
    SystemPrompt,
    TextPrompt,
    ImagePrompt,
    ToolInput,
    ToolCall,
    ToolResponse,
    MediaRef,
)
from TIYA.llm.provider import get_provider


def function(value: str) -> str:
    """Return a value."""
    return value


@pytest.mark.parametrize(
    "category", ["chat_completions", "responses_api", "anthropic_api"]
)
def test_provider_encodes_complete_tool_loop_and_preserves_native_parameters(category):
    route = candidate(
        f"{category}:arbitrary:m",
        category,
        "arbitrary",
        "m",
        {"endpoint": "https://test"},
        {"custom_flag": False, "temperature": 0, "max_token": 15, "stream": True},
    )
    provider = get_provider(route)
    picture = ImagePrompt("user", b"")
    context = Context(
        SystemPrompt("rules"),
        ToolInput(function, "alias"),
        TextPrompt("user", "question"),
        picture,
        TextPrompt("assistant", "calling", "reason"),
        ToolCall("alias", "c", {"value": "x"}),
        ToolResponse("alias", "c", {"result": "x"}),
        TextPrompt("user", "continue"),
    )
    payload = provider.encode(
        context, {picture.id: [MediaRef("base64", "YWJj", "image/png")]}
    )
    assert payload["custom_flag"] is False
    assert payload["temperature"] == 0
    assert payload["max_token"] == 15
    assert "alias" in json.dumps(payload)
    assert "image/png" in json.dumps(payload)
    assert "c" in json.dumps(payload)
    if category == "responses_api":
        assert payload["instructions"] == "rules"
        assert payload["tools"][0]["strict"] is False
    elif category == "anthropic_api":
        assert payload["system"] == "rules"
        assert payload["messages"][1]["content"][1]["type"] == "tool_use"
    else:
        assert payload["stream_options"]["include_usage"] is True


def test_image_file_id_shapes_are_explicit_and_not_inferred_from_names():
    picture = ImagePrompt("user", b"")
    context = Context(picture)
    media = {picture.id: [MediaRef("remote", "file-id", "image/png")]}
    ds = get_provider(
        candidate(
            "chat_completions:random:m",
            "chat_completions",
            "random",
            "m",
            {"chat_completions": {"image_file_format": "deepseek"}},
        )
    )
    assert ds.encode(context, media)["messages"][0]["content"] == [
        {"type": "file", "file_id": "file-id"}
    ]
    oa = get_provider(
        candidate(
            "chat_completions:deepseek:m", "chat_completions", "deepseek", "m", {}
        )
    )
    with pytest.raises(ValueError):
        oa.encode(context, media)
    anthropic = get_provider(
        candidate(
            "anthropic_api:random:m",
            "anthropic_api",
            "random",
            "m",
            {},
            files_settings={
                "endpoint": "https://files.invalid/v1/files",
                "dialect": "anthropic",
                "anthropic_beta": "files-api-2025-04-14",
            },
        )
    )
    assert anthropic.encode(context, media)["messages"][0]["content"][0]["source"] == {
        "type": "file",
        "file_id": "file-id",
    }
    assert "authorization" not in {k.lower() for k in anthropic.headers()}
    assert anthropic.headers()["anthropic-beta"] == "files-api-2025-04-14"


def test_chat_stream_reassembles_tool_arguments_and_usage_once():
    provider = get_provider(
        candidate("chat_completions:x:m", "chat_completions", "x", "m", {})
    )
    stream = provider.stream_decoder()
    stream.feed(
        {
            "id": "r",
            "model": "resolved",
            "choices": [
                {
                    "delta": {
                        "content": "hello",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c",
                                "function": {"name": "f", "arguments": '{"value":'},
                            }
                        ],
                    }
                }
            ],
        }
    )
    stream.feed(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    stream.feed(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "prompt_cache_hit_tokens": 10,
            },
        }
    )
    result = stream.finish()
    assert result.complete
    assert result.prompts[1].arguments == {"value": "x"}
    assert result.reported_model == "resolved"
    assert result.response_id == "r"
    assert result.usage["cached"] == 10


def test_responses_stream_preserves_opaque_items_and_duplicate_done_is_idempotent():
    provider = get_provider(
        candidate("responses_api:x:m", "responses_api", "x", "m", {})
    )
    stream = provider.stream_decoder()
    call = {
        "type": "function_call",
        "id": "item",
        "call_id": "c",
        "name": "f",
        "arguments": "{}",
    }
    stream.feed(
        {"type": "response.created", "response": {"id": "r", "status": "in_progress"}}
    )
    stream.feed({"type": "response.output_item.added", "output_index": 0, "item": call})
    stream.feed({"type": "response.function_call_arguments.delta", "delta": "{}"})
    for _ in range(2):
        stream.feed(
            {"type": "response.output_item.done", "output_index": 0, "item": call}
        )
    stream.feed(
        {
            "type": "response.completed",
            "response": {
                "id": "r",
                "status": "completed",
                "usage": {"input_tokens": 20, "output_tokens": 4},
            },
        }
    )
    result = stream.finish()
    assert result.complete and stream.partial
    assert len(result.prompts.prompts) == 1
    assert result.prompts[0].call_id == "c"
    assert result.usage["output"] == 4


def test_anthropic_stream_retains_thinking_signature_and_tool_arguments():
    provider = get_provider(
        candidate("anthropic_api:x:m", "anthropic_api", "x", "m", {})
    )
    stream = provider.stream_decoder()
    stream.feed(
        {
            "type": "message_start",
            "message": {"id": "r", "model": "m", "usage": {"input_tokens": 10}},
        }
    )
    stream.feed(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        }
    )
    stream.feed(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "opaque"},
        }
    )
    stream.feed(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signed"},
        }
    )
    stream.feed(
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "c", "name": "f", "input": {}},
        }
    )
    stream.feed(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"value":"x"}'},
        }
    )
    stream.feed({"type": "message_stop"})
    result = stream.finish()
    assert result.native[0]["signature"] == "signed"
    assert result.prompts[0].arguments == {"value": "x"}
    replay = provider.encode(Context(result.prompts), {})
    assert replay["messages"][0]["content"][0]["thinking"] == "opaque"
