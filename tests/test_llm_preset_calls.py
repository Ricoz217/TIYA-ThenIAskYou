"""Compatibility is checked through configuration and the public Chat API."""

import asyncio
import copy
import json

import httpx
import pytest

from TIYA import config
from TIYA.LLM_connect import Chat, ToolCall, ToolInput, parse_llm_setting


def mock_model(monkeypatch, handler):
    real_client = httpx.AsyncClient

    def client(**params):
        params.update(
            mounts={}, trust_env=False, transport=httpx.MockTransport(handler)
        )
        return real_client(**params)

    monkeypatch.setattr(httpx, "AsyncClient", client)


def document(version):
    if version == 1:
        return {
            "LLM_List": [
                {
                    "preset_name": "MAIN",
                    "endpoint": "https://primary.invalid/chat/completions",
                    "token": "dummy",
                    "model": "model",
                    "api_type": "openai",
                    "proxy_mode": "",
                    "max_context": 10000,
                    "auto_compress_rate": 0.7,
                    "price": {},
                    "extra_parameter": {"stream": False, "temperature": 0},
                }
            ]
        }
    return {
        "schema_version": 2,
        "Providers": {
            "chat_completions": {
                "primary": {
                    "endpoint": "https://primary.invalid/chat/completions",
                    "key": "dummy",
                },
                "backup": {
                    "endpoint": "https://backup.invalid/chat/completions",
                    "key": "other",
                },
            }
        },
        "LLM_List": {
            "MAIN": {
                "auto_compress_rate": 0.7,
                "proxy_mode": "",
                "providers": [
                    {
                        "llm": "chat_completions:primary:model",
                        "max_context": 10000,
                        "llm_params": {"stream": False, "temperature": 0},
                    },
                    {
                        "llm": "chat_completions:backup:other-model",
                        "llm_params": {"stream": False},
                    },
                ],
            }
        },
    }


@pytest.mark.parametrize("version", [1, 2])
def test_old_entry_points_send_tools_and_return_the_same_contract(monkeypatch, version):
    original = document(version)
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(copy.deepcopy(original)))
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query":"value"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    mock_model(monkeypatch, handler)

    def lookup(query: str):
        """Look up a value."""
        return query

    async def run():
        setting = parse_llm_setting("MAIN")
        async with Chat(config=setting, log_off=True) as chat:
            chat.add_tools(ToolInput(lookup))
            response = await chat.ask("question", timeout=5)
            assert response[0].text == "answer"
            assert isinstance(response[1], ToolCall)
            assert response[1].function_name == "lookup"
            assert response[1].arguments == {"query": "value"}
            assert sent[0]["model"] == "model"
            assert sent[0]["temperature"] == 0
            assert sent[0]["stream"] is False
            assert sent[0]["tools"][0]["function"]["name"] == "lookup"
            history = (await chat.get_context()).to_prompts()
            assert [p.text for p in history if hasattr(p, "text")] == [
                "question",
                "answer",
            ]
        setting.model_params["temperature"] = 1
        assert parse_llm_setting("MAIN").model_params["temperature"] == 0

    asyncio.run(run())
    assert dict(config.LLM_CFG) == original


def test_v2_fallback_works_through_the_existing_setting_entry(monkeypatch):
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(document(2)))
    hosts, models = [], []

    def handler(request):
        hosts.append(request.url.host)
        models.append(json.loads(request.content)["model"])
        if request.url.host == "primary.invalid":
            return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "fallback answer"}}]}
        )

    mock_model(monkeypatch, handler)

    async def run():
        async with Chat(log_off=True) as chat:
            chat.setting(parse_llm_setting("MAIN"))
            response = await chat.ask("question", timeout=5)
            assert response[0].text == "fallback answer"
            assert [p.text for p in (await chat.get_context()).to_prompts()] == [
                "question",
                "fallback answer",
            ]

    asyncio.run(run())
    assert hosts == ["primary.invalid", "backup.invalid"]
    assert models == ["model", "other-model"]
