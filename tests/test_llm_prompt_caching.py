"""Provider 开启服务端缓存；不管理节点或修改业务 Context。"""

import asyncio
import copy
import json

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, ImagePrompt, MediaRef, Prompts, SystemPrompt, TextPrompt
from TIYA.llm.provider import get_provider
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_requests_enable_server_managed_caching_each_time(stream):
    route = candidate(
        "", "anthropic_api", "arbitrary", "model",
        {"endpoint": "https://compatible.test/messages"}, {"stream": stream},
    )
    provider = get_provider(route)
    picture = ImagePrompt("user", b"original")
    context = Context(SystemPrompt("stable system"), TextPrompt("user", "question"), picture)
    before = context.signature
    original_params = copy.deepcopy(route.model_params)
    media = {picture.id: [MediaRef("base64", "aW1hZ2U=", "image/png")]}
    for _ in range(2):
        payload = provider.encode(context, media)
        assert payload["cache_control"] == {"type": "ephemeral"}
        assert payload["stream"] is stream
        assert "cache_control" not in payload["messages"][-1]["content"][-1]

    assert context.signature == before
    assert route.model_params == original_params
    assert picture.image_input == b"original"


def test_anthropic_respects_native_cache_parameters_and_payload_ownership():
    params = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    route = candidate("", "anthropic_api", "test", "model", {}, params)
    payload = get_provider(route).encode(Context(TextPrompt("user", "hi")), {})
    assert payload["cache_control"] == params["cache_control"]
    payload["cache_control"]["ttl"] = "5m"
    assert route.model_params["cache_control"]["ttl"] == "1h"


@pytest.mark.parametrize("params", [{}, {"cache_control": {"type": "ephemeral", "ttl": "1h"}}])
def test_anthropic_compatibility_option_only_disables_default_injection(params):
    route = candidate("", "anthropic_api", "test", "model", {"prompt_caching": False}, params)
    payload = get_provider(route).encode(Context(TextPrompt("user", "hi")), {})
    assert payload.get("cache_control") == params.get("cache_control")
    assert ("cache_control" in payload) == ("cache_control" in params)


@pytest.mark.parametrize("category", ["chat_completions", "responses_api"])
def test_implicit_cache_protocols_do_not_inject_vendor_specific_cache_fields(category):
    route = candidate("", category, "arbitrary", "model", {})
    payload = get_provider(route).encode(Context(TextPrompt("user", "hi")), {})
    assert "cache_control" not in payload
    assert "enable_cache_control" not in payload
    assert "prompt_cache_options" not in payload


@pytest.mark.parametrize("category", ["chat_completions", "responses_api"])
def test_explicit_native_cache_options_remain_passthrough_for_compatible_services(category):
    params = {"prompt_cache_key": "stable-key", "prompt_cache_options": {"ttl": "30m"}}
    route = candidate("", category, "test", "model", {}, params)
    payload = get_provider(route).encode(Context(TextPrompt("user", "hi")), {})
    assert payload["prompt_cache_key"] == "stable-key"
    assert payload["prompt_cache_options"] == {"ttl": "30m"}


def test_runtime_sends_cache_opt_in_on_every_turn_and_records_reported_cache_usage(tmp_path):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["cache_control"] == {"type": "ephemeral"}
        return httpx.Response(200, json={
            "id": f"message-{len(requests)}",
            "content": [{"type": "text", "text": "answer"}],
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 2000,
                "output_tokens": 4,
            },
        })

    async def run():
        route = candidate(
            "", "anthropic_api", "arbitrary", "model",
            {"endpoint": "https://compatible.test/messages"}, {"stream": False},
        )
        ledger = Ledger(tmp_path / "usage.db")
        runtime = Runtime(StateStore(tmp_path / "state.db"), ledger)
        context = Context(SystemPrompt("system"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            for index in range(2):
                pending = await runtime.execute(
                    context, Prompts(TextPrompt("user", f"question-{index}")),
                    configuration("test", [route]), gateway, 5,
                )
                runtime.commit(context, pending)

        rows = ledger.query_requests()
        assert len(rows) == 2
        for row in rows:
            assert row["reported"]["input"] == 3010
            assert row["reported"]["cached"] == 2000
            assert row["reported"]["cache_write"] == 1000

    asyncio.run(run())
    assert len(requests) == 2
