import asyncio
import copy
import json

import httpx
import pytest

from TIYA.utils import fingerprint
from llm_config_fixtures import configuration, candidate
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, Prompts, TextPrompt, SystemPrompt
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


def services(tmp_path):
    return Runtime(
        StateStore(tmp_path / "runtime.sqlite3"), Ledger(tmp_path / "usage.sqlite3")
    )


def test_fallback_cooldown_survives_restart_and_does_not_slide(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "primary":
            return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    config = configuration(
        "p",
        [
            candidate(
                "chat_completions:a:m",
                "chat_completions",
                "a",
                "m",
                {"endpoint": "https://primary"},
            ),
            candidate(
                "chat_completions:b:m",
                "chat_completions",
                "b",
                "m",
                {"endpoint": "https://backup"},
            ),
        ],
    )

    async def run():
        rt = services(tmp_path)
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            first = await rt.execute(
                context, Prompts(TextPrompt("user", "one")), config, gateway, 5
            )
            rt.commit(context, first)
            key = fingerprint([context.session_id, config.revision])
            until = rt.state.get("cooldown", key)["until"]
            rt = services(tmp_path)
            restored = Context.from_dict(context.to_dict(), {})
            second = await rt.execute(
                restored, Prompts(TextPrompt("user", "two")), config, gateway, 5
            )
            rt.commit(restored, second)
            assert calls == ["primary", "backup", "backup"]
            assert rt.state.get("cooldown", key)["until"] == until
            rt.state.put("cooldown", key, {"index": 1, "until": 0})
            await rt.execute(
                restored, Prompts(TextPrompt("user", "three")), config, gateway, 5
            )
            assert calls[-2:] == ["primary", "backup"]

    asyncio.run(run())


def test_system_prompt_is_stable_and_rollback_can_use_an_earlier_checkpoint(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": f"r{len(calls)}",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
        )

    route = candidate(
        "responses_api:a:m",
        "responses_api",
        "a",
        "m",
        {
            "endpoint": "https://test",
            "responses": {"supports_previous_response_id": True},
        },
    )
    config = configuration("p", [route])

    async def run():
        rt = services(tmp_path)
        context = Context(SystemPrompt("stable"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            for text in ("one", "two"):
                response = await rt.execute(
                    context, Prompts(TextPrompt("user", text)), config, gateway, 5
                )
                rt.commit(context, response)
            assert calls[1]["previous_response_id"] == "r1"
            assert calls[1]["instructions"] == "stable"
            context.back2last_round()
            response = await rt.execute(
                context, Prompts(TextPrompt("user", "replacement")), config, gateway, 5
            )
            assert calls[2]["previous_response_id"] == "r1"
            assert calls[2]["input"][0]["content"][0]["text"] == "replacement"

    asyncio.run(run())


def test_stream_partial_failure_stops_all_candidates_and_preserves_usage(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":20}}\n\n'
                'data: {"error":{"code":"rate_limit_exceeded"}}\n\n'
            ),
        )

    route = candidate(
        "chat_completions:a:m",
        "chat_completions",
        "a",
        "m",
        {"endpoint": "https://test"},
        {"stream": True},
    )

    async def run():
        rt = services(tmp_path)
        config = configuration("p", [route, copy.deepcopy(route)])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            with pytest.raises(Exception):
                await rt.execute(
                    Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5
                )
        assert len(calls) == 1
        row = rt.ledger.query_requests()[0]
        assert row["reported"]["input"] == 20
        assert row["diagnostic"]["partial"] is True

    asyncio.run(run())


def test_invalid_parent_is_not_probed_again_after_failed_full_replay(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 2:
            return httpx.Response(
                400, json={"error": {"code": "previous_response_not_found"}}
            )
        if len(calls) == 3:
            raise httpx.ReadTimeout("unknown")
        return httpx.Response(
            200,
            json={
                "id": f"r{len(calls)}",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
        )

    async def run():
        rt = services(tmp_path)
        route = candidate(
            "responses_api:x:m",
            "responses_api",
            "x",
            "m",
            {
                "endpoint": "https://test",
                "responses": {"supports_previous_response_id": True},
            },
        )
        config = configuration("p", [route])
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            first = await rt.execute(
                context, Prompts(TextPrompt("user", "one")), config, gateway, 5
            )
            rt.commit(context, first)
            with pytest.raises(httpx.ReadTimeout):
                await rt.execute(
                    context, Prompts(TextPrompt("user", "two")), config, gateway, 5
                )
            rt = services(tmp_path)
            await rt.execute(
                context, Prompts(TextPrompt("user", "new intent")), config, gateway, 5
            )
        assert "previous_response_id" not in calls[3]

    asyncio.run(run())


def test_cross_model_fallback_commits_with_existing_model_usage(tmp_path):
    from TIYA.llm.model import TokenUsage

    async def run():
        rt = services(tmp_path)
        context = Context(
            TextPrompt("user", "old"),
            TextPrompt("assistant", "old answer"),
            TokenUsage(10, 2, 0, 12, "large"),
        )
        config = configuration(
            "p",
            [
                candidate(
                    "chat_completions:a:large",
                    "chat_completions",
                    "a",
                    "large",
                    {"endpoint": "https://primary"},
                ),
                candidate(
                    "chat_completions:b:small",
                    "chat_completions",
                    "b",
                    "small",
                    {"endpoint": "https://backup"},
                ),
            ],
        )

        def handler(request):
            if request.url.host == "primary":
                return httpx.Response(
                    402, json={"error": {"code": "insufficient_quota"}}
                )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "new"}}],
                    "usage": {"prompt_tokens": 15, "completion_tokens": 1},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            pending = await rt.execute(
                context, Prompts(TextPrompt("user", "continue")), config, gateway, 5
            )
            rt.commit(context, pending)
        assert context._messages[-1].text == "new"
        assert context.usage.model == "small"
        assert context.usage.models == ("large", "small")
        assert context.usage.input_t == 25
        assert context.usage.total_t == 28
        assert context.usage.request_ids == (pending.result.request_id,)
        assert context.usage.sources["input"] == "mixed"
        assert len(context._messages) == 4

    asyncio.run(run())
