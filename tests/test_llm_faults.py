import asyncio
import sqlite3

import httpx
import pytest

from llm_config_fixtures import configuration, candidate
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, Prompts, TextPrompt, ContextOverflowError
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


@pytest.mark.parametrize("stage", ["begin_request", "observe_usage", "finish_request"])
def test_accounting_failure_never_regenerates(tmp_path, monkeypatch, stage):
    ledger = Ledger(tmp_path / "usage.sqlite3")
    rt = Runtime(StateStore(tmp_path / "runtime.sqlite3"), ledger)
    calls = []

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk fault")

    monkeypatch.setattr(ledger, stage, fail)

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        config = configuration(
            "p",
            [
                candidate(
                    "chat_completions:a:m",
                    "chat_completions",
                    "a",
                    "m",
                    {"endpoint": "https://test"},
                )
            ],
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            with pytest.raises(sqlite3.OperationalError):
                await rt.execute(
                    Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5
                )
        assert len(calls) == (0 if stage == "begin_request" else 1)

    asyncio.run(run())


def test_smaller_fallback_uses_server_overflow_without_mutating_context(tmp_path):
    ledger = Ledger(tmp_path / "usage.sqlite3")
    rt = Runtime(StateStore(tmp_path / "runtime.sqlite3"), ledger)
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.host == "backup":
            return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})

        return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})

    async def run():
        primary = candidate(
            "chat_completions:a:m",
            "chat_completions",
            "a",
            "m",
            {"endpoint": "https://primary"},
        )
        backup = candidate(
            "chat_completions:b:m",
            "chat_completions",
            "b",
            "m",
            {"endpoint": "https://backup"},
            max_context=1,
        )
        larger = candidate(
            "chat_completions:c:m",
            "chat_completions",
            "c",
            "m",
            {"endpoint": "https://larger"},
            max_context=100000,
        )
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            with pytest.raises(ContextOverflowError, match="chat_completions:b:m"):
                await rt.execute(
                    context,
                    Prompts(TextPrompt("user", "hi")),
                    configuration("p", [primary, backup, larger]),
                    gateway,
                    5,
                )
        assert [request.url.host for request in calls] == ["primary", "backup"]
        assert not context

    asyncio.run(run())


def test_unknown_gateway_does_not_send_parent_and_store_false_disables_it(tmp_path):
    async def run():
        rt = Runtime(
            StateStore(tmp_path / "runtime.sqlite3"), Ledger(tmp_path / "usage.sqlite3")
        )
        bodies = []

        def handler(request):
            import json

            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "r", "output": []})

        route = candidate(
            "responses_api:x:m", "responses_api", "x", "m", {"endpoint": "https://test"}
        )
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            for _ in range(2):
                result = await rt.execute(
                    context,
                    Prompts(TextPrompt("user", "hi")),
                    configuration("p", [route]),
                    gateway,
                    5,
                )
                rt.commit(context, result)
            route.provider_options["responses"] = {
                "supports_previous_response_id": True
            }
            route.model_params["store"] = False
            await rt.execute(
                context,
                Prompts(TextPrompt("user", "hi")),
                configuration("p", [route]),
                gateway,
                5,
            )
        assert all("previous_response_id" not in body for body in bodies)
        assert bodies[-1]["store"] is False

    asyncio.run(run())


def test_invalid_tool_json_still_records_non_stream_usage(tmp_path):
    async def run():
        rt = Runtime(
            StateStore(tmp_path / "runtime.sqlite3"), Ledger(tmp_path / "usage.sqlite3")
        )
        route = candidate(
            "chat_completions:x:m",
            "chat_completions",
            "x",
            "m",
            {"endpoint": "https://test"},
        )
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c", "function": {"name": "f", "arguments": "{"}}
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body)
            )
        ) as client:

            async def gateway(route):
                return Gateway(client)

            with pytest.raises(ValueError):
                await rt.execute(
                    Context(),
                    Prompts(TextPrompt("user", "hi")),
                    configuration("p", [route]),
                    gateway,
                    5,
                )
        row = rt.ledger.query_requests()[0]
        assert row["reported"]["input"] == 20
        assert row["reported"]["output"] == 5
        assert row["remote_state"] == "completed"

    asyncio.run(run())


def test_failed_attempt_is_mirrored_with_the_same_request_id(tmp_path):
    async def run():
        primary = Ledger(tmp_path / "usage.sqlite3")
        subscriber = Ledger(tmp_path / "subscriber.sqlite3")
        rt = Runtime(
            StateStore(tmp_path / "runtime.sqlite3"), primary, subscribers=[subscriber]
        )
        route = candidate(
            "chat_completions:x:m",
            "chat_completions",
            "x",
            "m",
            {"endpoint": "https://test"},
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    500, json={"error": {"code": "server_error"}}
                )
            )
        ) as client:

            async def gateway(route):
                return Gateway(client)

            with pytest.raises(Exception):
                await rt.execute(
                    Context(),
                    Prompts(TextPrompt("user", "hi")),
                    configuration("p", [route]),
                    gateway,
                    5,
                )
        assert primary.query_requests()[0]["id"] == subscriber.query_requests()[0]["id"]
        assert subscriber.query_requests()[0]["remote_state"] == "unknown"

    asyncio.run(run())
