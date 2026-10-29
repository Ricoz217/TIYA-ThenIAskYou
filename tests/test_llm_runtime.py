import asyncio

import httpx
import pytest

from llm_config_fixtures import configuration, candidate
from TIYA.llm.model import Context, Prompts, TextPrompt


def preset():
    return configuration(
        "p",
        [
            candidate(
                "responses_api:a:m",
                "responses_api",
                "a",
                "m",
                {
                    "endpoint": "https://test/responses",
                    "responses": {"supports_previous_response_id": True},
                },
                {"stream": False},
            )
        ],
    )


def runtime(tmp_path):
    from TIYA.llm.runtime import Runtime
    from TIYA.llm.state import StateStore
    from TIYA.llm.usage import Ledger

    return Runtime(
        StateStore(tmp_path / "runtime.sqlite3"), Ledger(tmp_path / "usage.sqlite3")
    )


def test_only_committed_matching_checkpoint_is_used_and_invalid_parent_replays(
    tmp_path,
):
    from TIYA.llm.gateway import Gateway

    requests = []

    def handler(request):
        import json

        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 3:
            return httpx.Response(
                400, json={"error": {"code": "previous_response_not_found"}}
            )
        return httpx.Response(
            200,
            json={
                "id": f"r{len(requests)}",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
        )

    async def run():
        rt = runtime(tmp_path)
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            first = await rt.execute(
                context, Prompts(TextPrompt("user", "first")), preset(), gateway, 10
            )
            assert "previous_response_id" not in requests[0]
            rt.commit(context, first)
            second = await rt.execute(
                context, Prompts(TextPrompt("user", "second")), preset(), gateway, 10
            )
            assert requests[1]["previous_response_id"] == "r1"
            assert len(requests[1]["input"]) == 1
            rt.commit(context, second)
            third = await rt.execute(
                context, Prompts(TextPrompt("user", "third")), preset(), gateway, 10
            )
            assert "previous_response_id" in requests[2]
            assert "previous_response_id" not in requests[3]
            assert len(requests[3]["input"]) == 5
            rt.commit(context, third)

    asyncio.run(run())


def test_partial_output_or_unknown_execution_never_falls_back(tmp_path):
    from TIYA.llm.gateway import Gateway

    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("remote unknown")

    async def run():
        rt = runtime(tmp_path)
        config = preset()
        config.candidates.append(
            candidate(
                "chat_completions:b:m",
                "chat_completions",
                "b",
                "m",
                {"endpoint": "https://backup"},
            )
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            with pytest.raises(httpx.ReadTimeout):
                await rt.execute(
                    Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 10
                )
        assert len(calls) == 1
        assert rt.ledger.query_requests()[0]["status"] == "unknown"

    asyncio.run(run())


def test_context_mutation_prevents_commit(tmp_path):
    from TIYA.llm.gateway import Gateway

    async def run():
        rt = runtime(tmp_path)
        context = Context()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    json={
                        "id": "r",
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "ok"}],
                            }
                        ],
                    },
                )
            )
        ) as client:

            async def gateway(route):
                return Gateway(client)

            result = await rt.execute(
                context, Prompts(TextPrompt("user", "hi")), preset(), gateway, 10
            )
            context.append(TextPrompt("user", "external mutation"))
            with pytest.raises(RuntimeError, match="Context"):
                rt.commit(context, result)
            assert len(context._messages) == 1

    asyncio.run(run())


def test_platform_request_id_and_reported_model_are_kept_separately(tmp_path):
    from TIYA.llm.gateway import Gateway

    async def run():
        rt = runtime(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200,
                    headers={"x-request-id": "server-request"},
                    json={
                        "id": "response-object",
                        "model": "resolved-model",
                        "output": [],
                    },
                )
            )
        ) as client:

            async def gateway(route):
                return Gateway(client)

            pending = await rt.execute(
                Context(), Prompts(TextPrompt("user", "hi")), preset(), gateway, 5
            )
        record = rt.ledger.get_request(pending.result.request_id)
        assert record["id"] != "server-request"
        assert record["diagnostic"]["platform_request_id"] == "server-request"
        assert record["diagnostic"]["reported_model"] == "resolved-model"

    asyncio.run(run())
