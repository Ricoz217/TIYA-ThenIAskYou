import asyncio
import json

import httpx

from llm_config_fixtures import configuration, candidate
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, Prompts, TextPrompt
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


def test_explicit_background_job_is_cancelled_by_protocol(tmp_path):
    calls = []
    ready = asyncio.Event()

    async def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"id": "r1", "status": "cancelled"})
        if request.method == "GET":
            ready.set()
            await asyncio.Event().wait()
        return httpx.Response(200, json={"id": "r1", "status": "queued", "output": []})

    async def run():
        ledger = Ledger(tmp_path / "usage.sqlite3")
        rt = Runtime(StateStore(tmp_path / "runtime.sqlite3"), ledger)
        route = candidate(
            "responses_api:x:m",
            "responses_api",
            "x",
            "m",
            {
                "endpoint": "https://test/responses",
                "responses": {"supports_background": True},
            },
            {"background": True},
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def gateway(route):
                return Gateway(client)

            task = asyncio.create_task(
                rt.execute(
                    Context(),
                    Prompts(TextPrompt("user", "hi")),
                    configuration("p", [route]),
                    gateway,
                    10,
                )
            )
            await asyncio.wait_for(ready.wait(), 3)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert ("POST", "/responses/r1/cancel") in calls
        model = next(
            r
            for r in ledger.query_requests()
            if r["identity"].get("operation") == "responses_api"
        )
        assert model["remote_state"] == "cancelled"

    asyncio.run(run())
