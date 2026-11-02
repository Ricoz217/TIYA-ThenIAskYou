import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.gateway import Gateway, HTTPFailure
from TIYA.llm.model import Context, ContextOverflowError, Prompts, TextPrompt
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger
from TIYA.utils import fingerprint


def setup_routes(tmp_path, *, attempts=6, windows=(None, None, None)):
    routes = [
        candidate(
            f"chat_completions:{name}:model-{name}",
            "chat_completions",
            name,
            f"model-{name}",
            {"endpoint": f"https://{name}/chat"},
            {"stream": False},
            max_context=window,
        )
        for name, window in zip(("a", "b", "c"), windows)
    ]
    config = configuration("main", routes, options={"max_attempts": attempts})
    runtime = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))
    context = Context()
    key = fingerprint([context.session_id, config.revision])
    return runtime, config, context, key


def rejected():
    return httpx.Response(429, json={"error": {"code": "insufficient_quota"}})


def accepted():
    return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})


@pytest.mark.parametrize("selected,expected", [(0, ["a", "b", "c"]), (2, ["c"])])
def test_failed_last_candidate_deletes_cooldown_and_next_call_starts_primary(tmp_path, selected, expected):
    runtime, config, context, key = setup_routes(tmp_path)
    if selected:
        until = time.time() + 1000
        runtime.state.put("cooldown", key, {"index": selected, "until": until}, until)

    calls = []
    healthy = [False]

    def handler(request):
        calls.append(request.url.host)
        if healthy[0] and request.url.host == "a":
            assert runtime.state.get("cooldown", key) is None
            return accepted()

        return rejected()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(HTTPFailure):
                await runtime.execute(context, Prompts(TextPrompt("user", "first")), config, gateway, 5)

            assert not context
            assert calls == expected
            assert runtime.state.get("cooldown", key) is None
            healthy[0] = True
            pending = await runtime.execute(context, Prompts(TextPrompt("user", "second")), config, gateway, 5)
            assert pending.route_index == 0
            runtime.commit(context, pending)
            assert runtime.state.get("cooldown", key) is None

    asyncio.run(run())
    assert calls == expected + ["a"]


def test_failing_backup_advances_and_refreshes_cooldown_before_sending_next(tmp_path):
    runtime, config, context, key = setup_routes(tmp_path)
    previous_until = time.time() + 120
    runtime.state.put("cooldown", key, {"index": 1, "until": previous_until}, previous_until)
    calls = []
    deadlines = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "b":
            return rejected()

        if request.url.host == "c":
            cooldown = runtime.state.get("cooldown", key)
            assert cooldown["index"] == 2
            assert cooldown["until"] > previous_until
            deadlines.append(cooldown["until"])

        return accepted()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            first = await runtime.execute(context, Prompts(TextPrompt("user", "one")), config, gateway, 5)
            runtime.commit(context, first)
            second = await runtime.execute(context, Prompts(TextPrompt("user", "two")), config, gateway, 5)
            runtime.commit(context, second)
            assert deadlines[0] == deadlines[1]
            runtime.state.put("cooldown", key, {"index": 2, "until": 0}, 0)
            await runtime.execute(context, Prompts(TextPrompt("user", "three")), config, gateway, 5)

    asyncio.run(run())
    assert calls == ["b", "c", "c", "a"]


@pytest.mark.parametrize("attempts,expected_index", [(1, 1), (2, 2), (3, None), (7, None)])
def test_candidates_are_not_revisited_even_when_total_budget_remains(tmp_path, attempts, expected_index):
    runtime, config, context, key = setup_routes(tmp_path, attempts=attempts)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return rejected()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(HTTPFailure) as caught:
                await runtime.execute(context, Prompts(TextPrompt("user", "hi")), config, gateway, 5)

            assert caught.value.llm_outcome.category == "quota"
            rows = runtime.ledger.query_requests(call_id=caught.value.llm_outcome.call_id)
            assert len(rows) == min(attempts, 3)
            assert len({row["id"] for row in rows}) == min(attempts, 3)
            assert not context
            cooldown = runtime.state.get("cooldown", key)
            if expected_index is None:
                assert cooldown is None

            else:
                assert cooldown["index"] == expected_index

    asyncio.run(run())
    assert calls == ["a", "b", "c"][:attempts]


@pytest.mark.parametrize("status,code,failed_sends", [
    (401, "invalid_api_key", 1),
    (429, "rate_limit_exceeded", 2),
])
def test_single_candidate_failure_stops_and_next_call_retries_primary(tmp_path, status, code, failed_sends):
    runtime, config, context, key = setup_routes(tmp_path, windows=(None,))
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if len(calls) <= failed_sends:
            return httpx.Response(status, json={"error": {"code": code}})

        return accepted()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(HTTPFailure):
                await runtime.execute(context, Prompts(TextPrompt("user", "first")), config, gateway, 5)

            assert calls == ["a"] * failed_sends
            assert runtime.state.get("cooldown", key) is None
            assert not context
            pending = await runtime.execute(context, Prompts(TextPrompt("user", "second")), config, gateway, 5)
            runtime.commit(context, pending)

    asyncio.run(run())
    assert calls == ["a"] * (failed_sends + 1)


@pytest.mark.parametrize("kind", ["unknown", "partial"])
def test_unsafe_last_candidate_failure_does_not_wrap_or_clear_cooldown(tmp_path, kind):
    runtime, config, context, key = setup_routes(tmp_path)
    config.candidates[2].model_params["stream"] = kind == "partial"
    config.refresh_fingerprints()
    key = fingerprint([context.session_id, config.revision])
    until = time.time() + 1000
    runtime.state.put("cooldown", key, {"index": 2, "until": until}, until)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if kind == "unknown":
            return httpx.Response(503, json={"error": {"message": "unavailable"}})

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=(
            'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            'data: {"error":{"code":"rate_limit_exceeded"}}\n\n'
        ))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(Exception):
                await runtime.execute(context, Prompts(TextPrompt("user", "hi")), config, gateway, 5)

            assert runtime.state.get("cooldown", key) == {"index": 2, "until": until}

    asyncio.run(run())
    assert calls == ["c"]


def test_local_overflow_of_selected_candidate_stops_without_trying_larger_candidates(tmp_path):
    runtime, config, context, key = setup_routes(tmp_path, windows=(1, 100000, 100000))
    calls = []

    async def gateway(route):
        calls.append(route.connection_name)
        raise AssertionError("窗口不足不能发送，也不能自动换候选")

    with pytest.raises(ContextOverflowError, match="model-a"):
        asyncio.run(runtime.execute(context, Prompts(TextPrompt("user", "hi")), config, gateway, 5))

    assert calls == []
    assert runtime.ledger.query_requests() == []
    assert runtime.state.get("cooldown", key) is None


@pytest.mark.parametrize("selected", [0, 1])
def test_server_context_overflow_is_reported_for_actual_candidate_without_fallback(tmp_path, selected):
    runtime, config, context, key = setup_routes(tmp_path)
    if selected:
        until = time.time() + 1000
        runtime.state.put("cooldown", key, {"index": selected, "until": until}, until)

    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(ContextOverflowError, match=f"model-{'ab'[selected]}") as caught:
                await runtime.execute(context, Prompts(TextPrompt("user", "hi")), config, gateway, 5)

            assert caught.value.llm_outcome.category == "context_overflow"
            assert len(runtime.ledger.query_requests()) == 1
            assert not context

    asyncio.run(run())
    assert calls == ["ab"[selected]]


def test_cancelling_last_candidate_does_not_wrap_or_clear_cooldown(tmp_path):
    runtime, config, context, key = setup_routes(tmp_path)
    until = time.time() + 1000
    runtime.state.put("cooldown", key, {"index": 2, "until": until}, until)
    calls = []

    async def run():
        started = asyncio.Event()

        async def handler(request):
            calls.append(request.url.host)
            started.set()
            await asyncio.Event().wait()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            running = asyncio.create_task(runtime.execute(
                context, Prompts(TextPrompt("user", "hi")), config, gateway, 5
            ))
            await started.wait()
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running

            assert runtime.state.get("cooldown", key) == {"index": 2, "until": until}
            assert not context

    asyncio.run(run())
    assert calls == ["c"]


def test_route_switch_does_not_reset_total_deadline(tmp_path, monkeypatch):
    from TIYA.llm import runtime as module
    runtime, config, context, key = setup_routes(tmp_path)
    clock = [time.monotonic()]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=time.time, monotonic=lambda: clock[0]))
    real_advance = runtime._advance_route
    calls = []

    def delayed_advance(call, index):
        real_advance(call, index)
        clock[0] += 10

    monkeypatch.setattr(runtime, "_advance_route", delayed_advance)

    def handler(request):
        calls.append(request.url.host)
        return rejected()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(TimeoutError):
                await runtime.execute(context, Prompts(TextPrompt("user", "hi")), config, gateway, 5)

            assert runtime.state.get("cooldown", key)["index"] == 1

    asyncio.run(run())
    assert calls == ["a"]
