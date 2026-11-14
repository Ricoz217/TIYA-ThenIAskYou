import asyncio
import json

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.chat import Chat
from TIYA.llm.gateway import Gateway, HTTPFailure
from TIYA.llm.model import Context, ContextOverflowError, TextPrompt
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger
from TIYA.llm.usage_model import TokenUsage


def preset(model="old", *, fallback=False, backup_window=1000):
    routes = [candidate(
        f"chat_completions:a:{model}", "chat_completions", "a", model,
        {"endpoint": "https://test/a"}, {"stream": False}, max_context=1000,
    )]
    if fallback:
        routes.append(candidate(
            "chat_completions:b:backup", "chat_completions", "b", "backup",
            {"endpoint": "https://test/b"}, {"stream": False}, max_context=backup_window,
        ))

    return configuration("test", routes, auto_compress_rate=0.7)


def runtime(tmp_path):
    return Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))


def response(total=100):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": "answer"}}],
        "usage": {"prompt_tokens": total - 50, "completion_tokens": 50, "total_tokens": total},
    })


def test_queued_request_checks_the_usage_committed_while_it_waited(tmp_path):
    async def run():
        chat = Chat(config=preset(), log_off=True)
        chat._runtime = runtime(tmp_path)
        started = asyncio.Event()
        release = asyncio.Event()
        requests = []

        async def handler(request):
            requests.append(request)
            started.set()
            await release.wait()
            return response(800)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            chat._gateway = gateway
            try:
                first = asyncio.create_task(chat.ask("first", timeout=5))
                await started.wait()
                second = asyncio.create_task(chat.ask("queued", timeout=5))
                async with asyncio.timeout(1):
                    while len(chat._ask_tasks) != 2:
                        await asyncio.sleep(0)

                release.set()
                assert (await first)[0].text == "answer"
                with pytest.raises(ContextOverflowError):
                    await second

                assert len(requests) == 1
                assert len(chat._runtime.ledger.query_requests()) == 1
                assert chat.last_outcome.category == "context_overflow"
                assert chat.last_outcome.remote_state == "not_sent"

            finally:
                release.set()
                await chat.close()

    asyncio.run(run())


def test_queue_false_admission_is_reserved_before_async_estimation(tmp_path, monkeypatch):
    import TIYA.llm.runtime as runtime_module

    async def run():
        chat = Chat(config=preset(), context=Context(TextPrompt("user", "history")), log_off=True)
        chat._runtime = runtime(tmp_path)
        estimating = asyncio.Event()
        release = asyncio.Event()
        requests = []
        estimates = []

        async def estimate(context):
            estimates.append(context)
            estimating.set()
            await release.wait()
            return {"input": 10}

        monkeypatch.setattr(runtime_module, "estimate_async", estimate)
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: requests.append(request) or response()
        )) as client:
            async def gateway(route):
                return Gateway(client)

            chat._gateway = gateway
            first = asyncio.create_task(chat.ask("first", queue=False, timeout=5))
            try:
                await asyncio.wait_for(estimating.wait(), 1)
                assert await asyncio.wait_for(chat.ask("second", queue=False, timeout=5), 0.2) is None
                release.set()
                assert (await first)[0].text == "answer"
                assert len(estimates) == 1
                assert len(requests) == 1

            finally:
                release.set()
                await asyncio.gather(first, return_exceptions=True)
                await chat.close()

    asyncio.run(run())


@pytest.mark.parametrize("inflight", [False, True])
def test_model_switch_does_not_reuse_old_model_measurements(tmp_path, inflight):
    async def run():
        chat = Chat(config=preset(), log_off=True)
        chat._runtime = runtime(tmp_path)
        started = asyncio.Event()
        release = asyncio.Event()
        models = []

        async def handler(request):
            model = json.loads(request.content)["model"]
            models.append(model)
            if model == "old":
                started.set()
                await release.wait()
                return response(800)

            return response()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            chat._gateway = gateway
            try:
                first = asyncio.create_task(chat.ask("first", timeout=5))
                await started.wait()
                if inflight:
                    chat.setting(preset("new"))

                release.set()
                assert await first is not None
                if not inflight:
                    chat.setting(preset("new"))

                assert (await chat.ask("second", timeout=5))[0].text == "answer"
                assert models == ["old", "new"]
                assert chat._last_usage.model == "new"

            finally:
                release.set()
                await chat.close()

    asyncio.run(run())


def test_successful_fallback_reuses_its_real_usage_without_tokenizer(tmp_path, monkeypatch):
    import TIYA.llm.runtime as module

    async def run():
        chat = Chat(config=preset(fallback=True), log_off=True)
        chat._runtime = runtime(tmp_path)
        paths = []

        def handler(request):
            paths.append(request.url.path)
            if request.url.path == "/a":
                return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})

            return response()

        async def forbidden(context):
            pytest.fail("a successful backup must reuse its reported usage on the next turn")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            chat._gateway = gateway
            try:
                assert await chat.ask("first", timeout=5) is not None
                monkeypatch.setattr(module, "estimate_async", forbidden)
                assert await chat.ask("second", timeout=5) is not None
                assert paths == ["/a", "/b", "/b"]

            finally:
                await chat.close()

    asyncio.run(run())


def test_direct_runtime_reuses_committed_usage_without_a_chat(tmp_path, monkeypatch):
    import TIYA.llm.runtime as module

    async def run():
        rt = runtime(tmp_path)
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response(800))) as client:
            async def gateway(route):
                return Gateway(client)

            pending = await rt.execute(context, TextPrompt("user", "first"), preset(), gateway, 5)
            await rt.commit_async(context, pending)

            async def forbidden(context):
                pytest.fail("committed reported usage must take priority over tokenizer")

            monkeypatch.setattr(module, "estimate_async", forbidden)
            with pytest.raises(ContextOverflowError):
                await rt.execute(context, TextPrompt("user", "second"), preset(), gateway, 5)

            assert len(rt.ledger.query_requests()) == 1

    asyncio.run(run())


def test_actual_fallback_window_uses_one_estimate_when_measurement_source_changes(tmp_path, monkeypatch):
    import TIYA.llm.runtime as module

    async def run():
        rt = runtime(tmp_path)
        context = Context(TextPrompt("user", "history"))
        config = preset(fallback=True, backup_window=20)
        estimates = []
        requests = []

        async def estimate(context):
            estimates.append(context)
            return {"input": 30}

        monkeypatch.setattr(module, "estimate_async", estimate)
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
        )) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(ContextOverflowError, match="backup"):
                await rt.execute(
                    context, TextPrompt("user", "next"), config, gateway, 5,
                    context_usage=TokenUsage(model="old", reported={"total": 10}),
                    usage_signature=context.signature,
                    usage_scope=config.primary.scope,
                )

        assert len(requests) == 1
        assert len(estimates) == 1
        assert len(rt.ledger.query_requests()) == 1

    asyncio.run(run())


def test_fallbacks_share_the_same_local_estimate(tmp_path, monkeypatch):
    import TIYA.llm.runtime as module

    async def run():
        rt = runtime(tmp_path)
        estimates = []

        async def estimate(context):
            estimates.append(context)
            return {"input": 30}

        monkeypatch.setattr(module, "estimate_async", estimate)
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
        )) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(HTTPFailure):
                await rt.execute(Context(TextPrompt("user", "history")), TextPrompt("user", "next"),
                                 preset(fallback=True), gateway, 5)

        assert len(estimates) == 1

    asyncio.run(run())


def test_watermark_reads_the_last_request_not_cumulative_tool_round_usage():
    from TIYA.llm.usage import reported_context_tokens

    usage = TokenUsage(model="old", reported={"total": 800})
    usage += TokenUsage(model="old", reported={"total": 100})
    assert usage.reported["total"] == 900
    assert reported_context_tokens(usage, preset().primary) == 100


def test_unrelated_settings_keep_real_measurements_valid(tmp_path, monkeypatch):
    import TIYA.llm.runtime as module

    async def run():
        chat = Chat(config=preset(), log_off=True)
        chat._runtime = runtime(tmp_path)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response())) as client:
            async def gateway(route):
                return Gateway(client)

            async def forbidden(context):
                pytest.fail("changing a threshold does not invalidate model measurements")

            chat._gateway = gateway
            try:
                assert await chat.ask("first", timeout=5) is not None
                updated = preset()
                updated.primary.max_context = 2000
                updated.auto_compress_rate = 0.6
                updated.refresh_fingerprints()
                chat.setting(updated)
                monkeypatch.setattr(module, "estimate_async", forbidden)
                assert await chat.ask("second", timeout=5) is not None

            finally:
                await chat.close()

    asyncio.run(run())


def test_runtime_does_not_reuse_old_measurement_after_context_clear(tmp_path, monkeypatch):
    import TIYA.llm.runtime as module

    async def run():
        rt = runtime(tmp_path)
        context = Context()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response(800))) as client:
            async def gateway(route):
                return Gateway(client)

            first = await rt.execute(context, TextPrompt("user", "first"), preset(), gateway, 5)
            await rt.commit_async(context, first)
            context.clear()

            async def forbidden(context):
                pytest.fail("an empty Context must not start tokenizer")

            monkeypatch.setattr(module, "estimate_async", forbidden)
            second = await rt.execute(context, TextPrompt("user", "new session"), preset(), gateway, 5)
            assert second.result.prompts[0].text == "answer"
            assert len(rt.ledger.query_requests()) == 2

    asyncio.run(run())


def test_missing_latest_reported_usage_does_not_reuse_an_older_sample():
    from TIYA.llm.usage import reported_context_tokens

    usage = TokenUsage(model="old", reported={"total": 800})
    usage += TokenUsage(model="old", estimated={"input": 10, "output": 2})
    assert reported_context_tokens(usage, preset().primary) is None


def test_same_model_on_a_different_connection_does_not_reuse_measurement():
    from TIYA.llm.usage import reported_context_tokens

    original = preset().primary
    other = candidate("chat_completions:other:old", "chat_completions", "other", "old",
                      {"endpoint": "https://other"}, max_context=1000)
    usage = TokenUsage(model="old", reported={"total": 800})
    assert reported_context_tokens(usage, other, original.scope) is None
    renamed_endpoint = preset().primary
    renamed_endpoint.endpoint = "https://different-service/test"
    renamed_endpoint.refresh_fingerprints()
    assert renamed_endpoint.ref == original.ref
    assert reported_context_tokens(usage, renamed_endpoint, original.scope) is None


def test_runtime_ignores_a_measurement_bound_to_another_context(tmp_path, monkeypatch):
    import TIYA.llm.runtime as module

    async def run():
        rt = runtime(tmp_path)
        estimates = []
        context = Context(TextPrompt("user", "history"))

        async def estimate(snapshot):
            estimates.append(snapshot)
            return {"input": 10}

        monkeypatch.setattr(module, "estimate_async", estimate)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response())) as client:
            async def gateway(route):
                return Gateway(client)

            config = preset()
            pending = await rt.execute(
                context, TextPrompt("user", "next"), config, gateway, 5,
                context_usage=TokenUsage(model="old", reported={"total": 800}),
                usage_signature="stale-context",
                usage_scope=config.primary.scope,
            )
            assert pending.result.prompts[0].text == "answer"
            assert len(estimates) == 1

    asyncio.run(run())
