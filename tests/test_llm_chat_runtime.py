import asyncio

import httpx

from TIYA.LLM_connect import Chat, ChatConfig


def config():
    return ChatConfig(
        endpoint="https://test/chat",
        model="m",
        token="k",
        api_provider="openai",
        model_params={"stream": False},
        log_off=True,
    )


def test_queue_and_nowait_commit_once_and_cancel_never_commits(monkeypatch):
    from TIYA import LLM_connect

    real_client = httpx.AsyncClient
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def handler(request):
        calls.append(request)
        started.set()
        await release.wait()
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(
        LLM_connect.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler)),
    )

    async def run():
        chat = Chat(config=config())
        try:
            task = asyncio.create_task(
                chat.ask("first", task_id="cancel-me", timeout=5)
            )
            await started.wait()
            await chat.stop_from_id("cancel-me")
            assert await task is None
            assert not chat._context._messages
            release.set()
            response = await chat.ask("second", no_wait=True, timeout=5)
            assert response[0].text == "ok"
            assert len(chat._context._messages) == 2
            response = await chat.ask("third", timeout=5)
            assert len(chat._context._messages) == 4
            assert len(calls) == 3
        finally:
            await chat.close()

    asyncio.run(run())


def test_shared_context_serializes_two_chats_and_copies_isolate_parameters(monkeypatch):
    from TIYA.llm import chat as module
    from TIYA.llm.model import Context, TextPrompt

    real_client = httpx.AsyncClient
    active, peak = 0, 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler)),
    )

    async def run():
        context = Context()
        first, second = (
            Chat(config=config(), context=context),
            Chat(config=config(), context=context),
        )
        copied = first.copy()
        try:
            copied.model["parameter"]["custom"] = {"value": 1}
            assert "custom" not in first.model["parameter"]
            assert copied._context.session_id != context.session_id
            await asyncio.gather(
                first.ask("one", timeout=5), second.ask("two", timeout=5)
            )
            assert peak == 1
            assert len(context._messages) == 4
            assert first.context._messages[0] is not context._messages[0]
            assert await first.get_context() is context
            first.back_to_last_round()
            assert len(context._messages) == 2
            await first.clear()
            assert not context
        finally:
            await first.close()
            await second.close()
            await copied.close()

    asyncio.run(run())


def test_stop_all_does_not_start_queued_requests(monkeypatch):
    from TIYA.llm import chat as module

    real_client = httpx.AsyncClient
    ready = asyncio.Event()
    calls = []

    async def handler(request):
        calls.append(request)
        ready.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler)),
    )

    async def run():
        async with Chat(config=config()) as chat:
            tasks = [asyncio.create_task(chat.ask(str(i), timeout=5)) for i in range(5)]
            await asyncio.wait_for(ready.wait(), 2)
            assert await chat.ask("skip", queue=False, timeout=1) is None
            await chat.stop_all()
            assert await asyncio.gather(*tasks) == [None] * 5
            assert len(calls) == 1
            assert not chat._context

    asyncio.run(run())
