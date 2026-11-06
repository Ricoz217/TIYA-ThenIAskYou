import asyncio
import json

import httpx
import pytest

from TIYA.LLM_connect import Chat, ChatConfig, LLMConfig


def config():
    return ChatConfig(
        [
            LLMConfig(
                "chat_completions",
                "test",
                "https://test/chat",
                "k",
                "m",
                model_params={"stream": False},
            )
        ],
        log_off=True,
    )


@pytest.mark.parametrize("no_wait", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("log_off", [False, True])
def test_round_usage_cache_hit_and_elapsed_are_logged_once(
    tmp_path,
    monkeypatch,
    no_wait,
    stream,
    log_off,
):
    from TIYA import LLM_connect
    from TIYA.llm.model import Prompts, TextPrompt
    from TIYA.logger import Logger, LoggerConfig

    logger = Logger(LoggerConfig(logs_dir=tmp_path / "logs", use_console=False))
    real_client = httpx.AsyncClient
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 800},
    }

    def handler(request):
        if stream:
            event = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}], "usage": usage}
            return httpx.Response(
                200,
                text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )

        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": usage})

    monkeypatch.setattr(
        LLM_connect.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler)),
    )

    async def run():
        settings = config()
        settings.logger = logger
        settings.log_off = log_off
        settings.primary.model_params["stream"] = stream
        chat = Chat(config=settings)
        try:
            # 复用旧消息不应把消息创建至今的时间当作本次请求耗时。
            prompts = Prompts(TextPrompt("user", "hi", create_time="1"))
            response = await chat.ask(prompts, no_wait=no_wait, timeout=5)
            assert response is not None
            assert response[0].text == "ok"
            assert 0 <= response.duration < 5
            assert prompts.time == "1"
            return response.duration

        finally:
            await chat.close()

    try:
        duration = asyncio.run(run())
        logger.flush()
        text = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "logs").rglob("app.log"))
        expected = (
            "本轮消耗 | 总tokens: [1,200]; 输入: [1,000]; 输出: [200]; "
            "输入(缓存): [800]; 缓存命中率: [80.00%]"
        )
        assert text.count(expected) == (0 if log_off else 1)
        assert text.count(f"耗时：[{duration:.2f}]秒") == (0 if log_off else 1)

    finally:
        logger.shutdown()


@pytest.mark.parametrize("log_off", [False, True])
def test_failed_round_logs_elapsed_without_success_usage(tmp_path, monkeypatch, log_off):
    from TIYA import LLM_connect
    from TIYA.logger import Logger, LoggerConfig

    logger = Logger(LoggerConfig(logs_dir=tmp_path / "logs", use_console=False))
    real_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(400, json={"error": {"message": "invalid messages"}})

    monkeypatch.setattr(
        LLM_connect.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler)),
    )

    async def run():
        settings = config()
        settings.logger = logger
        settings.log_off = log_off
        chat = Chat(config=settings)
        try:
            assert await chat.ask("hi", timeout=5) is None

        finally:
            await chat.close()

    try:
        asyncio.run(run())
        logger.flush()
        text = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "logs").rglob("app.log"))
        assert "本轮消耗" not in text
        assert text.count("耗时：[") == (0 if log_off else 1)

    finally:
        logger.shutdown()


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


def test_chat_copies_isolate_parameters_and_can_explicitly_share_history(monkeypatch):
    from TIYA.llm import chat as module
    from TIYA.llm.model import Context

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
        first = Chat(config=config(), context=context)
        second = first.copy(new_context=False)
        copied = first.copy()
        try:
            copied.model["parameter"]["custom"] = {"value": 1}
            assert "custom" not in first.model["parameter"]
            assert copied._context.session_id != context.session_id
            # 共享只保留旧接口语义；跨 Chat 的执行顺序由调用方控制。
            await first.ask("one", timeout=5)
            await second.ask("two", timeout=5)
            assert peak == 1
            assert len(context._messages) == 4
            assert first.context._messages[0] is context._messages[0]
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
