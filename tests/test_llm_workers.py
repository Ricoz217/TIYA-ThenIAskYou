import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Event, current_thread

import pytest


def test_llm_work_is_isolated_from_default_pool_and_preserves_context(monkeypatch):
    from TIYA.llm import workers

    marker = ContextVar("llm_request")
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-test") as pool:
        monkeypatch.setattr(workers, "LLM_EXECUTOR", pool)

        async def run():
            def default_pool_was_used(*args, **kwargs):
                raise AssertionError("LLM work used the asyncio default executor")

            monkeypatch.setattr(asyncio, "to_thread", default_pool_was_used)
            marker.set("request-123")
            result = await workers.run_sync(
                lambda value, *, suffix: (
                    current_thread().name,
                    marker.get(),
                    value + suffix,
                ),
                "a",
                suffix="b",
            )
            assert result[0].startswith("llm-test")
            assert result[1:] == ("request-123", "ab")

        asyncio.run(run())


def test_llm_pool_enforces_worker_limit_and_cancels_queued_work(monkeypatch):
    from TIYA.llm import workers

    started, release, second_started = Event(), Event(), Event()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-test") as pool:
        monkeypatch.setattr(workers, "LLM_EXECUTOR", pool)

        def blocking():
            started.set()
            assert release.wait(5)

        async def run():
            first = asyncio.create_task(workers.run_sync(blocking))
            async with asyncio.timeout(5):
                while not started.is_set():
                    await asyncio.sleep(0)
            second = asyncio.create_task(workers.run_sync(second_started.set))
            await asyncio.sleep(0)
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert not second_started.is_set()
            # Cancelling an awaiter cannot stop a function already running in a thread.
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not release.is_set()
            release.set()

        try:
            asyncio.run(run())
        finally:
            release.set()
    assert not second_started.is_set()


def test_llm_worker_exceptions_propagate_and_pool_survives_multiple_event_loops(
    monkeypatch,
):
    from TIYA.llm import workers

    def fail():
        raise OSError("disk unavailable")

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(workers, "LLM_EXECUTOR", pool)
        with pytest.raises(OSError, match="disk unavailable"):
            asyncio.run(workers.run_sync(fail))
        assert asyncio.run(workers.run_sync(lambda: 42)) == 42


def test_global_executor_shutdown_also_drains_llm_pool(monkeypatch):
    from TIYA import executor

    stopped = []

    class Pool:
        def __init__(self, name):
            self.name = name

        def shutdown(self, wait=True, *, cancel_futures=False):
            stopped.append((self.name, wait, cancel_futures))

    for name in (
        "GLOBAL_EXECUTOR",
        "AGENT_EXECUTOR",
        "FILE_CACHE_EXECUTOR",
        "AUTOMAPPING_EXECUTOR",
        "LLM_EXECUTOR",
    ):
        monkeypatch.setattr(executor, name, Pool(name))
    executor.shutdown_all()
    assert ("LLM_EXECUTOR", True, False) in stopped


def test_chat_request_uses_llm_pool_without_asyncio_default_executor(monkeypatch):
    import httpx
    from test_llm_preset_calls import mock_model

    from TIYA import executor
    from TIYA.LLM_connect import Chat

    def response(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            },
        )

    mock_model(monkeypatch, response)

    async def run():
        loop = asyncio.get_running_loop()
        submit = loop.run_in_executor
        used = []

        def checked(pool, function, *args):
            assert pool is not None, (
                "The Chat path submitted work to the default executor"
            )
            used.append(pool)
            return submit(pool, function, *args)

        monkeypatch.setattr(loop, "run_in_executor", checked)
        async with Chat(
            endpoint="https://model.invalid/chat",
            model="model",
            model_params={"stream": False},
        ) as chat:
            result = await chat.ask("hello", timeout=10)
            assert result is not None
            assert result[0].text == "done"
        assert executor.LLM_EXECUTOR in used

    asyncio.run(run())
