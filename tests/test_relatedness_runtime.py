import asyncio
import time

from TIYA.relatedness.runtime import RelatednessRuntime, RuntimeOverloadedError


def test_realtime_work_runs_outside_event_loop() -> None:
    async def run() -> None:
        runtime = RelatednessRuntime(realtime_workers=1, maintenance_workers=1, queue_limit=2)
        loop_thread = None

        def work() -> int:
            nonlocal loop_thread
            import threading

            loop_thread = threading.get_ident()
            return 42

        current_thread = __import__("threading").get_ident()
        assert await runtime.run_realtime(work) == 42
        assert loop_thread != current_thread
        await runtime.close()

    asyncio.run(run())


def test_realtime_queue_is_bounded() -> None:
    async def run() -> None:
        runtime = RelatednessRuntime(realtime_workers=1, maintenance_workers=1, queue_limit=1)
        release = __import__("threading").Event()
        first = asyncio.create_task(runtime.run_realtime(release.wait))
        await asyncio.sleep(0.02)

        try:
            await runtime.run_realtime(time.sleep, 0.01, enqueue_timeout=0)
        except RuntimeOverloadedError:
            pass
        else:
            raise AssertionError("queue limit was not enforced")

        release.set()
        await first
        await runtime.close()

    asyncio.run(run())


def test_maintenance_calls_with_same_key_are_coalesced() -> None:
    async def run() -> None:
        runtime = RelatednessRuntime(realtime_workers=1, maintenance_workers=1)
        calls = 0

        def work() -> int:
            nonlocal calls
            calls += 1
            time.sleep(0.03)
            return calls

        first, second = await asyncio.gather(
            runtime.run_maintenance("group:hotwords", work),
            runtime.run_maintenance("group:hotwords", work),
        )

        assert first == second == 1
        assert calls == 1
        await runtime.close()

    asyncio.run(run())
