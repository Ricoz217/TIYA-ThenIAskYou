from __future__ import annotations

import asyncio
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class RuntimeOverloadedError(RuntimeError):
    pass


class RelatednessRuntime:
    def __init__(
        self,
        *,
        realtime_workers: int = 1,
        maintenance_workers: int = 1,
        queue_limit: int = 128,
    ):
        self._realtime_executor = ThreadPoolExecutor(
            max_workers=realtime_workers,
            thread_name_prefix="tiya-relatedness",
        )
        self._maintenance_executor = ThreadPoolExecutor(
            max_workers=maintenance_workers,
            thread_name_prefix="tiya-relatedness-maintenance",
        )
        self._realtime_slots = asyncio.Semaphore(max(1, queue_limit))
        self._maintenance_lock = asyncio.Lock()
        self._maintenance_tasks: dict[str, asyncio.Task[Any]] = {}
        self._closed = False

    async def run_realtime(
        self,
        function: Callable[..., T],
        *args: Any,
        enqueue_timeout: float | None = 0.25,
    ) -> T:
        if self._closed:
            raise RuntimeError("relatedness runtime is closed")
        if enqueue_timeout is not None and enqueue_timeout <= 0:
            if self._realtime_slots.locked():
                raise RuntimeOverloadedError("relatedness realtime queue is full")
            await self._realtime_slots.acquire()
        elif enqueue_timeout is None:
            await self._realtime_slots.acquire()
        else:
            try:
                await asyncio.wait_for(
                    self._realtime_slots.acquire(),
                    timeout=enqueue_timeout,
                )
            except TimeoutError as exc:
                raise RuntimeOverloadedError(
                    "relatedness realtime queue is full"
                ) from exc
        try:
            loop = asyncio.get_running_loop()
            call = functools.partial(function, *args)
            return await loop.run_in_executor(self._realtime_executor, call)
        finally:
            self._realtime_slots.release()

    async def _execute_maintenance(
        self,
        key: str,
        function: Callable[..., T],
        args: tuple[Any, ...],
    ) -> T:
        try:
            loop = asyncio.get_running_loop()
            call = functools.partial(function, *args)
            return await loop.run_in_executor(self._maintenance_executor, call)
        finally:
            async with self._maintenance_lock:
                current = self._maintenance_tasks.get(key)
                if current is asyncio.current_task():
                    self._maintenance_tasks.pop(key, None)

    async def run_maintenance(
        self,
        key: str,
        function: Callable[..., T],
        *args: Any,
    ) -> T:
        if self._closed:
            raise RuntimeError("relatedness runtime is closed")
        async with self._maintenance_lock:
            task = self._maintenance_tasks.get(key)
            if task is None:
                task = asyncio.create_task(
                    self._execute_maintenance(key, function, args)
                )
                self._maintenance_tasks[key] = task
        return await asyncio.shield(task)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._maintenance_lock:
            tasks = tuple(self._maintenance_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._realtime_executor.shutdown(wait=False, cancel_futures=True)
        self._maintenance_executor.shutdown(wait=False, cancel_futures=True)


_RUNTIME: RelatednessRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def get_relatedness_runtime(
    *,
    realtime_workers: int = 1,
    maintenance_workers: int = 1,
    queue_limit: int = 128,
) -> RelatednessRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = RelatednessRuntime(
                realtime_workers=realtime_workers,
                maintenance_workers=maintenance_workers,
                queue_limit=queue_limit,
            )
    return _RUNTIME


async def close_relatedness_runtime() -> None:
    global _RUNTIME
    with _RUNTIME_LOCK:
        runtime = _RUNTIME
        _RUNTIME = None
    if runtime is not None:
        await runtime.close()
