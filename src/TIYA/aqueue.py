"""
aqueue.py
一个大异步队列，作为守门员
"""

from __future__ import annotations

__version__ = "2.0.0"

import asyncio
import traceback
import functools
from dataclasses import dataclass, field
from typing import Coroutine, Callable, Any

from TIYA.logger import get_logger
from TIYA.config import CONFIG_OBSERVER, SETTING_CFG

_log = get_logger()
_DEFAULT_STOP_TIMEOUT = object()


@dataclass(slots=True)
class AqueueTask:
    """把协程和超时封在一起"""
    coro: Coroutine
    timeout: float | int | Callable[..., float | int]
    result: list[Any] = field(default_factory=list)
    exception: list[Exception] = field(default_factory=list)
    parsed_timeout: int | float = field(init=False)
    event: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self):
        if callable(self.timeout):
            self.parsed_timeout = self.timeout()

        else:
            self.parsed_timeout = self.timeout

    def __await__(self):
        return self.coro.__await__()

    async def wait(self):
        await self.event.wait()

    @property
    def done(self) -> bool:
        return self.event.is_set()

    @property
    def cancelled(self) -> bool:
        return bool(self.exception)

    @property
    def name(self) -> str:
        return self.coro.__name__


class AqueueWorker:
    """消费者"""
    def __init__(self, queue: asyncio.Queue[AqueueTask]):
        self._queue = queue
        self.event = asyncio.Event()  # 仅作为状态位
        self.event.set()
        self._enable = True
        self._work_loop = asyncio.create_task(self.run())

    async def run(self):
        while self._enable:
            task = await self._queue.get()
            self.event.clear()
            try:
                if task.parsed_timeout > 0:
                    result = await asyncio.wait_for(task, timeout=task.parsed_timeout)

                else:
                    result = await task

            except (asyncio.CancelledError, asyncio.TimeoutError):
                task.exception.append(TimeoutError())
                continue

            except Exception as E:
                task.exception.append(E)
                _log.error(f"处理任务 [{task.name}] 异常: {E}")
                _log.debug(f"处理任务 [{task.name}] 异常:\n {traceback.format_exc()}")

            else:
                task.result.append(result)

            finally:
                self._queue.task_done()
                self.event.set()
                task.event.set()

    @property
    def suspend(self) -> bool:
        return self.event.is_set()

    @property
    def is_alive(self) -> bool:
        return not self._work_loop.done()

    async def wait(self):
        await self.event.wait()

    async def kill(self):
        self._enable = False
        if not self.suspend:
            await self._work_loop

        else:
            self.kill_nowait()

    def kill_nowait(self):
        self._enable = False
        self._work_loop.cancel()


class Aqueue:
    """
    异步任务队列
    创建时必须在稳定的有event_loop的线程下
    否则会出不可预料且无报错的bug
    """
    def __init__(
            self,
            min_worker: int = None,
            max_worker: int = None,
            default_timeout = None,
            *,
            observe_config: bool = True
    ):
        if max_worker is None:
            max_worker = SETTING_CFG.Aqueue.MaxWorkers  # 默认16

        if min_worker is None:
            min_worker = SETTING_CFG.Aqueue.MinWorkers  # 默认4

        if min_worker > max_worker:
            raise ValueError("<min_worker> can't greater than <max_worker>")

        if default_timeout is None:
            default_timeout = SETTING_CFG.Aqueue.DefaultTaskTimeout  # 默认60

        self._task_queue: asyncio.Queue[AqueueTask] = asyncio.Queue()
        self._min_worker = min_worker
        self._max_worker = max_worker
        self._timeout = default_timeout
        self._workers: list[AqueueWorker] = []
        if observe_config:
            CONFIG_OBSERVER.register(self)

    @property
    def running_worker(self) -> int:
        return len([w for w in self._workers if not w.suspend])

    @property
    def suspend_worker(self) -> int:
        return len([w for w in self._workers if w.suspend])

    def update_config(self):
        self._min_worker = SETTING_CFG.Aqueue.MinWorkers
        self._max_worker = SETTING_CFG.Aqueue.MaxWorkers
        self._timeout = SETTING_CFG.Aqueue.DefaultTaskTimeout

    def dynamic_worker(self):
        """动态调整Worker数量"""
        for worker in self._workers.copy():
            if not worker.is_alive:
                if worker in self._workers:
                    self._workers.remove(worker)

        pending_tasks = self._task_queue.qsize()
        current_workers = len(self._workers)
        suspend_workers = self.suspend_worker

        # 补全
        if current_workers < self._min_worker:
            for _ in range(self._min_worker - current_workers):
                self._workers.append(AqueueWorker(self._task_queue))

            return

        # 增设
        if pending_tasks > 1 * current_workers and current_workers < self._max_worker:
            add_count = min(self._max_worker - current_workers, max(1, (pending_tasks - suspend_workers) // 2))
            if add_count <= 0:
                return

            for _ in range(add_count):
                self._workers.append(AqueueWorker(self._task_queue))

        # 缩减
        if suspend_workers > 1 * pending_tasks and current_workers > self._min_worker:
            subtract_count = min(current_workers - self._min_worker, max(1, (suspend_workers - pending_tasks) // 2))
            if subtract_count <= 0:
                return

            suspends = [w for w in self._workers if w.suspend]
            for _ in range(subtract_count):
                if not suspends:
                    return

                suspends[-1].kill_nowait()
                suspends.pop()

    def add_task(self, coro: Coroutine, timeout: int | float | Callable[..., int | float] = None) -> AqueueTask:
        """添加任务，注意添加的是Coroutine不是Awaitable"""
        if timeout is None:
            timeout = self._timeout

        self.dynamic_worker()
        new_task = AqueueTask(coro, timeout)
        # _log.debug(f"已创建 Aqueue任务: [{new_task.name}]")
        asyncio.create_task(self._task_queue.put(new_task))
        return new_task

    def aqueue_wraps(self, func: Callable[..., Coroutine]):
        """也可通过装饰器的方式入库任务，使用后原函数会返回一个异步队列任务，而不是原结果"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> AqueueTask:
            return self.add_task(func(*args, **kwargs))

        return wrapper

    def info(self) -> str:
        pending_tasks = self._task_queue.qsize()
        return f"队列状态: 待处理任务 {pending_tasks} | 活跃工作器数量 {self.running_worker}"

    def kill_nowait(self):
        for w in self._workers:
            w.kill_nowait()

    async def kill(self, timeout: float | None | object = _DEFAULT_STOP_TIMEOUT):
        async def _wait():
            await self._task_queue.join()
            running = [w.wait() for w in self._workers]
            await asyncio.gather(*running)

        try:
            if timeout is _DEFAULT_STOP_TIMEOUT:
                timeout = SETTING_CFG.Aqueue.StopTimeout  # 默认15

            if timeout is None:
                await _wait()

            else:
                await asyncio.wait_for(_wait(), timeout=timeout)

        except (asyncio.CancelledError, asyncio.TimeoutError, RuntimeError):
            pass

        self.kill_nowait()
