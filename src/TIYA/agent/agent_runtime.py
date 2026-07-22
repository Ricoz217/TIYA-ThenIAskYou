from __future__ import annotations
__version__ = "0.1.0"

import traceback
import contextvars
import asyncio
import inspect
from functools import partial
from typing import TYPE_CHECKING, Callable
from TIYA.config import SETTING_CFG
from TIYA.executor import AGENT_EXECUTOR
from TIYA.logger import get_logger

if TYPE_CHECKING:
    from TIYA.agent.agent_structure import AgentTaskQueue, AgentTask

_log = get_logger()

class RetryLimitError(Exception):
    """任务重试超出限制"""
    pass


class Worker:
    def __init__(self, host: AgentWorkerManager):
        self._host = host
        self._timeout: float = SETTING_CFG.Agent.AgentWorkerTimeout  # 默认15分钟
        self.current_task: str = ""
        self.processing = False
        self.live = True
        self.running_loop = asyncio.create_task(self.worker_loop())

    def _callback(self, task: AgentTask):
        if task.callback.value == "NEVER":
            asyncio.create_task(self._host.callback(run=False, call_from=task.id))
            return

        elif task.callback.value == "EXCEPTION":
            if task.status.value not in {"EXCEPTION", "TIMEOUT", "FAILED"}:
                asyncio.create_task(self._host.callback(run=False, call_from=task.id))
                return

        elif task.callback.value == "CALL":
            asyncio.create_task(self._host.callback(run=True, call_only=True, call_from=task.id))
            return

        asyncio.create_task(self._host.callback(run=True, call_from=task.id))

    async def worker_loop(self):
        try:
            while self.live:
                await self._host.pause_event.wait()
                task = await self._task_get()
                task.set_running()
                timeout = task.timeout if task.timeout > 0 else self._timeout
                token = self._host.manager.task_context.set(self.current_task)  # 设置上下文
                function = next(iter(task.function.values()))
                task.attempt += 1
                try:
                    if 0 < task.retry_limit < task.attempt:
                        raise RetryLimitError

                    if inspect.iscoroutinefunction(function):
                        result = await asyncio.wait_for(function(**task.input_params), timeout=timeout)

                    else:
                        ctx = contextvars.copy_context()
                        warp_function = partial(function, **task.input_params)
                        loop = asyncio.get_running_loop()
                        # noinspection PyTypeChecker
                        result = await asyncio.wait_for(loop.run_in_executor(AGENT_EXECUTOR, ctx.run, warp_function),
                                                        timeout=timeout)

                except asyncio.TimeoutError:
                    if task.retry_limit > 0 and task.attempt < task.retry_limit:
                        task.set_ready()
                        await self._host.manager.add(task)

                    else:
                        task.set_timeout(True)
                        self._callback(task)

                    continue

                except asyncio.CancelledError:
                    task.set_cancel()
                    self._callback(task)
                    return

                except Exception as E:
                    _log.error(E)
                    _log.debug(traceback.format_exc())
                    if task.retry_limit > 0 and task.attempt < task.retry_limit:
                        task.set_exception(E)
                        task.set_ready()
                        await self._host.manager.add(task)

                    else:
                        task.set_exception(E, True)
                        self._callback(task)

                    continue

                else:
                    task.results["result"] = result
                    task.set_done()
                    self._callback(task)

                finally:
                    self._task_done(task)
                    self._host.manager.task_context.reset(token)
                    await self._host.manager.notify_once()

        except asyncio.CancelledError:
            return

    async def _task_get(self) -> AgentTask:
        task = await self._host.manager.get()
        self.current_task = task.id
        self._host.task_mapping[self.current_task] = self
        self.processing = True
        return task

    def _task_done(self, task: AgentTask):
        self._host.task_mapping.pop(self.current_task, None)
        self.current_task = ""
        if task.status.value != "READY":
            self._host.manager.add_accomplish(task)

        self.processing = False

    def restart(self):
        self.running_loop.cancel()
        self.processing = False
        self._host.task_mapping.pop(self.current_task, None)
        self.current_task = ""
        self.running_loop = asyncio.create_task(self.worker_loop())

    async def kill(self, join = True):
        self.live = False
        if join and self.processing:
            await self.running_loop

        self.running_loop.cancel()
        self.processing = False
        self._host.task_mapping.pop(self.current_task, None)
        self.current_task = ""
        if self in self._host.workers:
            self._host.workers.remove(self)


class AgentWorkerManager:
    def __init__(self, manager: AgentTaskQueue, callback: Callable, max_worker: int = None):
        if max_worker is None:
            max_worker = SETTING_CFG.Agent.MaxWorker  # 默认4

        self.manager = manager
        self.callback = callback
        self.max_worker = max_worker
        self.task_mapping: dict[str, Worker] = {}
        self.pause_event = asyncio.Event()
        self.workers: list[Worker] = []
        asyncio.create_task(self.initiate())

    async def initiate(self):
        await self.adjust_worker()
        self.resume()

    def pause(self, execute = False):
        self.pause_event.clear()
        if execute:
            for worker in self.workers:
                worker.restart()

    def resume(self):
        self.pause_event.set()

    async def shutdown(self, wait=False):
        for worker in self.workers[:]:
            await worker.kill(wait)

    async def adjust_worker(self, max_worker: int = None):
        if max_worker:
            self.max_worker = max_worker

        if len(self.workers) > self.max_worker:
            del_workers = self.workers[self.max_worker:]
            for worker in del_workers:
                await worker.kill()

        elif len(self.workers) < self.max_worker:
            adds = self.max_worker - len(self.workers)
            for _ in range(adds):
                self.workers.append(Worker(self))


    def execute_task(self, task_id: str):
        if task_id in self.task_mapping:
            worker = self.task_mapping[task_id]
            worker.restart()

