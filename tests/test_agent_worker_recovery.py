from __future__ import annotations

import asyncio
import os
from unittest.mock import Mock, patch

with patch.dict(os.environ, {"LOG_FORMAT": "%(message)s"}):
    import TIYA.aqueue as aqueue_module
    import TIYA.agent.agent as agent_module
    import TIYA.agent.agent_runtime as agent_runtime


def test_aqueue_replaces_finished_workers(monkeypatch) -> None:
    class FakeWorker:
        def __init__(self, _queue, *, alive: bool = True):
            self.is_alive = alive
            self.suspend = True

    queue = object.__new__(aqueue_module.Aqueue)
    queue._task_queue = Mock()
    queue._task_queue.qsize.return_value = 0
    queue._min_worker = 1
    queue._max_worker = 1
    dead_worker = FakeWorker(queue._task_queue, alive=False)
    queue._workers = [dead_worker]
    monkeypatch.setattr(aqueue_module, "AqueueWorker", FakeWorker)

    queue.dynamic_worker()

    assert dead_worker not in queue._workers
    assert len(queue._workers) == 1
    assert queue._workers[0].is_alive


def test_adjust_worker_replaces_finished_workers(monkeypatch) -> None:
    class FakeWorker:
        def __init__(self, _host, *, alive: bool = True):
            self.is_alive = alive

    manager = object.__new__(agent_runtime.AgentWorkerManager)
    manager.max_worker = 2
    dead_worker = FakeWorker(manager, alive=False)
    alive_worker = FakeWorker(manager)
    manager.workers = [dead_worker, alive_worker]
    monkeypatch.setattr(agent_runtime, "Worker", FakeWorker)

    asyncio.run(manager.adjust_worker())

    assert dead_worker not in manager.workers
    assert alive_worker in manager.workers
    assert len(manager.workers) == 2
    assert all(worker.is_alive for worker in manager.workers)


def test_agent_task_submission_checks_worker_health() -> None:
    calls: list[str] = []

    class FakeTaskQueue:
        async def add(self, _task) -> None:
            calls.append("add")

    class FakeTaskRunner:
        async def adjust_worker(self) -> None:
            calls.append("adjust")

    async def tool() -> None:
        return None

    agent = object.__new__(agent_module.BaseAgent)
    agent._functions = {"tool": tool}
    agent._task_queue = FakeTaskQueue()
    agent.task_runner = FakeTaskRunner()
    agent._history = Mock()

    task = asyncio.run(agent.add_task("tool", retry=1))

    assert task is not None
    assert calls == ["adjust", "add"]
    agent._history.task_create.assert_called_once()
