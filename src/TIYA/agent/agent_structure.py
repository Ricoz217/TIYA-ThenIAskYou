from __future__ import annotations
"""
agent_structure.py
agent的数据结构
"""
__version__ = "0.1.0"

import traceback
import asyncio
import time
import heapq
import contextvars
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable, Awaitable, Any, TYPE_CHECKING
from enum import Enum
from TIYA.LLM_connect import Context
from TIYA.time_id import next_time_id
from TIYA.config import SETTING_CFG

if TYPE_CHECKING:
    from apscheduler.job import Job


class TaskInterruptedError(Exception):
    """任务运行中断错误"""
    pass


class TaskCycleError(Exception):
    """任务依赖链循环错误"""
    pass


class TaskQueueError(Exception):
    """任务队列内部错误"""
    pass


class SubTaskError(Exception):
    """子任务引发的异常"""
    ...


class TaskStatus(Enum):
    EXPIRED = "EXPIRED"  # 以过期，惰性删除
    RUNNING = "RUNNING"  # 运行中
    DONE = "DONE"  # 已完成
    EXCEPTION = "EXCEPTION"  # 出现异常
    PENDING = "PENDING"  # 排队中
    CANCELED = "CANCELED"  # 已取消
    TIMEOUT = "TIMEOUT"  # 已超时
    WAITING = "WAITING"  # 等待依赖
    READY = "READY"  # 依赖已完成，可以开始
    FAILED = "FAILED"  # 任务失败


class CallbackMode(Enum):
    ALWAYS = "ALWAYS"
    EXCEPTION = "EXCEPTION"
    NEVER = "NEVER"
    CALL = "CALL"


def translate_task_status(status: TaskStatus | str) -> str:
    if isinstance(status, TaskStatus):
        status = status.value

    mapping = {
        "EXPIRED": "已失效",
        "RUNNING": "执行中",
        "DONE": "已完成",
        "EXCEPTION": "执行异常",
        "PENDING": "计划中",
        "CANCELED": "已取消",
        "TIMEOUT": "执行超时",
        "WAITING": "等待依赖完成",
        "READY": "已就绪",
        "FAILED": "执行失败",
    }
    if status not in mapping:
        raise KeyError("出现未定义任务状态")

    return mapping[status]


_ts_ready = {
    TaskStatus.PENDING,
    TaskStatus.READY,
    TaskStatus.WAITING
}
_ts_done = {
    TaskStatus.FAILED,
    TaskStatus.CANCELED,
    TaskStatus.TIMEOUT,
    TaskStatus.EXPIRED,
    TaskStatus.EXCEPTION,
    TaskStatus.DONE
}

@dataclass(slots=True, kw_only=True)
class Trigger:
    """ISO 8601"""
    trigger_once: str = ""    # 一次性触发时间
    year: str = ""            # 年份：4位数字
    month: str = ""           # 月份：1-12
    day: str = ""             # 日期：1-31
    week: str = ""            # 周数：1-53
    day_of_week: str = ""     # 星期几：0-6 或 mon,tue,wed,thu,fri,sat,sun
    hour: str = ""            # 小时：0-23
    minute: str = ""          # 分钟：0-59
    second: str = ""          # 秒：0-59
    start_date: str = ""      # 开始日期
    end_date: str = ""        # 结束日期
    jitter: int = 5        # 随机抖动（秒）

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return Trigger(
            trigger_once=self.trigger_once,
            year=self.year,
            month=self.month,
            day=self.day,
            week=self.week,
            day_of_week=self.day_of_week,
            hour=self.hour,
            minute=self.minute,
            second=self.second,
            start_date=self.start_date,
            end_date=self.end_date,
            jitter=self.jitter
        )

    def __eq__(self, other):
        if not isinstance(other, Trigger):
            raise TypeError

        if self.trigger_once and other.trigger_once and self.trigger_once == other.trigger_once:
            return True

        other_data = other.to_dict()
        self_data = self.to_dict()
        if len(other_data) != len(self_data):
            return False

        for key, value in self_data.items():
            other_value = other_data.get(key, None)
            if other_value is None:
                return False

            if value != other_value:
                return False

        return True

    def to_dict(self):
        data = asdict(self)
        return {k: v for k, v in data.items() if v}

    @classmethod
    def from_dict(cls, data: dict):
        return Trigger(**data)


@dataclass(slots=True)
class AgentTask:
    function: dict[str, Callable[..., Awaitable | Any]]
    input_params: dict = field(default_factory=dict)
    name: str = ""
    description: str = ""
    call_id: str = ""
    priority: int = 50
    data: dict = field(default_factory=dict)
    dependents: set[str] = field(default_factory=set)
    blocks: set = field(default_factory=set)
    create_time: str = field(default_factory=lambda : str(time.time()))
    status: TaskStatus = TaskStatus.PENDING
    results: dict = field(default_factory=dict)
    exceptions: list[Exception] = field(default_factory=list)
    attempt: int = 0
    retry_limit: int = 0  # 默认3次
    timeout: float = 0
    event: asyncio.Event = field(default_factory=asyncio.Event)
    callback: CallbackMode = CallbackMode.NEVER
    is_wait: bool = False
    id: str = field(default_factory=lambda : str(next_time_id()))

    def set_running(self):
        self.status = TaskStatus.RUNNING
        self.event.clear()

    def set_exception(self, E: Exception, final=False):
        self.exceptions.append(E)
        self.status = TaskStatus.EXCEPTION
        if final:
            self.event.set()

    def set_timeout(self, final=False):
        self.status = TaskStatus.TIMEOUT
        if final:
            self.event.set()

    def set_cancel(self):
        self.status = TaskStatus.CANCELED
        self.event.set()

    def set_done(self):
        self.status = TaskStatus.DONE
        self.event.set()

    def set_ready(self):
        self.status = TaskStatus.READY

    def copy(self):
        return self.__copy__()

    async def wait(self):
        await self.event.wait()

    def __copy__(self):
        new_task = AgentTask(
            self.function,
            self.input_params.copy(),
            self.name,
            self.description,
            self.call_id,
            self.priority,
            self.data.copy(),
            self.dependents.copy(),
            self.blocks.copy(),
            self.create_time,
            self.status,
            self.results.copy(),
            self.exceptions.copy(),
            self.attempt,
            self.retry_limit,
            self.timeout,
            asyncio.Event(),
            self.callback,
            self.is_wait
        )
        if self.event.is_set():
            new_task.event.set()

        return new_task

    def to_dict(self) -> dict:
        dump_content = {
            "function": next(iter(self.function)),
            "input_params": self.input_params.copy(),
            "name": self.name,
            "description": self.description,
            "call_id": self.call_id,
            "priority": self.priority,
            "data": self.data.copy(),
            "dependents": list(self.dependents),
            "blocks": list(self.blocks),
            "create_time": self.create_time,
            "status": self.status.value,
            "results": self.results.copy(),
            "exceptions": [traceback.format_exception(e) for e in self.exceptions],
            "attempt": self.attempt,
            "retry": self.retry_limit,
            "timeout": self.timeout,
            "event": self.event.is_set(),
            "callback": self.callback.value,
            "wait": self.is_wait,
            "id": self.id
        }
        return dump_content

    def to_llm(self) -> dict:
        return {
            "task_id": self.id,
            "task_name": self.name if self.name else "untitled",
            "tool_name": next(iter(self.function)),
            "status": translate_task_status(self.status),
            "result": self.results.get("result", None),
            "exceptions": [f"{e.__class__}: {e}" for e in self.exceptions]
        }

    @classmethod
    def from_dict(cls, data: dict):
        function = data.get("function")
        if function is None:
            return None

        new_task = AgentTask(
            function=function,
            input_params=data.get("input_params", {}),
            name=data.get("name", ""),
            description=data.get("description", ""),
            call_id=data.get("call_id", ""),
            priority=data.get("priority", 50),
            data=data.get("data", {}),
            dependents=set(data.get("dependents", [])),
            blocks=set(data.get("blocks", [])),
            create_time=data.get("create_time", str(time.time())),
            status=TaskStatus(data.get("status", "PENDING")),
            results=data.get("results", {}),
            attempt=data.get("attempt", 0),
            retry_limit=data.get("retry", 0),
            timeout=data.get("timeout", 0),
            callback=CallbackMode(data.get("callback", "NEVER")),
            is_wait=data.get("wait", False),
            id=data.get("id", str(next_time_id()))
        )
        if data.get("event", False):
            new_task.event.set()

        return new_task

    def __gt__(self, other):
        if not isinstance(other, AgentTask):
            raise TypeError("Not Agent Task")

        return self.priority > other.priority

    def __lt__(self, other):
        if not isinstance(other, AgentTask):
            raise TypeError("Not Agent Task")

        return self.priority < other.priority


@dataclass(slots=True)
class AgentScheTask(AgentTask):
    priority: int = 0
    last_fire_at: float = 0
    next_run_at: str = ""
    trigger_spec: Trigger = field(default_factory=Trigger)
    paused: bool = False

    def is_once(self):
        return bool(self.trigger_spec.trigger_once)

    def finish_fire(self):
        self.last_fire_at = time.time()
        self.event.set()

    def to_dict(self) -> dict:
        base_content = super(AgentScheTask, self).to_dict()
        base_content["last_fire_at"] = self.last_fire_at
        base_content["next_run_at"] = self.next_run_at
        base_content["trigger_spec"] = self.trigger_spec.to_dict()
        base_content["paused"] = self.paused
        return base_content

    def to_llm(self) -> dict:
        base_content = super(AgentScheTask, self).to_llm()
        if self.last_fire_at > 0:
            base_content["last_fire_at"] = f"{datetime.fromtimestamp(self.last_fire_at):%y-%m-%d %H:%M:%S}"

        else:
            base_content["last_fire_at"] = None

        base_content["next_run_at"] = self.next_run_at if self.next_run_at else None
        return base_content

    @classmethod
    def from_dict(cls, data: dict):
        function = data.get("function")
        if function is None:
            return None

        new_task = AgentScheTask(
            function=function,
            input_params=data.get("input_params", {}),
            name=data.get("name", ""),
            description=data.get("description", ""),
            call_id=data.get("call_id", ""),
            priority=data.get("priority", 0),
            data=data.get("data", {}),
            dependents=set(data.get("dependents", [])),
            blocks=set(data.get("blocks", [])),
            create_time=data.get("create_time", str(time.time())),
            status=TaskStatus(data.get("status", "PENDING")),
            results=data.get("results", {}),
            attempt=data.get("attempt", 0),
            retry_limit=data.get("retry", 0),
            timeout=data.get("timeout", 0),
            callback=CallbackMode(data.get("callback", "NEVER")),
            is_wait=data.get("wait", False),
            id=data.get("id", str(next_time_id())),
            last_fire_at=data.get("last_fire_at", 0),
            next_run_at=data.get("next_run_at", ""),
            trigger_spec=Trigger.from_dict(data.get("trigger_spec", {})),
            paused=data.get("paused", False)
        )
        if data.get("event", False):
            new_task.event.set()

        return new_task

    def set_ready(self):
        self.attempt = 0
        self.exceptions.clear()
        self.status = TaskStatus.READY

    def __copy__(self):
        new_task = AgentScheTask(
            self.function,
            self.input_params.copy(),
            self.name,
            self.description,
            self.call_id,
            self.priority,
            self.data.copy(),
            self.dependents.copy(),
            self.blocks.copy(),
            self.create_time,
            self.status,
            self.results.copy(),
            self.exceptions.copy(),
            self.attempt,
            self.retry_limit,
            self.timeout,
            callback=self.callback,
            last_fire_at=self.last_fire_at,
            next_run_at=self.next_run_at,
            trigger_spec = self.trigger_spec.copy(),
            paused = self.paused
        )
        if self.event.is_set():
            new_task.event.set()

        return new_task


class AgentTaskQueue:
    def __init__(self, max_accomplished_task: int = None):
        if max_accomplished_task is None:
            max_accomplished_task = SETTING_CFG.Agent.MaxAgentDoneTask  # 默认100

        self._heap: list[AgentTask] = []
        self.tasks: dict[str, AgentTask] = {}
        self.accomplished_task: dict[str, AgentTask] = {}
        self.show_tasks: set[str] = set()
        self.max_task_history: int = max_accomplished_task
        self.task_context = contextvars.ContextVar("task_id", default="")
        self._condition = asyncio.Condition()

    def _rearrange_priority(self, task: AgentTask):
        if not task.dependents:
            return

        max_priority = 0
        def _recursion(_task: AgentTask, check_ring: list = None):
            if check_ring is None:
                check_ring = []

            if _task.id in check_ring:
                raise TaskCycleError("任务依赖链有循环")

            if not _task.dependents:
                return

            if _task.status not in {TaskStatus.WAITING, TaskStatus.READY, TaskStatus.PENDING, TaskStatus.RUNNING}:
                return

            nonlocal max_priority
            max_priority = max(max_priority, _task.priority)
            check_ring.append(_task.id)
            for _task_id in _task.dependents:
                if _task_id not in self.tasks:
                    if _task_id not in self.accomplished_task:
                        raise KeyError("依赖任务必须先添加")

                    continue

                _sub_task = self.tasks[_task_id]
                _recursion(_sub_task, check_ring.copy())

        _recursion(task)
        if max_priority:
            task.status = TaskStatus.WAITING
            task.priority = max_priority + 1

    def add_accomplish(self, task: AgentTask):
        self.accomplished_task[task.id] = task
        self.tasks.pop(task.id, None)
        self.show_tasks.add(task.id)

        while len(self.accomplished_task) > self.max_task_history:
            key = next(iter(self.accomplished_task))
            del self.accomplished_task[key]
            if key in self.show_tasks:
                self.show_tasks.remove(key)

    async def add(self, task: AgentTask):
        async with self._condition:
            self.tasks[task.id] = task
            self._rearrange_priority(task)
            heapq.heappush(self._heap, task)
            self._condition.notify()

    async def pinned(self, task_id: AgentTask | str, call_mapping: dict = None) -> str:
        """置顶任务，返回新的id"""
        if isinstance(task_id, AgentTask):
            task_id = task_id.id

        if task_id not in self.tasks:
            return ""

        task = self.tasks[task_id]
        if task.status not in _ts_ready:
            return ""

        def _modify_blocks(_old_task: AgentTask, _new_task: AgentTask):
            if not _old_task.blocks:
                return

            _task_lookup: dict[str, AgentTask] = {**self.tasks, **self.accomplished_task}
            for _tid in _old_task.blocks:
                if _tid in _task_lookup:
                    _block_task = _task_lookup[_tid]
                    if _old_task.id in _block_task.dependents:
                        _block_task.dependents.remove(_old_task.id)
                        _block_task.dependents.add(_new_task.id)

        if call_mapping is not None:
            t_mapping = {k: v for v, k in call_mapping.items()}

        else:
            t_mapping = {}

        if not task.dependents:
            new_task = task.copy()
            task.status = TaskStatus.EXPIRED
            new_task.priority = 0
            _modify_blocks(task, new_task)
            if task.id in t_mapping:
                call_mapping[t_mapping[task.id]] = new_task.id

            async with self._condition:
                self.tasks[new_task.id] = new_task
                await self.add(new_task)
                self._condition.notify()

            return new_task.id

        reorder_heap = []
        def _recursion(_task: AgentTask):
            heapq.heappush(reorder_heap, _task)
            if not _task.dependents:
                return

            for _sub_task_id in _task.dependents:
                if _sub_task_id in self.tasks:
                    _sub_task = self.tasks[_sub_task_id]
                    if _sub_task.status not in _ts_ready:
                        return

                    _recursion(_sub_task)

        _recursion(task)
        new_task_id = ""
        async with self._condition:
            for i, tsk in enumerate(reorder_heap, start=1):
                new_task = tsk.copy()
                _modify_blocks(tsk, new_task)
                new_task.priority = i
                if tsk.id in t_mapping:
                    call_mapping[t_mapping[tsk.id]] = new_task.id

                tsk.status = TaskStatus.EXPIRED
                self.tasks[new_task.id] = new_task
                heapq.heappush(self._heap, new_task)
                if tsk.id == task_id:
                    new_task_id = new_task.id

            self._condition.notify()

        return new_task_id

    async def get(self) -> AgentTask:
        is_callable_task = False
        async with self._condition:
            while not self._heap or not is_callable_task:
                if not self._heap:
                    await self._condition.wait()
                    continue

                task = self._heap[0]
                if task.status in _ts_done:
                    self.add_accomplish(task)
                    heapq.heappop(self._heap)
                    continue

                elif task.status is TaskStatus.WAITING:
                    ready = True
                    lookup_task = {**self.tasks, **self.accomplished_task}
                    for tid in task.dependents:
                        if tid in lookup_task:
                            block_task = lookup_task[tid]
                            if block_task.status is TaskStatus.DONE:
                                self.add_accomplish(block_task)

                            elif block_task.status in {TaskStatus.PENDING, TaskStatus.READY,
                                                       TaskStatus.WAITING, TaskStatus.RUNNING}:
                                ready = False
                                break

                            elif block_task.status in {TaskStatus.FAILED, TaskStatus.EXCEPTION,
                                                       TaskStatus.CANCELED, TaskStatus.TIMEOUT}:
                                task.status = TaskStatus.FAILED
                                ready = False

                            elif block_task.status is TaskStatus.EXPIRED:
                                raise TaskQueueError("依赖链有失效任务")

                    if ready:
                        task.status = TaskStatus.READY
                        is_callable_task = True

                    else:
                        task = heapq.heappop(self._heap)
                        await self._condition.wait()
                        heapq.heappush(self._heap, task)
                        continue

                elif task.status is TaskStatus.RUNNING:
                    heapq.heappop(self._heap)
                    continue

                else:
                    is_callable_task = True

            return heapq.heappop(self._heap)

    async def notify_once(self):
        async with self._condition:
            self._condition.notify()

    def to_dict(self) -> dict:
        tasks = self.tasks.copy()
        accomplished_task = self.accomplished_task.copy()
        return {
            "pending": {k: v.to_dict() for k, v in tasks.items()},
            "accomplished": {k: v.to_dict() for k, v in accomplished_task.items()}
        }

    async def from_dict(self, data: dict, function_mapping: dict):
        self.tasks.clear()
        self.accomplished_task.clear()
        self._heap.clear()
        tasks = data.get("pending", {})
        accomplished = data.get("accomplished", {})
        for tid, task_data in tasks.items():
            function_name = task_data["function"]
            function = function_mapping.get(function_name, None)
            if function is None:
                continue

            task_data["function"] = {function_name: function}
            task = AgentTask.from_dict(task_data)
            if task is None:
                continue

            if task.status is TaskStatus.RUNNING:
                task.status = TaskStatus.READY

            await self.add(task)

        for tid, task_data in accomplished.items():
            function_name = task_data["function"]
            function = function_mapping.get(function_name, None)
            if function is None:
                continue

            task_data["function"] = {function_name: function}
            task = AgentTask.from_dict(task_data)
            if task is None:
                continue

            self.accomplished_task[task.id] = task

    def from_checkpoint(self, data: dict):
        pending_task = {}
        accomplished_task = {}
        task_lookup = {**self.tasks, **self.accomplished_task}
        for tid, tst in data["tasks"].items():
            if tid not in task_lookup:
                continue

            task = task_lookup[tid]
            task.status = TaskStatus(tst)
            pending_task[task.id] = task

        for tid, tst in data["accomplished"].items():
            if tid not in task_lookup:
                continue

            task = task_lookup[tid]
            task.status = TaskStatus(tst)
            accomplished_task[task.id] = task

        self.tasks.clear()
        self.tasks.update(pending_task)
        self.accomplished_task.clear()
        self.accomplished_task.update(accomplished_task)
        self._heap.clear()
        self._heap.extend(list(pending_task.values()))
        heapq.heapify(self._heap)


class AgentScheTaskManage:
    def __init__(self):
        self.tasks: dict[str, AgentScheTask] = {}
        self.pending_task: dict[str, AgentScheTask] = {}
        self.timeout: float = SETTING_CFG.Agent.AgentWorkerTimeout  # 默认15分钟

    def check_scheduled(self, jobs: list[Job]) -> list[str]:
        expired_jobs = []
        self.pending_task.clear()
        for job in jobs:
            if job.id in self.tasks:
                if job.next_run_time is not None:
                    self.pending_task[job.id] = self.tasks[job.id]

            else:
                expired_jobs.append(job.id)

        return expired_jobs
                # print(jobs)

    def clear(self):
        self.tasks.clear()
        self.pending_task.clear()

    def remove(self, task: str | AgentScheTask):
        if isinstance(task, str):
            self.tasks.pop(task, None)
            self.pending_task.pop(task, None)

        else:
            self.tasks.pop(task.id, None)
            self.pending_task.pop(task.id, None)

    def is_exist(self, task: str | AgentScheTask) -> bool:
        if isinstance(task, str):
            return task in self.tasks

        if task.id in self.tasks:
            return True

        for tsk in self.tasks.values():
            if next(iter(task.function)) == next(iter(tsk.function)):
                if task.trigger_spec == tsk.trigger_spec:
                    return True

        return False

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.pending_task.copy().items()}

    def from_dict(self, data: dict, function_mapping: dict):
        self.tasks.clear()
        self.pending_task.clear()
        for tid, task_data in data.items():
            function_name = task_data["function"]
            function = function_mapping.get(function_name, None)
            if function is None:
                continue

            task_data["function"] = {function_name: function}
            task = AgentScheTask.from_dict(task_data)
            if task is None:
                continue

            self.tasks[task.id] = task


class AgentMemory:
    """负责用户个性化记忆和其他偏好记忆"""
    ...


class AgentContext:
    def __init__(self, max_history: int = None):
        if max_history is None:
            max_history = SETTING_CFG.Agent.MaxAgentContext  # 默认10

        self.max_history = max_history
        self.current_window = Context()
        self.window_history: dict[str, Context] = {}

    def new(self) -> Context:
        new_context = Context()
        new_context.append(self.current_window.system)
        new_context.append(self.current_window.tools)
        self.window_history[str(time.time())] = self.current_window
        if len(self.window_history) > self.max_history:
            self.window_history.pop(next(iter(self.window_history)))

        self.current_window = new_context
        return new_context

    def to_dict(self) -> dict:
        dump_content = {
            "current": self.current_window.to_dict(),
            "history": {k: v.to_dict() for k, v in self.window_history.items()}
        }
        return dump_content

    def from_dict(self, data: dict, function_mapping: dict) -> Context:
        self.current_window = Context.from_dict(data["current"], function_mapping)
        self.window_history.update({k: Context.from_dict(v, function_mapping) for k, v in data["history"].items()})
        return self.current_window

    def from_checkpoint(self, data: dict, function_mapping: dict):
        self.current_window = Context.from_dict(data["current"], function_mapping)
        self.window_history = {k: v for k, v in self.window_history.items() if k in data["history"]}


class HistoryType(Enum):
    TASK_CREATE = "TASK_CREATE"
    TASK_DONE = "TASK_DONE"
    SCHE_CREATE = "SCHE_CREATE"
    SCHE_FIRE = "SCHE_FIRE"
    USER_ASK = "USER_ASK"
    LLM_ASK = "LLM_ASK"
    LLM_RESP = "LLM_RESP"
    COMP_CONTEXT = "COMP_CONTEXT"
    RELOAD_CONTEXT = "RELOAD_CONTEXT"


@dataclass(slots=True)
class SummaryHint:
    title: str
    information: dict | list
    event_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        dt = datetime.fromtimestamp(self.event_time)
        return {
            "title": self.title,
            "information": self.information,
            "event_time": f"{dt:%y-%m-%d %H:%M:%S}"
        }


@dataclass(slots=True)
class AgentHistory:
    event_type: HistoryType
    event_data: Any
    event_id: str = ""
    event_time: str = field(default_factory=lambda :str(time.time()))
    id: str = field(default_factory=lambda : str(next_time_id()))

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "event_data": self.event_data,
            "event_id": self.event_id,
            "event_time": self.event_time,
            "id": self.id
        }

    @classmethod
    def from_dict(cls, data: dict):
        new_event = AgentHistory(
            HistoryType(data["event_type"]),
            data.get("event_data", None),
            data.get("event_id", ""),
            data.get("event_time", str(time.time())),
            data.get("id", str(next_time_id()))
        )
        return new_event


class AgentHistoryManager:
    """负责整个Agent历史，事件制。支持回滚、快照"""
    def __init__(self, save_path: Path, max_history: int = None):
        if max_history is None:
            max_history = SETTING_CFG.Agent.MaxAgentHistory  # 默认1000

        self._max_history = max_history
        self._save_path = save_path
        self.checkpoint_path = save_path / "checkpoint"
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.history_file = save_path / "agent_history.json"
        self.task_file = save_path / "agent_tasks.json"
        self.schedule_task_file = save_path / "schedule_task.json"
        self.current_context_file = save_path / "context_current.json"
        self.context_history_file = save_path / "context_history.json"
        # 旧版上下文文件，仅用于兼容加载。
        self.context_file = save_path / "context.json"
        self.history: deque[AgentHistory] = deque(maxlen=self._max_history)
        self.checkpoint: deque[tuple[str, Path]] = deque(maxlen=5)

    def task_create(self, data: Any, eid: str = ""):
        new_event = AgentHistory(HistoryType.TASK_CREATE, data, eid)
        self.history.append(new_event)

    def task_done(self, data: Any, eid: str = ""):
        new_event = AgentHistory(HistoryType.TASK_DONE, data, eid)
        self.history.append(new_event)

    def schedule_task_create(self, data: Any, eid: str = ""):
        new_event = AgentHistory(HistoryType.SCHE_CREATE, data, eid)
        self.history.append(new_event)

    def schedule_task_fire(self, data: Any, eid: str = ""):
        new_event = AgentHistory(HistoryType.SCHE_FIRE, data, eid)
        self.history.append(new_event)

    def user_ask(self, data: Any, eid: str = ""):
        new_event = AgentHistory(HistoryType.USER_ASK, data, eid)
        self.history.append(new_event)

    def llm_ask(self, data: Any, eid: str = ""):
        new_event = AgentHistory(HistoryType.LLM_ASK, data, eid)
        self.history.append(new_event)

    def llm_response(self, data: Any, eid: str = ""):
        new_event = AgentHistory(HistoryType.LLM_RESP, data, eid)
        self.history.append(new_event)

    def compress_context(self, data: Any, eid: str = ""):
        new_event = AgentHistory(HistoryType.COMP_CONTEXT, data, eid)
        self.history.append(new_event)

    def reload_context(self, data: Any, eid: str = ""):
        new_event = AgentHistory(HistoryType.RELOAD_CONTEXT, data, eid)
        self.history.append(new_event)

    def undo(self, data: Any):
        ...

    def redo(self, data: Any):
        ...

    def from_checkpoint(self, data: dict):
        history_list = [h for h in list(self.history) if h.id in data["history"]]
        self.history.clear()
        self.history.extend(history_list)
