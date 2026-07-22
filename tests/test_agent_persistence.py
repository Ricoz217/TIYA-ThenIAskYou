import asyncio
import json
import threading
import time

from TIYA.LLM_connect import Context, TextPrompt
from TIYA.agent.agent import BaseAgent
from TIYA.agent.agent_structure import (
    AgentContext,
    AgentHistory,
    AgentHistoryManager,
    AgentTaskQueue,
    HistoryType,
)


class _Logger:
    def __init__(self):
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def debug(self, _message: str) -> None:
        pass


class _SerializedTask:
    def __init__(
            self,
            value: str,
            started: threading.Event | None = None,
            resume: threading.Event | None = None,
    ) -> None:
        self.value = value
        self.started = started
        self.resume = resume

    def to_dict(self) -> dict[str, str]:
        if self.started is not None:
            self.started.set()
        if self.resume is not None and not self.resume.wait(timeout=1):
            raise TimeoutError("task serialization did not resume")
        return {"value": self.value}


def _context(text: str) -> Context:
    return Context(TextPrompt(role="user", text=text))


def _bare_agent(tmp_path) -> BaseAgent:
    agent = object.__new__(BaseAgent)
    agent._history = AgentHistoryManager(tmp_path, max_history=10)
    agent._context = AgentContext(max_history=10)
    agent._functions = {}
    agent._logger = _Logger()
    agent._log_prefix = lambda: "Agent test"
    return agent


def test_agent_history_serializes_event_id() -> None:
    history = AgentHistory(
        event_type=HistoryType.USER_ASK,
        event_data={"text": "hello"},
        event_id="event-123",
    )

    assert history.to_dict()["event_id"] == "event-123"


def test_split_context_files_round_trip(tmp_path) -> None:
    agent = _bare_agent(tmp_path)
    agent._context.current_window = _context("current")
    agent._context.window_history = {"old": _context("history")}

    BaseAgent._save_current_context(agent)
    BaseAgent._save_context_history(agent)

    loaded = _bare_agent(tmp_path)
    BaseAgent._load_context(loaded)

    assert loaded._context.current_window.to_dict() == agent._context.current_window.to_dict()
    assert {
        key: value.to_dict() for key, value in loaded._context.window_history.items()
    } == {
        key: value.to_dict() for key, value in agent._context.window_history.items()
    }
    assert agent._history.current_context_file.is_file()
    assert agent._history.context_history_file.is_file()
    assert not agent._history.context_file.exists()
    assert not loaded._logger.errors


def test_new_current_context_uses_legacy_history_fallback(tmp_path) -> None:
    legacy = AgentContext(max_history=10)
    legacy.current_window = _context("legacy current")
    legacy.window_history = {"legacy": _context("legacy history")}

    agent = _bare_agent(tmp_path)
    agent._history.context_file.write_text(
        json.dumps({"data": legacy.to_dict()}, ensure_ascii=False),
        encoding="utf-8",
    )
    agent._context.current_window = _context("new current")
    BaseAgent._save_current_context(agent)

    loaded = _bare_agent(tmp_path)
    BaseAgent._load_context(loaded)

    assert loaded._context.current_window.to_dict() == agent._context.current_window.to_dict()
    assert loaded._context.window_history["legacy"].to_dict() == legacy.window_history["legacy"].to_dict()
    assert not loaded._logger.errors


def test_ordinary_save_does_not_create_checkpoint() -> None:
    calls: list[str] = []
    agent = object.__new__(BaseAgent)
    agent._save_task = lambda _dt: calls.append("task")
    agent._save_schedule_task = lambda: calls.append("schedule")
    agent._save_history = lambda _dt: calls.append("history")
    agent._save_current_context = lambda _dt: calls.append("current")
    agent._save_context_history = lambda _dt: calls.append("context_history")
    agent._save_checkpoint = lambda _dt=None: calls.append("checkpoint")
    agent._save_websearch = lambda: calls.append("websearch")

    BaseAgent.save_agent(agent)
    assert calls == ["task", "schedule", "history", "current", "websearch"]

    calls.clear()
    BaseAgent.save_agent(agent, save_context_history=True)
    assert calls == ["task", "schedule", "history", "current", "context_history", "websearch"]


def test_async_saves_use_agent_executor_and_serialize_per_agent() -> None:
    async def run_test() -> None:
        agent = object.__new__(BaseAgent)
        agent._persistence_lock = asyncio.Lock()
        event_loop_thread = threading.get_ident()
        worker_threads: list[int] = []
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        def save_agent(*, save_context_history: bool = False) -> None:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            worker_threads.append(threading.get_ident())
            time.sleep(0.02)
            with state_lock:
                active -= 1

        agent.save_agent = save_agent
        await asyncio.gather(
            BaseAgent.save_agent_async(agent),
            BaseAgent.save_agent_async(agent, save_context_history=True),
        )

        assert worker_threads
        assert all(thread_id != event_loop_thread for thread_id in worker_threads)
        assert max_active == 1

    asyncio.run(run_test())


def test_task_queue_serialization_copies_pending_tasks_before_iteration() -> None:
    queue = AgentTaskQueue(max_accomplished_task=10)
    started = threading.Event()
    resume = threading.Event()
    queue.tasks["first"] = _SerializedTask("first", started, resume)
    queue.tasks["second"] = _SerializedTask("second")

    def mutate() -> None:
        if started.wait(timeout=1):
            queue.tasks["late"] = _SerializedTask("late")
        resume.set()

    mutation = threading.Thread(target=mutate)
    mutation.start()
    try:
        snapshot = queue.to_dict()
    finally:
        resume.set()
        mutation.join(timeout=1)

    assert set(snapshot["pending"]) == {"first", "second"}


def test_task_queue_serialization_copies_accomplished_tasks_before_iteration() -> None:
    queue = AgentTaskQueue(max_accomplished_task=10)
    started = threading.Event()
    resume = threading.Event()
    queue.accomplished_task["first"] = _SerializedTask("first", started, resume)
    queue.accomplished_task["second"] = _SerializedTask("second")

    def mutate() -> None:
        if started.wait(timeout=1):
            queue.accomplished_task["late"] = _SerializedTask("late")
        resume.set()

    mutation = threading.Thread(target=mutate)
    mutation.start()
    try:
        snapshot = queue.to_dict()
    finally:
        resume.set()
        mutation.join(timeout=1)

    assert set(snapshot["accomplished"]) == {"first", "second"}
