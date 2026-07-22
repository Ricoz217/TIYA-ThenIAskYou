import unittest
from types import SimpleNamespace
from unittest.mock import patch

import TIYA.agent.agent_structure as agent_structure_module
from TIYA.agent.agent import BaseAgent
from TIYA.agent.agent_structure import (
    AgentScheTask,
    AgentScheTaskManage,
    AgentTaskQueue,
    TaskStatus,
    Trigger,
)


class _RecordingLogger:
    def __init__(self):
        self.debug_messages: list[str] = []
        self.error_messages: list[str] = []

    def debug(self, message):
        self.debug_messages.append(str(message))

    def error(self, message):
        self.error_messages.append(str(message))


class TestAgentScheduleExecution(unittest.IsolatedAsyncioTestCase):
    def _new_agent(self, task: AgentScheTask):
        setting = SimpleNamespace(
            Agent=SimpleNamespace(
                AgentWorkerTimeout=900,
                MaxAgentDoneTask=100,
            )
        )
        with patch.object(agent_structure_module, "SETTING_CFG", setting):
            tasks = AgentScheTaskManage()
            task_queue = AgentTaskQueue()
        tasks.tasks[task.id] = task
        logger = _RecordingLogger()
        callbacks: list[AgentScheTask] = []
        agent = SimpleNamespace(
            _initiated=True,
            _schedule_tasks=tasks,
            _task_queue=task_queue,
            _logger=logger,
            _callback=callbacks.append,
        )
        return agent, logger, callbacks

    async def test_recurring_task_gets_a_fresh_retry_budget_each_fire(self):
        call_count = 0

        async def succeed():
            nonlocal call_count
            call_count += 1
            return "new result"

        task = AgentScheTask(
            function={"succeed": succeed},
            id="recurring-task",
            name="recurring task",
            trigger_spec=Trigger(minute="0"),
            retry_limit=3,
            timeout=1,
            attempt=4,
            status=TaskStatus.RUNNING,
            results={"result": "stale result"},
            exceptions=[RuntimeError("stale failure")],
        )
        agent, logger, callbacks = self._new_agent(task)

        with patch("TIYA.agent.agent._log", logger):
            await BaseAgent.run_schedule_task(agent, task.id)

        self.assertEqual(call_count, 1)
        self.assertEqual(task.status, TaskStatus.READY)
        self.assertEqual(task.results, {"result": "new result"})
        self.assertEqual(task.exceptions, [])
        self.assertEqual(callbacks, [task])

    async def test_retry_exhaustion_logs_and_keeps_failure_status(self):
        call_count = 0

        async def fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("scheduled failure")

        task = AgentScheTask(
            function={"fail": fail},
            id="failing-task",
            name="failing task",
            trigger_spec=Trigger(minute="0"),
            retry_limit=2,
            timeout=1,
        )
        agent, logger, callbacks = self._new_agent(task)

        with patch("TIYA.agent.agent._log", logger):
            await BaseAgent.run_schedule_task(agent, task.id)

        self.assertEqual(call_count, 2)
        self.assertEqual(task.status, TaskStatus.EXCEPTION)
        self.assertTrue(task.event.is_set())
        self.assertTrue(any("已达到重试上限" in message for message in logger.error_messages))
        self.assertEqual(callbacks, [task])


if __name__ == "__main__":
    unittest.main(verbosity=2)
