import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from TIYA.llm.model import CallOutcome, Context, Prompts, SystemPrompt, TextPrompt


def test_agent_does_not_retry_a_terminal_communication_result(monkeypatch):
    import TIYA.agent.agent as module

    monkeypatch.setattr(module, "_log", Mock())
    monkeypatch.setattr(
        module,
        "SETTING_CFG",
        SimpleNamespace(Agent=SimpleNamespace(MaxReactTimes=20, NoteReactTimes=15)),
    )

    async def run():
        agent = module.BaseAgent.__new__(module.BaseAgent)
        agent._initiated = True
        agent._suspending = False
        agent._react_times = 0
        agent._retry_times = 0
        agent._pending_upload_image = []
        agent.control_retry_limit = 3
        agent.control_timeout = 1
        agent.control = SimpleNamespace(
            context=Context(SystemPrompt("system")),
            ask=AsyncMock(return_value=None),
            last_outcome=CallOutcome("failed", False, remote_state="unknown"),
        )
        agent._logger = Mock()
        agent._log_prefix = lambda: "test"
        agent._update_status = lambda: None
        agent.get_tool_call_result = lambda _: Prompts()
        agent.event = asyncio.Event()
        agent.event.set()
        agent.suspend_event = asyncio.Event()
        assert await agent.run(Prompts(TextPrompt("user", "hi"))) is None
        assert agent.control.ask.await_count == 1
        assert agent.event.is_set()

    asyncio.run(run())
