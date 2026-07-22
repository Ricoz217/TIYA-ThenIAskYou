from __future__ import annotations

import inspect
import json
import unittest

from TIYA.LLM_connect import (
    Chat,
    Context,
    Prompts,
    SystemPrompt,
    TextPrompt,
    ToolCall,
    ToolResponse,
)
from TIYA.agent.agent import BaseAgent
from TIYA.agent.group_chat_agent import GroupChatAgent
from TIYA.agent.agent_structure import TaskStatus


class _DummyLogger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message: str):
        self.warnings.append(message)


class _DummyTask:
    def __init__(self):
        self.id = "task-1"
        self.function = {"demo_tool": lambda: None}
        self.status = TaskStatus.RUNNING


class TestAgentToolProtocolFallback(unittest.IsolatedAsyncioTestCase):
    def test_detach_preserves_task_and_clears_native_call_mapping(self):
        task = _DummyTask()
        agent = BaseAgent.__new__(BaseAgent)
        agent._last_call_ids = {"call-1": task.id}
        agent._logger = _DummyLogger()
        agent._get_task_lookup = lambda schedule=False: {task.id: task}

        notice = agent._detach_tool_calls_for_context_reset("测试刷新")

        self.assertEqual(agent._last_call_ids, {})
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertIn('"task_id": "task-1"', notice)
        self.assertIn('"call_id": "call-1"', notice)
        self.assertIn("get_task_data", notice)

    def test_both_compress_implementations_detach_native_tool_protocol(self):
        base_source = inspect.getsource(BaseAgent.compress_context)
        group_source = inspect.getsource(GroupChatAgent.compress_context)

        self.assertIn("_detach_tool_calls_for_context_reset", base_source)
        self.assertIn("_detach_tool_calls_for_context_reset", group_source)

    def test_run_rolls_back_before_collecting_tool_responses(self):
        run_source = inspect.getsource(BaseAgent.run)

        self.assertLess(
            run_source.index("back2last_round"),
            run_source.index("get_tool_call_result"),
        )

    def test_back2last_round_preserves_system_prompt(self):
        context = Context(SystemPrompt("system"))
        context.append(TextPrompt("user", "request"))
        context.append(ToolCall("demo_tool", "call-1", {}))

        removed = context.back2last_round()

        self.assertTrue(any(isinstance(prompt, ToolCall) for prompt in removed))
        self.assertIsNotNone(context.system)
        self.assertEqual(context.system.text, "system")
        self.assertEqual(
            [prompt.role for prompt in context.to_prompts()],
            ["system"],
        )

    async def test_payload_downgrades_orphan_tool_response(self):
        context = Context(SystemPrompt("system"))
        context.append(
            Prompts(
                ToolResponse("demo_tool", "missing-call", {"ok": True}),
                TextPrompt("user", "continue"),
            )
        )
        chat = Chat(context=context, api_provider="openai", log_off=True)
        try:
            payload = await chat._payload_constructor(context)
        finally:
            await chat.close()

        roles = [message["role"] for message in payload["messages"]]
        self.assertNotIn("tool", roles)
        fallback_text = json.dumps(payload["messages"], ensure_ascii=False)
        self.assertIn("missing-call", fallback_text)
        self.assertIn("孤立", fallback_text)

        remaining = context.to_prompts()
        self.assertFalse(
            any(
                isinstance(prompt, ToolResponse)
                and prompt.call_id == "missing-call"
                for prompt in remaining
            )
        )

        second_payload = await chat._payload_constructor(context)
        second_text = json.dumps(second_payload["messages"], ensure_ascii=False)
        self.assertNotIn("missing-call", second_text)

    def test_discarding_from_context_copy_does_not_modify_original_rounds(self):
        context = Context(SystemPrompt("system"))
        orphan = ToolResponse("demo_tool", "missing-call", {"ok": True})
        context.append(orphan)
        context_copy = context.copy()

        context_copy.discard_prompt_ids({orphan.id})

        self.assertTrue(
            any(prompt.id == orphan.id for prompt in context.to_prompts())
        )
        self.assertFalse(
            any(prompt.id == orphan.id for prompt in context_copy.to_prompts())
        )

    async def test_payload_cleanup_reaches_chat_managed_context(self):
        context = Context(SystemPrompt("system"))
        orphan = ToolResponse("demo_tool", "missing-call", {"ok": True})
        context.append(orphan)
        chat = Chat(context=context, api_provider="openai", log_off=True)
        request_context = context.copy()
        request_context.append(TextPrompt("user", "continue"))
        try:
            await chat._payload_constructor(request_context)
        finally:
            await chat.close()

        self.assertFalse(
            any(prompt.id == orphan.id for prompt in context.to_prompts())
        )

    async def test_payload_keeps_valid_tool_response(self):
        context = Context(SystemPrompt("system"))
        context.append(TextPrompt("user", "request"))
        context.append(ToolCall("demo_tool", "call-1", {}))
        context.append(ToolResponse("demo_tool", "call-1", {"ok": True}))
        chat = Chat(context=context, api_provider="openai", log_off=True)
        try:
            payload = await chat._payload_constructor(context)
        finally:
            await chat.close()

        tool_messages = [
            message
            for message in payload["messages"]
            if message["role"] == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call-1")

    async def test_payload_closes_unfinished_tool_call_before_user_message(self):
        context = Context(SystemPrompt("system"))
        context.append(TextPrompt("user", "request"))
        context.append(ToolCall("demo_tool", "call-1", {}))
        context.append(TextPrompt("user", "next request"))
        chat = Chat(context=context, api_provider="openai", log_off=True)
        try:
            payload = await chat._payload_constructor(context)
        finally:
            await chat.close()

        messages = payload["messages"]
        tool_index = next(
            index
            for index, message in enumerate(messages)
            if message["role"] == "tool"
        )
        next_user_index = next(
            index
            for index, message in enumerate(messages)
            if (
                message["role"] == "user"
                and any(
                    part.get("text") == "next request"
                    for part in message["content"]
                )
            )
        )
        self.assertLess(tool_index, next_user_index)
        self.assertEqual(messages[tool_index]["tool_call_id"], "call-1")

    async def test_payload_closes_pending_call_before_orphan_notice(self):
        context = Context(SystemPrompt("system"))
        context.append(TextPrompt("user", "request"))
        context.append(ToolCall("demo_tool", "pending-call", {}))
        context.append(ToolResponse("demo_tool", "orphan-call", {"ok": True}))
        chat = Chat(context=context, api_provider="openai", log_off=True)
        try:
            payload = await chat._payload_constructor(context)
        finally:
            await chat.close()

        messages = payload["messages"]
        pending_response_index = next(
            index
            for index, message in enumerate(messages)
            if (
                message["role"] == "tool"
                and message["tool_call_id"] == "pending-call"
            )
        )
        orphan_notice_index = next(
            index
            for index, message in enumerate(messages)
            if (
                message["role"] == "user"
                and "orphan-call" in json.dumps(message, ensure_ascii=False)
            )
        )
        self.assertLess(pending_response_index, orphan_notice_index)


if __name__ == "__main__":
    unittest.main(verbosity=2)
