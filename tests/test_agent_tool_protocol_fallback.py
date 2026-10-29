from __future__ import annotations

import asyncio
import inspect
import json
import unittest

import httpx
import pytest

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


class TestAgentToolProtocolFallback(unittest.TestCase):
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
        self.assertEqual(context.system.text, "system")
        self.assertEqual([prompt.role for prompt in context.to_prompts()], ["system"])

    def test_discarding_from_context_copy_does_not_modify_original_rounds(self):
        context = Context(SystemPrompt("system"))
        orphan = ToolResponse("demo_tool", "missing-call", {"ok": True})
        context.append(orphan)
        context_copy = context.copy()
        context_copy.discard_prompt_ids({orphan.id})
        self.assertTrue(any(prompt.id == orphan.id for prompt in context.to_prompts()))
        self.assertFalse(
            any(prompt.id == orphan.id for prompt in context_copy.to_prompts())
        )


@pytest.fixture
def model_http(monkeypatch):
    requests, control = [], {"status": 200}
    real_client = httpx.AsyncClient

    def handler(request):
        requests.append(json.loads(request.content))
        if control["status"] != 200:
            return httpx.Response(
                control["status"], json={"error": {"code": "invalid_request_error"}}
            )
        if request.url.path == "/messages":
            body = {"id": "msg-test", "content": [{"type": "text", "text": "ok"}]}
        else:
            body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handler)),
    )
    return requests, control


def chat_for(context, provider):
    return Chat(
        endpoint="https://model.invalid/"
        + ("messages" if provider == "anthropic" else "chat/completions"),
        token="dummy",
        model="model",
        context=context,
        api_provider=provider,
        model_params={"stream": False},
        log_off=True,
    )


def tool_results(payload, provider):
    if provider == "openai":
        return [
            (message["tool_call_id"], message["content"])
            for message in payload["messages"]
            if message["role"] == "tool"
        ]
    return [
        (block["tool_use_id"], block["content"])
        for message in payload["messages"]
        for block in message["content"]
        if block["type"] == "tool_result"
    ]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("no_wait", [False, True])
def test_orphan_result_becomes_one_notice_after_success(model_http, provider, no_wait):
    requests, _ = model_http
    context = Context(SystemPrompt("system"))
    context.append(ToolResponse("demo_tool", "missing-call", {"ok": True}))

    async def run():
        async with chat_for(context, provider) as chat:
            response = await chat.ask("continue", no_wait=no_wait, timeout=5)
            assert response[0].text == "ok"
            assert tool_results(requests[0], provider) == []
            assert "missing-call" in json.dumps(requests[0], ensure_ascii=False)
            committed = (await chat.get_context()).to_prompts()
            assert not any(isinstance(p, ToolResponse) for p in committed)
            notices = [
                p
                for p in committed
                if isinstance(p, TextPrompt) and "孤立 TOOL RESPONSE" in p.text
            ]
            assert len(notices) == 1
            assert (await chat.ask("next", no_wait=no_wait, timeout=5))[0].text == "ok"
            assert (
                json.dumps(requests[1], ensure_ascii=False).count("孤立 TOOL RESPONSE")
                == 1
            )

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("no_wait", [False, True])
def test_failed_request_keeps_original_tool_history(model_http, provider, no_wait):
    requests, control = model_http
    context = Context(SystemPrompt("system"))
    orphan = ToolResponse("demo_tool", "missing-call", {"ok": True})
    context.append(orphan)
    original_ids = [p.id for p in context.to_prompts()]

    async def run():
        async with chat_for(context, provider) as chat:
            control["status"] = 400
            assert await chat.ask("continue", no_wait=no_wait, timeout=5) is None
            assert [p.id for p in context.to_prompts()] == original_ids
            assert not any(isinstance(p, TextPrompt) for p in context.to_prompts())
            control["status"] = 200
            assert (await chat.ask("retry", no_wait=no_wait, timeout=5))[0].text == "ok"
            assert len(requests) == 2
            assert not any(isinstance(p, ToolResponse) for p in context.to_prompts())

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("no_wait", [False, True])
def test_valid_tool_result_survives_send_and_commit(model_http, provider, no_wait):
    requests, _ = model_http
    context = Context(SystemPrompt("system"))
    context.append(
        Prompts(TextPrompt("user", "request"), ToolCall("demo_tool", "call-1", {}))
    )
    result = ToolResponse("demo_tool", "call-1", {"ok": True})

    async def run():
        async with chat_for(context, provider) as chat:
            assert (await chat.ask(Prompts(result), no_wait=no_wait, timeout=5))[
                0
            ].text == "ok"
            assert tool_results(requests[0], provider) == [
                ("call-1", json.dumps({"ok": True}))
            ]
            assert (
                sum(
                    isinstance(p, ToolResponse) and p.call_id == "call-1"
                    for p in context.to_prompts()
                )
                == 1
            )

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("no_wait", [False, True])
@pytest.mark.parametrize("next_kind", ["user", "orphan"])
def test_pending_tool_is_closed_before_later_input(
    model_http, provider, no_wait, next_kind
):
    requests, _ = model_http
    context = Context(SystemPrompt("system"))
    context.append(
        Prompts(
            TextPrompt("user", "request"), ToolCall("demo_tool", "pending-call", {})
        )
    )
    later = (
        TextPrompt("user", "next request")
        if next_kind == "user"
        else ToolResponse("demo_tool", "orphan-call", {"ok": True})
    )

    async def run():
        async with chat_for(context, provider) as chat:
            assert (await chat.ask(Prompts(later), no_wait=no_wait, timeout=5))[
                0
            ].text == "ok"
            results = tool_results(requests[0], provider)
            assert len(results) == 1 and results[0][0] == "pending-call"
            assert json.loads(results[0][1])["system"]["status"] == "协议已中止"
            wire_text = json.dumps(requests[0], ensure_ascii=False)
            marker = "next request" if next_kind == "user" else "orphan-call"
            assert wire_text.index("协议已中止") < wire_text.index(marker)
            committed = context.to_prompts()
            assert (
                sum(
                    isinstance(p, ToolCall) and p.call_id == "pending-call"
                    for p in committed
                )
                == 1
            )
            assert (
                sum(
                    isinstance(p, ToolResponse) and p.call_id == "pending-call"
                    for p in committed
                )
                == 1
            )

    asyncio.run(run())
