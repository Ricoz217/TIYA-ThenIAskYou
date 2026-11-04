"""Provider 公共链路：schema 复用、原生校验和实时日志的业务回归。"""

import asyncio
import copy
import json

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm import model
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, ImagePrompt, Prompts, TextPrompt, ToolInput
from TIYA.llm.provider import get_provider
from TIYA.llm.provider.base import Result, classify_error
from TIYA.llm.provider.common import attach_native, native_items
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger, estimate


def test_tool_schema_is_built_once_across_encoding_hashing_estimates_and_snapshots(monkeypatch):
    def tool(value: str) -> str:
        return value

    original = model.build_tool_payloads
    builds = []

    def build(function):
        builds.append(function)
        return original(function)

    monkeypatch.setattr(model, "build_tool_payloads", build)
    context = Context(ToolInput(tool, "alias"), TextPrompt("user", "hi"))
    _ = context.signature, context.header_signature
    estimate(context)
    snapshot = copy.deepcopy(context)
    for category in ("responses_api", "chat_completions", "anthropic_api"):
        provider = get_provider(candidate("", category, "a", "m", {}))
        first = provider.encode(snapshot, {})
        second = provider.encode(snapshot, {})
        assert "alias" in json.dumps(first)
        assert first == second

    assert builds == [tool]
    # Responses 移除 returns、各协议改别名，不能污染共享 schema。
    assert context.tools[0].tool_payloads["openai"]["function"]["name"] == "tool"
    assert "returns" in context.tools[0].tool_payloads["openai"]["function"]


def test_message_signature_is_cached_and_explicit_refresh_invalidates(monkeypatch):
    message = TextPrompt("assistant", "before")
    context = Context(message)
    expected = message.signature
    original = model.fingerprint
    calls = []

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(model, "fingerprint", counted)
    assert message.signature == expected
    assert calls == []
    message.text = "after"
    context.refresh_fingerprints()
    assert message.signature != expected
    assert len(calls) == 1


def test_native_replay_uses_message_index_and_rejects_changed_signature():
    message = TextPrompt("assistant", "before")
    result = Result(prompts=Prompts(message), native=[{"type": "message", "content": []}])
    attach_native(result, "responses_api", "scope")
    index = {message.id: message}
    assert native_items(message, "responses_api", "scope", index)
    assert native_items(message, "responses_api", "other", index) is None
    message.text = "changed"
    message.refresh_fingerprints()
    assert native_items(message, "responses_api", "scope", index) is None


@pytest.mark.parametrize("dialect", ["openai", "deepseek", "anthropic", "unknown"])
def test_vendor_codes_do_not_leak_into_other_dialects(dialect):
    failure = classify_error(200, {"error": {"code": "1305"}}, dialect)
    assert failure.category == "unknown"
    assert not failure.retryable
    assert not failure.fallback_allowed


def test_glm_code_and_anthropic_type_have_independent_rules():
    assert classify_error(200, {"error": {"code": "1305"}}, "glm").category == "unavailable"
    failure = classify_error(200, {"error": {"type": "rate_limit_error"}}, "anthropic")
    assert failure.retryable and failure.safe
    # 未声明方言不能套用其他服务的错误语义。
    assert classify_error(200, {"error": {"type": "rate_limit_error"}}, "other").category == "unknown"


def test_new_error_dialect_uses_the_same_table_contract(monkeypatch):
    from TIYA.llm.provider.error_codes import DIALECTS, ErrorDialect, ErrorRule

    monkeypatch.setitem(DIALECTS, "custom", ErrorDialect(
        {"busy": ErrorRule("capacity", retryable=True, fallback_allowed=False)},
        error_path=("result", "failure"), code_fields=("reason",),
    ))
    failure = classify_error(200, {"result": {"failure": {"reason": "busy"}}}, "custom")
    assert failure.category == "capacity"
    assert failure.safe and failure.retryable
    assert not failure.fallback_allowed


def test_cold_tool_snapshots_share_schema_and_explicit_refresh_rebuilds(monkeypatch):
    def first(value: str):
        return value

    def second(value: int):
        return value

    original = model.build_tool_payloads
    builds = []

    def build(function):
        builds.append(function)
        return original(function)

    monkeypatch.setattr(model, "build_tool_payloads", build)
    tool = ToolInput(first)
    previous = copy.deepcopy(tool)
    current = copy.deepcopy(tool)
    assert previous.tool_payloads is current.tool_payloads is tool.tool_payloads
    tool.function = second
    tool.refresh_schema()
    assert tool.tool_payloads["openai"]["function"]["name"] == "second"
    assert previous.tool_payloads["openai"]["function"]["name"] == "first"
    assert builds == [first, second]


def test_chat_stream_buffers_fragments_and_preserves_image_order(monkeypatch):
    provider = get_provider(candidate("", "chat_completions", "a", "m", {}))
    decoder = provider.stream_decoder()
    chunks = ["hello", " world", [{"type": "image_url", "image_url": {"url": "https://image"}}], "after", " image"]

    def forbidden(*args, **kwargs):
        raise AssertionError("流式片段缓存不应复制收到的内容块")

    monkeypatch.setattr("TIYA.llm.provider.openai_chat.copy.deepcopy", forbidden)
    for chunk in chunks:
        decoder.feed({"choices": [{"delta": {"content": chunk}}]})

    decoder.feed({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    result = decoder.finish()
    assert result.prompts[0].text == "hello world"
    assert isinstance(result.prompts[1], ImagePrompt)
    assert result.prompts[2].text == "after image"
    assert chunks[2] == [{"type": "image_url", "image_url": {"url": "https://image"}}]


class RecordingLogger:
    def __init__(self):
        self.writes = []
        self.closed = 0
        self.opened = 0

    def info(self, text, *, stream=False):
        if stream:
            self.opened += 1
            return self

    def write(self, text):
        self.writes.append(text)

    def close(self):
        self.closed += 1


@pytest.mark.parametrize("category", ["chat_completions", "responses_api", "anthropic_api"])
@pytest.mark.parametrize("log_off", [False, True])
def test_runtime_logs_during_stream_not_only_after_finish(tmp_path, category, log_off):
    logger = RecordingLogger()
    route = candidate("", category, "a", "m", {"endpoint": "https://a"}, {"stream": True})
    config = configuration("test", [route])
    config.log_off = log_off
    if category == "chat_completions":
        events = [
            {"choices": [{"delta": {"reasoning": "thinking"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
    elif category == "responses_api":
        events = [
            {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
            {"type": "response.output_text.delta", "delta": "answer"},
            {"type": "response.completed", "response": {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}]}},
        ]
    else:
        events = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "answer"}},
            {"type": "message_stop"},
        ]

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index, event in enumerate(events):
                if index == 2 and not log_off:
                    assert "answer" in "".join(logger.writes)

                yield f"data: {json.dumps(event)}\n\n".encode()

    async def run():
        runtime = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"), logger=logger)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Body(), headers={"content-type": "text/event-stream"}))) as client:
            async def gateway(route):
                return Gateway(client)

            result = await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5)
            assert result.result.prompts[0].text == "answer"

    asyncio.run(run())
    assert logger.opened == (0 if log_off else 1)
    assert logger.closed == logger.opened
    if log_off:
        assert logger.writes == []


@pytest.mark.parametrize("ending", ["error", "cancel"])
def test_stream_log_closes_after_failure_or_cancellation(tmp_path, ending):
    logger = RecordingLogger()

    async def run():
        started = asyncio.Event()

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                started.set()
                if ending == "error":
                    raise httpx.ReadError("stream lost")

                await asyncio.Event().wait()

        route = candidate("", "chat_completions", "a", "m", {"endpoint": "https://a"}, {"stream": True})
        config = configuration("test", [route])
        runtime = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"), logger=logger)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Body(), headers={"content-type": "text/event-stream"}))) as client:
            async def gateway(route):
                return Gateway(client)

            task = asyncio.create_task(runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5))
            await asyncio.wait_for(started.wait(), 2)
            if ending == "cancel":
                task.cancel()

            with pytest.raises(asyncio.CancelledError if ending == "cancel" else httpx.ReadError):
                await task

    asyncio.run(run())
    assert "partial" in "".join(logger.writes)
    assert logger.opened == logger.closed == 1


def test_streamed_text_is_persisted_by_the_real_project_logger(tmp_path):
    from TIYA.logger import Logger, LoggerConfig

    log_root = tmp_path / "logs"
    logger = Logger(LoggerConfig(logs_dir=log_root, use_console=False, typing_pause_enabled=False))

    async def run():
        route = candidate("", "chat_completions", "a", "m", {"endpoint": "https://a"}, {"stream": True})
        config = configuration("test", [route])
        runtime = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"), logger=logger)
        stream = (
            'data: {"choices":[{"delta":{"reasoning_content":"review-reason"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"review-answer"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"}))) as client:
            async def gateway(route):
                return Gateway(client)

            await runtime.execute(Context(), Prompts(TextPrompt("user", "hi")), config, gateway, 5)

    try:
        asyncio.run(run())
        logger.flush()
        text = "".join(path.read_text(encoding="utf-8") for path in log_root.rglob("app.log"))
        assert text.count("review-reason") == 1
        assert text.count("review-answer") == 1
        assert "思维链结束，最终回答" in text

    finally:
        logger.shutdown()


def test_legacy_stream_tool_call_and_reasoning_alias_are_preserved():
    provider = get_provider(candidate("", "chat_completions", "a", "m", {}))
    decoder = provider.stream_decoder()
    decoder.feed({"choices": [{"delta": {"reasoning": "reason", "content": "text", "function_call": {"name": "tool", "arguments": '{"value":'}}}]})
    decoder.feed({"choices": [{"delta": {"function_call": {"arguments": '"x"}'}}, "finish_reason": "function_call"}]})
    result = decoder.finish()
    assert result.prompts[0].reasoning_content == "reason"
    assert result.prompts[1].function_name == "tool"
    assert result.prompts[1].arguments == {"value": "x"}

