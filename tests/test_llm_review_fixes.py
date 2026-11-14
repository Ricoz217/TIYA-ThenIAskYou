import asyncio
import copy
import threading
from typing import Literal

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.chat import Chat
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, ContextOverflowError, Prompts, TextPrompt
from TIYA.llm.pricing import PriceCatalog
from TIYA.llm.provider import get_provider
from TIYA.llm.runtime import Pending, Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.tool_schema import build_tool_payloads
from TIYA.llm.usage import Ledger, estimate
from TIYA.llm.usage_model import TokenUsage


def config(*, max_context=None):
    return configuration("test", [candidate(
        "chat_completions:test:m", "chat_completions", "test", "m",
        {"endpoint": "https://test/chat"}, {"stream": False}, max_context=max_context,
    )])


def runtime(tmp_path):
    return Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))


def test_special_token_literals_are_ordinary_user_and_output_text():
    for role in ("user", "assistant"):
        assert estimate(Context(TextPrompt(role, "explain <|endoftext|>")))["input"] > 0


def test_nullable_enum_accepts_null_without_an_extra_schema_branch():
    def tool(value: Literal["a", "b"] | None = None):
        return value

    parameters = build_tool_payloads(tool)["openai"]["function"]["parameters"]
    assert parameters["properties"]["value"] == {
        "type": ["string", "null"], "enum": ["a", "b", None], "default": None,
    }
    assert "required" not in parameters


@pytest.mark.parametrize("tiers", [
    [{"rates": {"input": 1}}, {"up_to": 100, "rates": {"input": 2}}, {"rates": {"input": 3}}],
    [{"up_to": 100, "rates": {"input": 1}}, {"rates": {"input": 2}}, {"rates": {"input": 3}}],
])
def test_only_final_price_tier_may_be_unbounded(tiers):
    with pytest.raises(ValueError):
        PriceCatalog._validate({"currency": "USD", "tiers": tiers})


@pytest.mark.parametrize("end", [float("inf"), float("nan"), True])
def test_price_tier_boundary_must_be_a_finite_number(end):
    with pytest.raises(ValueError):
        PriceCatalog._validate({"currency": "USD", "tiers": [
            {"up_to": end, "rates": {"input": 1}}, {"rates": {"input": 2}},
        ]})


def test_repeated_mirror_updates_mutable_snapshot_without_duplicate_requests(tmp_path):
    source = Ledger(tmp_path / "source.db")
    target = Ledger(tmp_path / "target.db")
    call_id = source.begin_call("test")
    request_id = source.begin_request(call_id, {"model": "m"}, {"input": 10})
    first = source.get_request(request_id)
    target.mirror_request(first)
    source.observe_usage(request_id, "estimated", {"output": 7}, "local_output")
    source.observe_usage(request_id, "reported", {"input": 12, "output": 8}, "final")
    source.finish_request(request_id, "completed", remote_state="completed", response_id="r1")
    final = source.get_request(request_id)
    target.mirror_request(final)
    target.mirror_request(final)
    assert target.get_request(request_id) == final
    assert len(target.query_requests()) == 1


def test_runtime_prefers_real_usage_without_tokenizing_for_accounting(tmp_path, monkeypatch):
    import TIYA.llm.usage as usage

    def forbidden(*args, **kwargs):
        pytest.fail("Runtime accounting must not invoke tokenizer")

    monkeypatch.setattr(usage.tiktoken, "get_encoding", forbidden)

    async def run():
        rt = runtime(tmp_path)
        context = Context(TextPrompt("user", "history" * 100))
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
            })
        )) as client:
            async def gateway(route):
                return Gateway(client)

            preset = config(max_context=1000)
            result = await rt.execute(
                context, TextPrompt("user", "next"), preset, gateway, 5,
                context_usage=TokenUsage(model="m", reported={"input": 20, "output": 4}),
                usage_signature=context.signature,
                usage_scope=preset.primary.scope,
            )
            row = rt.ledger.get_request(result.result.request_id)
            assert row["reported"]["input"] == 20
            assert row["estimated"]["input"] > 0
            assert row["estimated"]["output"] > 0
            assert "characters" in row["estimated"]["algorithm"]

    asyncio.run(run())


@pytest.mark.parametrize("reported,overflow", [
    ({"input": 500, "output": 100}, False),
    ({"input": 600, "output": 100}, True),
    ({"total": 700}, True),
    ({"input": 0, "output": 0}, False),
])
def test_chat_uses_reported_usage_without_tokenizing(
    tmp_path,
    monkeypatch,
    reported,
    overflow,
):
    async def run():
        chat = Chat(config=config(max_context=1000), log_off=True)
        chat._runtime = runtime(tmp_path)
        chat._last_usage = TokenUsage(model="m", reported=reported, estimated={"input": 10000})
        chat._usage_context_signature = chat._context.signature

        async def forbidden(*args):
            pytest.fail("reported usage must take precedence over local estimates")

        async def no_post(task):
            return None

        monkeypatch.setattr("TIYA.llm.runtime.estimate_async", forbidden)
        monkeypatch.setattr(chat, "post", no_post)
        try:
            if overflow:
                with pytest.raises(ContextOverflowError):
                    await chat.ask("next", timeout=1)

            else:
                assert await chat.ask("next", timeout=1) is None

        finally:
            await chat.close()

    asyncio.run(run())


def test_chat_falls_back_to_estimation_only_without_complete_reported_usage(tmp_path, monkeypatch):
    async def run():
        chat = Chat(config=config(max_context=1000), context=Context(TextPrompt("user", "history")), log_off=True)
        chat._runtime = runtime(tmp_path)
        chat._last_usage = TokenUsage(model="m", reported={"input": 10}, estimated={"output": 0})
        chat._usage_context_signature = chat._context.signature
        seen = []

        async def fallback(prompts):
            seen.append(prompts)
            return {"input": 700}

        monkeypatch.setattr("TIYA.llm.runtime.estimate_async", fallback)
        try:
            with pytest.raises(ContextOverflowError):
                await chat.ask("next", timeout=1)

            assert len(seen) == 1

        finally:
            await chat.close()

    asyncio.run(run())


@pytest.mark.parametrize("persist_error", [False, True])
def test_committed_answer_is_published_before_checkpoint_io(
    tmp_path,
    monkeypatch,
    persist_error,
):
    from TIYA.llm.provider.base import Result

    async def run():
        rt = runtime(tmp_path)
        entered = threading.Event()
        release = threading.Event()

        def persist(session, pending):
            entered.set()
            if persist_error:
                raise RuntimeError("checkpoint failed")

            release.wait(2)

        async def execute(context, prompts, preset, gateway, timeout, **kwargs):
            planned = copy.deepcopy(context)
            planned.append(prompts)
            return Pending(
                Result(prompts=Prompts(TextPrompt("assistant", "answer"))),
                planned, context.signature, context.session_id, preset, preset.primary, 0, "call", None,
            )

        monkeypatch.setattr(rt, "_persist_commit", persist)
        monkeypatch.setattr(rt, "execute", execute)
        chat = Chat(config=config(), log_off=True)
        chat._runtime = rt
        try:
            answer = await chat.ask("question", timeout=0.1)
            assert answer[0].text == "answer"
            assert chat.last_outcome.committed
            assert chat.last_outcome.status == "completed"
            assert [p.text for p in chat._context] == ["question", "answer"]

        finally:
            release.set()
            for _ in range(100):
                if chat._running.is_set():
                    break

                await asyncio.sleep(0.005)

            await chat.close()

    asyncio.run(run())


def test_timeout_racing_with_commit_returns_the_committed_answer(tmp_path, monkeypatch):
    real_wait_for = asyncio.wait_for

    async def timeout_after_completion(awaitable, timeout):
        await real_wait_for(awaitable, timeout)
        raise TimeoutError

    async def run():
        from TIYA.llm.provider.base import Result

        rt = runtime(tmp_path)

        async def execute(context, prompts, preset, gateway, timeout, **kwargs):
            planned = copy.deepcopy(context)
            planned.append(prompts)
            return Pending(
                Result(prompts=Prompts(TextPrompt("assistant", "answer"))),
                planned, context.signature, context.session_id, preset, preset.primary, 0, "call", None,
            )

        monkeypatch.setattr(rt, "execute", execute)
        monkeypatch.setattr(asyncio, "wait_for", timeout_after_completion)
        chat = Chat(config=config(), log_off=True)
        chat._runtime = rt
        try:
            assert (await chat.ask("question", timeout=1))[0].text == "answer"
            assert chat.last_outcome.committed

        finally:
            await chat.close()

    asyncio.run(run())


def test_restored_round_uses_its_reported_usage_before_local_estimation(tmp_path, monkeypatch):
    async def run():
        round_prompts = Prompts(TextPrompt("user", "question"), TextPrompt("assistant", "answer"))
        round_prompts.usage = TokenUsage(model="m", reported={"input": 600, "output": 100})
        chat = Chat(config=config(max_context=1000), context=Context(round_prompts), log_off=True)
        chat._runtime = runtime(tmp_path)

        async def forbidden(*args):
            pytest.fail("restored reported usage must not start tokenizer")

        monkeypatch.setattr("TIYA.llm.runtime.estimate_async", forbidden)
        try:
            with pytest.raises(ContextOverflowError):
                await chat.ask("next", timeout=1)

        finally:
            await chat.close()

    asyncio.run(run())


def test_accounting_assumptions_reuse_cached_structured_content_and_preserve_signatures(monkeypatch):
    from TIYA.llm import usage
    from TIYA.llm.model import ToolResponse

    context = Context(ToolResponse("tool", "c", {"value": "long output" * 1000}))
    before = context.to_dict()
    signature = context.signature
    render = usage._prompt_text
    calls = []

    def counted(prompt):
        calls.append(prompt)
        return render(prompt)

    monkeypatch.setattr(usage, "_prompt_text", counted)
    expected = usage.assume_usage(context)
    assert usage.assume_usage(context) == expected
    assert usage.assume_usage(copy.deepcopy(context)) == expected
    assert len(calls) == 1
    assert context.signature == signature
    assert context.to_dict() == before
    context.refresh_fingerprints()
    usage.assume_usage(context)
    assert len(calls) == 2


@pytest.mark.parametrize("mutation", ["clear", "replace", "rollback", "direct_clear"])
def test_old_reported_usage_does_not_block_a_changed_context(tmp_path, monkeypatch, mutation):
    async def run():
        first = Prompts(TextPrompt("user", "old"), TextPrompt("assistant", "answer"))
        first.usage = TokenUsage(model="m", reported={"input": 150, "output": 50})
        last = Prompts(TextPrompt("user", "recent"), TextPrompt("assistant", "answer"))
        last.usage = TokenUsage(model="m", reported={"input": 600, "output": 100})
        context = Context(first, last)
        chat = Chat(config=config(max_context=1000), context=context, log_off=True)
        chat._runtime = runtime(tmp_path)

        async def no_post(task):
            return None

        monkeypatch.setattr(chat, "post", no_post)
        try:
            with pytest.raises(ContextOverflowError):
                chat._check_context_window()

            if mutation == "clear":
                await chat.clear()

            elif mutation == "replace":
                chat.replace_context(Context(TextPrompt("user", "shorter history")))

            elif mutation == "rollback":
                chat.back_to_last_round()

            else:
                context.clear()

            assert await chat.ask("next", timeout=1) is None

        finally:
            await chat.close()

    asyncio.run(run())


def test_chat_completions_rejects_multiple_branches_before_sending():
    preset = config()
    preset.primary.model_params["n"] = 2
    with pytest.raises(ValueError, match="单候选|n=1"):
        get_provider(preset.primary).encode(Context(TextPrompt("user", "question")), {})


def test_runtime_deadline_covers_materialization_without_regeneration(tmp_path, monkeypatch):
    from TIYA.llm.model import ImageResponseItem, MediaRef

    async def run():
        rt = runtime(tmp_path)
        requests = []

        async def materialize(item, gateway):
            await asyncio.sleep(0.3)
            return ImageResponseItem(MediaRef(hash_name="cached"))

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": [{"type": "image_url", "image_url": "https://test/image"}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            })

        monkeypatch.setattr(rt.media, "materialize", materialize)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            with pytest.raises(TimeoutError):
                await rt.execute(Context(), TextPrompt("user", "draw"), config(), gateway, 0.1)

        assert len(requests) == 1
        row = rt.ledger.query_requests()[0]
        assert row["remote_state"] == "completed"
        assert row["reported"]["output"] == 3

    asyncio.run(run())
