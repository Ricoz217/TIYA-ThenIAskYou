import asyncio
import copy
import json

import httpx
import pytest

from llm_config_fixtures import candidate, configuration
from TIYA.llm.config_model import LLMConfig
from TIYA.llm.gateway import Gateway
from TIYA.llm.model import Context, Prompts, SystemPrompt, TextPrompt
from TIYA.llm.provider.common import normalize
from TIYA.llm.runtime import Runtime
from TIYA.llm.state import StateStore
from TIYA.llm.usage import Ledger


def test_retry_reuses_working_history_and_borrows_candidate(tmp_path, monkeypatch):
    config = configuration(
        "main",
        [candidate("chat_completions:a:m", "chat_completions", "a", "m",
                   {"endpoint": "https://test/chat"}, {"stream": False})],
    )
    context = Context(SystemPrompt("system"), TextPrompt("user", "history"))
    before = context.to_dict()
    context_copies = []
    real_deepcopy = Context.__deepcopy__

    def copy_history(self, memo):
        context_copies.append(self)
        return real_deepcopy(self, memo)

    def copy_route(self, memo):
        pytest.fail("Runtime must borrow the already captured candidate")

    monkeypatch.setattr(Context, "__deepcopy__", copy_history)
    monkeypatch.setattr(LLMConfig, "__deepcopy__", copy_route, raising=False)
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})

        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runtime = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def gateway(route):
                return Gateway(client)

            pending = await runtime.execute(context, Prompts(TextPrompt("user", "new")),
                                            config, gateway, 5)
            assert context.to_dict() == before
            runtime.commit(context, pending)
            assert context.message.prompts[-1].text == "ok"
            assert pending.route is config.primary

    asyncio.run(run())
    assert len(context_copies) == 1
    assert payloads[0] == payloads[1]


def test_normalization_shares_unchanged_messages_without_editing_source():
    original = Context(SystemPrompt("system"), TextPrompt("user", "hello"))
    before = copy.deepcopy(original.to_dict())
    normalized = normalize(original)
    assert original.to_dict() == before
    assert normalized.message.prompts is not original.message.prompts
    assert normalized.message[1] is original.message[1]


def test_message_slice_is_a_shallow_prompts_container():
    source = Context(SystemPrompt("system"), TextPrompt("user", "one"),
                     TextPrompt("assistant", "two"))
    suffix = source.message[1:]
    assert isinstance(suffix, Prompts)
    assert suffix[0] is source.message[1]
    suffix.append(TextPrompt("user", "three"))
    assert len(source.message.prompts) == 3


def test_explicit_refresh_after_prompt_edit_rejects_stale_response(tmp_path):
    config = configuration(
        "main",
        [candidate("chat_completions:a:m", "chat_completions", "a", "m",
                   {"endpoint": "https://test/chat"}, {"stream": False})],
    )
    prompt = TextPrompt("user", "history")
    context = Context(prompt)

    async def run():
        runtime = Runtime(StateStore(tmp_path / "state.db"), Ledger(tmp_path / "usage.db"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )) as client:
            async def gateway(route):
                return Gateway(client)

            pending = await runtime.execute(context, Prompts(TextPrompt("user", "new")),
                                            config, gateway, 5)
            prompt.text = "changed"
            context.refresh_fingerprints()
            with pytest.raises(RuntimeError, match="Context"):
                runtime.commit(context, pending)

    asyncio.run(run())


def test_context_fingerprints_are_cached_until_framework_changes_history(monkeypatch):
    from TIYA.llm import model as module
    calls = []
    real_fingerprint = module.fingerprint

    def count(value):
        calls.append(value)
        return real_fingerprint(value)

    monkeypatch.setattr(module, "fingerprint", count)
    context = Context(SystemPrompt("stable"), TextPrompt("user", "one"))
    first = context.signature
    before = len(calls)
    assert context.signature == first
    assert context.message_fingerprints == context.message_fingerprints
    assert len(calls) == before
    context.append(TextPrompt("assistant", "two"))
    assert context.signature != first
    assert len(calls) > before


def test_configuration_fingerprints_cache_and_refresh_through_update(monkeypatch):
    from TIYA.llm import config_model as module
    calls = []
    real_fingerprint = module.fingerprint

    def count(value):
        calls.append(value)
        return real_fingerprint(value)

    config = configuration("main", [candidate("chat_completions:a:m", "chat_completions",
                                             "a", "m", {"endpoint": "https://test"})])
    monkeypatch.setattr(module, "fingerprint", count)
    route = config.primary
    scope = route.scope
    revision = config.revision
    before = len(calls)
    assert route.scope == scope
    assert config.revision == revision
    assert len(calls) == before
    config.model = "new-model"
    assert config.revision != revision
    assert route.scope != scope


def test_context_edits_invalidate_message_and_round_fingerprints():
    first = TextPrompt("user", "one")
    answer = TextPrompt("assistant", "two")
    context = Context(SystemPrompt("system"), first, answer)
    original = context.signature
    first_round = next(iter(context.rounds.values()))
    round_signature = first_round.signature
    answer.text = "changed"
    context.refresh_fingerprints()
    assert context.signature != original
    assert first_round.signature != round_signature
    original = context.signature
    context.discard_prompt_ids({answer.id})
    assert context.signature != original
    original = context.signature
    context.clear()
    assert context.signature != original
    assert len(context.message.prompts) == 1


def test_prompts_cache_refreshes_on_append_and_discard():
    prompt = TextPrompt("user", "one")
    prompts = Prompts(prompt)
    first = prompts.signature
    prompts.append(TextPrompt("assistant", "two"))
    assert prompts.signature != first
    first = prompts.signature
    prompts.discard_prompt_ids({prompt.id})
    assert prompts.signature != first
