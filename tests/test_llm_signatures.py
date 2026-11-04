"""消息签名只从对象接口读取，容器复用成员缓存，不探测原位写入。"""

import copy

import pytest

from TIYA.llm import model
from TIYA.llm.model import Context, Prompts, SystemPrompt, TextPrompt, ToolCall, ToolInput
from TIYA.llm.provider.base import Result
from TIYA.llm.provider.common import attach_native, native_items


def test_prompt_signature_is_cached_until_explicit_refresh(monkeypatch):
    message = TextPrompt("assistant", "before")
    original = model.fingerprint
    hashed = []

    def count(value):
        hashed.append(value)
        return original(value)

    monkeypatch.setattr(model, "fingerprint", count)
    before = message.signature
    assert message.signature == before
    assert len(hashed) == 1
    message.text = "after"
    assert message.signature == before
    assert len(hashed) == 1
    message.refresh_fingerprints()
    assert message.signature != before
    assert len(hashed) == 2


def test_context_and_prompts_reuse_member_signatures_without_rescanning_content(monkeypatch):
    def tool(value: str):
        return value

    system = SystemPrompt("system")
    history = TextPrompt("user", "history" * 1000)
    tool_input = ToolInput(tool)
    context = Context(system, history, tool_input)
    members = [system, history, tool_input]
    expected = [message.signature for message in members]

    def forbidden(value):
        pytest.fail("成员签名已有缓存，上层不应再次展开原始消息内容")

    monkeypatch.setattr(model, "stable", forbidden)
    assert context.message_fingerprints == tuple(expected[:2])
    assert context.signature
    assert context.header_signature
    assert Prompts(members).signature


def test_append_only_hashes_the_new_message_and_container_aggregation(monkeypatch):
    context = Context(SystemPrompt("system"), TextPrompt("user", "history" * 1000))
    original_context_signature = context.signature
    _ = context.header_signature
    previous_round = next(iter(context.rounds.values()))
    _ = previous_round.signature
    original_stable = model.stable
    scanned = []

    def inspect(value):
        if isinstance(value, dict) and "contain_type" in value:
            scanned.append(value["id"])

        return original_stable(value)

    monkeypatch.setattr(model, "stable", inspect)
    incoming = TextPrompt("assistant", "new")
    context.append(incoming)
    assert context.signature != original_context_signature
    assert context.header_signature
    assert context.message.signature
    assert next(iter(context.rounds.values())).signature
    assert scanned == [incoming.id]


def test_native_signature_snapshot_uses_the_same_message_signature():
    answer = TextPrompt("assistant", "answer")
    call = ToolCall("tool", "call", {"value": "x"})
    prompts = Prompts(answer, call)
    context = Context(prompts)
    before = context.signature, prompts.signature, answer.signature, call.signature
    result = Result(prompts=prompts, native=[{"type": "opaque", "value": "native"}])
    attach_native(result, "responses_api", "scope")
    record = answer.native["responses_api"]
    assert record["signatures"] == {answer.id: answer.signature, call.id: call.signature}
    assert "projection_hashes" not in record
    context.refresh_fingerprints(content_changed=False)
    assert (context.signature, prompts.signature, answer.signature, call.signature) == before
    index = {message.id: message for message in prompts}
    assert native_items(answer, "responses_api", "scope", index) is record
    call.arguments["value"] = "changed"
    # 原位写入不触发框架探测；只有显式刷新后才获得新签名。
    assert native_items(answer, "responses_api", "scope", index) is record
    context.refresh_fingerprints()
    assert native_items(answer, "responses_api", "scope", index) is None


def test_missing_native_member_does_not_replay_the_original_group():
    answer = TextPrompt("assistant", "answer")
    call = ToolCall("tool", "call", {})
    result = Result(prompts=Prompts(answer, call), native=[{"type": "opaque"}])
    attach_native(result, "responses_api", "scope")
    assert native_items(answer, "responses_api", "scope", {answer.id: answer}) is None


def test_signature_cache_is_not_serialized_and_native_survives_archive():
    answer = TextPrompt("assistant", "answer")
    result = Result(prompts=Prompts(answer), native=[{"type": "opaque"}])
    attach_native(result, "responses_api", "scope")
    context = Context(result.prompts)
    expected = context.signature
    data = context.to_dict()
    assert "_signature" not in answer.to_dict()
    restored = Context.from_dict(data, {})
    assert restored.signature == expected
    restored_answer = restored.message[0]
    assert native_items(restored_answer, "responses_api", "scope", {restored_answer.id: restored_answer})


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy])
def test_prompt_snapshots_reuse_a_warm_signature(clone, monkeypatch):
    message = TextPrompt("user", "history")
    expected = message.signature
    duplicate = clone(message)

    def forbidden(value):
        pytest.fail("保留内容和 ID 的副本应复用已计算的签名")

    monkeypatch.setattr(model, "stable", forbidden)
    assert duplicate.signature == expected


def test_tool_deepcopy_and_schema_refresh_keep_the_single_signature_contract(monkeypatch):
    def first(value: str):
        return value

    def second(value: int):
        return value

    tool = ToolInput(first)
    expected = tool.signature
    duplicate = copy.deepcopy(tool)
    assert duplicate.signature == expected
    assert duplicate.tool_payloads is tool.tool_payloads
    tool.function = second
    assert tool.signature == expected
    tool.refresh_schema()
    assert tool.signature != expected
    assert duplicate.signature == expected


def test_system_shallow_copy_gets_a_signature_for_its_new_message_id():
    system = SystemPrompt("system")
    signature = system.signature
    duplicate = system.copy()
    assert duplicate.id != system.id
    assert duplicate.signature != signature
    assert copy.deepcopy(system).signature == signature
