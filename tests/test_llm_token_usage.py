import copy

import pytest

from TIYA.LLM_connect import Context, Prompts, TextPrompt, TokenUsage


def test_reported_estimated_and_missing_values_remain_distinguishable():
    usage = TokenUsage(
        model="model-a",
        request_id="request-a",
        call_id="call-a",
        reported={"input": 0, "cached": None},
        estimated={"input": 100, "output": 12, "algorithm": "local-test"},
    )
    assert usage.input_t == 0
    assert usage.output_t == usage.total_t == 12
    assert usage.cache_hit_t == 0
    assert usage.effective["cached"] is None
    assert usage.reported == {"input": 0, "cached": None}
    assert usage.estimated == {"input": 100, "output": 12}
    assert usage.sources == {"input": "reported", "output": "estimated", "total": "derived"}
    assert "cached" in usage.incomplete_fields
    assert usage.request_ids == ("request-a",)
    assert usage.call_ids == ("call-a",)
    assert usage.samples[0].estimated["algorithm"] == "local-test"


def test_effective_is_resolved_per_request_before_aggregation():
    reported = TokenUsage(model="a", reported={"input": 100, "output": 2}, estimated={"input": 150}, request_id="r1")
    estimated = TokenUsage(model="b", estimated={"input": 20, "output": 3}, request_id="r2")
    combined = reported + estimated
    assert combined.input_t == 120
    assert combined.output_t == 5
    assert combined.total_t == 125
    assert combined.reported["input"] == 100
    assert combined.estimated["input"] == 170
    assert combined.sources["input"] == "mixed"
    assert combined.models == ("a", "b")
    assert combined.model == "b"
    assert reported.model == "a" and reported.input_t == 100
    assert combined.request_ids == ("r1", "r2")


def test_same_request_snapshot_is_updated_without_counting_it_twice():
    first = TokenUsage(model="a", estimated={"input": 100}, request_id="r1")
    later = TokenUsage(model="a", reported={"input": 80, "output": 4}, request_id="r1")
    combined = first + later
    assert combined.input_t == 80
    assert combined.total_t == 84
    assert len(combined.samples) == 1
    assert combined.request_ids == ("r1",)


def test_legacy_fields_and_archive_are_read_as_unknown_source():
    archive = {"input_t": 10, "output_t": 2, "cache_hit_t": 3, "total_t": 12, "model": "old", "id": "old-id"}
    usage = TokenUsage.from_dict(archive)
    assert usage.id == "old-id"
    assert usage.input_t == 10 and usage.total_t == 12
    assert usage.reported == {} and usage.estimated == {}
    assert usage.unknown == {"input": 10, "output": 2, "cached": 3, "total": 12}
    assert set(usage.sources.values()) == {"unknown"}
    assert usage.request_ids == ()
    assert TokenUsage(10, 2, 3, 12, "old").input_t == 10


def test_new_context_archive_preserves_sources_and_references():
    usage = TokenUsage(model="a", request_id="r1", call_id="c1", reported={"input": 10}, estimated={"output": 2})
    context = Context(Prompts(TextPrompt("user", "question"), TextPrompt("assistant", "answer"), usage))
    archive = context.to_dict()
    restored = Context.from_dict(copy.deepcopy(archive), {})
    assert restored.to_dict() == archive
    assert restored.usage.request_ids == ("r1",)
    assert restored.usage.sources["output"] == "estimated"
    assert list(restored.rounds.values())[0].usage.request_ids == ("r1",)


def test_old_context_archive_remains_readable_without_changing_source_data():
    context = Context(Prompts(TextPrompt("user", "question"), TextPrompt("assistant", "answer"), TokenUsage(10, 2, 0, 12, "old")))
    archive = context.to_dict()
    legacy = {"input_t": 10, "output_t": 2, "cache_hit_t": 0, "total_t": 12, "model": "old", "id": "old-usage"}
    archive["usage"] = legacy
    for round_data in archive["round"].values():
        round_data["usage"] = dict(legacy)

    original = copy.deepcopy(archive)
    restored = Context.from_dict(archive, {})
    assert archive == original
    assert restored.usage.input_t == 10
    assert restored.usage.sources["input"] == "unknown"
    assert restored.usage.request_ids == ()


def test_deepcopy_keeps_request_identity_and_isolates_evidence():
    reported = {"input": 10}
    usage = TokenUsage(model="a", request_id="r1", reported=reported)
    reported["input"] = 999
    duplicate = copy.deepcopy(usage)
    assert duplicate.id == usage.id
    assert duplicate.request_ids == usage.request_ids
    duplicate.samples[0].reported["input"] = 20
    assert usage.input_t == 10
    assert duplicate.input_t == 20


def test_subtracting_request_subset_preserves_remaining_provenance():
    first = TokenUsage(model="a", request_id="r1", reported={"input": 10, "output": 2})
    second = TokenUsage(model="b", request_id="r2", estimated={"input": 20, "output": 3})
    remainder = first + second - first.copy()
    assert remainder.input_t == 20
    assert remainder.request_ids == ("r2",)
    assert remainder.sources["input"] == "estimated"
    assert remainder.models == ("b",)


def test_unidentified_numeric_difference_does_not_claim_reported_usage():
    first = TokenUsage(model="a", request_id="r1", reported={"input": 10, "output": 2})
    difference = first - TokenUsage(1, 1, 0, 2, "a")
    assert difference.input_t == 9 and difference.total_t == 10
    assert difference.reported == {} and difference.request_ids == ()
    assert difference.sources["input"] == "unknown"


def test_partial_aggregate_does_not_treat_unavailable_cached_usage_as_reported_zero():
    a = TokenUsage(model="a", reported={"input": 10, "cached": 2})
    b = TokenUsage(model="a", reported={"input": 20})
    combined = a + b
    assert combined.cache_hit_t == 2
    assert "cached" in combined.incomplete_fields
    assert "cached" not in b.reported


@pytest.mark.parametrize("values", [{"input": -1}, {"input": True}, {"input": 1.5}, {"input": "10"}, {"images": float("nan")}])
def test_invalid_measurements_are_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        TokenUsage(reported=values)


def test_empty_clear_and_old_comparisons():
    usage = TokenUsage(10, 2, 3, 12, "m")
    assert usage > TokenUsage(1, 1, 0, 2, "other")
    assert TokenUsage(1, 1, 0, 2, "other") < usage
    assert not TokenUsage()
    assert TokenUsage(reported={"output_image": 1})
    usage.clear()
    assert not usage and usage.samples == []
    assert usage.model == "" and usage.request_ids == ()


def test_unknown_future_usage_archive_is_rejected():
    with pytest.raises(ValueError, match="版本"):
        TokenUsage.from_dict({"schema_version": 99})


def test_subtracting_empty_usage_preserves_request_evidence():
    usage = TokenUsage(model="a", request_id="r1", reported={"input": 10})
    difference = usage - TokenUsage()
    assert difference.input_t == 10
    assert difference.request_ids == ("r1",)
    assert difference.sources["input"] == "reported"


def test_round_usage_survives_next_input_and_protocol_normalization():
    from TIYA.llm.provider.common import normalize

    answer = TextPrompt("assistant", "answer", create_time=2.0)
    usage = TokenUsage(model="a", request_id="r1", reported={"input": 10, "output": 2})
    context = Context(Prompts(TextPrompt("user", "first", create_time=1.0), answer, usage))
    context.append(Prompts(TextPrompt("user", "next", create_time=3.0)))
    normalized = normalize(context)
    old_round = next(round for round in normalized.rounds.values() if any(prompt.id == answer.id for prompt in round))
    assert old_round.usage is not None
    assert old_round.usage.request_ids == ("r1",)
    assert normalized.usage.input_t == 10


def test_repaired_tool_round_preserves_its_usage_without_mutating_original():
    from TIYA.LLM_connect import ToolResponse
    from TIYA.llm.provider.common import normalize

    usage = TokenUsage(model="a", request_id="r1", reported={"input": 10, "output": 2})
    context = Context(Prompts(ToolResponse("tool", "missing-call", {"value": 1}), usage))
    original = context.to_dict()
    normalized = normalize(context)
    assert context.to_dict() == original
    assert list(normalized.rounds.values())[0].usage.request_ids == ("r1",)
    assert normalized.usage.request_ids == ("r1",)


def test_runtime_returns_ledger_evidence_and_preserves_each_committed_round(tmp_path, monkeypatch):
    import asyncio
    import httpx
    from itertools import count
    from types import SimpleNamespace

    from test_llm_runtime import runtime, preset
    from TIYA.llm import model
    from TIYA.llm.gateway import Gateway

    # Context 的轮次以 Prompt 时间为键；不要让零延迟模拟响应依赖系统时钟精度。
    timestamps = count(1)
    monkeypatch.setattr(model, "time", SimpleNamespace(time=lambda: float(next(timestamps))))

    def response(request):
        return httpx.Response(200, json={
            "id": "response-id",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}],
            "usage": {"input_tokens": 20, "output_tokens": 2},
        })

    async def run():
        rt = runtime(tmp_path)
        context = Context()
        identifiers = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            async def gateway(route):
                return Gateway(client)

            for question in ("first", "next"):
                before = context.to_dict()
                pending = await rt.execute(context, Prompts(TextPrompt("user", question)), preset(), gateway, 5)
                assert context.to_dict() == before
                result = rt.commit(context, pending)
                identifiers.append(pending.result.request_id)
                row = rt.ledger.get_request(pending.result.request_id)
                assert result.usage.samples[0].reported == row["reported"]
                assert result.usage.samples[0].estimated == row["estimated"]
                assert result.usage.call_ids == (pending.call_id,)
                assert result.usage.sources["input"] == "reported"
                assert result.usage.sources["cached"] == "estimated"

        assert context.usage.input_t == 40
        assert context.usage.total_t == 44
        assert context.usage.request_ids == tuple(identifiers)
        per_round = [identifier for round in context.rounds.values() if round.usage is not None for identifier in round.usage.request_ids]
        assert per_round == identifiers
        restored = Context.from_dict(context.to_dict(), {})
        assert restored.usage.request_ids == tuple(identifiers)
        assert restored.usage.input_t == 40

    asyncio.run(run())
