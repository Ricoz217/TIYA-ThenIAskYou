import asyncio
import base64
import copy
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import pytest

from TIYA.LLM_connect import (
    BasePrompt,
    Context,
    ContextEmptyError,
    ImagePrompt,
    Prompts,
    SystemPrompt,
    TextPrompt,
    TokenUsage,
    ToolCall,
    ToolInput,
    ToolResponse,
    build_tool_payloads,
)
from TIYA.LLM_usage import ApiPrice, LLMUsage, UsageStorage


def tool(value: Annotated[str, "the value"], count: int = 1) -> str:
    """Repeat a value.

    :param value: Value to repeat.
    :param count: Repeat count.
    :return: Repeated text.
    """
    return value * count


def test_complete_context_archive_preserves_ids_tools_reasoning_and_rounds(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"already cached")
    image = ImagePrompt("user", path, "image-name")
    image.image = ["cached-image"]
    context = Context(
        SystemPrompt("system"),
        ToolInput(tool),
        Prompts(TextPrompt("user", "hi"), image),
        Prompts(
            TextPrompt("assistant", "answer", "reasoning"),
            ToolCall("tool", "c", {"value": "v"}),
        ),
        ToolResponse("tool", "c", {"result": "v"}),
        TokenUsage(10, 2, 4, 12, "m"),
    )
    archive = context.to_dict()
    restored = Context.from_dict(copy.deepcopy(archive), {"tool": tool})
    assert restored.to_dict() == archive
    assert restored.tools[0].function is tool
    assert restored.system.text == "system"
    assert restored.has_content
    assert restored.usage.total_t == 12
    assert list(reversed(restored)) == list(reversed(list(restored)))
    assert "reasoning" not in str(
        restored
    )  # Reasoning is metadata, not visible user text.
    assert image == ImagePrompt.from_dict(image.to_dict())
    assert restored.discard_prompt_ids({image.id}) == 1
    assert restored.discard_prompt_ids(set()) == 0
    assert restored.discard_prompt_ids({image.id}) == 0
    restored.back2last_round()
    restored.clear(system=True, tools=True)
    assert restored.system is None and not restored.tools and not restored.has_content
    with pytest.raises(ContextEmptyError):
        restored.back2last_round()


def test_prompts_flatten_copy_duration_and_legacy_arithmetic():
    user, answer = TextPrompt("user", "question"), TextPrompt("assistant", "answer")
    usage = TokenUsage(10, 2, 3, 12, "m")
    prompts = Prompts([user, answer], usage, duration=1.5)
    assert prompts.role == "MIX"
    assert prompts.duration == 1.5
    assert user in prompts and "unrelated" not in prompts
    assert len(list(reversed(prompts))) == 2
    assert len(list(prompts - answer)) == 1
    assert len(list(prompts - Prompts(answer))) == 1
    assert (prompts - usage).usage.total_t == 0
    assert len(list(prompts + user)) == 3
    assert len(list(user + answer)) == 2
    assert Prompts(role="user", duration=0).duration == 0
    text = Prompts("hello", role="user")
    text.extend("world")
    assert text[1].text == "world"
    assert text.discard_prompt_ids({text[0].id}) == 1
    assert text.discard_prompt_ids(set()) == 0
    assert "world" in str(text)
    assert str(BasePrompt("user", "CUSTOM"))
    with pytest.raises(TypeError):
        Prompts(object())
    with pytest.raises(ValueError):
        Prompts("missing role")
    with pytest.raises(KeyError):
        prompts["x"]
    with pytest.raises(TypeError):
        prompts + object()
    with pytest.raises(TypeError):
        prompts - object()


def test_usage_model_arithmetic_and_price_copy():
    original = TokenUsage(10, 2, 3, 12, "m")
    restored = TokenUsage.from_dict(original.to_dict())
    assert restored == original
    assert original.copy().id != original.id
    assert (original + TokenUsage(1, 1, 0, 2, "m")).total_t == 14
    assert original > TokenUsage(1, 1, 0, 2, "m")
    assert original - TokenUsage(1, 1, 0, 2, "m")
    restored.clear()
    assert not restored
    price = ApiPrice(1, 2, 0, "USD")
    assert ApiPrice.from_dict(price.to_dict()) == price.copy()
    assert bool(price) and not ApiPrice()
    missing = UsageStorage("m", None)
    assert missing.copy().price is None
    priced = UsageStorage("m", price)
    assert UsageStorage.from_dict(priced.to_dict()).price == price
    priced.add(priced.copy())
    assert priced.total_t == {"verified": 0, "estimate": 0}


def test_legacy_usage_query_and_json_import_are_backed_by_sqlite(tmp_path):
    path = tmp_path / "usage.json"
    usage = LLMUsage(path)
    usage.update("m", {"input": 100, "cached": 50, "output": 10}, ApiPrice(1, 2, 0))
    usage.update("m", {"input": 20, "cached": 0, "output": 2}, verified=False)
    assert not path.exists()
    assert path.with_suffix(".sqlite3").exists()
    overview = LLMUsage(path).overview("ALL")
    assert overview["input"] == {"verified": 100, "estimate": 20}
    assert overview["cost"]["CNY"] == pytest.approx(0.00007)
    assert usage.query()
    now = __import__("time").time()
    assert usage.query(time_window=(now - 1, now + 1))
    assert usage.query(hash_key="not-present") == {}
    assert usage.to_dict()["storage"]
    usage.save()
    assert usage.load() is usage.ledger
    with pytest.raises(ValueError):
        usage.overview("INVALID")


class Choice(Enum):
    A = "a"
    B = "b"


def test_schema_keeps_enum_defaults_collections_and_optional_contract():
    def typed(
        choice: Choice = Choice.A,
        values: list[int] | None = None,
        mode: Literal["a", "b"] = "a",
        mapping: dict[str, float] | None = None,
    ) -> tuple[str, int]:
        return "ok", 1

    schema = build_tool_payloads(typed)["openai"]["function"]
    fields = schema["parameters"]["properties"]
    assert fields["choice"]["enum"] == ["a", "b"]
    assert fields["choice"]["default"] == "a"
    assert fields["mode"]["enum"] == ["a", "b"]
    assert schema["returns"]["type"] == "array"
    assert build_tool_payloads(tool)["anthropic"]["name"] == "tool"
    with pytest.raises(TypeError):
        build_tool_payloads(123)


def test_estimator_does_not_tokenize_opaque_replay_metadata_twice():
    from TIYA.llm.usage import estimate

    prompt = TextPrompt("assistant", "short answer")
    before = estimate(Context(prompt))["input"]
    prompt.native = {"responses_api": {"encrypted_content": "opaque" * 10000}}
    assert estimate(Context(prompt))["input"] == before
