import copy
import threading

import pytest

from TIYA.LLM_connect import Context, Prompts, SystemPrompt, TextPrompt, TokenUsage, ToolInput
from TIYA.llm.config_model import ChatConfig, LLMConfig, RequestPolicy
from TIYA.LLM_usage import ApiPrice, UsageStorage


@pytest.mark.parametrize("clone", [copy.copy, lambda value: value.copy()])
def test_context_shallow_copy_has_independent_containers_and_shared_contents(clone):
    system = SystemPrompt("system")
    message = TextPrompt("user", "question")
    context = Context(system, message)
    duplicate = clone(context)
    assert duplicate.session_id != context.session_id
    assert duplicate._messages is not context._messages
    assert duplicate._rounds is not context._rounds
    assert duplicate._tools is not context._tools
    assert duplicate._system_prompt is system
    assert duplicate._messages[1] is message
    message.text = "shared"
    assert duplicate._messages[1].text == "shared"
    duplicate._messages.clear()
    assert len(context._messages) == 2


@pytest.mark.parametrize("clone", [copy.copy, lambda value: value.copy()])
def test_prompts_shallow_copy_shares_prompt_objects_and_preserves_duration(clone):
    original = Prompts(TextPrompt("user", "question"), duration=2)
    duplicate = clone(original)
    assert duplicate.duration == 2
    assert duplicate.prompts is not original.prompts
    assert duplicate[0] is original[0]
    duplicate[0].text = "shared"
    assert original[0].text == "shared"
    duplicate.prompts.clear()
    assert len(original.prompts) == 1


def test_deepcopy_isolates_context_and_preserves_internal_aliases_and_callable():
    class Owner:
        def __init__(self):
            self.lock = threading.Lock()

        def tool(self, value: str):
            return value

    owner = Owner()
    tool = ToolInput(owner.tool)
    original = Context(SystemPrompt("system"), TextPrompt("user", "question"), tool)
    duplicate = copy.deepcopy(original)
    assert duplicate.session_id == original.session_id
    assert duplicate._system_prompt is duplicate._messages[0]
    assert duplicate._system_prompt is not original._system_prompt
    assert duplicate._tools[0] is not tool
    assert duplicate._tools[0].function is tool.function
    for round in duplicate.rounds.values():
        for prompt in round:
            assert any(prompt is message for message in duplicate._messages)

    duplicate._messages[1].text = "isolated"
    assert original._messages[1].text == "question"


def test_prompts_deepcopy_isolates_prompt_and_nested_native_data():
    prompt = TextPrompt("assistant", "answer")
    prompt.native = {"protocol": {"items": ["native"]}}
    original = Prompts(prompt, duration=2)
    duplicate = copy.deepcopy(original)
    duplicate[0].native["protocol"]["items"].append("isolated")
    assert original[0].native["protocol"]["items"] == ["native"]
    assert duplicate.duration == 2


def test_system_prompt_copy_preserves_native_data_with_shallow_semantics():
    original = SystemPrompt("system")
    original.native = {"protocol": {"items": ["native"]}}
    shallow = original.copy()
    assert shallow.native == original.native
    assert shallow.native is not original.native
    assert shallow.native["protocol"] is original.native["protocol"]
    deep = copy.deepcopy(original)
    deep.native["protocol"]["items"].append("isolated")
    assert original.native["protocol"]["items"] == ["native"]


@pytest.mark.parametrize("clone", [copy.copy, lambda value: value.copy()])
def test_token_usage_shallow_copy_shares_samples_not_outer_collection(clone):
    original = TokenUsage(model="m", request_id="r1", reported={"input": 10})
    duplicate = clone(original)
    assert duplicate.id != original.id
    assert duplicate.request_ids == original.request_ids
    assert duplicate.samples is not original.samples
    assert duplicate.samples[0] is original.samples[0]
    duplicate.samples[0].reported["input"] = 20
    assert original.input_t == 20
    duplicate.samples.clear()
    assert original.input_t == 20


def test_config_shallow_copy_and_deepcopy_have_explicit_ownership():
    logger = object()
    candidate = LLMConfig("chat_completions", "a", "https://test/chat", "key", "m", model_params={"custom": {"nested": [1]}})
    original = ChatConfig([candidate], logger=logger, request_policy=RequestPolicy())
    shallow = copy.copy(original)
    assert shallow.candidates is not original.candidates
    assert shallow.primary is original.primary
    assert shallow.request_policy is original.request_policy
    assert shallow.logger is logger
    assert original.copy().primary is original.primary
    deep = copy.deepcopy(original)
    assert deep.logger is logger
    assert deep.primary is not original.primary
    assert deep.request_policy is not original.request_policy
    deep.primary.model_params["custom"]["nested"].append(2)
    assert original.primary.model_params["custom"]["nested"] == [1]


def test_partial_config_deepcopy_keeps_parameter_update_independent():
    original = ChatConfig(model_params={"custom": {"values": [1]}})
    duplicate = copy.deepcopy(original)
    duplicate.model_params["custom"]["values"].append(2)
    assert duplicate.candidates is None
    assert original.model_params == {"custom": {"values": [1]}}


def test_usage_storage_copy_shares_nested_counts_and_deepcopy_isolates_them():
    original = UsageStorage("m", ApiPrice(1, 2, 3))
    shallow = copy.copy(original)
    assert shallow.usage is not original.usage
    assert shallow.usage["input"] is original.usage["input"]
    assert shallow.price is original.price
    deep = copy.deepcopy(original)
    deep.usage["input"]["verified"] = 10
    deep.price.input = 5
    assert original.usage["input"]["verified"] == 0
    assert original.price.input == 1
