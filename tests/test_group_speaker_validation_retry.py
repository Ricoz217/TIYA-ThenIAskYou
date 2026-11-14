from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import TIYA.agent.group_chat_agent as group_chat_agent_module
from TIYA.LLM_connect import (
    Chat,
    Context,
    ContextEmptyError,
    Prompts,
    SystemPrompt,
    TextPrompt,
)
from TIYA.agent.group_chat_agent import GroupChatSpeaker


class _Message:
    def __init__(self, msg_id: str):
        self.msg_id = msg_id
        self.done = True
        self.broken = False

    async def to_llm(self, wait: bool = False) -> dict:
        return {
            "message_id": self.msg_id,
            "user_id": "user-1",
            "message_sequence": ["text_1"],
            "message_content": {"text_1": {"text": "hello"}},
        }


class _History:
    def __init__(self, messages: list[_Message]):
        self.messages = messages

    def get_last_message(self, count: int) -> list[_Message]:
        return self.messages[-count:]

    def get_new_message_start_from(self, message_id: str) -> list[_Message]:
        return self.messages


class _Fav:
    @property
    def fav_list(self) -> dict:
        return {}

    def get_fav(self, title: str) -> str:
        return ""


class _LLM:
    def __init__(
            self,
            responses: list[Prompts | None],
            rollback_error: Exception | None = None
    ):
        self.responses = responses
        self.rollback_error = rollback_error
        self.prompts: list[str] = []
        self.rollback_count = 0

    async def ask(self, prompt, timeout: float) -> Prompts | None:
        self.prompts.append(str(prompt))
        return self.responses.pop(0)

    def back_to_last_round(self) -> Prompts:
        self.rollback_count += 1
        if self.rollback_error is not None:
            raise self.rollback_error

        return Prompts()


def _response(payload: dict) -> Prompts:
    return Prompts(
        TextPrompt("assistant", json.dumps(payload, ensure_ascii=False))
    )


def _invalid_nested_response(secret: str) -> Prompts:
    return _response({
        "message_sequence": ["text_1"],
        "message_content": {
            "text_1": {"text": secret},
            "to_agent": "",
            "intention": {},
        },
    })


def _valid_response() -> Prompts:
    return _response({
        "message_sequence": ["text_1"],
        "message_content": {"text_1": {"text": "corrected"}},
        "to_agent": "",
        "intention": {},
    })


async def _make_speaker(monkeypatch, tmp_path: Path, llm: _LLM) -> GroupChatSpeaker:
    monkeypatch.setattr(
        group_chat_agent_module,
        "SETTING_CFG",
        SimpleNamespace(
            LLM=SimpleNamespace(SpeakerRequestTimeout=1),
            Agent=SimpleNamespace(SpeakerInitInputMessages=50),
        ),
    )

    def missing_character(_: str):
        raise KeyError

    monkeypatch.setattr(group_chat_agent_module, "get_character", missing_character)
    history = _History([_Message("new-message")])
    host = SimpleNamespace(
        data_path=tmp_path,
        message_history=history,
        speaker_round_limit=10,
        member_alias={},
    )
    speaker = GroupChatSpeaker(host=host, character="测试角色")
    await speaker.LLM.close()
    speaker.LLM = llm
    speaker.FAV = _Fav()
    speaker._initiated = True
    speaker.last_message = "old-message"
    return speaker


def test_speaker_retries_with_sanitized_error_and_commits_cursor(monkeypatch, tmp_path: Path):
    async def run():
        secret = "INVALID_SPEECH_MUST_NOT_BE_ECHOED"
        llm = _LLM([_invalid_nested_response(secret), _valid_response()])
        speaker = await _make_speaker(monkeypatch, tmp_path, llm)

        result = await speaker.speak(speak_prompt="重新组织一次发言")

        assert result is not None
        assert llm.rollback_count == 1
        assert len(llm.prompts) == 2
        assert "错误类型" in llm.prompts[1]
        assert "错误位置" in llm.prompts[1]
        assert "错误原因" in llm.prompts[1]
        assert "JSON 根节点" in llm.prompts[1]
        assert "to_agent" in llm.prompts[1]
        assert secret not in llm.prompts[1]
        assert speaker.last_message == "new-message"
        assert speaker._run_rounds == 1

    asyncio.run(run())


def test_speaker_keeps_cursor_when_all_validation_attempts_fail(monkeypatch, tmp_path: Path):
    async def run():
        llm = _LLM([
            _invalid_nested_response("first invalid speech"),
            _invalid_nested_response("second invalid speech"),
        ])
        speaker = await _make_speaker(monkeypatch, tmp_path, llm)

        result = await speaker.speak()

        assert result is None
        assert llm.rollback_count == 2
        assert speaker.last_message == "old-message"
        assert speaker._run_rounds == 0

    asyncio.run(run())


def test_speaker_stops_retry_when_context_rollback_is_empty(monkeypatch, tmp_path: Path):
    async def run():
        llm = _LLM(
            [_invalid_nested_response("invalid speech"), _valid_response()],
            rollback_error=ContextEmptyError("empty context"),
        )
        speaker = await _make_speaker(monkeypatch, tmp_path, llm)

        result = await speaker.speak()

        assert result is None
        assert len(llm.prompts) == 1
        assert speaker.last_message == "old-message"
        assert speaker._run_rounds == 0
        assert speaker._force_refresh is True
        assert speaker.last_failure_retryable is False
        assert "ContextEmptyError" in speaker.last_failure_reason

    asyncio.run(run())


def test_speaker_does_not_rollback_none_response(monkeypatch, tmp_path: Path):
    async def run():
        llm = _LLM([None, _valid_response()])
        speaker = await _make_speaker(monkeypatch, tmp_path, llm)

        result = await speaker.speak()

        assert result is not None
        assert llm.rollback_count == 0
        assert len(llm.prompts) == 2
        assert "空响应" in llm.prompts[1]
        assert speaker.last_message == "new-message"
        assert speaker._run_rounds == 1

    asyncio.run(run())


def test_chat_back_to_last_round_returns_removed_prompts():
    async def run():
        context = Context(SystemPrompt("system"))
        context.append(TextPrompt("user", "request"))
        context.append(TextPrompt("assistant", "response"))
        chat = Chat(context=context, log_off=True)
        try:
            removed = chat.back_to_last_round()
        finally:
            await chat.close()

        assert isinstance(removed, Prompts)
        assert [prompt.role for prompt in removed] == ["user", "assistant"]
        assert [prompt.role for prompt in context.to_prompts()] == ["system"]

    asyncio.run(run())
