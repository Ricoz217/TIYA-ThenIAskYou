from __future__ import annotations

import asyncio
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from TIYA.agent.private_chat_agent import (
    PrivateAgentPostPayload,
    PrivateChatAgent,
    PrivateChatSpeaker,
    PrivatePostReason,
    PrivateSpeakTask,
    PrivateSpeakerPostPayload,
    PrivatePostTask,
    post_reason_prompt_maker,
)
from TIYA.agent.agent import BaseAgent
from TIYA.dialog import private_dialog as private_dialog_module
from TIYA.dialog.base_dialog import BaseDialog
from TIYA.dialog.private_dialog import PrivateMainDialog
from TIYA.LLM_connect import TextPrompt, build_tool_payloads
from TIYA.model.chat_notification import ChatNotification
from TIYA.model.message import MessageManager, PrivateMsg, TextMsg
from TIYA.model.persona import PrivatePersonaResult


def _private_message(msg_id: str, user_id: str, text: str) -> PrivateMsg:
    return PrivateMsg(
        msg_id=msg_id,
        user_id=user_id,
        username=user_id,
        nickname=user_id,
        content=[TextMsg(text=text)],
    )


class _DummyFav:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    @property
    def fav_list(self) -> dict[str, set[str]]:
        return {
            title: {hash_name}
            for title, hash_name in self.mapping.items()
        }

    def get_fav(self, title: str) -> str:
        return self.mapping.get(title, "")

    def __or__(self, other):
        return _DummyFav({**other.mapping, **self.mapping})


class _DummyLLM:
    def __init__(self, response) -> None:
        self.response = response

    async def ask(self, _prompt: str, timeout: float):
        return self.response


def test_base_dialog_owns_configurable_message_history() -> None:
    dialog = BaseDialog(max_message=37)

    assert isinstance(dialog.message_history, MessageManager)
    assert dialog.message_history._max_message == 37


def test_private_main_history_is_independent_but_aqueue_is_shared(
    monkeypatch,
    tmp_path,
) -> None:
    import TIYA.agent.private_chat_agent as private_agent_module
    import TIYA.model.persona as persona_module

    class _Agent:
        def __init__(self, **_kwargs) -> None:
            self.last_message = ""

    class _Speaker:
        def __init__(self, **kwargs) -> None:
            self.message_history = kwargs["host"].message_history
            self.last_message = ""

    class _Persona:
        def __init__(self, *_args) -> None:
            pass

    class _Logger:
        def error(self, *_args) -> None:
            pass

    class _Host:
        def __init__(self) -> None:
            self.user_id = "10001"
            self.username = "Alice"
            self.data_path = tmp_path
            self.message_history = MessageManager(max_message=99)
            self.aqueue = object()
            self.fav = _DummyFav({})
            self.logger = _Logger()
            self.private_config = SimpleNamespace(
                agent_round_limit=50,
                speaker_round_limit=30,
                default_character="抹布",
                chat_model="local",
                agent_model="local",
                speak_rate_min=0.3,
                speak_rate_max=1.0,
                attention_fade_out_time=300,
            )

    monkeypatch.setattr(private_agent_module, "PrivateChatAgent", _Agent)
    monkeypatch.setattr(private_agent_module, "PrivateChatSpeaker", _Speaker)
    monkeypatch.setattr(persona_module, "PrivatePersona", _Persona)
    monkeypatch.setattr(private_dialog_module, "PrivateChatAgent", _Agent)
    monkeypatch.setattr(private_dialog_module, "PrivateChatSpeaker", _Speaker)
    monkeypatch.setattr(private_dialog_module, "PrivatePersona", _Persona)
    monkeypatch.setattr(private_dialog_module, "get_character", lambda _name: object())
    monkeypatch.setattr(
        private_dialog_module,
        "SETTING_CFG",
        SimpleNamespace(
            PrivateChat=SimpleNamespace(
                AgentRequestTime=300,
                MaxMessageHistory=37,
            ),
        ),
    )

    host = _Host()
    dialog = PrivateMainDialog(host=host)

    assert dialog.message_history is not host.message_history
    assert dialog.message_history._max_message == 37
    assert dialog.SPEAKER.message_history is dialog.message_history
    assert dialog.aqueue is host.aqueue
    assert dialog.suspending
    assert [
        handle.__name__
        for handle in dialog._message_handle_sequence
    ] == [
        "_commit_new_message",
        "_agent_chat",
    ]


def test_private_main_commits_only_messages_it_accepts() -> None:
    async def run() -> None:
        class _Aqueue:
            def add_task(self, coro, **_kwargs) -> None:
                coro.close()

        host_history = MessageManager(max_message=10)
        main_history = MessageManager(max_message=10)
        msg = _private_message("user-1", "10001", "hello")
        await host_history.add_message(msg)

        dialog = object.__new__(PrivateMainDialog)
        dialog.host = SimpleNamespace(aqueue=_Aqueue())
        dialog.message_history = main_history
        dialog.NOTICE = ChatNotification()
        dialog._pause = False
        dialog._memory_summary_message_count = 0
        dialog._message_handle_sequence = [dialog._commit_new_message]

        assert await dialog.accept_message(msg) is False
        host_copy = await host_history.get_message("user-1")
        main_copy = await main_history.get_message("user-1")

        assert host_copy is msg
        assert main_copy is not None
        assert main_copy is not host_copy

    asyncio.run(run())

@pytest.mark.parametrize(
    ("reason", "instruction"),
    [
        (PrivatePostReason.USER_MESSAGE, "应当回应用户"),
        (PrivatePostReason.AGENT_INTERNAL, "自行判断是否发言"),
        (PrivatePostReason.INITIATE, "请勿发言"),
        (PrivatePostReason.SAVE_MEMORY, "请勿发言"),
    ],
)
def test_private_post_reason_always_uses_a_speaking_group(
    reason: PrivatePostReason,
    instruction: str,
) -> None:
    assert instruction in post_reason_prompt_maker(reason)


def test_private_post_task_matches_group_single_reason_queue() -> None:
    task = PrivatePostTask(PrivatePostReason.USER_MESSAGE, timeout=30)

    assert task.reason is PrivatePostReason.USER_MESSAGE
    assert not task.expired.is_set()


def test_private_speak_tool_is_configured_as_background_task(monkeypatch, tmp_path) -> None:
    def base_init(self, **_kwargs) -> None:
        self.functions_default_setting = {}

    monkeypatch.setattr(BaseAgent, "__init__", base_init)

    class _Host:
        user_id = "10001"
        data_path = tmp_path

    agent = PrivateChatAgent(host=_Host())

    assert agent.functions_default_setting["speak"] == {
        "wait": False,
        "callback": "EXCEPTION",
    }


def test_private_payloads_anchor_bot_and_keep_lunar_time(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        monkeypatch.setattr(private_agent_module, "get_bot_uid", lambda: "bot-100")
        monkeypatch.setattr(private_agent_module, "get_bot_name", lambda: "抹布")
        monkeypatch.setattr(private_agent_module, "_format_lunar", lambda _now: "农历测试 节气测试")

        agent_prompt = await PrivateAgentPostPayload(
            reason=PrivatePostReason.USER_MESSAGE,
            messages=[],
        ).get_prompt()
        speaker_prompt = await PrivateSpeakerPostPayload(messages=[]).get_prompt()

        for prompt in (agent_prompt, speaker_prompt):
            assert "bot-100" in prompt
            assert "抹布" in prompt
            assert "农历测试 节气测试" in prompt
            assert "# 私聊对象" not in prompt

    asyncio.run(run())


def test_private_agent_payload_includes_internal_notification(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        monkeypatch.setattr(private_agent_module, "get_bot_uid", lambda: "bot-100")
        monkeypatch.setattr(private_agent_module, "get_bot_name", lambda: "抹布")
        monkeypatch.setattr(private_agent_module, "_format_lunar", lambda _now: "农历测试")

        prompt = await PrivateAgentPostPayload(
            reason=PrivatePostReason.USER_MESSAGE,
            messages=[],
            notice="## 记忆通知\n\n1. 你需要保存一条有效记忆",
        ).get_prompt()
        empty_prompt = await PrivateAgentPostPayload(
            reason=PrivatePostReason.USER_MESSAGE,
            messages=[],
        ).get_prompt()

        assert "# 系统通知" in prompt
        assert "你需要保存一条有效记忆" in prompt
        assert "**无新通知**" in empty_prompt

    asyncio.run(run())


def test_private_agent_payload_constructor_reads_notice() -> None:
    async def run() -> None:
        notice = ChatNotification()
        notice.add_notice(
            content="主控需要积极传递信息给 Speaker",
            title="发言规范",
            alive_until_messages=3,
        )
        agent = object.__new__(PrivateChatAgent)
        agent.last_message = "cursor"
        agent._host = SimpleNamespace(
            message_history=MessageManager(max_message=10),
            NOTICE=notice,
            participant_aliases={},
        )
        agent.said_intention = {}

        payload = await agent._payload_constructor(
            PrivatePostTask(PrivatePostReason.USER_MESSAGE, timeout=30),
        )

        assert "主控需要积极传递信息给 Speaker" in payload.notice

    asyncio.run(run())


def test_private_payloads_attach_aliases_by_message_sender(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        monkeypatch.setattr(private_agent_module, "get_bot_uid", lambda: "bot-100")
        monkeypatch.setattr(private_agent_module, "get_bot_name", lambda: "抹布")
        monkeypatch.setattr(private_agent_module, "_format_lunar", lambda _now: "农历测试")
        monkeypatch.setattr(
            private_agent_module,
            "SETTING_CFG",
            SimpleNamespace(Groups=SimpleNamespace(MessageParseTimeout=15)),
        )
        messages = [
            _private_message("user-1", "10001", "你好"),
            _private_message("bot-1", "bot-100", "你好呀"),
        ]
        aliases = {
            "10001": ["宝"],
            "bot-100": ["抹布"],
        }

        agent_prompt = await PrivateAgentPostPayload(
            reason=PrivatePostReason.USER_MESSAGE,
            messages=messages,
            participant_aliases=aliases,
        ).get_prompt()
        speaker_prompt = await PrivateSpeakerPostPayload(
            messages=messages,
            participant_aliases=aliases,
        ).get_prompt()

        for prompt in (agent_prompt, speaker_prompt):
            assert '"user_aliases": [\n      "宝"\n    ]' in prompt
            assert '"user_aliases": [\n      "抹布"\n    ]' in prompt

    asyncio.run(run())


def test_private_main_dialog_keeps_separate_user_and_bot_aliases(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.dialog.private_dialog as private_dialog

        monkeypatch.setattr(private_dialog, "get_bot_uid", lambda: "bot-100", raising=False)
        dialog = object.__new__(PrivateMainDialog)
        dialog.host = SimpleNamespace(user_id="10001", username="Alice")
        dialog._persona_lock = asyncio.Lock()
        dialog._persona_cache = PrivatePersonaResult.empty(
            user_id="10001",
            user_name="Alice",
            bot_id="bot-100",
            bot_name="抹布",
        )
        memories: list[str] = []

        async def add_memory(memory: str):
            memories.append(memory)
            return SimpleNamespace(success=True, message="")

        dialog.PERSONA = SimpleNamespace(add_memory=add_memory)

        await dialog.set_alias("10001", "宝")
        await dialog.set_alias("bot-100", "抹布老师")
        await dialog.set_alias("10001", "宝")

        assert dialog._persona_cache.user.alias == ["宝"]
        assert dialog._persona_cache.bot.alias == ["抹布老师"]
        assert dialog.participant_aliases == {
            "10001": ["宝"],
            "bot-100": ["抹布老师"],
        }
        assert memories == [
            "用户[10001]在当前私聊关系中的爱称新增: 宝",
            "BOT[bot-100]在当前私聊关系中的爱称新增: 抹布老师",
        ]

        with pytest.raises(KeyError):
            await dialog.set_alias("other-user", "陌生人")

    asyncio.run(run())


def test_private_alias_cache_is_unchanged_when_memory_write_fails(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.dialog.private_dialog as private_dialog

        monkeypatch.setattr(private_dialog, "get_bot_uid", lambda: "bot-100", raising=False)
        dialog = object.__new__(PrivateMainDialog)
        dialog.host = SimpleNamespace(user_id="10001", username="Alice")
        dialog._persona_lock = asyncio.Lock()
        dialog._persona_cache = PrivatePersonaResult.empty(
            user_id="10001",
            user_name="Alice",
            bot_id="bot-100",
            bot_name="抹布",
        )

        async def add_memory(_memory: str):
            return SimpleNamespace(success=False, message="failed")

        dialog.PERSONA = SimpleNamespace(add_memory=add_memory)

        with pytest.raises(RuntimeError, match="failed"):
            await dialog.set_alias("10001", "宝")

        assert dialog._persona_cache.user.alias == []

    asyncio.run(run())


def test_private_speaker_payload_omits_empty_optional_sections() -> None:
    async def run() -> None:
        prompt = await PrivateSpeakerPostPayload(messages=[]).get_prompt()

        assert "# 相关信息" not in prompt
        assert "# 相关记忆" not in prompt
        assert "# 主控正在做的事" not in prompt

    asyncio.run(run())


def test_fallback_only_observes_whether_speak_was_requested() -> None:
    async def run() -> None:
        agent = object.__new__(PrivateChatAgent)
        agent._initiated = True
        agent._speak_request_generation = 4
        calls: list[dict] = []

        async def add_task(**kwargs) -> None:
            calls.append(kwargs)

        agent.add_task = add_task
        await agent._observe_force_speak_fallback(4, 0)

        assert len(calls) == 1
        assert calls[0]["function_name"] == "speak"
        assert calls[0]["callback"] == "EXCEPTION"
        assert agent._speak_request_generation == 5

    asyncio.run(run())


def test_old_fallback_observer_is_invalidated_by_any_speak_request() -> None:
    async def run() -> None:
        agent = object.__new__(PrivateChatAgent)
        agent._initiated = True
        agent._speak_request_generation = 5
        calls: list[str] = []

        async def add_task(**_kwargs) -> None:
            calls.append("speak")

        agent.add_task = add_task
        await agent._observe_force_speak_fallback(4, 0)

        assert calls == []

    asyncio.run(run())


def test_private_said_review_uses_internal_run_without_fallback(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        monkeypatch.setattr(private_agent_module, "get_bot_uid", lambda: "bot-100")
        monkeypatch.setattr(
            private_agent_module,
            "SETTING_CFG",
            SimpleNamespace(
                Agent=SimpleNamespace(SaidReviewPeriod=2),
                Groups=SimpleNamespace(MessageParseTimeout=15),
            ),
        )
        history = MessageManager(max_message=10)
        await history.add_message(_private_message("bot-1", "bot-100", "第一句"))
        await history.add_message(_private_message("bot-2", "bot-100", "第二句"))
        run_calls: list[tuple[TextPrompt, bool]] = []

        async def agent_run(prompt, from_user=True):
            run_calls.append((prompt, from_user))
            return None

        class _Aqueue:
            def __init__(self) -> None:
                self.task: asyncio.Task | None = None

            def add_task(self, coro, timeout):
                assert timeout == -1
                self.task = asyncio.create_task(coro)

        aqueue = _Aqueue()
        agent = object.__new__(PrivateChatAgent)
        agent.said_review_count = 2
        agent.said_intention = {
            "bot-1": {"intent": "react", "effect": "neutral"},
            "bot-2": {"intent": "answer", "effect": "neutral"},
        }
        agent._said_review_lock = asyncio.Lock()
        agent._host = SimpleNamespace(
            message_history=history,
            participant_aliases={"bot-100": ["抹布"]},
            aqueue=aqueue,
        )
        agent.run = agent_run

        await agent._auto_said_review()
        assert aqueue.task is not None
        await aqueue.task

        assert agent.said_review_count == 0
        assert agent.said_intention == {}
        assert len(run_calls) == 1
        prompt, from_user = run_calls[0]
        assert from_user is False
        assert "BOT 发言结果审阅任务" in prompt.text
        assert "本次发言意图" in prompt.text
        assert "用户意图" not in prompt.text
        assert "正确判断" not in prompt.text

    asyncio.run(run())


def test_private_speaker_merges_calls_arriving_while_busy() -> None:
    async def run() -> None:
        speaker = object.__new__(PrivateChatSpeaker)
        speaker._request_lock = asyncio.Lock()
        speaker._running = False
        speaker._speak_queue = deque()
        speaker.last_failure_reason = ""
        speaker.last_failure_retryable = False
        speaker.logger = SimpleNamespace(
            error=lambda *_args, **_kwargs: None,
            debug=lambda *_args, **_kwargs: None,
        )
        started = asyncio.Event()
        release = asyncio.Event()
        payloads: list[PrivateSpeakerPostPayload] = []

        def build_payload(**kwargs) -> PrivateSpeakerPostPayload:
            payload = PrivateSpeakerPostPayload(messages=[], **kwargs)
            payloads.append(payload)
            return payload

        result = {
            "message_sequence": ["text_1"],
            "message_content": {"text_1": {"text": "hello"}},
            "to_agent": "",
            "intention": {
                "target_message_id": "user-1",
                "intent": "answer",
                "effect": "neutral",
            },
        }

        async def request_payload(_payload):
            if len(payloads) == 1:
                started.set()
                await release.wait()
            return result

        speaker._build_payload = build_payload
        speaker._request_payload = request_payload

        first = PrivateSpeakTask(call_id="call-1", speak_prompt="第一件事")
        second = PrivateSpeakTask(
            call_id="call-2",
            speak_prompt="第二件事",
            working="处理中",
            relative_information={"新闻": "A"},
        )
        third = PrivateSpeakTask(
            call_id="call-3",
            speak_prompt="第三件事",
            working="处理完成",
            relative_information={"新闻": "B", "天气": "晴"},
            relative_memory={"约定": "稍后回复"},
        )

        await speaker.speak(first)
        await started.wait()
        await speaker.speak(second)
        await speaker.speak(third)
        release.set()

        await first.event.wait()
        assert first.result == result
        assert not first.merged_to
        first.delivered.set()

        await second.event.wait()
        assert second.result is None
        assert second.merged_to == "call-3"
        second.delivered.set()

        await third.event.wait()
        assert third.result == result
        third.delivered.set()
        await _wait_until(lambda: not speaker.running)

        assert len(payloads) == 2
        assert payloads[0].speak_prompt == "第一件事"
        assert payloads[1].speak_prompt == "第二件事\n\n第三件事"
        assert payloads[1].working == "处理完成"
        assert payloads[1].relative_information == {"新闻": "B", "天气": "晴"}
        assert payloads[1].relative_memory == {"约定": "稍后回复"}

    async def _wait_until(predicate) -> None:
        for _ in range(20):
            if predicate():
                return
            await asyncio.sleep(0)
        raise AssertionError("condition did not become true")

    asyncio.run(run())


def test_private_speaker_releases_busy_state_after_unexpected_failure() -> None:
    async def run() -> None:
        speaker = object.__new__(PrivateChatSpeaker)
        speaker._request_lock = asyncio.Lock()
        speaker._running = False
        speaker._speak_queue = deque()
        speaker.last_failure_reason = ""
        speaker.last_failure_retryable = False
        speaker.logger = SimpleNamespace(error=lambda *_args, **_kwargs: None, debug=lambda *_args, **_kwargs: None)
        speaker._build_payload = lambda **kwargs: PrivateSpeakerPostPayload(
            messages=[],
            **kwargs,
        )

        async def fail(_payload):
            raise RuntimeError("speaker crashed")

        speaker._request_payload = fail
        task = PrivateSpeakTask(call_id="call-1")
        await speaker.speak(task)
        await task.event.wait()

        assert isinstance(task.exception, RuntimeError)
        assert "speaker crashed" in str(task.exception)
        task.delivered.set()
        await _wait_until(lambda: not speaker.running)
        assert not speaker.running

    async def _wait_until(predicate) -> None:
        for _ in range(20):
            if predicate():
                return
            await asyncio.sleep(0)
        raise AssertionError("condition did not become true")

    asyncio.run(run())


def test_private_speaker_payload_includes_private_and_character_favs(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        monkeypatch.setattr(
            private_agent_module,
            "get_character",
            lambda _: SimpleNamespace(fav=_DummyFav({"角色表情": "character-hash"})),
        )
        monkeypatch.setattr(
            private_agent_module,
            "SETTING_CFG",
            SimpleNamespace(Agent=SimpleNamespace(SpeakerInitInputMessages=30)),
        )
        speaker = object.__new__(PrivateChatSpeaker)
        speaker.last_message = ""
        speaker.HISTORY = MessageManager(max_message=10)
        speaker.character = "测试角色"
        speaker.FAV = _DummyFav({"私聊表情": "private-hash"})
        speaker._host = SimpleNamespace(participant_aliases={})

        payload = speaker._build_payload()

        assert payload.fav_list == ["私聊表情", "角色表情"]

    asyncio.run(run())


def test_private_speaker_resolves_private_fav_before_character_fav(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        monkeypatch.setattr(
            private_agent_module,
            "get_character",
            lambda _: SimpleNamespace(fav=_DummyFav({"同名表情": "character-hash"})),
        )
        monkeypatch.setattr(
            private_agent_module,
            "SETTING_CFG",
            SimpleNamespace(LLM=SimpleNamespace(SpeakerRequestTimeout=12)),
        )
        speaker = object.__new__(PrivateChatSpeaker)
        speaker._initiated = True
        speaker.last_failure_reason = ""
        speaker.last_failure_retryable = False
        speaker._force_refresh = False
        speaker._force_clear = False
        speaker._run_rounds = 0
        speaker._host = SimpleNamespace(speaker_round_limit=10)
        speaker.character = "测试角色"
        speaker._allow_key_type = {"reply", "break", "text", "image", "fav"}
        speaker.FAV = _DummyFav({"同名表情": "private-hash"})
        speaker.LLM = _DummyLLM([
            TextPrompt(
                "assistant",
                """{
  "message_sequence": ["fav_1"],
  "message_content": {
    "fav_1": {
      "fav_title": "同名表情"
    }
  },
  "to_agent": "",
  "intention": {
    "target_message_id": "user-1",
    "intent": "react",
    "effect": "neutral"
  }
}""",
            )
        ])

        result = await speaker._speak_payload(PrivateSpeakerPostPayload(messages=[]))

        assert result is not None
        assert result["message_content"]["fav_1"] == {"hash_name": "private-hash"}
        assert result["intention"]["intent"] == "react"

    asyncio.run(run())


def test_private_speaker_rejects_response_without_intention(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        monkeypatch.setattr(
            private_agent_module,
            "get_character",
            lambda _: SimpleNamespace(fav=_DummyFav({})),
        )
        monkeypatch.setattr(
            private_agent_module,
            "SETTING_CFG",
            SimpleNamespace(LLM=SimpleNamespace(SpeakerRequestTimeout=12)),
        )
        speaker = object.__new__(PrivateChatSpeaker)
        speaker._initiated = True
        speaker.last_failure_reason = ""
        speaker.last_failure_retryable = False
        speaker._force_refresh = False
        speaker._force_clear = False
        speaker._run_rounds = 0
        speaker._host = SimpleNamespace(speaker_round_limit=10)
        speaker.character = "测试角色"
        speaker._allow_key_type = {"reply", "break", "text", "image", "fav"}
        speaker.FAV = _DummyFav({})
        speaker.LLM = _DummyLLM([
            TextPrompt(
                "assistant",
                json.dumps({
                    "message_sequence": ["text_1"],
                    "message_content": {"text_1": {"text": "hello"}},
                    "to_agent": "",
                }, ensure_ascii=False),
            )
        ])

        result = await speaker._speak_payload(PrivateSpeakerPostPayload(messages=[]))

        assert result is None
        assert "字段不完整" in speaker.last_failure_reason

    asyncio.run(run())


@pytest.mark.parametrize(
    "response_update",
    [
        {"intention": "not-an-object"},
        {
            "intention": {
                "target_message_id": "user-1",
                "intent": "answer",
                "effect": "neutral",
            },
            "unexpected": True,
        },
    ],
)
def test_private_speaker_rejects_invalid_intention_schema(
    monkeypatch,
    response_update,
) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        monkeypatch.setattr(
            private_agent_module,
            "get_character",
            lambda _: SimpleNamespace(fav=_DummyFav({})),
        )
        monkeypatch.setattr(
            private_agent_module,
            "SETTING_CFG",
            SimpleNamespace(LLM=SimpleNamespace(SpeakerRequestTimeout=12)),
        )
        response = {
            "message_sequence": ["text_1"],
            "message_content": {"text_1": {"text": "hello"}},
            "to_agent": "",
        }
        response.update(response_update)
        speaker = object.__new__(PrivateChatSpeaker)
        speaker._initiated = True
        speaker.last_failure_reason = ""
        speaker.last_failure_retryable = False
        speaker._force_refresh = False
        speaker._force_clear = False
        speaker._run_rounds = 0
        speaker._host = SimpleNamespace(speaker_round_limit=10)
        speaker.character = "测试角色"
        speaker._allow_key_type = {"reply", "break", "text", "image", "fav"}
        speaker.FAV = _DummyFav({})
        speaker.LLM = _DummyLLM(
            [TextPrompt("assistant", json.dumps(response, ensure_ascii=False))]
        )

        assert await speaker._speak_payload(
            PrivateSpeakerPostPayload(messages=[])
        ) is None

    asyncio.run(run())


def test_private_agent_get_fav_list_returns_private_and_character_favs(monkeypatch) -> None:
    import TIYA.agent.private_chat_agent as private_agent_module

    monkeypatch.setattr(
        private_agent_module,
        "get_character",
        lambda _: SimpleNamespace(fav=_DummyFav({"角色表情": "character-hash"})),
    )
    agent = object.__new__(PrivateChatAgent)
    agent._host = SimpleNamespace(
        SPEAKER=SimpleNamespace(character="测试角色"),
        FAV=_DummyFav({"私聊表情": "private-hash"}),
    )

    assert agent.get_fav_list() == {
        "private_fav": ["私聊表情"],
        "character_fav": ["角色表情"],
    }


def test_private_agent_uses_single_bucket_memory_tools() -> None:
    async def run() -> None:
        class _Host:
            def __init__(self) -> None:
                self.added: list[str] = []

            async def add_memory(self, memory: str) -> dict:
                self.added.append(memory)
                return {"success": True, "memory": memory, "message": ""}

            async def query_memory(self, query: str) -> dict:
                return {"success": True, "answer": query, "matches": []}

        agent = object.__new__(PrivateChatAgent)
        agent._host = _Host()
        agent._memory_lock = asyncio.Lock()
        agent._last_add_memory_time = 0.0

        added = await agent.add_memory("双方正在重构私聊画像")
        queried = await agent.query_memory("我们在做什么")

        assert added["success"]
        assert agent._host.added == ["双方正在重构私聊画像"]
        assert queried["answer"] == "我们在做什么"

    asyncio.run(run())


def test_private_agent_set_alias_delegates_to_main_dialog() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []

        async def set_alias(user_id: str, alias: str) -> None:
            calls.append((user_id, alias))

        agent = object.__new__(PrivateChatAgent)
        agent._host = SimpleNamespace(set_alias=set_alias)

        await agent.set_alias("10001", "宝")

        assert calls == [("10001", "宝")]

    asyncio.run(run())


def test_private_agent_tool_schemas_include_real_descriptions() -> None:
    agent = object.__new__(PrivateChatAgent)
    tools = (
        "speak",
        "get_single_message",
        "get_time_interval_messages",
        "get_image_detail",
        "get_news_by_date",
        "fetch_news",
        "add_memory",
        "query_memory",
        "get_persona",
        "set_alias",
        "get_fav_list",
        "add_private_fav",
    )

    for name in tools:
        schema = build_tool_payloads(getattr(agent, name))["openai"]["function"]
        assert not schema["description"].startswith("Tool generated from")
        for parameter in schema["parameters"]["properties"].values():
            assert parameter.get("description")


def test_private_agent_add_private_fav_returns_private_result(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        image_path = tmp_path / "fav.png"
        Image.new("RGB", (16, 16), "red").save(image_path)
        monkeypatch.setattr(
            private_agent_module,
            "get_file_path_async",
            AsyncMock(return_value=image_path),
        )

        class _Fav:
            async def add_fav(self, hash_name: str, fav_title: str = "") -> str:
                assert hash_name == "hash-name"
                assert fav_title == "标题"
                return "标题"

        agent = object.__new__(PrivateChatAgent)
        agent._host = SimpleNamespace(FAV=_Fav())

        result = await agent.add_private_fav("hash-name", "标题")

        assert result == {
            "success": True,
            "hash_name": "hash-name",
            "fav_title": "标题",
            "fav_type": "private",
            "message": "成功添加私聊表情",
        }

    asyncio.run(run())


def test_private_agent_add_private_fav_rejects_truncated_image(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        valid_image = tmp_path / "valid.png"
        truncated_image = tmp_path / "truncated.png"
        Image.new("RGB", (16, 16), "red").save(valid_image)
        truncated_image.write_bytes(valid_image.read_bytes()[:60])
        monkeypatch.setattr(
            private_agent_module,
            "get_file_path_async",
            AsyncMock(return_value=truncated_image),
        )

        agent = object.__new__(PrivateChatAgent)
        agent._host = SimpleNamespace(
            FAV=SimpleNamespace(
                add_fav=lambda *args, **kwargs: pytest.fail(
                    "损坏图片不应进入私聊表情写入流程"
                )
            )
        )

        with pytest.raises(OSError):
            await agent.add_private_fav("hash-name", "标题")

    asyncio.run(run())


def test_private_memory_summary_keeps_fixed_nonempty_subjects() -> None:
    content = {
        "USER": {
            "event": {"项目": "用户正在重构私聊 Agent"},
            "occupation": "程序员",
        },
        "BOT": {},
        "RELATIONSHIP": {"event": {"重构": "双方正在重构私聊 Agent"}},
        "UNKNOWN": {"free": "不应保存"},
    }

    result = PrivateMainDialog._normalize_memory_summary(content)

    assert result == {
        "USER": {
            "event": {"项目": "用户正在重构私聊 Agent"},
            "occupation": "程序员",
        },
        "RELATIONSHIP": {"event": {"重构": "双方正在重构私聊 Agent"}},
    }


def test_private_memory_summary_rejects_missing_required_events() -> None:
    assert PrivateMainDialog._normalize_memory_summary({}) == {}
    assert PrivateMainDialog._normalize_memory_summary({
        "RELATIONSHIP": {"interaction_style": "长期协作"},
    }) == {}
    assert PrivateMainDialog._normalize_memory_summary({
        "USER": {"occupation": "程序员"},
        "RELATIONSHIP": {"event": {"协作": "双方讨论了实现"}},
    }) == {
        "RELATIONSHIP": {"event": {"协作": "双方讨论了实现"}},
    }


def test_private_dialog_restores_saved_message_cursors(tmp_path) -> None:
    async def run() -> None:
        saved_history = MessageManager(max_message=10)
        await saved_history.add_message(
            _private_message("saved-user-message", "10001", "saved")
        )
        payload = {
            "version": "0.1.0",
            "data": {
                "messages": saved_history.to_dict(),
                "news": {"26-07-04": {"topic": ["item"]}},
                "agent_last_message": "old-user-message",
                "speaker_last_message": "old-bot-message",
                "persona_cache": {
                    "user": {"alias": ["阿A"]},
                    "bot": {"style": "先分析再行动"},
                    "relationship": {"relationship_type": "长期协作"},
                },
                "last_persona_time": 123.0,
                "memory_summary_message_count": 9,
                "clear_private_context_epoch_seen": 12.0,
                "notification": {
                    "notices": {},
                    "recordings": {},
                    "messages_now": 2,
                    "self_messages_now": 1,
                },
            },
        }
        (tmp_path / "private_main_dialog_persist.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        dialog = object.__new__(PrivateMainDialog)
        dialog.host = SimpleNamespace(
            data_path=tmp_path,
            user_id="10001",
            username="Alice",
        )
        dialog.message_history = MessageManager(max_message=10)
        dialog.relative_news = {}
        dialog.AGENT = SimpleNamespace(last_message="new-message-must-not-win")
        dialog.SPEAKER = SimpleNamespace(last_message="")
        dialog.NOTICE = ChatNotification()
        dialog._persona_cache = None
        dialog._last_persona_time = 0.0
        dialog._memory_summary_message_count = 0
        dialog._clear_private_context_epoch_seen = 0.0

        await dialog.load()

        assert dialog.AGENT.last_message == "old-user-message"
        assert dialog.SPEAKER.last_message == "old-bot-message"
        assert dialog._persona_cache.user.participant_id == "10001"
        assert dialog._persona_cache.user.alias == ["阿A"]
        assert dialog._persona_cache.relationship.relationship_type == "长期协作"
        assert dialog._memory_summary_message_count == 9
        assert dialog._clear_private_context_epoch_seen == 12.0
        assert await dialog.message_history.get_message("saved-user-message") is not None

    asyncio.run(run())


def test_private_main_dialog_start_clears_persisted_notice_and_recreates_memory_tip(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        monkeypatch.setattr(private_dialog_module, "PRIVATE_CHATS_DIR", tmp_path)

        class _Agent:
            async def start(self) -> None:
                return None

            async def shutdown(self) -> None:
                return None

        class _Speaker:
            async def initiate(self) -> None:
                return None

            async def shutdown(self) -> None:
                return None

        dialog = object.__new__(PrivateMainDialog)
        dialog._initiated = False
        dialog._pause = True
        dialog._auto_save_loop = None
        dialog.host = SimpleNamespace(private_config=SimpleNamespace(chat=False))
        dialog.NOTICE = ChatNotification()
        dialog.NOTICE.add_notice(
            content="这条持久化通知启动时应该被清掉",
            title="旧通知",
            alive_until_messages=3,
        )
        dialog.AGENT = _Agent()
        dialog.SPEAKER = _Speaker()
        dialog.load = lambda: _load(dialog)
        dialog.save = lambda: _noop()
        dialog._initiate_main_agent = lambda: None

        await dialog.start()
        try:
            text = dialog.NOTICE.notice2text()
            assert "这条持久化通知启动时应该被清掉" not in text
            assert dialog._memory_tip_notice
            assert dialog.NOTICE.get_notice(dialog._memory_tip_notice) is not None
            assert dialog.memory_save_notice

        finally:
            await dialog.shutdown()

    async def _load(dialog) -> None:
        dialog.NOTICE.add_notice(
            content="这条持久化通知启动时应该被清掉",
            title="旧通知",
            alive_until_messages=3,
            exist_ok=True,
        )

    async def _noop() -> None:
        return None

    asyncio.run(run())


def test_private_main_dialog_start_marks_stale_global_clear_epoch(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        class _Agent:
            def __init__(self) -> None:
                self.cleared = 0

            async def start(self) -> None:
                return None

            async def shutdown(self) -> None:
                return None

            def clear_history(self) -> None:
                self.cleared += 1

        class _Speaker:
            def __init__(self) -> None:
                self.cleared = 0

            async def initiate(self) -> None:
                return None

            async def shutdown(self) -> None:
                return None

            def clear_history(self) -> None:
                self.cleared += 1

        monkeypatch.setattr(private_dialog_module, "PRIVATE_CHATS_DIR", tmp_path)
        (tmp_path / "context_epoch.json").write_text(
            json.dumps({"clear_private_context_epoch": 25.0}),
            encoding="utf-8",
        )
        dialog = object.__new__(PrivateMainDialog)
        dialog._initiated = False
        dialog._pause = True
        dialog._auto_save_loop = None
        dialog.host = SimpleNamespace(private_config=SimpleNamespace(chat=False))
        dialog.NOTICE = ChatNotification()
        dialog.AGENT = _Agent()
        dialog.SPEAKER = _Speaker()
        dialog.load = lambda: _load(dialog)
        dialog.save = Mock(side_effect=_noop)
        dialog._initiate_main_agent = lambda: None

        await dialog.start()
        try:
            assert dialog.AGENT.cleared == 1
            assert dialog.SPEAKER.cleared == 1
            assert dialog._clear_private_context_epoch_seen == 25.0
            assert dialog.save.called

        finally:
            await dialog.shutdown()

    async def _load(dialog) -> None:
        dialog._clear_private_context_epoch_seen = 10.0

    async def _noop() -> None:
        return None

    asyncio.run(run())


def test_private_speaker_does_not_block_private_chat_queue(monkeypatch) -> None:
    async def run() -> None:
        class _Speaker:
            async def speak(self, task: PrivateSpeakTask):
                await asyncio.sleep(0)
                task.result = {
                    "message_sequence": ["text_1"],
                    "message_content": {"text_1": {"text": "hello"}},
                    "to_agent": "",
                    "intention": {
                        "target_message_id": "user-1",
                        "intent": "answer",
                        "effect": "neutral",
                    },
                }
                task.event.set()

            def mark_failure(self, *_args, **_kwargs) -> None:
                raise AssertionError("speaker should not fail")

        class _Host:
            def __init__(self) -> None:
                self.sent = []
                self.aqueue = self._Aqueue()

            class _Task:
                def __init__(self, coro) -> None:
                    self.coro = coro
                    self.result = []
                    self.exception = []
                    self.cancelled = False

                async def wait(self) -> None:
                    self.result.append(await self.coro)

            class _Aqueue:
                def add_task(self, coro, timeout):
                    return _Host._Task(coro)

            async def say(self, message):
                self.sent.append(message)
                return [_private_message("bot-1", "bot", "hello")]

        dialog = object.__new__(PrivateMainDialog)
        dialog.message_history = MessageManager(max_message=10)
        dialog.SPEAKER = _Speaker()
        dialog.host = _Host()
        dialog.NOTICE = ChatNotification()
        dialog._last_bot_said = deque(maxlen=20)
        dialog._paraphrase = 0
        dialog.AGENT = SimpleNamespace(said_intention={}, said_review_count=0)

        result = await dialog.speak()

        assert result == ""
        assert len(dialog.host.sent) == 1
        assert await dialog.message_history.get_message("bot-1") is not None
        assert dialog.NOTICE.to_dict()["self_messages_now"] == 1
        assert dialog.AGENT.said_intention["bot-1"]["intent"] == "answer"
        assert dialog.AGENT.said_review_count == 1

    asyncio.run(run())


def test_private_main_dialog_adds_notice_when_bot_repeats_paraphrase_words() -> None:
    async def run() -> None:
        class _Speaker:
            async def speak(self, task: PrivateSpeakTask):
                task.result = {
                    "message_sequence": ["text_1"],
                    "message_content": {"text_1": {"text": "确实确实，这波确实"}},
                    "to_agent": "",
                    "intention": {
                        "target_message_id": "user-1",
                        "intent": "agree",
                        "effect": "neutral",
                    },
                }
                task.event.set()

            def mark_failure(self, *_args, **_kwargs) -> None:
                raise AssertionError("speaker should not fail")

        class _Host:
            async def say(self, message):
                return [_private_message("bot-1", "bot", "确实确实，这波确实")]

        dialog = object.__new__(PrivateMainDialog)
        dialog.message_history = MessageManager(max_message=10)
        dialog.SPEAKER = _Speaker()
        dialog.host = _Host()
        dialog.NOTICE = ChatNotification()
        dialog._last_bot_said = deque(maxlen=20)
        dialog._last_bot_said.extend(["确实", "确实"])
        dialog._paraphrase = 2
        dialog.AGENT = SimpleNamespace(said_intention={}, said_review_count=0)

        await dialog.speak()

        notice_text = dialog.NOTICE.notice2text()
        assert "发言规范" in notice_text
        assert "复述" in notice_text

    asyncio.run(run())


def test_private_main_dialog_ignores_merged_speaker_task() -> None:
    async def run() -> None:
        class _Speaker:
            async def speak(self, task: PrivateSpeakTask):
                task.merged_to = "call-5"
                task.event.set()

            def mark_failure(self, *_args, **_kwargs) -> None:
                raise AssertionError("speaker should not fail")

        class _Host:
            def __init__(self) -> None:
                self.sent = []

            async def say(self, message):
                index = len(self.sent) + 1
                self.sent.append(message)
                return [_private_message(f"bot-{index}", "bot", f"reply-{index}")]

        dialog = object.__new__(PrivateMainDialog)
        dialog.message_history = MessageManager(max_message=10)
        dialog.SPEAKER = _Speaker()
        dialog.host = _Host()

        result = await dialog.speak(call_id="call-1")

        assert result == "已合并发言任务至 call-5"
        assert dialog.host.sent == []

    asyncio.run(run())


def test_private_speaker_llm_request_uses_shared_aqueue(monkeypatch) -> None:
    async def run() -> None:
        import TIYA.agent.private_chat_agent as private_agent_module

        recorded: list[float] = []

        class _Task:
            def __init__(self, coro) -> None:
                self.coro = coro
                self.result = []
                self.exception = []
                self.cancelled = False

            async def wait(self) -> None:
                self.result.append(await self.coro)

        class _Aqueue:
            def add_task(self, coro, timeout):
                recorded.append(timeout)
                return _Task(coro)

        monkeypatch.setattr(
            private_agent_module,
            "SETTING_CFG",
            SimpleNamespace(LLM=SimpleNamespace(SpeakerRequestTimeout=12)),
        )
        speaker = object.__new__(PrivateChatSpeaker)
        speaker.aqueue = _Aqueue()
        speaker.last_failure_reason = ""
        speaker.last_failure_retryable = False
        speaker.mark_failure = lambda *_args, **_kwargs: None

        async def speak_payload(_payload):
            return {
                "message_sequence": ["text_1"],
                "message_content": {"text_1": {"text": "hello"}},
                "to_agent": "",
                "intention": {
                    "target_message_id": "user-1",
                    "intent": "answer",
                    "effect": "neutral",
                },
            }

        speaker._speak_payload = speak_payload
        result = await speaker._request_payload(PrivateSpeakerPostPayload(messages=[]))

        assert result is not None
        assert recorded == [42]

    asyncio.run(run())
