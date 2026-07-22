from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from TIYA import aqueue
from TIYA import private_chat as private_chat_module
from TIYA.config import ConfigMap, DEFAULT_SETTING, SETTING_CFG
from TIYA.global_vars import PRIVATE_CHATS
from TIYA.model import message as message_module
from TIYA.model.message import (
    PrivateMsg,
    SendBreak,
    SendMessage,
    SendText,
    TextMsg,
)
from ncatbot.core.message import MessageChain, Text


@pytest.fixture(autouse=True)
def _private_runtime_settings(monkeypatch, tmp_path):
    """Keep runtime tests independent from application startup order and data."""
    for section, values in DEFAULT_SETTING.items():
        if section not in SETTING_CFG:
            monkeypatch.setitem(SETTING_CFG, section, ConfigMap(values))

    from TIYA import private_chat

    monkeypatch.setattr(private_chat, "PRIVATE_CHATS_DIR", tmp_path)
    original_get_config = private_chat.get_private_user_config

    def _notice_only_config(user_id: str) -> ConfigMap:
        config = original_get_config(user_id)
        config.chat = False
        return config

    monkeypatch.setattr(
        private_chat,
        "get_private_user_config",
        _notice_only_config,
    )


class _FakePrivateMessage:
    def __init__(
            self,
            message_id: str,
            user_id: str,
            text: str,
            *,
            username: str = "User"
    ):
        self.message_id = message_id
        self.user_id = user_id
        self.sender = SimpleNamespace(nickname=username, card="")
        self.message = [{"type": "text", "data": {"text": text}}]


def _private_msg(message_id: str, user_id: str, text: str) -> PrivateMsg:
    return PrivateMsg(
        msg_id=message_id,
        user_id=user_id,
        username="User",
        nickname="",
        content=[TextMsg(text=text)]
    )


def test_private_msg_round_trip_and_copy(monkeypatch) -> None:
    monkeypatch.setattr(message_module, "PrivateMessage", _FakePrivateMessage)
    raw = _FakePrivateMessage("message-1", "user-1", "hello")

    message = PrivateMsg.from_message(raw)

    assert message is not None
    assert message.msg_id == "message-1"
    assert message.user_id == "user-1"
    assert message.username == "User"
    assert isinstance(message.content[0], TextMsg)
    assert message.content[0].text == "hello"

    restored = PrivateMsg.from_dict(message.to_dict())
    assert restored is not None
    assert restored.to_dict() == message.to_dict()

    copied = message.copy()
    assert isinstance(copied, PrivateMsg)
    assert copied is not message
    assert copied.content == message.content


def test_private_msg_from_send_uses_successful_message_ids() -> None:
    sent = PrivateMsg.from_send(
        messages_id=["sent-1", "sent-2"],
        send=SendMessage("first"),
    )

    assert len(sent) == 1
    assert sent[0] is not None
    assert sent[0].msg_id == "sent-1"
    assert sent[0].content == [TextMsg(text="first")]


def test_aqueue_can_skip_config_observer(monkeypatch) -> None:
    register = Mock()
    monkeypatch.setattr(aqueue.CONFIG_OBSERVER, "register", register)

    aqueue.Aqueue(min_worker=1, max_worker=1, observe_config=False)

    register.assert_not_called()


def test_private_chat_processes_messages_in_fifo_order(monkeypatch) -> None:
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        monkeypatch.setattr(message_module, "PrivateMessage", _FakePrivateMessage)
        calls: list[str] = []

        class _Dialog:
            async def accept_message(self, message: PrivateMsg) -> bool:
                await asyncio.sleep(0)
                calls.append(message.msg_id)
                return False

        chat = PrivateChat("user-1", "User")
        chat.dialog_list.append(_Dialog())
        try:
            tasks = [
                await chat.accept_message(_FakePrivateMessage(str(index), "user-1", str(index)))
                for index in range(3)
            ]
            await asyncio.gather(*(task.event.wait() for task in tasks if task is not None))

            assert calls == ["0", "1", "2"]
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_private_chat_initializes_own_fav_store(monkeypatch, tmp_path) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        calls: list[tuple[object, object]] = []
        fav = SimpleNamespace(auto_fav=AsyncMock())

        def fake_get_autofav(index_file, logger=None):
            calls.append((index_file, logger))
            return fav

        monkeypatch.setattr(private_chat_module, "get_autofav", fake_get_autofav)
        chat = PrivateChat("user-fav", "User")
        try:
            assert chat.fav is fav
            assert calls == [(tmp_path / "user-fav" / "fav_index.json", chat.logger)]
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_private_chat_message_flow_updates_own_fav_store(monkeypatch) -> None:
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        chat = PrivateChat("user-fav-flow", "User")
        chat.fav = SimpleNamespace(auto_fav=AsyncMock())
        try:
            message = _private_msg("message-fav", "user-fav-flow", "hello")

            await chat.message_flow(message)
            await asyncio.sleep(0)

            chat.fav.auto_fav.assert_awaited_once_with(message)
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_notice_requests_are_consumed_fifo_and_respect_intercept() -> None:
    from TIYA.private_chat import NoticeResponse, PrivateChat

    async def run_test() -> None:
        propagated: list[str] = []

        class _Dialog:
            async def accept_message(self, message: PrivateMsg) -> bool:
                propagated.append(message.msg_id)
                return False

        chat = PrivateChat("user-1", "User")
        chat.dialog_list.append(_Dialog())
        first = NoticeResponse(
            echo="first",
            user_id="user-1",
            message_id="sent-1",
            expected_replies=2,
            intercept=True
        )
        second = NoticeResponse(
            echo="second",
            user_id="user-1",
            message_id="sent-2",
            expected_replies=1,
            intercept=False
        )
        chat.register_notice(first)
        chat.register_notice(second)
        try:
            assert await chat.message_flow(_private_msg("reply-1", "user-1", "one")) is True
            assert await chat.message_flow(_private_msg("reply-2", "user-1", "two")) is True
            assert first.done
            assert [message.msg_id for message in first.replies] == ["reply-1", "reply-2"]
            assert not second.done

            assert await chat.message_flow(_private_msg("reply-3", "user-1", "three")) is False
            assert second.done
            assert propagated == ["reply-3"]
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_wait_any_cancels_other_targets(monkeypatch) -> None:
    from TIYA.private_chat import NoticeManager, PrivateChat

    async def run_test() -> None:
        PRIVATE_CHATS.clear()
        first_chat = PrivateChat("user-1", "One")
        second_chat = PrivateChat("user-2", "Two")
        PRIVATE_CHATS.update({
            "user-1": first_chat,
            "user-2": second_chat,
        })

        async def _say(self, message):
            return [
                PrivateMsg(
                    msg_id=f"sent-{self.user_id}",
                    user_id="bot",
                    username="BOT",
                    nickname="",
                    content=[TextMsg(text=str(message))]
                )
            ]

        monkeypatch.setattr(PrivateChat, "say", _say)
        try:
            assert await NoticeManager.request_someone("user-1", "token", echo="batch") == "batch"
            assert await NoticeManager.request_someone("user-2", "token", echo="batch") == "batch"

            waiter = asyncio.create_task(NoticeManager.wait_any("batch"))
            await first_chat.message_flow(_private_msg("reply-1", "user-1", "answer"))
            response = await waiter

            assert response is not None
            assert response.user_id == "user-1"
            assert response.done
            assert second_chat.get_notice("batch") is None
        finally:
            await first_chat.shutdown()
            await second_chat.shutdown()
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_wait_any_ignores_cancelled_target(monkeypatch) -> None:
    from TIYA.private_chat import NoticeManager, PrivateChat

    async def run_test() -> None:
        PRIVATE_CHATS.clear()
        first_chat = PrivateChat("user-1", "One")
        second_chat = PrivateChat("user-2", "Two")

        async def _say(self, message):
            return [
                PrivateMsg(
                    msg_id=f"sent-{self.user_id}",
                    user_id="bot",
                    username="BOT",
                    nickname="",
                    content=[TextMsg(text=str(message))]
                )
            ]

        monkeypatch.setattr(PrivateChat, "say", _say)
        try:
            await NoticeManager.request_someone("user-1", "token", echo="batch")
            await NoticeManager.request_someone("user-2", "token", echo="batch")
            waiter = asyncio.create_task(NoticeManager.wait_any("batch"))
            await asyncio.sleep(0)

            first_chat.remove_notice("batch", cancel=True)
            await second_chat.message_flow(_private_msg("reply-2", "user-2", "answer"))
            response = await waiter

            assert response is not None
            assert response.user_id == "user-2"
        finally:
            await first_chat.shutdown()
            await second_chat.shutdown()
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_wait_someone_timeout_cancels_notice() -> None:
    from TIYA.private_chat import NoticeManager, NoticeResponse, PrivateChat

    async def run_test() -> None:
        PRIVATE_CHATS.clear()
        chat = PrivateChat("user-1", "User")
        response = NoticeResponse(
            echo="pending",
            user_id="user-1",
            message_id="sent",
            response_timeout=1,
        )
        chat.register_notice(response)
        try:
            result = await NoticeManager.wait_someone(
                "user-1",
                "pending",
                timeout=0.01,
            )

            assert result is None
            assert response.cancelled
            assert chat.get_notice("pending") is None
        finally:
            await chat.shutdown()
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_wait_any_timeout_cancels_all_targets() -> None:
    from TIYA.private_chat import NoticeManager, NoticeResponse, PrivateChat

    async def run_test() -> None:
        PRIVATE_CHATS.clear()
        chats = [
            PrivateChat("user-1", "One"),
            PrivateChat("user-2", "Two"),
        ]
        responses = []
        for chat in chats:
            response = NoticeResponse(
                echo="batch",
                user_id=chat.user_id,
                message_id=f"sent-{chat.user_id}",
                response_timeout=1,
            )
            chat.register_notice(response)
            responses.append(response)

        try:
            result = await NoticeManager.wait_any("batch", timeout=0.01)

            assert result is None
            assert all(response.cancelled for response in responses)
            assert all(chat.get_notice("batch") is None for chat in chats)
        finally:
            await asyncio.gather(*(chat.shutdown() for chat in chats))
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_notice_response_expires_without_waiter() -> None:
    from TIYA.private_chat import NoticeResponse, PrivateChat

    async def run_test() -> None:
        PRIVATE_CHATS.clear()
        chat = PrivateChat("user-1", "User")
        response = NoticeResponse(
            echo="expiring",
            user_id="user-1",
            message_id="sent",
            response_timeout=0.01,
            intercept=True,
        )
        chat.register_notice(response)
        try:
            await asyncio.sleep(0.03)

            assert response.expired
            assert response.event.is_set()
            assert chat.get_notice("expiring") is None
            assert await chat.message_flow(
                _private_msg("late-reply", "user-1", "too late")
            ) is False
        finally:
            await chat.shutdown()
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_send_failure_does_not_register_notice(monkeypatch) -> None:
    from TIYA.private_chat import NoticeManager, PrivateChat

    async def run_test() -> None:
        PRIVATE_CHATS.clear()
        chat = PrivateChat("user-1", "User")
        PRIVATE_CHATS["user-1"] = chat
        monkeypatch.setattr(chat, "say", AsyncMock(return_value=[None]))
        try:
            echo = await NoticeManager.request_someone("user-1", "request", echo="failed")

            assert echo is None
            assert chat.get_notice("failed") is None
        finally:
            await chat.shutdown()
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_private_chat_say_records_successful_message(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        chat = PrivateChat("user-1", "User")
        monkeypatch.setattr(
            private_chat_module.api,
            "say",
            AsyncMock(return_value={
                "status": "ok",
                "data": {"message_id": "sent-1"}
            })
        )
        try:
            sent = await chat.say("hello")

            assert len(sent) == 1
            assert sent[0] is not None
            assert sent[0].msg_id == "sent-1"
            assert chat.message_history["sent-1"].msg_id == "sent-1"
        finally:
            await chat.shutdown()
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_private_chat_say_delays_only_after_first_message_chain(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    class _MessageHistory:
        async def add_message(self, message):
            return message

    async def run_test() -> None:
        chat = object.__new__(PrivateChat)
        chat.user_id = "user-1"
        chat.message_history = _MessageHistory()
        sleep = AsyncMock()
        monkeypatch.setattr(private_chat_module.asyncio, "sleep", sleep)
        monkeypatch.setattr(private_chat_module.random, "uniform", lambda *_: 0.25)
        monkeypatch.setattr(
            private_chat_module,
            "SETTING_CFG",
            SimpleNamespace(
                PrivateChat=SimpleNamespace(
                    ApiTimeout=60,
                    SleepPerWord=0.4,
                )
            ),
        )
        monkeypatch.setattr(
            private_chat_module.api,
            "say",
            AsyncMock(side_effect=[
                {"status": "ok", "data": {"message_id": "sent-1"}},
                {"status": "ok", "data": {"message_id": "sent-2"}},
            ]),
        )
        monkeypatch.setattr(chat, "_log_message", AsyncMock())

        sent = await chat.say(SendMessage(
            SendText("first"),
            SendBreak(),
            SendText("second"),
        ))

        assert [message.msg_id for message in sent if message is not None] == [
            "sent-1",
            "sent-2",
        ]
        assert [call.args[0] for call in sleep.await_args_list] == [0.5, 4.25]

    asyncio.run(run_test())


def test_private_chat_system_send_uses_default_fixed_delay(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    class _MessageHistory:
        async def add_message(self, message):
            return message

    async def run_test() -> None:
        chat = object.__new__(PrivateChat)
        chat.user_id = "user-1"
        chat.message_history = _MessageHistory()
        sleep = AsyncMock()
        monkeypatch.setattr(private_chat.asyncio, "sleep", sleep)
        monkeypatch.setattr(
            private_chat,
            "SETTING_CFG",
            SimpleNamespace(PrivateChat=SimpleNamespace(ApiTimeout=60)),
        )
        monkeypatch.setattr(
            private_chat.api,
            "say",
            AsyncMock(side_effect=[
                {"status": "ok", "data": {"message_id": "sent-1"}},
                {"status": "ok", "data": {"message_id": "sent-2"}},
            ]),
        )
        monkeypatch.setattr(chat, "_log_message", AsyncMock())

        sent = await chat.system_send(SendMessage(
            SendText("first"),
            SendBreak(),
            SendText("second"),
        ))

        assert [message.msg_id for message in sent if message is not None] == [
            "sent-1",
            "sent-2",
        ]
        assert [call.args[0] for call in sleep.await_args_list] == [0.3]

    asyncio.run(run_test())


def test_private_chat_system_send_accepts_callable_delay(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    class _MessageHistory:
        async def add_message(self, message):
            return message

    async def run_test() -> None:
        chat = object.__new__(PrivateChat)
        chat.user_id = "user-1"
        chat.message_history = _MessageHistory()
        sleep = AsyncMock()
        delay = Mock(side_effect=[0.1, 0.2])
        monkeypatch.setattr(private_chat.asyncio, "sleep", sleep)
        monkeypatch.setattr(
            private_chat,
            "SETTING_CFG",
            SimpleNamespace(PrivateChat=SimpleNamespace(ApiTimeout=60)),
        )
        monkeypatch.setattr(
            private_chat.api,
            "say",
            AsyncMock(side_effect=[
                {"status": "ok", "data": {"message_id": "sent-1"}},
                {"status": "ok", "data": {"message_id": "sent-2"}},
                {"status": "ok", "data": {"message_id": "sent-3"}},
            ]),
        )
        monkeypatch.setattr(chat, "_log_message", AsyncMock())

        await chat.system_send(
            SendMessage(
                SendText("first"),
                SendBreak(),
                SendText("second"),
                SendBreak(),
                SendText("third"),
            ),
            delay=delay,
        )

        assert [call.args[0] for call in sleep.await_args_list] == [0.1, 0.2]
        assert delay.call_count == 2

    asyncio.run(run_test())


def test_notice_targets_filter_placeholders_and_invalid_values(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import NoticeManager

    monkeypatch.setattr(
        private_chat_module,
        "BASE_CFG",
        SimpleNamespace(
            NoticeList=[
                "114514",
                "1919810",
                "",
                [],
                "10001",
                10002,
                "10001",
            ]
        )
    )

    assert NoticeManager._notice_targets() == ["10001", "10002"]


def test_request_all_registers_same_echo_for_every_target(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import NoticeManager, PrivateChat

    async def run_test() -> None:
        PRIVATE_CHATS.clear()
        monkeypatch.setattr(
            private_chat,
            "BASE_CFG",
            SimpleNamespace(NoticeList=["user-1", "user-2"])
        )

        async def _say(self, message):
            return [
                PrivateMsg(
                    msg_id=f"sent-{self.user_id}",
                    user_id="bot",
                    username="BOT",
                    nickname="",
                    content=[TextMsg(text=str(message))]
                )
            ]

        monkeypatch.setattr(PrivateChat, "say", _say)
        try:
            echo = await NoticeManager.request_all("request", echo="batch")

            assert echo == "batch"
            assert PRIVATE_CHATS["user-1"].get_notice("batch") is not None
            assert PRIVATE_CHATS["user-2"].get_notice("batch") is not None
        finally:
            await asyncio.gather(*(
                chat.shutdown()
                for chat in list(PRIVATE_CHATS.values())
            ))
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_request_all_applies_response_timeout(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import NoticeManager, PrivateChat

    async def run_test() -> None:
        PRIVATE_CHATS.clear()
        monkeypatch.setattr(
            private_chat_module,
            "BASE_CFG",
            SimpleNamespace(NoticeList=["user-1"]),
        )

        async def _say(self, message):
            return [
                PrivateMsg(
                    msg_id=f"sent-{self.user_id}",
                    user_id="bot",
                    username="BOT",
                    nickname="",
                    content=[TextMsg(text=str(message))],
                )
            ]

        monkeypatch.setattr(PrivateChat, "say", _say)
        try:
            echo = await NoticeManager.request_all(
                "request",
                echo="batch",
                response_timeout=12.5,
            )
            response = PRIVATE_CHATS["user-1"].get_notice("batch")

            assert echo == "batch"
            assert response is not None
            assert response.response_timeout == 12.5
        finally:
            await asyncio.gather(*(
                chat.shutdown()
                for chat in list(PRIVATE_CHATS.values())
            ))
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_private_api_sends_message_chain(monkeypatch) -> None:
    from TIYA.api import private_api

    async def run_test() -> None:
        bot = SimpleNamespace(
            api=SimpleNamespace(
                post_private_msg=AsyncMock(return_value={
                    "status": "ok",
                    "data": {"message_id": 123}
                })
            )
        )
        monkeypatch.setattr(private_api, "get_bot", lambda: bot)
        chain = MessageChain([Text("hello")])

        response = await private_api.say("user-1", chain)

        assert response["status"] == "ok"
        bot.api.post_private_msg.assert_awaited_once_with("user-1", rtf=chain)

    asyncio.run(run_test())


def test_shutdown_cancels_waiters_and_stops_queue() -> None:
    from TIYA.private_chat import NoticeManager, NoticeResponse, PrivateChat

    async def run_test() -> None:
        chat = PrivateChat("user-1", "User")
        response = NoticeResponse(
            echo="pending",
            user_id="user-1",
            message_id="sent",
            expected_replies=1,
            intercept=True
        )
        chat.register_notice(response)
        PRIVATE_CHATS["user-1"] = chat
        waiter = asyncio.create_task(NoticeManager.wait_someone("user-1", "pending"))

        await chat.shutdown()
        result = await waiter

        assert result is not None
        assert result.cancelled
        assert not chat.enable
        assert all(worker._work_loop.done() for worker in chat.aqueue._workers)
        PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_private_main_dialog_is_loaded_only_for_unhandled_chat(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        starts: list[str] = []

        class _MainDialog:
            def __init__(self, *, host):
                self.host = host

            async def start(self):
                starts.append("start")

            async def accept_message(self, _message):
                starts.append("accept")
                return True

            async def shutdown(self):
                starts.append("shutdown")

            async def save(self):
                return None

            def update_config(self):
                return None

            @property
            def idle(self):
                return True

        monkeypatch.setattr(private_chat, "PrivateMainDialog", _MainDialog)
        chat = PrivateChat("lazy-user", "User")
        chat.private_config.chat = True
        command = chat.command_dialog

        class _Intercept:
            async def accept_message(self, _message):
                return True

        chat.dialog_list = [_Intercept()]
        try:
            assert await chat.message_flow(_private_msg("1", "lazy-user", "manage"))
            assert chat.main_dialog is None

            chat.dialog_list = [command]
            assert await chat.message_flow(_private_msg("2", "lazy-user", "hello"))
            assert chat.main_dialog is not None
            assert starts[:2] == ["start", "accept"]
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_private_chat_idle_expiry_defers_while_main_dialog_busy(monkeypatch) -> None:
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        chat = PrivateChat("idle-user", "User")
        chat._chat_object_timeout = 0.01
        chat._last_activity_at -= 1
        class _BusyMain:
            idle = False

            async def save(self):
                return None

            async def shutdown(self):
                return None

        busy = _BusyMain()
        chat.main_dialog = busy

        await chat._expire_if_idle()
        assert chat.enable
        assert chat.user_id in PRIVATE_CHATS

        busy.idle = True
        await chat._expire_if_idle()
        assert not chat.enable
        assert chat.user_id not in PRIVATE_CHATS

    asyncio.run(run_test())


def test_private_chat_activity_refresh_only_updates_timestamp(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        monkeypatch.setattr(private_chat_module, "monotonic", lambda: 100.0)
        chat = PrivateChat("timestamp-user", "User")
        try:
            reaper = private_chat_module._PRIVATE_CHAT_REAPER_TASK
            assert chat._last_activity_at == 100.0
            assert not hasattr(chat, "_idle_task")

            monkeypatch.setattr(private_chat_module, "monotonic", lambda: 125.0)
            chat.touch_activity()
            assert chat._last_activity_at == 125.0
            assert private_chat_module._PRIVATE_CHAT_REAPER_TASK is reaper
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_private_chats_share_one_central_reaper() -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        first = PrivateChat("reaper-user-1", "One")
        reaper = private_chat_module._PRIVATE_CHAT_REAPER_TASK
        second = PrivateChat("reaper-user-2", "Two")
        try:
            assert reaper is not None
            assert private_chat_module._PRIVATE_CHAT_REAPER_TASK is reaper
            assert not reaper.done()
        finally:
            await first.shutdown()
            await second.shutdown()

        assert private_chat_module._PRIVATE_CHAT_REAPER_TASK is None
        assert reaper.done()

    asyncio.run(run_test())


def test_central_reaper_reclaims_an_inactive_private_chat(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        monkeypatch.setitem(
            private_chat_module.SETTING_CFG.PrivateChat,
            "ChatObjectTimeout",
            0.01,
        )
        chat = PrivateChat("expired-user", "User")

        await asyncio.wait_for(
            _wait_until(lambda: chat.user_id not in PRIVATE_CHATS),
            timeout=2.0,
        )

        assert chat.user_id not in PRIVATE_CHATS
        assert private_chat_module._PRIVATE_CHAT_REAPER_TASK is None

    async def _wait_until(predicate) -> None:
        while not predicate():
            await asyncio.sleep(0)

    asyncio.run(run_test())


def test_central_reaper_waits_for_busy_private_chat(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        monkeypatch.setitem(
            private_chat_module.SETTING_CFG.PrivateChat,
            "ChatObjectTimeout",
            0.01,
        )

        class _BusyMain:
            idle = False

            async def save(self):
                return None

            async def shutdown(self):
                return None

        chat = PrivateChat("busy-reaper-user", "User")
        busy = _BusyMain()
        chat.main_dialog = busy
        try:
            await asyncio.sleep(0.03)
            assert chat.enable
            assert chat.user_id in PRIVATE_CHATS

            busy.idle = True
            private_chat._notify_private_chat_reaper()
            await asyncio.wait_for(
                _wait_until(lambda: chat.user_id not in PRIVATE_CHATS),
                timeout=2.0,
            )
        finally:
            if chat.enable:
                await chat.shutdown()

    async def _wait_until(predicate) -> None:
        while not predicate():
            await asyncio.sleep(0)

    asyncio.run(run_test())


def test_activity_refresh_restarts_a_stopped_reaper() -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        chat = PrivateChat("self-heal-user", "User")
        stopped = private_chat._PRIVATE_CHAT_REAPER_TASK
        assert stopped is not None
        stopped.cancel()
        await asyncio.gather(stopped, return_exceptions=True)
        assert stopped.done()

        try:
            chat.touch_activity()
            restarted = private_chat_module._PRIVATE_CHAT_REAPER_TASK
            assert restarted is not None
            assert restarted is not stopped
            assert not restarted.done()
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_one_chat_reap_failure_does_not_stop_other_expirations(monkeypatch) -> None:
    from TIYA import private_chat
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        monkeypatch.setitem(
            private_chat_module.SETTING_CFG.PrivateChat,
            "ChatObjectTimeout",
            0.01,
        )
        broken = PrivateChat("broken-reap-user", "Broken")
        healthy = PrivateChat("healthy-reap-user", "Healthy")

        async def fail_reap(*, now=None):
            raise RuntimeError("simulated reap failure")

        monkeypatch.setattr(broken, "_expire_if_idle", fail_reap)
        broken._last_activity_at -= 1
        healthy._last_activity_at -= 1
        private_chat_module._notify_private_chat_reaper()
        try:
            await asyncio.wait_for(
                _wait_until(lambda: healthy.user_id not in PRIVATE_CHATS),
                timeout=2.0,
            )
            reaper = private_chat._PRIVATE_CHAT_REAPER_TASK
            assert broken.user_id in PRIVATE_CHATS
            assert reaper is not None
            assert not reaper.done()
        finally:
            await broken.shutdown()
            if healthy.enable:
                await healthy.shutdown()

    async def _wait_until(predicate) -> None:
        while not predicate():
            await asyncio.sleep(0)

    asyncio.run(run_test())


def test_reaper_loop_retries_after_unexpected_cycle_error(monkeypatch) -> None:
    from TIYA import private_chat

    async def run_test() -> None:
        calls = 0

        async def cycle() -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated reaper cycle crash")
            return False

        monkeypatch.setattr(private_chat_module, "_private_chat_reaper_cycle", cycle)
        monkeypatch.setattr(private_chat_module, "_PRIVATE_CHAT_REAPER_FAILURE_RETRY", 0)
        private_chat._ensure_private_chat_reaper()
        task = private_chat_module._PRIVATE_CHAT_REAPER_TASK
        assert task is not None

        await task

        assert calls == 2
        assert private_chat_module._PRIVATE_CHAT_REAPER_TASK is None

    asyncio.run(run_test())
