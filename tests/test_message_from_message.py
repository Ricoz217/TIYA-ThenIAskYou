import asyncio
from types import SimpleNamespace

from TIYA.model import message as message_module
from TIYA.model.message import (
    FileMsg,
    GroupMsg,
    ImgMsg,
    MessageManager,
    PrivateMsg,
    SendFile,
    SendImage,
    SendMessage,
    TextMsg,
)


class _FakeGroupMessage:
    def __init__(self) -> None:
        self.message_id = 123
        self.user_id = 456
        self.group_id = 789
        self.sender = SimpleNamespace(nickname="UserName", card="NickName")
        self.message = [
            {
                "type": "text",
                "data": {
                    "text": "hello",
                },
            },
        ]


def test_group_msg_from_message_supports_slots_dataclass() -> None:
    raw_message = _FakeGroupMessage()

    message = GroupMsg.from_message(raw_message)

    assert message is not None
    assert message.msg_id == "123"
    assert message.user_id == "456"
    assert message.group_id == "789"
    assert message.username == "UserName"
    assert message.nickname == "NickName"
    assert len(message.content) == 1
    assert isinstance(message.content[0], TextMsg)
    assert message.content[0].text == "hello"


def test_message_manager_add_message_accepts_group_message(monkeypatch) -> None:
    async def run_test() -> None:
        monkeypatch.setattr(message_module, "GroupMessage", _FakeGroupMessage)
        manager = MessageManager(max_message=10)

        message = await manager.add_message(_FakeGroupMessage())

        assert isinstance(message, GroupMsg)
        assert message.msg_id == "123"
        assert message.group_id == "789"

    asyncio.run(run_test())


def test_group_msg_from_send_uses_non_mutating_file_check(monkeypatch) -> None:
    existing = "a" * 32
    missing = "b" * 32
    checked: list[str] = []

    def check_file_exists(hash_name: str) -> bool:
        checked.append(hash_name)
        return hash_name == existing

    monkeypatch.setattr(message_module, "check_file_exists", check_file_exists)
    monkeypatch.setattr(message_module, "get_bot_uid", lambda: "bot")
    monkeypatch.setattr(message_module, "get_bot_name", lambda: "BOT")

    messages = GroupMsg.from_send(
        group_id="group",
        bot_nickname="Bot",
        messages_id=["message"],
        send=SendMessage(SendImage(existing), SendImage(missing)),
        send_time=1.0,
    )

    assert checked == [existing, missing]
    assert len(messages) == 1
    assert messages[0] is not None
    images = [item for item in messages[0].content if isinstance(item, ImgMsg)]
    assert [image.hash_name for image in images] == [existing]


def test_private_msg_from_send_uses_non_mutating_file_check(monkeypatch) -> None:
    existing_image = "a" * 32
    existing_file = "b" * 32
    missing = "c" * 32
    existing = {existing_image, existing_file}
    checked: list[str] = []

    def check_file_exists(hash_name: str) -> bool:
        checked.append(hash_name)
        return hash_name in existing

    monkeypatch.setattr(message_module, "check_file_exists", check_file_exists)
    monkeypatch.setattr(message_module, "get_bot_uid", lambda: "bot")
    monkeypatch.setattr(message_module, "get_bot_name", lambda: "BOT")

    messages = PrivateMsg.from_send(
        messages_id=["message"],
        send=SendMessage(
            SendImage(existing_image),
            SendImage(missing),
            SendFile(existing_file),
            SendFile(missing),
        ),
        send_time=1.0,
    )

    assert checked == [existing_image, missing, existing_file, missing]
    assert len(messages) == 1
    assert messages[0] is not None
    assert any(
        isinstance(item, ImgMsg) and item.hash_name == existing_image
        for item in messages[0].content
    )
    assert any(
        isinstance(item, FileMsg) and item.hash_name == existing_file
        for item in messages[0].content
    )
