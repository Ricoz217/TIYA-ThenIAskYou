from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from TIYA.config import ConfigMap, SETTING_CFG
from TIYA.model.message import (
    ImgMsg,
    PrivateMsg,
    SendBreak,
    SendImage,
    SendMessage,
    SendText,
    TextMsg,
)


@pytest.fixture(autouse=True)
def _message_history_setting(monkeypatch):
    if "Groups" not in SETTING_CFG:
        SETTING_CFG["Groups"] = ConfigMap()
    if "PrivateChat" not in SETTING_CFG:
        SETTING_CFG["PrivateChat"] = ConfigMap()
    monkeypatch.setitem(SETTING_CFG.Groups, "MaxMessageHistory", 10)
    monkeypatch.setitem(SETTING_CFG.PrivateChat, "ManagementDialogTimeout", 1)


class _Store:
    def __init__(self, mapping: dict[str, set[str]] | None = None):
        self.mapping = {
            title: set(hash_names)
            for title, hash_names in (mapping or {}).items()
        }
        self.add_calls: list[list[tuple[str, str]]] = []
        self.rename_calls: list[tuple[str, str]] = []
        self.remove_calls: list[list[str]] = []

    @property
    def fav_list(self) -> dict[str, set[str]]:
        return {
            title: set(hash_names)
            for title, hash_names in self.mapping.items()
        }

    async def add_favs(self, entries: list[tuple[str, str]]) -> int:
        self.add_calls.append(entries.copy())
        for hash_name, title in entries:
            self.mapping.setdefault(title, set()).add(hash_name)
        return len(entries)

    async def rename_fav(self, hash_name: str, title: str) -> bool:
        self.rename_calls.append((hash_name, title))
        found = False
        for old_title, hash_names in list(self.mapping.items()):
            if hash_name not in hash_names:
                continue
            found = True
            hash_names.remove(hash_name)
            if not hash_names:
                del self.mapping[old_title]
        if found:
            self.mapping.setdefault(title, set()).add(hash_name)
        return found

    async def remove_favs(self, hash_names: list[str]) -> int:
        self.remove_calls.append(hash_names.copy())
        removed: set[str] = set()
        for title, members in list(self.mapping.items()):
            matches = members.intersection(hash_names)
            members.difference_update(matches)
            removed.update(matches)
            if not members:
                del self.mapping[title]
        return len(removed)


class _Sender:
    def __init__(self):
        self.messages: list[str | SendMessage] = []
        self.fail_images = False

    async def __call__(self, message):
        self.messages.append(message)
        if not isinstance(message, SendMessage):
            return [object()]

        chain_count = 0
        has_items = False
        for item in message.items:
            if isinstance(item, SendBreak):
                if has_items:
                    chain_count += 1
                    has_items = False
                continue
            has_items = True
        if has_items:
            chain_count += 1

        if self.fail_images and any(
                isinstance(item, SendImage)
                for item in message.items
        ):
            return [None] * chain_count
        return [object()] * chain_count

    @property
    def image_hashes(self) -> list[str]:
        return [
            item.image
            for message in self.messages
            if isinstance(message, SendMessage)
            for item in message.items
            if isinstance(item, SendImage)
        ]

    @property
    def text(self) -> str:
        return "\n".join(
            message
            for message in self.messages
            if isinstance(message, str)
        )

    @property
    def embedded_text(self) -> str:
        return "\n".join(
            item.text
            for message in self.messages
            if isinstance(message, SendMessage)
            for item in message.items
            if isinstance(item, SendText)
        )


def _message(
        text: str = "",
        *,
        images: list[ImgMsg] | None = None,
        user_id: str = "admin",
) -> PrivateMsg:
    content = []
    if text:
        content.append(TextMsg(text=text))
    content.extend(images or [])
    return PrivateMsg(
        msg_id=f"message-{id(content)}",
        user_id=user_id,
        username="Admin",
        nickname="",
        content=content,
    )


def _image(hash_name: str = "", *, done: bool = True) -> ImgMsg:
    image = ImgMsg(name=hash_name or "missing", hash_name=hash_name)
    if done:
        image.event.set()
    return image


def _make_system(
        mapping: dict[str, set[str]] | None = None,
        *,
        system_sender: _Sender | None = None,
):
    from TIYA.fav_system import FavSystem, FavTarget

    store = _Store(mapping)
    sender = _Sender()
    exits: list[bool] = []

    async def close() -> bool:
        exits.append(True)
        return True

    system = FavSystem(
        target=FavTarget(
            kind="group",
            identifier="100",
            display_name="测试群(100)",
            store=store,
        ),
        say=sender,
        system_send=system_sender or sender,
        close=close,
        image_wait_timeout=0.01,
    )
    return system, store, sender, exits


def test_list_builds_unique_ordered_selection_without_sending_images() -> None:
    async def run_test() -> None:
        system, _, sender, _ = _make_system({
            "生气": {"hash-a", "hash-b"},
            "别名": {"hash-a"},
        })

        assert await system.handle_message(_message("--list")) is True

        assert [item.hash_name for item in system.selection] == [
            "hash-a",
            "hash-b",
        ]
        assert "1/2" in sender.text
        assert "2/2" in sender.text
        assert "hash-a" not in sender.text
        assert sender.image_hashes == []

    asyncio.run(run_test())


def test_navigation_sends_one_image_and_does_not_wrap() -> None:
    async def run_test() -> None:
        system, _, sender, _ = _make_system({"标题": {"hash-a", "hash-b"}})
        await system.handle_message(_message("--list"))

        await system.handle_message(_message("--next"))
        await system.handle_message(_message("--next"))
        await system.handle_message(_message("--next"))

        assert sender.image_hashes == ["hash-a", "hash-b"]
        assert system.cursor_index == 1
        assert "已到底" in sender.text

        await system.handle_message(_message("--prev"))
        await system.handle_message(_message("--prev"))
        assert sender.image_hashes[-1] == "hash-a"
        assert "已到开头" in sender.text

    asyncio.run(run_test())


def test_image_output_uses_system_sender_instead_of_say() -> None:
    async def run_test() -> None:
        system_sender = _Sender()
        system, _, sender, _ = _make_system(
            {"标题": {"hash-a"}},
            system_sender=system_sender,
        )
        await system.handle_message(_message("--list"))

        await system.handle_message(_message("--next"))

        assert sender.image_hashes == []
        assert system_sender.image_hashes == ["hash-a"]

    asyncio.run(run_test())


def test_select_crops_by_two_boundaries_and_rebuilds_indexes() -> None:
    async def run_test() -> None:
        system, _, sender, _ = _make_system({
            "标题": {"hash-a", "hash-b", "hash-c", "hash-d"},
        })
        await system.handle_message(_message("--list"))
        sender.messages.clear()

        await system.handle_message(_message("--select 2, 3"))

        assert [item.hash_name for item in system.selection] == [
            "hash-b",
            "hash-c",
        ]
        assert "1/2" in sender.text
        assert "2/2" in sender.text
        assert sender.image_hashes == []

    asyncio.run(run_test())


def test_images_ignore_attached_text_and_skip_missing_hash_name() -> None:
    async def run_test() -> None:
        system, store, sender, _ = _make_system({"旧标题": {"old-hash"}})
        await system.handle_message(_message("--list"))
        await system.handle_message(_message("--next"))
        sender.messages.clear()

        await system.handle_message(_message(
            "不能成为标题",
            images=[_image("new-hash"), _image()],
        ))

        assert [item.hash_name for item in system.pending_add] == ["new-hash"]
        assert system.pending_add[0].title == ""
        assert store.rename_calls == []
        assert sender.image_hashes == []

    asyncio.run(run_test())


def test_image_waits_until_download_provides_hash_name() -> None:
    async def run_test() -> None:
        system, _, _, _ = _make_system()
        image = _image(done=False)

        async def finish_download() -> None:
            await asyncio.sleep(0)
            image.hash_name = "downloaded-hash"

        task = asyncio.create_task(finish_download())
        await system.handle_message(_message(images=[image]))
        await task

        assert [item.hash_name for item in system.pending_add] == [
            "downloaded-hash",
        ]

    asyncio.run(run_test())


def test_check_add_requires_titles_then_adds_the_confirmed_whole_queue() -> None:
    async def run_test() -> None:
        system, store, sender, _ = _make_system()
        await system.handle_message(_message(
            images=[_image("new-a"), _image("new-b")],
        ))

        await system.handle_message(_message("--check-add"))
        assert sender.image_hashes == ["new-a", "new-b"]
        assert system.add_confirmed is False

        await system.handle_message(_message("--jump 1"))
        await system.handle_message(_message("标题甲"))
        await system.handle_message(_message("--next"))
        await system.handle_message(_message("标题乙"))
        image_count = len(sender.image_hashes)

        await system.handle_message(_message("--check-add"))
        assert system.add_confirmed is True
        await system.handle_message(_message("--add"))

        assert store.add_calls == [[
            ("new-a", "标题甲"),
            ("new-b", "标题乙"),
        ]]
        assert system.pending_add == ()
        assert system.selection == ()
        assert len(sender.image_hashes) == image_count + 2

    asyncio.run(run_test())


def test_any_mutation_invalidates_confirmation() -> None:
    async def run_test() -> None:
        system, store, _, _ = _make_system()
        await system.handle_message(_message(images=[_image("new-a")]))
        await system.handle_message(_message("--check-add"))
        await system.handle_message(_message("--jump 1"))
        await system.handle_message(_message("标题"))
        await system.handle_message(_message("--check-add"))
        assert system.add_confirmed is True

        await system.handle_message(_message("--jump 1"))
        assert system.add_confirmed is False
        await system.handle_message(_message("--add"))
        assert store.add_calls == []

    asyncio.run(run_test())


def test_delete_requires_check_and_removes_all_staged_hashes() -> None:
    async def run_test() -> None:
        system, store, sender, _ = _make_system({
            "标题甲": {"hash-a", "hash-b"},
            "别名": {"hash-a"},
        })
        await system.handle_message(_message("--list"))
        await system.handle_message(_message("--add-delete"))
        assert system.selection == ()

        await system.handle_message(_message("--delete"))
        assert store.remove_calls == []

        await system.handle_message(_message("--check-delete"))
        assert sender.image_hashes == ["hash-a", "hash-b"]
        assert system.delete_confirmed is True
        await system.handle_message(_message("--delete"))

        assert store.remove_calls == [["hash-a", "hash-b"]]
        assert store.mapping == {}
        assert system.pending_delete == ()
        assert system.selection == ()

    asyncio.run(run_test())


def test_add_delete_with_cursor_stages_only_current_item() -> None:
    async def run_test() -> None:
        system, _, sender, _ = _make_system({
            "标题": {"hash-a", "hash-b", "hash-c"},
        })
        await system.handle_message(_message("--list"))
        await system.handle_message(_message("--jump 2"))
        sender.messages.clear()

        await system.handle_message(_message("--add-delete"))

        assert [item.hash_name for item in system.pending_delete] == ["hash-b"]
        assert [item.hash_name for item in system.selection] == [
            "hash-a",
            "hash-b",
            "hash-c",
        ]
        assert system.cursor_index == 1
        assert "当前 2/3" in sender.text
        assert sender.image_hashes == []

    asyncio.run(run_test())


def test_remove_updates_backing_queue_and_previews_adjusted_cursor() -> None:
    async def run_test() -> None:
        system, _, sender, _ = _make_system()
        await system.handle_message(_message(
            images=[_image("new-a"), _image("new-b"), _image("new-c")],
        ))
        await system.handle_message(_message("--check-add"))
        await system.handle_message(_message("--jump 2"))
        sender.messages.clear()

        await system.handle_message(_message("--remove"))

        assert [item.hash_name for item in system.pending_add] == [
            "new-a",
            "new-c",
        ]
        assert [item.hash_name for item in system.selection] == [
            "new-a",
            "new-c",
        ]
        assert system.cursor_index == 1
        assert sender.text == ""
        assert sender.image_hashes == ["new-c"]
        assert "2/2" in sender.embedded_text

    asyncio.run(run_test())


def test_plain_text_renames_current_hash_without_resending_image() -> None:
    async def run_test() -> None:
        system, store, sender, _ = _make_system({
            "旧标题": {"hash-a", "hash-b"},
            "另一个旧标题": {"hash-a"},
        })
        await system.handle_message(_message("--list"))
        await system.handle_message(_message("--jump 1"))
        sender.messages.clear()

        await system.handle_message(_message("新标题"))

        assert store.rename_calls == [("hash-a", "新标题")]
        assert store.mapping == {
            "旧标题": {"hash-b"},
            "新标题": {"hash-a"},
        }
        assert sender.image_hashes == []

    asyncio.run(run_test())


def test_unknown_command_is_not_used_as_a_title() -> None:
    async def run_test() -> None:
        system, store, sender, _ = _make_system({"标题": {"hash-a"}})
        await system.handle_message(_message("--list"))
        await system.handle_message(_message("--jump 1"))
        sender.messages.clear()

        await system.handle_message(_message("--does-not-exist"))

        assert store.rename_calls == []
        assert "命令错误" in sender.text

    asyncio.run(run_test())


def test_fav_help_exports_command_set_help_without_sending_images() -> None:
    async def run_test() -> None:
        system, _, sender, _ = _make_system()

        await system.handle_message(_message("--help"))

        assert "--list" in sender.text
        assert "--show" in sender.text
        assert "--remove" in sender.text
        assert "--exit" in sender.text
        assert sender.image_hashes == []

        sender.messages.clear()
        await system.handle_message(_message("--help select"))
        assert "用法: --select" in sender.text
        assert "--show" not in sender.text

    asyncio.run(run_test())


def test_check_does_not_confirm_when_image_delivery_fails() -> None:
    async def run_test() -> None:
        system, _, sender, _ = _make_system()
        await system.handle_message(_message(images=[_image("new-a")]))
        await system.handle_message(_message("--check-add"))
        await system.handle_message(_message("--jump 1"))
        await system.handle_message(_message("标题"))
        sender.fail_images = True

        await system.handle_message(_message("--check-add"))

        assert system.add_confirmed is False
        assert "发送失败" in sender.text

    asyncio.run(run_test())


def test_exit_uses_close_callback() -> None:
    async def run_test() -> None:
        system, _, _, exits = _make_system()

        await system.handle_message(_message("--exit"))

        assert exits == [True]

    asyncio.run(run_test())


def test_private_fav_entry_requires_management_and_transfers_target(monkeypatch) -> None:
    import TIYA.dialog.private_dialog as private_dialog
    from TIYA.global_vars import PRIVATE_CHATS, QQ_GROUPS

    async def run_test() -> None:
        store = _Store()
        QQ_GROUPS.clear()
        QQ_GROUPS["100"] = SimpleNamespace(
            group_id="100",
            group_name="测试群",
            fav=store,
        )
        private_store = _Store()
        PRIVATE_CHATS["200"] = SimpleNamespace(
            user_id="200",
            username="测试用户",
            fav=private_store,
        )
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )

        class Host:
            user_id = "admin"
            username = "Admin"
            logger = Mock()
            management_dialog = None
            fav_dialog = None

            def __init__(self):
                self.sent: list[str] = []
                self.opened = []

            async def say(self, message):
                self.sent.append(str(message))
                return [object()]

            async def system_send(self, message, *, delay=0.3):
                return await self.say(message)

            async def open_fav_dialog(self, target):
                self.opened.append(target)
                return True

        host = Host()
        dialog = private_dialog.PrivateCommandDialog(host=host)
        try:
            await dialog.accept_message(_message("#fav --group 100"))
            assert host.opened == []
            assert any("管理界面" in text for text in host.sent)

            host.management_dialog = object()
            await dialog.accept_message(_message("#fav --group 100"))
            assert len(host.opened) == 1
            assert host.opened[0].store is store

            await dialog.accept_message(_message("#fav --private 200"))
            assert len(host.opened) == 2
            assert host.opened[1].kind == "private"
            assert host.opened[1].identifier == "200"
            assert host.opened[1].display_name == "测试用户(200)"
            assert host.opened[1].store is private_store
        finally:
            QQ_GROUPS.clear()
            PRIVATE_CHATS.clear()

    asyncio.run(run_test())


def test_private_fav_target_rejects_multiple_target_kinds(monkeypatch) -> None:
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )

        class Host:
            user_id = "admin"
            username = "Admin"
            logger = Mock()
            management_dialog = object()

            def __init__(self):
                self.sent: list[str] = []
                self.opened = []

            async def system_send(self, message, *, delay=0.3):
                self.sent.append(str(message))
                return [object()]

            async def open_fav_dialog(self, target):
                self.opened.append(target)
                return True

        host = Host()
        dialog = private_dialog.PrivateCommandDialog(host=host)
        await dialog.accept_message(
            _message("#fav --group 100 --private 200"),
        )

        assert host.opened == []
        assert "必须且只能指定" in host.sent[-1]

    asyncio.run(run_test())


def test_private_fav_target_opens_an_existing_offline_user_directory(
        monkeypatch,
        tmp_path,
) -> None:
    import TIYA.fav_system as fav_system
    from TIYA.global_vars import PRIVATE_CHATS

    store = _Store()
    user_dir = tmp_path / "300"
    user_dir.mkdir()
    PRIVATE_CHATS.clear()
    monkeypatch.setattr(fav_system, "PRIVATE_CHATS_DIR", tmp_path)
    monkeypatch.setattr(fav_system, "get_autofav", lambda path: store)

    target = fav_system.resolve_fav_target(private_id="300")

    assert target.kind == "private"
    assert target.identifier == "300"
    assert target.display_name == "300"
    assert target.store is store


def test_private_fav_target_rejects_an_unknown_user(monkeypatch, tmp_path) -> None:
    import TIYA.fav_system as fav_system
    from TIYA.global_vars import PRIVATE_CHATS

    PRIVATE_CHATS.clear()
    monkeypatch.setattr(fav_system, "PRIVATE_CHATS_DIR", tmp_path)

    with pytest.raises(fav_system.FavTargetError, match="不存在"):
        fav_system.resolve_fav_target(private_id="404")


def test_private_fav_target_rejects_invalid_qq_and_uninitialized_store() -> None:
    import TIYA.fav_system as fav_system
    from TIYA.global_vars import PRIVATE_CHATS

    PRIVATE_CHATS.clear()
    try:
        with pytest.raises(fav_system.FavTargetError, match="不合法"):
            fav_system.resolve_fav_target(private_id="not-a-qq")

        PRIVATE_CHATS["500"] = SimpleNamespace(username="用户", fav=None)
        with pytest.raises(fav_system.FavTargetError, match="尚未初始化"):
            fav_system.resolve_fav_target(private_id="500")
    finally:
        PRIVATE_CHATS.clear()


def test_private_help_exports_only_commands_allowed_by_permission(monkeypatch) -> None:
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )

        class Host:
            username = "User"
            logger = Mock()
            management_dialog = None

            def __init__(self, user_id: str):
                self.user_id = user_id
                self.sent: list[str] = []

            async def say(self, message):
                self.sent.append(str(message))
                return [object()]

            async def system_send(self, message, *, delay=0.3):
                return await self.say(message)

        ordinary = Host("ordinary")
        ordinary_dialog = private_dialog.PrivateCommandDialog(host=ordinary)
        await ordinary_dialog.accept_message(
            _message("#help", user_id="ordinary"),
        )
        assert "#help" in ordinary.sent[-1]
        assert "#manage" not in ordinary.sent[-1]
        assert "#fav" not in ordinary.sent[-1]

        admin = Host("admin")
        admin_dialog = private_dialog.PrivateCommandDialog(host=admin)
        await admin_dialog.accept_message(_message("#help fav"))
        assert "用法: #fav" in admin.sent[-1]
        assert "--group" in admin.sent[-1]
        assert "--private" in admin.sent[-1]
        assert "#manage" not in admin.sent[-1]

    asyncio.run(run_test())


def test_private_management_messages_use_system_send(monkeypatch) -> None:
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )

        class Host:
            user_id = "admin"
            username = "Admin"
            logger = Mock()
            management_dialog = None

            def __init__(self):
                self.say_messages = []
                self.system_messages = []

            async def say(self, message):
                self.say_messages.append(message)
                return [object()]

            async def system_send(self, message, *, delay=0.3):
                self.system_messages.append((message, delay))
                return [object()]

        host = Host()
        dialog = private_dialog.PrivateCommandDialog(host=host)

        await dialog.accept_message(_message("#help"))

        assert host.say_messages == []
        assert len(host.system_messages) == 1
        assert host.system_messages[0][1] == 0.3

    asyncio.run(run_test())


def test_private_fav_dialog_is_a_thin_transfer_object() -> None:
    from TIYA.dialog.private_dialog import PrivateFavDialog
    from TIYA.fav_system import FavSystem, FavTarget

    async def run_test() -> None:
        store = _Store()

        class Host:
            user_id = "admin"
            username = "Admin"
            logger = Mock()
            management_dialog = object()
            fav_dialog = None

            def __init__(self):
                self.sent = []

            async def say(self, message):
                self.sent.append(message)
                return [object()]

            async def system_send(self, message, *, delay=0.3):
                return await self.say(message)

            async def close_fav_dialog(self, expected=None):
                return expected is self.fav_dialog

        host = Host()
        dialog = PrivateFavDialog(
            host=host,
            target=FavTarget("group", "100", "测试群(100)", store),
        )
        host.fav_dialog = dialog

        assert isinstance(dialog.system, FavSystem)
        assert await dialog.accept_message(_message("--list")) is True

    asyncio.run(run_test())


def test_private_chat_places_fav_before_management_and_destroys_it(monkeypatch) -> None:
    from TIYA.fav_system import FavTarget
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        chat = PrivateChat("admin", "Admin")
        sender = _Sender()
        monkeypatch.setattr(chat, "say", sender)
        monkeypatch.setattr(chat, "system_send", sender)
        target = FavTarget("group", "100", "测试群(100)", _Store())
        try:
            assert await chat.open_management_dialog() is True
            assert await chat.open_fav_dialog(target) is True
            fav_dialog = chat.fav_dialog
            management_dialog = chat.management_dialog

            assert fav_dialog is not None
            assert management_dialog is not None
            assert chat.dialog_list == [
                chat.command_dialog,
                fav_dialog,
                management_dialog,
            ]

            assert await chat.close_management_dialog() is True
            assert chat.fav_dialog is None
            assert chat.management_dialog is None
            assert fav_dialog.suspending
            assert management_dialog.closed
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_fav_activity_refreshes_management_timeout(monkeypatch) -> None:
    from TIYA.fav_system import FavTarget
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        chat = PrivateChat("admin", "Admin")
        sender = _Sender()
        monkeypatch.setattr(chat, "say", sender)
        monkeypatch.setattr(chat, "system_send", sender)
        try:
            await chat.open_management_dialog()
            await chat.open_fav_dialog(
                FavTarget("group", "100", "测试群(100)", _Store()),
            )
            management = chat.management_dialog
            assert management is not None
            old_timeout = management.timeout_task

            assert await chat.message_flow(_message("--list")) is True

            assert management.timeout_task is not old_timeout
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_fav_exit_destroys_only_fav_dialog(monkeypatch) -> None:
    from TIYA.fav_system import FavTarget
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        chat = PrivateChat("admin", "Admin")
        sender = _Sender()
        monkeypatch.setattr(chat, "say", sender)
        monkeypatch.setattr(chat, "system_send", sender)
        try:
            await chat.open_management_dialog()
            await chat.open_fav_dialog(
                FavTarget("group", "100", "测试群(100)", _Store()),
            )
            fav_dialog = chat.fav_dialog

            assert await chat.message_flow(_message("--exit")) is True

            assert chat.fav_dialog is None
            assert fav_dialog is not None and fav_dialog.suspending
            assert chat.management_dialog is not None
            assert chat.dialog_list == [
                chat.command_dialog,
                chat.management_dialog,
            ]
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_auto_fav_crud_preserves_many_hashes_but_rename_collapses_titles(
        monkeypatch,
        tmp_path,
) -> None:
    import TIYA.auto_fav as auto_fav_module
    from TIYA.auto_fav import AutoFav

    async def run_test() -> None:
        monkeypatch.setattr(
            auto_fav_module,
            "get_file_path_async",
            AsyncMock(side_effect=lambda hash_name: tmp_path / hash_name),
        )
        monkeypatch.setattr(auto_fav_module, "set_fav_title", lambda *_: None)
        monkeypatch.setitem(auto_fav_module.SETTING_CFG.Groups, "AutoFavExpire", 0)
        fav = AutoFav(tmp_path / "fav.json")
        await fav.add_favs([
            ("hash-a", "标题甲"),
            ("hash-a", "标题乙"),
            ("hash-b", "标题甲"),
        ])

        assert await fav.rename_fav("hash-a", "新标题") is True
        assert fav.fav_list == {
            "标题甲": {"hash-b"},
            "新标题": {"hash-a"},
        }

        assert await fav.remove_favs(["hash-a"]) == 1
        assert fav.fav_list == {"标题甲": {"hash-b"}}

    asyncio.run(run_test())
