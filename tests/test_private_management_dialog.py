from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from TIYA.config import ConfigMap, SETTING_CFG
from TIYA.model.message import PrivateMsg, TextMsg


@pytest.fixture(autouse=True)
def _private_dialog_settings(monkeypatch, tmp_path):
    """Keep these tests independent from config initialization order."""
    if "Groups" not in SETTING_CFG:
        SETTING_CFG["Groups"] = ConfigMap()
    if "PrivateChat" not in SETTING_CFG:
        SETTING_CFG["PrivateChat"] = ConfigMap()
    monkeypatch.setitem(SETTING_CFG.Groups, "MaxMessageHistory", 10)
    monkeypatch.setitem(
        SETTING_CFG.PrivateChat,
        "ManagementDialogTimeout",
        1,
    )
    from TIYA import private_chat

    monkeypatch.setattr(private_chat, "PRIVATE_CHATS_DIR", tmp_path)


def _message(text: str, *, user_id: str = "admin") -> PrivateMsg:
    return PrivateMsg(
        msg_id=f"message-{text}",
        user_id=user_id,
        username="User",
        nickname="",
        content=[TextMsg(text=text)],
    )


class _Host:
    def __init__(self, user_id: str = "admin"):
        self.user_id = user_id
        self.username = "User"
        self.logger = Mock()
        self.sent: list[str] = []
        self.management_dialog = None
        self.main_dialog = None

    async def say(self, message):
        self.sent.append(str(message))
        return []

    async def system_send(self, message, *, delay=0.3):
        return await self.say(message)

    async def close_management_dialog(self, expected=None) -> bool:
        dialog = self.management_dialog
        if dialog is None or expected is not None and dialog is not expected:
            return False

        self.management_dialog = None
        await dialog.shutdown()
        return True


async def _make_chat(monkeypatch, *, user_id: str = "admin"):
    from TIYA.private_chat import PrivateChat

    chat = PrivateChat(user_id, "User")
    sent: list[str] = []

    async def say(message):
        sent.append(str(message))
        return []

    monkeypatch.setattr(chat, "say", say)
    monkeypatch.setattr(chat, "system_send", say)
    return chat, sent


def test_management_dialog_is_paused_until_started() -> None:
    from TIYA.dialog.private_dialog import PrivateManagementDialog

    async def run_test() -> None:
        host = _Host()
        dialog = PrivateManagementDialog(host=host, idle_timeout=1)

        assert dialog.suspending
        assert await dialog.accept_message(_message("hello")) is False

        assert await dialog.start() is True
        assert dialog.enable
        await dialog.shutdown()

    asyncio.run(run_test())


def test_management_dialog_intercepts_and_refreshes_timeout() -> None:
    from TIYA.dialog.private_dialog import PrivateManagementDialog

    async def run_test() -> None:
        host = _Host()
        dialog = PrivateManagementDialog(host=host, idle_timeout=0.04)
        host.management_dialog = dialog
        await dialog.start()
        first_timeout_task = dialog.timeout_task

        assert await dialog.accept_message(_message("first")) is True
        assert dialog.timeout_task is not first_timeout_task

        await asyncio.sleep(0.025)
        assert await dialog.accept_message(_message("second")) is True
        await asyncio.sleep(0.025)
        assert dialog.enable

        await host.close_management_dialog(dialog)

    asyncio.run(run_test())


def test_management_dialog_timeout_asks_host_to_destroy_it() -> None:
    from TIYA.dialog.private_dialog import PrivateManagementDialog

    async def run_test() -> None:
        host = _Host()
        dialog = PrivateManagementDialog(host=host, idle_timeout=0.01)
        host.management_dialog = dialog
        await dialog.start()

        await asyncio.sleep(0.03)

        assert host.management_dialog is None
        assert dialog.closed
        assert dialog.timeout_task is None
        assert any("超时" in message for message in host.sent)

    asyncio.run(run_test())


def test_closed_management_dialog_cannot_be_reused() -> None:
    from TIYA.dialog.private_dialog import PrivateManagementDialog

    async def run_test() -> None:
        host = _Host()
        dialog = PrivateManagementDialog(host=host, idle_timeout=1)

        assert await dialog.start() is True
        assert await dialog.start() is False
        assert await dialog.shutdown() is True
        assert await dialog.shutdown() is False

        with pytest.raises(RuntimeError, match="已关闭"):
            await dialog.start()

    asyncio.run(run_test())


def test_manage_command_supports_positional_and_cli_exit(monkeypatch) -> None:
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        chat, _ = await _make_chat(monkeypatch)
        try:
            assert await chat.command_dialog.accept_message(_message("#manage")) is True
            first_dialog = chat.management_dialog
            assert first_dialog is not None

            assert await chat.command_dialog.accept_message(
                _message("#manage --exit")
            ) is True
            assert chat.management_dialog is None
            assert first_dialog.closed
            assert first_dialog not in chat.dialog_list

            assert await chat.command_dialog.accept_message(_message("#manage")) is True
            second_dialog = chat.management_dialog
            assert second_dialog is not None
            assert second_dialog is not first_dialog

            assert await chat.command_dialog.accept_message(
                _message("#manage exit")
            ) is True
            assert chat.management_dialog is None
            assert second_dialog.closed
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_groups_command_lists_current_loaded_groups(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        monkeypatch.setattr(
            private_dialog,
            "QQ_GROUPS",
            {
                "200": SimpleNamespace(group_id="200", group_name="Beta"),
                "100": SimpleNamespace(group_id="100", group_name="Alpha"),
            },
            raising=False,
        )
        host = _Host()
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(_message("#groups")) is True

        assert host.sent == [
            "当前已加载 2 个群：\n"
            "1. Alpha: 100\n"
            "2. Beta: 200"
        ]

    asyncio.run(run_test())


def test_groups_command_reports_empty_runtime(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        monkeypatch.setattr(private_dialog, "QQ_GROUPS", {}, raising=False)
        host = _Host()
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(_message("#groups")) is True
        assert host.sent == ["当前没有已加载的群"]

    asyncio.run(run_test())


def test_reload_command_uses_shared_runtime_hot_reload(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    from TIYA.runtime_control import GroupLoadResult
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        reload_runtime = AsyncMock(
            return_value=GroupLoadResult(started=["300"], pending_config=["400"])
        )
        monkeypatch.setattr(private_dialog.runtime_control, "hot_reload", reload_runtime)
        host = _Host()
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(_message("#reload")) is True

        reload_runtime.assert_awaited_once_with()
        assert host.sent == [
            "热重载完成：新增群 1 个，待配置群 1 个\n"
            "新增群：300\n"
            "待配置群：400"
        ]

    asyncio.run(run_test())


def test_reload_command_reports_runtime_failure(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        reload_runtime = AsyncMock(side_effect=RuntimeError("配置读取失败"))
        monkeypatch.setattr(private_dialog.runtime_control, "hot_reload", reload_runtime)
        host = _Host()
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(_message("#reload")) is True
        assert host.sent == ["热重载失败: 配置读取失败"]

    asyncio.run(run_test())


def test_restart_command_is_owner_only_and_requests_restart_after_reply(
    monkeypatch,
) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        host = _Host(user_id="owner")

        def request_restart() -> None:
            assert host.sent == ["正在重启 TIYA..."]

        restart_request = Mock(side_effect=request_restart)
        monkeypatch.setattr(
            private_dialog.runtime_control,
            "request_restart",
            restart_request,
        )
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(
            _message("#restart", user_id="owner")
        ) is True

        restart_request.assert_called_once_with()

    asyncio.run(run_test())


def test_restart_command_rejects_bot_admin(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        restart_request = Mock()
        monkeypatch.setattr(
            private_dialog.runtime_control,
            "request_restart",
            restart_request,
        )
        host = _Host(user_id="admin")
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(
            _message("#restart", user_id="admin")
        ) is True

        restart_request.assert_not_called()
        assert host.sent == ["权限不足喵~"]

    asyncio.run(run_test())


def test_restart_command_rejects_injected_arguments(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        restart_request = Mock()
        monkeypatch.setattr(
            private_dialog.runtime_control,
            "request_restart",
            restart_request,
        )
        host = _Host(user_id="owner")
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(
            _message(
                "#restart __import__('os').system('whoami')",
                user_id="owner",
            )
        ) is True

        restart_request.assert_not_called()
        assert len(host.sent) == 1
        assert host.sent[0].startswith("命令错误:")

    asyncio.run(run_test())


def test_cleanup_command_scans_then_executes_pending_plan(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    from TIYA.storage_cleanup import (
        CleanupBreakdown,
        CleanupResult,
        CleanupStats,
    )
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        summary = CleanupBreakdown(
            group_checkpoints=CleanupStats(files=100, bytes=4 * 1024 ** 3),
            private_checkpoints=CleanupStats(files=20, bytes=2 * 1024 ** 3),
            logs=CleanupStats(files=30, bytes=512 * 1024 ** 2),
        )
        plan = SimpleNamespace(summary=summary)
        cleaner = SimpleNamespace(
            scan=AsyncMock(return_value=plan),
            execute=AsyncMock(return_value=CleanupResult(
                deleted=summary,
                skipped_files=2,
                failed_files=1,
            )),
        )
        monkeypatch.setattr(
            private_dialog.storage_cleanup,
            "get_storage_cleanup_service",
            Mock(return_value=cleaner),
        )
        host = _Host(user_id="owner")
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(
            _message("#cleanup", user_id="owner")
        ) is True
        assert await dialog.accept_message(
            _message("#cleanup --confirm", user_id="owner")
        ) is True

        cleaner.scan.assert_awaited_once_with()
        cleaner.execute.assert_awaited_once_with(plan)
        assert "Agent checkpoint：120 个，6.00 GB" in host.sent[0]
        assert "历史日志：30 个，512.00 MB" in host.sent[0]
        assert "预计释放：150 个文件，6.50 GB" in host.sent[0]
        assert "checkpoint_" not in host.sent[0]
        assert host.sent[-2:] == [
            "正在清理过期数据...",
            "清理完成：删除 150 个文件，释放 6.50 GB\n跳过 2 个，失败 1 个",
        ]
        assert dialog._pending_cleanup is None

    asyncio.run(run_test())


def test_cleanup_command_handles_empty_scan_without_pending_plan(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    from TIYA.storage_cleanup import CleanupBreakdown
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        cleaner = SimpleNamespace(
            scan=AsyncMock(return_value=SimpleNamespace(summary=CleanupBreakdown())),
            execute=AsyncMock(),
        )
        monkeypatch.setattr(
            private_dialog.storage_cleanup,
            "get_storage_cleanup_service",
            Mock(return_value=cleaner),
        )
        host = _Host(user_id="owner")
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(
            _message("#cleanup", user_id="owner")
        ) is True

        assert host.sent == ["当前没有可清理的过期数据"]
        assert dialog._pending_cleanup is None
        cleaner.execute.assert_not_awaited()

    asyncio.run(run_test())


def test_cleanup_command_can_abort_pending_plan(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    from TIYA.storage_cleanup import CleanupBreakdown, CleanupStats
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        plan = SimpleNamespace(summary=CleanupBreakdown(
            logs=CleanupStats(files=1, bytes=100),
        ))
        cleaner = SimpleNamespace(
            scan=AsyncMock(return_value=plan),
            execute=AsyncMock(),
        )
        monkeypatch.setattr(
            private_dialog.storage_cleanup,
            "get_storage_cleanup_service",
            Mock(return_value=cleaner),
        )
        host = _Host(user_id="owner")
        dialog = PrivateCommandDialog(host=host)

        await dialog.accept_message(_message("#cleanup", user_id="owner"))
        await dialog.accept_message(_message("#cleanup --abort", user_id="owner"))

        assert host.sent[-1] == "已取消清理操作"
        assert dialog._pending_cleanup is None
        cleaner.execute.assert_not_awaited()

    asyncio.run(run_test())


def test_cleanup_command_rejects_expired_or_conflicting_confirmation(
    monkeypatch,
) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    from TIYA.storage_cleanup import CleanupBreakdown, CleanupStats
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        times = iter((100.0, 401.0, 402.0))
        monkeypatch.setattr(private_dialog, "monotonic", lambda: next(times))
        plan = SimpleNamespace(summary=CleanupBreakdown(
            logs=CleanupStats(files=1, bytes=100),
        ))
        cleaner = SimpleNamespace(
            scan=AsyncMock(return_value=plan),
            execute=AsyncMock(),
        )
        monkeypatch.setattr(
            private_dialog.storage_cleanup,
            "get_storage_cleanup_service",
            Mock(return_value=cleaner),
        )
        host = _Host(user_id="owner")
        dialog = PrivateCommandDialog(host=host)

        await dialog.accept_message(_message("#cleanup", user_id="owner"))
        await dialog.accept_message(_message("#cleanup --confirm", user_id="owner"))
        await dialog.accept_message(
            _message("#cleanup --confirm --abort", user_id="owner")
        )

        assert "已过期" in host.sent[-2]
        assert host.sent[-1].startswith("命令错误:")
        assert dialog._pending_cleanup is None
        cleaner.execute.assert_not_awaited()

    asyncio.run(run_test())


def test_cleanup_command_requires_owner_permission(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        get_cleaner = Mock()
        monkeypatch.setattr(
            private_dialog.storage_cleanup,
            "get_storage_cleanup_service",
            get_cleaner,
        )
        host = _Host(user_id="admin")
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(
            _message("#cleanup", user_id="admin")
        ) is True

        get_cleaner.assert_not_called()
        assert host.sent == ["权限不足喵~"]

    asyncio.run(run_test())


def test_cleanup_rescan_failure_discards_previous_plan(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    from TIYA.storage_cleanup import CleanupBreakdown, CleanupStats
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        plan = SimpleNamespace(summary=CleanupBreakdown(
            logs=CleanupStats(files=1, bytes=100),
        ))
        cleaner = SimpleNamespace(
            scan=AsyncMock(side_effect=(plan, RuntimeError("scan failed"))),
            execute=AsyncMock(),
        )
        monkeypatch.setattr(
            private_dialog.storage_cleanup,
            "get_storage_cleanup_service",
            Mock(return_value=cleaner),
        )
        host = _Host(user_id="owner")
        dialog = PrivateCommandDialog(host=host)

        await dialog.accept_message(_message("#cleanup", user_id="owner"))
        assert dialog._pending_cleanup is not None

        await dialog.accept_message(_message("#cleanup", user_id="owner"))

        assert dialog._pending_cleanup is None
        cleaner.execute.assert_not_awaited()
        assert host.sent[-1] == "命令执行失败，看看日志咋回事"

    asyncio.run(run_test())


def test_manage_command_requires_bot_admin(monkeypatch) -> None:
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        chat, sent = await _make_chat(monkeypatch, user_id="ordinary-user")
        try:
            assert await chat.command_dialog.accept_message(
                _message("#manage", user_id="ordinary-user")
            ) is True
            assert chat.management_dialog is None
            assert any("权限不足" in message for message in sent)
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_non_command_passes_through_command_dialog() -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog

    async def run_test() -> None:
        host = _Host()
        command_dialog = PrivateCommandDialog(host=host)

        assert await command_dialog.accept_message(_message("hello")) is False

    asyncio.run(run_test())


def test_private_clear_and_refresh_commands_use_main_dialog(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        host = _Host()
        host.main_dialog = SimpleNamespace(
            clear_history=Mock(),
            refresh_context=Mock(),
        )
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(_message("#clear")) is True
        assert await dialog.accept_message(_message("#refresh")) is True

        host.main_dialog.clear_history.assert_called_once_with()
        host.main_dialog.refresh_context.assert_called_once_with()

    asyncio.run(run_test())


def test_private_named_context_commands_support_current_active_and_all(
    monkeypatch,
    tmp_path,
) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        monkeypatch.setattr(private_dialog, "PRIVATE_CHATS_DIR", tmp_path)
        current = _Host()
        current.main_dialog = SimpleNamespace(
            clear_history=Mock(),
            refresh_context=Mock(),
            mark_private_context_epoch_seen=Mock(),
        )
        other = SimpleNamespace(
            main_dialog=SimpleNamespace(
                clear_history=Mock(),
                refresh_context=Mock(),
                mark_private_context_epoch_seen=Mock(),
            )
        )
        lightweight = SimpleNamespace(main_dialog=None)
        monkeypatch.setattr(
            private_dialog,
            "PRIVATE_CHATS",
            {"current": current, "other": other, "light": lightweight},
            raising=False,
        )
        dialog = PrivateCommandDialog(host=current)

        assert await dialog.accept_message(_message("#clear_private_context")) is True
        assert await dialog.accept_message(_message("#refresh_private_context")) is True
        assert await dialog.accept_message(_message("#clear_private_context --active")) is True
        assert await dialog.accept_message(_message("#refresh_private_context --active")) is True
        assert await dialog.accept_message(_message("#clear_private_context --all")) is True

        assert current.main_dialog.clear_history.call_count == 3
        assert current.main_dialog.refresh_context.call_count == 2
        assert other.main_dialog.clear_history.call_count == 2
        assert other.main_dialog.refresh_context.call_count == 1
        assert lightweight.main_dialog is None
        assert current.main_dialog.mark_private_context_epoch_seen.called
        assert other.main_dialog.mark_private_context_epoch_seen.called
        assert (tmp_path / "context_epoch.json").is_file()
        assert any("系统提示词" in message for message in current.sent)

    asyncio.run(run_test())


def test_refresh_private_context_rejects_all_option(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        host = _Host()
        host.main_dialog = SimpleNamespace(refresh_context=Mock())
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(_message("#refresh_private_context --all")) is True

        host.main_dialog.refresh_context.assert_not_called()
        assert any("命令错误" in message for message in host.sent)

    asyncio.run(run_test())


def test_private_model_switch_commands_support_agent_and_speaker(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        monkeypatch.setattr(private_dialog, "get_llm_list", lambda: ["local", "cloud"])
        host = _Host()
        host.main_dialog = SimpleNamespace(
            AGENT=SimpleNamespace(switch_model=Mock(return_value=True)),
            SPEAKER=SimpleNamespace(switch_model=Mock(return_value=True)),
        )
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(_message("#switch_agent_model 2")) is True
        assert await dialog.accept_message(_message("#switch_speaker_model local")) is True

        host.main_dialog.AGENT.switch_model.assert_called_once_with("cloud")
        host.main_dialog.SPEAKER.switch_model.assert_called_once_with("local")

    asyncio.run(run_test())


def test_private_context_commands_do_not_start_an_agent_for_lightweight_chat(
    monkeypatch,
) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog
    import TIYA.dialog.private_dialog as private_dialog

    async def run_test() -> None:
        monkeypatch.setattr(
            private_dialog,
            "BASE_CFG",
            SimpleNamespace(OwnerList=["owner"], AdminList=["admin"]),
        )
        host = _Host()
        dialog = PrivateCommandDialog(host=host)

        assert await dialog.accept_message(_message("#clear")) is True

        assert host.main_dialog is None
        assert any("尚未启动" in message for message in host.sent)

    asyncio.run(run_test())


def test_private_chat_places_notice_trigger_before_dialogs() -> None:
    from TIYA.private_chat import NoticeResponse, PrivateChat

    async def run_test() -> None:
        calls: list[str] = []

        class _Dialog:
            async def accept_message(self, message: PrivateMsg) -> bool:
                calls.append(message.msg_id)
                return False

        chat = object.__new__(PrivateChat)
        response = NoticeResponse(
            echo="notice",
            user_id="admin",
            message_id="sent",
            intercept=True,
        )
        chat._notice_queue = deque([response])
        chat._notice_responses = {response.echo: response}
        chat._notice_expiry_tasks = {}
        chat.dialog_list = [_Dialog()]

        assert await chat.message_flow(_message("answer")) is True
        assert response.done
        assert calls == []

    asyncio.run(run_test())


def test_private_chat_uses_dialog_list_order() -> None:
    from TIYA.private_chat import PrivateChat

    async def run_test() -> None:
        calls: list[str] = []

        class _Dialog:
            def __init__(self, name: str, intercept: bool = False):
                self.name = name
                self.intercept = intercept

            async def accept_message(self, message: PrivateMsg) -> bool:
                calls.append(self.name)
                return self.intercept

        chat = object.__new__(PrivateChat)
        chat._notice_queue = deque()
        chat._notice_responses = {}
        chat._notice_expiry_tasks = {}
        chat.dialog_list = [
            _Dialog("command"),
            _Dialog("management", intercept=True),
            _Dialog("trailing"),
        ]

        assert await chat.message_flow(_message("hello")) is True
        assert calls == ["command", "management"]

    asyncio.run(run_test())


def test_private_chat_inserts_management_dialog_only_while_open(monkeypatch) -> None:
    from TIYA.dialog.private_dialog import PrivateCommandDialog, PrivateManagementDialog

    async def run_test() -> None:
        chat, _ = await _make_chat(monkeypatch)
        trailing = Mock()
        chat.dialog_list.append(trailing)
        try:
            assert chat.management_dialog is None
            assert chat.dialog_list == [chat.command_dialog, trailing]
            assert isinstance(chat.command_dialog, PrivateCommandDialog)

            assert await chat.open_management_dialog() is True
            dialog = chat.management_dialog
            assert isinstance(dialog, PrivateManagementDialog)
            assert chat.dialog_list == [chat.command_dialog, dialog, trailing]

            assert await chat.close_management_dialog() is True
            assert chat.management_dialog is None
            assert chat.dialog_list == [chat.command_dialog, trailing]
            assert dialog.closed
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_reopening_management_dialog_does_not_inherit_context(monkeypatch) -> None:
    async def run_test() -> None:
        chat, _ = await _make_chat(monkeypatch)
        try:
            await chat.open_management_dialog()
            first = chat.management_dialog
            assert first is not None
            first.context_marker = "old context"

            await chat.close_management_dialog()
            await chat.open_management_dialog()
            second = chat.management_dialog

            assert second is not None
            assert second is not first
            assert not hasattr(second, "context_marker")
            assert len(second.message_history) == 0
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_opening_existing_management_dialog_only_refreshes_timeout(monkeypatch) -> None:
    async def run_test() -> None:
        chat, _ = await _make_chat(monkeypatch)
        try:
            assert await chat.open_management_dialog() is True
            dialog = chat.management_dialog
            assert dialog is not None
            first_timeout_task = dialog.timeout_task

            assert await chat.open_management_dialog() is False
            assert chat.management_dialog is dialog
            assert dialog.timeout_task is not first_timeout_task
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_stale_dialog_cannot_close_a_new_management_dialog(monkeypatch) -> None:
    async def run_test() -> None:
        chat, _ = await _make_chat(monkeypatch)
        try:
            await chat.open_management_dialog()
            first = chat.management_dialog
            await chat.close_management_dialog()
            await chat.open_management_dialog()
            second = chat.management_dialog

            assert first is not None
            assert second is not None
            assert await chat.close_management_dialog(expected=first) is False
            assert chat.management_dialog is second
            assert second in chat.dialog_list
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_automatic_timeout_removes_and_destroys_management_dialog(monkeypatch) -> None:
    async def run_test() -> None:
        chat, sent = await _make_chat(monkeypatch)
        chat._management_dialog_timeout = 0.01
        try:
            await chat.open_management_dialog()
            dialog = chat.management_dialog
            assert dialog is not None

            await asyncio.sleep(0.03)

            assert chat.management_dialog is None
            assert dialog not in chat.dialog_list
            assert dialog.closed
            assert any("超时" in message for message in sent)
        finally:
            await chat.shutdown()

    asyncio.run(run_test())


def test_private_chat_shutdown_destroys_active_management_dialog(monkeypatch) -> None:
    async def run_test() -> None:
        chat, _ = await _make_chat(monkeypatch)
        await chat.open_management_dialog()
        dialog = chat.management_dialog
        assert dialog is not None

        await chat.shutdown()

        assert chat.management_dialog is None
        assert dialog.closed
        assert dialog not in chat.dialog_list

    asyncio.run(run_test())
