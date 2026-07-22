from __future__ import annotations

"""Private-chat dialogs used by the rebuild runtime."""

import asyncio
import inspect
import json
import random
import time
import traceback
import weakref
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, cast

from TIYA.agent.agent_prompt import AgentPrompt
from TIYA.agent.private_chat_agent import PrivatePostReason, PrivateChatAgent, PrivateChatSpeaker
from TIYA.command import (
    CommandArgs,
    CommandError,
    CommandInvocation,
    CommandPermission,
    CommandSet,
)
from TIYA.config import (
    BASE_CFG,
    PRIVATE_CHATS_DIR,
    PROMPTS_DIR,
    SETTING_CFG,
    get_bot_uid,
    get_llm_list,
    get_private_user_config,
)
from TIYA.fav_system import (
    FavSystem,
    FavTarget,
    FavTargetError,
    resolve_fav_target,
)
from TIYA.LLM_connect import Chat, parse_llm_setting
from TIYA.global_vars import PRIVATE_CHATS, QQ_GROUPS
from TIYA import runtime_control
from TIYA import storage_cleanup
from TIYA.model.attention_speak import AttentionSpeakProbability
from TIYA.model.character import get_character
from TIYA.model.chat_notification import ChatNotification, Notice
from TIYA.model.message import PrivateMsg, SendMessage, TextMsg
from TIYA.model.persona import PrivatePersona, PrivatePersonaResult
from TIYA.utils import atomic_save_json, clean_json_text_from_llm, timestamp2text

from .base_dialog import BaseDialog

if TYPE_CHECKING:
    from TIYA.aqueue import Aqueue
    from TIYA.logger import BlockHandle
    from TIYA.private_chat import PrivateChat
    from TIYA.auto_fav import AutoFav


def _private_dialog_max_message(host: object) -> int:
    history = getattr(host, "message_history", None)
    value = getattr(history, "_max_message", None)
    if value is not None:
        return int(value)

    try:
        return int(SETTING_CFG.PrivateChat.MaxMessageHistory)
    except Exception:
        return 5000


def _private_context_epoch_file() -> Path:
    return PRIVATE_CHATS_DIR / "context_epoch.json"


def get_private_clear_context_epoch() -> float:
    epoch_file = _private_context_epoch_file()
    if not epoch_file.is_file():
        return 0.0

    try:
        content = json.loads(epoch_file.read_text(encoding="utf-8"))
        return float(content.get("clear_private_context_epoch", 0.0) or 0.0)

    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def bump_private_clear_context_epoch() -> float:
    epoch = time.time()
    atomic_save_json(
        {"clear_private_context_epoch": epoch},
        _private_context_epoch_file(),
    )
    return epoch


_PARAPHRASE_WORDS = {
    "这波",
    "确实",
    "搁这",
    "属实",
    "复读",
    "是吧",
    "这操作",
    "太狠了",
    "太骚了",
    "太真实",
}


class PrivateDialogAttrs:
    """Flatten the private-chat host interfaces needed by dialogs."""

    def __init__(self, *, host: PrivateChat, **kwargs):
        super().__init__(**kwargs)
        self.host: PrivateChat = weakref.proxy(host)

    async def say(self, message):
        return await self.host.system_send(message)

    async def system_send(self, message, *, delay=0.3):
        return await self.host.system_send(message, delay=delay)

    @property
    def user_id(self) -> str:
        return self.host.user_id

    @property
    def username(self) -> str:
        return self.host.username

    @property
    def logger(self) -> BlockHandle:
        return self.host.logger

    @property
    def management_dialog(self) -> PrivateManagementDialog | None:
        return self.host.management_dialog

    @property
    def data_path(self) -> Path:
        return self.host.data_path

    @property
    def aqueue(self) -> Aqueue:
        return self.host.aqueue

    @property
    def private_config(self):
        return self.host.private_config

    @property
    def FAV(self) -> AutoFav:
        return self.host.fav


class PrivateFavDialog(BaseDialog, PrivateDialogAttrs):
    """Thin private-chat adapter for the independent favorite system."""

    def __init__(self, *, host: PrivateChat, target: FavTarget, **kwargs):
        super().__init__(
            host=host,
            max_message=_private_dialog_max_message(host),
            **kwargs,
        )
        self.system = FavSystem(
            target=target,
            say=self.say,
            system_send=self.system_send,
            close=self._close,
        )

    async def message_flow(self, msg: PrivateMsg) -> bool:
        return await self.system.handle_message(msg)

    async def _close(self) -> bool:
        return await self.host.close_fav_dialog(expected=self)

    async def shutdown(self) -> bool:
        if self.suspending:
            return False

        self.pause()
        return True


@dataclass(frozen=True, slots=True)
class PrivateCommandContext:
    message: PrivateMsg
    invocation: CommandInvocation

    @property
    def user_id(self) -> str:
        return self.message.user_id


@dataclass(frozen=True, slots=True)
class PendingCleanup:
    plan: storage_cleanup.CleanupPlan
    owner_id: str
    expires_at: float


class PrivateCommandDialog(BaseDialog, PrivateDialogAttrs):
    """Parse private management commands before modal business dialogs."""

    COMMANDS = CommandSet(prefix="#")

    def __init__(self, *, host: PrivateChat, **kwargs):
        super().__init__(
            host=host,
            max_message=_private_dialog_max_message(host),
            **kwargs,
        )
        self._command_set = self.COMMANDS
        self._pending_cleanup: PendingCleanup | None = None
        self._cleanup_command_lock = asyncio.Lock()

    async def message_flow(self, msg: PrivateMsg) -> bool:
        command_text = self._extract_command_text(msg)
        if command_text is None:
            return False

        try:
            invocation = self._command_set.parse(command_text)
            context = PrivateCommandContext(msg, invocation)
            if not self._can_use_permission(
                    self.get_user_permission(context.user_id),
                    invocation.spec.permission,
            ):
                await self.say("权限不足喵~")
                return True

            result = invocation.spec.handler(self, context, invocation.args)
            if inspect.isawaitable(result):
                await result

        except CommandError as exc:
            await self.say(f"命令错误: {exc}")

        except Exception as exc:
            self.logger.error(f"私聊命令执行异常: {exc}")
            self.logger.debug(traceback.format_exc())
            await self.say("命令执行失败，看看日志咋回事")

        return True

    @staticmethod
    def _extract_command_text(msg: PrivateMsg) -> str | None:
        text = "".join(
            item.text for item in msg
            if isinstance(item, TextMsg)
        ).strip()
        if not text.startswith("#"):
            return None

        command_name = text.removeprefix("#").lstrip()
        if not command_name:
            return None

        if not (command_name[0].isascii() and command_name[0].isalpha()):
            return None

        return text

    @classmethod
    def get_user_permission(cls, user_id: str) -> CommandPermission:
        if user_id in cls._configured_user_ids("OwnerList"):
            return CommandPermission.OWNER

        if user_id in cls._configured_user_ids("AdminList"):
            return CommandPermission.BOT_ADMIN

        return CommandPermission.OPEN

    @staticmethod
    def _configured_user_ids(config_key: str) -> set[str]:
        values = getattr(BASE_CFG, config_key, [])
        if not isinstance(values, (list, tuple, set)):
            return set()

        return {
            str(value)
            for value in values
            if isinstance(value, (str, int)) and str(value)
        }

    @staticmethod
    def _can_use_permission(
            actual: CommandPermission,
            required: CommandPermission,
    ) -> bool:
        return actual >= required

    @COMMANDS.command(
        name="help",
        aliases=("commands",),
        summary="查看命令列表与指定命令的用法",
    )
    @COMMANDS.argument(
        name="command",
        nargs="?",
        default=None,
        _help="可选，要查看的命令名",
    )
    async def command_help(
            self,
            context: PrivateCommandContext,
            args: CommandArgs,
    ) -> None:
        help_text = self._command_set.format_help(
            args.command,
            permission=self.get_user_permission(context.user_id),
        )
        await self.say(help_text)

    @COMMANDS.command(
        name="manage",
        permission=CommandPermission.BOT_ADMIN,
        summary="进入或退出私聊管理界面",
    )
    @COMMANDS.argument(
        name="action",
        nargs="?",
        default="enter",
        choices=("enter", "exit"),
        _help="不传入时进入管理界面，也可以传入 exit",
    )
    @COMMANDS.option(
        "--exit",
        dest="exit_requested",
        action="store_true",
        default=False,
        _type=bool,
        _help="退出管理界面",
    )
    async def manage(
            self,
            context: PrivateCommandContext,
            args: CommandArgs,
    ):
        if args.exit_requested or args.action == "exit":
            changed = await self.host.close_management_dialog()
            if changed:
                await self.say("已退出管理界面")

            else:
                await self.say("管理界面尚未启动")

            return

        changed = await self.host.open_management_dialog()
        if changed:
            await self.say("已进入管理界面；使用 #manage exit 或 #manage --exit 退出")

        else:
            await self.say("管理界面已在运行，空闲计时已刷新")

    @COMMANDS.command(
        name="groups",
        permission=CommandPermission.BOT_ADMIN,
        summary="列出当前已加载的群",
    )
    async def groups(
            self,
            _context: PrivateCommandContext,
            _args: CommandArgs,
    ) -> None:
        groups = sorted(
            tuple(QQ_GROUPS.values()),
            key=lambda group: str(group.group_id),
        )
        if not groups:
            await self.say("当前没有已加载的群")
            return

        lines = [f"当前已加载 {len(groups)} 个群："]
        # noinspection PyUnresolvedReferences
        lines.extend(
            f"{index}. {group.group_name}: {group.group_id}"
            for index, group in enumerate(groups, start=1)
        )
        await self.say("\n".join(lines))

    @COMMANDS.command(
        name="reload",
        permission=CommandPermission.BOT_ADMIN,
        summary="热重载配置并发现、启动新增群",
    )
    async def reload_runtime(
            self,
            _context: PrivateCommandContext,
            _args: CommandArgs,
    ) -> None:
        try:
            result = await runtime_control.hot_reload()
        except RuntimeError as exc:
            await self.say(f"热重载失败: {exc}")
            return

        lines = [
            f"热重载完成：新增群 {len(result.started)} 个，"
            f"待配置群 {len(result.pending_config)} 个"
        ]
        if result.started:
            lines.append(f"新增群：{', '.join(result.started)}")

        if result.pending_config:
            lines.append(f"待配置群：{', '.join(result.pending_config)}")

        await self.say("\n".join(lines))

    @COMMANDS.command(
        name="restart",
        permission=CommandPermission.OWNER,
        summary="正常关闭并重启 TIYA",
    )
    async def restart_runtime(
            self,
            _context: PrivateCommandContext,
            _args: CommandArgs,
    ) -> None:
        await self.say("正在重启 TIYA...")
        runtime_control.request_restart()

    @COMMANDS.command(
        name="cleanup",
        permission=CommandPermission.OWNER,
        summary="扫描并清理过期 checkpoint 与日志",
    )
    @COMMANDS.option(
        "--confirm",
        dest="confirm",
        action="store_true",
        default=False,
        _type=bool,
        _help="确认执行上一次扫描生成的清理计划",
    )
    @COMMANDS.option(
        "--abort",
        dest="abort",
        action="store_true",
        default=False,
        _type=bool,
        _help="取消待确认的清理计划",
    )
    async def cleanup_storage(
            self,
            context: PrivateCommandContext,
            args: CommandArgs,
    ) -> None:
        if args.confirm and args.abort:
            raise CommandError("--confirm 和 --abort 不能同时使用")

        async with self._cleanup_command_lock:
            pending = self._pending_cleanup
            if args.abort:
                self._pending_cleanup = None
                if pending is None:
                    await self.say("没有待取消的清理计划")
                else:
                    await self.say("已取消清理操作")
                return

            if args.confirm:
                if pending is None:
                    await self.say("没有待确认的清理计划，请先使用 #cleanup 扫描")
                    return

                if monotonic() >= pending.expires_at:
                    self._pending_cleanup = None
                    await self.say("清理计划已过期，请重新使用 #cleanup 扫描")
                    return

                if context.user_id != pending.owner_id:
                    await self.say("该清理计划由其他 Owner 创建，不能确认")
                    return

                cleaner = storage_cleanup.get_storage_cleanup_service()
                await self.say("正在清理过期数据...")
                try:
                    result = await cleaner.execute(pending.plan)
                finally:
                    self._pending_cleanup = None

                await self.say(self._format_cleanup_result(result))
                return

            cleaner = storage_cleanup.get_storage_cleanup_service()
            self._pending_cleanup = None
            plan = await cleaner.scan()
            if plan.summary.total.files == 0:
                await self.say("当前没有可清理的过期数据")
                return

            self._pending_cleanup = PendingCleanup(
                plan=plan,
                owner_id=context.user_id,
                expires_at=monotonic() + 300.0,
            )
            await self.say(self._format_cleanup_plan(plan))

    @classmethod
    def _format_cleanup_plan(
            cls,
            plan: storage_cleanup.CleanupPlan,
    ) -> str:
        summary = plan.summary
        checkpoint_files = (
            summary.group_checkpoints.files
            + summary.private_checkpoints.files
        )
        checkpoint_bytes = (
            summary.group_checkpoints.bytes
            + summary.private_checkpoints.bytes
        )
        return (
            "发现可清理数据：\n"
            f"Agent checkpoint：{checkpoint_files} 个，"
            f"{cls._format_storage_size(checkpoint_bytes)}\n"
            f"历史日志：{summary.logs.files} 个，"
            f"{cls._format_storage_size(summary.logs.bytes)}\n"
            f"预计释放：{summary.total.files} 个文件，"
            f"{cls._format_storage_size(summary.total.bytes)}\n\n"
            "此操作不可逆，请在 5 分钟内使用 #cleanup --confirm 确认。"
        )

    @classmethod
    def _format_cleanup_result(
            cls,
            result: storage_cleanup.CleanupResult,
    ) -> str:
        return (
            f"清理完成：删除 {result.deleted.files} 个文件，"
            f"释放 {cls._format_storage_size(result.deleted.bytes)}\n"
            f"跳过 {result.skipped_files} 个，失败 {result.failed_files} 个"
        )

    @staticmethod
    def _format_storage_size(size: int) -> str:
        value = float(max(0, size))
        units = ("B", "KB", "MB", "GB", "TB")
        unit = units[0]
        for unit in units:
            if value < 1024 or unit == units[-1]:
                break
            value /= 1024

        if unit == "B":
            return f"{int(value)} B"
        return f"{value:.2f} {unit}"

    async def _get_active_main_dialog(self):
        main_dialog = self.host.main_dialog
        if main_dialog is None:
            await self.say("私聊主对话尚未启动，没有可操作的 Agent/Speaker 上下文")

        return main_dialog

    @COMMANDS.command(
        name="clear",
        permission=CommandPermission.BOT_ADMIN,
        summary="清空当前私聊 Agent 与 Speaker 上下文",
    )
    async def clear_context(
            self,
            _context: PrivateCommandContext,
            _args: CommandArgs,
    ) -> None:
        main_dialog = await self._get_active_main_dialog()
        if main_dialog is None:
            return

        main_dialog.clear_history()
        await self.say("已清空当前私聊的 Agent 与 Speaker 上下文")

    @staticmethod
    def _active_main_dialogs() -> list[PrivateMainDialog]:
        dialogs: list[PrivateMainDialog] = []
        for chat in tuple(PRIVATE_CHATS.values()):
            main_dialog = getattr(chat, "main_dialog", None)
            if main_dialog is not None:
                dialogs.append(main_dialog)

        return dialogs

    @staticmethod
    async def _save_main_dialog(main_dialog: PrivateMainDialog) -> None:
        save = getattr(main_dialog, "save", None)
        if not callable(save):
            return

        result = save()
        if inspect.isawaitable(result):
            await result

    @COMMANDS.command(
        name="clear_private_context",
        permission=CommandPermission.BOT_ADMIN,
        summary="清空私聊 Agent 与 Speaker 上下文，并刷新系统提示词",
    )
    @COMMANDS.option(
        "--active",
        dest="active",
        action="store_true",
        default=False,
        _type=bool,
        _help="清空所有当前存活且已启动的私聊上下文",
    )
    @COMMANDS.option(
        "--all",
        dest="all",
        action="store_true",
        default=False,
        _type=bool,
        _help="标记所有私聊下次启动时清空上下文并刷新系统提示词",
    )
    async def clear_private_context(
            self,
            _context: PrivateCommandContext,
            args: CommandArgs,
    ) -> None:
        """清空私聊上下文；该操作会刷新系统提示词。"""
        if args.active and args.all:
            raise CommandError("--active 和 --all 不能同时使用")

        if args.all:
            epoch = bump_private_clear_context_epoch()
            dialogs = self._active_main_dialogs()
            for main_dialog in dialogs:
                main_dialog.clear_history()
                main_dialog.mark_private_context_epoch_seen(epoch)
                await self._save_main_dialog(main_dialog)

            await self.say(
                "已标记所有私聊上下文过期；未启动私聊将在下次启动时清空上下文并刷新系统提示词。"
                f"已立即处理 {len(dialogs)} 个存活私聊。"
            )
            return

        if args.active:
            dialogs = self._active_main_dialogs()
            for main_dialog in dialogs:
                main_dialog.clear_history()

            await self.say(
                f"已清空 {len(dialogs)} 个存活私聊的 Agent 与 Speaker 上下文，并将在下一轮刷新系统提示词"
            )
            return

        main_dialog = await self._get_active_main_dialog()
        if main_dialog is None:
            return

        main_dialog.clear_history()
        await self.say("已清空当前私聊的 Agent 与 Speaker 上下文，并将在下一轮刷新系统提示词")

    @COMMANDS.command(
        name="refresh",
        permission=CommandPermission.BOT_ADMIN,
        summary="刷新当前私聊 Agent 与 Speaker 上下文",
    )
    async def refresh_context(
            self,
            _context: PrivateCommandContext,
            _args: CommandArgs,
    ) -> None:
        main_dialog = await self._get_active_main_dialog()
        if main_dialog is None:
            return
        main_dialog.refresh_context()
        await self.say("已标记当前私聊的 Agent 与 Speaker 刷新上下文")

    @COMMANDS.command(
        name="refresh_private_context",
        permission=CommandPermission.BOT_ADMIN,
        summary="刷新私聊 Agent 与 Speaker 上下文",
    )
    @COMMANDS.option(
        "--active",
        dest="active",
        action="store_true",
        default=False,
        _type=bool,
        _help="刷新所有当前存活且已启动的私聊上下文",
    )
    async def refresh_private_context(
            self,
            _context: PrivateCommandContext,
            args: CommandArgs,
    ) -> None:
        """刷新私聊上下文；该操作是压缩/整理上下文，不等同于清空。"""
        if args.active:
            dialogs = self._active_main_dialogs()
            for main_dialog in dialogs:
                main_dialog.refresh_context()

            await self.say(f"已标记 {len(dialogs)} 个存活私聊刷新 Agent 与 Speaker 上下文")
            return

        main_dialog = await self._get_active_main_dialog()
        if main_dialog is None:
            return

        main_dialog.refresh_context()
        await self.say("已标记当前私聊的 Agent 与 Speaker 刷新上下文")

    @COMMANDS.command(
        name="switch_agent_model",
        aliases=("switch_a",),
        permission=CommandPermission.BOT_ADMIN,
        summary="切换当前私聊 Agent 模型",
    )
    @COMMANDS.argument(
        name="llm_preset",
        default=None,
        nargs="?",
        _help="LLM 预设的序号或预设名",
    )
    async def switch_agent_model(
            self,
            _context: PrivateCommandContext,
            args: CommandArgs,
    ) -> None:
        main_dialog = await self._get_active_main_dialog()
        if main_dialog is None:
            return
        await self._switch_model(
            args.llm_preset,
            switcher=main_dialog.AGENT.switch_model,
            target_name="私聊 Agent",
            command_name="switch_agent_model",
        )

    @COMMANDS.command(
        name="switch_speaker_model",
        aliases=("switch_s",),
        permission=CommandPermission.BOT_ADMIN,
        summary="切换当前私聊 Speaker 模型",
    )
    @COMMANDS.argument(
        name="llm_preset",
        default=None,
        nargs="?",
        _help="LLM 预设的序号或预设名",
    )
    async def switch_speaker_model(
            self,
            _context: PrivateCommandContext,
            args: CommandArgs,
    ) -> None:
        main_dialog = await self._get_active_main_dialog()
        if main_dialog is None:
            return
        await self._switch_model(
            args.llm_preset,
            switcher=main_dialog.SPEAKER.switch_model,
            target_name="私聊 Speaker",
            command_name="switch_speaker_model",
        )

    async def _switch_model(
            self,
            llm_preset: str | None,
            *,
            switcher,
            target_name: str,
            command_name: str,
    ) -> None:
        model_list = get_llm_list()
        if llm_preset is None:
            if not model_list:
                await self.say("无可用模型")
                return
            lines = [
                f"用法: '#{command_name} <preset_name/preset_index>'",
                "目前有以下模型:",
            ]
            lines.extend(
                f"\t{index}. {preset}"
                for index, preset in enumerate(model_list, start=1)
            )
            await self.say("\n".join(lines))
            return

        preset = llm_preset
        if llm_preset.isdigit():
            index = int(llm_preset)
            if not 0 < index <= len(model_list):
                await self.say("输入的序号超出范围")
                return
            preset = model_list[index - 1]

        if not switcher(preset):
            await self.say(f"LLM 预设 [{preset}] 不合法，无法切换")
            return
        await self.say(f"{target_name} 已切换至 [{preset}]")

    @COMMANDS.command(
        name="fav",
        permission=CommandPermission.BOT_ADMIN,
        summary="进入表情管理对话",
    )
    @COMMANDS.option(
        "-g",
        "--group",
        dest="group_id",
        default=None,
        _help="目标群号",
    )
    @COMMANDS.option(
        "-p",
        "--private",
        dest="private_id",
        default=None,
        _help="目标私聊用户 QQ号",
    )
    @COMMANDS.option(
        "-c",
        "--character",
        dest="character_name",
        default=None,
        _help="目标角色名",
    )
    async def fav(
            self,
            _context: PrivateCommandContext,
            args: CommandArgs,
    ) -> None:
        if self.management_dialog is None:
            await self.say("请先使用 #manage 进入管理界面")
            return

        try:
            target = resolve_fav_target(
                group_id=args.group_id,
                character_name=args.character_name,
                private_id=args.private_id,
            )
        except FavTargetError as exc:
            await self.say(f"表情管理目标错误: {exc}")
            return

        if await self.host.open_fav_dialog(target):
            await self.say(
                f"已进入[{target.display_name}]表情管理；"
                "使用 --exit 退出"
            )
        else:
            await self.say("表情管理对话已在运行，请先使用 --exit 退出")


class PrivateManagementDialog(BaseDialog, PrivateDialogAttrs):
    """Modal shell for future private management business."""

    def __init__(
            self,
            *,
            host: PrivateChat,
            idle_timeout: float = 1800,
            **kwargs,
    ):
        if idle_timeout <= 0:
            raise ValueError("idle_timeout 必须大于 0")

        super().__init__(
            host=host,
            max_message=_private_dialog_max_message(host),
            **kwargs,
        )
        self.idle_timeout = float(idle_timeout)
        self._timeout_task: asyncio.Task[None] | None = None
        self._closed = False
        super().pause()

    async def start(self) -> bool:
        if self._closed:
            raise RuntimeError("管理对话已关闭，不能再次启动")

        was_suspending = self.suspending
        super().resume()
        self.refresh_timeout()
        return was_suspending

    async def shutdown(self) -> bool:
        if self._closed:
            return False

        self._closed = True
        super().pause()
        self._cancel_timeout()
        return True

    async def message_flow(self, msg: PrivateMsg) -> bool:
        self.refresh_timeout()
        return True

    def refresh_timeout(self):
        if self._closed:
            return

        self._cancel_timeout()
        self._timeout_task = asyncio.create_task(self._wait_for_timeout())

    def _cancel_timeout(self):
        task = self._timeout_task
        self._timeout_task = None
        if task is None or task is asyncio.current_task():
            return

        task.cancel()

    async def _wait_for_timeout(self):
        try:
            await asyncio.sleep(self.idle_timeout)
        except asyncio.CancelledError:
            return

        self._timeout_task = None
        if await self.host.close_management_dialog(expected=self):
            await self.say("管理界面已因长时间无操作自动超时退出")

    @property
    def timeout_task(self) -> asyncio.Task[None] | None:
        return self._timeout_task

    @property
    def closed(self) -> bool:
        return self._closed


class PrivateMainDialog(BaseDialog, PrivateDialogAttrs):
    """Normal one-to-one conversation backed by a controller and Speaker."""

    def __init__(self, *, host: PrivateChat, **kwargs):
        super().__init__(
            host=host,
            max_message=int(SETTING_CFG.PrivateChat.MaxMessageHistory),
            **kwargs,
        )

        # 标识符
        self.pause()
        self._last_persona_time = time.time()
        self._memory_summary_message_count = 0
        self._initiated = False
        self._memory_tip_notice: str = ""
        self.memory_save_notice: Notice | None = None
        self._last_bot_said: deque[str] = deque(maxlen=20)
        self._paraphrase = 0
        self._clear_private_context_epoch_seen = 0.0

        # 配置
        self.agent_round_limit = int(self.private_config.agent_round_limit)
        self.speaker_round_limit = int(self.private_config.speaker_round_limit)

        # 数据结构
        self.relative_news: dict[str, dict[str, list]] = {}

        # 锁
        self._persona_lock = asyncio.Lock()
        self._memory_summary_lock = asyncio.Lock()

        # 工具对象
        self._persona_cache: PrivatePersonaResult | None = None
        self.NOTICE = ChatNotification()
        character_name = str(self.private_config.default_character)
        try:
            get_character(character_name)

        except KeyError:
            self.logger.error(f"私聊人格[{character_name}]不存在，回退到[抹布]")
            character_name = "抹布"

        self.SPEAKER = PrivateChatSpeaker(
            host=self,  # type: ignore
            character=character_name,
            llm_preset=str(self.private_config.chat_model),
            logger=self.logger,
        )
        self.AGENT = PrivateChatAgent(
            host=self,  # type: ignore
            llm_preset=str(self.private_config.agent_model),
            agent_name=f"Private_{self.user_id}",
            control_timeout=float(SETTING_CFG.PrivateChat.AgentRequestTime),
            logger=self.logger,
        )
        self.PERSONA = PrivatePersona(self.user_id, self.username)
        self.SPEAK_PROBABILITY = AttentionSpeakProbability(
            min_prob=float(self.private_config.speak_rate_min),
            max_prob=float(self.private_config.speak_rate_max),
            time_window=int(self.private_config.attention_fade_out_time),
        )

        # LOOP
        self._auto_save_loop: asyncio.Task | None = None
        self._build_handle_sequence()

    def _build_handle_sequence(self):
        sequence = [
            self._commit_new_message,
            self._agent_chat,
        ]
        self._message_handle_sequence.extend(sequence)

    async def message_flow(self, msg: PrivateMsg) -> bool:
        for handle in self._message_handle_sequence:
            skip = await handle(msg)
            if skip:
                return False

        return False

    # =========================================================
    # Trigger区，统一规则，True代表跳过
    # =========================================================

    async def _commit_new_message(self, msg: PrivateMsg) -> bool:
        """提交私聊主对话消息记录"""
        await self.message_history.add_message(msg)
        self.NOTICE.notice_message_trigger()
        self._memory_summary_message_count += 1
        # noinspection PyAsyncCall
        self.aqueue.add_task(self.auto_memory_summary(), timeout=-1)
        return False

    async def _agent_chat(self, msg: PrivateMsg) -> bool:
        """整个BOT私聊逻辑的主入口，接入Agent"""
        speak_prob = self.SPEAK_PROBABILITY.get_probability()
        random_flag = random.random()
        self.logger.debug(f"当前发言概率: [{speak_prob:.2%}]")

        # 先判断强制发言
        if self.is_reply(msg):
            self.SPEAK_PROBABILITY.wake()

            # noinspection PyAsyncCall
            self.aqueue.add_task(self.AGENT.run_in_queue(PrivatePostReason.REPLY), timeout=-1)

        elif random_flag < speak_prob:
            self.SPEAK_PROBABILITY.wake()
            # noinspection PyAsyncCall
            self.aqueue.add_task(
                self.AGENT.run_in_queue(PrivatePostReason.USER_MESSAGE),
                timeout=-1,
            )

        return False

    async def speak(
        self,
        *,
        speak_prompt: str = "",
        working: str = "",
        relative_information: dict[str, str] | None = None,
        relative_memory: dict[str, str] | None = None,
        call_id: str = "",
    ) -> str | None:
        from TIYA.agent.private_chat_agent import PrivateSpeakTask

        task = PrivateSpeakTask(
            call_id=call_id or f"system_{time.time_ns()}",
            speak_prompt=speak_prompt,
            working=working,
            relative_information=dict(relative_information or {}),
            relative_memory=dict(relative_memory or {}),
        )
        await self.SPEAKER.speak(task)
        await task.event.wait()
        try:
            if task.exception is not None:
                raise task.exception

            if task.merged_to:
                return f"已合并发言任务至 {task.merged_to}"

            if task.result is None:
                return None

            result = dict(task.result)
            sent = await self.host.say(SendMessage.from_dict(result))
            successful = [
                message for message in sent
                if isinstance(message, PrivateMsg)
            ]
            if not successful:
                self.SPEAKER.mark_failure("私聊发言发送失败", retryable=True)
                raise RuntimeError("私聊发言发送失败")

            self.NOTICE.notice_message_trigger(True)
            for message in successful:
                added_message = await self.message_history.add_message(message.copy())
                text = (await added_message.to_llm(wait=False))["format_text"]
                self._last_bot_said.append(text)
                self.AGENT.said_intention[added_message.msg_id] = result["intention"]
                self.AGENT.said_review_count += 1

            if any(flag in check for check in self._last_bot_said for flag in _PARAPHRASE_WORDS):
                self._paraphrase += 1

            if self._paraphrase >= 3:
                if self._paraphrase > 3:
                    clear_paraphrase = True
                    for message in successful:
                        text = (await message.to_llm(wait=False))["format_text"]
                        if any(flag in text for flag in _PARAPHRASE_WORDS):
                            clear_paraphrase = False
                            break

                    if clear_paraphrase:
                        self._last_bot_said.clear()
                        self._paraphrase = 0

                if self._paraphrase >= 3:
                    self.NOTICE.add_notice(
                        content="Speaker 近期多次复述用户的话，这是不被允许的。通过 `speak_prompt` 明确告知并且调整发言方向",
                        title="发言规范",
                        priority=30,
                        alive_until_self_messages=1,
                        exist_ok=True,
                    )

            to_agent = result.get("to_agent", "")
            if to_agent:
                return f"# 来自私聊 Speaker 的信息\n\n{to_agent}"

            return ""

        finally:
            task.delivered.set()

    def _initiate_main_agent(self) -> None:
        initiate = self.AGENT.register_prompt(
            name="private_chat_initiate_prompt",
            path=PROMPTS_DIR / "private_chat_initiate",
            description="私聊主控初始化 prompt",
            role="user",
            default_input={
                "user_id": lambda: self.user_id,
                "username": lambda: self.username,
                "private_summary": self.get_private_summary,
                "old_messages": self.get_old_message,
                "news": self.get_private_news,
            },
        )
        self.AGENT.add_initiate_prompts(initiate)

    def _initiate_memory_tip_notification(self) -> None:
        private_setting = getattr(SETTING_CFG, "PrivateChat", None)
        messages_tip = int(getattr(private_setting, "ForceMemoryTipMessages", 100))
        said_tip = int(getattr(private_setting, "ForceMemoryTipSaid", 25))
        messages_save = int(getattr(private_setting, "ForceMemorySaveMessages", 150))
        said_save = int(getattr(private_setting, "ForceMemorySaveSaid", 30))
        memory_tip = self.NOTICE.add_notice(
            content="你已太长时间未添加记忆，尽快保存一条有效记忆",
            title="记忆通知",
            priority=45,
            alive_until_messages=-messages_tip,
            alive_until_self_messages=-said_tip,
            exist_ok=True,
        )
        self._memory_tip_notice = memory_tip.uid
        self.memory_save_notice = self.NOTICE.add_notice(
            content="你已太长时间未添加记忆，立即保存一条有效记忆",
            title="记忆通知",
            priority=40,
            alive_until_messages=-messages_save,
            alive_until_self_messages=-said_save,
            exist_ok=True,
        )

    def get_old_message(self, count: int | None = None) -> list[PrivateMsg]:
        if count is None:
            count = int(SETTING_CFG.Agent.AgentInitInputMessages)

        messages = cast(list[PrivateMsg], self.message_history.get_last_message(count))
        if messages:
            self.AGENT.last_message = messages[-1].msg_id

        return messages

    async def get_private_persona(
        self,
        immediate: bool = False,
        force: bool = False,
    ) -> PrivatePersonaResult:
        async with self._persona_lock:
            if self._persona_cache is not None and not force:
                age = time.time() - self._last_persona_time
                if age < float(SETTING_CFG.PrivateChat.PersonaCacheTime) or immediate:
                    return self._persona_cache

            if immediate:
                from TIYA.model.persona import PrivatePersonaResult

                return self._persona_cache or PrivatePersonaResult.empty(
                    user_id=self.user_id,
                    user_name=self.username,
                )

            result = await self.PERSONA.get_persona()
            self._persona_cache = result
            self._last_persona_time = time.time()
            return result

    async def get_private_summary(self, immediate: bool = False) -> str:
        persona = await self.get_private_persona(immediate=immediate)
        return json.dumps(persona.to_dict(), ensure_ascii=False, indent=2)

    @property
    def participant_aliases(self) -> dict[str, list[str]]:
        """从当前画像缓存派生私聊双方爱称，不维护第二份持久化数据。"""
        if self._persona_cache is None:
            return {}

        return {
            self._persona_cache.user.participant_id: list(
                self._persona_cache.user.alias
            ),
            self._persona_cache.bot.participant_id: list(
                self._persona_cache.bot.alias
            ),
        }

    async def set_alias(self, user_id: str, alias: str) -> None:
        """设置当前私聊用户或 BOT 的爱称，并写入唯一关系记忆桶。"""
        bot_id = str(get_bot_uid())
        if user_id not in {self.user_id, bot_id}:
            raise KeyError("指定的参与者不是当前私聊用户或 BOT，添加爱称失败")

        persona = self._persona_cache
        if persona is None:
            persona = await self.get_private_persona()
        async with self._persona_lock:
            current = self._persona_cache or persona
            if user_id == self.user_id:
                participant = current.user
                memory = f"用户[{user_id}]在当前私聊关系中的爱称新增: {alias}"

            else:
                participant = current.bot
                memory = f"BOT[{user_id}]在当前私聊关系中的爱称新增: {alias}"

            if alias in participant.alias:
                return

            response = await self.PERSONA.add_memory(memory)
            if not response.success:
                raise RuntimeError(response.message or "添加爱称记忆失败")

            participant.alias.append(alias)
            self._persona_cache = current
            self._last_persona_time = time.time()

    async def add_memory(self, memory: str) -> dict:
        response = await self.PERSONA.add_memory(memory)
        self._persona_cache = None
        return {"success": response.success, "memory": memory, "message": response.message}

    async def query_memory(self, query: str) -> dict:
        response = await self.PERSONA.query_memory(query)
        if not response.success or response.result_source == "LOCAL":
            return {
                "success": False,
                "main_answer": "",
                "sub_answer": "",
                "matches": [],
                "message": response.message or "有效结果不足",
            }

        matches = [
            match
            for match in response.matches
            if match.final_score > SETTING_CFG.Persona.MemoryConfidenceGate
        ]
        packed = []
        for match in matches:
            record = await self.PERSONA.get_memory_content(match.key)
            if record is None:
                continue

            packed.append({
                "summary": record.summary,
                "content": record.content,
                "create_at": record.created_at,
                "score": match.final_score,
            })
        main_answer = response.sub_answer or response.answer
        sub_answer = response.answer if response.sub_answer else ""
        return {
            "success": bool(packed),
            "main_answer": main_answer if packed else "",
            "sub_answer": sub_answer if packed else "",
            "matches": packed,
            "message": response.message if packed else "有效结果不足",
        }

    async def get_private_news(self) -> dict[str, list]:
        key = timestamp2text(datetime.now().timestamp(), "%y-%m-%d")
        news = self.relative_news.get(key, {})
        limit = int(SETTING_CFG.PrivateChat.MaxRelativeNews)
        return dict(list(news.items())[-limit:])

    async def get_news_by_date(self, date: str | None = None) -> dict:
        if date is None:
            target = datetime.now()

        else:
            target = datetime.fromisoformat(date)

        key = timestamp2text(target.timestamp(), "%y-%m-%d")
        return {"date": key, "news": self.relative_news.get(key, {})}

    @staticmethod
    def _normalize_memory_summary(content: dict) -> dict[str, dict]:
        """校验主体的必填事件，避免空总结或无依据画像写入关系桶。"""
        if not isinstance(content, dict):
            return {}

        def normalize_subject(_subject: str) -> dict | None:
            _section = content.get(_subject)
            if not isinstance(_section, dict):
                return None

            event = _section.get("event")
            if not isinstance(event, dict):
                return None

            valid_event = {
                title: description
                for title, description in event.items()
                if (
                    isinstance(title, str)
                    and title.strip()
                    and isinstance(description, str)
                    and description.strip()
                )
            }
            if not valid_event:
                return None

            normalized = dict(_section)
            normalized["event"] = valid_event
            return normalized

        relationship = normalize_subject("RELATIONSHIP")
        if relationship is None:
            return {}

        summary: dict[str, dict] = {}
        for subject in ("USER", "BOT"):
            section = normalize_subject(subject)
            if section is not None:
                summary[subject] = section

        summary["RELATIONSHIP"] = relationship

        return summary

    async def auto_memory_summary(self) -> None:
        async with self._memory_summary_lock:
            trigger = int(SETTING_CFG.PrivateChat.MemorySummaryMessageCount)
            if self._memory_summary_message_count < trigger:
                return

            messages = self.message_history.get_last_message(trigger)
            if len(messages) < trigger // 2:
                return

            self._memory_summary_message_count = 0
            prompt = AgentPrompt(
                name="private_memory_summary_system",
                data_path=PROMPTS_DIR / "private_memory_summary_system",
                description="私聊自动记忆总结系统提示词",
                role="system",
            )
            setting = parse_llm_setting(BASE_CFG.Agents.MemorySummaryModel)
            if setting is None:
                return

            setting.keep_alive = False
            setting.system_prompt = await prompt.get_prompt()
            tasks = [asyncio.create_task(msg.to_llm(wait=True, timeout=15)) for msg in messages]
            await asyncio.gather(*tasks, return_exceptions=True)
            payload = [
                task.result()
                for task in tasks
                if task.done() and not task.cancelled() and task.exception() is None
            ]
            try:
                async with Chat(config=setting) as chat:
                    response = await chat.ask(json.dumps(payload, ensure_ascii=False))

                if not response or not hasattr(response[0], "text"):
                    return

                content = json.loads(clean_json_text_from_llm(response[0].text))
            except Exception as exc:
                self.logger.error(f"私聊自动记忆总结失败: {exc}")
                self.logger.debug(traceback.format_exc())
                return

            summary = self._normalize_memory_summary(content)
            if summary:
                result = await self.PERSONA.add_memory(
                    json.dumps(summary, ensure_ascii=False),
                )
                if result.success:
                    self._persona_cache = None

    def to_dict(self) -> dict:
        return {
            "version": "0.1.0",
            "data": {
                "messages": self.message_history.to_dict(),
                "news": self.relative_news,
                "notification": self.NOTICE.to_dict(),
                "agent_last_message": self.AGENT.last_message,
                "speaker_last_message": self.SPEAKER.last_message,
                "persona_cache": (
                    self._persona_cache.to_dict()
                    if self._persona_cache is not None
                    else None
                ),
                "last_persona_time": self._last_persona_time,
                "memory_summary_message_count": self._memory_summary_message_count,
                "clear_private_context_epoch_seen": getattr(
                    self,
                    "_clear_private_context_epoch_seen",
                    0.0,
                ),
            },
        }

    async def load(self) -> None:
        save_path = self.data_path / "private_main_dialog_persist.json"
        if not save_path.is_file():
            return

        try:
            content = json.loads(save_path.read_text(encoding="utf-8")).get("data", {})

        except (OSError, json.JSONDecodeError):
            return

        messages = content.get("messages", {})
        if isinstance(messages, dict):
            await self.message_history.load_dict(messages)

        self.relative_news.update(content.get("news", {}))
        notification = content.get("notification", {})
        if isinstance(notification, dict) and notification:
            try:
                self.NOTICE.load_dict(notification)

            except (KeyError, TypeError, ValueError):
                self.logger.error("私聊通知系统持久化数据加载失败，已跳过")

        self.AGENT.last_message = str(content.get("agent_last_message", "") or "")
        self.SPEAKER.last_message = str(content.get("speaker_last_message", "") or "")
        persona_cache = content.get("persona_cache")
        if isinstance(persona_cache, dict):
            from TIYA.model.persona import PrivatePersonaResult

            self._persona_cache = PrivatePersonaResult.from_dict(
                persona_cache,
                user_id=self.user_id,
                user_name=self.username,
            )

        self._last_persona_time = content.get("last_persona_time", time.time())
        self._memory_summary_message_count = content.get("memory_summary_message_count", 0)
        try:
            self._clear_private_context_epoch_seen = float(
                content.get("clear_private_context_epoch_seen", 0.0) or 0.0,
            )

        except (TypeError, ValueError):
            self._clear_private_context_epoch_seen = 0.0

    async def save(self) -> None:
        atomic_save_json(self.to_dict(), self.data_path / "private_main_dialog_persist.json")

    def mark_private_context_epoch_seen(self, epoch: float) -> None:
        self._clear_private_context_epoch_seen = max(
            float(getattr(self, "_clear_private_context_epoch_seen", 0.0) or 0.0),
            float(epoch),
        )

    async def _apply_private_context_epoch(self) -> None:
        epoch = get_private_clear_context_epoch()
        if epoch <= float(getattr(self, "_clear_private_context_epoch_seen", 0.0) or 0.0):
            return

        self.clear_history()
        self.mark_private_context_epoch_seen(epoch)
        await self.save()

    async def _auto_save(self) -> None:
        try:
            while True:
                await asyncio.sleep(float(SETTING_CFG.PrivateChat.AutoSaveTime))
                await self.save()

        except asyncio.CancelledError:
            return

    async def start(self) -> None:
        if self._initiated:
            return

        await self.load()
        await self._apply_private_context_epoch()
        self.NOTICE.clear()
        self._initiate_memory_tip_notification()
        self._initiate_main_agent()
        try:
            await asyncio.gather(self.AGENT.start(), self.SPEAKER.initiate())

        except Exception:
            await asyncio.gather(
                self.AGENT.shutdown(),
                self.SPEAKER.shutdown(),
                return_exceptions=True,
            )
            raise

        self._auto_save_loop = asyncio.create_task(self._auto_save())
        self._initiated = True
        if bool(self.private_config.chat):
            self.resume()

    async def shutdown(self) -> None:
        if not self._initiated:
            self.pause()
            return

        self.pause()
        await self.save()
        if self._auto_save_loop is not None:
            self._auto_save_loop.cancel()
            await asyncio.gather(self._auto_save_loop, return_exceptions=True)

        await asyncio.gather(self.AGENT.shutdown(), self.SPEAKER.shutdown())
        self._initiated = False

    def update_config(self) -> None:
        self.host.private_config = get_private_user_config(self.user_id)
        self.agent_round_limit = int(self.private_config.agent_round_limit)
        self.speaker_round_limit = int(self.private_config.speaker_round_limit)
        self.SPEAK_PROBABILITY.min_prob = float(self.private_config.speak_rate_min)
        self.SPEAK_PROBABILITY.max_prob = float(self.private_config.speak_rate_max)
        self.SPEAK_PROBABILITY.time_window = int(self.private_config.attention_fade_out_time)
        if bool(self.private_config.chat):
            self.resume()

        else:
            self.pause()

    def refresh_context(self) -> None:
        self.AGENT.force_refresh()
        self.SPEAKER.force_refresh()

    def clear_history(self) -> None:
        self.AGENT.clear_history()
        self.SPEAKER.clear_history()

    @property
    def idle(self) -> bool:
        return (
            not self.AGENT.running
            and self.AGENT.run_queue_empty
            and not self.SPEAKER.running
        )
