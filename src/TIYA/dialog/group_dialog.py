from __future__ import annotations
"""
group_dialog.py
TIYA_BOT的群组相关对话对象
"""
__version__ = "0.3.0"

import asyncio
import inspect
import traceback
import time
import weakref
import random
import json
import TIYA.api.group_api as api

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, TypedDict, Any, Callable, Coroutine
from collections import defaultdict, deque
from itertools import islice
from pathlib import Path
from datetime import datetime
from io import BytesIO
from functools import partial

from .base_dialog import BaseDialog
from TIYA.command import *
from TIYA.global_vars import QQ_GROUPS
from TIYA.time_id import next_time_id
from TIYA.setu.setu import (
    Setu,
    PixivPayload,
    SetuSendReceipt,
    Illustration,
    Illust,
    POST_PROCESS_HASH_MAPPING,
)
from TIYA.setu.ocr import detect_advertisement
from TIYA.file_cache import add_file_async, get_file_metadata_async, get_file_path_async
from TIYA.executor import GLOBAL_EXECUTOR
from TIYA.model.message import (
    BaseSendItem,
    GroupMsg,
    ReplyMsg,
    ImgMsg,
    SetuMsg,
    SendMessage,
    TextMsg,
    SendReply,
    SendText,
    SendImage
)
from TIYA.model.attention_speak import AttentionSpeakProbability
from TIYA.model.chat_notification import ChatNotification
from TIYA.model.character import get_group_character_manager
from TIYA.relatedness import BotCommunity, GroupRelatedness, MemberAffinityScore
from TIYA.auto_fav import AutoFav
from TIYA.model.persona import (
    GroupPersona,
    GroupMemberPersona,
    GroupPersonaResult,
    MemberPersonaResult,
    GROUP_ACTION_TYPE_MAPPING,
)
from TIYA.LLM_connect import parse_llm_setting, Chat
from TIYA.agent.group_chat_agent import GroupChatAgent, GroupChatSpeaker, PostReason
from TIYA.agent.agent_prompt import AgentPrompt
from TIYA.re_filter import _pixiv_link_filter
from TIYA.config import BASE_CFG, PROMPTS_DIR, SETTING_CFG, CONFIG_OBSERVER, get_llm_list, config_update
from TIYA.utils import (
    DecoratedDict,
    timestamp2text,
    atomic_save_json,
    compress_gif,
    compress_image,
    postprocess_image,
    clean_json_text_from_llm
)
from TIYA.logger import BlockHandle

if TYPE_CHECKING:
    from TIYA.qq_group import QQGroup, MemberList, Member
    from TIYA.aqueue import Aqueue


_SETU_COMMAND = CommandSet(prefix='#')
_SETU_GLOBAL_PERMISSION = CommandPermission.BOT_ADMIN
_PARAPHRASE_WORDS = {"这波", "确实", "搁这", "属实", "是吧", "这操作", "太狠了", "太骚了", "太真实", "太草了", "属于是"}


class SetuHelp(Exception):
    """并非错误，而是返回帮助"""
    ...


class GroupDialogAttrs:
    """存放群对话特有的属性，以及平铺宿主的接口"""
    def __init__(self, *, host: QQGroup, **kwargs):
        super().__init__(**kwargs)
        self.host: QQGroup = weakref.proxy(host)

    async def say(self, message: GroupMsg | SendMessage | BaseSendItem | str) -> list[GroupMsg | None]:
        """统一转换为 message chain 发送，并把成功的结果包装成 GroupMsg"""
        return await self.host.say(message)

    async def upload_group_file(
            self,
            file: str,
            filename: str,
            folder_id: str = '/'
    ) -> tuple[bool, GroupMsg | None]:
        return await self.host.upload_group_file(file, filename, folder_id)

    async def update_member(self, force: bool = False):
        return await self.host.update_member(force)

    async def mute(self, user_id: str, duration: float = None):
        return await self.host.mute(user_id, duration)

    async def unmute(self, user_id: str):
        return await self.host.unmute(user_id)

    async def unmute_all(self):
        return await self.host.unmute_all()

    def is_muted(self, user_id: str) -> bool:
        return self.host.is_muted(user_id)

    @property
    def group_id(self) -> str:
        return self.host.group_id

    @property
    def group_name(self) -> str:
        return self.host.group_name

    @property
    def group_config(self) -> DecoratedDict:
        return self.host.group_config

    @property
    def data_path(self) -> Path:
        return self.host.data_path

    @property
    def member_list(self) -> MemberList | None:
        return self.host.member_list

    @property
    def aqueue(self) -> Aqueue:
        return self.host.aqueue

    @property
    def logger(self) -> BlockHandle:
        return self.host.logger

    @property
    def FAV(self) -> AutoFav:
        return self.host.fav

    @property
    def main_dialog(self) -> GroupMainDialog:
        return self.host.main_dialog


@dataclass(frozen=True, slots=True)
class GroupCommandContext:
    message: GroupMsg
    invocation: CommandInvocation

    @property
    def user_id(self) -> str:
        return self.message.user_id

    @property
    def reply(self) -> ReplyMsg | None:
        return self.message.reply
    
    @property
    def msg_id(self) -> str:
        return self.message.msg_id


class GroupCommandDialog(BaseDialog, GroupDialogAttrs):
    COMMANDS = CommandSet(prefix="#")

    def __init__(self, *, host: QQGroup, **kwargs):
        super().__init__(host=host, **kwargs)
        self._command_set = self.COMMANDS

        # 数据结构
        self._pending_delete_memory: dict[str, Any] = {}
        self._pending_reset_memory = {}

        # 标识符
        self._remove_memory_processing = False

        # 锁
        self._remove_memory_lock = asyncio.Lock()

    async def message_flow(self, msg: GroupMsg) -> bool:
        """Return True when command-like input should stop propagating."""
        command_text = self._extract_command_text(msg)
        if command_text is None:
            return False

        try:
            invocation = self._command_set.parse(command_text)
            context = GroupCommandContext(message=msg, invocation=invocation)
            if not await self.check_command_permission(context):
                await self.say(SendMessage(
                    SendReply(msg.msg_id),
                    SendText("权限不足喵~")
                ))
                return True

            result = invocation.spec.handler(self, context, invocation.args)
            if inspect.isawaitable(result):
                await result

        except CommandError as exc:
            await self.say(f"命令错误: {exc}")

        except Exception as exc:
            self.logger.error(f"命令执行异常: {exc}")
            self.logger.debug(traceback.format_exc())
            await self.say(SendMessage(
                SendReply(msg.msg_id),
                SendText("命令执行失败，看看日志咋回事")
            ))

        return True

    async def check_command_permission(self, context: GroupCommandContext) -> bool:
        required_permission = context.invocation.spec.permission
        if required_permission is CommandPermission.OPEN:
            return True

        actual_permission = await self.get_user_permission(context.user_id)
        return self.can_use_permission(actual_permission, required_permission)

    async def get_user_permission(self, user_id: str) -> CommandPermission:
        """Return the highest command permission available to this group user."""
        owner_ids = self._get_configured_user_ids("OwnerList")
        if user_id in owner_ids:
            return CommandPermission.OWNER

        if user_id in self._get_configured_user_ids("AdminList"):
            return CommandPermission.BOT_ADMIN

        if self.member_list is None:
            await self.update_member()

        member_list = self.member_list
        if member_list is None:
            return CommandPermission.OPEN

        group_owner = member_list.get_owner
        is_group_owner = (
            group_owner is not None
            and group_owner.user_id == user_id
        )
        is_group_admin = any(
            member.user_id == user_id
            for member in member_list.get_admins
        )
        if is_group_owner or is_group_admin:
            return CommandPermission.GROUP_ADMIN

        return CommandPermission.OPEN

    @staticmethod
    def can_use_permission(
            actual_permission: CommandPermission,
            required_permission: CommandPermission
    ) -> bool:
        return actual_permission >= required_permission

    @staticmethod
    def _get_configured_user_ids(config_key: str) -> set[str]:
        values = getattr(BASE_CFG, config_key, [])
        if not isinstance(values, list):
            return set()

        return {
            str(value)
            for value in values
            if isinstance(value, (str, int)) and str(value)
        }

    def _extract_command_text(self, msg: GroupMsg) -> str | None:
        text = "".join(item.text for item in msg if isinstance(item, TextMsg)).strip()
        if not text.startswith(self._command_set.prefix):
            return None

        check_alpha = ''.join(text.split(self._command_set.prefix)[1:]).strip()
        if not check_alpha:
            return None

        if not (check_alpha[0].isalpha() and check_alpha[0].isascii()):
            return None

        return text

    @COMMANDS.command(
        name="help",
        aliases=("commands",),
    )
    @COMMANDS.argument(
        name="command",
        nargs='?',
        default=None,
        _help="查看指定命令的使用方式"
    )
    async def command_help(self, context: GroupCommandContext, args: CommandArgs):
        """获取命令列表与使用方式"""
        permission = await self.get_user_permission(context.user_id)
        help_text = self._command_set.format_help(
            args.command,
            permission=permission
        )
        await self.say(help_text)

    @COMMANDS.command(
        name="ai",
        aliases=("switch_ai",),
        permission=CommandPermission.GROUP_ADMIN
    )
    async def switch_ai(self, context: GroupCommandContext, args: CommandArgs):
        """开关群聊主Agent"""
        if self.main_dialog.suspending:
            self.main_dialog.resume()
            self.group_config.chat = True
            config_update()
            await self.host.say("来咯！")

        else:
            self.main_dialog.pause()
            self.group_config.chat = False
            config_update()
            await self.host.say("走了喵")

    @COMMANDS.command(
        name="clear",
        permission=CommandPermission.BOT_ADMIN
    )
    @COMMANDS.option(
        "--all", "-a",
        dest="all",
        action="store_true",
        default=False,
        _type=bool,

    )
    @COMMANDS.option(
        "--group", "-g",
        dest="group",
        default=""
    )
    async def clear(self, context: GroupCommandContext, args: CommandArgs):
        """强制清除当前上下文"""
        if args.all:
            for group in QQ_GROUPS.values():
                group.main_dialog.AGENT.clear_history()

            await self.say(SendMessage(SendReply(context.msg_id), SendText(text="已清除所有群上下文")))

        elif args.group:
            group_id = args.group
            if group_id not in QQ_GROUPS:
                await self.say(SendMessage(SendReply(context.msg_id), SendText(text=f"群 [{group_id}] 不存在")))

            else:
                QQ_GROUPS[group_id].main_dialog.AGENT.clear_history()
                await self.say(SendMessage(SendReply(context.msg_id), SendText(text=f"已清除群 [{group_id}] 的上下文")))

        else:
            self.main_dialog.AGENT.clear_history()
            await self.say(SendMessage(SendReply(context.msg_id), SendText(text="已清除上下文")))

    @COMMANDS.command(
        name="refresh",
        permission=CommandPermission.BOT_ADMIN
    )
    @COMMANDS.option(
        "--all", "-a",
        dest="all",
        action="store_true",
        default=False,
        _type=bool,

    )
    @COMMANDS.option(
        "--group", "-g",
        dest="group",
        default=""
    )
    async def refresh_context(self, context: GroupCommandContext, args: CommandArgs):
        """强制刷新一次上下文"""
        if args.all:
            for group in QQ_GROUPS.values():
                group.main_dialog.refresh_context()

            await self.say(SendMessage(SendReply(context.msg_id), SendText(text="已刷新所有群上下文")))

        elif args.group:
            group_id = args.group
            if group_id not in QQ_GROUPS:
                await self.say(SendMessage(SendReply(context.msg_id), SendText(text=f"群 [{group_id}] 不存在")))

            else:
                QQ_GROUPS[group_id].main_dialog.refresh_context()
                await self.say(
                    SendMessage(SendReply(context.msg_id), SendText(text=f"已刷新群 [{group_id}] 的上下文")))

        else:
            self.main_dialog.refresh_context()
            await self.say(SendMessage(SendReply(context.msg_id), SendText(text="已刷新上下文")))

    @COMMANDS.command(
        name="speak_delay",
        aliases=("delay",),
        permission=CommandPermission.BOT_ADMIN
    )
    async def speak_delay(self, context: GroupCommandContext, args: CommandArgs):
        delay = self.main_dialog.get_average_speak_delay()
        if delay is None:
            delay = "暂时无法统计"

        else:
            delay = f"{delay:.2f}"

        await self.say(
            SendMessage(SendReply(context.msg_id), SendText(text=f"当前上下文窗口发言平均延迟: [{delay}]")))

    @COMMANDS.command(
        name="stop",
        aliases=("stop_agent",),
        permission=CommandPermission.GROUP_ADMIN
    )
    async def stop_agent(self, context: GroupCommandContext, args: CommandArgs):
        """中断本次Agent任务，但不中断发言"""
        await self.main_dialog.AGENT.interrupt_control()
        await self.say(SendMessage(SendReply(context.msg_id), SendText(text="已挂起一次Agent")))

    @COMMANDS.command(
        name="muted",
    )
    async def get_muted(self, context: GroupCommandContext, args: CommandArgs):
        """获取拉黑列表"""
        muted_set = self.host.muted_list
        if muted_set:
            lines = ["以下群友已被拉黑喵: "]
            for index, member in enumerate(muted_set, start=1):
                lines.append(f"\t{index}. {member.user_id}: {member.nickname or member.username}")

            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText('\n'.join(lines))
            ))

        else:
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText("还没有人被拉黑喵")
            ))

    @COMMANDS.command(
        name="unmute",
        aliases=("unban",),
        permission=CommandPermission.GROUP_ADMIN
    )
    @COMMANDS.argument(
        name="user_id",
        default=None,
        _help="要解除拉黑的QQ号，可以传 'all'"
    )
    async def unmute_user(self, context: GroupCommandContext, args: CommandArgs):
        """解除拉黑某人，可输入 'all'"""
        if context.message.user_id == args.user_id:
            return

        if args.user_id.lower() == "all":
            await self.unmute_all()
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText("已解除所有人的拉黑")
            ))
            return

        if self.member_list is None:
            await self.host.update_member()

        if self.member_list is None:
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText(f"没找到 [{args.user_id}] 这个人")
            ))

        elif args.user_id not in self.member_list:
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText(f"没找到 [{args.user_id}] 这个人")
            ))

        else:
            member = self.member_list.get_member(user_id=args.user_id)[0]
            await self.unmute(member.user_id)
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText(f"已解除对 [{member.user_id}: {member.nickname or member.username}] 的拉黑")
            ))

    @COMMANDS.command(
        name="switch_agent_model",
        aliases=("switch_a",),
        permission=CommandPermission.BOT_ADMIN
    )
    @COMMANDS.argument(
        name="llm_preset",
        default=None,
        nargs='?',
        _help="LLM预设的序号或预设名"
    )
    async def switch_agent_model(self, context: GroupCommandContext, args: CommandArgs):
        """切换群聊 Agent 的模型，运行中切换有风险，若无限报错则需要清空一次上下文"""
        llm_preset: str = args.llm_preset
        model_list = get_llm_list()
        if llm_preset is None:
            if not model_list:
                await self.say(SendMessage(
                    SendReply(context.msg_id),
                    SendText("无可用模型")
                ))
                return

            lines = [
                "用法: '#switch_agent_model <preset_name/preset_index>'\n",
                "目前有以下模型: "
            ]
            for index, preset in enumerate(model_list, start=1):
                lines.append(f"\t{index}. {preset}")

            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText('\n'.join(lines))
            ))

        else:
            if llm_preset.isdigit():
                llm_index = int(llm_preset)
                if 0 < llm_index <= len(model_list):
                    preset = model_list[llm_index - 1]
                    success = self.main_dialog.AGENT.switch_model(preset)
                    if not success:
                        await self.say(SendMessage(
                            SendReply(context.msg_id),
                            SendText(f"LLM 预设 [{preset}] 不合法，无法切换")
                        ))

                    else:
                        await self.say(SendMessage(
                            SendReply(context.msg_id),
                            SendText(f"群聊 Agent 已切换至 [{preset}]"))
                        )

                else:
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("输入的序号超出范围"))
                    )

            else:
                success = self.main_dialog.AGENT.switch_model(llm_preset)
                if not success:
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText(f"LLM 预设 [{llm_preset}] 不合法，无法切换")
                    ))

                else:
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText(f"群聊 Agent 已切换至 [{llm_preset}]"))
                    )

    @COMMANDS.command(
        name="switch_speaker_model",
        aliases=("switch_s",),
        permission=CommandPermission.BOT_ADMIN
    )
    @COMMANDS.argument(
        name="llm_preset",
        default=None,
        nargs='?',
        _help="LLM预设的序号或预设名"
    )
    async def switch_speaker_model(self, context: GroupCommandContext, args: CommandArgs):
        """切换发言 LLM 的模型，运行中切换有风险，若无限报错则需要清空一次上下文"""
        llm_preset: str = args.llm_preset
        model_list = get_llm_list()
        if llm_preset is None:
            if not model_list:
                await self.say(SendMessage(
                    SendReply(context.msg_id),
                    SendText("无可用模型")
                ))
                return

            lines = [
                "用法: '#switch_speaker_model <preset_name/preset_index>'\n",
                "目前有以下模型: "
            ]
            for index, preset in enumerate(model_list, start=1):
                lines.append(f"\t{index}. {preset}")

            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText('\n'.join(lines))
            ))

        else:
            if llm_preset.isdigit():
                llm_index = int(llm_preset)
                if 0 < llm_index <= len(model_list):
                    preset = model_list[llm_index - 1]
                    success = self.main_dialog.SPEAKER.switch_model(preset)
                    if not success:
                        await self.say(SendMessage(
                            SendReply(context.msg_id),
                            SendText(f"LLM 预设 [{preset}] 不合法，无法切换")
                        ))

                    else:
                        await self.say(SendMessage(
                            SendReply(context.msg_id),
                            SendText(f"群聊 Speaker 已切换至 [{preset}]"))
                        )

                else:
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("输入的序号超出范围"))
                    )

            else:
                success = self.main_dialog.SPEAKER.switch_model(llm_preset)
                if not success:
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText(f"LLM 预设 [{llm_preset}] 不合法，无法切换")
                    ))

                else:
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText(f"群聊 Speaker 已切换至 [{llm_preset}]"))
                    )

    @COMMANDS.command(
        name="setu",
    )
    @COMMANDS.argument(
        name="setu_args",
        default="",
        raw_remainder=True,
        _help="传给 setu 的参数，可使用 --help 获取用法"
    )
    async def get_setu(self, context: GroupCommandContext, args: CommandArgs):
        """发送色图"""
        timeout = SETTING_CFG.SETU.SendSETUTaskTimeout
        try:
            task = self.aqueue.add_task(self.main_dialog.setu(args.setu_args), timeout)
            await task.wait()
            if task.exception:
                raise task.exception[0]

        except SetuHelp as help_message:
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText(str(help_message))
            ))

        except Exception as E:
            self.logger.error(f"色图模块出错: {E}")
            self.logger.debug(traceback.format_exc())
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText(f"色图模块出错: {E}")
            ))

    @COMMANDS.command(
        name="add_memory",
        aliases=("add_m",),
        permission=CommandPermission.BOT_ADMIN
    )
    @COMMANDS.option(
        "--target", "-t",
        dest="target",
        default="",
        required=True,
        _help="指定记忆桶，可填 'GROUP' | 'BOT' | QQ号"
    )
    @COMMANDS.option(
        "--memory", "-m",
        dest="memory",
        default="",
        required=True,
        _help="要添加的记忆内容"
    )
    async def add_memory(self, context: GroupCommandContext, args: CommandArgs):
        """手动添加一条记忆"""
        await self.say(SendMessage(
            SendReply(context.msg_id),
            SendText("正在添加记忆...")
        ))
        result = await self.main_dialog.add_memory(target=args.target, memory=args.memory)
        if result["success"]:
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText(f"已成功添加记忆: {args.memory}")
            ))

        else:
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText(f"添加记忆失败: {result['message']}")
            ))

    @COMMANDS.command(
        name="remove_memory",
        permission=CommandPermission.BOT_ADMIN
    )
    @COMMANDS.option(
        "--target", "-t",
        dest="target",
        default="",
        _help="指定记忆桶，可填 'GROUP' | 'BOT' | QQ号"
    )
    @COMMANDS.option(
        "--query", "-q",
        dest="query",
        default="",
        _help="要删除的记忆内容"
    )
    @COMMANDS.option(
        "--count", "-c",
        dest="count",
        _type=int,
        default=0,
        _help="要删除的记忆单片数量"
    )
    @COMMANDS.option(
        "--threshold", "-th",
        dest="threshold",
        _type=float,
        default=0.5,
        _help="筛选可删除的阈值"
    )
    @COMMANDS.option(
        "--confirm",
        dest="confirm",
        action="store_true",
        default=False,
        _help="确认删除操作"
    )
    @COMMANDS.option(
        "--abort",
        dest="abort",
        action="store_true",
        default=False,
        _help="放弃删除操作"
    )
    async def remove_memory(self, context: GroupCommandContext, args: CommandArgs):
        """删除指定记忆"""
        timeout = SETTING_CFG.Groups.PendingDeleteMemoryExpire
        if self._remove_memory_lock.locked():
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText("记忆系统运算中，请等待数据处理完成后再使用本指令")
            ))
            return

        if self._remove_memory_processing:
            if self._pending_reset_memory.get("create_time", 0) + timeout >= time.time():
                await self.say(SendMessage(
                    SendReply(context.msg_id),
                    SendText("有缓存中的记忆重置操作，请等待其他操作完成或使用 '#reset_memory --abort' 取消")
                ))
                return

            elif self._pending_delete_memory.get("create_time", 0) + timeout >= time.time():
                if context.message.user_id != self._pending_delete_memory.get("user_id", ""):
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("有其他管理员正在进行的记忆删除操作，请等待他人操作完成")
                    ))
                    return

                if args.abort:
                    async with self._remove_memory_lock:
                        self._pending_delete_memory.clear()
                        await self.say(SendMessage(
                            SendReply(context.msg_id),
                            SendText("已放弃删除操作")
                        ))
                        self._remove_memory_processing = False
                        return

                if args.confirm:
                    async with self._remove_memory_lock:
                        await self.say(SendMessage(
                            SendReply(context.msg_id),
                            SendText("正在删除记忆...")
                        ))
                        await self.main_dialog.remove_memory(self._pending_delete_memory.copy())
                        self._pending_delete_memory.clear()
                        self._remove_memory_processing = False
                        return

            else:
                async with self._remove_memory_lock:
                    self._remove_memory_processing = False

        async with self._remove_memory_lock:
            if args.abort:
                if self._pending_delete_memory:
                    self._remove_memory_processing = True
                    self._pending_delete_memory.clear()
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("已放弃删除操作")
                    ))
                    self._remove_memory_processing = False

                else:
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("无可放弃的操作")
                    ))

                return

            if args.confirm:
                if self._pending_delete_memory.get("create_time", 0) + timeout >= time.time():
                    self._remove_memory_processing = True
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("正在删除记忆...")
                    ))
                    await self.main_dialog.remove_memory(self._pending_delete_memory.copy())
                    self._pending_delete_memory.clear()
                    self._remove_memory_processing = False

                else:
                    self._pending_delete_memory.clear()
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("未获取要删除的记忆列表，请先获取")
                    ))

                return

            if not (args.target.strip() and args.query.strip() and args.count):
                await self.say(SendMessage(
                    SendReply(context.msg_id),
                    SendText("'--target', '--query', '--count' 为必填字段")
                ))
                return

            if args.threshold < 0.2 or args.threshold > 1:
                await self.say(SendMessage(
                    SendReply(context.msg_id),
                    SendText("'--threshold' 必须在 0.2~1.0 之间")
                ))
                return

            if args.count < 10 or args.count > 500:
                await self.say(SendMessage(
                    SendReply(context.msg_id),
                    SendText("'--count' 必须在 10~500 之间")
                ))
                return

            self._remove_memory_processing = True
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText("正在获取要删除的记忆列表...")
            ))
            try:
                result = await self.main_dialog.get_remove_memory_list(
                    target=args.target,
                    query=args.query,
                    count=args.count,
                    threshold=args.threshold
                )
                
            except Exception as E:
                await self.say(SendMessage(
                    SendReply(context.msg_id),
                    SendText(str(E))
                ))
                self.logger.error(E)
                self.logger.debug(traceback.format_exc())
                self._remove_memory_processing = False
                return 
            
            if not result or not result.get("memories", {}):
                await self.say(SendMessage(
                    SendReply(context.msg_id),
                    SendText("没有找到能删除的记忆")
                ))
                self._remove_memory_processing = False
                return
            
            result["user_id"] = context.user_id
            self._pending_delete_memory = result
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText(f"符合筛选的删除条件的记忆数量: {len(result['memories'])}\n"
                         f"此操作不可逆，确认无误后，附带 '--confirm' 关键字确认删除操作")
            ))
            return

    @COMMANDS.command(
        name="reset_memory",
        permission=CommandPermission.BOT_ADMIN
    )
    @COMMANDS.option(
        "--confirm",
        dest="confirm",
        action="store_true",
        default=False,
        _help="确认删除操作"
    )
    @COMMANDS.option(
        "--abort",
        dest="abort",
        action="store_true",
        default=False,
        _help="放弃重置操作"
    )
    async def reset_memory(self, context: GroupCommandContext, args: CommandArgs):
        """重置整个记忆库"""
        timeout = SETTING_CFG.Groups.PendingDeleteMemoryExpire
        if self._remove_memory_lock.locked():
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText("记忆系统运算中，请等待数据处理完成后再使用本指令")
            ))
            return

        if self._remove_memory_processing:
            if self._pending_delete_memory.get("create_time", 0) + timeout >= time.time():
                await self.say(SendMessage(
                    SendReply(context.msg_id),
                    SendText("有缓存中的指定记忆删除操作，请等待其他操作完成或使用 '#remove_memory --abort' 取消")
                ))
                return

            elif self._pending_reset_memory.get("create_time", 0) + timeout >= time.time():
                if context.message.user_id != self._pending_reset_memory.get("user_id", ""):
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("有其他管理员正在进行的记忆重置操作，请等待他人操作完成")
                    ))
                    return

                if args.abort:
                    async with self._remove_memory_lock:
                        self._pending_reset_memory.clear()
                        await self.say(SendMessage(
                            SendReply(context.msg_id),
                            SendText("已放弃重置操作")
                        ))
                        self._remove_memory_processing = False
                        return

                if args.confirm:
                    async with self._remove_memory_lock:
                        await self.say(SendMessage(
                            SendReply(context.msg_id),
                            SendText("正在重置记忆...")
                        ))
                        await self.main_dialog.reset_memory()
                        self._pending_reset_memory.clear()
                        self._remove_memory_processing = False

                        return

            else:
                async with self._remove_memory_lock:
                    self._pending_reset_memory.clear()
                    self._remove_memory_processing = False

        async with self._remove_memory_lock:
            if args.abort:
                if self._pending_reset_memory:
                    self._remove_memory_processing = True
                    self._pending_reset_memory.clear()
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("已放弃重置操作")
                    ))
                    self._remove_memory_processing = False

                else:
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("无可放弃的操作")
                    ))

                return

            if args.confirm:
                if self._pending_reset_memory.get("create_time", 0) + timeout >= time.time():
                    self._remove_memory_processing = True
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("正在重置记忆库...")
                    ))
                    await self.main_dialog.reset_memory()
                    self._pending_reset_memory.clear()
                    self._remove_memory_processing = False

                else:
                    self._pending_reset_memory.clear()
                    await self.say(SendMessage(
                        SendReply(context.msg_id),
                        SendText("需要先运行一次，再执行确认")
                    ))

                return

            self._remove_memory_processing = True
            self._pending_reset_memory = {
                "create_time": time.time(),
                "user_id": context.user_id
            }
            await self.say(SendMessage(
                SendReply(context.msg_id),
                SendText("**危险提示**\n\n你正在试图重置本群记忆库，此操作不可逆，且会丢失至今为止的所有记忆(无自动备份)\n"
                         "请确认你已经知晓这是什么操作，以及会带来的后果。确认无误后，附带 '--confirm' 关键字确认重置操作")
            ))
            return


class GroupMainDialog(BaseDialog, GroupDialogAttrs):
    def __init__(self, *, host: QQGroup, **kwargs):
        super().__init__(host=host, **kwargs)
        self._pause = True

        # 配置
        self._call_or_at_speak_decreasing = SETTING_CFG.Groups.CallOrAtProbabilityDecreasing  # 默认1.6%
        self.agent_round_limit = self.group_config.agent_round_limit  # 默认30次
        self.speaker_round_limit = self.group_config.speaker_round_limit  # 默认30次
        agent_timeout = SETTING_CFG.Groups.AgentRequestTime  # 默认300秒

        # 标识符
        self._last_group_persona_time = time.time()
        self._last_member_persona_time: dict[str, float] = defaultdict(float)
        self._memory_tip_notice: str = ""
        self.memory_save_notice: str = ""
        self._image_repeat = 0
        self._memory_summary_message_count = 0
        self._new_words_analysis_count = 0
        self._last_bot_said: deque[dict] = deque(maxlen=10)
        self._paraphrase = 0
        self._long_speak = 0
        self._community_ready = False
        self.speaker_busy = False
        self._speaker_task_count = 0
        self._last_try_repeat_image_time = 0

        # 缓存
        self._group_persona_cache: GroupPersonaResult | None = None
        self._member_persona_cache: dict[str, MemberPersonaResult] = {}

        # 数据结构
        self.relative_news: dict[str, dict[str, list]] = {}
        self.member_alias: dict[str, list[str]] = {}

        # 工具对象
        GCM = get_group_character_manager(self.host.group_id)
        if GCM is None:
            self.host.logger.error("获取配置人格失败，返回默认人格[抹布]")
            character = "抹布"

        else:
            character = GCM.character_now

        self.GCM = GCM
        self.NOTICE = ChatNotification()  # 通知系统
        self.SPEAKER = GroupChatSpeaker(
            host=self,
            character=character,
            llm_preset=self.host.group_config.chat_model,
            logger=self.logger
        )
        self.AGENT = GroupChatAgent(
            host=self,  # type: ignore
            llm_preset=self.host.group_config.agent_model,
            agent_name=f"{self.host.group_name}_Chat",
            control_timeout=agent_timeout,
            logger=self.logger
        )
        self.GROUP_PERSONA = GroupPersona(self.host)
        self.MEMBER_PERSONA: dict[str, GroupMemberPersona] = {}
        self.SPEAK_PROBABILITY = AttentionSpeakProbability(
            min_prob=self.host.group_config.speak_rate_min,
            max_prob=self.host.group_config.speak_rate_max,
            time_window=self.host.group_config.attention_fade_out_time
        )
        self.SETU = Setu(self.group_id, self.group_config.setu_nsfw)
        self.GROUP_COMMUNITY = GroupRelatedness(
            group_id=self.group_id,
            data_path=self.data_path / "relatedness"
        )
        self.BOT_COMMUNITY = BotCommunity(
            relatedness=self.GROUP_COMMUNITY,
            bot_id=self.bot_id,
            data_path=self.data_path/ "relatedness"
        )

        # 锁
        self._group_persona_lock = asyncio.Lock()
        self._member_persona_lock = asyncio.Lock()
        self._speak_lock = asyncio.Lock()
        self._memory_summary_lock = asyncio.Lock()
        self._new_words_refresh_lock = asyncio.Lock()
        self._auto_mission_check_lock = asyncio.Lock()

        # LOOP
        self._auto_save_loop: asyncio.Task | None = None
        self._auto_clear_loop: asyncio.Task | None = None
        self._auto_mission_loop: asyncio.Task | None = None

        # 初始化
        CONFIG_OBSERVER.register(self)
        self._build_handle_sequence()

    def _build_handle_sequence(self):
        sequence = [
            self._muted_trigger,
            self._commit_new_message,
            self._got_banned_trigger,
            self._setu_trigger,
            self._repeat_trigger,
            self._agent_chat
        ]
        self._message_handle_sequence.extend(sequence)

    async def message_flow(self, msg: GroupMsg) -> bool:
        await self._check_auto_mission()
        for handle in self._message_handle_sequence:
            skip = await handle(msg)
            if skip:
                return False

        return False

    async def _check_auto_mission(self):
        """维护自动任务"""
        async with self._auto_mission_check_lock:
            if self._auto_mission_loop is None:
                self._auto_mission_loop = asyncio.create_task(self.auto_mission_loop())

            elif self._auto_mission_loop.done():
                self._auto_mission_loop = asyncio.create_task(self.auto_mission_loop())

    async def auto_mission_loop(self):
        """主对话的周期性自动化任务"""
        handlers: dict[int, Callable[..., Coroutine]] = {
            next_time_id(): self._auto_fetch_new_illusts
        }
        tasks: dict[int, asyncio.Task] = {}
        sleep = SETTING_CFG.Common.LoopTasksGuardPeriod
        if not isinstance(sleep, (float, int)) or sleep < 60:
            sleep = 3600  # 保底，防止无限循环检查

        try:
            while True:
                for key, handler in handlers.items():
                    if key not in tasks:
                        tasks[key] = asyncio.create_task(handler())

                    else:
                        task = tasks[key]
                        if task.done():
                            tasks[key] = asyncio.create_task(handler())

                await asyncio.sleep(sleep)

        except asyncio.CancelledError:
            for task in tasks.values():
                task.cancel()

            return

    # =========================================================
    # Trigger区，统一规则，True代表跳过
    # =========================================================

    async def _muted_trigger(self, msg: GroupMsg) -> bool:
        """判断用户是否被BOT拉黑，不加入历史记录"""
        if self.host.is_muted(msg.user_id):
            return True

        return False

    async def _setu_trigger(self, msg: GroupMsg) -> bool:
        """解析对已发送色图的回复互动。"""
        text = (await msg.pure_text()).strip()
        if not text:
            return False

        pixiv_targets: list[tuple[str, str]] = []
        seen_targets: set[tuple[str, str]] = set()
        for match in _pixiv_link_filter.finditer(text):
            target = (match.group("kind").lower(), match.group("target_id"))
            if target in seen_targets:
                continue

            seen_targets.add(target)
            pixiv_targets.append(target)

        if (
                pixiv_targets
                and self.group_config.setu
                and BASE_CFG.Module.setu
        ):
            timeout = SETTING_CFG.SETU.SendSETUTaskTimeout
            for kind, target_id in pixiv_targets:
                if kind == "artworks":
                    args = f"--count 5 --illust {target_id} --origin"

                else:
                    args = f"--count 5 --artist {target_id}"

                # noinspection PyAsyncCall
                self.aqueue.add_task(self.setu(args), timeout=timeout)

            await self.say(SendMessage(
                SendReply(msg.msg_id),
                SendText("好耶！来点色图！")
            ))
            return True

        if not self.is_reply(msg):
            return False

        reply = msg.reply
        if reply is None:
            return False

        allow_global = False
        global_targets: tuple[Setu, ...] = ()
        if "--global" in text.split():
            permission = await self.host.command_dialog.get_user_permission(msg.user_id)
            allow_global = permission >= _SETU_GLOBAL_PERMISSION
            global_targets = tuple(
                group.main_dialog.SETU
                for group in QQ_GROUPS.values()
                if group.main_dialog is not None
            )

        try:
            response = await self.SETU.respond_to_result(
                reply.msg_id,
                msg.user_id,
                text,
                allow_global=allow_global,
                global_targets=global_targets
            )

        except Exception as exc:
            self.logger.error(f"色图互动处理异常: {exc}")
            self.logger.debug(traceback.format_exc())
            await self.say(SendMessage(
                SendReply(msg.msg_id),
                SendText("色图互动处理失败，看看日志咋回事")
            ))
            return True

        if response is None:
            return False

        await self.say(SendMessage(
            SendReply(msg.msg_id),
            SendText(response)
        ))
        return True

    async def _repeat_trigger(self, msg: GroupMsg) -> bool:
        """复读机"""
        random_flag = random.random()
        common_rate = SETTING_CFG.Groups.CommonRepeatRate  # 默认0.8
        image_rate = SETTING_CFG.Groups.ImageRepeatRate  # 默认0.9
        async def _image_repeat_protect():
            await asyncio.sleep(15)
            self._image_repeat = 0

        def _get_repeat_image() -> str:
            image_window = SETTING_CFG.Groups.ImageRepeatMessageWindow  # 默认15
            image_msgs = self.message_history.get_last_message(self._image_repeat or image_window, True)
            image_hash_counts = defaultdict(int)
            for _msg in image_msgs:
                if _msg.user_id == self.bot_id:
                    return ""

                for item in _msg:
                    if isinstance(item, ImgMsg):
                        if not item.hash_name:
                            continue

                        image_hash_counts[item.hash_name] += 1
                        if image_hash_counts[item.hash_name] >= 3:
                            return item.hash_name

            return ""

        common_count = 1
        last_hash = ""
        can_repeat_msg = None
        for msg in reversed(self.message_history):
            if msg.user_id == self.bot_id:
                break

            # 先判断普通发言复读
            repeat_hash = msg.repeatable_hash()
            if not last_hash:
                last_hash = repeat_hash
                can_repeat_msg = msg
                continue

            if repeat_hash == last_hash:
                common_count += 1

            else:
                break

        if common_count >= 3 and can_repeat_msg is not None:
            if random_flag < common_rate:
                self._image_repeat = 1
                asyncio.create_task(_image_repeat_protect())
                said = await self.say(can_repeat_msg)
                if said and said[0] is not None:
                    said_msg = await self.message_history.add_message(said[0].copy())
                    await self.community_process(cast(GroupMsg, said_msg), bot_said=True)
                    return True

        # 再判断图片复读
        can_repeat_img = _get_repeat_image()
        if can_repeat_img and random_flag < image_rate:
            if time.time() - self._last_try_repeat_image_time < 60:
                return False

            self._last_try_repeat_image_time = time.time()
            said = await self.say(SendImage(can_repeat_img))
            if said and said[0] is not None:
                said_msg = await self.message_history.add_message(said[0].copy())
                await self.community_process(cast(GroupMsg, said_msg), bot_said=True)
                return True

        return False

    async def _got_banned_trigger(self, msg: GroupMsg) -> bool:
        if self.host.got_banned:
            notice = self.NOTICE.add_notice(
                content="你近期已被禁言，注意发言，通过 `speak_prompt` 明确告知 Speaker 注意",
                title="禁言与拉黑",
                priority=10,
                alive_until_self_messages=5,
                alive_until_time=600,
                exist_ok=True
            )
            self.NOTICE.refresh_notice(notice)
            return True

        return False

    # =========================================================
    # 主功能: 群聊对话入口与逻辑
    # =========================================================

    async def _commit_new_message(self, msg: GroupMsg) -> bool:
        await self._ensure_member_list()
        receive_msg = await self.message_history.add_message(msg, self.host.member_list)
        await self.community_process(cast(GroupMsg, receive_msg))
        self.NOTICE.notice_message_trigger()
        self._memory_summary_message_count += 1

        if self._memory_summary_message_count >= SETTING_CFG.Groups.MemorySummaryMessageCount:  #
            # noinspection PyAsyncCall
            self.aqueue.add_task(self.auto_memory_summary(), timeout=-1)

        if self._new_words_analysis_count >= self.message_history.capacity - 500:
            # noinspection PyAsyncCall
            self.aqueue.add_task(self.auto_refresh_community_new_words(), timeout=-1)

        return False

    async def _agent_chat(self, msg: GroupMsg) -> bool:
        """整个BOT群聊逻辑的主入口，接入Agent"""
        speak_prob = self.SPEAK_PROBABILITY.get_probability()
        random_flag = random.random()
        bot_prob = await self.BOT_COMMUNITY.score(msg.msg_id)
        combo_prop = speak_prob + bot_prob.continuity + bot_prob.interest
        max_prob = self.group_config.speak_rate_max
        combo_prop = min(max_prob, combo_prop)
        self.logger.debug(f"当前发言概率: [{combo_prop:.2%}]; 注意力: [{speak_prob:.2f}]; "
            f"兴趣分: [{bot_prob.interest:.2f}]; 连续性: [{bot_prob.continuity:.2f}]")

        # 先判断强制发言
        if self.is_reply_or_at(msg):
            if random_flag < (1 - self._call_or_at_speak_decreasing):
                self.SPEAK_PROBABILITY.wake()

                # noinspection PyAsyncCall
                self.aqueue.add_task(self.AGENT.run_in_queue(PostReason.CALL_OR_REPLY), timeout=-1)

        # 随机发言，采用动态热度和呼吸机制，同时不占用繁忙
        elif random_flag < combo_prop:
            if self.speaker_busy:
                return False

            self.SPEAK_PROBABILITY.wake()

            # noinspection PyAsyncCall
            self.aqueue.add_task(self.AGENT.run_in_queue(PostReason.RANDOM_SPEAK), timeout=-1)

        return False

    async def speak(
            self,
            *,
            speak_prompt: str = "",
            working: str = "",
            relative_information: dict[str, str] = None,
            relative_memory: dict[str, str] = None
    ) -> str | None:
        timeout = SETTING_CFG.LLM.SpeakerRequestTimeout  # 默认180
        queue_started_at = time.monotonic()
        self._speaker_task_count += 1
        self.speaker_busy = True
        try:
            # 防止上一句话还没说完就构造下一句话
            async with self._speak_lock:
                if time.monotonic() - queue_started_at >= timeout:
                    self.SPEAKER.mark_failure(
                        "Speaker 队列等待超时，发言内容已过期，任务已结束，无需重试",
                        retryable=False
                    )
                    return None

                # 给一点默认信息
                if relative_memory is None:
                    relative_memory = {}

                if not self.SPEAKER.had_group_persona:
                    group_persona = await self.get_group_persona(immediate=True)
                    if group_persona is not None:
                        relative_memory.update({"本群画像": group_persona.to_dict()})  # type: ignore
                        self.SPEAKER.had_group_persona = True

                last_few_msgs = self.message_history.get_last_message(20)
                users = set()
                for msg in last_few_msgs:
                    users.add(msg.user_id)

                for user in users:
                    member_persona = await self.get_member_persona(user_id=user, immediate=True)
                    if member_persona is None:
                        continue

                    member_persona_dict = member_persona.to_dict()
                    hash_flag = hash(str(member_persona_dict))
                    if user in self.SPEAKER.member_persona and hash_flag == self.SPEAKER.member_persona[user]:
                        continue

                    relative_memory.update({f"群员 {user} 画像": member_persona_dict})  # type: ignore
                    self.SPEAKER.member_persona[user] = hash_flag

                bot_info = await self.get_bot_info()
                task = self.aqueue.add_task(self.SPEAKER.speak(
                    speak_prompt=speak_prompt,
                    working=working,
                    relative_information=relative_information,
                    relative_memory=relative_memory,
                    bot_info=bot_info
                ), timeout=timeout + 30)
                await task.wait()
                if task.cancelled or not task.result:
                    reason = "发言 LLM 任务执行异常或超时"
                    if task.exception:
                        reason = f"{reason}: {task.exception[-1]}"

                    self.SPEAKER.mark_failure(reason, retryable=True)
                    return None

                result: dict = task.result[0]
                if not isinstance(result, dict):
                    if not self.SPEAKER.last_failure_reason:
                        self.SPEAKER.mark_failure("发言 LLM 未返回合法发言结果")

                    return None

                said = await self.say(SendMessage.from_dict(result))
                success_said: list[GroupMsg] = []
                callback_text = []
                for msg in said:
                    if msg is None:
                        continue

                    added_msg = await self.message_history.add_message(msg, self.member_list)
                    self._last_bot_said.append((await added_msg.to_llm(wait=False)))
                    self.AGENT.said_intention[added_msg.msg_id] = result["intention"]
                    success_said.append(cast(GroupMsg, added_msg))
                    self.AGENT.said_review_count += 1
                    await self.community_process(cast(GroupMsg, added_msg), bot_said=True)

        finally:
            self._speaker_task_count -= 1
            self.speaker_busy = self._speaker_task_count > 0

        # 等到发言录入消息队列就解锁
        if success_said:
            # 推进自己发言的计数器
            self.NOTICE.notice_message_trigger(True)
            if any(flag in check["format_text"] for check in self._last_bot_said for flag in _PARAPHRASE_WORDS):
                self._paraphrase += 1

            if any(len(flag) > 20 for check in self._last_bot_said for flag in self.message_history[check["message_id"]].text):
                self._long_speak += 1

            if self._paraphrase >= 3:
                if self._paraphrase > 3:
                    clear_paraphrase = True
                    for check in success_said:
                        text = (await check.to_llm(wait=False))["format_text"]
                        if any(flag in text for flag in _PARAPHRASE_WORDS):
                            clear_paraphrase = False
                            break

                    if clear_paraphrase:
                        self._clear_short_said_indicator()

                if self._paraphrase >= 3:
                    self.NOTICE.add_notice(
                        content="Speaker 近期多次迎合、复述群友的话，发言质量低下、这是不被允许的。"
                                "通过 `speak_prompt` 明确告知 Speaker **禁止迎合他人观点**， 并且调整为合适的发言方向，"
                                "同时积极向 Speaker 传递信息，帮助它提高发言质量",
                        title="发言规范",
                        priority=30,
                        alive_until_self_messages=1,
                        exist_ok=True
                    )

            if self._long_speak >= 2:
                if self._long_speak > 2:
                    clear_long_speak = True
                    texts = []
                    for check in success_said:
                        texts.append(check.text)
                        if any(len(flag) > 20 for text in texts for flag in text):
                            clear_long_speak = False
                            break

                    if clear_long_speak:
                        self._clear_short_said_indicator()

                if self._long_speak >= 2:
                    self.NOTICE.add_notice(
                        content="Speaker 近期多次发言单句过长。"
                                "通过 `speak_prompt` 告知 Speaker 可以用 **分句** 将长句拆分成多个短句发送",
                        title="发言规范",
                        priority=31,
                        alive_until_self_messages=1,
                        exist_ok=True
                    )

        else:
            self.SPEAKER.mark_failure("发言消息发送失败或未取得消息回执", retryable=True)
            return None

        if result["to_agent"]:
            callback_text.append("# 来自 `SPEAK LLM` 的信息(要求))\n")
            callback_text.append(result["to_agent"])
            callback_text.append("\n\n# 上次发言\n")
            if not success_said:
                callback_text.append("**无成功发言**")

            else:
                timeout = SETTING_CFG.Groups.MessageParseTimeout  # 默认15秒
                for msg in success_said:
                    text = await msg.to_llm(wait=True, timeout=timeout)
                    callback_text.append(f"- `{msg.msg_id}`: {text}")

            return '\n'.join(callback_text)

        return ""

    def _initiate_main_agent(self):
        group_chat_initiate_input = {
            "group_name": lambda : self.host.group_name,
            "group_id": lambda : self.host.group_id,
            "group_summary": self.get_group_summary,
            "member_list": self._ensure_member_list,
            "active_member_info": self.get_active_member_info,
            "old_messages": self.get_old_message,
            "news": self.get_group_news
        }

        group_chat_initiate_prompt = self.AGENT.register_prompt(
            name="group_chat_initiate_prompt",
            path=PROMPTS_DIR / "group_chat_initiate",
            description="群聊主控agent的初始化prompt",
            role="user",
            default_input=group_chat_initiate_input
        )

        self.AGENT.add_initiate_prompts(group_chat_initiate_prompt)

    def _initiate_memory_tip_notification(self):
        messages_tip = SETTING_CFG.Groups.ForceMemoryTipMessages  # 默认50
        said_tip = SETTING_CFG.Groups.ForceMemoryTipSaid  # 默认15
        messages_save = SETTING_CFG.Groups.ForceMemorySaveMessages  # 默认75
        said_save = SETTING_CFG.Groups.ForceMemorySaveSaid  # 默认20
        # 确保记忆提醒存在
        memory_tip = self.NOTICE.add_notice(
            content="你已太长时间未添加记忆，尽快保存一条有效记忆",
            title="记忆通知",
            priority=45,
            alive_until_messages=-messages_tip,
            alive_until_self_messages=-said_tip,
            exist_ok=True
        )
        self._memory_tip_notice = memory_tip.uid
        memory_save = self.NOTICE.add_notice(
            content="你已太长时间未添加记忆，立即保存一条有效记忆",
            title="记忆通知",
            priority=40,
            alive_until_messages=-messages_save,
            alive_until_self_messages=-said_save,
            exist_ok=True
        )
        self.memory_save_notice = memory_save

    async def get_group_summary(self, immediate: bool = False) -> str:
        """获取群总结文本"""
        result = await self.get_group_persona(immediate)
        output_lines = [
            f"- 群类型: {GROUP_ACTION_TYPE_MAPPING[result.group_action_type.value]}\n",
        ]
        if result.group_hobby:
            output_lines.append(f"- 群爱好、喜欢的东西: {result.group_hobby}\n")

        if result.group_disgust:
            output_lines.append(f"- 群厌恶、讨厌的东西: {result.group_disgust}\n")

        member_list = await self._ensure_member_list()
        if member_list is not None and result.group_vip:
            vip_lines = []
            i = 1
            for user_id in result.group_vip:
                if user_id not in member_list:
                    continue

                member = member_list.get_member(user_id=user_id)[0]
                vip_lines.append(f"\t{i}. user_id: {user_id}; username: {member.username}; nickname: {member.nickname}")
                i += 1

            if vip_lines:
                output_lines.append(f"- 重要群员:  \n")
                output_lines.append('\n'.join(vip_lines))
                output_lines.append('\n')

        if result.group_event:
            output_lines.append(f"- 群近期事件、活动: {result.group_event}\n")

        if result.group_memes:
            memes_lines = []
            for meme in result.group_memes:
                memes_lines.append(f"\t- {meme}")

            if memes_lines:
                output_lines.append(f"- 群常用梗:  \n")
                output_lines.append('\n'.join(memes_lines))
                output_lines.append('\n')

        if result.group_relations:
            relations_lines = []
            for title, content in result.group_relations.items():
                relations_lines.append(f"\t- {title}: {content}")

            if relations_lines:
                output_lines.append(f"- 群内事物关系图:  \n")
                output_lines.append('\n'.join(relations_lines))
                output_lines.append('\n')

        return '\n'.join(output_lines)

    async def get_active_member_info(self, immediate: bool = False) -> list[dict]:
        """获取活跃群员信息"""
        member_list = await self._ensure_member_list()
        if member_list is None:
            return []

        input_limit = SETTING_CFG.Groups.ActiveMemberPersonaLimit  # 默认5个
        active_members = self.message_history.get_active_users()
        tasks: dict[str, asyncio.Task] = {}
        for user_id in active_members:
            if user_id not in member_list:
                continue

            if user_id not in self.MEMBER_PERSONA:
                member = member_list.get_member(user_id=user_id)[0]
                member_persona = GroupMemberPersona(member)
                if await member_persona.is_empty():
                    continue

                self.MEMBER_PERSONA[user_id] = member_persona

            tasks[user_id] = asyncio.create_task(self.get_member_persona(user_id, immediate))

        output = []
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        for k, v in tasks.items():
            try:
                if len(output) > input_limit:
                    break

                if not v.done() or v.cancelled() or v.exception() is not None:
                    continue

                result: MemberPersonaResult = v.result()
                if result is None:
                    continue

                output.append(result.to_dict())

            except Exception as E:
                self.host.logger.error(f"{self.host.log_prefix()}获取成员画像出现错误: {E}")
                self.host.logger.debug(traceback.format_exc())
                continue

        return output

    async def get_bot_info(self) -> Member | None:
        member_list = await self._ensure_member_list()
        if member_list is None:
            return None

        return member_list.get_bot_info()

    def get_old_message(self, count: int = None) -> list[GroupMsg]:
        """获取历史聊天记录"""
        if count is None:
            count = SETTING_CFG.Agent.AgentInitInputMessages  # 默认100条

        messages = self.message_history.get_last_message(count)
        if messages:
            self.AGENT.last_message = messages[-1].msg_id

        return cast(list[GroupMsg], messages)

    async def get_group_news(self) -> dict[str, list]:
        """获取与群有关的新闻"""
        today = datetime.now().timestamp()
        key = timestamp2text(today, "%y-%m-%d")
        news = self.relative_news.get(key, {})
        news_limit: int = SETTING_CFG.Groups.MaxRelativeNews  # 默认5个
        return dict(reversed(list(islice(reversed(news.items()), news_limit))))

    async def _ensure_member_list(self) -> MemberList | None:
        if self.member_list is None:
            await self.update_member()

        if self.member_list is None:
            self.host.logger.error(F"{self.host.log_prefix()}无法获取群列表")

        return self.member_list

    # =========================================================
    # 数据处理区，要频繁类实例的数据交互，例如BOT分析、相似度分析、通知管理
    # =========================================================

    async def get_group_persona(self, immediate: bool = False, force: bool =False) -> GroupPersonaResult:
        """获取群画像"""
        async with self._group_persona_lock:
            if self._group_persona_cache is not None and not force:
                if time.time() - self._last_group_persona_time < SETTING_CFG.Groups.GroupPersonaCacheTime:  # 默认3小时
                    return self._group_persona_cache

            if immediate:
                if self._group_persona_cache is None:
                    return GroupPersonaResult(self.group_id, self.group_name)

                else:
                    return self._group_persona_cache

            result = await self.GROUP_PERSONA.get_persona()
            self._group_persona_cache = result
            self._last_group_persona_time = time.time()
            return result

    async def get_member_persona(
            self,
            user_id: str,
            immediate: bool = False,
            force: bool = False
    ) -> MemberPersonaResult | None:
        """获取用户画像"""
        async with self._member_persona_lock:
            if user_id not in self.MEMBER_PERSONA:
                member_list = await self._ensure_member_list()
                member = member_list.get_member(user_id=user_id)
                if not member:
                    return None

                member = member[0]
                member_persona = GroupMemberPersona(member)
                if await member_persona.is_empty():
                    return None

                else:
                    self.MEMBER_PERSONA[user_id] = member_persona

            if user_id in self._member_persona_cache and not force:
                if time.time() - self._last_member_persona_time[user_id] < SETTING_CFG.Groups.MemberPersonaCacheTime:  # 默认1天
                    return self._member_persona_cache[user_id]

            if immediate:
                return self._member_persona_cache.get(user_id, None)

            result = await self.MEMBER_PERSONA[user_id].get_persona()
            if result.alias:
                alias = self.member_alias.setdefault(user_id, [])
                alias.clear()
                alias.extend(result.alias)

            self._member_persona_cache[user_id] = result
            self._last_member_persona_time[user_id] = time.time()
            return result

    async def add_memory(self, target: str, memory: str) -> dict:
        class _AddResult(TypedDict):
            success: bool
            memory: str
            message: str

        self.AGENT.force_save_memory_retry = 0
        member_list = await self._ensure_member_list()
        if member_list is None:
            return dict(_AddResult(
                success=False,
                memory="",
                message="获取成员列表失败，无法添加记忆"
            ))

        if target == "BOT":
            target = self.bot_id

        if target == "GROUP":
            response = await self.GROUP_PERSONA.add_memory(content=memory)

        else:
            if target not in self.MEMBER_PERSONA:
                member = member_list.get_member(user_id=target)
                if not member:
                    return dict(_AddResult(
                        success=False,
                        memory="",
                        message="成员不存在，无法添加记忆"
                    ))

                self.MEMBER_PERSONA[target] = GroupMemberPersona(member[0])

            response = await self.MEMBER_PERSONA[target].add_memory(memory)

        if self._memory_tip_notice or self.memory_save_notice:
            try:
                self.NOTICE.notice_suppress(self._memory_tip_notice)
                self.NOTICE.notice_suppress(self.memory_save_notice)

            except KeyError:
                self._initiate_memory_tip_notification()

        else:
            self._initiate_memory_tip_notification()

        return dict(_AddResult(
            success=response.success,
            memory=memory,
            message=response.message
        ))

    async def get_memory(self, target: str, query: str) -> dict:
        class _GetResult(TypedDict):
            success: bool
            main_answer: str
            sub_answer: str
            matches: list
            message: str

        member_list = await self._ensure_member_list()
        if member_list is None:
            return dict(_GetResult(
                success=False,
                main_answer="",
                sub_answer="",
                matches=[],
                message="获取成员列表失败，无法获取记忆"
            ))

        if target == "BOT":
            target = self.bot_id

        if target == "GROUP":
            bucket = self.GROUP_PERSONA
            response = await bucket.get_memory(query)

        else:
            if target not in self.MEMBER_PERSONA:
                member = member_list.get_member(user_id=target)
                if not member:
                    return dict(_GetResult(
                        success=False,
                        main_answer="",
                        sub_answer="",
                        matches=[],
                        message="成员不存在，无法获取记忆"
                    ))

                self.MEMBER_PERSONA[target] = GroupMemberPersona(member[0])

            bucket = self.MEMBER_PERSONA[target]
            response = await bucket.get_memory(query)

        if not response.success:
            return dict(_GetResult(
                success=False,
                main_answer="",
                sub_answer="",
                matches=[],
                message=response.message
            ))

        elif response.result_source == "LOCAL":
            return dict(_GetResult(
                success=False,
                main_answer="",
                sub_answer="",
                matches=[],
                message="有效结果不足"
            ))

        else:
            matches = []
            gate = SETTING_CFG.Persona.MemoryConfidenceGate  # 默认0.6
            for match in response.matches:
                if match.final_score > gate:
                    matches.append(match)

            if not matches:
                return dict(_GetResult(
                    success=False,
                    main_answer="",
                    sub_answer="",
                    matches=[],
                    message="有效结果不足"
                ))

            sub_answer = ""
            if response.sub_answer:
                main_answer = response.sub_answer
                sub_answer = response.answer

            else:
                main_answer = response.answer

            matches_dict = []
            for match in matches:
                memory_content = await bucket.get_memory_content(match.key)
                matches_dict.append({
                    "summary": memory_content.summary,
                    "content": memory_content.content,
                    "create_at": memory_content.created_at,
                    "score": match.final_score
                })

            return dict(_GetResult(
                success=True,
                main_answer=main_answer,
                sub_answer=sub_answer,
                matches=matches_dict,
                message=response.message
            ))

    async def search_memory(self, query: str) -> dict:
        class _SearchResult(TypedDict):
            success: bool
            answer: str
            matches: list
            message: str

        member_list = await self._ensure_member_list()
        if member_list is None:
            return dict(_SearchResult(
                success=False,
                answer="",
                matches=[],
                message="获取成员列表失败，无法获取记忆"
            ))

        bucket = self.GROUP_PERSONA
        response = await bucket.search_memory(query)
        if not response:
            return dict(_SearchResult(
                success=False,
                answer="",
                matches=[],
                message="记忆库异常，无返回结果"
            ))

        else:
            answer = response["answer"]
            matches = []
            gate = SETTING_CFG.Persona.SearchMemoryConfidenceGate  # 默认0.5
            self.logger.debug(response)
            for match in response["matches"]:
                if match.get("score", 0) > gate:
                    matches.append(match)

            matches_dict = []
            reversed_mapping: dict[str, str] = {self.GROUP_PERSONA.bucker_id: "GROUP"}
            bot_id = self.bot_id
            if bot_id not in self.MEMBER_PERSONA:
                bot_member = member_list.get_member(user_id=bot_id)
                self.MEMBER_PERSONA[bot_id] = GroupMemberPersona(bot_member[0])

            reversed_mapping[self.MEMBER_PERSONA[bot_id].bucker_id] = "BOT"
            for user_id, persona in self.MEMBER_PERSONA.items():
                if user_id == bot_id:
                    continue

                reversed_mapping[persona.bucker_id] = user_id

            for match in matches:
                if key := match.get("mem_id", ""):
                    memory_record = await bucket.get_memory_content(key)
                    memory_content = memory_record.content

                else:
                    memory_content = ""

                matches_dict.append({
                    "summary": match.get("summary", ""),
                    "content": memory_content,
                    "reason": match.get("reason", ""),
                    "score": match.get("score", 0),
                    "target": reversed_mapping.get(match.get("bucket_id", ""), "")
                })

            return dict(_SearchResult(
                success=True,
                answer=answer,
                matches=matches_dict,
                message=""
            ))

    async def set_alias(self, user_id: str, alias: str) -> None:
        """设置爱称"""
        member_list = await self._ensure_member_list()
        if member_list is None:
            raise KeyError("无法获取群员列表，添加爱称失败")

        user = member_list.get_member(user_id=user_id)
        if not user:
            raise KeyError("指定的成员不存在，添加爱称失败")

        aliases = self.member_alias.setdefault(user_id, [])
        if alias not in aliases:
            aliases.append(alias)

            await self.add_memory(
                target=user_id,
                memory=f"群员新增一个爱称: {alias}"
            )

    async def get_news_by_date(self, date: str = None) -> dict:
        """
        获取指定日期的新闻。
        :param date: 日期: 标准 `ISO 8601`字符串(无时区)，不传入则默认为今日。
        :return: 成功: 该日期的新闻字典; 无新闻: 空字典; 失败: 抛出异常
        """
        class _NewsResult(TypedDict):
            date: str
            news: dict[str, list]

        if date is None:
            date = datetime.now().timestamp()

        else:
            try:
                date = datetime.fromisoformat(date).astimezone().timestamp()

            except (TypeError, ValueError):
                raise ValueError("解析起始时间字符串失败")

        key = timestamp2text(date, "%y-%m-%d")
        if key in self.relative_news:
            return dict(_NewsResult(
                date=key,
                news=self.relative_news[key]
            ))

        else:
            news_file = self.data_path / "news" / f"{key}.json"
            if not news_file.is_file():
                return {}

            try:
                news: dict[str, list] = json.loads(news_file.read_text("utf-8"))

            except json.JSONDecodeError:
                return {}

            self.relative_news[key] = news
            return dict(_NewsResult(
                date=key,
                news=news
            ))

    async def auto_memory_summary(self):
        """根据聊天记录数量自动总结记忆"""
        trigger = SETTING_CFG.Groups.MemorySummaryMessageCount  # 默认 1000 条
        if self._memory_summary_lock.locked():
            return

        async with self._memory_summary_lock:
            if self._memory_summary_message_count < trigger:
                return

            member_list = await self._ensure_member_list()
            if member_list is None:
                self.logger.error("无法获取群员列表，记忆总结失败")
                return

            self.logger.info("正在总结群聊...")
            self._memory_summary_message_count = 0
            await self.save()
            messages_list = self.message_history.get_last_message(trigger)
            if len(messages_list) < 0.5 * trigger:
                # 数据不足
                return

            system_prompt = AgentPrompt(
                name="memory_summary_system_prompt",
                data_path=PROMPTS_DIR / "group_memory_summary_system",
                description="自动记忆总结的系统提示词",
                role="system"
            )
            try:
                llm_setting = parse_llm_setting(BASE_CFG.Agents.MemorySummaryModel)

            except Exception as E:
                self.logger.error(f"获取记忆总结模型出错: {E}")
                self.logger.debug(traceback.format_exc())
                return

            if llm_setting is None:
                self.logger.error("获取记忆总结模型出错")
                return

            llm_setting.keep_alive = False
            old_model_params = llm_setting.model_params
            if old_model_params is None:
                new_model_params = {"stream": False}

            else:
                new_model_params = old_model_params
                new_model_params.update({"stream": False})

            llm_setting.model_params = new_model_params
            llm_setting.system_prompt = await system_prompt.get_prompt()
            bot_info = member_list.get_bot_info()
            bot_ident = f"QQ号(user_id): [{bot_info.user_id}]; 群昵称: [{bot_info.nickname}]; QQ昵称: [{bot_info.username}]"
            tasks: list[asyncio.Task] = [asyncio.create_task(msg.to_llm(wait=True)) for msg in messages_list]
            await asyncio.gather(*tasks, return_exceptions=True)
            msg_payload = []
            for task in tasks:
                if not task.done() or task.cancelled() or task.exception() is not None:
                    continue

                origin_dict: dict = task.result()
                origin_dict.pop("message_type", None)
                origin_dict.pop("message_tips", None)
                origin_dict.pop("message_sequence", None)
                origin_dict.pop("message_content", None)
                msg_payload.append(origin_dict)

            payload = (f"# 聊天记录\n\n```json\n{json.dumps(msg_payload, ensure_ascii=False, indent=2)}\n```\n\n\n"
                       f"# BOT信息\n\n{bot_ident}")

            async with Chat(config=llm_setting) as chat:
                content = {}
                for _ in range(2):  # 写死，重试一次
                    response = await chat.ask(payload)
                    if response:
                        text_response = response[0]
                        if not (hasattr(text_response, "text") and text_response.text):
                            continue

                        json_text = clean_json_text_from_llm(text_response.text)
                        try:
                            content = json.loads(json_text)

                        except json.JSONDecodeError:
                            continue

                        else:
                            if not content or not isinstance(content, dict):
                                continue

                            break

                tasks: dict[str, asyncio.Task] = {}
                for k, v in content.items():  # type: str, dict
                    if k == "GROUP":
                        if not isinstance(v, dict) or not v.get("group_event", {}):
                            continue

                        target = self.GROUP_PERSONA

                    elif k == "BOT":
                        if not isinstance(v, dict) or not v.get("event", {}):
                            continue

                        target = self.MEMBER_PERSONA.setdefault(self.bot_id, GroupMemberPersona(bot_info))

                    elif k.isdigit():
                        if not isinstance(v, dict) or not v.get("event", {}):
                            continue

                        if k not in member_list:
                            continue

                        member = member_list.get_member(user_id=k)
                        target = self.MEMBER_PERSONA.setdefault(k, GroupMemberPersona(member[0]))

                    tasks[k] = asyncio.create_task(target.add_memory(json.dumps(v, ensure_ascii=False, indent=2)))

                await asyncio.gather(*tasks.values(), return_exceptions=True)
                success = []
                fails = []
                for target, task in tasks.items():
                    if task.done() and not task.cancel() and task.exception() is None:
                        success.append(target)

                    else:
                        fails.append(target)

                fail_tip = ""
                if fails:
                    fail_tip = f"，{fails} 记忆总结失败"

                self.logger.info(f"自动记忆总结已完成，新增 {success} 的记忆{fail_tip}")

    async def auto_refresh_community_new_words(self):
        """自动更新 NLP 新词"""
        if self._new_words_refresh_lock.locked():
            return

        async with self._new_words_refresh_lock:
            if self._new_words_analysis_count < max(3000, self.message_history.capacity - 500):
                return

            self.logger.info("正在发现新词...")
            try:
                await self.GROUP_COMMUNITY.refresh_new_words(self.message_history)

            except Exception as E:
                self.logger.error(f"新词发现失败: {E}")
                self.logger.debug(traceback.format_exc())

            else:
                self._new_words_analysis_count = 0
                await self.save()

    async def community_process(self, msg: GroupMsg, bot_said: bool = False) -> MemberAffinityScore | None:
        if not self._community_ready:
            return None

        try:
            await self.GROUP_COMMUNITY.ingest(msg)
            if bot_said:
                await self.BOT_COMMUNITY.record_output(msg.msg_id)
                return None

            else:
                bot_score: MemberAffinityScore = await self.BOT_COMMUNITY.score(msg.msg_id)
                return bot_score

        except Exception as E:
            self.logger.error(f"相关性社区模型链路错误: {E}")
            self.logger.debug(traceback.format_exc())
            return None

    # =========================================================
    # 色图，核心功能这一块
    # =========================================================

    @_SETU_COMMAND.command(
        name="setu"
    )
    @_SETU_COMMAND.option(
        "--help",
        dest="help",
        action="store_true",
        default=False,
        _type=bool
    )
    @_SETU_COMMAND.option(
        "--count", "-c",
        dest="count",
        default=1,
        _type=int,
    )
    @_SETU_COMMAND.option(
        "--illust", "-i",
        dest="illust_id",
        default="",
    )
    @_SETU_COMMAND.option(
        "--artist", "-a",
        dest="artist_id",
        default=""
    )
    @_SETU_COMMAND.option(
        "--query", "-q",
        dest="query_text",
        default=""
    )
    @_SETU_COMMAND.option(
        "--origin", "-o",
        dest="origin",
        action="store_true",
        default=False,
        _type=bool
    )
    @_SETU_COMMAND.option(
        "--no_related", "-n",
        dest="related",
        action="store_false",
        default=True,
        _type=bool
    )
    async def _setu_command_parser(
            self,
            _command: CommandArgs
    ) -> str | PixivPayload | None:
        # 先做 help
        if _command.help:
            return self._setu_help()

        # 按 illust_id -> artist_id -> query_text -> random 获取色图
        expect_count = min(5, max(_command.count, 1))
        if _command.illust_id:
            include_origin = _command.origin
            include_related = _command.related
            return await self.SETU.get_setu(
                expect_count=expect_count,
                illust_id=_command.illust_id,
                include_origin=include_origin,
                include_related=include_related
            )

        elif _command.artist_id:
            return await self.SETU.get_setu(
                expect_count=expect_count,
                artist_id=_command.artist_id
            )

        elif _command.query_text:
            return await self.SETU.get_setu(
                expect_count=expect_count,
                query_text=_command.query_text
            )

        else:
            return await self.SETU.get_setu(
                expect_count=expect_count
            )

    @staticmethod
    def _setu_help() -> str:
        tips = [
            "可用参数如下，所有参数均为可选，当不传入任何参数时，返回一张随机色图。  ",
            "`--count/--c`: 指定要获取的色图数量"
            "`--illust/--i`: 获取指定id的色图本身/相关色图，通过 `--origin/--no_related` 控制行为"
            "`--artist/--a`: 获取指定画师的色图"
            "`--query/--q`: 通过自然语言获取指定色图"
            "`--origin/--o`: 获取指定id色图时，是否包括自身，默认不包括"
            "`--no_related`: 获取指定id色图时，是否获取相关色图，默认获取"
        ]
        return '\n'.join(tips)

    async def setu(self, args: str) -> None:
        """
        <--count/--c> 要获取的色图数量
        <--illust/--i> 获取指定色图id的相关色图
        <--artist/--a> 获取指定画师id的色图
        <--query/--q> 获取指定色图，自然语言输入
        <--origin/--o> 获取指定色图id时，是否包含自身，默认不包含(False)
        <--no_related> 获取指定色图id时，是否不获取相关色图，默认获取(True)
        <--help> 帮助
        :param args:
        :return:
        """
        async def _process_worker(
                _illustration: Illustration,
                _illust: Illust,
                _image_bytes: bytes,
                _freeze_id: str,
                _success: list[str],
                _fails: list[str]
        ):
            iid = _illustration.iid
            key = f"{iid}_p{_illust.pages[0]}"

            # 查询缓存，若有直接发送
            send_hash_name = POST_PROCESS_HASH_MAPPING.get(key)
            if send_hash_name:
                try:
                    await get_file_path_async(send_hash_name)

                except FileNotFoundError:
                    pass

                else:
                    metadata = await get_file_metadata_async(send_hash_name)
                    if metadata is None:
                        pass

                    else:
                        if metadata.note == "经过后处理的色图，使用文件发送":
                            upload_file = True

                        else:
                            upload_file = False

                        receipt = await _send_setu(
                            _illustration=_illustration,
                            _illust=_illust,
                            _hash_name=send_hash_name,
                            _upload_file=upload_file
                        )
                        if receipt.sent:
                            await self.SETU.mark_used(
                                message_id=receipt.message_id,
                                illust=_illust,
                                freeze_id=_freeze_id
                            )
                            _success.append(iid)
                            return

            # 判断广告
            if _illustration.type.value == "ugoira":
                is_ads = False

            else:
                is_ads = await detect_advertisement(_image_bytes)

            if is_ads:
                self.logger.debug(f"[PIXIV][测试]检测到广告: {iid}_p{_illust.pages[0]}")
                await self.say(f"[PIXIV][测试]检测到广告: {iid}_p{_illust.pages[0]}")
                await self.SETU.mark_ban_pages({iid: _illust})
                _fails.append(iid)
                return

            # 压缩
            loop = asyncio.get_running_loop()
            if _illustration.type.value == "ugoira":
                future = loop.run_in_executor(GLOBAL_EXECUTOR, compress_gif,
                                              _image_bytes, gif_size_limit)
                compressed: BytesIO | None = await future

            else:
                future = loop.run_in_executor(GLOBAL_EXECUTOR, compress_image, _image_bytes, image_size_limit)
                compressed, _ = await future

            # 判断上传分支
            upload_file = False
            if compressed is None:
                upload_file = True
                compressed = BytesIO(_image_bytes)

            # 后处理
            try:
                # noinspection PyTypeChecker
                future = loop.run_in_executor(
                    GLOBAL_EXECUTOR,
                    partial(
                        postprocess_image,
                        compressed,
                        noise_pixels=noise_pixels
                    )
                )
                postprocessed = await future

            except ValueError:
                postprocessed = compressed

            send_hash_name = await add_file_async(
                postprocessed,
                file_type="image",
                note="经过后处理的色图，使用文件发送" if upload_file else "经过后处理的色图，使用普通发送"
            )
            POST_PROCESS_HASH_MAPPING[key] = send_hash_name

            # 上传
            receipt = await _send_setu(
                _illustration=_illustration,
                _illust=_illust,
                _hash_name=send_hash_name,
                _upload_file=upload_file
            )
            if receipt.sent:
                await self.SETU.mark_used(
                    message_id=receipt.message_id,
                    illust=_illust,
                    freeze_id=_freeze_id
                )
                _success.append(iid)

            else:
                _fails.append(iid)

            return

        async def _process_payload(_payload: PixivPayload):
            """按count并发处理，节省计算量"""
            async def _worker_helper(
                    _illustration: Illustration,
                    _illust: Illust,
                    _image_bytes: bytes
            ):
                try:
                    await _process_worker(
                        _illustration=_illustration,
                        _illust=_illust,
                        _image_bytes=_image_bytes,
                        _freeze_id=_payload.freeze_id,
                        _success=success,
                        _fails=fails
                    )

                except Exception as E:
                    fails.append(_illust.iid)
                    self.logger.error(f"色图 [{_illust.iid}] 处理失败: {E}")
                    self.logger.debug(traceback.format_exc())

            if not _payload:
                raise RuntimeError("色图模块内部错误，获取失败")

            fails: list[str] = []
            success: list[str] = []
            candidates: list[tuple[Illustration, Illust, bytes]] = []

            for iid, illust in _payload.illusts.items():
                hash_name = _payload.hash_names.get(iid, "")
                illustration = _payload.illustrations.get(iid, None)
                if illustration is None or not hash_name:
                    fails.append(iid)
                    continue

                try:
                    filepath = await get_file_path_async(hash_name)

                except FileNotFoundError:
                    fails.append(iid)
                    continue

                image_bytes = filepath.read_bytes()
                candidates.append((illustration, illust, image_bytes))

            while candidates and len(success) < _payload.count:
                remaining = _payload.count - len(success)
                batch = candidates[:remaining]
                del candidates[:remaining]
                await asyncio.gather(*(
                    _worker_helper(illustration, illust, image_bytes)
                    for illustration, illust, image_bytes in batch
                ))

            if not success:
                await self.say("色图全部发送失败了喵...")

            elif len(success) < _payload.count:
                await self.say(f"有 {len(fails)} 张色图发送失败了")

            await self.SETU.unfreeze(_payload.freeze_id)

        async def _send_setu(
                _illustration: Illustration,
                _illust: Illust,
                _hash_name: str,
                _upload_file: bool
        ) -> SetuSendReceipt:
            """单独一个发送逻辑，不走 say"""
            member_list = await self._ensure_member_list()
            timeout = SETTING_CFG.Groups.ApiTimeout  # 默认60秒
            setu_msg = SetuMsg(
                illust_id=_illustration.iid,
                page=_illust.pages[0],
                hash_name=_hash_name,
                artist_id=_illustration.artist_id,
                artist_name=_illustration.artist_name,
                type=_illustration.type.value,
                title=_illustration.title,
                caption=_illustration.caption,
                tags=_illustration.tags_list
            )
            send_msg = SendMessage.from_message(setu_msg)

            # 分链路发送
            if _upload_file:
                filename = f"{setu_msg.illust_id}_p{setu_msg.page}"
                if setu_msg.type == "ugoira":
                    filename += ".gif"

                else:
                    filename += ".png"

                try:
                    response = await asyncio.wait_for(
                        api.upload_group_file(
                            self.group_id,
                            _hash_name,
                            filename,
                            '/'
                        ),
                        timeout
                    )

                except asyncio.TimeoutError:
                    return SetuSendReceipt(False)

                if response is None:
                    return SetuSendReceipt(False)

                if response.get("status") != "ok":
                    return SetuSendReceipt(False)

                msg_id = await api.get_file_message_id(
                    self.group_id,
                    filename,
                    retry_interval=5
                )
                if not msg_id:
                    self.logger.warning(
                        f"色图已上传到群文件，但消息记录尚未同步: "
                        f"filename=[{filename}]"
                    )
                    return SetuSendReceipt(True)

            else:
                try:
                    response = await asyncio.wait_for(
                        api.say(
                            self.group_id,
                            (await send_msg.message_chain_async())[0]
                        ),
                        timeout
                    )

                except asyncio.TimeoutError:
                    return SetuSendReceipt(False)

                if response is None:
                    return SetuSendReceipt(False)

                if "say" not in response:
                    return SetuSendReceipt(False)

                say_result = response.get("say", {})
                if not say_result:
                    return SetuSendReceipt(False)

                if say_result["status"] != "ok":
                    return SetuSendReceipt(False)

                msg_id = str(say_result["data"]["message_id"])

            # 构造色图消息
            if member_list is None:
                bot_nickname = ""

            else:
                bot_info = member_list.get_bot_info()
                if bot_info is None:
                    bot_nickname = ""

                else:
                    bot_nickname = bot_info.nickname

            group_msg = GroupMsg(
                group_id=self.group_id,
                msg_id=msg_id,
                user_id=self.bot_id,
                username=self.bot_name,
                nickname=bot_nickname,
                content=[setu_msg]
            )
            mirror = cast(GroupMsg, await self.host.message_history.add_message(group_msg.copy(), member_list))
            said_msg = await self.message_history.add_message(group_msg, member_list)
            await self.community_process(cast(GroupMsg, said_msg), bot_said=True)
            asyncio.create_task(self.host.log_message(mirror))
            return SetuSendReceipt(True, msg_id)

        image_size_limit = SETTING_CFG.SETU.MaxImageFileSize  # 默认12mb
        gif_size_limit = SETTING_CFG.SETU.MaxGifFileSize  # 默认6mb
        noise_pixels = SETTING_CFG.SETU.NoisePixels  # 默认 5 个

        if not (self.group_config.setu and BASE_CFG.Module.setu):
            raise PermissionError("群未启用色图或色图模块总开关未开启")

        # 接受指令
        command = f"#setu {args}"
        try:
            invocation = _SETU_COMMAND.parse(command)
            result = await invocation.spec.handler(self, invocation.args)

            # 处理 help
            if isinstance(result, str):
                raise SetuHelp(result)

            # 处理色图
            if result is None:
                raise RuntimeError("色图模块内部错误，获取失败")

            await _process_payload(result)

        except SetuHelp:
            raise

        except CommandError as exc:
            self.logger.error(f"色图指令错误: {exc}")
            raise exc

        except Exception as exc:
            self.logger.error(f"色图指令错误: {exc}")
            self.logger.debug(traceback.format_exc())
            await self.say("色图链路错误，看看日志啥情况")
            raise exc

        return None

    async def _auto_fetch_new_illusts(self):
        """自动刷新群色图库"""
        try:
            while True:
                if self.group_config.setu and BASE_CFG.Module.setu:
                    await self.SETU.get_new_illusts()

                sleep = SETTING_CFG.SETU.AutoFetchPeriod
                await asyncio.sleep(sleep)

        except asyncio.CancelledError:
            return

    # =========================================================
    # 外部指令接口区
    # =========================================================

    def refresh_context(self):
        """强制刷新上下文窗口"""
        self.AGENT.force_refresh()
        self.SPEAKER.force_refresh()

    def clear_history(self):
        """强制清楚聊天记录，并刷新上下文窗口"""
        self.AGENT.clear_history()
        if self.message_history:
            self.SPEAKER.last_message = self.message_history[-1].msg_id

        self.SPEAKER.force_refresh()

    def get_average_speak_delay(self) -> float | None:
        """获取上下文窗口内的发言平均延迟"""
        return self.AGENT.get_average_speak_delay()

    async def show_group_persona(self) -> GroupPersonaResult:
        return await self.get_group_persona(immediate=False, force=True)

    async def show_member_persona(self, user_id) -> MemberPersonaResult | None:
        return await self.get_member_persona(user_id, False, True)

    async def get_remove_memory_list(
            self,
            *,
            target: str,
            query: str,
            count: int = 50,
            threshold: float = 0.5
    ) -> dict[str, Any]:
        """获取要删除的记忆列表"""
        member_list = await self._ensure_member_list()
        if member_list is None:
            raise RuntimeError("获取用户列表失败，无法获取删除记忆列表")

        if target.lower() == "group":
            bucket = self.GROUP_PERSONA

        elif target.lower() == "bot":
            user_id = self.bot_id
            member = member_list.get_member(user_id=user_id)
            if not member:
                raise KeyError("用户不存在，无法获取删除记忆列表")

            member = member[0]
            if target in self.MEMBER_PERSONA:
                bucket = self.MEMBER_PERSONA[target]

            else:
                self.MEMBER_PERSONA[target] = bucket = GroupMemberPersona(member)

        elif target.isdigit():
            member = member_list.get_member(user_id=target)
            if not member:
                raise KeyError("用户不存在，无法获取删除记忆列表")

            member = member[0]
            if target in self.MEMBER_PERSONA:
                bucket = self.MEMBER_PERSONA[target]

            else:
                self.MEMBER_PERSONA[target] = bucket = GroupMemberPersona(member)

        else:
            raise KeyError("指定目标错误，无法获取删除记忆列表")

        remove_system = AgentPrompt(
            name="remove_system",
            data_path=PROMPTS_DIR / "group_remove_memory_system",
            description="全库删除指定记忆的系统提示词",
            role="system"
        )
        system_prompt = await remove_system.get_prompt()
        time_info = f"> 当前时间: {timestamp2text(time.time())}; 期望获取记忆数量: `{count}`  \n\n"
        command = f"# 查询内容\n\n{query}\n\n\n"
        content = {}
        await bucket._ensure_bucket_handle()
        for _ in range(2):  # 只重试一次，写死
            try:
                response = await bucket.bucket.advance_query(
                    system_prompt=system_prompt,
                    command=time_info + command
                )
                if not response:
                    continue

                text_response = response[0]
                if not (hasattr(text_response, "text") and text_response.text):
                    continue

                json_text = clean_json_text_from_llm(text_response.text)
                try:
                    content = json.loads(json_text)

                except json.JSONDecodeError:
                    continue

                else:
                    if not content or not isinstance(content, dict):
                        continue

                    break

            except Exception as E:
                self.logger.error(f"删除指定记忆发生错误: {E}")
                self.logger.debug(traceback.format_exc())
                continue

        if not content:
            return {}

        memory_aliases = []
        for raw_key, score in content.items():
            if not isinstance(score, (float, int)) or not 0 <= score <= 1:
                continue
            key = str(raw_key).strip()
            if key.startswith("memory_"):
                memory_aliases.append(key)
        resolved_aliases = await bucket.bucket.resolve_aliases(memory_aliases) if memory_aliases else {}

        p0 = 0
        p20 = 0
        p50 = 0
        p80 = 0
        can_delete = {}
        for key, score in content.items():  # type: str, float
            if not isinstance(score, (float, int)) or not 0 <= score <= 1:
                continue

            key = key.strip()
            if not (key.startswith("bucket_") or key.startswith("memory_")):
                continue

            # 不直接删桶
            if key.strip().startswith("bucket_"):
                continue

            mem_id = resolved_aliases.get(key, "")
            if not mem_id:
                continue

            memory_record = await bucket.bucket.get_memory(mem_id)
            if memory_record is None or memory_record.kind == "bucket":
                continue

            if score > threshold:
                can_delete[mem_id] = score

            if score < 0.2:
                p0 += 1

            elif score < 0.5:
                p20 += 1

            elif score < 0.8:
                p50 += 1

            else:
                p80 += 1

        if p80 or p50 or p20 or p0:
            result_text = (f"已完成待删除记忆筛选: \n"
                           f"目标删除内容: {query}\n"
                           f"- 0.0~0.2: {p0} 条\n"
                           f"- 0.2~0.5: {p20} 条\n"
                           f"- 0.5~0.8: {p50} 条\n"
                           f"- 0.8~1.0: {p80} 条")
            await self.say(result_text)

        can_delete = dict(sorted(can_delete.items(), key=lambda x: x[1], reverse=True)[:count])

        return {
            "bucket": bucket,
            "memories": can_delete,
            "create_time": time.time()
        }

    async def remove_memory(self, pending_delete: dict[str, Any]):
        """删除指定记忆，操作不可逆"""
        memories = pending_delete.get("memories", {})
        bucket: GroupMemberPersona | GroupPersona | None = pending_delete.get("bucket", None)
        if not pending_delete or not memories or bucket is None:
            await self.say("未发现可删除记忆")
            return

        success = []
        fail = []
        tasks: dict[str, asyncio.Task] = {}
        await bucket._ensure_bucket_handle()
        if isinstance(bucket, GroupPersona):
            self._group_persona_cache = None

        else:
            self._member_persona_cache.pop(bucket.user_id, None)

        for key in memories.keys():
            tasks[key] = asyncio.create_task(bucket.bucket.delete_memory(key, reason="用户主动删除"))

        await asyncio.gather(*tasks.values(), return_exceptions=True)
        for k, v in tasks.items():
            if not v.done() or v.cancelled() or v.exception() is not None:
                fail.append(k)
                continue

            result = v.result()
            if result.success:
                success.append(k)

            else:
                fail.append(k)

        try:
            result = await bucket.bucket.optimize(reason="用户删除记忆后优化")

        except Exception as E:
            self.logger.error(f"优化记忆库失败: {E}")
            self.logger.debug(traceback.format_exc())

        else:
            self.logger.debug(result.to_dict())

        await self.say(f"成功删除 [{len(success)}] 条记忆; 失败 [{len(fail)}] 条")
        self.logger.debug(f"群[{self.group_id}] 已删除以下记忆: \n{'\n'.join(success)}\n\n失败: \n{'\n'.join(fail)}")

    async def reset_memory(self):
        """重置整个群的记忆库，不可逆操作，极其危险"""
        bucket_id = self.GROUP_PERSONA.parent_bucket.bucket_id
        result = await self.GROUP_PERSONA.parent_bucket.delete_memory(bucket_id, reason="用户重置记忆库")
        if result.success:
            self.logger.debug(f"群[{self.group_id}] 已重置记忆库")
            self._member_persona_cache.clear()
            self._group_persona_cache = None
            self.MEMBER_PERSONA.clear()
            self.GROUP_PERSONA = GroupPersona(self.host)  # 重建
            await asyncio.gather(
                asyncio.create_task(self.get_group_persona()),
                asyncio.create_task(self.get_active_member_info()),
                return_exceptions=True
            )
            await self.say("已重置记忆库")

        else:
            await self.say(f"记忆库重置失败: {result.message}")

    # =========================================================
    # 其他函数，放一下无关紧要的东西
    # =========================================================

    def update_config(self):
        self._call_or_at_speak_decreasing = SETTING_CFG.Groups.CallOrAtProbabilityDecreasing
        self.SPEAK_PROBABILITY.min_prob = self.group_config.speak_rate_min
        self.SPEAK_PROBABILITY.max_prob = self.group_config.speak_rate_max
        self.SPEAK_PROBABILITY.time_window = self.group_config.attention_fade_out_time
        self.agent_round_limit = self.group_config.agent_round_limit  # 默认30次
        self.speaker_round_limit = self.group_config.speaker_round_limit  # 默认30次
        self.SETU.nsfw = self.group_config.setu_nsfw

    def _clear_short_said_indicator(self):
        self._last_bot_said.clear()
        self._long_speak = 0
        self._paraphrase = 0

    # =========================================================
    # 持久化
    # =========================================================

    def to_dict(self) -> dict:
        """AGENT自带持久化逻辑，SPEAKER/缓存不持久化"""
        dt = datetime.now().timestamp()
        dump_content = {
            "update": timestamp2text(dt),
            "description": "此文件为群聊天对话的持久化文件",
            "version": "0.1.0",
            "data": {
                "pause": self._pause,
                "messages": self.message_history.to_dict(),
                "news": self.relative_news.copy(),
                "notification": self.NOTICE.to_dict(),
                "last_group_persona_time": self._last_group_persona_time,
                "last_member_persona_time": self._last_member_persona_time.copy(),
                "memory_summary_message_count": self._memory_summary_message_count,
                "new_words_analysis_count": self._new_words_analysis_count
            }
        }
        return dump_content

    async def _load_dict(self, data: dict):
        content = data.get("data", {})
        if not content:
            return

        messages = content.get("messages", {})
        news = content.get("news", {})
        notification = content.get("notification", {})
        last_group_persona_time = content.get("last_group_persona_time", time.time())
        last_member_persona_time = content.get("last_member_persona_time", {})
        memory_summary_message_count = content.get("memory_summary_message_count", 0)
        new_words_analysis_count = content.get("new_words_analysis_count", 0)

        if messages:
            await self.message_history.load_dict(messages)

        if self.message_history:
            self.AGENT.last_message = self.message_history[-1].msg_id

        if news:
            self.relative_news.update(news)

        if notification:
            self.NOTICE.load_dict(notification)

        self._last_group_persona_time = last_group_persona_time
        if last_member_persona_time:
            self._last_member_persona_time.update(last_member_persona_time)

        self._memory_summary_message_count = memory_summary_message_count
        self._new_words_analysis_count = new_words_analysis_count

    async def _auto_save(self):
        sleep_time = SETTING_CFG.Groups.AutoSaveTime  # 默认15分钟
        try:
            while True:
                await asyncio.sleep(sleep_time)
                await self.save()

        except asyncio.CancelledError:
            return

    async def save(self):
        def _json_helper(obj: Any):
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return obj.to_dict()

            else:
                raise TypeError(f"{type(obj)} 无法序列化")

        save_path = self.data_path / "group_main_dialog_persist.json"
        atomic_save_json(
            content=self.to_dict(),
            target_path=save_path,
            default=_json_helper
        )
        await self.GROUP_COMMUNITY.save()
        await self.BOT_COMMUNITY.save()
        if self._auto_save_loop is not None:
            self._auto_save_loop.cancel()

        self._auto_save_loop = asyncio.create_task(self._auto_save())

    async def load(self):
        save_path = self.data_path / "group_main_dialog_persist.json"
        if not save_path.is_file():
            return

        try:
            load_content: dict = json.loads(save_path.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError):
            return

        else:
            await self._load_dict(load_content)

    async def _clear_cache(self):
        now = datetime.now()
        for date, news in self.relative_news.copy().items():
            timestamp = datetime.strptime(date, "%y-%m-%d")
            if now.timestamp() - timestamp.timestamp() > 86400:
                self.relative_news.pop(date, None)

        member_list = await self._ensure_member_list()
        if member_list is None:
            return

        for user_id in self.MEMBER_PERSONA.copy():
            if user_id not in member_list:
                self.MEMBER_PERSONA.pop(user_id)

        if self._auto_clear_loop is not None:
            self._auto_clear_loop.cancel()

        self._auto_clear_loop = asyncio.create_task(self._auto_clear_cache())

    async def _auto_clear_cache(self):
        sleep = SETTING_CFG.Groups.ClearCacheInterval  # 默认三天
        try:
            while True:
                await asyncio.sleep(sleep)
                asyncio.create_task(self._clear_cache())

        except asyncio.CancelledError:
            return

    # =========================================================
    # 开关
    # =========================================================

    async def start(self):
        """每个对象只能调用一次，不然容易炸"""
        async def _community_helper():
            try:
                await self.GROUP_COMMUNITY.start(self.message_history)
                await self.BOT_COMMUNITY.start()

            except Exception as E:
                self.logger.error(f"社区模型初始化失败: {E}")
                self.logger.debug(traceback.format_exc())

            else:
                self._community_ready = True

        await self.load()
        pause = not self.group_config.chat
        self.NOTICE.clear()
        self._initiate_memory_tip_notification()
        self._initiate_main_agent()
        tasks: list[asyncio.Task] = [
            asyncio.create_task(self.AGENT.start()),
            asyncio.create_task(self.SPEAKER.initiate()),
            asyncio.create_task(self._check_auto_mission()),
            asyncio.create_task(_community_helper()),
            asyncio.create_task(self.get_group_persona()),
            asyncio.create_task(self.get_active_member_info())
        ]
        if self._auto_save_loop is None:
            self._auto_save_loop = asyncio.create_task(self._auto_save())

        if self._auto_clear_loop is None:
            self._auto_clear_loop = asyncio.create_task(self._auto_clear_cache())

        await asyncio.gather(*tasks, return_exceptions=True)
        if not pause:
            self.resume()

    async def shutdown(self):
        """每个对象只能调用一次，不然容易炸"""
        self.pause()
        await self.save()
        if self._auto_save_loop is not None:
            self._auto_save_loop.cancel()

        if self._auto_clear_loop is not None:
            self._auto_clear_loop.cancel()

        if self._auto_mission_loop is not None:
            self._auto_mission_loop.cancel()

        tasks: list[asyncio.Task] = [
            asyncio.create_task(self.AGENT.shutdown()),
            asyncio.create_task(self.SPEAKER.shutdown()),
            asyncio.create_task(self.BOT_COMMUNITY.close()),
            asyncio.create_task(self.GROUP_COMMUNITY.close())
        ]
        await asyncio.gather(*tasks)
