"""
qq_group.py
TIYA_BOT的Q群对象，负责管理群特有接口
划清职责:
- 群对象/私聊对象: 负责接口、数据存储、命令管理、实现对外功能
- 对话对象: 负责处理消息，调用接口，执行具体的消息逻辑
"""
from __future__ import annotations

__version__ = "3.1.0"

import asyncio
import time
import json
import random

from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast
from enum import Enum

import TIYA.api.group_api as api
from TIYA.dialog.group_dialog import GroupMainDialog, GroupCommandDialog
from TIYA.model.message import MessageManager, GroupMsg, SendMessage, BaseSendItem
from TIYA.auto_fav import get_autofav
from TIYA.aqueue import Aqueue
from TIYA.config import GROUPS_DIR, SETTING_CFG, get_bot_uid
from TIYA.logger import get_logger, BlockHandle
from TIYA.utils import atomic_save_json, timestamp2text
from TIYA.global_vars import QQ_GROUPS


if TYPE_CHECKING:
    from TIYA.dialog.base_dialog import BaseDialog
    from TIYA.utils import DecoratedDict
    from ncatbot.core.message import GroupMessage, MessageChain

_log = get_logger()

class _MemberRole(Enum):
    MEMBER = "member"
    OWNER = "owner"
    ADMIN = "admin"
    ROBOT = "robot"


@dataclass(slots=True)
class Member:
    group_id: str
    user_id: str
    username: str  # 用户名
    nickname: str
    title: str  # 群头衔
    level: str  # 群等级
    join_time: float
    role: _MemberRole
    got_banned: float | None = None  # 被禁言
    muted: float | None = None  # 已屏蔽

    def __str__(self) -> str:
        output = f"QQ号: `{self.user_id}`"
        if self.username:
            output += f"\nQQ名: `{self.username}`"

        if self.nickname:
            output += f"\n群昵称: `{self.nickname}`"

        if self.title:
            output += f"\n群头衔: `{self.title}`"

        return output

    def __hash__(self):
        return hash((self.group_id, self.user_id))

    def __eq__(self, other):
        if not isinstance(other, Member):
            return NotImplemented

        return self.group_id == other.group_id and self.user_id == other.user_id


@dataclass(slots=True)
class MemberList:
    """仅作为容器，不要频繁操作，线程不安全"""
    _group_id: str
    _data: dict[str, Member] = field(default_factory=dict)  # user_id mapping
    _username_mapping: dict[str, set[Member]] = field(init=False, default_factory=lambda : defaultdict(set))
    _nickname_mapping: dict[str, set[Member]] = field(init=False, default_factory=lambda : defaultdict(set))
    _owner_id: str = ""
    _admins: set[str] = field(default_factory=set)
    _robots: set[str] = field(default_factory=set)
    last_update: float = field(default_factory=time.time)

    def __post_init__(self):
        self._username_mapping.clear()
        self._nickname_mapping.clear()
        for member in self._data.values():
            if member.username:
                self._username_mapping[member.username].add(member)

            if member.nickname:
                self._nickname_mapping[member.nickname].add(member)

    def update(self, member_data: list[dict], ban_data: list[dict]):
        if not member_data:
            return

        exist = self._data.copy()
        self._admins.clear()

        for m in member_data:
            user_id = str(m.get("user_id", ""))
            if not user_id:
                continue

            group_id = str(m.get("group_id", ""))
            if group_id != self._group_id:
                continue

            username = str(m.get("nickname", ""))
            nickname = str(m.get("card", ""))
            title = str(m.get("title", ""))
            level = str(m.get("level", "0"))
            join_time = float(m.get("join_time", time.time()))
            role = _MemberRole(str(m.get("role", "member")))
            if m.get("is_robot", False):
                role = _MemberRole.ROBOT

            if role is _MemberRole.OWNER:
                self._owner_id = user_id

            elif role is _MemberRole.ADMIN:
                self._admins.add(user_id)

            elif role is _MemberRole.ROBOT:
                self._robots.add(user_id)

            if user_id in exist:
                member = exist.pop(user_id)
                member.username = username
                member.nickname = nickname
                member.title = title
                member.level = level
                member.join_time = join_time
                member.role = role
                member.got_banned = None

            else:
                member = Member(
                    group_id=group_id,
                    user_id=user_id,
                    username=username,
                    nickname=nickname,
                    title=title,
                    level=level,
                    join_time=join_time,
                    role=role
                )
                self._data[user_id] = member

        for ban in ban_data:
            ban_id = str(ban.get("uin", ""))
            if ban_id not in self._data:
                continue

            uptime = float(ban.get("shutUpTime", 0))
            if not uptime:
                continue

            self._data[ban_id].got_banned = uptime

        for delete_id in exist:
            self.pop(delete_id)

        self.__post_init__()

    def get_member(self, user_id: str = "", user_name: str = "", nickname: str = "") -> list[Member]:
        if not (user_id or user_name or nickname):
            return []

        if user_id:
            if user_id in self._data:
                return [self._data[user_id]]

        if user_name:
            if user_name in self._username_mapping:
                return list(self._username_mapping[user_name])

        if nickname:
            if nickname in self._nickname_mapping:
                return list(self._nickname_mapping[nickname])

        return []

    def get_bot_info(self) -> Member | None:
        bot_uid = get_bot_uid()
        if bot_uid not in self._data:
            return None

        return self._data[bot_uid]

    def add_member(self, member: Member | dict) -> Member | None:
        if isinstance(member, Member):
            if member.user_id in self._data:
                return self._data[member.user_id]

            if member.group_id != self._group_id:
                return None

            self._data[member.user_id] = member
            if member.username:
                self._username_mapping.setdefault(member.username, set()).add(member)

            if member.nickname:
                self._nickname_mapping.setdefault(member.nickname, set()).add(member)

            return member

        elif isinstance(member, dict):
            group_id = str(member.get("group_id"), "")
            if group_id != self._group_id:
                return None

            user_id = str(member.get("user_id", ""))
            if user_id in self._data:
                return self._data[user_id]

            username = str(member.get("nickname", ""))
            nickname = str(member.get("card", ""))
            title = str(member.get("title", ""))
            level = str(member.get("level", "0"))
            join_time = float(member.get("join_time", time.time()))
            role = _MemberRole(str(member.get("role", "member")))
            join_time = float(member.get("join_time", time.time()))
            got_banned = float(member.get("got_banned", 0))
            muted = float(member.get("muted", 0))
            if not got_banned:
                got_banned = None

            if not muted:
                muted = None

            obj = Member(
                group_id=group_id,
                user_id=user_id,
                username=username,
                nickname=nickname,
                title=title,
                level=level,
                join_time=join_time,
                role=role,
                got_banned=got_banned,
                muted=muted
            )
            if not (obj.group_id and obj.user_id):
                raise ValueError("Not enough data")

            self._data[obj.user_id] = obj
            if username:
                self._username_mapping[username].add(obj)

            if nickname:
                self._nickname_mapping[nickname].add(obj)
            return obj

        else:
            raise TypeError

    def pop(self, user_id: str | Member) -> Member | None:
        if isinstance(user_id, Member):
            user_id = user_id.user_id

        if user_id in self._data:
            obj = self._data.pop(user_id)
            user_name_set = self._username_mapping.get(obj.username, set())
            user_name_set.discard(obj)
            nickname_set = self._nickname_mapping.get(obj.nickname, set())
            nickname_set.discard(obj)
            return obj

        return None

    @property
    def get_owner(self) -> Member | None:
        if not self._owner_id:
            return None

        if self._owner_id not in self._data:
            return None

        return self._data[self._owner_id]

    @property
    def get_admins(self) -> set[Member]:
        if not self._admins:
            return set()

        output = set()
        for admin in self._admins:
            if admin not in self._data:
                continue

            output.add(self._data[admin])

        return output

    @property
    def get_robots(self) -> set[Member]:
        if not self._robots:
            return set()

        output = set()
        for robot in self._robots:
            if robot not in self._data:
                continue

            output.add(self._data[robot])

        return output

    @property
    def get_muted(self) -> set[Member]:
        output = set()
        for member in self._data.values():
            if member.muted:
                output.add(member)

        return output

    def __iter__(self):
        return iter(self._data.values())

    def __reversed__(self):
        return reversed(self._data.values())

    def __getitem__(self, item) -> list[Member]:
        return self.get_member(item, item, item)

    def __contains__(self, item: Member | str) -> bool:
        if isinstance(item, Member):
            item = item.user_id

        elif isinstance(item, str):
            pass

        else:
            raise TypeError

        return item in self._data


class QQGroup:
    """群对象"""
    def __init__(
            self,
            group_id : str,
            group_name: str,
            max_member: int,
            group_config: DecoratedDict,
            logger: BlockHandle
    ):
        # 基本数据
        self.group_id: str = group_id
        self.group_name: str = group_name
        self.max_member: int = max_member
        self.data_path = GROUPS_DIR / self.group_id
        self.data_path.mkdir(exist_ok=True, parents=True)
        self.bot_id = get_bot_uid()

        # 群数据
        self.member_list: MemberList | None = None
        self.message_history = MessageManager()
        self.dialog_list: list[BaseDialog] = []  # 对话列表
        self.group_config: DecoratedDict = group_config

        # 工具对象
        self.logger: BlockHandle = logger
        self.aqueue = Aqueue()
        self.main_dialog: GroupMainDialog | None = GroupMainDialog(host=self)
        self.command_dialog: GroupCommandDialog = GroupCommandDialog(host=self)
        self.fav = get_autofav(self.data_path / "fav_index.json", logger=logger)

        # 状态位
        self._enable = False

    @classmethod
    def make_group(cls, data: dict, config: DecoratedDict) -> QQGroup:
        """创建群对象"""
        group_id = str(data["group_id"])
        group_name = data["group_name"]
        max_member = data["max_member_count"]

        logger = _log.create_block(group_id, group_name)
        new_group = QQGroup(group_id, group_name, max_member, config, logger)
        if group_id not in QQ_GROUPS:
            QQ_GROUPS[group_id] = new_group

        return new_group

    # 入口
    async def accept_message(self, msg: GroupMessage):
        if not self._enable:
            return

        parsed_msg = await self.message_history.add_message(msg, self.member_list)
        if parsed_msg is None:
            return

        if not isinstance(parsed_msg, GroupMsg):
            return

        asyncio.create_task(self.log_message(parsed_msg))
        # noinspection PyAsyncCall
        self.aqueue.add_task(self.message_flow(parsed_msg), timeout=-1)

    # 入口
    async def message_flow(self, msg: GroupMsg):
        reorder_dialog: list[BaseDialog] = [self.command_dialog, self.main_dialog] + self.dialog_list
        # 先获取表情
        asyncio.create_task(self.fav.auto_fav(msg))
        for dialog in reorder_dialog:
            skip = await dialog.accept_message(msg)
            if skip:
                return

    # =========================================================
    # 群方法
    # =========================================================

    async def log_message(self, msg: GroupMsg):
        msg_dict = await msg.to_llm(wait=True,timeout=1.5)
        self.logger.info(msg_dict["format_text"])

    async def _ensure_member_list(self):
        if self.member_list is None:
            await self.update_member()

        return self.member_list

    async def update_member(self, force=False):
        """更新群员信息"""
        if self.member_list is not None:
            if not force and time.time() - self.member_list.last_update < SETTING_CFG.Groups.MemberUpdatePeriod:  # 默认300秒
                return

        member_response = asyncio.create_task(self._get_group_member_list())
        ban_response = asyncio.create_task(self._get_group_ban_list())
        await asyncio.gather(member_response, ban_response)
        if member_response.result() is not None and member_response.result()["status"] == "ok":
            member_data = member_response.result()["data"]

        else:
            member_data = []

        if ban_response.result() is not None and ban_response.result()["status"] == "ok":
            ban_data = ban_response.result()["data"]

        else:
            ban_data = []

        if self.member_list is None:
            self.member_list = MemberList(self.group_id)

        self.member_list.update(member_data, ban_data)

    def log_prefix(self) -> str:
        return f"群[{self.group_id}]({self.group_name}) "

    def is_muted(self, user_id: str) -> bool:
        """判断用户是否被bot拉黑"""
        if self.member_list is None:
            return False

        user = self.member_list.get_member(user_id=user_id)
        if not user:
            return False

        user = user[0]
        now = time.time()
        if user.muted is None:
            return False

        if now < user.muted:
            return True

        else:
            user.muted = None
            return False

    async def mute(self, user_id: str, duration: float = None) -> float | None:
        """拉黑他人"""
        if user_id == self.bot_id:
            return None

        member_list = await self._ensure_member_list()
        if duration is None:
            duration = SETTING_CFG.Groups.DefaultMuteTime  # 默认3小时

        if member_list is None:
            return None

        user = member_list.get_member(user_id=user_id)
        if not user:
            return None

        user = user[0]
        uptime = time.time() + duration
        user.muted = uptime
        return uptime

    async def unmute(self, user_id: str):
        """解除拉黑某人"""
        if user_id == self.bot_id:
            return

        member_list = await self._ensure_member_list()
        if member_list is None:
            return

        user = member_list.get_member(user_id=user_id)
        if not user:
            return

        user = user[0]
        user.muted = None

    async def unmute_all(self):
        member_list = await self._ensure_member_list()
        if member_list is None:
            return

        for member in member_list:
            member.muted = False

    def to_dict(self) -> dict:
        dt = datetime.now().timestamp()
        dump_content = {
            "update": timestamp2text(dt),
            "description": "此文件为群对象的持久化文件",
            "version": "0.1.0",
            "data": {
                "enable": self._enable,
                "messages": self.message_history.to_dict(),
            }
        }
        return dump_content

    async def save(self):
        def _json_helper(obj: Any):
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return obj.to_dict()

            else:
                raise TypeError(f"{type(obj)} 无法序列化")

        save_path = self.data_path / "group_persist.json"
        atomic_save_json(
            content=self.to_dict(),
            target_path=save_path,
            default=_json_helper
        )
        await self.main_dialog.save()

    async def _load_dict(self) -> bool:
        save_path = self.data_path / "group_persist.json"
        if not save_path.is_file():
            return True

        try:
            load_content = json.loads(save_path.read_text(encoding="utf-8"))

        except json.JSONDecodeError:
            return True

        content = load_content.get("data", {})
        if not content:
            return True

        enable = content.get("enable", True)
        messages = content.get("messages", {})
        if messages:
            await self.message_history.load_dict(messages)

        return enable

    @property
    def got_banned(self) -> bool:
        """判断bot是否被禁言"""
        if self.member_list is None:
            return False

        bot_info = self.member_list.get_bot_info()
        if bot_info is None:
            return False

        if bot_info.got_banned is None:
            return False

        if bot_info.got_banned > time.time():
            return True

        bot_info.got_banned = None
        return False

    @property
    def muted_list(self) -> set[Member]:
        if self.member_list is None:
            return set()

        return self.member_list.get_muted

    @property
    def enable(self) -> bool:
        return self._enable

    # =========================================================
    # 开关
    # =========================================================

    async def start(self):
        """只能执行一次"""
        await self._load_dict()
        await self._ensure_member_list()
        if self.group_config.enable:
            self.resume()

        await self.main_dialog.start()

    async def shutdown(self):
        """只能执行一次"""
        self.pause()
        for dialog in self.dialog_list:
            dialog.pause()

        await self.save()
        errors: list[Exception] = []
        try:
            await self.aqueue.kill()

        except asyncio.CancelledError:
            raise

        except Exception as error:
            errors.append(error)

        try:
            await self.main_dialog.shutdown()

        except asyncio.CancelledError:
            raise

        except Exception as error:
            errors.append(error)

        if errors:
            raise ExceptionGroup(f"群 [{self.group_id}] 关闭失败", errors)

    def resume(self):
        self._enable = True
        self.group_config.enable = True

    def pause(self):
        self._enable = False
        self.group_config.enable = False

    # =========================================================
    # 群API
    # =========================================================

    async def _get_group_member_list(self) -> dict | None:
        """获取群员列表"""
        timeout = SETTING_CFG.Groups.ApiTimeout  # 默认60秒
        try:
            response = await asyncio.wait_for(api.get_group_member_list(self.group_id), timeout)

        except asyncio.TimeoutError:
            return None

        else:
            return response

    async def _get_group_ban_list(self):
        """获取群禁言列表"""
        timeout = SETTING_CFG.Groups.ApiTimeout  # 默认60秒
        try:
            response = await asyncio.wait_for(api.get_group_ban_list(self.group_id), timeout)

        except asyncio.TimeoutError:
            return None

        else:
            return response

    async def _get_file_message_id(self, filename: str) -> str | None:
        """获取 **刚** 上传的文件消息id(我也不知道为什么上传接口不直接返回消息id，好蠢)"""
        timeout = SETTING_CFG.Groups.ApiTimeout  # 默认60秒
        try:
            response = await asyncio.wait_for(
                api.get_file_message_id(
                    self.group_id,
                    filename,
                    retry_interval=1
                ),
                timeout
            )

        except asyncio.TimeoutError:
            return None

        else:
            return response

    async def say(self, message: GroupMsg | SendMessage | BaseSendItem | str) -> list[GroupMsg | None]:
        """统一转换为 message chain 发送，并把成功的结果包装成 GroupMsg"""
        await self._ensure_member_list()
        timeout = SETTING_CFG.Groups.ApiTimeout  # 默认60秒
        message_chains: list[MessageChain] = []
        origin_send = message
        if isinstance(message, SendMessage):
            message_chains.extend(await message.message_chain_async())

        elif isinstance(message, (BaseSendItem, str)):
            origin_send = SendMessage(message)
            message_chains.extend(await origin_send.message_chain_async())

        elif isinstance(message, GroupMsg):
            origin_send = SendMessage.from_message(message)
            message_chains.extend(await origin_send.message_chain_async())

        else:
            raise TypeError(f"不支持的发送格式: {type(message)}")

        msg_ids: list[str | None] = []
        first_chain = True
        sleep_per_word = SETTING_CFG.Groups.SleepPerWord  # 默认0.6秒
        for send in message_chains:
            try:
                # 模拟打字延迟
                if not first_chain:
                    await asyncio.sleep(0.5)
                    words_sleep = []
                    for item in send.elements:  # type: dict
                        item_type = item.get("type", "")
                        if item_type != "text":
                            continue

                        text_data = item.get("data", {}).get("text", "")
                        words_sleep.append(min(10, len(text_data)) * sleep_per_word + random.uniform(0, 0.5))

                    await asyncio.sleep(sum(words_sleep))

                response = await asyncio.wait_for(api.say(self.group_id, send), timeout)

            except asyncio.TimeoutError:
                msg_ids.append(None)
                continue

            else:
                if "say" in response:
                    say_result = response.get("say", {})
                    if not say_result:
                        msg_ids.append(None)

                    elif say_result["status"] != "ok":
                        _log.info(f"{self.log_prefix()}发言失败了")
                        msg_ids.append(None)

                    else:
                        msg_ids.append(str(say_result["data"]["message_id"]))

                for k, v in response.get("files", {}).items():
                    if not v:
                        msg_ids.append(None)

                    elif v["status"] != "ok":
                        msg_ids.append(None)

                    else:
                        file_msg_id = await self._get_file_message_id(k)
                        msg_ids.append(file_msg_id)

            finally:
                first_chain = False

        if self.member_list is None:
            bot_nickname = ""

        else:
            bot_info = self.member_list.get_bot_info()
            if bot_info is None:
                bot_nickname = ""

            else:
                bot_nickname = bot_info.nickname

        said_messages = GroupMsg.from_send(
            group_id=self.group_id,
            bot_nickname=bot_nickname,
            messages_id=msg_ids,
            send=origin_send,

        )
        for said_msg in said_messages:
            if said_msg is None:
                continue

            mirror = cast(GroupMsg, await self.message_history.add_message(said_msg.copy(), self.member_list))
            asyncio.create_task(self.log_message(mirror))

        return said_messages

    async def upload_group_file(
            self,
            file: str,
            filename: str,
            folder_id: str = '/'
    ) -> tuple[bool, GroupMsg | None]:
        """
        上传群文件
        :param file: 本地缓存系统的哈希id
        :param filename: 上传后的文件名
        :param folder_id: 上传到群文件的哪个文件夹，要先通过API获取文件夹id
        :return:
        """
        await self._ensure_member_list()
        response = await api.upload_group_file(
            self.group_id,
            file,
            filename,
            folder_id
        )
        if response is None:
            return False, None

        if response.get("status") != "ok":
            return False, None

        msg_id = await api.get_file_message_id(
            self.group_id,
            filename,
            retry_interval=5
        )
        if not msg_id:
            _log.warning(
                f"群文件已上传，但消息记录尚未同步: "
                f"group_id=[{self.group_id}], filename=[{filename}]"
            )
            return True, None

        if self.member_list is None:
            bot_nickname = ""

        else:
            bot_info = self.member_list.get_bot_info()
            if bot_info is None:
                bot_nickname = ""

            else:
                bot_nickname = bot_info.nickname

        said_msg = GroupMsg.from_uploaded_file(
            group_id=self.group_id,
            bot_nickname=bot_nickname,
            message_id=msg_id,
            filename=filename,
            hash_name=file,
            folder_id=folder_id
        )
        mirror = cast(GroupMsg, await self.message_history.add_message(said_msg.copy(), self.member_list))
        asyncio.create_task(self.log_message(mirror))
        return True, said_msg
