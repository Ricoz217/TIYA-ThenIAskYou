from __future__ import annotations
"""
message.py
对接ncatbot的消息对象
"""
__version__ = "3.1.0"

import json
import random
import copy
import asyncio
import time
import weakref
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Literal, TYPE_CHECKING, cast
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from hashlib import blake2b
from uuid import uuid4

from TIYA.config import SETTING_CFG, get_bot_uid, get_bot_name, DATA_DIR
from TIYA.utils import AutoMapping, ARLock, httpx_downloader, timestamp2text
from TIYA.file_cache import check_file_exists, get_file_path, add_file_async, get_file_path_async
from TIYA.image_to_text import image2text, is_fav

from ncatbot.core.message import GroupMessage, PrivateMessage, MessageChain, Text, Image, \
    File, Reply, At


if TYPE_CHECKING:
    from TIYA.qq_group import MemberList


_MAPPING_FILE = DATA_DIR / "message" / "image_mapping.json"
_IMAGE_MAPPING: AutoMapping[str] = AutoMapping(_MAPPING_FILE, expire_day=lambda :SETTING_CFG.Common.FileCacheExpire)  # 默认30
_IMAGE_DOWNLOAD_LOCK = asyncio.Lock()
_IMAGE_DOWNLOAD_TASK: dict[tuple, asyncio.Task[str]] = {}

async def _create_download_task(url: str, image_name: str) -> str:
    direct_proxy = {
        "https": None,
        "http": None
    }
    image = await httpx_downloader(
        url=url,
        retry=3,
        default=None,
        proxy=direct_proxy
    )
    if image is None or not image.ok:
        return ""

    try:
        hash_name = await add_file_async(image.content, "image", image_name)

    except TypeError:
        return ""

    else:
        if hash_name:
            return hash_name

        return ""

async def _download_image(url: str, image_name: str) -> str:
    async with _IMAGE_DOWNLOAD_LOCK:
        key = (url, image_name)
        download_task = _IMAGE_DOWNLOAD_TASK.get(key, None)
        if download_task is None:
            download_task = _IMAGE_DOWNLOAD_TASK[key] = asyncio.create_task(_create_download_task(url, image_name))

    await asyncio.shield(download_task)
    if not download_task.done() or download_task.cancelled() or download_task.exception() is not None:
        return ""

    async with _IMAGE_DOWNLOAD_LOCK:
        # noinspection PyAsyncCall
        _IMAGE_DOWNLOAD_TASK.pop(key, None)

    return download_task.result()


@dataclass()
class BaseSendItem:
    """用于发送的基本内容，仅作为标记"""
    ...


@dataclass(slots=True)
class SendReply(BaseSendItem):
    origin_message_id: str


@dataclass(slots=True)
class SendAt(BaseSendItem):
    at_user_id: str


@dataclass(slots=True)
class SendBreak(BaseSendItem):
    ...


@dataclass(slots=True)
class SendText(BaseSendItem):
    text: str


@dataclass(slots=True)
class SendImage(BaseSendItem):
    image: str


@dataclass(slots=True)
class SendSetu(BaseSendItem):
    setu: str


@dataclass(slots=True)
class SendFile(BaseSendItem):
    file: str  # hash_name


class SendMessage:
    ITEM_TYPE_MAPPING = {
        "reply": SendReply,
        "at": SendAt,
        "break": SendBreak,
        "text": SendText,
        "image": SendImage,
        "file": SendFile,
        "fav": SendImage
    }
    def __init__(self, *args):
        self._sequence: list[BaseSendItem] = []
        for item in args:
            if isinstance(item, str):
                self._sequence.append(SendText(item))
                continue

            if not isinstance(item, BaseSendItem):
                continue

            self._sequence.append(item)

    def append(self, item: BaseSendItem | str):
        if isinstance(item, str):
            self._sequence.append(SendText(item))

        elif isinstance(item, BaseSendItem):
            self._sequence.append(item)

        else:
            raise TypeError(f"不支持的发送消息类型: {type(item)}")

    def extend(self, items: list[BaseSendItem] | SendMessage):
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, BaseSendItem):
                    raise TypeError(f"不支持的发送消息类型: {type(item)}")

            self._sequence.extend(items)

        elif isinstance(items, SendMessage):
            self._sequence.extend(items.items)

        else:
            raise TypeError(f"不支持的发送消息类型: {type(items)}")

    def clear(self):
        self._sequence.clear()

    def pop(self, index: int = -1):
        self._sequence.pop(index)

    def copy(self) -> SendMessage:
        new_instance = SendMessage()
        new_instance._sequence = self.items
        return new_instance

    @classmethod
    def from_dict(cls, data: dict) -> SendMessage:
        """来自 LLM 的字典，不是持久化的字典"""
        items = []
        sequence = data.get("message_sequence", [])
        content = data.get("message_content", {})
        if not sequence:
            return SendMessage()

        last_key_type = ""
        for key in sequence:
            if not (isinstance(key, str) and key):
                continue

            if key not in content:
                continue

            item_content: dict = content[key]
            key_type = key.split('_')[0].lower()
            if key_type not in cls.ITEM_TYPE_MAPPING:
                continue

            if key_type == "break":
                items.append(SendBreak())


            else:
                if not item_content:
                    continue

                # 自动在文字和图片中间添加 break
                if key_type in {"image", "fav"}:
                    if last_key_type == "text":
                        items.append(SendBreak())


                elif key_type == "text":
                    if last_key_type in {"image", "fav"}:
                        items.append(SendBreak())

                items.append(cls.ITEM_TYPE_MAPPING[key_type](list(item_content.values())[0]))

            last_key_type = key_type


        return SendMessage(*items)

    @classmethod
    def from_message(cls, message: BaseMsg | DirectMsg | ProcessMsg) -> SendMessage:
        items = []
        content = []
        if isinstance(message, (DirectMsg | ProcessMsg)):
            content.append(message)

        elif isinstance(message, BaseMsg):
            content = message.content

        else:
            raise TypeError("错误的类型")

        for item in content:
            if isinstance(item, TextMsg):
                items.append(SendText(item.text))

            elif isinstance(item, ImgMsg):
                items.append(SendImage(item.hash_name or item.url))

            elif isinstance(item, ReplyMsg):
                items.append(SendReply(item.msg_id))

            elif isinstance(item, AtMsg):
                items.append(SendAt(item.user_id))

            elif isinstance(item, SetuMsg):
                items.append(SendSetu(item.hash_name))

        return SendMessage(*items)

    def _to_message_chain(self) -> list[MessageChain]:
        message_chains = []
        send_list = []
        for item in self._sequence:
            if isinstance(item, SendText):
                send_list.append(Text(item.text))

            elif isinstance(item, SendImage):
                image = item.image
                if not (image.startswith("http://") or image.startswith("https://")):
                    try:
                        file = get_file_path(image)

                    except FileNotFoundError:
                        continue

                    else:
                        image = str(file)

                send_list.append(Image(image))

            elif isinstance(item, SendBreak):
                if not send_list:
                    continue

                message_chains.append(MessageChain(send_list))
                send_list.clear()

            elif isinstance(item, SendAt):
                send_list.append(At(item.at_user_id))

            elif isinstance(item, SendReply):
                send_list.append(Reply(item.origin_message_id))

            elif isinstance(item, SendFile):
                try:
                    file = get_file_path(item.file)

                except FileNotFoundError:
                    continue

                else:
                    send_list.append(File(str(file)))

            elif isinstance(item, SendSetu):
                try:
                    file = get_file_path(item.setu)

                except FileNotFoundError:
                    continue

                else:
                    send_list.append(Image(str(file)))

        if send_list:
            message_chains.append(MessageChain(send_list))

        return message_chains

    async def message_chain_async(self) -> list[MessageChain]:
        """使用异步文件系统的转换消息链方法，不会堵塞事件循环"""
        message_chains = []
        send_list = []
        for item in self._sequence:
            if isinstance(item, SendText):
                send_list.append(Text(item.text))

            elif isinstance(item, SendImage):
                image = item.image
                if not (image.startswith("http://") or image.startswith("https://")):
                    try:
                        file = await get_file_path_async(image)

                    except FileNotFoundError:
                        continue

                    else:
                        image = str(file)

                send_list.append(Image(image))

            elif isinstance(item, SendBreak):
                if not send_list:
                    continue

                message_chains.append(MessageChain(send_list))
                send_list.clear()

            elif isinstance(item, SendAt):
                send_list.append(At(item.at_user_id))

            elif isinstance(item, SendReply):
                send_list.append(Reply(item.origin_message_id))

            elif isinstance(item, SendFile):
                try:
                    file = await get_file_path_async(item.file)

                except FileNotFoundError:
                    continue

                else:
                    send_list.append(File(str(file)))

            elif isinstance(item, SendSetu):
                try:
                    file = await get_file_path_async(item.setu)

                except FileNotFoundError:
                    continue

                else:
                    send_list.append(Image(str(file)))

        if send_list:
            message_chains.append(MessageChain(send_list))

        return message_chains

    @property
    def message_chain(self) -> list[MessageChain]:
        return self._to_message_chain()

    @property
    def items(self) -> list[BaseSendItem]:
        return self._sequence.copy()

    def __add__(self, other) -> SendMessage:
        if not isinstance(other, SendMessage):
            raise TypeError(f"不支持的发送消息类型: {type(other)}")

        new_instance = SendMessage()
        new_instance.extend(self)
        new_instance.extend(other)
        return new_instance

    def __iter__(self):
        return iter(self._sequence)


@dataclass(frozen=True, slots=True)
class RelativeCalData:
    text: str = ""
    semantic_text: str = ""
    media_ids: tuple[str, ...] = ()
    reply_to: str | None = None
    mention_ids: frozenset[str] = frozenset()
    pending_media: bool = False


@dataclass(slots=True, kw_only=True, weakref_slot=True)
class BaseMsg:
    msg_id: str
    user_id: str
    username: str
    nickname: str
    content: list[DirectMsg | ProcessMsg] = field(default_factory=list)
    time: float = field(default_factory=time.time)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    broken: bool = False  # 标记消息处理失败，不再重试、等待
    _process_task: asyncio.Task | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    # 指针
    previous_message: weakref.ReferenceType[BaseMsg] | None = None
    next_message: weakref.ReferenceType[BaseMsg] | None = None

    def _format_text_llm(self) -> dict:
        output = {
            "message_type": "普通消息",
            "message_tips": [],
            "message_id": self.msg_id,
            "message_time": timestamp2text(self.time),
            "user_id": self.user_id,
            "username": self.username,
            "nickname": self.nickname,
            "message_sequence": [],
            "message_content": {}
        }
        text_index = 1
        img_index = 1
        at_index = 1
        file_index = 1
        etc_index = 1
        setu_index = 1
        sequence = output["message_sequence"]
        data = output["message_content"]
        tips = output["message_tips"]
        bot_id = get_bot_uid()
        interaction_text = "".join(
            item.text for item in self.content if isinstance(item, TextMsg)
        ).strip()
        format_text = f"{self.nickname or self.username}({self.user_id}): "
        if bot_id and bot_id == self.user_id:
            tips.append("BOT(你自己)发送的消息")

        for c in self.content:
            if isinstance(c, ReplyMsg):
                key = "reply"
                reply_content = {"origin_message_id": c.msg_id}
                origin = c.origin
                if origin is not None:
                    reply_content["origin_user_id"] = origin.user_id
                    reply_content["origin_username"] = origin.username
                    reply_content["origin_nickname"] = origin.nickname
                    reply_content["origin_message_time"] = timestamp2text(origin.time)
                    try:
                        reply_content["origin_message_format_text"] = str(origin)

                    except Exception as e:
                        reply_content["origin_message_format_text"] = f"原消息格式化失败: {type(e).__name__}"

                    setu = next(
                        (item for item in origin if isinstance(item, SetuMsg)),
                        None,
                    )
                    if setu is not None:
                        tips.append(_format_setu_reply_tip(setu, interaction_text))

                    elif origin.user_id == bot_id:
                        tips.append(f"回复你的消息，原消息id: [{c.msg_id}]")

                    else:
                        origin_name = origin.nickname or origin.username or origin.user_id
                        tips.append(f"回复[{origin_name}]的消息，原消息id: [{c.msg_id}]")

                else:
                    reply_content["origin_user_id"] = "原消息已丢失"
                    reply_content["origin_username"] = "原消息已丢失"
                    reply_content["origin_nickname"] = "原消息已丢失"
                    reply_content["origin_message_time"] = "原消息已丢失"
                    reply_content["origin_message_format_text"] = "原消息已丢失"

                data[key] = reply_content
                format_text += f"<reply:{c.msg_id}>"

            elif isinstance(c, TextMsg):
                key = f"text_{text_index}"
                data[key] = {"text": c.text}
                format_text += c.text
                text_index += 1

            elif isinstance(c, ImgMsg):
                key = f"image_{img_index}"
                if c.detail:
                    last_key = next(reversed(c.detail))

                else:
                    last_key = None

                data[key] = {
                    "image_name": c.name,
                    "hash_name": c.hash_name,
                    "image_type": c.type,
                    "description": c.description if c.done else "未解析完毕或已过期图片",
                    "last_query": {last_key: c.detail[last_key]} if last_key else {}
                }
                format_text += f"<image:{c.description if c.done else '未解析完毕或已过期图片'}>"
                img_index += 1

            elif isinstance(c, AtMsg):
                key = f"at_{at_index}"
                data[key] = {
                    "at_user_id": c.user_id,
                    "at_username": c.username,
                    "at_nickname": c.nickname
                }
                if c.user_id and c.user_id == bot_id:
                    tips.append("@了你")

                elif c.nickname or c.username or c.user_id:
                    tips.append(f"@了{c.nickname or c.username or c.user_id}")

                format_text += f"<at:{c.user_id}>"
                at_index += 1

            elif isinstance(c, RawMsg):
                key = f"etc_{etc_index}"
                data[key] = {
                    "raw_message_type": c.type,
                    "raw_message_content": c.content.copy()
                }

                format_text += f"<etc:{c.type}>"
                etc_index += 1

            elif isinstance(c, FileMsg):
                key = f"file_{file_index}"
                data[key] = {
                    "filename": c.filename,
                    "hash_name": c.hash_name,
                    "folder_id": c.folder_id
                }

                format_text += f"<file:{c.filename}>"
                file_index += 1

            elif isinstance(c, SetuMsg):
                key = f"setu_{setu_index}"
                data[key] = {
                    "illust_id": c.illust_id,
                    "page": c.page,
                    "hash_name": c.hash_name,
                    "artist_id": c.artist_id,
                    "artist_name": c.artist_name,
                    "type": c.type,
                    "title": c.title,
                    "caption": c.caption,
                    "tags": c.tags.copy()
                }

                output["message_type"] = "色图消息"
                format_text += f"<setu: id[{c.illust_id}] 标题[{c.title}] 画师[{c.artist_name or c.artist_id}] 标签[{c.tags}]>"
                setu_index += 1

            else:
                raise TypeError

            sequence.append(key)

        output["format_text"] = format_text
        return output

    def parse_content(self, data: list[dict]):
        """改为按message列表解析，不再使用正则处理CQ码"""
        for msg in  data:
            msg_type: str = msg.get("type")
            msg_data: dict = msg.get("data", {})
            if not (msg_type and msg_data):
                continue

            if msg_type == "text":
                self.content.append(TextMsg(text=msg_data.get("text", "")))

            elif msg_type == "at":
                self.content.append(AtMsg(user_id=str(msg_data.get("qq", ""))))

            elif msg_type == "reply":
                self.content.insert(0, ReplyMsg(msg_id=str(msg_data.get("id"))))

            elif msg_type == "image":
                img_name = msg_data.get("file", "")
                url = msg_data.get("url", "")
                self.content.append(ImgMsg(name=img_name, url=url))

            else:
                self.content.append(RawMsg(type=msg_type, content=msg_data))

    def to_dict(self) -> dict:
        base_dict = {
            "msg_id": self.msg_id,
            "user_id": self.user_id,
            "username": self.username,
            "nickname": self.nickname,
            "content": [c.to_dict() for c in self.content],
            "time": self.time,
            "event": self.done,
            "broken": self.broken
        }
        return base_dict

    async def to_llm(self, wait: bool = False, timeout: float = None) -> dict:
        """
        发送到LLM的信息，改为返回json字典
        :param wait: 是否等待解析
        :param timeout: 默认60秒
        :return:
        """
        if not wait or self.broken:
            return self._format_text_llm()

        if timeout is None:
            timeout = SETTING_CFG.Message.MessageParseTimeout  # 默认15秒

        futures = [c.wait() for c in self.content if isinstance(c, ProcessMsg)]
        try:
            await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=timeout)

        except asyncio.TimeoutError:
            return self._format_text_llm()

        return self._format_text_llm()

    @classmethod
    def from_dict(cls, data: dict) -> BaseMsg:
        raise NotImplemented

    async def process(self, **kwargs):
        raise NotImplemented

    async def pure_text(self, wait: bool = False, timeout: float = None) -> str:
        """只包含普通消息和图片信息"""
        output = ""
        llm_dict = await self.to_llm(wait, timeout)
        data: dict[str, dict] = llm_dict["message_content"]
        for k, v in data.items():
            if k.startswith("text"):
                output += v["text"]

            elif k.startswith("image"):
                output += f"<image:{v['description']}>"

        return output

    def copy(self) -> BaseMsg:
        """返回新的消息对象，切断上下引用"""
        return self.__copy__()

    def __iter__(self):
        return iter(self.content)

    def __copy__(self) -> BaseMsg:
        """返回新的消息对象，切断上下引用"""
        raise NotImplemented

    def __hash__(self):
        return hash(self.msg_id)

    def __eq__(self, other):
        if not isinstance(other, BaseMsg):
            return NotImplemented

        return self.msg_id == other.msg_id

    def to_relative_cal(self) -> RelativeCalData:
        """Return structured evidence for lightweight relatedness calculation."""
        text_parts: list[str] = []
        semantic_parts: list[str] = []
        media_ids: list[str] = []
        reply_to: str | None = None
        mention_ids: set[str] = set()
        pending_media = False

        for item in self.content:
            if isinstance(item, TextMsg):
                if text := item.text.strip():
                    text_parts.append(text)

            elif isinstance(item, ReplyMsg):
                reply_to = item.msg_id or reply_to

            elif isinstance(item, AtMsg):
                if item.user_id:
                    mention_ids.add(item.user_id)

            elif isinstance(item, ImgMsg):
                if item.hash_name:
                    media_ids.append(f"image:{item.hash_name}")
                if description := item.description.strip():
                    semantic_parts.append(description)
                elif not self.broken:
                    pending_media = True

            elif isinstance(item, FileMsg):
                if item.filename.strip():
                    semantic_parts.append(item.filename.strip())

            elif isinstance(item, SetuMsg):
                media_ids.append(f"setu:{item.illust_id}:{item.page}")
                setu_text = " ".join(
                    value
                    for value in (
                        item.title.strip(),
                        item.artist_name.strip(),
                        item.caption.strip(),
                        " ".join(tag.strip() for tag in item.tags if tag.strip()),
                    )
                    if value
                )
                if setu_text:
                    semantic_parts.append(setu_text)

            elif isinstance(item, RawMsg) and item.type == "face":
                face_id = str(item.content.get("id", "")).strip()
                if face_id:
                    media_ids.append(f"face:{face_id}")

        return RelativeCalData(
            text="\n".join(text_parts),
            semantic_text="\n".join(semantic_parts),
            media_ids=tuple(dict.fromkeys(media_ids)),
            reply_to=reply_to,
            mention_ids=frozenset(mention_ids),
            pending_media=pending_media,
        )

    def repeatable_hash(self) -> str:
        """判断复读用的哈希"""
        sequence = []
        for item in self.content:
            if isinstance(item, TextMsg):
                sequence.append([f"text_{item.text}"])

            elif isinstance(item, AtMsg):
                sequence.append([f"at_{item.user_id}"])

            elif isinstance(item, ImgMsg):
                if not item.hash_name:
                    # 破坏哈希
                    sequence.append([f"image_{random.random()}"])

                else:
                    sequence.append([f"image_{item.hash_name}"])

        if not sequence:
            # 破坏哈希，防止空值
            sequence.append(str(uuid4()))

        json_text = json.dumps(sequence, ensure_ascii=False)
        h = blake2b(json_text.encode(), digest_size=16)
        return h.hexdigest()

    @classmethod
    def from_message(cls, data: GroupMessage | PrivateMessage) -> BaseMsg | None:
        """从ncatbot的消息对象创建"""
        msg_id = str(data.message_id)
        user_id = str(data.user_id)
        username = str(data.sender.nickname) if data.sender.nickname else ""
        nickname = str(data.sender.card) if data.sender.card else ""
        message = data.message
        if not message:
            return None

        new_msg = BaseMsg(
            msg_id=msg_id,
            user_id=user_id,
            username=username,
            nickname=nickname,
        )
        new_msg.parse_content(message)
        return new_msg

    @property
    def done(self) -> bool:
        return self.event.is_set()

    @property
    def is_reply(self) -> bool:
        return any(isinstance(x, ReplyMsg) for x in self.content)

    @property
    def reply(self) -> ReplyMsg | None:
        for item in self.content:
            if isinstance(item, ReplyMsg):
                return item

        return None

    @property
    def is_at(self) -> bool:
        return any(isinstance(x, AtMsg) for x in self.content)

    @property
    def at_ids(self) -> set[str]:
        return {x.user_id for x in self.content if isinstance(x, AtMsg)}

    @property
    def images(self) -> list[ImgMsg]:
        return [img for img in self.content if isinstance(img, ImgMsg)]

    @property
    def text(self) -> list[str]:
        """返回纯文本消息"""
        output = []
        for item in self.content:
            if not isinstance(item, TextMsg):
                continue

            output.append(item.text)

        return output

    def __str__(self):
        """只能用作阅读，不可信，没有任何转移"""
        return self._format_text_llm()["format_text"]


@dataclass(slots=True, kw_only=True, weakref_slot=True)
class GroupMsg(BaseMsg):
    group_id: str
    async def process(self, **kwargs):
        """需要reply_helper和member_list"""
        if self.broken:
            return

        reply_helper = kwargs.get("reply_helper")
        member_list = kwargs.get("member_list")
        tasks: list[asyncio.Task] = []
        process_item: list[ProcessMsg] = []
        for item in self.content:
            if not isinstance(item, ProcessMsg):
                continue

            if isinstance(item, ReplyMsg):
                if not item.msg_id:
                    continue

                if callable(reply_helper):
                    data = reply_helper(item.msg_id)
                    if data is None:
                        item.event.set()
                        continue

                    process_item.append(item)
                    tasks.append(asyncio.create_task(item.process(data)))

            elif isinstance(item, ImgMsg):
                process_item.append(item)
                tasks.append(asyncio.create_task(item.process()))

            elif isinstance(item, AtMsg):
                process_item.append(item)
                tasks.append(asyncio.create_task(item.process(member_list)))

        try:
            await asyncio.gather(*tasks, return_exceptions=True)

        except (asyncio.CancelledError, asyncio.TimeoutError):
            return

        else:
            if not any(not item.done for item in process_item):
                self.event.set()

    def to_dict(self) -> dict:
        base_dict = super(GroupMsg, self).to_dict()
        base_dict["group_id"] = self.group_id
        return base_dict

    def __copy__(self):
        new_copy = GroupMsg(
            msg_id=self.msg_id,
            user_id=self.user_id,
            username=self.username,
            nickname=self.nickname,
            content=self.content.copy(),
            time=self.time,
            group_id=self.group_id
        )
        for _ in range(len(new_copy.content)):
            if isinstance(new_copy.content[_], ReplyMsg):
                old_reply = cast(ReplyMsg, new_copy.content[_])
                new_copy.content[_] = old_reply.copy()
                break

        return new_copy

    @classmethod
    def from_message(cls, data: GroupMessage) -> GroupMsg | None:
        base_msg = BaseMsg.from_message(data)
        if base_msg is None:
            return None

        group_id = str(data.group_id)
        new_msg =GroupMsg(
            msg_id=base_msg.msg_id,
            user_id=base_msg.user_id,
            username=base_msg.username,
            nickname=base_msg.nickname,
            content=base_msg.content.copy(),
            time=base_msg.time,
            group_id=group_id
        )
        if base_msg.done:
            new_msg.event.set()

        return new_msg

    @classmethod
    def from_dict(cls, data: dict) -> GroupMsg | None:
        msg_id = data.get("msg_id", "")
        user_id = data.get("user_id", "")
        group_id = data.get("group_id", "")
        broken = data.get("broken", False)
        raw_content: list[dict] = data.get("content", [])
        if not (group_id and msg_id and user_id):
            return None

        content = []
        for c in raw_content:
            msg_obj = _msg_builder(c)
            if msg_obj is None:
                continue

            content.append(msg_obj)

        new_group = GroupMsg(
            msg_id=msg_id,
            user_id=user_id,
            username=data.get("username", ""),
            nickname=data.get("nickname", ""),
            content=content,
            time=data.get("time", time.time()),
            broken=broken,
            group_id=group_id
        )
        return new_group

    @classmethod
    def from_send(
            cls,
            group_id: str,
            bot_nickname: str,
            messages_id: list[str | None],
            send: SendMessage,
            send_time: float = None
    ) -> list[GroupMsg | None]:
        def _next_time():
            nonlocal send_time
            send_time += 0.1
            return send_time

        def _get_msg_id() -> str | None:
            try:
                _msg_id = next(ids_iter)

            except StopIteration:
                return None

            else:
                return _msg_id

        def _commit_message():
            if not buffer:
                return

            _files: list[FileMsg] = []
            _content = []

            # 先把文件单独拉出来
            for _i in buffer:
                if isinstance(_i, SendText):
                    _content.append(TextMsg(text=_i.text))

                elif isinstance(_i, SendImage):
                    image = _i.image
                    url = ""
                    hash_name = ""
                    if not (image.startswith("http://") or image.startswith("https://")):
                        hash_name = image
                        if not check_file_exists(hash_name):
                            continue

                    else:
                        url = image

                    _content.append(ImgMsg(
                        name=hash_name or url,
                        url=url,
                        hash_name=hash_name
                    ))

                elif isinstance(_i, SendAt):
                    _content.append(AtMsg(
                        user_id=_i.at_user_id
                    ))

                elif isinstance(_i, SendReply):
                    _content.append(ReplyMsg(
                        msg_id=_i.origin_message_id
                    ))

                elif isinstance(_i, SendFile):
                    _files.append(FileMsg(
                        filename=_i.file,
                        hash_name=_i.file
                    ))

            buffer.clear()

            # 创建基本内容发言消息
            if _content:
                _msg_id = _get_msg_id()
                if _msg_id is not None:
                    _new_message = GroupMsg(
                        group_id=group_id,
                        msg_id=_msg_id,
                        user_id=bot_uid,
                        username=bot_username,
                        nickname=bot_nickname,
                        content=_content.copy(),
                        time=_next_time()
                    )
                    split_message.append(_new_message)

                else:
                    split_message.append(None)

            # 再创建单独文件消息
            for _f in _files:
                _msg_id = _get_msg_id()
                if _msg_id is None:
                    split_message.append(None)
                    continue

                _new_message = GroupMsg(
                    group_id=group_id,
                    msg_id=_msg_id,
                    user_id=bot_uid,
                    username=bot_username,
                    nickname=bot_nickname,
                    content=[_f],
                    time=_next_time()
                )
                split_message.append(_new_message)

        if send_time is None:
            send_time = time.time()

        bot_uid = get_bot_uid()
        bot_username = get_bot_name()
        ids_iter = iter(messages_id)
        split_message: list[GroupMsg | None] = []
        buffer: list[BaseSendItem] = []
        for item in send.items:
            if not isinstance(item, BaseSendItem):
                continue

            if isinstance(item, SendBreak):
                _commit_message()

            else:
                buffer.append(item)

        _commit_message()

        return split_message

    @classmethod
    def from_uploaded_file(
            cls,
            group_id: str,
            bot_nickname: str,
            message_id: str,
            filename: str,
            hash_name: str,
            folder_id: str = '/',
            send_time: float = None
    ) -> GroupMsg:
        bot_uid = get_bot_uid()
        bot_username = get_bot_name()
        if send_time is None:
            send_time = time.time()

        return GroupMsg(
            group_id=group_id,
            msg_id=message_id,
            user_id=bot_uid,
            username=bot_username,
            nickname=bot_nickname,
            content=[FileMsg(
                filename=filename,
                hash_name=hash_name,
                folder_id=folder_id
            )],
            time=send_time
        )


@dataclass(slots=True, kw_only=True, weakref_slot=True)
class PrivateMsg(BaseMsg):
    async def process(self, **kwargs):
        if self.broken:
            return

        reply_helper = kwargs.get("reply_helper")
        tasks: list[asyncio.Task] = []
        process_items: list[ProcessMsg] = []
        for item in self.content:
            if not isinstance(item, ProcessMsg):
                continue

            if isinstance(item, ReplyMsg):
                if not item.msg_id:
                    item.event.set()
                    continue

                data = reply_helper(item.msg_id) if callable(reply_helper) else None
                if data is None:
                    item.event.set()
                    continue

                process_items.append(item)
                tasks.append(asyncio.create_task(item.process(data)))

            elif isinstance(item, ImgMsg):
                process_items.append(item)
                tasks.append(asyncio.create_task(item.process()))

            else:
                item.event.set()

        try:
            await asyncio.gather(*tasks, return_exceptions=True)

        except (asyncio.CancelledError, asyncio.TimeoutError):
            return

        if not any(not item.done for item in process_items):
            self.event.set()

    def __copy__(self):
        new_copy = PrivateMsg(
            msg_id=self.msg_id,
            user_id=self.user_id,
            username=self.username,
            nickname=self.nickname,
            content=self.content.copy(),
            time=self.time,
            broken=self.broken
        )
        for index, item in enumerate(new_copy.content):
            if isinstance(item, ReplyMsg):
                new_copy.content[index] = item.__copy__()
                break

        if self.done:
            new_copy.event.set()

        return new_copy

    @classmethod
    def from_message(cls, data: PrivateMessage) -> PrivateMsg | None:
        base_msg = BaseMsg.from_message(data)
        if base_msg is None:
            return None

        new_msg = PrivateMsg(
            msg_id=base_msg.msg_id,
            user_id=base_msg.user_id,
            username=base_msg.username,
            nickname=base_msg.nickname,
            content=base_msg.content.copy(),
            time=base_msg.time
        )
        if base_msg.done:
            new_msg.event.set()

        return new_msg

    @classmethod
    def from_dict(cls, data: dict) -> PrivateMsg | None:
        msg_id = data.get("msg_id", "")
        user_id = data.get("user_id", "")
        if not (msg_id and user_id):
            return None

        content = []
        for raw_item in data.get("content", []):
            msg_obj = _msg_builder(raw_item)
            if msg_obj is not None:
                content.append(msg_obj)

        new_msg = PrivateMsg(
            msg_id=msg_id,
            user_id=user_id,
            username=data.get("username", ""),
            nickname=data.get("nickname", ""),
            content=content,
            time=data.get("time", time.time()),
            broken=data.get("broken", False)
        )
        if data.get("event", False):
            new_msg.event.set()

        return new_msg

    @classmethod
    def from_send(
            cls,
            messages_id: list[str | None],
            send: SendMessage,
            send_time: float = None
    ) -> list[PrivateMsg | None]:
        def _to_content(_items: list[BaseSendItem]) -> list[DirectMsg | ProcessMsg]:
            _content: list[DirectMsg | ProcessMsg] = []
            for _item in _items:
                if isinstance(_item, SendText):
                    _content.append(TextMsg(text=_item.text))

                elif isinstance(_item, SendImage):
                    image = _item.image
                    if image.startswith(("http://", "https://")):
                        _content.append(ImgMsg(name=image, url=image))
                        continue

                    if not check_file_exists(image):
                        continue

                    _content.append(ImgMsg(name=image, hash_name=image))

                elif isinstance(_item, SendReply):
                    _content.append(ReplyMsg(msg_id=_item.origin_message_id))

                elif isinstance(_item, SendAt):
                    _content.append(AtMsg(user_id=_item.at_user_id))

                elif isinstance(_item, SendFile):
                    if not check_file_exists(_item.file):
                        continue

                    _content.append(FileMsg(filename=_item.file, hash_name=_item.file))

            return _content

        if send_time is None:
            send_time = time.time()

        split_items: list[list[BaseSendItem]] = []
        buffer: list[BaseSendItem] = []
        for item in send.items:
            if isinstance(item, SendBreak):
                if buffer:
                    split_items.append(buffer)
                    buffer = []
                continue

            buffer.append(item)

        if buffer:
            split_items.append(buffer)

        bot_uid = get_bot_uid()
        bot_username = get_bot_name()
        result: list[PrivateMsg | None] = []
        message_id_iter = iter(messages_id)
        for index, items in enumerate(split_items):
            content = _to_content(items)
            if not content:
                continue

            try:
                message_id = next(message_id_iter)
            except StopIteration:
                message_id = None

            if message_id is None:
                result.append(None)
                continue

            result.append(PrivateMsg(
                msg_id=message_id,
                user_id=bot_uid,
                username=bot_username,
                nickname="",
                content=content,
                time=send_time + index * 0.1
            ))

        return result


@dataclass(slots=True, kw_only=True)
class DirectMsg:
    """抽象类，无需等待"""
    def to_dict(self) -> dict:
        return {"msg_type": _MSG_TYPE_MAPPING_R[self.__class__.__name__]}

    @classmethod
    def from_dict(cls, data: dict) -> DirectMsg:
        raise NotImplemented


@dataclass(slots=True, kw_only=True)
class ProcessMsg:
    """抽象类，需要等待，带有process, event, wait, done"""
    event: asyncio.Event = field(default_factory=asyncio.Event)

    async def process(self, *args, **kwargs):
        raise NotImplemented

    async def wait(self):
        await self.event.wait()

    def to_dict(self) -> dict:
        return {
            "msg_type": _MSG_TYPE_MAPPING_R[self.__class__.__name__],
            "event": self.done
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProcessMsg:
        raise NotImplemented

    @property
    def done(self) -> bool:
        return self.event.is_set()


@dataclass(slots=True, kw_only=True)
class TextMsg(DirectMsg):
    text: str

    def to_dict(self) -> dict:
        base_dict = super(TextMsg, self).to_dict()
        base_dict["text"] = self.text
        return base_dict

    @classmethod
    def from_dict(cls, data: dict) -> TextMsg:
        return TextMsg(text=data.get("text", ""))


@dataclass(slots=True, kw_only=True)
class ReplyMsg(ProcessMsg):
    msg_id: str
    _origin: weakref.ReferenceType[BaseMsg] | None = None

    async def process(self, *args, **kwargs):
        if self.event.is_set():
            return

        if not args:
            return

        data = args[0]
        if not isinstance(data, BaseMsg):
            return

        self._origin = weakref.ref(data)
        self.event.set()

    def to_dict(self) -> dict:
        base_dict = super(ReplyMsg, self).to_dict()
        base_dict["msg_id"] = self.msg_id
        return base_dict

    def copy(self) -> ReplyMsg:
        return self.__copy__()

    @classmethod
    def from_dict(cls, data: dict) -> ReplyMsg | None:
        msg_id = data.get("msg_id", "")
        if not msg_id:
            return None

        return ReplyMsg(msg_id=msg_id)

    @property
    def origin(self) -> GroupMsg | None:
        if self._origin is None:
            return None

        return self._origin()

    @property
    def alive(self) -> bool:
        if self.origin is None:
            return False

        return True

    @property
    def user_id(self) -> str:
        if not self.alive:
            return ""

        return self.origin.user_id

    @property
    def username(self) -> str:
        if not self.alive:
            return ""

        return self.origin.username

    @property
    def nickname(self) -> str:
        if not self.alive:
            return ""

        return self.origin.nickname

    @property
    def time(self) -> float | None:
        if not self.alive:
            return None

        return self.origin.time

    def __copy__(self) -> ReplyMsg:
        new_instance = ReplyMsg(
            msg_id=self.msg_id,
            _origin=self._origin
        )
        new_instance.event.clear()
        return new_instance


@dataclass(slots=True, kw_only=True)
class ImgMsg(ProcessMsg):
    name: str
    url: str = ""
    hash_name: str = ""
    description: str = ""
    detail: dict = field(default_factory=dict)  # 存储query
    type: Literal["FAV", "IMAGE"] = "IMAGE"

    async def process(self, *args, **kwargs):
        if self.event.is_set():
            return

        if self.description:
            self.event.set()
            return

        if self.name:
            hash_name = _IMAGE_MAPPING.get(self.name)
            if hash_name is not None:
                self.hash_name = hash_name

        if not (self.hash_name or self.url):
            return

        if not self.hash_name:
            hash_name = await _download_image(self.url, self.name)
            if hash_name:
                self.hash_name = hash_name

            else:
                return

        if self.name:
            _IMAGE_MAPPING[self.name] = self.hash_name

        description = await image2text(
            hash_name=self.hash_name,
            image_name=self.name,
            mode="NORMAL",
            prefer_fav=True
        )
        if is_fav(self.hash_name):
            self.type = "FAV"

        if description:
            self.description = description
            self.event.set()

    def to_dict(self):
        base_dict = super(ImgMsg, self).to_dict()
        base_dict["name"] = self.name
        base_dict["url"] = self.url
        base_dict["hash_name"] = self.hash_name
        base_dict["description"] = self.description
        base_dict["detail"] = copy.deepcopy(self.detail)
        base_dict["type"] = self.type
        return base_dict

    @classmethod
    def from_dict(cls, data: dict) -> ImgMsg | None:
        name = data.get("name", "")
        url = data.get("url", "")
        hash_name = data.get("hash_name", "")
        if not (name and hash_name):
            return None

        description = data.get("description", "")
        detail = data.get("detail", {})
        _type = data.get("type", "IMAGE")

        if _type not in {"IMAGE", "FAV"}:
            return None

        new_img = ImgMsg(
            name=name,
            url=url,
            hash_name=hash_name,
            description=description,
            detail=detail,
            type=_type
        )
        if new_img.description:
            new_img.event.set()

        return new_img


@dataclass(slots=True, kw_only=True)
class RawMsg(DirectMsg):
    type: str
    content: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        base_dict = super(RawMsg, self).to_dict()
        base_dict["type"] = self.type
        base_dict["content"] = copy.deepcopy(self.content)
        return base_dict

    @classmethod
    def from_dict(cls, data: dict) -> RawMsg:
        return RawMsg(
            type=data.get("type", ""),
            content=data.get("content", {})
        )


@dataclass(slots=True, kw_only=True)
class AtMsg(ProcessMsg):
    user_id: str
    username: str = ""
    nickname: str = ""

    async def process(self, *args, **kwargs):
        if self.event.is_set():
            return

        if not args:
            return

        if not (hasattr(args[0], "get_member") and callable(args[0].get_member)):
            return

        member_list: MemberList = args[0]
        member = member_list.get_member(user_id=self.user_id)
        if not member:
            self.event.set()
            return

        member = member[0]
        self.username = member.username
        self.nickname = member.nickname
        self.event.set()

    def to_dict(self) -> dict:
        base_dict = super(AtMsg, self).to_dict()
        base_dict["user_id"] = self.user_id
        base_dict["username"] = self.username
        base_dict["nickname"] = self.nickname
        return base_dict

    @classmethod
    def from_dict(cls, data: dict) -> AtMsg | None:
        user_id = data.get("user_id", "")
        if not user_id:
            return None

        new_at = AtMsg(
            user_id=user_id,
            username=data.get("username", ""),
            nickname=data.get("nickname", "")
        )
        if data.get("event", False):
            new_at.event.set()

        return new_at


@dataclass(slots=True, kw_only=True)
class FileMsg(DirectMsg):
    """先只保存自己发送出去的，后续再做他人的文件解析"""
    filename: str
    hash_name: str
    folder_id: str = '/'

    def to_dict(self) -> dict:
        base_dict = super(FileMsg, self).to_dict()
        base_dict["filename"] = self.filename
        base_dict["hash_name"] = self.hash_name
        base_dict["folder_id"] = self.folder_id
        return base_dict

    @classmethod
    def from_dict(cls, data: dict) -> FileMsg | None:
        filename = data.get("filename", "")
        hash_name = data.get("hash_name", "")
        if not (filename and hash_name):
            return None

        folder_id = data.get("folder_id", '/')
        return FileMsg(
            filename=filename,
            hash_name=hash_name,
            folder_id=folder_id
        )


@dataclass(slots=True, kw_only=True)
class SetuMsg(DirectMsg):
    illust_id: str
    page: int
    hash_name: str
    artist_id: int
    artist_name: str
    type: str
    title: str
    caption: str
    tags: list[str]

    def to_dict(self) -> dict:
        base_dict = super(SetuMsg, self).to_dict()
        base_dict.update(asdict(self))
        return base_dict

    @classmethod
    def from_dict(cls, data: dict) -> SetuMsg | None:
        illust_id = data.get("illust_id", "")
        page = data.get("page", None)
        hash_name = data.get("hash_name", "")
        artist_id = data.get("artist_id", None)
        artist_name = data.get("artist_name", "")
        _type = data.get("type", "")
        title = data.get("title", "")
        caption = data.get("caption", "")
        tags = data.get("tags", [])

        if any(not attr for attr in (illust_id, hash_name, _type)):
            return None

        if page is None or artist_id is None:
            return None

        # noinspection PyTypeChecker
        return SetuMsg(
            illust_id=illust_id,
            page=page,
            hash_name=hash_name,
            artist_id=artist_id,
            artist_name=artist_name,
            type=_type,
            title=title,
            caption=caption,
            tags=tags.copy()
        )

def _format_setu_reply_tip(setu: SetuMsg, text: str) -> str:
    target = f"色图[{setu.illust_id}_p{setu.page}]《{setu.title}》"
    try:
        score = Decimal(text)

    except InvalidOperation:
        score = None

    if score is not None and not score.is_nan():
        if score.is_infinite():
            score = Decimal("10") if score > 0 else Decimal("0")

        score = min(Decimal("10"), max(Decimal("0"), score))
        score = score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return f"对{target}打分 {score:.2f} 分"

    if text.startswith("--"):
        return f"对{target}发送互动指令：{text}"

    return f"对{target}进行回复或评价"


_MSG_TYPE_MAPPING = {
    "TIYA_TEXT": TextMsg,
    "TIYA_REPLY": ReplyMsg,
    "TIYA_IMAGE": ImgMsg,
    "TIYA_RAW": RawMsg,
    "TIYA_AT": AtMsg,
    "TIYA_FILE": FileMsg,
    "TIYA_SETU": SetuMsg
}
_MSG_TYPE_MAPPING_R = {v.__name__: k for k, v in _MSG_TYPE_MAPPING.items()}

def _msg_builder(data: dict[str, Any]) -> DirectMsg | ProcessMsg | None:
    msg_type = data.get("msg_type", "")
    if not msg_type:
        return None

    if msg_type not in _MSG_TYPE_MAPPING:
        return None

    return _MSG_TYPE_MAPPING[msg_type].from_dict(data)


class MessageManager:
    """消息管理器，作为历史记录使用, 不做群号和对话用户检测"""
    def __init__(self, max_message: int = None):
        if max_message is None:
            max_message = SETTING_CFG.Common.MaxMessageHistory  # 默认5000

        self._max_message = max_message
        self._lock = ARLock()

        # 双数据结构
        self._message_list: deque[BaseMsg] = deque(maxlen=self._max_message)
        self._message_dict: dict[str, BaseMsg] = {}  # 不保证顺序

        # 记录自己发送过的消息id
        self._bot_id = get_bot_uid()
        self.bot_msg_id: set[str] = set()
        self.user_msg_mapping: dict[str, set[str]] = {}  # 记录每个人发送了什么消息: {user_id: set[msg_id]}
        self._process_fails: dict[str, int] = defaultdict(int)  # 存储初始化失败的消息

    def _index_right(self, item: BaseMsg | str) -> int:
        if isinstance(item, BaseMsg):
            item = item.msg_id

        for i, e in enumerate(reversed(self._message_list)):
            if item == e.msg_id:
                return len(self._message_list) - 1 - i

        raise ValueError(f"<{item}> not in history")

    def _rerank_by_time(self, msg: BaseMsg) -> int:
        """根据使用场景考虑，不再使用二分法。注意输出的是负数"""
        msg_time = msg.time
        n = len(self._message_list)
        if n == 0:
            return 0

        # 高频快路：尾插
        if float(self._message_list[-1].time) <= msg_time:
            return 0

        for i, e in enumerate(reversed(self._message_list)):
            if e.time <= msg_time:
                return -i

        return -len(self._message_list)

    def _parse_new_msg(self, msg: BaseMsg):
        self._message_dict[msg.msg_id] = msg
        if msg.user_id == self._bot_id:
            self.bot_msg_id.add(msg.msg_id)

        self.user_msg_mapping.setdefault(msg.user_id, set()).add(msg.msg_id)

    @staticmethod
    def _weakref_msg(msg: BaseMsg | None) -> weakref.ReferenceType[BaseMsg] | None:
        return weakref.ref(msg) if msg is not None else None

    async def _add(self, msg: BaseMsg):
        if not self._message_list:
            await self.clear()  # 额外清除一次
            msg.previous_message = None
            msg.next_message = None
            self._message_list.append(msg)
            self._parse_new_msg(msg)
            return

        if msg.msg_id in self._message_dict:
            await self._replace(msg)
            return

        # 重排时间，注意边界处理
        index = self._rerank_by_time(msg)
        true_index = len(self._message_list) + index
        pop_item = None

        """重写，缕清思路"""
        # 1. 队列满了，新消息是最旧消息，直接返回不用加入
        if len(self._message_list) == self._max_message:
            if true_index == 0:
                return

            # 2. 获取需要去除的消息
            pop_item = self._message_list.popleft()
            true_index -= 1

            # 清空头指针
            if self._message_list:
                self._message_list[0].previous_message = None

        # 3. 按顺序插入新消息
        if true_index >= len(self._message_list):
            self._message_list.append(msg)
            cur_index = len(self._message_list) - 1

        else:
            self._message_list.insert(true_index, msg)
            cur_index = true_index

        # 4. 统一设置指针
        prev_msg = self._message_list[cur_index - 1] if cur_index > 0 else None
        next_msg = self._message_list[cur_index + 1] if cur_index + 1 < len(self._message_list) else None

        msg.previous_message = self._weakref_msg(prev_msg)
        msg.next_message = self._weakref_msg(next_msg)

        if prev_msg is not None:
            prev_msg.next_message = self._weakref_msg(msg)

        if next_msg is not None:
            next_msg.previous_message = self._weakref_msg(msg)

        # 5. 添加新消息到其他数据结构
        self._parse_new_msg(msg)

        # 6. 处理去除的消息
        if pop_item is None:
            return

        pop_item.previous_message = None
        pop_item.next_message = None
        self._message_dict.pop(pop_item.msg_id, None)

        if pop_item.user_id == self._bot_id:
            self.bot_msg_id.discard(pop_item.msg_id)

        if pop_item.user_id in self.user_msg_mapping:
            self.user_msg_mapping[pop_item.user_id].discard(pop_item.msg_id)

            if not self.user_msg_mapping[pop_item.user_id]:
                del self.user_msg_mapping[pop_item.user_id]

    async def _remove(self, msg_id: str):
        async with self._lock:
            if msg_id not in self._message_dict:
                return

            msg = self._message_dict[msg_id]
            index = self._message_list.index(msg)
            if index == 0:
                if len(self._message_list) > 1:
                    self._message_list[1].previous_message = None

                self._message_list.popleft()

            elif index == len(self._message_list) - 1:
                self._message_list[-2].next_message = None
                self._message_list.pop()

            else:
                self._message_list[index - 1].next_message = msg.next_message
                self._message_list[index + 1].previous_message = msg.previous_message
                self._message_list.remove(msg)

            if msg.user_id == self._bot_id:
                self.bot_msg_id.discard(msg.msg_id)

            if msg.user_id in self.user_msg_mapping:
                self.user_msg_mapping[msg.user_id].discard(msg.msg_id)

                if not self.user_msg_mapping[msg.user_id]:
                    del self.user_msg_mapping[msg.user_id]

            del self._message_dict[msg_id]

    async def _replace(self, msg: BaseMsg):
        async with self._lock:
            if msg.msg_id not in self._message_dict:
                return

            old = self._message_dict[msg.msg_id]
            index = self._index_right(msg)
            msg.previous_message = old.previous_message
            msg.next_message = old.next_message
            if old.previous_message is not None:
                if prev_m := old.previous_message():
                    prev_m.next_message = weakref.ref(msg)

            if old.next_message is not None:
                if next_m := old.next_message():
                    next_m.previous_message = weakref.ref(msg)

            self._message_list[index] = msg
            self._message_dict[msg.msg_id] = msg

    def _reply_helper(self, msg_id: str) -> BaseMsg | None:
        if not msg_id:
            return None

        if msg_id not in self._message_dict:
            return None

        return self._message_dict[msg_id]

    @staticmethod
    def _start_message_process(
            msg: BaseMsg,
            reply_helper,
            group_member_list: MemberList | None,
    ) -> tuple[asyncio.Task, bool]:
        task = msg._process_task
        if task is not None and not task.done():
            return task, False

        task = asyncio.create_task(
            msg.process(reply_helper=reply_helper, member_list=group_member_list)
        )
        msg._process_task = task

        def clear_process_task(completed_task: asyncio.Task) -> None:
            if msg._process_task is completed_task:
                msg._process_task = None

        task.add_done_callback(clear_process_task)
        return task, True

    async def _message_process_helper(
            self,
            msg: BaseMsg,
            group_member_list: MemberList | None,
            *,
            track_failure: bool = True,
    ):
        task, owns_process = self._start_message_process(
            msg,
            self._reply_helper,
            group_member_list,
        )
        await task
        if not (track_failure and owns_process):
            return

        if msg.broken or msg.done:
            self._process_fails.pop(msg.msg_id, None)

        else:
            self._process_fails[msg.msg_id] += 1

    async def _message_process_retry(self, group_member_list: MemberList | None):
        tasks: dict[str, asyncio.Task] = {}
        for k, v in self._process_fails.copy().items():
            if k not in self._message_dict:  # 默认3
                self._process_fails.pop(k, None)
                continue

            msg = self._message_dict[k]
            if msg.broken:
                self._process_fails.pop(k, None)
                continue

            if v > SETTING_CFG.Message.ProcessRetry:
                msg.broken = True
                self._process_fails.pop(k, None)
                continue

            tasks[msg.msg_id] = asyncio.create_task(
                self._message_process_helper(msg, group_member_list)
            )

        await asyncio.gather(*tasks.values())

    async def add_message(
            self,
            data: BaseMsg | GroupMessage | PrivateMessage | dict,
            group_member_list: MemberList = None
    ) -> BaseMsg | None:
        """
        新增消息，也可以直接更新，按msg_id界定
        :param data:
        :param group_member_list:
        :return:
        """
        # 先重试失败任务
        asyncio.create_task(self._message_process_retry(group_member_list))

        if isinstance(data, BaseMsg):
            msg = data

        elif isinstance(data, GroupMessage):
            msg = GroupMsg.from_message(data)

        elif isinstance(data, PrivateMessage):
            msg = PrivateMsg.from_message(data)

        elif isinstance(data, dict):
            if "group_id" in data:
                msg = GroupMsg.from_dict(data)

            else:
                msg = PrivateMsg.from_dict(data)

        else:
            raise TypeError("不支持的消息类型")

        if msg is None:
            return None

        async with self._lock:
            await self._add(msg)
            asyncio.create_task(self._message_process_helper(msg, group_member_list))

        return msg

    async def get_message(self, msg_id: str) -> BaseMsg | None:
        async with self._lock:
            if msg_id not in self._message_dict:
                return None

            return self._message_dict[msg_id]

    async def remove_message(self, msg_id: str):
        if msg_id not in self._message_dict:
            return

        await self._remove(msg_id)

    async def pop(self, msg_id: str):
        msg = await self.get_message(msg_id)
        if msg is None:
            return None

        await self._remove(msg_id)
        return msg_id

    def to_dict(self) -> dict:
        return {msg.msg_id: msg.to_dict() for msg in self._message_list.copy()}

    async def load_dict(self, data: dict):
        if not data:
            return

        msg_buffer: list[BaseMsg] = []
        for msg in data.values():
            if "msg_id" not in msg:
                continue

            if "group_id" in msg:
                new_msg = GroupMsg.from_dict(msg)
                if new_msg is None:
                    continue

                msg_buffer.append(new_msg)

            else:
                new_msg = PrivateMsg.from_dict(msg)
                if new_msg is None:
                    continue

                msg_buffer.append(new_msg)

        if not msg_buffer:
            return

        msg_buffer.sort(key=lambda x: x.time)
        async with self._lock:
            await self.clear()
            last_msg = msg_buffer[0]
            self._parse_new_msg(last_msg)
            self._message_list.append(last_msg)
            for msg in msg_buffer[1:]:
                last_msg.next_message = self._weakref_msg(msg)
                msg.previous_message = self._weakref_msg(last_msg)
                self._message_list.append(msg)
                self._parse_new_msg(msg)
                last_msg = msg

        for msg in self._message_list:
            # 不传member_list，跳过成员信息初始化。
            asyncio.create_task(
                self._message_process_helper(msg, None, track_failure=False)
            )

    async def extend(self, items: list[BaseMsg] | MessageManager):
        if not items:
            return

        if isinstance(items, list):
            if any(not isinstance(item, BaseMsg) for item in items):
                raise TypeError

        elif isinstance(items, MessageManager):
            pass

        else:
            raise TypeError

        async with self._lock:
            for item in items:
                await self._add(item)

            for item in items:
                asyncio.create_task(
                    self._message_process_helper(item, None, track_failure=False)
                )

    async def clear(self):
        """清空记录"""
        async with self._lock:
            for m in self._message_list:
                m.previous_message = None
                m.next_message = None

            self._message_list.clear()
            self._message_dict.clear()
            self.bot_msg_id.clear()
            self.user_msg_mapping.clear()

    # 映射同名函数
    async def append(
            self,
            data: BaseMsg | GroupMessage | PrivateMessage | dict,
            group_member_list: MemberList = None
    ) -> BaseMsg | None:
        return await self.add_message(data, group_member_list)

    async def get(self, msg_id: str) -> BaseMsg | None:
        return await self.get_message(msg_id)

    async def remove(self, msg_id: str):
        await self.remove_message(msg_id)

    # 特殊接口
    def search_by_time(self, message_time: float) -> BaseMsg | None:
        """返回最接近给定时间的消息对象"""
        if not self._message_list:
            return None

        beyond = None
        under = None
        for msg in reversed(self._message_list):
            if msg.time < message_time:
                under = msg
                break

            beyond = msg

        if under is None:
            return beyond

        if beyond is None:
            return under

        under_diff = message_time - under.time
        beyond_diff = beyond.time - message_time
        if under_diff > beyond_diff:
            return beyond

        else:
            return under

    def search_by_reply_user(self, user_id: str) -> list[BaseMsg]:
        """获取回复某人的所有消息"""
        return [m for m in self._message_list if m.reply and m.reply.user_id == user_id]

    def search_by_reply_message(self, message_id: str) -> list[BaseMsg]:
        """获取回复某条消息的所有消息"""
        return [m for m in self._message_list if m.reply and m.reply.msg_id == message_id]

    def search_by_at_user(self, user_id: str) -> list[BaseMsg]:
        """获取@某人的所有消息"""
        return [m for m in self._message_list if user_id in m.at_ids]

    def get_active_users(self, time_limit: int = None) -> list[str]:
        """返回活跃用户的QQ号，按时间阈值和发言数量排序，越活跃越靠前"""
        if time_limit is None:
            time_limit = SETTING_CFG.Message.ActiveUserTimeLimit  # 默认1天

        speak_limit = SETTING_CFG.Message.ActiveUserSpeakGate  # 默认3句
        speak_count: dict[str, int] = defaultdict(int)
        now = time.time()
        for msg in reversed(self._message_list):
            if now - msg.time > time_limit:
                break

            speak_count[msg.user_id] += 1

        speak_count = {k: v for k, v in speak_count.items() if v >= speak_limit}
        speak_count_sorted = dict(sorted(speak_count.items(), key=lambda x: x[1], reverse=True))
        return list(speak_count_sorted.keys())

    def get_message_from_user(self, user_id: str) -> list[BaseMsg]:
        """获取某人发送的全部消息"""
        if user_id not in self.user_msg_mapping:
            return []

        msgs = [self._message_dict[msg_id] for msg_id in self.user_msg_mapping[user_id]]
        msgs.sort(key=lambda x: x.time)
        return msgs

    def get_time_interval_message(self, start: float = None, end: float = None) -> list[BaseMsg]:
        """返回某个时间区间内的所有消息"""
        if start is None:
            start = 0.0

        if end is None:
            end = time.time()

        if start > end:
            return []

        return [m for m in self._message_list if start <= m.time <= end]

    def get_last_message(self, count: int = 1, reverse=False) -> list[BaseMsg]:
        """
        返回最后count条消息，若消息数量不足则取小值
        :param count: 消息数
        :param reverse: 是否倒叙，默认正序（旧到新）
        :return:
        """
        if not self._message_list:
            return []

        selected = []
        for i, m in enumerate(reversed(self._message_list)):
            if i >= count:
                break

            selected.append(m)

        if not reverse:
            return list(reversed(selected))

        return selected

    def get_new_message_start_from(self, message_id: str, reverse=False) -> list[BaseMsg]:
        """
        返回从 message_id 开始的新消息，不包括 message_id 本身
        :param message_id:
        :param reverse: 是否倒叙，默认正序（旧到新）
        :return:
        """
        if message_id not in self._message_dict:
            return []

        result = []
        for msg in reversed(self._message_list):
            if msg.msg_id == message_id:
                break

            result.append(msg)

        if not reverse:
            return list(reversed(result))

        return result

    @property
    def sended_msg(self) -> set[str]:
        """返回BOT自己发送过的消息，判断回复用"""
        return self.bot_msg_id.copy()

    @property
    def capacity(self) -> int:
        return self._max_message

    # 魔法方法
    def __contains__(self, item):
        if not isinstance(item, BaseMsg):
            raise TypeError

        return item.msg_id in self._message_dict

    def __bool__(self):
        return bool(self._message_list)

    def __iter__(self):
        return iter(self._message_list)

    def __reversed__(self):
        return reversed(self._message_list)

    def __len__(self):
        return len(self._message_list)

    def __str__(self):
        """返回总消息量、最新消息时间、最新十条消息的内容"""
        output = f"当前消息数量:[{len(self._message_list)}]"
        if self._message_list:
            dt_text = timestamp2text(self._message_list[-1].time)
            output += f"，最新消息的时间:[{dt_text}]\n"
            output += "最新十条消息："
            for i, m in enumerate(self.get_last_message(10), start=1):
                if i > 10:
                    break

                output += f"\n{str(m)}"

        return output

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self._message_list[item]

        elif isinstance(item, str):
            return self._message_dict[item]

        else:
            raise TypeError("Not support Key Type")
