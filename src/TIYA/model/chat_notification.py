from __future__ import annotations
"""
chat_notification.py
通知系统，通过消息数、发言数、时间三项决定存活
"""
__version__ = "0.1.2"

import json
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b
from typing import TypedDict

from TIYA.logger import get_logger

_log = get_logger()


@dataclass(slots=True)
class Notice:
    content: str
    title: str
    priority: int
    alive_until_messages: int
    alive_until_self_messages: int
    alive_until_time: float

    @property
    def uid(self) -> str:
        hash_str = json.dumps([self.content, self.title], ensure_ascii=False)
        h = blake2b(hash_str.encode(), digest_size=8)
        return h.hexdigest()

    def __hash__(self) -> int:
        return hash((self.content, self.title))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Notice):
            return NotImplemented

        return self.content == other.content and self.title == other.title

    # PERSIST
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Notice:
        return Notice(
            content=data["content"],
            title=data["title"],
            priority=data["priority"],
            alive_until_messages=data["alive_until_messages"],
            alive_until_self_messages=data["alive_until_self_messages"],
            alive_until_time=data["alive_until_time"]
        )


class ChatNotification:
    _MAX_COUNTER = 1_000_000

    class _NoticeRecord(TypedDict):
        message_start: int
        self_message_start: int
        time_start: float

    def __init__(self) -> None:
        self._notice_data: dict[str, Notice] = {}
        self._recording: dict[str, ChatNotification._NoticeRecord] = {}
        self._messages = 0
        self._self_messages = 0

    def add_notice(
            self,
            content: str = "",
            title: str = "普通通知",
            priority: int = 50,
            alive_until_messages: int = 0,
            alive_until_self_messages: int = 0,
            alive_until_time: float = 0,
            exist_ok: bool = False
    ) -> Notice:
        """
        添加一条通知，内容和标题一样的通知不会重复入库。非要这样做可以加多几个空格
        :param content: 通知内容
        :param title: 标题，分组用
        :param priority: 优先级，越小越靠前
        :param alive_until_messages: 多少条消息失效，可为负数
        :param alive_until_self_messages: 自己发言多少次失效，可为负数
        :param alive_until_time: 多少时间后失效，可为负数
        :param exist_ok: 是否静默返回已存在的通知
        :return:
        """
        # 至少得有一个限制
        if not (alive_until_messages or alive_until_self_messages or alive_until_time):
            raise ValueError("通知至少要有一个限制")

        if not ((alive_until_messages >= 0 and alive_until_self_messages >= 0 and alive_until_time >= 0
        ) or (alive_until_messages <= 0 and alive_until_self_messages <= 0 and alive_until_time <= 0)):
            raise ValueError("要么正向、要么反向，不能又正又反")

        new_notice = Notice(
            content=content,
            title=title,
            priority=priority,
            alive_until_messages=alive_until_messages,
            alive_until_self_messages=alive_until_self_messages,
            alive_until_time=alive_until_time
        )
        new_notice_id = new_notice.uid
        if new_notice_id in self._notice_data:
            if exist_ok:
                # self._update_notice(new_notice_id)
                if not self._check_alive(new_notice_id):
                    self.remove_notice(new_notice_id)
                    self._notice_data[new_notice_id] = new_notice
                    self._recording[new_notice_id] = {
                        "message_start": self._messages,
                        "self_message_start": self._self_messages,
                        "time_start": time.time()
                    }

                return self._notice_data[new_notice_id]

            else:
                _log.warning(f"通知 [{new_notice_id}] 已存在，添加失败")
                return self._notice_data[new_notice_id]

        self._notice_data[new_notice_id] = new_notice
        self._recording[new_notice_id] = {
            "message_start": self._messages,
            "self_message_start": self._self_messages,
            "time_start": time.time()
        }
        return new_notice

    def clear(self):
        self._notice_data.clear()
        self._recording.clear()

    def remove_notice(self, notice_id: str | Notice) -> None:
        if isinstance(notice_id, Notice):
            notice_id = notice_id.uid

        self._notice_data.pop(notice_id, None)
        self._recording.pop(notice_id, None)

    def notice_message_trigger(self, self_msg: bool = False) -> None:
        if self_msg:
            self._self_messages += 1

        else:
            self._messages += 1

    def notice_suppress(self, notice_id: str | Notice) -> None:
        """抑制反向通知"""
        if isinstance(notice_id, Notice):
            notice_id = notice_id.uid

        notice = self._notice_data.get(notice_id, None)
        if not notice:
            raise KeyError(f"通知 [{notice_id}] 不存在")

        if any(getattr(notice, attr) > 0 for attr in
               ("alive_until_messages", "alive_until_self_messages", "alive_until_time")):
            raise KeyError(f"通知 [{notice_id}] 不是反向通知")

        # noinspection PyTypeChecker
        self._recording.setdefault(notice_id, {}).update(
            {
                "message_start": self._messages,
                "self_message_start": self._self_messages,
                "time_start": time.time()
            }
        )

    def refresh_notice(self, notice_id: str | Notice) -> None:
        """重置通知的计时器"""
        if isinstance(notice_id, Notice):
            notice_id = notice_id.uid

        notice = self._notice_data.get(notice_id, None)
        if not notice:
            return

        # noinspection PyTypeChecker
        self._recording.setdefault(notice_id, {}).update(
            {
                "message_start": self._messages,
                "self_message_start": self._self_messages,
                "time_start": time.time()
            }
        )

    def get_notice(self, notice_id: str) -> Notice | None:
        """获取通知对象"""
        return self._notice_data.get(notice_id, None)

    def _shift_counter(self) -> None:
        if self._messages < self._MAX_COUNTER and self._self_messages < self._MAX_COUNTER:
            return

        messages_diff = self._messages
        self_messages_diff = self._self_messages
        self._messages = 0
        self._self_messages = 0
        for record in self._recording.values():
            record["message_start"] -= messages_diff
            record["self_message_start"] -= self_messages_diff

    def _check_alive(self, notice_id: str) -> bool:
        def _delete(_uid: str) -> None:
            self._notice_data.pop(_uid, None)
            self._recording.pop(_uid, None)

        notice = self._notice_data.get(notice_id, None)
        if notice is None:
            return False

        start = self._recording.get(notice_id)
        if not start:
            _delete(notice_id)
            return False

        # 正向和反向不同逻辑，正向是有任意条件达成就不通知、反向是任意条件达成就通知
        messages_flag = False
        self_said_flag = False
        time_flag = False
        if notice.alive_until_messages:
            value = notice.alive_until_messages
            if value > 0:
                if self._messages - start["message_start"] > value:
                    _delete(notice_id)
                    return False

                messages_flag = True
            else:
                if value + self._messages > start["message_start"]:
                    messages_flag = True

        if notice.alive_until_self_messages:
            value = notice.alive_until_self_messages
            if value > 0:
                if self._self_messages - start["self_message_start"] > value:
                    _delete(notice_id)
                    return False

                self_said_flag = True

            else:
                if value + self._self_messages > start["self_message_start"]:
                    self_said_flag = True

        if notice.alive_until_time:
            value = notice.alive_until_time
            if value > 0:
                if time.time() - start["time_start"] > value:
                    _delete(notice_id)
                    return False

                time_flag = True

            else:
                if value + time.time() > start["time_start"]:
                    time_flag = True

        if not (messages_flag or self_said_flag or time_flag):
            return False

        return True

    def _ensure_notice_available(self) -> list[Notice]:
        # 清零计算器和修改标志位
        available_notices = []
        for notice_id, notice in self._notice_data.copy().items():
            alive = self._check_alive(notice_id)
            if alive:
                available_notices.append(notice)

        return available_notices

    def is_notice_alive(self, notice: Notice | str) -> bool:
        """检查通知是否已生效"""
        if isinstance(notice, Notice):
            notice = notice.uid

        return self._check_alive(notice)

    def notice2text(self, level: str = "##") -> str:
        """获取完整通知文本，markdown格式。标题顺序由其最高优先度的元素决定。若相同则随机"""
        available_notices = self._ensure_notice_available()
        reorder = sorted(available_notices, key=lambda x: x.priority)
        titled: dict[str, list[Notice]] = {}
        for notice in reorder:
            titled.setdefault(notice.title, []).append(notice)

        output = ""
        for title, notices in titled.items():
            output += f"{level} {title}\n\n"
            for i, notice in enumerate(notices, start=1):
                output += f"{i}. {notice.content}\n"

            output += "\n\n"

        return output.strip('\n')

    # PERSIST
    def to_dict(self) -> dict:
        return {
            "notices": {k: v.to_dict() for k, v in self._notice_data.items()},
            "recordings": self._recording,
            "messages_now": self._messages,
            "self_messages_now": self._self_messages
        }

    def load_dict(self, data: dict) -> None:
        notices_data: dict = data["notices"]
        recordings: dict = data["recordings"]
        self._messages = data["messages_now"]
        self._self_messages = data["self_messages_now"]

        to_obj = {k: Notice.from_dict(v) for k, v in notices_data.items()}
        self._notice_data.clear()
        self._recording.clear()
        self._notice_data.update(to_obj)
        self._recording.update(recordings)
        self._shift_counter()
