from __future__ import annotations
"""
base_dialog.py
TIYA_BOT的对话对象，负责具体消息处理逻辑，独立消息队列
"""

__version__ = "3.1.0"

from typing import Awaitable, Callable

from TIYA.model.message import MessageManager, BaseMsg
from TIYA.config import get_bot_uid, get_bot_name


class BaseDialog:
    """基础对话对象"""
    def __init__(self, *, max_message: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.bot_id: str = get_bot_uid()
        self.bot_name: str = get_bot_name()
        self.message_history: MessageManager = MessageManager(
            max_message=max_message,
        )
        self._pause: bool = False

        # 其他属性
        self._message_handle_sequence: list[Callable[..., Awaitable]] = []  # message flow 顺序

    # 入口
    async def accept_message(self, msg: BaseMsg) -> bool:
        """可以返回拦截信号阻止消息流继续传递"""
        if self._pause:
            return False

        process_mirror = msg.copy()
        return await self.message_flow(process_mirror)

    # 虽然不建议在对话里嵌套消息流，不过还是做了。最好只实现一个自身的逻辑处理
    async def message_flow(self, msg: BaseMsg) -> bool:
        raise NotImplemented

    # =========================================================
    # 工具区
    # =========================================================

    def _register_message_handle(self, function: Callable[..., Awaitable]):
        """内部用，排序消息处理handle"""
        if function in self._message_handle_sequence:
            raise RuntimeError("重复注册handle")

        self._message_handle_sequence.append(function)

    def is_reply(self, msg: BaseMsg) -> bool:
        reply = msg.reply
        if reply is None:
            return False

        if reply.msg_id in self.message_history.bot_msg_id:
            return True

        return False

    def is_at(self, msg: BaseMsg) -> bool:
        if self.bot_id in msg.at_ids:
            return True

        return False

    def is_reply_or_at(self, msg: BaseMsg) -> bool:
        return self.is_at(msg) or self.is_reply(msg)

    def pause(self):
        """暂停接收新消息"""
        self._pause = True

    def resume(self):
        """继续接收新消息"""
        self._pause = False

    @property
    def suspending(self) -> bool:
        """是否挂起状态"""
        return self._pause

    @property
    def enable(self) -> bool:
        return not self._pause
