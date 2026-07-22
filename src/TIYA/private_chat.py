from __future__ import annotations
"""
private_chat.py
私聊消息流与通知广播
"""
__version__ = "0.1.0"

import asyncio
import json
import random
import traceback
from time import monotonic
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import TIYA.api.private_api as api
from TIYA.aqueue import Aqueue, AqueueTask
from TIYA.auto_fav import get_autofav
from TIYA.config import (
    BASE_CFG,
    CONFIG_OBSERVER,
    PRIVATE_CHATS_DIR,
    SETTING_CFG,
    get_private_user_config,
)
from TIYA.global_vars import PRIVATE_CHATS
from TIYA.logger import get_logger
from TIYA.time_id import next_time_id
from TIYA.dialog.base_dialog import BaseDialog
from TIYA.dialog.private_dialog import (
    PrivateCommandDialog,
    PrivateFavDialog,
    PrivateMainDialog,
    PrivateManagementDialog,
)
from TIYA.fav_system import FavTarget
from TIYA.model.message import (
    BaseSendItem,
    MessageManager,
    PrivateMsg,
    SendMessage,
)
from TIYA.utils import atomic_save_json

if TYPE_CHECKING:
    from ncatbot.core.message import PrivateMessage


_log = get_logger()
_NOTICE_PLACEHOLDERS = {"114514", "1919810"}
_PRIVATE_CHAT_BUSY_RECHECK = 60.0
_PRIVATE_CHAT_REAPER_FAILURE_RETRY = 1.0
_PRIVATE_CHAT_REAPER_TASK: asyncio.Task[None] | None = None
_PRIVATE_CHAT_REAPER_WAKE: asyncio.Event | None = None


def _private_setting(name: str, default: float | int) -> float | int:
    """Read a private setting while remaining usable before config startup."""
    private = getattr(SETTING_CFG, "PrivateChat", None)
    return getattr(private, name, default)

def _notify_private_chat_reaper() -> None:
    wake = _PRIVATE_CHAT_REAPER_WAKE
    if wake is not None:
        wake.set()

def _ensure_private_chat_reaper() -> None:
    """Run one deadline-driven reaper for every live private-chat object."""
    global _PRIVATE_CHAT_REAPER_TASK, _PRIVATE_CHAT_REAPER_WAKE

    task = _PRIVATE_CHAT_REAPER_TASK
    if task is not None and not task.done():
        _notify_private_chat_reaper()
        return

    _PRIVATE_CHAT_REAPER_WAKE = asyncio.Event()
    _PRIVATE_CHAT_REAPER_TASK = asyncio.create_task(_private_chat_reaper())

async def _private_chat_reaper() -> None:
    global _PRIVATE_CHAT_REAPER_TASK, _PRIVATE_CHAT_REAPER_WAKE

    current = asyncio.current_task()
    try:
        while True:
            try:
                keep_running = await _private_chat_reaper_cycle()

            except asyncio.CancelledError:
                return

            except Exception:
                _log.error(f"私聊对象回收器异常，将自动重试\n{traceback.format_exc()}")
                await asyncio.sleep(_PRIVATE_CHAT_REAPER_FAILURE_RETRY)
                continue

            if not keep_running:
                return

    finally:
        if _PRIVATE_CHAT_REAPER_TASK is current:
            _PRIVATE_CHAT_REAPER_TASK = None
            _PRIVATE_CHAT_REAPER_WAKE = None

async def _private_chat_reaper_cycle() -> bool:
    wake = _PRIVATE_CHAT_REAPER_WAKE
    if wake is None:
        return False

    wake.clear()
    chats = [
        chat
        for chat in tuple(PRIVATE_CHATS.values())
        if isinstance(chat, PrivateChat)
        and chat.enable
        and chat._chat_object_timeout > 0
    ]
    if not chats:
        return False

    now = monotonic()
    due = [
        chat
        for chat in chats
        if chat._last_activity_at + chat._chat_object_timeout <= now
        and chat._reap_retry_at <= now
    ]
    busy_due = False
    for chat in due:
        try:
            expired = await chat._expire_if_idle(now=now)

        except asyncio.CancelledError:
            raise

        except Exception as E:
            busy_due = True
            chat._reap_retry_at = now + _PRIVATE_CHAT_BUSY_RECHECK
            _log.error(
                f"私聊对象[{getattr(chat, 'user_id', 'unknown')}]回收失败，稍后重试: {E}\n"
                f"{traceback.format_exc()}"
            )

        else:
            if not expired:
                busy_due = True

    chats = [
        chat
        for chat in tuple(PRIVATE_CHATS.values())
        if isinstance(chat, PrivateChat)
        and chat.enable
        and chat._chat_object_timeout > 0
    ]
    if not chats:
        return False

    now = monotonic()
    future_delays = [
        chat._last_activity_at + chat._chat_object_timeout - now
        for chat in chats
        if chat._last_activity_at + chat._chat_object_timeout > now
    ]
    wait_time = min(future_delays) if future_delays else _PRIVATE_CHAT_BUSY_RECHECK
    if busy_due:
        wait_time = min(wait_time, _PRIVATE_CHAT_BUSY_RECHECK)

    try:
        await asyncio.wait_for(wake.wait(), timeout=max(0.001, wait_time))

    except asyncio.TimeoutError:
        pass

    return True

async def _stop_private_chat_reaper_if_unused() -> None:
    if any(
        isinstance(chat, PrivateChat) and chat.enable
        for chat in tuple(PRIVATE_CHATS.values())
    ):
        _ensure_private_chat_reaper()
        return

    task = _PRIVATE_CHAT_REAPER_TASK
    if task is None or task is asyncio.current_task():
        return

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@dataclass(slots=True)
class NoticeResponse:
    echo: str
    user_id: str
    message_id: str
    expected_replies: int = 1
    intercept: bool = False
    response_timeout: float | None = None
    replies: list[PrivateMsg] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    expired: bool = field(default=False, init=False)
    expires_at: float | None = field(default=None, init=False)

    def __post_init__(self):
        if self.expected_replies <= 0:
            raise ValueError("expected_replies 必须大于 0")

        if self.response_timeout is not None and self.response_timeout <= 0:
            raise ValueError("response_timeout 必须大于 0")

    @property
    def done(self) -> bool:
        return len(self.replies) >= self.expected_replies

    def append(self, message: PrivateMsg):
        if self.cancelled or self.expired or self.done:
            return

        self.replies.append(message)
        if self.done:
            self.event.set()

    def cancel(self):
        if self.done:
            self.event.set()
            return

        self.cancelled = True
        self.event.set()

    def expire(self):
        if self.cancelled or self.expired:
            self.event.set()
            return

        self.expired = True
        self.event.set()


class PrivateChat:
    def __init__(self, user_id: str, username: str = ""):
        self.user_id = str(user_id)
        self.username = username
        self.data_path = PRIVATE_CHATS_DIR / self.user_id
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.private_config = get_private_user_config(self.user_id)
        self.message_history = MessageManager(
            max_message=int(_private_setting("MaxMessageHistory", 5000))
        )
        self.aqueue = Aqueue(
            min_worker=1,
            max_worker=1,
            default_timeout=-1,
            observe_config=False
        )
        self._notice_queue: deque[NoticeResponse] = deque()
        self._notice_responses: dict[str, NoticeResponse] = {}
        self._notice_expiry_tasks: dict[str, asyncio.Task[None]] = {}
        self._enable = bool(self.private_config.enable)
        self._loaded = False

        # 配置
        self._management_dialog_timeout = float(
            _private_setting("ManagementDialogTimeout", 1800)
        )
        self._chat_object_timeout = float(_private_setting("ChatObjectTimeout", 10800))

        # 锁
        self._management_dialog_lock = asyncio.Lock()
        self._fav_dialog_lock = asyncio.Lock()
        self._main_dialog_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()

        # 工具对象
        self.logger = _log.create_block(f"private_{user_id}", username)
        self.command_dialog = PrivateCommandDialog(host=self)
        self.management_dialog: PrivateManagementDialog | None = None
        self.fav_dialog: PrivateFavDialog | None = None
        self.main_dialog: PrivateMainDialog | None = None
        self.fav = get_autofav(self.data_path / "fav_index.json", logger=self.logger)
        self._last_activity_at = monotonic()
        self._reap_retry_at = 0.0

        # 数据结构
        self.dialog_list: list[BaseDialog] = [self.command_dialog]

        PRIVATE_CHATS[self.user_id] = self
        CONFIG_OBSERVER.register(self)
        _ensure_private_chat_reaper()

    async def accept_message(self, message: PrivateMessage) -> AqueueTask | None:
        if not self._enable:
            return None

        await self._ensure_loaded()
        self.touch_activity()

        sender = getattr(message, "sender", None)
        username = str(getattr(sender, "nickname", "") or "")
        if username:
            self.username = username

        parsed_message = await self.message_history.add_message(message)
        if not isinstance(parsed_message, PrivateMsg):
            return None

        asyncio.create_task(self._log_message(parsed_message))
        return self.aqueue.add_task(self.message_flow(parsed_message), timeout=-1)

    async def message_flow(self, message: PrivateMsg) -> bool:
        # 先获取表情，保持和群聊消息流一致；不阻塞后续对话处理。
        fav = getattr(self, "fav", None)
        if fav is not None:
            asyncio.create_task(fav.auto_fav(message))

        if self._notice_trigger(message):
            return True

        management_dialog = getattr(self, "management_dialog", None)
        if management_dialog is not None:
            management_dialog.refresh_timeout()

        for dialog in self.dialog_list:
            if await dialog.accept_message(message):
                return True

        main_dialog = await self._ensure_main_dialog()
        if main_dialog is None:
            return False

        return await main_dialog.accept_message(message)

    def update_config(self) -> None:
        self.private_config = get_private_user_config(self.user_id)
        self._management_dialog_timeout = float(
            _private_setting("ManagementDialogTimeout", 1800)
        )
        self._chat_object_timeout = float(_private_setting("ChatObjectTimeout", 10800))
        self._enable = bool(self.private_config.enable)
        if self.main_dialog is not None:
            self.main_dialog.update_config()

        _ensure_private_chat_reaper()

    # =========================================================
    # 通知广播对话
    # =========================================================

    def _notice_trigger(self, message: PrivateMsg) -> bool:
        """Consume pending notice replies before any business dialog."""
        return self._consume_notice(message)

    def register_notice(self, response: NoticeResponse):
        if response.user_id != self.user_id:
            raise ValueError("通知目标与私聊对象不一致")

        if response.echo in self._notice_responses:
            raise ValueError(f"通知 echo 重复: {response.echo}")

        if response.cancelled or response.expired or response.done:
            raise ValueError("不能注册已结束的通知")

        response_timeout = response.response_timeout
        if response_timeout is None:
            response_timeout = float(
                _private_setting("NoticeResponseTimeout", 300)
            )

        response.response_timeout = response_timeout
        loop = asyncio.get_running_loop()
        response.expires_at = loop.time() + response_timeout
        self._notice_responses[response.echo] = response
        self._notice_queue.append(response)
        self._notice_expiry_tasks[response.echo] = asyncio.create_task(
            self._expire_notice(response)
        )

    def get_notice(self, echo: str) -> NoticeResponse | None:
        return self._notice_responses.get(echo)

    def _remove_notice_reference(self, echo: str) -> NoticeResponse | None:
        response = self._notice_responses.pop(echo, None)
        if response is None:
            return None

        self._notice_queue = deque(
            item for item in self._notice_queue
            if item.echo != echo
        )
        return response

    async def _expire_notice(self, response: NoticeResponse):
        response_timeout = response.response_timeout
        if response_timeout is None:
            return

        try:
            await asyncio.sleep(response_timeout)
        except asyncio.CancelledError:
            return

        if self._notice_responses.get(response.echo) is not response:
            return

        response.expire()
        self._remove_notice_reference(response.echo)
        # noinspection PyAsyncCall
        self._notice_expiry_tasks.pop(response.echo, None)

    def remove_notice(self, echo: str, *, cancel: bool = False) -> NoticeResponse | None:
        response = self._remove_notice_reference(echo)
        if response is None:
            return None

        expiry_task = self._notice_expiry_tasks.pop(echo, None)
        if expiry_task is not None:
            expiry_task.cancel()

        if cancel:
            response.cancel()

        return response

    def cancel_all_notices(self):
        for response in self._notice_responses.values():
            response.cancel()

        self._notice_queue.clear()
        for task in self._notice_expiry_tasks.values():
            task.cancel()

        self._notice_expiry_tasks.clear()

    def _consume_notice(self, message: PrivateMsg) -> bool:
        while self._notice_queue:
            response = self._notice_queue[0]
            if response.cancelled or response.expired or response.done:
                self._notice_queue.popleft()
                continue

            response.append(message)
            if response.done:
                self._notice_queue.popleft()

            return response.intercept

        return False

    # =========================================================
    # 各种trigger
    # =========================================================

    async def open_management_dialog(self) -> bool:
        """Create and insert a fresh management dialog when none is active."""
        async with self._management_dialog_lock:
            if self.management_dialog is not None:
                self.management_dialog.refresh_timeout()
                return False

            dialog = PrivateManagementDialog(
                host=self,
                idle_timeout=self._management_dialog_timeout,
            )
            self.management_dialog = dialog
            command_index = self.dialog_list.index(self.command_dialog)
            self.dialog_list.insert(command_index + 1, dialog)
            try:
                await dialog.start()

            except Exception:
                self.dialog_list.remove(dialog)
                self.management_dialog = None
                await dialog.shutdown()
                raise

            return True

    async def close_management_dialog(
            self,
            expected: PrivateManagementDialog | None = None,
    ) -> bool:
        """Remove and destroy the active management dialog."""
        async with self._management_dialog_lock:
            dialog = self.management_dialog
            if dialog is None:
                return False

            if expected is not None and dialog is not expected:
                return False

            async with self._fav_dialog_lock:
                await self._close_fav_dialog()
            self.management_dialog = None
            if dialog in self.dialog_list:
                self.dialog_list.remove(dialog)

            await dialog.shutdown()
            return True

    async def open_fav_dialog(self, target: FavTarget) -> bool:
        """Insert one fresh favorite dialog immediately before management."""
        async with self._management_dialog_lock:
            if self.management_dialog is None:
                return False

            async with self._fav_dialog_lock:
                if self.fav_dialog is not None:
                    return False

                dialog = PrivateFavDialog(host=self, target=target)
                management_index = self.dialog_list.index(self.management_dialog)
                self.dialog_list.insert(management_index, dialog)
                self.fav_dialog = dialog
                return True

    async def close_fav_dialog(
            self,
            expected: PrivateFavDialog | None = None,
    ) -> bool:
        """Remove and destroy the active favorite dialog."""
        async with self._fav_dialog_lock:
            return await self._close_fav_dialog(expected)

    async def _close_fav_dialog(
            self,
            expected: PrivateFavDialog | None = None,
    ) -> bool:
        dialog = self.fav_dialog
        if dialog is None:
            return False

        if expected is not None and dialog is not expected:
            return False

        self.fav_dialog = None
        if dialog in self.dialog_list:
            self.dialog_list.remove(dialog)

        await dialog.shutdown()
        return True

    # =========================================================
    # 私聊方法
    # =========================================================
    async def _ensure_main_dialog(self) -> PrivateMainDialog | None:
        if not self._enable or not bool(self.private_config.chat):
            return None

        async with self._main_dialog_lock:
            if self.main_dialog is not None:
                return self.main_dialog

            dialog = PrivateMainDialog(host=self)
            try:
                await dialog.start()

            except Exception:
                await dialog.shutdown()
                raise

            self.main_dialog = dialog
            return dialog

    async def _ensure_loaded(self) -> None:
        if not hasattr(self, "_loaded") or self._loaded:
            return

        async with self._load_lock:
            if self._loaded:
                return

            save_path = self.data_path / "private_persist.json"
            if save_path.is_file():
                try:
                    content = json.loads(save_path.read_text(encoding="utf-8")).get("data", {})

                except (OSError, json.JSONDecodeError):
                    content = {}

                messages = content.get("messages", {})
                if messages:
                    await self.message_history.load_dict(messages)

            self._loaded = True

    def save(self) -> None:
        atomic_save_json(
            {
                "version": "0.1.0",
                "data": {
                    "username": self.username,
                    "messages": self.message_history.to_dict(),
                },
            },
            self.data_path / "private_persist.json",
        )

    def log_prefix(self) -> str:
        return f"私聊[{self.user_id}]({self.username}) "

    async def _log_message(self, msg: PrivateMsg):
        msg_dict = await msg.to_llm(wait=True,timeout=1.5)
        self.logger.info(msg_dict["format_text"])

    async def shutdown(self):
        async with self._shutdown_lock:
            if not self._enable and PRIVATE_CHATS.get(self.user_id) is not self:
                return

            self._enable = False
            self.cancel_all_notices()
            # Let notice waiters observe cancellation before this chat leaves
            # the registry. This also removes a shutdown scheduling race.
            await asyncio.sleep(0)
            await self.close_management_dialog()
            if self.main_dialog is not None:
                await self.main_dialog.save()
                await self.main_dialog.shutdown()
                self.main_dialog = None

            self.save()
            await self.aqueue.kill(timeout=None)
            CONFIG_OBSERVER.remove(self)
            if PRIVATE_CHATS.get(self.user_id) is self:
                PRIVATE_CHATS.pop(self.user_id, None)

            await _stop_private_chat_reaper_if_unused()

    @property
    def enable(self) -> bool:
        return self._enable

    # =========================================================
    # 私聊API
    # =========================================================

    async def say(
            self,
            message: PrivateMsg | SendMessage | BaseSendItem | str  # 这里要加多一个 PrivateMsg
    ) -> list[PrivateMsg | None]:
        await self._ensure_loaded()
        self.touch_activity()
        timeout = float(_private_setting("ApiTimeout", 60))  # 默认60秒

        if isinstance(message, SendMessage):
            send_message = message

        elif isinstance(message, (BaseSendItem, str)):
            send_message = SendMessage(message)

        elif isinstance(message, PrivateMsg):
            send_message = SendMessage.from_message(message)

        else:
            raise TypeError(f"不支持的发送格式: {type(message)}")

        message_ids: list[str | None] = []
        first_chain = True
        sleep_per_word = float(_private_setting("SleepPerWord", 0.4))  # 默认0.4秒
        for message_chain in await send_message.message_chain_async():
            try:
                # 模拟打字延迟
                if not first_chain:
                    await asyncio.sleep(0.5)
                    words_sleep = []
                    for item in message_chain.elements:
                        item_type = item.get("type", "")
                        if item_type != "text":
                            continue

                        text_data = item.get("data", {}).get("text", "")
                        words_sleep.append(max(10, len(text_data)) * sleep_per_word + random.uniform(0, 0.5))

                    await asyncio.sleep(sum(words_sleep))

                response = await asyncio.wait_for(api.say(self.user_id, message_chain), timeout)

            except asyncio.TimeoutError:
                message_ids.append(None)
                continue

            else:
                if not isinstance(response, dict) or response.get("status") != "ok":
                    message_ids.append(None)
                    continue

                data = response.get("data")
                if not isinstance(data, dict) or not data.get("message_id"):
                    message_ids.append(None)
                    continue

                message_ids.append(str(data["message_id"]))

            finally:
                first_chain = False

        sent_messages = PrivateMsg.from_send(
            messages_id=message_ids,
            send=send_message
        )
        for sent_message in sent_messages:
            if sent_message is None:
                continue

            await self.message_history.add_message(sent_message.copy())
            asyncio.create_task(self._log_message(sent_message))

        return sent_messages

    async def system_send(
            self,
            message: PrivateMsg | SendMessage | BaseSendItem | str,
            *,
            delay: float | Callable[[], float] = 0.3,
    ) -> list[PrivateMsg | None]:
        """Send system output with a fixed or callable chain interval."""
        await self._ensure_loaded()
        self.touch_activity()
        timeout = float(_private_setting("ApiTimeout", 60))

        if isinstance(message, SendMessage):
            send_message = message

        elif isinstance(message, (BaseSendItem, str)):
            send_message = SendMessage(message)

        elif isinstance(message, PrivateMsg):
            send_message = SendMessage.from_message(message)

        else:
            raise TypeError(f"不支持的发送格式: {type(message)}")

        fixed_delay: float | None = None
        if not callable(delay):
            fixed_delay = float(delay)
            if fixed_delay < 0:
                raise ValueError("delay 不能小于 0")

        message_ids: list[str | None] = []
        first_chain = True
        for message_chain in await send_message.message_chain_async():
            try:
                if not first_chain:
                    if callable(delay):
                        current_delay = delay()
                    else:
                        assert fixed_delay is not None
                        current_delay = fixed_delay

                    current_delay = float(current_delay)
                    if current_delay < 0:
                        raise ValueError("delay 不能小于 0")

                    await asyncio.sleep(current_delay)

                response = await asyncio.wait_for(
                    api.say(self.user_id, message_chain),
                    timeout,
                )

            except asyncio.TimeoutError:
                message_ids.append(None)
                continue

            else:
                if not isinstance(response, dict) or response.get("status") != "ok":
                    message_ids.append(None)
                    continue

                data = response.get("data")
                if not isinstance(data, dict) or not data.get("message_id"):
                    message_ids.append(None)
                    continue

                message_ids.append(str(data["message_id"]))

            finally:
                first_chain = False

        sent_messages = PrivateMsg.from_send(
            messages_id=message_ids,
            send=send_message,
        )
        for sent_message in sent_messages:
            if sent_message is None:
                continue

            await self.message_history.add_message(sent_message.copy())
            asyncio.create_task(self._log_message(sent_message))

        return sent_messages

    # =========================================================
    # 生命周期管理
    # =========================================================

    @classmethod
    def get_or_create(cls, user_id: str, username: str = "") -> PrivateChat:
        """创建私聊对象"""
        user_id = str(user_id)
        chat = PRIVATE_CHATS.get(user_id)
        if isinstance(chat, cls) and chat.enable:
            if username:
                chat.username = username
            return chat

        return cls(user_id, username)

    def touch_activity(self) -> None:
        if not hasattr(self, "_last_activity_at"):
            return

        self._last_activity_at = monotonic()
        self._reap_retry_at = 0.0
        _ensure_private_chat_reaper()

    def _is_idle(self) -> bool:
        if self._notice_responses or self.management_dialog is not None or self.fav_dialog is not None:
            return False

        if self.aqueue._task_queue.qsize() or self.aqueue.running_worker:
            return False

        if self.main_dialog is not None and not self.main_dialog.idle:
            return False

        return True

    async def _expire_if_idle(self, *, now: float | None = None) -> bool:
        if not self._enable:
            return False

        if now is None:
            now = monotonic()

        if now - self._last_activity_at < self._chat_object_timeout:
            return False

        if not self._is_idle():
            return False

        await self.shutdown()
        return True


class NoticeManager:
    @staticmethod
    def _new_echo() -> str:
        return str(next_time_id())  # 现在可以改用 time_id了，更稳一点

    @staticmethod
    def _notice_targets() -> list[str]:
        raw_targets = BASE_CFG.NoticeList
        if not isinstance(raw_targets, (list, tuple, set)):
            return []

        targets: list[str] = []
        for raw_target in raw_targets:
            if not isinstance(raw_target, (str, int)):
                continue

            target = str(raw_target).strip()
            if not target or target in _NOTICE_PLACEHOLDERS or target in targets:  # 这样处理也行，先不改
                continue

            targets.append(target)

        return targets

    @staticmethod
    def _chat(target: str) -> PrivateChat:
        return PrivateChat.get_or_create(str(target))

    @classmethod
    async def notice_someone(
            cls,
            target: str,
            message: SendMessage | BaseSendItem | str
    ) -> list[PrivateMsg | None]:
        return await cls._chat(target).say(message)

    @classmethod
    async def notice_all(
            cls,
            message: SendMessage | BaseSendItem | str
    ) -> dict[str, list[PrivateMsg | None]]:
        targets = cls._notice_targets()
        results = await asyncio.gather(
            *(cls.notice_someone(target, message) for target in targets),
            return_exceptions=True
        )
        output: dict[str, list[PrivateMsg | None]] = {}
        for target, result in zip(targets, results):
            if isinstance(result, BaseException):
                _log.error(f"通知发送失败: user_id=[{target}], {result}")
                output[target] = [None]
            else:
                output[target] = result

        return output

    @classmethod
    async def request_someone(
            cls,
            target: str,
            message: SendMessage | BaseSendItem | str,
            *,
            expected_replies: int = 1,
            intercept: bool = False,
            echo: str | None = None,
            response_timeout: float | None = None
    ) -> str | None:
        if expected_replies <= 0:
            raise ValueError("expected_replies 必须大于 0")

        if response_timeout is not None and response_timeout <= 0:
            raise ValueError("response_timeout 必须大于 0")

        if echo is None:
            echo = cls._new_echo()

        chat = cls._chat(target)
        sent_messages = await chat.say(message)
        sent_message = next(
            (
                item for item in reversed(sent_messages)
                if isinstance(item, PrivateMsg)
            ),
            None
        )
        if sent_message is None:
            return None

        chat.register_notice(NoticeResponse(
            echo=echo,
            user_id=chat.user_id,
            message_id=sent_message.msg_id,
            expected_replies=expected_replies,
            intercept=intercept,
            response_timeout=response_timeout
        ))
        return echo

    @classmethod
    async def request_all(
            cls,
            message: SendMessage | BaseSendItem | str,
            *,
            expected_replies: int = 1,
            intercept: bool = False,
            echo: str | None = None,
            response_timeout: float | None = None
    ) -> str | None:
        if response_timeout is not None and response_timeout <= 0:
            raise ValueError("response_timeout 必须大于 0")

        if echo is None:
            echo = cls._new_echo()

        results = await asyncio.gather(
            *(
                cls.request_someone(
                    target,
                    message,
                    expected_replies=expected_replies,
                    intercept=intercept,
                    echo=echo,
                    response_timeout=response_timeout
                )
                for target in cls._notice_targets()
            ),
            return_exceptions=True
        )
        for result in results:
            if result == echo:
                return echo

        return None

    @classmethod
    async def wait_someone(
            cls,
            target: str,
            echo: str,
            timeout: float | None = None
    ) -> NoticeResponse | None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout 不能小于 0")

        chat = PRIVATE_CHATS.get(str(target))
        if not isinstance(chat, PrivateChat):
            return None

        response = chat.get_notice(echo)
        if response is None:
            return None

        try:
            await asyncio.wait_for(response.event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            if chat.get_notice(echo) is response:
                chat.remove_notice(echo, cancel=True)
            return None

        chat.remove_notice(echo)
        if response.expired:
            return None

        return response

    @classmethod
    async def wait_any(
            cls,
            echo: str,
            timeout: float | None = None
    ) -> NoticeResponse | None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout 不能小于 0")

        responses: list[tuple[PrivateChat, NoticeResponse]] = []
        for chat in list(PRIVATE_CHATS.values()):
            if not isinstance(chat, PrivateChat):
                continue

            response = chat.get_notice(echo)
            if response is not None:
                responses.append((chat, response))

        if not responses:
            return None

        tasks = {
            asyncio.create_task(response.event.wait()): (chat, response)
            for chat, response in responses
        }
        pending = set(tasks)
        winner: NoticeResponse | None = None
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while pending and winner is None:
            wait_timeout = None
            if deadline is not None:
                wait_timeout = max(0.0, deadline - loop.time())

            done, pending = await asyncio.wait(
                pending,
                timeout=wait_timeout,
                return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                break

            for completed_task in done:
                _, response = tasks[completed_task]
                if response.done and not response.cancelled and not response.expired:
                    winner = response
                    break

        for task in pending:
            task.cancel()

        await asyncio.gather(*pending, return_exceptions=True)
        if winner is None:
            for chat, response in responses:
                if chat.get_notice(echo) is response:
                    chat.remove_notice(echo, cancel=True)
            return None

        for chat, response in responses:
            if response is winner:
                chat.remove_notice(echo)
            else:
                chat.remove_notice(echo, cancel=True)

        return winner

    @classmethod
    def cancel(cls, echo: str, target: str | None = None):
        if target is not None:
            chat = PRIVATE_CHATS.get(str(target))
            if isinstance(chat, PrivateChat):
                chat.remove_notice(echo, cancel=True)
            return

        for chat in list(PRIVATE_CHATS.values()):
            if isinstance(chat, PrivateChat):
                chat.remove_notice(echo, cancel=True)
