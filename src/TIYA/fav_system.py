from __future__ import annotations

"""Image-first favorite management session for private administration."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

from TIYA.auto_fav import get_autofav
from TIYA.command import CommandArgs, CommandError, CommandInvocation, CommandSet
from TIYA.config import PRIVATE_CHATS_DIR
from TIYA.global_vars import PRIVATE_CHATS
from TIYA.model.character import get_character
from TIYA.model.message import (
    ImgMsg,
    PrivateMsg,
    SendBreak,
    SendImage,
    SendMessage,
    SendText,
    TextMsg,
)


class FavoriteStore(Protocol):
    """Storage operations required by the management session."""

    @property
    def fav_list(self) -> dict[str, set[str]]: ...

    async def add_favs(self, entries: list[tuple[str, str]]) -> int: ...

    async def rename_fav(self, hash_name: str, title: str) -> bool: ...

    async def remove_favs(self, hash_names: list[str]) -> int: ...


@dataclass(frozen=True, slots=True)
class FavTarget:
    kind: Literal["group", "character", "private"]
    identifier: str
    display_name: str
    store: FavoriteStore


class FavTargetError(ValueError):
    """Raised when a requested favorite library cannot be resolved."""


def resolve_fav_target(
        *,
        group_id: str | None = None,
        character_name: str | None = None,
        private_id: str | None = None,
) -> FavTarget:
    """Resolve exactly one configured favorite library."""
    group_id = (group_id or "").strip()
    character_name = (character_name or "").strip()
    private_id = (private_id or "").strip()
    if sum(map(bool, (group_id, character_name, private_id))) != 1:
        raise FavTargetError(
            "必须且只能指定 --group、--character 或 --private 其中一个目标"
        )

    if group_id:
        from TIYA.global_vars import QQ_GROUPS

        group = QQ_GROUPS.get(group_id)
        if group is None:
            raise FavTargetError(f"群[{group_id}]不存在或尚未加载")

        store = getattr(group, "fav", None)
        if store is None:
            raise FavTargetError(f"群[{group_id}]尚未初始化表情库")

        group_name = str(getattr(group, "group_name", "") or group_id)
        return FavTarget(
            kind="group",
            identifier=group_id,
            display_name=f"{group_name}({group_id})",
            store=store,
        )

    if private_id:
        if not private_id.isdigit():
            raise FavTargetError(f"私聊 QQ号[{private_id}]不合法")

        private_chat = PRIVATE_CHATS.get(private_id)
        if private_chat is not None:
            store = getattr(private_chat, "fav", None)
            if store is None:
                raise FavTargetError(f"私聊用户[{private_id}]尚未初始化表情库")

            username = str(getattr(private_chat, "username", "") or "").strip()
            display_name = f"{username}({private_id})" if username else private_id
            return FavTarget(
                kind="private",
                identifier=private_id,
                display_name=display_name,
                store=store,
            )

        private_dir = PRIVATE_CHATS_DIR / private_id
        if not private_dir.is_dir():
            raise FavTargetError(f"私聊用户[{private_id}]不存在或尚无私聊数据")

        return FavTarget(
            kind="private",
            identifier=private_id,
            display_name=private_id,
            store=get_autofav(private_dir / "fav_index.json"),
        )

    try:
        character = get_character(character_name)

    except KeyError as exc:
        raise FavTargetError(f"角色[{character_name}]不存在") from exc

    return FavTarget(
        kind="character",
        identifier=character_name,
        display_name=character_name,
        store=character.fav,
    )


@dataclass(slots=True)
class FavItem:
    hash_name: str
    title: str = ""

    @property
    def display_title(self) -> str:
        return self.title or "未命名"


class SelectionSource(Enum):
    LIBRARY = "library"
    PENDING_ADD = "pending_add"
    PENDING_DELETE = "pending_delete"


@dataclass(frozen=True, slots=True)
class FavCommandContext:
    message: PrivateMsg
    invocation: CommandInvocation


SayCallback = Callable[[str | SendMessage], Awaitable[list[object | None]]]
SystemSendCallback = Callable[[str | SendMessage], Awaitable[list[object | None]]]
CloseCallback = Callable[[], Awaitable[bool]]


class FavSystem:
    """Own one disposable favorite-management conversation."""

    COMMANDS = CommandSet(prefix="-")

    def __init__(
            self,
            *,
            target: FavTarget,
            say: SayCallback,
            system_send: SystemSendCallback,
            close: CloseCallback,
            image_wait_timeout: float = 15,
    ):
        if image_wait_timeout <= 0:
            raise ValueError("image_wait_timeout 必须大于 0")

        self.target = target
        self._say = say
        self._system_send = system_send
        self._close = close
        self._image_wait_timeout = float(image_wait_timeout)
        self._command_set = self.COMMANDS

        self._selection: list[FavItem] = []
        self._selection_source = SelectionSource.LIBRARY
        self._cursor: int | None = None
        self._pending_add: dict[str, FavItem] = {}
        self._pending_delete: dict[str, FavItem] = {}
        self._add_confirmation: tuple[tuple[str, str], ...] | None = None
        self._delete_confirmation: tuple[str, ...] | None = None

    async def handle_message(self, message: PrivateMsg) -> bool:
        """Intercept and process one message in the active favorite dialog."""
        images = message.images
        if images:
            await self._receive_images(images)
            return True

        text = self._message_text(message)
        if not text:
            await self._say("请输入表情管理命令")
            return True

        if text.startswith("-"):
            await self._run_command(message, text)
            return True

        await self._rename_current(text)
        return True

    async def _run_command(self, message: PrivateMsg, text: str) -> None:
        try:
            invocation = self._command_set.parse(text)
            context = FavCommandContext(message, invocation)
            result = invocation.spec.handler(self, context, invocation.args)
            if inspect.isawaitable(result):
                await result
        except CommandError as exc:
            await self._say(f"命令错误: {exc}")
        except (TypeError, ValueError) as exc:
            await self._say(f"命令错误: {exc}")

    @staticmethod
    def _message_text(message: PrivateMsg) -> str:
        return "".join(
            item.text
            for item in message
            if isinstance(item, TextMsg)
        ).strip()

    async def _receive_images(self, images: list[ImgMsg]) -> None:
        changed = False
        for image in images:
            hash_name = await self._wait_for_hash(image)
            if not hash_name or hash_name in self._pending_add:
                continue

            self._pending_add[hash_name] = FavItem(hash_name=hash_name)
            changed = True

        if changed:
            self._invalidate_confirmations()

    async def _wait_for_hash(self, image: ImgMsg) -> str:
        if image.hash_name:
            return image.hash_name

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._image_wait_timeout
        while not image.hash_name and not image.done:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.05, remaining))

        return image.hash_name

    def _invalidate_confirmations(self) -> None:
        self._add_confirmation = None
        self._delete_confirmation = None

    def _replace_selection(
            self,
            items: list[FavItem],
            source: SelectionSource,
    ) -> None:
        self._invalidate_confirmations()
        self._selection = [FavItem(item.hash_name, item.title) for item in items]
        self._selection_source = source
        self._cursor = None

    def _clear_selection(self) -> None:
        self._replace_selection([], SelectionSource.LIBRARY)

    @staticmethod
    def _items_from_mapping(mapping: dict[str, set[str]]) -> list[FavItem]:
        by_hash: dict[str, str] = {}
        for title in sorted(mapping):
            for hash_name in sorted(mapping[title]):
                if hash_name:
                    by_hash.setdefault(hash_name, title)

        return sorted(
            (FavItem(hash_name, title) for hash_name, title in by_hash.items()),
            key=lambda item: (item.title, item.hash_name),
        )

    @staticmethod
    def _hash_hint(hash_name: str) -> str:
        return f"…{hash_name[-6:]}" if len(hash_name) > 6 else f"…{hash_name[-4:]}"

    @classmethod
    def _format_item(cls, item: FavItem, index: int, total: int) -> str:
        return (
            f"{index}/{total}. [{item.display_title}]  "
            f"{cls._hash_hint(item.hash_name)}"
        )

    def _format_items(self, items: list[FavItem] | None = None) -> str:
        current = self._selection if items is None else items
        if not current:
            return "列表为空"

        total = len(current)
        return "\n".join(
            self._format_item(item, index, total)
            for index, item in enumerate(current, start=1)
        )

    async def _send_images(self, items: list[FavItem]) -> bool:
        if not items:
            return False

        message = SendMessage()
        total = len(items)
        for index, item in enumerate(items, start=1):
            message.append(SendImage(item.hash_name))
            message.append(SendText(f"\n{self._format_item(item, index, total)}"))
            if index < total:
                message.append(SendBreak())

        sent = await self._system_send(message)
        return len(sent) == total and all(result is not None for result in sent)

    async def _send_cursor(self) -> None:
        if self._cursor is None or not self._selection:
            return

        item = self._selection[self._cursor]
        await self._system_send(SendMessage(
            SendImage(item.hash_name),
            SendText(
                f"\n{self._format_item(item, self._cursor + 1, len(self._selection))}"
            ),
        ))

    async def _rename_current(self, title: str) -> None:
        if self._cursor is None or not self._selection:
            await self._say("当前没有可重命名的图片，请先移动游标")
            return

        title = title.strip()
        if not title:
            await self._say("表情标题不能为空")
            return

        if len(title) > 30:
            await self._say("表情标题不能超过30个字符")
            return

        self._invalidate_confirmations()
        current = self._selection[self._cursor]
        if self._selection_source is SelectionSource.PENDING_ADD:
            pending = self._pending_add.get(current.hash_name)
            if pending is None:
                await self._say("当前待新增项目已经不存在")
                return

            pending.title = title

        else:
            renamed = await self.target.store.rename_fav(current.hash_name, title)
            if not renamed:
                await self._say("当前表情已经不在目标库中")
                return

        self._update_title_references(current.hash_name, title)
        await self._say(
            f"已将当前表情重命名为[{title}]；"
            f"当前 {self._cursor + 1}/{len(self._selection)}"
        )

    def _update_title_references(self, hash_name: str, title: str) -> None:
        for collection in (
                self._selection,
                list(self._pending_add.values()),
                list(self._pending_delete.values()),
        ):
            for item in collection:
                if item.hash_name == hash_name:
                    item.title = title

    @COMMANDS.command(
        "--help",
        aliases=("-h", "--commands"),
        summary="查看命令列表与指定命令的用法",
    )
    @COMMANDS.argument("command", nargs="?", default=None)
    async def command_help(
            self,
            _context: FavCommandContext,
            args: CommandArgs,
    ) -> None:
        command = args.command
        if command and not command.startswith("-"):
            command = f"--{command}"

        await self._say(self._command_set.format_help(command))

    @COMMANDS.command("--list", aliases=("-l",), summary="文字列出全部表情")
    async def list_favs(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        items = self._items_from_mapping(self.target.store.fav_list)
        self._replace_selection(items, SelectionSource.LIBRARY)
        await self._say(
            f"目标：{self.target.display_name}\n"
            f"{self._format_items()}"
        )

    @COMMANDS.command("--show", aliases=("-v",), summary="发送整个选择集，慎用")
    async def show(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        if not self._selection:
            await self._say("当前选择集为空")
            return

        if not await self._send_images(self._selection):
            await self._say("选择集图片未全部发送成功")

    @COMMANDS.command("--select", aliases=("-s",), summary="按两个边界裁剪选择集")
    @COMMANDS.argument("bounds", nargs="+")
    async def select(
            self,
            _context: FavCommandContext,
            args: CommandArgs,
    ) -> None:
        if not self._selection:
            await self._say("当前选择集为空")
            return

        start, end = self._parse_bounds(args.bounds)
        if start < 1 or end > len(self._selection) or start > end:
            raise ValueError(f"选择范围必须位于 1 到 {len(self._selection)} 之间")

        selected = self._selection[start - 1:end]
        source = self._selection_source
        self._replace_selection(selected, source)
        await self._say(self._format_items())

    @staticmethod
    def _parse_bounds(values: list[str]) -> tuple[int, int]:
        raw = "".join(values).replace(" ", "").replace('，', ',')
        parts = raw.split(",")
        if len(parts) != 2:
            raise ValueError("--select 必须传入两个边界，例如 --select 2,5")

        try:
            return int(parts[0]), int(parts[1])

        except ValueError as exc:
            raise ValueError("--select 的边界必须是整数") from exc

    @COMMANDS.command(
        "--check-select",
        aliases=("-cs",),
        summary="文字列出当前选择集",
    )
    async def check_select(
            self,
            _context: FavCommandContext,
            _args: CommandArgs,
    ) -> None:
        await self._say(self._format_items())

    @COMMANDS.command("--jump", aliases=("-j",), summary="跳转到指定 index")
    @COMMANDS.argument("index", _type=int)
    async def jump(self, _context: FavCommandContext, args: CommandArgs) -> None:
        if not 1 <= args.index <= len(self._selection):
            await self._say(f"index 必须位于 1 到 {len(self._selection)} 之间")
            return

        self._invalidate_confirmations()
        self._cursor = args.index - 1
        await self._send_cursor()

    @COMMANDS.command("--next", aliases=("-n",), summary="查看下一张")
    async def next(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        if not self._selection:
            await self._say("当前选择集为空")
            return

        if self._cursor is not None and self._cursor >= len(self._selection) - 1:
            await self._say("已到底")
            return

        self._invalidate_confirmations()
        self._cursor = 0 if self._cursor is None else self._cursor + 1
        await self._send_cursor()

    @COMMANDS.command("--prev", aliases=("-p",), summary="查看上一张")
    async def previous(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        if not self._selection:
            await self._say("当前选择集为空")
            return

        if self._cursor is not None and self._cursor <= 0:
            await self._say("已到开头")
            return

        self._invalidate_confirmations()
        self._cursor = (
            len(self._selection) - 1
            if self._cursor is None
            else self._cursor - 1
        )
        await self._send_cursor()

    @COMMANDS.command(
        "--check-add",
        aliases=("-ca",),
        summary="展示并确认待新增清单",
    )
    async def check_add(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        items = [FavItem(item.hash_name, item.title) for item in self._pending_add.values()]
        self._replace_selection(items, SelectionSource.PENDING_ADD)
        if not items:
            await self._say("待新增清单为空")
            return

        delivered = await self._send_images(items)
        unnamed = sum(not item.title for item in items)
        if not delivered:
            await self._say("待新增图片发送失败，本次未确认")
            return

        if unnamed:
            await self._say(f"还有 {unnamed} 张表情未命名，本次未确认")
            return

        self._add_confirmation = self._add_snapshot()
        await self._say(f"已确认待新增清单，共 {len(items)} 张")

    @COMMANDS.command("--add", aliases=("-a",), summary="添加已确认的全部表情")
    async def add(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        snapshot = self._add_snapshot()
        if not snapshot:
            await self._say("待新增清单为空")
            return

        if any(not hash_name or not title for hash_name, title in snapshot):
            await self._say("待新增清单仍有未命名或缺少 hash_name 的项目")
            return

        if self._add_confirmation != snapshot:
            await self._say("待新增清单尚未确认或确认已经失效")
            return

        added = await self.target.store.add_favs(list(snapshot))
        self._pending_add.clear()
        self._add_confirmation = None
        if self._selection_source is SelectionSource.PENDING_ADD:
            self._clear_selection()

        else:
            self._invalidate_confirmations()

        await self._say(f"已添加 {added} 张表情")

    @COMMANDS.command("--clear-add", aliases=("-xa",), summary="清空待新增清单")
    async def clear_add(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        self._invalidate_confirmations()
        count = len(self._pending_add)
        self._pending_add.clear()
        if self._selection_source is SelectionSource.PENDING_ADD:
            self._clear_selection()

        await self._say(f"已清空待新增清单，共移除 {count} 项")

    @COMMANDS.command(
        "--add-delete",
        aliases=("-ad",),
        summary="将选择集或游标当前项加入待删除清单",
    )
    async def add_delete(
            self,
            _context: FavCommandContext,
            _args: CommandArgs,
    ) -> None:
        if not self._selection:
            await self._say("当前选择集为空")
            return

        self._invalidate_confirmations()
        if self._cursor is not None:
            item = self._selection[self._cursor]
            self._pending_delete.setdefault(
                item.hash_name,
                FavItem(item.hash_name, item.title),
            )
            await self._say(
                f"已将当前表情[{item.display_title}]加入待删除清单；"
                f"当前 {self._cursor + 1}/{len(self._selection)}"
            )
            return

        for item in self._selection:
            self._pending_delete.setdefault(
                item.hash_name,
                FavItem(item.hash_name, item.title),
            )

        count = len(self._selection)
        self._clear_selection()
        await self._say(f"已将 {count} 张表情加入待删除清单")

    @COMMANDS.command(
        "--check-delete",
        aliases=("-cd",),
        summary="展示并确认待删除清单",
    )
    async def check_delete(
            self,
            _context: FavCommandContext,
            _args: CommandArgs,
    ) -> None:
        items = [FavItem(item.hash_name, item.title) for item in self._pending_delete.values()]
        self._replace_selection(items, SelectionSource.PENDING_DELETE)
        if not items:
            await self._say("待删除清单为空")
            return

        if not await self._send_images(items):
            await self._say("待删除图片发送失败，本次未确认")
            return

        self._delete_confirmation = self._delete_snapshot()
        await self._say(f"已确认待删除清单，共 {len(items)} 张")

    @COMMANDS.command("--delete", aliases=("-d",), summary="删除已确认的全部表情")
    async def delete(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        snapshot = self._delete_snapshot()
        if not snapshot:
            await self._say("待删除清单为空")
            return

        if self._delete_confirmation != snapshot:
            await self._say("待删除清单尚未确认或确认已经失效")
            return

        removed = await self.target.store.remove_favs(list(snapshot))
        self._pending_delete.clear()
        self._delete_confirmation = None
        if self._selection_source is SelectionSource.PENDING_DELETE:
            self._clear_selection()

        else:
            self._invalidate_confirmations()

        await self._say(f"已删除 {removed} 张表情")

    @COMMANDS.command(
        "--clear-delete",
        aliases=("-xd",),
        summary="清空待删除清单",
    )
    async def clear_delete(
            self,
            _context: FavCommandContext,
            _args: CommandArgs,
    ) -> None:
        self._invalidate_confirmations()
        count = len(self._pending_delete)
        self._pending_delete.clear()
        if self._selection_source is SelectionSource.PENDING_DELETE:
            self._clear_selection()

        await self._say(f"已清空待删除清单，共移除 {count} 项")

    @COMMANDS.command("--remove", aliases=("-r",), summary="移除游标当前项")
    async def remove(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        if self._cursor is None or not self._selection:
            await self._say("当前没有可移除的游标项目")
            return

        self._invalidate_confirmations()
        removed = self._selection.pop(self._cursor)
        if self._selection_source is SelectionSource.PENDING_ADD:
            self._pending_add.pop(removed.hash_name, None)

        elif self._selection_source is SelectionSource.PENDING_DELETE:
            self._pending_delete.pop(removed.hash_name, None)

        if not self._selection:
            self._cursor = None
            await self._say(f"已移除[{removed.display_title}]；当前 0/0")
            return

        self._cursor = min(self._cursor, len(self._selection) - 1)
        await self._send_cursor()

    @COMMANDS.command("--exit", aliases=("-x",), summary="退出表情管理对话")
    async def exit(self, _context: FavCommandContext, _args: CommandArgs) -> None:
        await self._say("已退出表情管理对话")
        await self._close()

    def _add_snapshot(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (item.hash_name, item.title)
            for item in self._pending_add.values()
        )

    def _delete_snapshot(self) -> tuple[str, ...]:
        return tuple(self._pending_delete)

    @property
    def selection(self) -> tuple[FavItem, ...]:
        return tuple(self._selection)

    @property
    def pending_add(self) -> tuple[FavItem, ...]:
        # noinspection PyTypeChecker
        return tuple(self._pending_add.values())

    @property
    def pending_delete(self) -> tuple[FavItem, ...]:
        # noinspection PyTypeChecker
        return tuple(self._pending_delete.values())

    @property
    def cursor_index(self) -> int | None:
        return self._cursor

    @property
    def add_confirmed(self) -> bool:
        return bool(
            self._add_confirmation
            and self._add_confirmation == self._add_snapshot()
        )

    @property
    def delete_confirmed(self) -> bool:
        return bool(
            self._delete_confirmation
            and self._delete_confirmation == self._delete_snapshot()
        )


__all__ = [
    "FavItem",
    "FavSystem",
    "FavTarget",
    "FavTargetError",
    "FavoriteStore",
    "resolve_fav_target",
]
