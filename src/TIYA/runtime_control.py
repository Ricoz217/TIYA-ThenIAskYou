from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Protocol, TextIO

from TIYA.input_hub import get_input_hub


class SaveableGroup(Protocol):
    async def save(self) -> None:
        ...


class ShutdownGroup(Protocol):
    async def shutdown(self) -> None:
        ...


class StartableGroup(Protocol):
    async def start(self) -> None:
        ...


@dataclass(slots=True)
class GroupLoadResult:
    started: list[str]
    pending_config: list[str]


_hot_reload_handler: Callable[[], Awaitable[GroupLoadResult]] | None = None
_hot_reload_lock: asyncio.Lock | None = None


def set_hot_reload_handler(
        handler: Callable[[], Awaitable[GroupLoadResult]],
) -> None:
    """Register the active runtime's hot-reload implementation."""
    global _hot_reload_handler, _hot_reload_lock

    if not callable(handler):
        raise TypeError("handler must be callable")

    _hot_reload_handler = handler
    _hot_reload_lock = None


def clear_hot_reload_handler() -> None:
    """Remove the active runtime's hot-reload implementation."""
    global _hot_reload_handler, _hot_reload_lock

    _hot_reload_handler = None
    _hot_reload_lock = None


async def hot_reload() -> GroupLoadResult:
    """Run the active runtime's serialized hot-reload workflow."""
    global _hot_reload_lock

    handler = _hot_reload_handler
    if handler is None:
        raise RuntimeError("热重载运行时尚未就绪")

    if _hot_reload_lock is None:
        _hot_reload_lock = asyncio.Lock()

    async with _hot_reload_lock:
        return await handler()


def request_restart() -> None:
    """Wake the terminal command loop with the fixed internal restart command."""
    get_input_hub().publish("restart()")


def _config_value(config: object, key: str):
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


async def start_new_groups(
        group_list: Iterable[dict],
        *,
        groups: MutableMapping[str, StartableGroup],
        skipped_groups: Iterable[str],
        group_configs: Mapping[str, object],
        initialize_config: Callable[[dict[str, str]], None],
        make_group: Callable[[dict, object], StartableGroup],
        strict: bool = True
) -> GroupLoadResult:
    skipped = {str(group_id) for group_id in skipped_groups}
    new_group_data: dict[str, dict] = {}
    for group_data in group_list:
        group_id = str(group_data.get("group_id", ""))
        if not group_id or group_id in groups or group_id in skipped:
            continue

        new_group_data[group_id] = group_data

    initialize_config({
        group_id: group_data.get("group_name", "")
        for group_id, group_data in new_group_data.items()
    })

    pending_config = []
    for group_id in new_group_data:
        group_config = group_configs.get(group_id)
        if group_config is None:
            pending_config.append(group_id)
            continue

        chat_model = _config_value(group_config, "chat_model")
        agent_model = _config_value(group_config, "agent_model")
        if "YourModelPresetName" in (chat_model, agent_model):
            pending_config.append(group_id)

    if strict and pending_config:
        lines = ["以下群配置未完成，请填写配置后重启: "]
        lines.extend(f"\t- {group_id}" for group_id in pending_config)
        raise RuntimeError('\n'.join(lines))

    async def _start_group(group_id: str, group_data: dict) -> str:
        group = make_group(group_data, group_configs[group_id])
        await group.start()
        return group_id

    start_tasks = []
    for group_id, group_data in new_group_data.items():
        if group_id in pending_config:
            continue

        start_tasks.append(_start_group(group_id, group_data))

    started = []
    if start_tasks:
        results = await asyncio.gather(*start_tasks, return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]

        started = [result for result in results if isinstance(result, str)]

    return GroupLoadResult(started=started, pending_config=pending_config)


async def save_groups(groups: Iterable[SaveableGroup]) -> list[BaseException]:
    errors: list[BaseException] = []
    for group in list(groups):
        try:
            await group.save()
        except Exception as exc:
            errors.append(exc)
    return errors


async def shutdown_groups(
        groups: Iterable[ShutdownGroup],
        *,
        timeout: float | None = None
) -> list[BaseException]:
    async def _shutdown(group: ShutdownGroup):
        if timeout is None:
            await group.shutdown()
        else:
            await asyncio.wait_for(group.shutdown(), timeout=timeout)

    results = await asyncio.gather(
        *(_shutdown(group) for group in list(groups)),
        return_exceptions=True
    )
    return [result for result in results if isinstance(result, BaseException)]


def build_restart_command(
        *,
        executable: str | None = None,
        module: str = "TIYA.QQ_bot"
) -> list[str]:
    return [executable or sys.executable, "-m", module]


def spawn_restart_process(
        *,
        executable: str | None = None,
        cwd: str | Path,
        platform_name: str | None = None,
        popen=subprocess.Popen
):
    command = build_restart_command(executable=executable)
    environment = os.environ.copy()
    environment["TIYA_RESTARTED"] = "1"
    active_platform = os.name if platform_name is None else platform_name
    kwargs = {
        "cwd": str(cwd),
        "env": environment,
    }
    if active_platform == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore
        kwargs["close_fds"] = False  # type: ignore

    else:
        kwargs["start_new_session"] = True  # type: ignore

    return popen(command, **kwargs)


def pause_before_exit(
        message: str = "",
        *,
        stream: TextIO | None = None,
        input_func=input
) -> None:
    if message:
        print(message)

    active_stream = sys.stdin if stream is None else stream
    try:
        interactive = active_stream.isatty()
    except (AttributeError, OSError):
        interactive = False

    if not interactive:
        return

    try:
        input_func("按回车退出")
    except (EOFError, KeyboardInterrupt):
        return
