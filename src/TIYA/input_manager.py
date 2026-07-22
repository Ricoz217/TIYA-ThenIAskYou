"""Centralized command input helpers powered by InputHub."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from TIYA.input_hub import get_input_hub

if TYPE_CHECKING:
    from TIYA.logger import Logger

def _resolve_logger(logger: Logger | None) -> Logger:
    if logger is not None:
        return logger
    from TIYA.logger import get_logger

    return get_logger()


def read_command(prompt: str = "", *, logger: Logger | None = None, echo: bool = True) -> str:
    """Read one command via the process-global InputHub.

    This call is blocking for the caller, but InputHub collects terminal input in
    its own reader thread, so other application threads/tasks keep running.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "read_command() cannot run inside an active event loop. "
            "Use `await read_command_async(...)` instead."
        )

    hub = get_input_hub()
    active_logger: Logger | None = None
    if echo:
        active_logger = _resolve_logger(logger)
        hub.set_logger(active_logger)
    else:
        active_logger = logger
        hub.set_logger(None)
    hub.configure(prompt=prompt, echo_to_logger=echo)
    hub.start()
    while True:
        command = hub.read(timeout=0.2)
        if command is not None:
            if active_logger is not None:
                try:
                    active_logger.note_input_submit()
                except Exception:
                    pass
            return command


async def read_command_async(prompt: str = "", *, logger: Logger | None = None, echo: bool = True) -> str:
    """Read one command in async contexts via InputHub queue."""
    hub = get_input_hub()
    active_logger: Logger | None = None
    if echo:
        active_logger = _resolve_logger(logger)
        hub.set_logger(active_logger)
    else:
        active_logger = logger
        hub.set_logger(None)
    hub.configure(prompt=prompt, echo_to_logger=echo)
    hub.start()
    while True:
        command = await hub.read_async(timeout=0.2)
        if command is not None:
            if active_logger is not None:
                try:
                    active_logger.note_input_submit()
                except Exception:
                    pass
            return command
