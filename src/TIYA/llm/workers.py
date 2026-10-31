"""LLM 同步任务统一入口，线程数及退出收尾由 TIYA.executor 管理。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import copy_context
from functools import partial

from TIYA.executor import LLM_EXECUTOR


async def run_sync(function: Callable, /, *args, **kwargs):
    call = partial(function, *args, **kwargs)
    # noinspection PyTypeChecker
    return await asyncio.get_running_loop().run_in_executor(
        LLM_EXECUTOR, copy_context().run, call
    )