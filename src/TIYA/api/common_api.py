from __future__ import annotations
"""
common_api.py
通用的 ncatbot SDK api 接口
"""
__version__ = "0.1.0"

from typing import TYPE_CHECKING
from TIYA.config import get_bot

if TYPE_CHECKING:
    from TIYA.mybot import MyBot


async def get_group_list() -> dict:
    bot: MyBot = get_bot()
    if bot is None:
        return {}

    return await bot.api.get_group_list(True)