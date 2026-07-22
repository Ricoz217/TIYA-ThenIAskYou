from __future__ import annotations
"""
private_api.py
私聊相关的 ncatbot SDK API 接口
"""
__version__ = "0.1.0"

import traceback
from typing import TYPE_CHECKING

from ncatbot.core.message import MessageChain

from TIYA.config import get_bot
from TIYA.logger import get_logger

if TYPE_CHECKING:
    from TIYA.mybot import MyBot


_log = get_logger()


async def say(user_id: str, message_chain: MessageChain) -> dict | None:
    bot: MyBot = get_bot()
    if bot is None:
        _log.error(f"私聊消息发送失败: user_id=[{user_id}], BOT 实例未初始化")
        return None

    if not message_chain.chain:
        _log.debug(f"私聊消息发送跳过: user_id=[{user_id}], message_chain 为空")
        return None

    try:
        response = await bot.api.post_private_msg(user_id, rtf=message_chain)

    except Exception as exc:
        _log.error(f"私聊消息发送失败: user_id=[{user_id}], SDK 异常: {exc}")
        _log.debug(traceback.format_exc())
        return None

    if not _is_ok_response(response):
        _log.error(f"私聊消息发送失败: user_id=[{user_id}], SDK 返回异常: {response}")
        return None

    return response


def _is_ok_response(response: dict | None) -> bool:
    return isinstance(response, dict) and response.get("status") == "ok"
