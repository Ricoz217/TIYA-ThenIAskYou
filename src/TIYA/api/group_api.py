from __future__ import annotations
"""
group_api.py
群聊相关的 ncatbot SDK api 接口
"""
__version__ = "0.1.0"

import asyncio
import base64
import traceback
from typing import TYPE_CHECKING
from pathlib import Path

from TIYA.file_cache import get_file_path_async
from TIYA.config import get_bot, get_bot_uid
from TIYA.logger import get_logger

from ncatbot.core.message import MessageChain

if TYPE_CHECKING:
    from TIYA.mybot import MyBot

_log = get_logger()


async def get_group_member_list(group_id: str) -> dict | None:
    bot: MyBot = get_bot()
    if bot is None:
        return None

    return await bot.api.get_group_member_list(group_id, no_cache=True)

async def get_group_ban_list(group_id: str) -> dict | None:
    bot: MyBot = get_bot()
    if bot is None:
        return None

    return await bot.api.get_group_shut_list(group_id)

async def say(group_id: str, message_chain: MessageChain) -> dict:
    async def _none():
        return None

    bot: MyBot = get_bot()
    if not message_chain.chain:
        _log.debug(f"群消息发送跳过: group_id=[{group_id}], message_chain 为空")
        return {"files": {}}

    chain_mirror = message_chain.chain.copy()
    files: list[dict] = []
    for item in message_chain.chain:
        item_type = item.get("type", "")
        if not item_type:
            continue

        if item_type == "file":
            files.append(item)
            chain_mirror.remove(item)

    # 把基本内容和文件分开发送
    file_tasks: dict[str, asyncio.Task] = {}
    if chain_mirror and bot is not None:
        say_task = asyncio.create_task(bot.api.post_group_msg(group_id, rtf=MessageChain(chain_mirror)))

    else:
        if chain_mirror and bot is None:
            _log.error(f"群消息发送失败: group_id=[{group_id}], BOT 实例未初始化")
        say_task = asyncio.create_task(_none())

    for file in files:
        path = file["data"].get("file", "")
        if not path:
            _log.error(f"群文件发送跳过: group_id=[{group_id}], file 消息缺少路径: {file}")
            continue

        path = Path(path.replace("file:///", ""))
        name = path.stem
        try:
            base64_str = f"base64://{base64.b64encode(path.read_bytes()).decode()}"

        except OSError as exc:
            _log.error(f"群文件发送失败: group_id=[{group_id}], file=[{path}], 读取文件失败: {exc}")
            _log.debug(traceback.format_exc())
            continue

        if bot is None:
            _log.error(f"群文件发送失败: group_id=[{group_id}], file=[{path}], BOT 实例未初始化")
            file_tasks[name] = asyncio.create_task(_none())

        else:
            file_tasks[name] = asyncio.create_task(
                bot.api.upload_group_file(
                    group_id=group_id,
                    file=base64_str,
                    name=name,
                    folder_id='/'
                )
            )

    await asyncio.gather(say_task, *file_tasks.values(), return_exceptions=True)
    result = {}
    if chain_mirror:
        if not say_task.done() or say_task.cancelled() or say_task.exception() is not None:
            if say_task.cancelled():
                _log.error(f"群消息发送失败: group_id=[{group_id}], SDK 发送任务已取消")

            elif say_task.exception() is not None:
                exc = say_task.exception()
                _log.error(f"群消息发送失败: group_id=[{group_id}], SDK 异常: {exc}")
                _log.debug(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

            else:
                _log.error(f"群消息发送失败: group_id=[{group_id}], SDK 发送任务未完成")

            result["say"] = None

        else:
            result["say"] = say_task.result()
            if not _is_ok_response(result["say"]):
                _log.error(f"群消息发送失败: group_id=[{group_id}], SDK 返回异常: {result['say']}")

    result["files"] = {}

    for k, v in file_tasks.items():
        if not v.done() or v.cancelled() or v.exception() is not None:
            if v.cancelled():
                _log.error(f"群文件发送失败: group_id=[{group_id}], file=[{k}], SDK 上传任务已取消")

            elif v.exception() is not None:
                exc = v.exception()
                _log.error(f"群文件发送失败: group_id=[{group_id}], file=[{k}], SDK 异常: {exc}")
                _log.debug(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

            else:
                _log.error(f"群文件发送失败: group_id=[{group_id}], file=[{k}], SDK 上传任务未完成")
            result["files"][k] = None

        else:
            result["files"][k] = v.result()
            if not _is_ok_response(result["files"][k]):
                _log.error(f"群文件发送失败: group_id=[{group_id}], file=[{k}], SDK 返回异常: {result['files'][k]}")

    return result


def _is_ok_response(response: dict | None) -> bool:
    return isinstance(response, dict) and response.get("status") == "ok"

async def upload_group_file(group_id: str, file: str, filename: str, folder_id: str = '/') -> dict | None:
    """
    上传群文件
    :param group_id: 群号
    :param file: 本地缓存系统的哈希id
    :param filename: 上传后的文件名
    :param folder_id: 上传到群文件的哪个文件夹，要先通过API获取文件夹id
    :return:
    """
    bot: MyBot = get_bot()
    if bot is None:
        return None

    try:
        path = await get_file_path_async(file)

    except FileNotFoundError:
        _log.error(f"上传群文件失败: 文件 [{file}] 不存在")
        return None

    except ValueError:
        _log.error(f"上传群文件失败: 文件 [{file}] 哈希id错误")
        return None

    base64_str = f"base64://{base64.b64encode(path.read_bytes()).decode()}"
    response: dict = await bot.api.upload_group_file(
        group_id,
        base64_str,
        filename,
        folder_id
    )
    return response

async def get_file_message_id(
        group_id: str,
        filename: str,
        retry_interval: float = 1
) -> str | None:
    bot: MyBot = get_bot()
    if bot is None:
        return None

    bot_uid = get_bot_uid()
    for _ in range(3):
        await asyncio.sleep(retry_interval)
        response: dict = await bot.api.get_group_msg_history(group_id, 0, 100, False)
        if not response or response["status"] != "ok":
            continue

        messages: list[dict] = response["data"]["messages"]
        for msg in reversed(messages):
            if str(msg.get("user_id", "")) != str(bot_uid):
                continue

            for segment in msg.get("message", []):
                if segment.get("type") != "file":
                    continue

                if segment.get("data", {}).get("file") != filename:
                    continue

                msg_id = msg.get("message_seq") or msg.get("message_id")
                return str(msg_id) if msg_id is not None else None

    return None
