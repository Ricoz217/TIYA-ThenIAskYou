"""
image_to_text.py
通过VLM把图片转换为文本
"""

__version__ = "0.2.0"

import asyncio
import json
import time
import traceback
from threading import Lock
from typing import Literal
from pathlib import Path

from TIYA.config import BASE_CFG, SETTING_CFG, DATA_DIR
from TIYA.LLM_connect import Chat, Context, SystemPrompt, ImagePrompt, TextPrompt, Prompts, parse_llm_setting
from TIYA.file_cache import get_file_path_async
from TIYA.utils import AutoMapping
from TIYA.logger import get_logger
from TIYA.agent.agent_prompt import AgentPrompt


_log = get_logger()
_CACHE_FILE = DATA_DIR / "image2text" / "image_to_text_cache.json"
_IMAGE_CACHE: AutoMapping[dict] = AutoMapping(_CACHE_FILE, expire_day=lambda: SETTING_CFG.ImageRecognize.CacheExpire,
                           persist_period="SECOND")
_INITIATE_LOCK = Lock()
_POST_LOCK = asyncio.Lock()
_RUNNING_TASK: dict[tuple, asyncio.Task[str]] = {}


async def image2text(
        hash_name: str,
        query: str = "",
        *,
        image_name: str = "",
        prompt: str = None,
        system: str = None,
        mode: Literal["NORMAL", "FAV"] = "NORMAL",
        prefer_fav: bool = False,
        no_cache: bool = False
) -> str:
    """
    图片转文字
    :param hash_name: 项目缓存系统生成的哈希名字
    :param query: 要搜索的内容，若输入则为高精度模式
    :param image_name: 图片名字，唯一名字
    :param prompt:
    :param system:
    :param mode: 识图模型，是否返回表情
    :param prefer_fav: 普通识图时优先返回已登记的表情标题，不会创建表情标题
    :param no_cache:
    :return:
    """
    image_name = image_name or hash_name
    if not no_cache:
        cache = _get_cache(hash_name, query, mode, prefer_fav)
        if cache:
            return cache

    try:
        filepath = await get_file_path_async(hash_name)

    except FileNotFoundError:
        _log.error(f"Can not find cache <{hash_name}>")
        return ""

    except ValueError:
        _log.error(f"Hash name illegal <{hash_name}>")
        return ""

    if system is None:
        system_prompt = AgentPrompt(
            "内置识图系统提示词",
            DATA_DIR / "prompt" / "basic_image2text_system"
        )
        system = await system_prompt.get_prompt()

    if prompt is None:
        user_prompt = AgentPrompt(
            "内置识图用户提示词",
            DATA_DIR / "prompt" / "basic_image2text"
        )
        prompt = await user_prompt.get_prompt(query)

    post_task = await _create_or_get_post_task(
        hash_name=hash_name,
        query=query,
        image_name=image_name,
        filepath=filepath,
        system=system,
        prompt=prompt,
        mode=mode,
        no_cache=no_cache
    )
    await asyncio.shield(post_task)
    if not post_task.done() or post_task.cancelled() or post_task.exception() is not None:
        return ""

    return post_task.result()

async def _create_or_get_post_task(
        *,
        hash_name: str,
        query: str,
        image_name: str,
        filepath: Path,
        system: str,
        prompt: str,
        mode: Literal["NORMAL", "FAV"] = "NORMAL",
        no_cache: bool
) -> asyncio.Task[str]:
    if query:
        vlm_name = BASE_CFG.Agents.ImageModelHigh

    else:
        vlm_name = BASE_CFG.Agents.ImageModel

    key = (image_name, str(filepath), query, prompt, system, vlm_name, mode, no_cache)
    async with _POST_LOCK:
        post_task = _RUNNING_TASK.get(key, None)
        if post_task is None:
            post_task = _RUNNING_TASK[key] = asyncio.create_task(_create_post_task(
                hash_name=hash_name,
                query=query,
                image_name=image_name,
                filepath=filepath,
                prompt=prompt,
                system=system,
                vlm_name=vlm_name,
                mode=mode,
                no_cache=no_cache
            ))

        else:
            if post_task.done() and (post_task.cancelled() or post_task.exception() is not None):
                post_task = _RUNNING_TASK[key] = asyncio.create_task(_create_post_task(
                    hash_name=hash_name,
                    query=query,
                    image_name=image_name,
                    filepath=filepath,
                    prompt=prompt,
                    system=system,
                    vlm_name=vlm_name,
                    mode=mode,
                    no_cache=no_cache
                ))

        return post_task

async def _create_post_task(
        *,
        hash_name: str,
        query: str,
        image_name: str,
        filepath: Path,
        prompt: str,
        system: str,
        vlm_name: str,
        mode: Literal["NORMAL", "FAV"] = "NORMAL",
        no_cache: bool
) -> str:
    key = (image_name, str(filepath), query, prompt, system, vlm_name, mode, no_cache)
    context = Context()
    context.append(SystemPrompt(system))
    prompt = TextPrompt("user", prompt)
    new_prompt = Prompts(prompt, ImagePrompt("user", filepath, image_name))
    client = Chat(keep_alive=False)
    try:
        setting = parse_llm_setting(vlm_name)
        client.setting(setting)
        client.replace_context(context)
        retry_limit = SETTING_CFG.ImageRecognize.VLMRetryLimit  # 默认3
        timeout = SETTING_CFG.ImageRecognize.VLMTimeout  # 默认180
        for _ in range(retry_limit):
            try:
                response = await client.ask(new_prompt, timeout=timeout)

            except Exception as E:
                _log.error(E)
                _log.debug(traceback.format_exc())
                continue

            else:
                if not (response and isinstance(response.prompts[0], TextPrompt)):
                    continue

                text_result = response.prompts[0].text  # type: ignore[attr-defined]
                if not text_result:
                    continue

                if not query:
                    if mode == "FAV":
                        fav_title = _normalize_fav_title(text_result)
                        if not fav_title:
                            continue

                        if not no_cache:
                            set_fav_title(hash_name, fav_title)

                        return fav_title

                    if len(text_result) > 450:
                        text_result = text_result[450:]

                    if not no_cache:
                        _update_cache(hash_name, description=text_result)

                    return text_result

                else:
                    if len(text_result) > 5000:
                        text_result = text_result[5000:]

                    new_query = {query: {"time": time.time(), "result": text_result}}
                    if not no_cache:
                        _update_cache(hash_name, query=new_query, mode=mode)

                    return text_result

    except Exception as E:
        _log.error(E)
        _log.debug(traceback.format_exc())
        return ""

    finally:
        asyncio.create_task(client.close())
        await _end_and_clear(key)

    return ""

async def _end_and_clear(key: tuple):
    """退出时收尾"""
    async with _POST_LOCK:
        _RUNNING_TASK.pop(key, None)

def _update_cache(
    hash_name: str,
        description: str = "",
        query: dict[str, dict] = None,
        mode: Literal["NORMAL", "FAV"] = "NORMAL"
):
    if query is None:
        query = {}

    if not (description or query):
        return

    old_cache = _IMAGE_CACHE.setdefault(hash_name, {})
    if description:
        if mode == "NORMAL":
            old_cache["description"] = description

        elif mode == "FAV":
            old_cache["fav_title"] = description

    old_query = old_cache.setdefault("query", {})
    old_query.update(query)
    old_cache["time"] = time.time()
    _IMAGE_CACHE.touch(hash_name)

def _get_cache(
        hash_name: str,
        query: str = "",
        mode: Literal["NORMAL", "FAV"] = "NORMAL",
        prefer_fav: bool = False
) -> str:
    cache = _IMAGE_CACHE.get(hash_name)
    if cache is None:
        return ""

    record_time = cache.get("time", 0)
    if time.time() - record_time > SETTING_CFG.ImageRecognize.CacheExpire * 24 * 3600:
        _remove_cache(hash_name)
        return ""

    if query:
        query_cache = cache.setdefault("query", {})
        query_result = query_cache.get(query, {})
        if not query_result:
            return ""

        if time.time() - query_result["time"] > SETTING_CFG.ImageRecognize.CacheExpire * 24 * 3600:
            query_cache.pop(query)
            return ""

        return query_result.get("result", "")

    elif mode == "NORMAL":
        if prefer_fav:
            fav_title = _normalize_fav_title(cache.get("fav_title", ""))
            if fav_title:
                return fav_title

        return cache.get("description", "")

    elif mode == "FAV":
        return _normalize_fav_title(cache.get("fav_title", ""))

    return ""

def _remove_cache(hash_name: str):
    _IMAGE_CACHE.remove(hash_name)


def _normalize_fav_title(value: str) -> str:
    if not isinstance(value, str):
        return ""

    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)

    except json.JSONDecodeError:
        title = text

    else:
        if isinstance(parsed, dict):
            title = parsed.get("title", "")

        elif isinstance(parsed, str):
            title = parsed

        else:
            return ""

    if not isinstance(title, str):
        return ""

    title = title.strip()
    if not title or len(title) > 30:
        return ""

    return title


def set_fav_title(hash_name: str, title: str) -> str:
    """登记纯表情标题，供所有识图调用方共享。"""
    fav_title = _normalize_fav_title(title)
    if not fav_title:
        raise ValueError("表情标题必须是30字符以内的非空字符串")

    _update_cache(hash_name, description=fav_title, mode="FAV")
    return fav_title


def is_fav(hash_name: str) -> bool:
    """判断是否表情"""
    cache = _IMAGE_CACHE.get(hash_name)
    if cache is None:
        return False

    fav_title = _normalize_fav_title(cache.get("fav_title", ""))
    if not fav_title:
        return False

    return True
