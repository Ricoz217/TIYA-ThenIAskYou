from __future__ import annotations
"""
auto_fav.py
管理表情包，提供表情列表和表情识别业务。
"""
__version__ = "0.3.1"

import time
import asyncio
import random

from typing import TYPE_CHECKING
from pathlib import Path
from collections import defaultdict
from threading import Lock
from dataclasses import dataclass

from TIYA.logger import get_logger
from TIYA.auto_mapping import AutoMapping
from TIYA.config import SETTING_CFG, DATA_DIR
from TIYA.image_to_text import image2text, set_fav_title
from TIYA.file_cache import check_file_exists, get_file_path_async
from TIYA.agent.agent_prompt import AgentPrompt


if TYPE_CHECKING:
    from TIYA.logger import BlockHandle, LogHandle
    from model.message import BaseMsg

_PROMPT = AgentPrompt("内置系统提示词", DATA_DIR / "prompt" / "basic_autofav_system")
_log = get_logger()
_CREATE_LOCK = Lock()


@dataclass(slots=True)
class AutoFavView:
    _mapping: dict[str, set[str]]
    hosts: list[AutoFav]

    def get_fav(self, title: str) -> str:
        """
        随机返回一张同标题的表情"
        :param title:
        :return: 返回图片的哈希文件名
        """
        favs = self._mapping.get(title)
        if not favs:
            return ""

        chosen = random.choice(list(favs))
        if not check_file_exists(chosen):
            for host in self.hosts:
                host.remove_fav_hash(chosen)

            favs.discard(chosen)
            if not favs:
                self._mapping.pop(title, None)

            return ""

        for host in self.hosts:
            host.renew_fav_hash(title, chosen)

        return chosen

    def __or__(self, other):
        if not isinstance(other, AutoFav):
            return NotImplemented

        self_mapping = self._mapping
        other_mapping = other._mapping.mapping()
        combined_mapping = self_mapping.copy()
        for k, v in other_mapping.items():
            combined_mapping[k] = combined_mapping.get(k, set()).union(v)

        return AutoFavView(combined_mapping, [*self.hosts, other])


class AutoFav:
    def __init__(self, index_file: Path, logger: LogHandle | BlockHandle = None):
        if logger is None:
            logger = _log

        self._logger = logger
        self._recording = False
        self._record_start_time = time.time()
        self._sleep_task = None
        self._mutation_lock = Lock()

        # 缓存容器
        self._mapping: AutoMapping[set[str]] = AutoMapping(
            save_path=index_file,
            persist_period="UPDATE",
            value_type=set,
            expire_day=lambda: SETTING_CFG.Groups.AutoFavExpire  # 默认30天，一般很难触碰
        )  # 这个不是唯一映射: {title: set[hash_id]}
        self._hash_names: set[str] = set().union(*self._mapping.values())  # 查重表
        self._fav_buffer: dict[str, int] = defaultdict(int)
        self._sync_fav_titles()

    async def auto_fav(self, msg: BaseMsg):
        """传入消息，自动发现表情，纯本地算法，不准但经济"""
        images = msg.images
        if not images:
            return

        init_event = [img.wait() for img in images]
        try:
            await asyncio.wait_for(asyncio.gather(*init_event), SETTING_CFG.LLM.MessageParseTimeout)  # 默认60s

        except asyncio.TimeoutError:
            pass

        parsed_img = [img for img in images if img.done]
        for img in parsed_img:
            if not img.done:
                continue

            if img.hash_name in self._hash_names:
                continue

            self._fav_buffer[img.hash_name] += 1

        asyncio.create_task(self.run_period())

    async def add_fav(self, hash_name: str, title: str = "") -> str:
        """手动添加表情"""
        # 测试 hash_name，不存在这里直接抛出异常
        await get_file_path_async(hash_name)
        title = title.strip()
        if len(title) > 30:
            raise ValueError("表情标题不能超过30个字符")

        if not title:
            auto_fav_system = await _PROMPT.get_prompt()
            for _ in range(2):  # 只重试一次，写死
                title = await image2text(
                    hash_name=hash_name,
                    system=auto_fav_system,
                    prompt="按要求处理图片",
                    mode="FAV"
                )
                if not title:
                    continue

                if not isinstance(title, str):
                    continue

                title = title.strip()
                if not title or len(title) > 30:
                    continue

                else:
                    break

        if not title:
            raise RuntimeError("识图模型获取表情标题失败")

        with self._mutation_lock:
            self._mapping.setdefault(title, set()).add(hash_name)
            self._mapping.touch(title)
            self._refresh_hash()

        set_fav_title(hash_name, title)
        return title

    async def add_favs(self, entries: list[tuple[str, str]]) -> int:
        """Add a validated batch without invoking automatic title generation."""
        normalized: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for hash_name, title in entries:
            if not isinstance(hash_name, str) or not hash_name.strip():
                raise ValueError("表情 hash_name 不能为空")

            hash_name = hash_name.strip()
            await get_file_path_async(hash_name)
            title = self._validate_manual_title(title)
            pair = (hash_name, title)
            if pair in seen:
                continue

            seen.add(pair)
            normalized.append(pair)

        with self._mutation_lock:
            for hash_name, title in normalized:
                self._mapping.setdefault(title, set()).add(hash_name)
                self._mapping.touch(title)

            self._refresh_hash()

        for hash_name, title in normalized:
            set_fav_title(hash_name, title)

        return len(normalized)

    async def rename_fav(self, hash_name: str, title: str) -> bool:
        """Collapse one hash to a single title in this favorite library."""
        if not isinstance(hash_name, str) or not hash_name.strip():
            raise ValueError("表情 hash_name 不能为空")

        hash_name = hash_name.strip()
        title = self._validate_manual_title(title)
        found = False
        with self._mutation_lock:
            for old_title, hash_names in list(self._mapping.items()):
                if hash_name not in hash_names:
                    continue

                found = True
                hash_names.remove(hash_name)
                if hash_names:
                    self._mapping.touch(old_title)
                else:
                    self._mapping.pop(old_title)

            if found:
                self._mapping.setdefault(title, set()).add(hash_name)
                self._mapping.touch(title)
                self._refresh_hash()

        if found:
            set_fav_title(hash_name, title)

        return found

    async def remove_favs(self, hash_names: list[str]) -> int:
        """Remove every title relation for the hashes in this library."""
        targets = {
            hash_name.strip()
            for hash_name in hash_names
            if isinstance(hash_name, str) and hash_name.strip()
        }
        if not targets:
            return 0

        return self._remove_fav_hashes(targets)

    def remove_fav_hash(self, hash_name: str) -> bool:
        """Remove one hash from every title in this favorite library."""
        if not isinstance(hash_name, str) or not hash_name:
            return False

        return bool(self._remove_fav_hashes({hash_name}))

    def _remove_fav_hashes(self, targets: set[str]) -> int:
        removed: set[str] = set()
        with self._mutation_lock:
            for title, members in list(self._mapping.items()):
                matches = members.intersection(targets)
                if not matches:
                    continue

                removed.update(matches)
                members.difference_update(matches)
                if members:
                    self._mapping.touch(title)

                else:
                    self._mapping.pop(title)

            self._refresh_hash()

        return len(removed)

    @staticmethod
    def _validate_manual_title(title: str) -> str:
        if not isinstance(title, str):
            raise TypeError("表情标题必须是字符串")

        title = title.strip()
        if not title:
            raise ValueError("表情标题不能为空")
        if len(title) > 30:
            raise ValueError("表情标题不能超过30个字符")

        return title

    async def run_period(self):
        now = time.time()
        period = SETTING_CFG.AutoFav.TimePeriod  # 默认1小时
        # 兜底，防卡死
        if now - self._record_start_time > period + 2 * SETTING_CFG.AutoFav.ImageToTitleTimeout:
            self._recording = False
            if self._sleep_task is not None:
                self._sleep_task.cancel()

        if self._recording:
            return

        self._recording = True
        self._record_start_time = now
        self._sleep_task = asyncio.create_task(asyncio.sleep(period))
        try:
            await self._sleep_task

        except asyncio.CancelledError:
            return

        else:
            await self._update_fav()

        finally:
            self._recording = False

    async def _update_fav(self):
        img_rank = dict(sorted(self._fav_buffer.items(), key=lambda x: x[1], reverse=True))
        self._fav_buffer.clear()
        img_rank = {k: v for k, v in img_rank.items() if
                    v > SETTING_CFG.AutoFav.ImageRepeatGate and k not in self._hash_names}  # 默认3次
        if not img_rank:
            return

        self._logger.info("正在自动添加表情包...")
        auto_fav_system = await _PROMPT.get_prompt()
        tasks: dict[str, asyncio.Task] = {}
        for img in img_rank:
            task = asyncio.create_task(
                image2text(hash_name=img, system=auto_fav_system, prompt="按要求处理图片", mode="FAV"))
            tasks[img] = task

        try:
            await asyncio.wait_for(asyncio.gather(*tasks.values(), return_exceptions=True),
                                   SETTING_CFG.AutoFav.ImageToTitleTimeout)  # 默认120秒

        except asyncio.TimeoutError:
            pass

        new_add = []
        for k, v in tasks.items():
            if v.cancelled() or not v.done() or v.exception() is not None:
                continue

            title = v.result()
            if not isinstance(title, str):
                continue

            title = title.strip()
            if not title or len(title) > 30:
                continue

            new_add.append(title)
            self._mapping.setdefault(title, set()).add(k)
            self._mapping.touch(title)
            set_fav_title(k, title)

        self._refresh_hash()
        if new_add:
            title_list = ""
            for i, new_fav in enumerate(new_add, start=1):
                title_list += f"{i}. {new_fav}\n"

            title_list.strip()
            self._logger.info(f"已新增 [{len(new_add)}] 张表情:\n{title_list}")

        else:
            self._logger.info("无新增表情")

    def _refresh_hash(self):
        """重建刷新查重表，保证准确"""
        self._hash_names.clear()
        self._hash_names.update(*self._mapping.values())

    def _sync_fav_titles(self):
        """将持久化表情索引回填到共享识图缓存。"""
        for title, hash_names in self._mapping.items():
            for hash_name in hash_names:
                set_fav_title(hash_name, title)

    def get_fav(self, title: str) -> str:
        """
        随机返回一张同标题的表情"
        :param title:
        :return: 返回图片的哈希文件名
        """
        favs = self._mapping.get(title)
        if not favs:
            return ""

        chosen = random.choice(list(favs))
        if not check_file_exists(chosen):
            self.remove_fav_hash(chosen)
            return ""

        return chosen

    def renew_fav_hash(self,fav_title: str, fav_hash: str):
        """通过哈希名续期表情"""
        if not fav_hash in self._hash_names:
            return

        self._mapping.get(fav_title)

    @property
    def fav_list(self) -> dict[str, set[str]]:
        return {k: set(v) for k, v in self._mapping.items()}

    def __contains__(self, item):
        if not isinstance(item, str):
            raise TypeError("必须传入表情标题")

        return item in self._mapping

    def __or__(self, other) -> AutoFavView:
        if not isinstance(other, AutoFav):
            return NotImplemented

        self_mapping = self._mapping.mapping()
        other_mapping = other._mapping.mapping()
        combined_mapping = self_mapping.copy()
        for k, v in other_mapping.items():
            combined_mapping[k] = combined_mapping.get(k, set()).union(v)

        return AutoFavView(combined_mapping, [self, other])


_AUTO_FAV_STORAGE: dict[str, AutoFav] = {}

def get_autofav(index_file: Path, logger: LogHandle | BlockHandle = None) -> AutoFav:
    """用于获取AUTOFAV单例"""
    with _CREATE_LOCK:
        key = str(index_file)
        if key in _AUTO_FAV_STORAGE:
            return _AUTO_FAV_STORAGE[key]

        else:
            new_auto_fav = AutoFav(index_file, logger)
            _AUTO_FAV_STORAGE[key] = new_auto_fav
            return new_auto_fav
