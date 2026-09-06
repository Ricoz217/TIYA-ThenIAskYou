from __future__ import annotations
"""
setu.py，一个 Pixiv 的色图模块
"""
__version__ = "0.2.0"

import os
import time
import asyncio
import traceback
import json
import copy
import random
import zipfile
import tempfile
import math

from dataclasses import dataclass, field, asdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import datetime
from typing import TypedDict, Callable, Awaitable, Any
from enum import Enum, auto
from threading import Lock as TLock
from itertools import combinations
from io import BytesIO
from PIL import Image, UnidentifiedImageError, ImageFile
from collections import defaultdict

from pixivpy_async import PixivClient, PixivError
from pixivpy_async.aapi import AppPixivAPI

from TIYA.private_chat import NoticeManager
from TIYA.command import CommandArgs, CommandError, CommandSet
from TIYA.model.message import TextMsg
from TIYA.LLM_connect import Chat, parse_llm_setting, TextPrompt
from TIYA.agent.agent_prompt import AgentPrompt
from TIYA.file_cache import add_file_async, check_file_exists, get_file_path_async
from TIYA.time_id import next_time_id
from TIYA.config import (
    SETTING_CFG,
    BASE_CFG,
    get_proxy,
    config_update,
    PIXIV_CHUNKS_DIR,
    PROMPTS_DIR,
    CONFIG_OBSERVER,
    GROUPS_DIR
)
from TIYA.logger import get_logger
from TIYA.auto_mapping import AutoMapping
from TIYA.utils import atomic_save_json, httpx_downloader, HttpxResponse
from TIYA.re_filter import _chunk_index_filter
from TIYA.executor import GLOBAL_EXECUTOR
from .remote import (
    RemoteLoginCancelled,
    RemoteLoginError,
    RemoteLoginUnavailable,
    capture_pixiv_code
)
from .public_url import (
    discover_public_login_url,
    select_public_bind_host
)

_log = get_logger()
_SINGLETON_LOCK = TLock()
_LOGIN_LOCK = asyncio.Lock()
_LOGIN_TASK: asyncio.Task[str] | None = None
_IMAGE_DOWNLOAD_LOCK = asyncio.Lock()
_IMAGE_DOWNLOAD_TASKS: dict[str, asyncio.Task] = {}
_SETU_INTERACTION_COMMANDS = CommandSet(prefix="--")
_INTERNAL_PARAMS = {
    "max_cal_tags": 5,  # 最大计算tag数
    "max_searching_workers": 5,  # 最大并发搜索数
    "tags_sequence_reward_factor": 0.2,  # tag顺序奖励因子
    "multiply_tags_hit_reward_factor": 0.15,  # tag多次命中奖励
    "bookmark_score": {
        "top": {
            "bookmark": 10_000,
            "score_percentage": 1
        },
        "bottom": {
            "bookmark": 2000,
            "score_percentage": 0.5
        }
    },
    "score_weight": {
        "bookmark": 0.5,
        "tags": 0.5,
    },
    "group_score_weight": {
        "bookmark": 0.4,
        "artist": 0.3,
        "tags": 0.3
    },
    "used_penalty_factor": 0.06,
    "min_bookmark_threshold": 1000,
    "local_search_score_threshold": 0.6,  # 本地搜索分数阈值
    "related_search_score_threshold": 0.55,  # 相关画廊搜索分数阈值
    "online_search_score_threshold": 0.35,  # 原生搜索分数阈值
    "ugoira_missing_frames_threshold": 0.03,  # 动图帧丢失允许的阈值
    "rating_min_samples": 5,
}
_TEXT2TAG_PROMPT = AgentPrompt(
    name="text2tag_system",
    data_path=PROMPTS_DIR / "pixiv_text2tag_system",
    description="自然语言转标签的系统提示词",
    role="system"
)
_DOWNLOAD_HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.82 '
                          'Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Upgrade-Insecure-Requests': '1',
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Referer": "https://www.pixiv.net/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Sec-Ch-Ua": '"Chromium";v="124", "Microsoft Edge";v="124", "Not-A.Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"'
        }
POST_PROCESS_HASH_MAPPING: AutoMapping[str] = AutoMapping(
    save_path=PIXIV_CHUNKS_DIR.parent / "postprocess_mapping.json",
    expire_day=lambda : SETTING_CFG.Common.FileCacheExpire,
    persist_period="UPDATE"
)


class IllustType(Enum):
    ILLUST = "illust"
    UGOIRA = "ugoira"
    MANGA = "manga"


class PayloadType(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    RANDOM = auto()
    QUERY = auto()
    ARTIST = auto()
    ILLUST = auto()


class _User(TypedDict):
    id: int
    name: str
    is_followed: bool


class _Tag(TypedDict):
    name: str
    translated_name: str | None


class _PixivResults(TypedDict):
    illustrations: list[Illustration]
    illusts: list[Illust]


class _SearchResult(TypedDict):
    iid: str
    illustration: Illustration
    illust: Illust
    score: float


class _PixivUgoiraFrame(TypedDict):
    file: str
    index: int
    delay: int


@dataclass(slots=True)
class Illustration:
    """完整的插画信息"""
    id: int  # 插画id
    title: str  # 插画标题
    type: IllustType  # 插画类型
    caption: str  # 插画注释
    restrict: int  # 访问限制: {0: 公开, 1: 仅登录用户, 2: 仅自己}
    user: _User  # 作者信息，存了个字典
    tags: list[_Tag]  # 标签
    page_count: int  # 页数，即一个插图页图片总数
    sanity_level: int  # 敏感级别{0~2: 全年龄, 3~4: R15, 5~6: R18, 7+: R18G}，不准
    total_view: int  # 访问/阅读数
    total_bookmarks: int  # 红心数
    illust_ai_type: int  # 是否AI: {0: 非AI或无法判断, 1: AI生成, 2: 可能是AI}，很不准
    img_urls: dict[str, str]  # 图片数据链接
    is_bookmarked: bool  # 是否已收藏
    NSFW: bool  # 是否R18，不准，不要走这个
    visible: bool  # 是否还可见（是否已经被删除）
    add_time: float = field(default_factory=time.time)  # 入库时间
    hash_name: dict[str, str] = field(default_factory=dict)  # 本地缓存

    @property
    def iid(self) -> str:
        return str(self.id)

    @property
    def artist_id(self) -> int:
        return self.user["id"]

    @property
    def artist_name(self) -> str:
        return self.user["name"]

    @property
    def tags_set(self) -> set[str]:
        return {tag["name"] for tag in self.tags}

    @property
    def tags_list(self) -> list[str]:
        return [tag["name"] for tag in self.tags]

    @property
    def pages(self) -> list[int]:
        return [p for p in range(self.page_count)]

    @property
    def bookmark(self) -> int:
        return self.total_bookmarks

    def copy(self) -> Illustration:
        return self.__copy__()

    def __copy__(self) -> Illustration:
        return Illustration(
            id=self.id,
            title=self.title,
            type=self.type,
            caption=self.caption,
            restrict=self.restrict,
            user=copy.deepcopy(self.user),
            tags=copy.deepcopy(self.tags),
            page_count=self.page_count,
            sanity_level=self.sanity_level,
            total_view=self.total_view,
            total_bookmarks=self.total_bookmarks,
            illust_ai_type=self.illust_ai_type,
            img_urls=self.img_urls.copy(),
            is_bookmarked=self.is_bookmarked,
            NSFW=self.NSFW,
            visible=self.visible,
            add_time=self.add_time,
            hash_name=self.hash_name.copy()
        )

    def to_dict(self) -> dict:
        output = asdict(self)
        output["type"] = self.type.value
        return output

    @classmethod
    def from_dict(cls, data: dict) -> Illustration | None:
        if not isinstance(data, dict):
            return None

        _id = data.get("id", 0)
        title = data.get("title", "")
        illust_type = data.get("type", "")
        caption = data.get("caption", "")
        restrict = data.get("restrict", 0)
        user = data.get("user", {})
        tags = data.get("tags", [])
        page_count = data.get("page_count", 0)
        sanity_level = data.get("sanity_level", 0)
        total_view = data.get("total_view", 0)
        total_bookmarks = data.get("total_bookmarks", 0)
        illust_ai_type = data.get("illust_ai_type", 0)
        img_urls = data.get("img_urls", {})
        is_bookmarked = data.get("is_bookmarked", False)
        NSFW = data.get("NSFW", True)
        visible = data.get("visible", True)
        add_time = data.get("add_time", time.time())
        hash_name = data.get("hash_name", {})

        # 校验必须项
        if any(not item for item in (
            _id,
            illust_type,
            user,
            page_count,
            img_urls
        )):
            return None

        # 校验链接数量是否足够
        if len(img_urls) != page_count:
            return None

        # 校验枚举
        if illust_type not in IllustType._value2member_map_:
            return None

        return cls(
            id=_id,
            title=title,
            type=IllustType(illust_type),
            caption=caption,
            restrict=restrict,
            user=_User(**user),
            tags=tags.copy(),
            page_count=page_count,
            sanity_level=sanity_level,
            total_view=total_view,
            total_bookmarks=total_bookmarks,
            illust_ai_type=illust_ai_type,
            img_urls=img_urls.copy(),
            is_bookmarked=is_bookmarked,
            NSFW=NSFW,
            visible=visible,
            add_time=add_time,
            hash_name=hash_name.copy()
        )


@dataclass(slots=True)
class Illust:
    """轻量化的插画信息，带储存索引"""
    id: int  # 插画id
    artist_id: int  # 画师id
    artist_name: str  # 画师名字
    title: str  # 插画标题
    chunk: int  # chunk index
    pages: list[int]  # 可用分页
    tags: set[str]  # 标签
    bookmark: int  # 红心
    NSFW: bool  # 是否色图
    attempts: dict[str, int] = field(default_factory=dict)  # 记录发送尝试
    scores: dict[str, dict[str, float]] = field(default_factory=dict)  # page -> user_id -> [0, 1]

    @property
    def iid(self) -> str:
        return str(self.id)

    def to_dict(self) -> dict:
        output = asdict(self)
        output["tags"] = list(self.tags)
        return output

    @classmethod
    def from_dict(cls, data: dict) -> Illust | None:
        if not isinstance(data, dict):
            return None

        _id = data.get("id")
        artist_id = data.get("artist_id")
        artist_name = data.get("artist_name", "")
        title = data.get("title", "")
        chunk = data.get("chunk", 0)
        pages = data.get("pages", [])
        tags = data.get("tags", [])
        bookmark = data.get("bookmark", 0)
        NSFW = data.get("NSFW", False)
        attempts = data.get("attempts", {})
        raw_scores = data.get("scores", {})

        if any(not item for item in {
            _id,
            artist_id,
            artist_name,
            chunk
        }):
            return None

        tags = set(tags)
        scores: dict[str, dict[str, float]] = {}
        if isinstance(raw_scores, dict):
            for raw_page, raw_users in raw_scores.items():
                if not isinstance(raw_users, dict):
                    continue

                page_scores = {}
                for raw_user_id, raw_score in raw_users.items():
                    if not isinstance(raw_score, (int, float)):
                        continue

                    score = float(raw_score)
                    if not math.isfinite(score):
                        continue

                    page_scores[str(raw_user_id)] = min(1.0, max(0.0, score))

                if page_scores:
                    scores[str(raw_page)] = page_scores

        return cls(
            id=_id,
            artist_id=artist_id,
            artist_name=artist_name,
            title=title,
            chunk=chunk,
            pages=pages,
            tags=tags,
            bookmark=bookmark,
            NSFW=NSFW,
            attempts=attempts,
            scores=scores
        )

    def copy(self) -> Illust:
        return self.__copy__()

    def __copy__(self) -> Illust:
        return Illust(
            id=self.id,
            artist_id=self.artist_id,
            artist_name=self.artist_name,
            title=self.title,
            chunk=self.chunk,
            pages=self.pages.copy(),
            tags=self.tags.copy(),
            bookmark=self.bookmark,
            NSFW=self.NSFW,
            attempts=self.attempts.copy(),
            scores=copy.deepcopy(self.scores)
        )


@dataclass(slots=True)
class Ugoira:
    url: str
    frames: list[_PixivUgoiraFrame] = field(default_factory=list)


@dataclass(slots=True)
class PixivPayload:
    count: int
    freeze_id: str
    query_text: str
    query_tags: list[str]
    payload_type: PayloadType
    illusts: dict[str, Illust] = field(default_factory=dict)
    illustrations: dict[str, Illustration] = field(default_factory=dict)
    hash_names: dict[str, str] = field(default_factory=dict)
    create_time: float = field(default_factory=time.time)

    def __bool__(self):
        return bool(self.hash_names)

    def __len__(self):
        return len(self.hash_names)


@dataclass(frozen=True, slots=True)
class SetuSendReceipt:
    sent: bool
    message_id: str | None = None


@dataclass(frozen=True, slots=True)
class SetuInteractionContext:
    illust: Illust
    user_id: str
    allow_global: bool = False
    global_targets: tuple[Setu, ...] = ()


@dataclass(slots=True)
class PixivFreeze:
    illusts: dict[str, Illust] = field(default_factory=dict)
    freeze_id: str = field(default_factory=lambda : str(next_time_id()))
    freeze_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        illusts = {k: v.to_dict() for k, v in self.illusts.items()}
        return {
            "illusts": illusts,
            "freeze_id": self.freeze_id,
            "freeze_time": self.freeze_time
        }

    @classmethod
    def from_dict(cls, data: dict) -> PixivFreeze | None:
        if not isinstance(data, dict):
            return None

        illusts = data.get("illusts", {})
        freeze_id = data.get('freeze_id', "")
        freeze_time = data.get("freeze_time", time.time())
        if not freeze_id:
            return None

        parsed_illusts = {}
        for k, v in illusts.items():
            illust = Illust.from_dict(v)
            if illust is None:
                continue

            parsed_illusts[k] = illust

        return PixivFreeze(
            illusts=parsed_illusts,
            freeze_id=freeze_id,
            freeze_time=freeze_time
        )

    def __bool__(self) -> bool:
        return bool(self.illusts)


@dataclass(slots=True)
class PixivStorageChunk:
    index: int
    last_update: float = field(default_factory=time.time)
    data: dict[str, Illustration] = field(default_factory=dict)
    loaded: bool = False

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "last_update": self.last_update,
            "data": {
                iid: illustration.to_dict()
                for iid, illustration in self.data.items()
            },
        }

    def _load_dict(self, data: dict):
        index = data.get("index", -1)
        last_update = data.get("last_update", time.time())
        content = data.get("data", {})

        if self.index != index:
            _log.error(f"[PIXIV] chunk_{self.index}.json 已损坏，正在新建文件...")
            return

        self.last_update = last_update
        if not content:
            return

        self.data.clear()
        for raw_data in content.values():
            illustration = Illustration.from_dict(raw_data)
            if illustration is None:
                continue

            self.data[illustration.iid] = illustration

    def ensure_loaded(self):
        if self.loaded:
            return

        self.load()

    def load(self):
        filepath = PIXIV_CHUNKS_DIR / f"chunk_{self.index}.json"
        if not filepath.is_file():
            _log.info(f"[PIXIV] chunk_{self.index}.json 不存在，正在新建文件...")
            self.loaded = True
            return

        try:
            load_content: dict = json.loads(filepath.read_text(encoding="utf-8"))

        except json.JSONDecodeError:
            _log.error(f"[PIXIV] chunk_{self.index}.json 已损坏，正在新建文件...")
            return

        else:
            if not isinstance(load_content, dict):
                _log.error(f"[PIXIV] chunk_{self.index}.json 已损坏，正在新建文件...")
                return

            self._load_dict(load_content)

        finally:
            self.loaded = True

    def purge(self):
        self.data.clear()
        self.loaded = False

    def save(self):
        filepath = PIXIV_CHUNKS_DIR / f"chunk_{self.index}.json"
        atomic_save_json(
            content=self.to_dict(),
            target_path=filepath
        )

    def to_obj(self, illustration: Illustration) -> tuple[Illustration, Illust]:
        result_illust = Illust(
            id=illustration.id,
            artist_id=illustration.artist_id,
            artist_name=illustration.artist_name,
            title=illustration.title,
            chunk=self.index,
            pages=illustration.pages,
            tags=illustration.tags_set,
            bookmark=illustration.bookmark,
            NSFW=illustration.NSFW
        )
        return illustration, result_illust

    def _add_one(self, illustration: Illustration) -> tuple[Illustration, Illust]:
        self.data[illustration.iid] = illustration
        return self.to_obj(illustration)

    def add(self, illustration: dict | Illustration) -> tuple[Illustration, Illust]:
        if isinstance(illustration, dict):
            illustration = Illustration.from_dict(illustration)
            if illustration is None:
                raise ValueError("[PIXIV] 无法解析 Illustration 字典，数据有误")

        result_illustration, result_illust = self._add_one(illustration)
        self.save()
        return result_illustration, result_illust

    def extend(self, *args: dict | Illustration) -> list[tuple[Illustration, Illust]]:
        results = []
        for item in args:
            if not isinstance(item, (dict, Illustration)):
                continue

            if isinstance(item, dict):
                item = Illustration.from_dict(item)
                if item is None:
                    continue

            results.append(self._add_one(item))

        self.save()
        return results

    def get(self, iid: str | int) -> tuple[Illustration | None, Illust | None]:
        iid = str(iid)
        illustration = self.data.get(iid, None)
        if illustration is None:
            return None, None

        return self.to_obj(illustration)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data.values())

    def __reversed__(self):
        return reversed(self.data.values())

    def __contains__(self, item: str | Illustration):
        if isinstance(item, Illustration):
            item = item.iid

        elif not isinstance(item, str):
            raise TypeError("不支持的类型")

        return item in self.data

    def __del__(self):
        self.purge()


class PixivAccessTokenExpiredError(PixivError):
    pass


class PixivAuthFailError(PixivError):
    pass


class MyAppPixivAPI(AppPixivAPI):
    def __init__(self, **requests_kwargs):
        super().__init__(**requests_kwargs)

    @staticmethod
    async def wait_private_chat_token_response(echo: str) -> str | None:
        timeout = SETTING_CFG.SETU.TokenResponseTimeout
        token_response = await NoticeManager.wait_any(echo, timeout=timeout)
        if token_response is None or not token_response.replies:
            return None

        token_message = token_response.replies[0]
        text = ""
        for item in token_message:
            if isinstance(item, TextMsg):
                text = item.text
                break

        if not text:
            return None

        if text.lower() != "#retry":
            await NoticeManager.notice_someone(token_response.user_id, f"[Pixiy] 正在以 [{text}] 登录...")
            _log.info(f"[Pixiy] 正在以 [{text}] 登录...")

        return text

    async def process_token_login(self) -> dict | None:
        """处理竟态问题，只允许一个登录链路"""
        global _LOGIN_TASK
        # 查询任务
        async with _LOGIN_LOCK:
            # 只要新任务
            if _LOGIN_TASK is None or _LOGIN_TASK.done():
                token_task = _LOGIN_TASK = asyncio.create_task(self._token_login())

            else:
                return None

        # 让异常传递
        return await token_task

    @staticmethod
    async def _notify_remote_login_url(login_url: str) -> None:
        security_note = (
            "\n警告：当前链接使用未加密 HTTP，请仅在可信网络中使用。"
            if login_url.startswith("http://")
            else ""
        )
        message = (
            "[Pixiv] 请打开下面的一次性链接完成登录：\n"
            f"{login_url}\n"
            "这是运行在 TIYA 服务器上的临时浏览器，登录完成后链接会自动失效。"
            f"{security_note}"
        )
        results = await NoticeManager.notice_all(message)
        sent = any(
            any(item is not None for item in messages)
            for messages in results.values()
        )
        if not sent:
            raise RemoteLoginUnavailable("无法向通知者发送远程登录链接")

    async def _request_manual_login_code(self, tip_text: str) -> str | None:
        response_timeout = SETTING_CFG.SETU.TokenResponseTimeout
        notice_echo = None
        for _ in range(2):
            notice_echo = await NoticeManager.request_all(
                tip_text,
                response_timeout=response_timeout,
                intercept=True,
            )
            if notice_echo is not None:
                break

        if notice_echo is None:
            raise PixivError(
                "无法发送 token 获取消息，请检查网络或是否添加了通知者QQ号"
            )
        return await self.wait_private_chat_token_response(notice_echo)

    async def _obtain_authorization_code(
            self,
            login_url: str,
            manual_tip_text: str,
    ) -> str | None:
        public_url = str(
            BASE_CFG.SETU.get("PixivLoginPublicUrl", "") or ""
        ).strip()
        auto_discovered = False
        if not public_url:
            if not BASE_CFG.SETU.get("PixivLoginAutoDiscoverIP", True):
                return await self._request_manual_login_code(manual_tip_text)
            bind_port = int(
                BASE_CFG.SETU.get("PixivLoginBindPort", 8765)
            )
            public_port = int(
                BASE_CFG.SETU.get("PixivLoginPublicPort", 0)
            ) or bind_port
            try:
                public_url = await discover_public_login_url(
                    public_port,
                    ipv4_endpoints=BASE_CFG.SETU.get(
                        "PixivLoginIPv4Endpoints",
                        None,
                    ),
                    ipv6_endpoints=BASE_CFG.SETU.get(
                        "PixivLoginIPv6Endpoints",
                        None,
                    ),
                )
                auto_discovered = True
                _log.warning(
                    "[Pixiv] 未配置公网 URL，已自动生成未加密登录地址"
                )
            except (RemoteLoginError, TypeError, ValueError) as exc:
                _log.warning(
                    f"[Pixiv] 无法自动获取公网登录地址，切换到手动登录: {exc}"
                )
                return await self._request_manual_login_code(manual_tip_text)

        proxy_config = get_proxy(
            BASE_CFG.SETU.get("ProxyMode", "NecessaryProxy")
        )
        proxy = proxy_config.get("https") or proxy_config.get("http")
        bind_host = select_public_bind_host(
            public_url,
            str(
                BASE_CFG.SETU.get(
                    "PixivLoginBindHost",
                    "0.0.0.0",
                )
            ),
            auto_discovered=auto_discovered,
        )
        try:
            return await capture_pixiv_code(
                login_url,
                public_base_url=public_url,
                bind_host=bind_host,
                bind_port=int(
                    BASE_CFG.SETU.get("PixivLoginBindPort", 8765)
                ),
                notify_login_url=self._notify_remote_login_url,
                timeout=SETTING_CFG.SETU.TokenResponseTimeout,
                proxy=proxy,
                locale=str(
                    BASE_CFG.SETU.get("PixivLoginLocale", "zh-HK")
                ),
                tls_certfile=(
                    ""
                    if auto_discovered
                    else BASE_CFG.SETU.get("PixivLoginTLSCert", "")
                ),
                tls_keyfile=(
                    ""
                    if auto_discovered
                    else BASE_CFG.SETU.get("PixivLoginTLSKey", "")
                ),
                allow_insecure_public_url=auto_discovered,
            )
        except RemoteLoginCancelled as exc:
            raise PixivError("用户取消了 Pixiv 登录") from exc

        except (RemoteLoginError, TypeError, ValueError) as exc:
            _log.warning(
                f"[Pixiv] 远程一键登录不可用，切换到手动登录: {exc}"
            )
            return await self._request_manual_login_code(manual_tip_text)

    async def _token_login(self):
        from base64 import urlsafe_b64encode
        from hashlib import sha256
        from secrets import token_urlsafe
        from urllib.parse import urlencode

        LOGIN_URL = "https://app-api.pixiv.net/web/v1/login"
        REDIRECT_URI = "https://app-api.pixiv.net/web/v1/users/auth/pixiv/callback"

        tip_text = ("请按照提示进行pixiv账号登录：\n"
                    "1. 使用浏览器访问下方的登录链接，在登录前按'F12'打开控制台；\n"
                    "2. 以chrome为例，切换至<网络>选项卡，勾选保留日志；\n"
                    "3. 正常登录pixiv账号，直到出现一个空白的页面；\n"
                    "4. 在搜索框中输入'callback?'，点击结果中的':path:'条目，会找到一个开头为'/web/v1/users/auth/pixiv/callback?'url；\n"
                    "5. 将上述url末尾处的code=xxxxxx复制下来(不包括code=)，xxxxxx就是登录需要的token；\n"
                    "6. 在本私聊对话中直接输入复制下来的token，或在程序终端中输入pixiv_token(复制下来的token)；\n"
                    "7. 请注意，该token的有效期很短，请在看到空白页面后两分钟内输入，否则需要重新登录。\n"
                    "完整文档请查看https://gist.github.com/ZipFile/c9ebedb224406f4f11845ab700124362\n\n")

        # print("please visit this before continue:\n\thttps://gist.github.com/ZipFile/c9ebedb224406f4f11845ab700124362")

        def s256(_data):
            """S256 transformation method."""
            return urlsafe_b64encode(sha256(_data).digest()).rstrip(b"=").decode("ascii")

        """Proof Key for Code Exchange by OAuth Public Clients (RFC7636)."""
        code_verifier = token_urlsafe(32)
        code_challenge = s256(code_verifier.encode("ascii"))
        login_params = {
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "client": "pixiv-android",
        }
        login_url = f"{LOGIN_URL}?{urlencode(login_params)}"
        # print(f"login link:\n\t{login_url}")
        tip_text += f"登录链接：{login_url}"
        # code = input("code: ").strip()

        code = await self._obtain_authorization_code(login_url, tip_text)
        if code == "#retry":
            return await self.login(refresh_token=BASE_CFG.SETU.PixivToken)

        if not code:
            raise PixivError("没有获取到 token，色图模块初始化失败，请重启后重试")

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "code_verifier": code_verifier,
            "grant_type": "authorization_code",
            "include_policy": "true",
            "redirect_uri": REDIRECT_URI,
        }
        headers = {
            'User-Agent': self.user_agent
        }

        # return auth/token response
        return await self.auth_req(self.api.auth, headers, data)

    async def login_web(self):
        return await self.process_token_login()

    # 用于检测 token 过期
    async def requests_call(
            self,
            method,
            url,
            headers=None,
            params=None,
            data=None,
    ):
        result = await super().requests_call(
            method,
            url,
            headers=headers,
            params=params,
            data=data,
        )

        error = result.get("error") if isinstance(result, dict) else None
        message = error.get("message", "") if isinstance(error, dict) else ""

        if "invalid_grant" in message or "Access Token" in message:
            raise PixivAccessTokenExpiredError(message)

        return result


class AsyncPixivApi:
    """连接pixiv，单例"""
    _instance: AsyncPixivApi = None
    def __init__(self):
        # 数据存储
        self._user_id = ""
        self._access_token = ""
        self._refresh_token = ""
        self._chunks: dict[int, PixivStorageChunk] = {}
        timeout = SETTING_CFG.SETU.IdsExpireDay  # 默认 30 天
        self._ids_mapping: AutoMapping[int] = AutoMapping(
            save_path=PIXIV_CHUNKS_DIR.parent / "pixiv_ids.json",
            expire_day=timeout,
            persist_period="SECOND"
        )
        self._tags_mapping: AutoMapping[float] = AutoMapping(
            save_path=PIXIV_CHUNKS_DIR.parent / "pixiv.tags.json",
            expire_day=timeout,
            persist_period="SECOND"
        )

        # 配置
        self._max_loaded_chunk = SETTING_CFG.SETU.MaxLoadedChunks  # 默认 50 个

        # 标识符
        self._last_login_time = 0
        self._index_now = self._get_lastest_index()

        # 锁
        self._login_lock = asyncio.Lock()
        self._chunk_lock = TLock()
        self._chunk_purge_lock = TLock()
        self._add_lock = TLock()
        self._fetch_lock = asyncio.Lock()

        # 工具对象
        self._client: PixivClient | None = None
        self.PIXIV: MyAppPixivAPI | None = None

        # 持续任务
        self._fetch_task: asyncio.Task | None = None

    @classmethod
    def get_api(cls) -> AsyncPixivApi:
        """单例取对象"""
        with _SINGLETON_LOCK:
            if cls._instance is None:
                cls._instance = cls()

            return cls._instance

    # =========================================================
    # 登录和验证相关
    # =========================================================

    async def _ensure_api(self):
        if self.PIXIV is None:
            if self._client is not None:
                await self._client.close()

            proxy_dict: dict = get_proxy(BASE_CFG.SETU.ProxyMode)
            self._client = client = PixivClient(proxy=proxy_dict.get("http", None))
            self.PIXIV = MyAppPixivAPI(client=client)

        await self._ensure_auth()

        # 维护自动任务
        if self._fetch_task is None:
            self._fetch_task = asyncio.create_task(self._pixiv_auto_fetch_new())

        elif self._fetch_task.done():
            self._fetch_task.cancel()
            self._fetch_task = asyncio.create_task(self._pixiv_auto_fetch_new())

    async def _ensure_auth(self, force=False):
        self._user_id = self.PIXIV.user_id
        access_token_now = self._access_token = self.PIXIV.access_token
        async with self._login_lock:
            if time.time() - self._last_login_time < 3300 and not force:
                return

            if force and access_token_now != self._access_token:
                return

            await self.pixiv_login()

    async def pixiv_login(self):
        async def _login_and_update_token():
            try:
                await self.PIXIV.login_web()

            except Exception as E:
                _log.error(f"[Pixiv] 登录失败，请重启后重试，错误如下: {E}")
                _log.debug(traceback.format_exc())
                raise PixivAuthFailError("登录失败")

            else:
                self._user_id = self.PIXIV.user_id
                self._access_token = self.PIXIV.access_token
                self._refresh_token = self.PIXIV.refresh_token
                self._last_login_time = time.time()
                self._save_refresh_token()
                return

        if not self._refresh_token:
            save_token = BASE_CFG.SETU.get("PixivToken", "")
            if not save_token:
                await _login_and_update_token()
                return

            else:
                self._refresh_token = save_token

        try:
            await self.PIXIV.login(refresh_token=self._refresh_token)

        except PixivError:
            await _login_and_update_token()
            return

        else:
            self._user_id = self.PIXIV.user_id
            self._access_token = self.PIXIV.access_token
            self._refresh_token = self.PIXIV.refresh_token
            self._last_login_time = time.time()
            self._save_refresh_token()

    def _save_refresh_token(self):
        if self._refresh_token:
            BASE_CFG.SETU.PixivToken = self._refresh_token
            config_update()

    # =========================================================
    # 本地文件数据库接口
    # =========================================================

    @staticmethod
    def _get_lastest_index() -> int:
        chunks = list(PIXIV_CHUNKS_DIR.iterdir())
        if not chunks:
            return 1

        chunks = [chunk for chunk in chunks if chunk.is_file() and _chunk_index_filter.match(chunk.name)]
        if not chunks:
            return 1

        lastest_chunk = max(int(_chunk_index_filter.match(chunk.name).group(1)) for chunk in chunks)
        return lastest_chunk

    def get_or_load_chunk(self, index: int = None) -> PixivStorageChunk:
        def _check_chunk_limit():
            """超了就清理最老的一个 chunk"""
            with self._chunk_purge_lock:
                need_del = len(self._chunks) - self._max_loaded_chunk
                if need_del <= 0:
                    return

                keys = sorted(self._chunks.keys())[:need_del]
                for key in keys:
                    self._chunks.pop(key, None)

        if index is None:
            index = self._index_now

        else:
            with self._chunk_lock:
                chunk = self._chunks.setdefault(index, PixivStorageChunk(index=index))
                chunk.ensure_loaded()
                _check_chunk_limit()
                return chunk

        with self._chunk_lock:
            chunk = self._chunks.get(index, None)
            if chunk is not None:
                if len(chunk) <= 500:
                    _check_chunk_limit()
                    return chunk

                index += 1
                self._index_now = index

            chunk = self._chunks[index] = PixivStorageChunk(
                index=index
            )
            chunk.ensure_loaded()
            _check_chunk_limit()
            return chunk

    @staticmethod
    def _parse_img_urls(raw_data: dict) -> dict[str, str]:
        output = {}
        page_count = raw_data.get("page_count", 0)
        if not page_count:
            return {}

        if page_count > 1:
            pages: list[dict] = raw_data["meta_pages"]
            for i, url in enumerate(pages):
                output[str(i)] = url["image_urls"]["original"]

        else:
            output['0'] = raw_data["meta_single_page"]["original_image_url"]

        return output

    @staticmethod
    def _parse_illustration(raw_data: dict) -> Illustration | None:
        """从 API 的响应中解析"""
        R18_TAG = {
            "R-18",
            "R-18G",
            "子宮",
            "中出し",
            "中出"
        }

        iid = raw_data.get("id", 0)
        title = raw_data.get("title", "")
        illust_type = raw_data.get("type", "")
        caption = raw_data.get("caption", "")
        restrict = raw_data.get("restrict", 0)
        user = raw_data.get("user", {})
        tags = raw_data.get("tags", [])
        page_count = raw_data.get("page_count", 0)
        sanity_level = raw_data.get("sanity_level", 0)
        total_view = raw_data.get("total_view", 0)
        total_bookmarks = raw_data.get("total_bookmarks", 0)
        illust_ai_type = raw_data.get("illust_ai_type", 0)
        img_urls = AsyncPixivApi._parse_img_urls(raw_data)
        is_bookmarked = raw_data.get("is_bookmarked", False)
        visible = raw_data.get("visible", True)
        NSFW = any(tag["name"] in R18_TAG for tag in tags)
        add_time = time.time()

        # 校验必须项
        if any(not item for item in (
            iid,
            illust_type,
            user,
            page_count,
            img_urls
        )):
            return None

        # 校验链接数量是否足够
        if len(img_urls) != page_count:
            return None

        # 校验枚举
        if illust_type not in IllustType._value2member_map_:
            return None

        # 构建TypeDict
        user = _User(
            id=int(user["id"]),
            name=user["name"],
            is_followed=user["is_followed"]
        )

        tags = [
            _Tag(
                name=tag["name"],
                translated_name=tag["translated_name"]
            )
            for tag in tags
        ]

        return Illustration(
            id=int(iid),
            title=title,
            type=IllustType(illust_type),
            caption=caption,
            restrict=restrict,
            user=user,
            tags=tags,
            page_count=page_count,
            sanity_level=sanity_level,
            total_view=total_view,
            total_bookmarks=total_bookmarks,
            illust_ai_type=illust_ai_type,
            img_urls=img_urls,
            is_bookmarked=is_bookmarked,
            NSFW=NSFW,
            visible=visible,
            add_time=add_time
        )

    def _add_new_illust_executor(self, input_data: list[Illustration], force_add: bool, results_container: list):
        need_add = []
        need_update = []
        with self._add_lock:
            for new_illustration in input_data:
                if not force_add and new_illustration.iid in self._ids_mapping:
                    need_update.append(new_illustration)
                    continue

                need_add.append(new_illustration)

            if not (need_add or need_update):
                return

            # 先处理新图片
            if need_add:
                chunk = self.get_or_load_chunk()
                results_container.extend(chunk.extend(*need_add))
                for illustration, _ in results_container:
                    self._ids_mapping[illustration.iid] = chunk.index

                self.update_tags(*need_add)

            # 再处理老图片更新
            chunk_mapping = {}
            for new_illustration in need_update:
                chunk_mapping.setdefault(self._ids_mapping[new_illustration.iid], []).append(new_illustration)

            for index, illustrations in chunk_mapping.items():
                chunk = self.get_or_load_chunk(index=index)
                for new_illustration in illustrations:
                    if new_illustration not in chunk:
                        continue

                    old_illustration, _ = chunk.get(new_illustration.iid)
                    if old_illustration is None:
                        chunk.add(new_illustration)
                        continue

                    old_illustration.title = new_illustration.title
                    old_illustration.caption = new_illustration.caption
                    old_illustration.tags = new_illustration.tags.copy()
                    old_illustration.restrict = new_illustration.restrict
                    old_illustration.sanity_level = new_illustration.sanity_level
                    old_illustration.total_view = new_illustration.total_view
                    old_illustration.total_bookmarks = new_illustration.total_bookmarks
                    old_illustration.illust_ai_type = new_illustration.illust_ai_type
                    old_illustration.img_urls = new_illustration.img_urls.copy()
                    old_illustration.is_bookmarked = new_illustration.is_bookmarked
                    old_illustration.visible = new_illustration.visible
                    old_illustration.hash_name.update(new_illustration.hash_name)

                chunk.save()

        if need_add:
            _log.info(f"[PIXIV] 已新增 [{len(need_add)}] 个画廊")

        if need_update:
                _log.info(f"[PIXIV] 已更新 [{len(need_update)}] 个画廊")

    async def add_new_illust(
            self,
            *args: Illustration,
            force: bool = False
    ) -> list[tuple[Illustration, Illust]]:
        """带查重的新增画廊"""
        if not args:
            return []

        clear_new_illustrations = [_new for _new in args if isinstance(_new, Illustration)]
        if not clear_new_illustrations:
            return []

        results = []
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            GLOBAL_EXECUTOR,
            self._add_new_illust_executor,
            clear_new_illustrations, force, results
        )
        self._ids_mapping.persist()
        return results

    def get_illustration(self, illust: Illust | str) -> tuple[Illustration | None, Illust | None]:
        """直接从数据库取回对象，保证是数据库内的，分为两种模式"""
        if isinstance(illust, Illust):
            illust_id = illust.iid
            chunk = self.get_or_load_chunk(illust.chunk)

        elif isinstance(illust, str):
            illust_id = illust
            if illust not in self._ids_mapping:
                return None, None

            index = self._ids_mapping[illust]
            chunk = self.get_or_load_chunk(index)

        else:
            raise TypeError(f"不支持的类型: {type(illust)}")

        return chunk.get(illust_id)

    @staticmethod
    async def download_image(illustration: Illustration, _pages: list[int]) -> dict[str, str]:
        """采用唯一任务制，防止同一个画廊重复下载"""
        sem = asyncio.Semaphore(SETTING_CFG.SETU.ConcurrentDownloadLimit)  # 默认 5 个
        async def sem_helper(_url: str):
            async with sem:
                return await httpx_downloader(
                    url=_url,
                    retry=retry_limit,
                    timeout=timeout,
                    proxy=proxy,
                    sleep=lambda: random.uniform(1, 3),
                    headers=_DOWNLOAD_HEADERS
                )

        async def _get_or_create_download_task(_url: str) -> asyncio.Future:
            async with _IMAGE_DOWNLOAD_LOCK:
                _task = _IMAGE_DOWNLOAD_TASKS.get(_url, None)
                if _task is None or not _task.done() or _task.cancelled() or _task.exception() is not None:
                    _task = _IMAGE_DOWNLOAD_TASKS[_url] = asyncio.create_task(sem_helper(_url))

                return asyncio.shield(_task)

        if not illustration.img_urls:
            return {}

        # 获取 proxy，没有填默认 NecessaryProxy
        if SETTING_CFG.SETU.DownloaderProxy:
            proxy = get_proxy(SETTING_CFG.SETU.DownloaderProxy)

        else:
            proxy = get_proxy("NecessaryProxy")

        timeout = SETTING_CFG.SETU.DownloaderTimeout  # 默认 180 秒
        retry_limit = SETTING_CFG.SETU.DownloaderRetry  # 默认 3 次

        tasks: dict[str, asyncio.Future[HttpxResponse]] = {}
        download_results = {}
        for k, v in illustration.img_urls.items():
            if int(k) not in _pages:
                continue

            tasks[k] = await _get_or_create_download_task(v)

        # 等待任务返回
        await asyncio.gather(*tasks.values(), return_exceptions=True)

        # 跳过失败
        for k, v in tasks.items():
            if not v.done() or v.cancelled() or v.exception() is not None:
                continue

            response = v.result()
            if not response.ok:
                continue

            content = response.content
            if not content:
                continue

            try:
                _hash_name = await add_file_async(content, file_type="image", note="色图")

            except ValueError:
                continue

            download_results[k] = _hash_name

        return download_results

    async def _get_ugoira_file(self, illustration: Illustration) -> dict[str, str]:
        """直接返回合成好的原始gif"""
        async def _get_or_create_download_task(_url: str) -> asyncio.Task:
            async with _IMAGE_DOWNLOAD_LOCK:
                _task = _IMAGE_DOWNLOAD_TASKS.get(_url, None)
                if _task is None or not _task.done() or _task.cancelled() or _task.exception() is not None:
                    _task = _IMAGE_DOWNLOAD_TASKS[_url] = asyncio.create_task(httpx_downloader(
                url=_url,
                retry=retry_limit,
                timeout=timeout,
                proxy=proxy,
                sleep=lambda: random.uniform(1, 3),
                headers=_DOWNLOAD_HEADERS
            ))
                return _task

        # 先检查是否已有缓存
        if illustration.hash_name:
            hash_name = next(iter(illustration.hash_name.values()))
            try:
                await get_file_path_async(hash_name)

            except (ValueError, FileNotFoundError):
                pass

            else:
                return illustration.hash_name.copy()

        metadata = await self.pixiv_get_ugoira_data(illustration.iid)
        if metadata is None:
            return {}

        # 下载动图
        timeout = SETTING_CFG.SETU.DownloaderTimeout  # 默认 180 秒
        retry_limit = SETTING_CFG.SETU.DownloaderRetry  # 默认 3 次

        # 获取 proxy，没有填默认 NecessaryProxy
        if SETTING_CFG.SETU.DownloaderProxy:
            proxy = get_proxy(SETTING_CFG.SETU.DownloaderProxy)

        else:
            proxy = get_proxy("NecessaryProxy")

        try:
            task = await _get_or_create_download_task(metadata.url)
            await task
            response = task.result()

        except (Exception, asyncio.TimeoutError) as E:
            _log.error(f"[PIXIV] 动图下载失败: {E}")
            _log.debug(traceback.format_exc())
            return {}

        except asyncio.CancelledError:
            _log.error(f"[PIXIV] 动图下载已取消")
            return {}

        else:
            if not response.ok:
                _log.error(f"[PIXIV] 动图下载失败")
                return {}

            content = response.content
            with tempfile.TemporaryDirectory() as temp_dir:
                with BytesIO(content) as zip_buffer:
                    try:
                        with zipfile.ZipFile(zip_buffer) as zip_file:
                            zip_file.extractall(temp_dir)

                    except Exception as E:
                        _log.error(f"[PIXIV] 动图解压失败: {E}")
                        _log.debug(traceback.format_exc())
                        return {}

                # 校验帧列表
                exist_frames = os.listdir(temp_dir)
                checked_frames = {frame["index"]: frame for frame in metadata.frames if frame["file"] in exist_frames}

                if len(checked_frames) / len(metadata.frames) < 1 - _INTERNAL_PARAMS["ugoira_missing_frames_threshold"]:
                    _log.error(f"[PIXIV] 动图下载失败，丢失帧过多")
                    return {}

                # 合成GIF
                opened_frames: dict[int, ImageFile.ImageFile] = {}
                for index, frame in checked_frames.items():
                    frame_path = os.path.join(temp_dir, frame["file"])
                    try:
                        frame_obj = Image.open(frame_path)
                        if frame_obj.mode != "RGBA":
                            frame_obj = frame_obj.convert("RGBA")

                        opened_frames[index] = frame_obj

                    except UnidentifiedImageError:
                        continue

                if len(opened_frames) < 2:
                    _log.error(f"[PIXIV] 动图下载失败，损坏帧过多")
                    return {}

                frames_index = list(opened_frames.keys())
                first_frame = opened_frames[frames_index[0]]
                follow_frames = [opened_frames[index] for index in frames_index[1:]]
                durations = [checked_frames[index]["delay"] for index in frames_index]
                with BytesIO() as gif_buffer:
                    first_frame.save(
                        gif_buffer,
                        format='GIF',
                        append_images=follow_frames,
                        save_all=True,
                        duration=durations,
                        loop=0,
                        optimize=True
                    )
                    try:
                        hash_name = await add_file_async(gif_buffer, file_type="image", note="色图")

                    except TypeError:
                        _log.error(f"[PIXIV] 动图解析失败")
                        return {}

                    else:
                        illustration.hash_name.clear()
                        illustration.hash_name.update({'0': hash_name})
                        await self.add_new_illust(illustration)
                        async with _IMAGE_DOWNLOAD_LOCK:
                            # noinspection PyAsyncCall
                            _IMAGE_DOWNLOAD_TASKS.pop(metadata.url, None)

                        return {'0': hash_name}

                    finally:
                        for frame in opened_frames.values():
                            frame.close()

    async def get_image(self, illustration: Illustration | str) -> dict[str, str]:
        """获取图片，返回的是哈希id字典。如果传入对象一定保证是数据库内的对象，不会做二次哈希名验证，没有就直接下载"""
        if isinstance(illustration, Illustration):
            pass

        elif isinstance(illustration, str):
            illustration, _ = self.get_illustration(illustration)
            if illustration is None:
                return {}

        else:
            raise TypeError(f"不支持的类型: {type(illustration)}")

        # 先检查总库里有没有，没有的话把不在索引里面的图片丢出去，包括失效的
        if not await self.check_illustration_visible(illustration):
            if illustration.iid not in self._ids_mapping:
                _log.warning(f"[PIXIV] 获取图片文件失败，[{illustration.iid}] 不在色图库内")
                return {}

        # 把动图分出去
        if illustration.type is IllustType.UGOIRA:
            return await self._get_ugoira_file(illustration)

        # 下载缺失的图片
        missing_pages = illustration.pages
        need_del = []
        for page, hash_name in illustration.hash_name.items():
            if hash_name:
                try:
                    await get_file_path_async(hash_name)

                except (ValueError, FileNotFoundError):
                    need_del.append(page)
                    continue

                else:
                    if int(page) in missing_pages:
                        missing_pages.remove(int(page))

        for page in need_del:
            illustration.hash_name.pop(page, None)

        if missing_pages:
            download_results = await self.download_image(illustration, missing_pages)
            for page, hash_name in download_results.items():
                illustration.hash_name[page] = hash_name

            # 更新数据库
            await self.add_new_illust(illustration)
            async with _IMAGE_DOWNLOAD_LOCK:
                for page in download_results:
                    key = illustration.img_urls.get(page, "")
                    # noinspection PyAsyncCall
                    _IMAGE_DOWNLOAD_TASKS.pop(key, None)

        return copy.deepcopy(illustration.hash_name.copy())

    def update_tags(self, *args: Illustration):
        now = time.time()
        new_tags = {tag: now for illustration in args if isinstance(illustration, Illustration) for tag in illustration.tags_set}
        self._tags_mapping.update_from_dict(new_tags)
        _log.info(f"[PIXIV] 已更新 [{len(new_tags)}] 个 tag")

    async def illustration_to_illust(self, *args: Illustration) -> dict[str, Illust]:
        """这个链路会调用入库"""
        illusts = {}
        chunk_cache: dict[int, PixivStorageChunk] = {}  # chunk缓存
        ids_mapping = self._ids_mapping.mapping()
        need_add = []
        for illustration in args:
            if not isinstance(illustration, Illustration):
                continue

            iid = illustration.iid
            if not iid in ids_mapping:
                need_add.append(illustration)
                continue

            index = ids_mapping[iid]
            chunk = chunk_cache.setdefault(index, self.get_or_load_chunk(index))
            if iid not in chunk:
                continue

            _, illust = chunk.to_obj(illustration)
            illusts[iid] = illust

        # 处理需要新增的
        new_illusts = await self.add_new_illust(*need_add)
        for _, new_illust in new_illusts:
            illusts[new_illust.iid] = new_illust

        return illusts

    def illust_to_illustration(self, *args: Illust) -> dict[str, Illustration]:
        """这个链路不会调用入库，不存在的会直接跳过"""
        missing = 0
        chunk_mapping: dict[int, dict[str, Illust]] = {}
        for illust in args:
            if not isinstance(illust, Illust):
                continue

            chunk_mapping.setdefault(illust.chunk, {})[illust.iid] = illust

        results: dict[str, Illustration] = {}
        for chunk_index, content in chunk_mapping.items():
            chunk = self.get_or_load_chunk(chunk_index)
            for iid in content:
                illustration, _ = chunk.get(iid)
                if illustration is None:
                    missing += 1
                    continue

                results[iid] = illustration

        if missing:
            _log.warning(f"[PIXIV] 丢失了 {missing} 个画廊")

        return results

    async def check_illustration_visible(self, illustration: Illustration | str) -> bool:
        """请求API，确保画廊还可用，避免作者已删除"""
        if isinstance(illustration, Illustration):
            illust_id = illustration.iid
            if len(illustration.hash_name) == illustration.page_count:
                # 只要有一张图失败都扔掉
                all_exist = True
                for hash_name in illustration.hash_name.values():
                    if not check_file_exists(hash_name):
                        all_exist = False

                if all_exist:
                    return True

        elif isinstance(illustration, str):
            illust_id = illustration
            if illust_id in self._ids_mapping:
                chunk_index = self._ids_mapping[illust_id]
                chunk = self.get_or_load_chunk(chunk_index)
                illustration, illust = chunk.get(illust_id)
                if illustration is not None:
                    if len(illustration.hash_name) == illustration.page_count:
                        # 只要有一张图失败都扔掉
                        all_exist = True
                        for hash_name in illustration.hash_name.values():
                            if not check_file_exists(hash_name):
                                all_exist = False

                        if all_exist:
                            return True

        else:
            raise TypeError(f"不支持的类型: {type(illustration)}")

        check_illustration = await self.pixiv_get_illust_detail(illust_id)
        if check_illustration is None:
            return False

        return check_illustration.visible

    # =========================================================
    # 业务相关
    # =========================================================

    @staticmethod
    def _illustration_filter(
            illustration: Illustration,
            min_bookmarks: int = None
    ) -> bool:
        """过滤器，判断是否要入库"""
        if not illustration.visible:
            return False

        allowed_type = (IllustType.ILLUST, IllustType.UGOIRA)
        if illustration.type not in allowed_type:
            return False

        if min_bookmarks is None:
            min_bookmarks = SETTING_CFG.SETU.MinBookmarksGate  # 默认 2000 个

        if illustration.bookmark < min_bookmarks:
            return False

        return True

    def _is_new_illust(self, illust_id: str) -> bool:
        """查重，真为新"""
        return illust_id not in self._ids_mapping

    def is_valid_illust(self, illust_id: Illust | str) -> bool:
        """简单快速判断画廊id是否还在映射表内，即是否有效"""
        if isinstance(illust_id, Illust):
            illust_id = Illust.iid

        return illust_id in self._ids_mapping

    async def _parse_raw_to_result(self, raw_dicts: list[dict], bookmark_threshold: int = None) -> _PixivResults:
        output = _PixivResults(
            illustrations=[],
            illusts=[]
        )
        if not raw_dicts:
            return output

        illustrations = []
        for raw_data in raw_dicts:
            illustration = self._parse_illustration(raw_data)
            if illustration is not None and self._illustration_filter(illustration, bookmark_threshold):
                illustrations.append(illustration)

        self.update_tags(*illustrations)

        add_results = await self.add_new_illust(*illustrations)
        for illustration, illust in add_results:
            output["illustrations"].append(illustration)
            output["illusts"].append(illust)

        return output

    async def _pixiv_get_next_page(
            self,
            next_url: Any,
            method: Callable[..., Awaitable[Any]],
            pages: int = None,
            expect_new_count: int = None,
            logger_prefix: str = ""
    ) -> list:
        output = []
        new_count = 0
        if pages is None:
            pages = SETTING_CFG.SETU.CrawlPages  # 默认3页

        next_page = self.PIXIV.parse_qs(next_url)
        for _ in range(pages):
            next_page.pop("viewed", None)
            try:
                response = await method(**next_page)

            except PixivAccessTokenExpiredError:
                await self._ensure_auth(True)
                response = await method(**next_page)

            except Exception as E:
                _log.error(f"{logger_prefix}{E}")
                _log.debug(traceback.format_exc())
                break

            if not response.illusts:
                break

            output.extend(response.illusts)
            if expect_new_count and expect_new_count > 0:
                for raw_data in response.illusts:
                    illustration = self._parse_illustration(raw_data)
                    if illustration is None:
                        continue

                    if self._illustration_filter(illustration) and self._is_new_illust(illustration.iid):
                        new_count += 1

                if new_count >= expect_new_count:
                    break

            if not response.next_url:
                break

            next_page = self.PIXIV.parse_qs(response.next_url)

        return output

    async def pixiv_get_ugoira_data(
            self,
            illust_id: str
    ) -> Ugoira | None:
        await self._ensure_api()

        try:
            response = await self.PIXIV.ugoira_metadata(int(illust_id))

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.ugoira_metadata(int(illust_id))

        except Exception as E:
            _log.error(f"[PIXIV] 获取动图元数据失败: {E}")
            _log.debug(traceback.format_exc())
            return None

        if not response.ugoira_metadata:
            _log.warning(f"[PIXIV] 动图 [{illust_id}] 元数据为空，可能不是动图。")
            return None

        url = response.ugoira_metadata["zip_urls"]["medium"]
        frames = []
        for index, frame_data in enumerate(response.ugoira_metadata["frames"]):
            frames.append(_PixivUgoiraFrame(
                file=frame_data["file"],
                index=index,
                delay=frame_data["delay"]
            ))

        return Ugoira(
            url=url,
            frames=frames
        )

    def get_local_illustration(self, illust_id: str) -> tuple[Illustration | None, Illust | None]:
        if illust_id not in self._ids_mapping:
            return None, None

        chunk_index = self._ids_mapping[illust_id]
        chunk = self.get_or_load_chunk(chunk_index)
        return chunk.get(illust_id)

    async def pixiv_get_ranking(self) -> _PixivResults:
        """获取周排行"""
        raw_results = []
        await self._ensure_api()
        try:
            response = await self.PIXIV.illust_ranking("week")

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.illust_ranking("week")

        except Exception as E:
            _log.error(f"[PIXIV] 获取每周热门失败: {E}")
            _log.debug(traceback.format_exc())
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )

        if not response.illusts:
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )

        raw_results.extend(response.illusts)
        # 获取后续页面
        if response.next_url:
            next_pages_result = await self._pixiv_get_next_page(
                response.next_url,
                self.PIXIV.illust_ranking,
                logger_prefix="[PIXIV] 获取每周热门失败: "
            )
            raw_results.extend(next_pages_result)

        return await self._parse_raw_to_result(raw_results)

    async def pixiv_get_follow_illusts(self) -> _PixivResults:
        """获取关注画师新画廊"""
        raw_results = []
        await self._ensure_api()

        # 先获取公开关注
        try:
            response = await self.PIXIV.illust_follow()

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.illust_follow()

        except Exception as E:
            _log.error(f"[PIXIV] 获取关注列表新画廊失败: {E}")
            _log.debug(traceback.format_exc())
            response = None

        if response is None:
            pass

        elif not response.illusts:
            _log.warning(f"[PIXIV] 关注列表为空，是否还未关注任何画师？")

        else:
            raw_results.extend(response.illusts)
            if response.next_url:
                next_page_result = await self._pixiv_get_next_page(
                    next_url=response.next_url,
                    method=self.PIXIV.illust_follow,
                    logger_prefix="[PIXIV] 获取关注列表新画廊失败: "
                )
                raw_results.extend(next_page_result)

        # 再获取私人关注
        try:
            response = await self.PIXIV.illust_follow(restrict="private")

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.illust_follow(restrict="private")

        except Exception as E:
            _log.error(f"[PIXIV] 获取关注列表新画廊失败: {E}")
            _log.debug(traceback.format_exc())
            response = None

        if response is None or not response.illusts:
            pass

        else:
            raw_results.extend(response.illusts)
            if response.next_url:
                next_page_result = await self._pixiv_get_next_page(
                    next_url=response.next_url,
                    method=self.PIXIV.illust_follow,
                    logger_prefix="[PIXIV] 获取关注列表新画廊失败: "
                )

                raw_results.extend(next_page_result)

        return await self._parse_raw_to_result(raw_results)

    async def pixiv_get_recommend(self):
        """获取为你推荐"""
        raw_results = []
        await self._ensure_api()

        try:
            response = await self.PIXIV.illust_recommended()

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.illust_recommended()

        except Exception as E:
            _log.error(f"[PIXIV] 获取为推荐失败: {E}")
            _log.debug(traceback.format_exc())
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )
        if not response.illusts:
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )

        raw_results.extend(response.illusts)
        # 获取后续页面
        if response.next_url:
            next_pages_result = await self._pixiv_get_next_page(
                response.next_url,
                self.PIXIV.illust_recommended,
                logger_prefix="[PIXIV] 获取为推荐失败: "
            )
            raw_results.extend(next_pages_result)

        return await self._parse_raw_to_result(raw_results)

    async def pixiv_get_favourite(self):
        """获取新收藏的画廊"""
        raw_results = []
        await self._ensure_api()

        # 先爬公开的
        try:
            response = await self.PIXIV.user_bookmarks_illust(user_id=int(self._user_id))

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.user_bookmarks_illust(user_id=int(self._user_id))

        except Exception as E:
            _log.error(f"[PIXIV] 获取收藏新画廊失败: {E}")
            _log.debug(traceback.format_exc())
            response = None

        if response is None:
            pass

        elif not response.illusts:
            _log.warning(f"[PIXIV] 收藏为空，是否还未任何收藏？")

        else:
            raw_results.extend(response.illusts)
            if response.next_url:
                next_page_result = await self._pixiv_get_next_page(
                    next_url=response.next_url,
                    method=self.PIXIV.user_bookmarks_illust,
                    logger_prefix="[PIXIV] 获取收藏新画廊失败: "
                )
                raw_results.extend(next_page_result)

        # 再爬私人的
        try:
            response = await self.PIXIV.user_bookmarks_illust(user_id=int(self._user_id), restrict="private")

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.user_bookmarks_illust(user_id=int(self._user_id), restrict="private")

        except Exception as E:
            _log.error(f"[PIXIV] 获取收藏新画廊失败: {E}")
            _log.debug(traceback.format_exc())
            response = None

        if response is None or not response.illusts:
            pass

        else:
            raw_results.extend(response.illusts)
            if response.next_url:
                next_page_result = await self._pixiv_get_next_page(
                    next_url=response.next_url,
                    method=self.PIXIV.user_bookmarks_illust,
                    logger_prefix="[PIXIV] 获取收藏新画廊失败: "
                )

                raw_results.extend(next_page_result)

        return await self._parse_raw_to_result(raw_results, 0)

    async def pixiv_get_new_illusts(self):
        """自动调用 `排行榜/关注列表/推荐列表/收藏列表` 获取新图"""
        await self._ensure_api()
        tasks: list[asyncio.Task] = [
            asyncio.create_task(self.pixiv_get_ranking()),
            asyncio.create_task(self.pixiv_get_follow_illusts()),
            asyncio.create_task(self.pixiv_get_recommend()),
            asyncio.create_task(self.pixiv_get_favourite())
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _pixiv_auto_fetch_new(self):
        """维护一个持久化任务，周期自动更新"""
        period = SETTING_CFG.SETU.AutoFetchPeriodAutoFetchPeriod  # 默认 1 天
        while True:
            await asyncio.sleep(period)
            async with self._fetch_lock:
                await self.pixiv_get_new_illusts()

    async def pixiv_get_illust_detail(self, illust_id: str) -> Illustration | None:
        """获取画廊详情"""
        await self._ensure_api()

        try:
            response = await self.PIXIV.illust_detail(int(illust_id))

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.illust_detail(int(illust_id))

        except Exception as E:
            _log.error(f"[PIXIV] 获取画廊 [{illust_id}] 失败: {E}")
            _log.debug(traceback.format_exc())
            return None

        if not response.illust:
            _log.warning(f"[PIXIV] 获取画廊 [{illust_id}] 失败")
            return None

        illustration = self._parse_illustration(response.illust)
        if illustration is None:
            _log.warning(f"[PIXIV] 获取画廊 [{illust_id}] 失败")
            return None

        return illustration

    async def pixiv_get_related(
            self,
            illust_id: str,
            new_illusts_expect: int = 5,
            include_origin: bool = False
    ) -> _PixivResults:
        """获取指定画廊的相关画廊，可用做搜索算法"""
        raw_results = []
        new_count = 0
        await self._ensure_api()

        # 先获取自己
        try:
            origin_response = await self.PIXIV.illust_detail(int(illust_id))

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            origin_response = await self.PIXIV.illust_detail(int(illust_id))

        except Exception as E:
            _log.error(f"[PIXIV] 获取相关画廊失败: {E}")
            _log.debug(traceback.format_exc())
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )

        if not origin_response.illust:
            _log.warning(f"[PIXIV] 画廊[{illust_id}]不存在，无法获取相关画廊")
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )

        if include_origin:
            raw_results.append(origin_response.illust)

        # 再获取相关
        try:
            response = await self.PIXIV.illust_related(int(illust_id))

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.illust_related(int(illust_id))

        except Exception as E:
            _log.error(f"[PIXIV] 获取相关画廊失败: {E}")
            _log.debug(traceback.format_exc())
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )

        if not response.illusts:
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )

        raw_results.extend(response.illusts)

        # 计数
        for raw_data in response.illusts:
            illustration = self._parse_illustration(raw_data)
            if illustration is None:
                continue

            if self._illustration_filter(illustration) and self._is_new_illust(illustration.iid):
                new_count += 1

        # 爬取直到满足上限或指定新图数量
        if response.next_url and new_illusts_expect > new_count:
            pages_limit = SETTING_CFG.SETU.CrawlForSearchPagesLimit  # 默认 10 页
            next_page_result = await self._pixiv_get_next_page(
                next_url=response.next_url,
                method=self.PIXIV.illust_related,
                pages=pages_limit,
                expect_new_count=new_illusts_expect - new_count,
                logger_prefix="[PIXIV] 获取相关画廊失败: "
            )
            raw_results.extend(next_page_result)

        return await self._parse_raw_to_result(raw_results)

    async def pixiv_get_artist_work(
            self,
            artist_id: int,
            new_illusts_expect: int = 5
    ):
        """获取指定画师作品"""
        raw_results = []
        new_count = 0
        await self._ensure_api()

        try:
            response = await self.PIXIV.user_illusts(artist_id)

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.user_illusts(artist_id)

        except Exception as E:
            _log.error(f"[PIXIV] 获取指定画师作品失败: {E}")
            _log.debug(traceback.format_exc())
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )

        if not response.illusts:
            return _PixivResults(
                illustrations=[],
                illusts=[]
            )

        raw_results.extend(response.illusts)

        # 计数
        for raw_data in response.illusts:
            illustration = self._parse_illustration(raw_data)
            if illustration is None:
                continue

            if self._illustration_filter(illustration, min_bookmarks=_INTERNAL_PARAMS[
                "min_bookmark_threshold"]) and self._is_new_illust(illustration.iid):
                new_count += 1

        # 爬取直到满足上限或指定新图数量
        if response.next_url and new_illusts_expect > new_count:
            pages_limit = SETTING_CFG.SETU.CrawlForSearchPagesLimit  # 默认 10 页
            next_page_result = await self._pixiv_get_next_page(
                next_url=response.next_url,
                method=self.PIXIV.user_illusts,
                pages=pages_limit,
                expect_new_count=new_illusts_expect - new_count,
                logger_prefix="[PIXIV] 获取指定画师作品失败: "
            )
            raw_results.extend(next_page_result)

        return await self._parse_raw_to_result(raw_results)

    async def pixiv_update_trending_tags(self):
        await self._ensure_api()

        try:
            response = await self.PIXIV.trending_tags_illust()

        except PixivAccessTokenExpiredError:
            await self._ensure_auth(True)
            response = await self.PIXIV.trending_tags_illust()

        except Exception as E:
            _log.error(f"[PIXIV] 更新热门标签失败: {E}")
            _log.debug(traceback.format_exc())
            return

        if not response.trend_tags:
            return

        now = time.time()
        trending_tags = {}
        for tag_dict in response.trend_tags:
            trending_tags[tag_dict["tag"]] = now
            for tag in tag_dict["illust"]["tags"]:
                trending_tags[tag["name"]] = now

        self._tags_mapping.update_from_dict(trending_tags)

    def _get_search_range(self, available_dict: dict[str, Illust] = None) -> dict[int, set[str]]:
        search_range: dict[int, set[str]] = defaultdict(set)
        if isinstance(available_dict, dict):
            for k, v in available_dict.items():
                if not v.pages:
                    continue

                search_range[int(v.chunk)].add(k)

        else:
            for index, chunk in self._chunks.items():
                search_range[index].update(*chunk.keys())

        return search_range

    def _get_search_illustrations(
            self,
            search_range: dict[int, set[str]]
    ) -> tuple[dict[str, Illustration], dict[str, Illust]]:
        illustrations = {}
        illusts = {}
        for index, iid_set in search_range.items():
            chunk = self.get_or_load_chunk(index)
            for iid in iid_set:
                illustration, illust = chunk.get(iid)
                if illustration is not None:
                    illustrations[iid] = illustration
                    illusts[iid] = illust

        return illustrations, illusts

    @staticmethod
    def bookmark_score(bookmark: int) -> float:
        max_bookmark = _INTERNAL_PARAMS["bookmark_score"]["top"]["bookmark"]
        min_bookmark = _INTERNAL_PARAMS["bookmark_score"]["bottom"]["bookmark"]
        max_score = _INTERNAL_PARAMS["bookmark_score"]["top"]["score_percentage"]
        min_score = _INTERNAL_PARAMS["bookmark_score"]["bottom"]["score_percentage"]

        bookmark = min(max_bookmark, max(min_bookmark, bookmark))
        return ((bookmark - min_bookmark) / (max_bookmark - min_bookmark)) * (max_score - min_score) + min_score

    async def _tags_searching(
            self,
            query_tags: list[str],
            search_data: dict[str, Illustration],
            nsfw: bool = False,
            expect_count: int = 1,
            tags_weight: dict[str, float] = None,
            artist_weight: dict[str, float] = None,
            ban_tags: set[str] = None,
            ban_artists: set[int] = None,
            cal_cache: dict = None,
            group_used: dict[str, Illust] = None
    ) -> dict[str, dict]:
        """
        tag匹配算法
        :param query_tags: 要查询的tag
        :param search_data: 给定的数据库
        :param nsfw: 是否包含色图
        :param expect_count: 期望结果数量
        :param tags_weight: tags权重字典，[0, 1]
        :param artist_weight: 作者权重字典，[0, 1]
        :param ban_tags: tag 黑名单
        :param ban_artists: 画师黑名单
        :param cal_cache: 外部传入一个字典，用于获取最终排序缓存，节省计算量
        :return:
        """
        # 去重，复制，保证顺序
        check_repeat = []
        for tag in query_tags:
            if tag not in check_repeat:
                check_repeat.append(tag)

        query_tags = check_repeat

        if not search_data:
            return {}

        if tags_weight is None:
            tags_weight = {}

        if artist_weight is None:
            artist_weight = {}

        if ban_tags is None:
            ban_tags = set()

        if ban_artists is None:
            ban_artists = set()

        if group_used is None:
            group_used = set()

        else:
            group_used = set(group_used.keys())

        # 先排除 R-18 tag，这个 tag 用作判断是否 NSFW
        if "R-18" in query_tags:
            query_tags.remove("R-18")

        # AI 给我滚出去
        if "AI生成" in query_tags:
            query_tags.remove("AI生成")

        # 限制tag数
        max_cal_tags = _INTERNAL_PARAMS["max_cal_tags"]
        if len(query_tags) > max_cal_tags:
            query_tags = query_tags[:max_cal_tags]

        # 如果是色图，则把 R-18 添加到末尾
        if nsfw:
            query_tags.append("R-18")

        search_score: dict[str, dict] = {}
        bookmark_threshold = _INTERNAL_PARAMS["min_bookmark_threshold"]

        # 计算 rating
        tags_size = len(query_tags)

        # 构建排序奖励
        tags_sequence_reward_mapping: dict[str, float] = {}
        max_size = max(1, min(max_cal_tags, tags_size))
        reward_factor = _INTERNAL_PARAMS["tags_sequence_reward_factor"]
        for i, tag in enumerate(query_tags, start=0):
            multiply = max_size - min(max_size, i)
            reward_weight = multiply * reward_factor / max_size
            tags_sequence_reward_mapping[tag] = reward_weight

        # 构建多次命中奖励
        multiply_hit_factor = _INTERNAL_PARAMS["multiply_tags_hit_reward_factor"]
        max_multiply_hit_reward = multiply_hit_factor * max_cal_tags
        max_hit_reward = min(
            max_multiply_hit_reward,
            multiply_hit_factor * tags_size,
        ) / max_size + 1
        max_tags_score = (
            sum(tags_sequence_reward_mapping.values())
            * max_hit_reward
        )
        bookmark_weight = _INTERNAL_PARAMS["score_weight"]["bookmark"]
        tags_score_weight = _INTERNAL_PARAMS["score_weight"]["tags"]
        score_threshold = _INTERNAL_PARAMS["local_search_score_threshold"]

        for iid, illustration in search_data.items():
            # 过滤色图
            if group_used and iid in group_used:
                continue

            if illustration.bookmark < bookmark_threshold:
                continue

            tags_set = illustration.tags_set
            if any(tag in ban_tags for tag in tags_set):
                continue

            if illustration.artist_id in ban_artists:
                continue

            if not nsfw:
                if illustration.NSFW:
                    continue

            # 按顺序匹配tag
            score = 0
            multiply_hit_reward = 0
            for tag in query_tags:
                if tag in tags_set:
                    tag_score = tags_sequence_reward_mapping[tag] * (
                                tags_weight.get(tag, 0.5) + artist_weight.get(str(illustration.artist_id), 0.5)) / 2
                    multiply_hit_reward += multiply_hit_factor
                    score += tag_score

            multiply_hit_reward = min(max_multiply_hit_reward, multiply_hit_reward) / max_size + 1
            normalized_tags_score = (
                score * multiply_hit_reward / max_tags_score
                if max_tags_score > 0
                else 0.0
            )

            # noinspection PyTypeChecker
            final_score = (
                bookmark_weight * self.bookmark_score(illustration.bookmark)
                + tags_score_weight * normalized_tags_score
            )
            final_score = min(1.0, max(0.0, final_score))
            if final_score < score_threshold:
                continue

            search_score[iid] = {
                "score": final_score,
                "illustration": illustration
            }

        # 排序并验证结果可用
        final_illustrations = {}
        search_result_sorted = dict(
            sorted(search_score.items(), key=lambda x: (round(x[1]["score"], 4), x[1]["illustration"].add_time),
                   reverse=True))

        # 存储计算缓存
        if isinstance(cal_cache, dict):
            cal_cache.clear()
            cal_cache.update(search_result_sorted)

        for iid, content in search_result_sorted.items():
            illustration = content["illustration"]
            available = await self.check_illustration_visible(illustration)
            if not available:
                continue

            final_illustrations[iid] = content
            if len(final_illustrations) >= expect_count:
                break

        return final_illustrations

    async def _related_searching(
            self,
            query_tags: list[str],
            search_data: dict[str, dict],
            nsfw: bool = False,
            expect_count: int = 1,
            tags_weight: dict[str, float] = None,
            artist_weight: dict[str, float] = None,
            ban_tags: set[str] = None,
            ban_artists: set[int] = None,
            cal_cache: dict[str, dict] = None
    ) -> dict[str, dict]:
        """
        相关画廊搜索算法
        :param query_tags: 要查询的tag
        :param search_data: 给定的数据库
        :param nsfw: 是否包含色图
        :param expect_count: 期望结果数量
        :param tags_weight: tags权重字典，[0, 1]
        :param artist_weight: 作者权重字典，[0, 1]
        :param ban_tags: tag 黑名单
        :param ban_artists: 画师黑名单
        :param cal_cache: 外部传入一个字典，用于获取最终排序缓存，节省计算量
        :return:
        """
        def _get_next_base_illustration() -> Illustration | None:
            try:
                _iid, _content = next(iter(search_data.items()))

            except StopIteration:
                return None

            search_data.pop(_iid, None)
            return _content["illustration"]

        async def _related_search_worker():
            _base_illustration = _get_next_base_illustration()
            if _base_illustration is None:
                return

            _page_now = 0
            _page_limit = SETTING_CFG.SETU.RelatedSearchCrawlPages  # 默认 10 页
            _next_request_params = {"illust_id": _base_illustration.id}
            while len(search_score) < expect_count and _page_now < _page_limit:
                if not _next_request_params:
                    return

                try:
                    _response = await self.PIXIV.illust_related(**_next_request_params)

                except PixivAccessTokenExpiredError:
                    await self._ensure_auth(True)
                    _response = await self.PIXIV.illust_related(**_next_request_params)

                except Exception as E:
                    _log.error(f"[PIXIV] 获取相关画廊失败: {E}")
                    _log.debug(traceback.format_exc())
                    return

                if not _response.illusts:
                    return

                await asyncio.sleep(random.uniform(0, 3))
                _page_now += 1
                if _response.next_url:
                    _next_request_params = self.PIXIV.parse_qs(_response.next_url)

                else:
                    _next_request_params = None

                # 判断结果
                _new_illustrations: list[Illustration] = []
                for raw_data in _response.illusts:
                    _new_illustration = self._parse_illustration(raw_data)
                    if _new_illustration is None:
                        continue

                    if not (self._illustration_filter(_new_illustration, bookmark_threshold) and self._is_new_illust(
                            _new_illustration.iid)):
                        continue

                    _new_illustrations.append(_new_illustration)

                # 计算得分
                for _new_illustration in _new_illustrations:
                    # 过滤
                    if _new_illustration.bookmark < bookmark_threshold:
                        continue

                    _tags_set = _new_illustration.tags_set
                    if any(_tag in ban_tags for _tag in _tags_set):
                        continue

                    if _new_illustration.artist_id in ban_artists:
                        continue

                    if not nsfw:
                        if _new_illustration.NSFW:
                            continue

                    # tag 匹配
                    _score = 0
                    _multiply_hit_reward = 0
                    for _tag in query_tags:
                        if _tag in _tags_set:
                            _tag_score = tags_sequence_reward_mapping[_tag] * (
                                    tags_weight.get(_tag, 0.5) + artist_weight.get(str(_new_illustration.artist_id), 0.5)) / 2
                            _multiply_hit_reward += multiply_hit_factor
                            _score += _tag_score

                    _multiply_hit_reward = min(max_multiply_hit_reward, _multiply_hit_reward) / max_size + 1
                    _normalized_tags_score = (
                        _score * _multiply_hit_reward / max_tags_score
                        if max_tags_score > 0
                        else 0.0
                    )

                    # noinspection PyTypeChecker
                    _final_score = (
                            bookmark_weight * self.bookmark_score(_new_illustration.bookmark)
                            + tags_score_weight * _normalized_tags_score
                    )
                    _final_score = min(1.0, max(0.0, _final_score))
                    if _final_score < score_threshold:
                        continue

                    _iid = _new_illustration.iid
                    search_score[_iid] = {
                        "score": _final_score,
                        "illustration": _new_illustration
                    }

        async def _related_search_runtime():
            nonlocal workers_now
            workers_now += 1
            try:
                await _related_search_worker()

            except (Exception, asyncio.CancelledError) as E:
                _log.error(f"[PIXIV] 相关画廊搜索爬虫 Worker 故障: {E}")
                _log.error(traceback.format_exc())
                return

            finally:
                workers_now -= 1

        async def _related_search_tasks_group():
            async with asyncio.TaskGroup() as tg:
                # 创建初始任务
                for _ in range(max_workers):
                    tg.create_task(_related_search_runtime())

                await asyncio.sleep(0.1)  # 等待计数器
                while len(search_score) < expect_count and search_data:
                    if workers_now < max_workers:
                        tg.create_task(_related_search_runtime())

                    await asyncio.sleep(1)

        await self._ensure_api()
        max_workers = min(expect_count, _INTERNAL_PARAMS["max_searching_workers"])  # 确定一个并发数量
        workers_now = 0
        if not search_data:
            return {}

        # 去重，复制，保证顺序
        check_repeat = []
        for tag in query_tags:
            if tag not in check_repeat:
                check_repeat.append(tag)

        query_tags = check_repeat
        search_data = search_data.copy()

        if tags_weight is None:
            tags_weight = {}

        if artist_weight is None:
            artist_weight = {}

        if ban_tags is None:
            ban_tags = set()

        if ban_artists is None:
            ban_artists = set()

        # 先排除 R-18 tag，这个 tag 用作判断是否 NSFW
        if "R-18" in query_tags:
            query_tags.remove("R-18")

        # AI 给我滚出去
        if "AI生成" in query_tags:
                query_tags.remove("AI生成")

        # 限制tag数
        max_cal_tags = _INTERNAL_PARAMS["max_cal_tags"]
        if len(query_tags) > max_cal_tags:
            query_tags = query_tags[:max_cal_tags]

        # 如果是色图，则把 R-18 添加到末尾
        if nsfw:
            query_tags.append("R-18")

        search_score: dict[str, dict] = {}
        bookmark_threshold = _INTERNAL_PARAMS["min_bookmark_threshold"]

        # 计算 rating
        tags_size = len(query_tags)

        # 构建排序奖励
        tags_sequence_reward_mapping: dict[str, float] = {}
        max_size = max(1, min(max_cal_tags, tags_size))
        reward_factor = _INTERNAL_PARAMS["tags_sequence_reward_factor"]
        for i, tag in enumerate(query_tags, start=0):
            multiply = max_size - min(max_size, i)
            reward_weight = multiply * reward_factor / max_size
            tags_sequence_reward_mapping[tag] = reward_weight

        # 构建多次命中奖励
        multiply_hit_factor = _INTERNAL_PARAMS["multiply_tags_hit_reward_factor"]
        max_multiply_hit_reward = multiply_hit_factor * max_cal_tags
        max_hit_reward = min(
            max_multiply_hit_reward,
            multiply_hit_factor * tags_size,
        ) / max_size + 1
        max_tags_score = (
                sum(tags_sequence_reward_mapping.values())
                * max_hit_reward
        )
        bookmark_weight = _INTERNAL_PARAMS["score_weight"]["bookmark"]
        tags_score_weight = _INTERNAL_PARAMS["score_weight"]["tags"]
        score_threshold = _INTERNAL_PARAMS["related_search_score_threshold"]

        # 超级并发
        await _related_search_tasks_group()

        # 返回结果，不做数量校验，不够也没办法
        final_illustrations = {}
        search_result_sorted = dict(
            sorted(search_score.items(), key=lambda x: (round(x[1]["score"], 4), x[1]["illustration"].add_time),
                   reverse=True))

        # 存储计算缓存
        if isinstance(cal_cache, dict):
            cal_cache.clear()
            cal_cache.update(search_result_sorted)

        for iid, content in search_result_sorted.items():
            illustration = content["illustration"]
            available = await self.check_illustration_visible(illustration)
            if not available:
                continue

            final_illustrations[iid] = content
            if len(final_illustrations) >= expect_count:
                break

        return final_illustrations

    async def _online_searching(
            self,
            query_tags: list[str],
            nsfw: bool = False,
            expect_count: int = 1,
            tags_weight: dict[str, float] = None,
            artist_weight: dict[str, float] = None,
            ban_tags: set[str] = None,
            ban_artists: set[int] = None,
            cal_cache: dict[str, dict] = None
    ) -> dict[str, dict]:
        """
        执行针对指定 tag 的原生 Pixiv 搜索
        :param query_tags: 要查询的 tag
        :param nsfw: 是否包含色图
        :param expect_count: 期望结果数量
        :param tags_weight: tags权重字典，[0, 1]
        :param artist_weight: 作者权重字典，[0, 1]
        :param ban_tags: tag 黑名单
        :param ban_artists: 画师黑名单
        :param cal_cache: 外部传入一个字典，用于获取最终排序缓存，节省计算量
        :return:
        """
        def _get_next_search_tag() -> str | None:
            try:
                return search_combinations.pop()

            except IndexError:
                return None

        async def _online_search_worker():
            _search_tag = _get_next_search_tag()
            if _search_tag is None:
                return

            _page_now = 0
            _page_limit = SETTING_CFG.SETU.OnlineSearchCrawlPages  # 默认 5 页
            _next_request_params = {"word": _search_tag, "min_bookmarks": bookmark_threshold}
            while len(search_score) < expect_count and _page_now < _page_limit:
                if not _next_request_params:
                    return

                try:
                    _response = await self.PIXIV.search_illust(**_next_request_params)

                except PixivAccessTokenExpiredError:
                    await self._ensure_auth(True)
                    _response = await self.PIXIV.search_illust(**_next_request_params)

                except Exception as E:
                    _log.error(f"[PIXIV] 搜索画廊失败: {E}")
                    _log.debug(traceback.format_exc())
                    return

                if not _response.illusts:
                    return

                await asyncio.sleep(random.uniform(0, 3))
                _page_now += 1
                if _response.next_url:
                    _next_request_params = self.PIXIV.parse_qs(_response.next_url)
                    if "bookmark_num_min" in _next_request_params:
                        _next_request_params["min_bookmarks"] = _next_request_params.pop("bookmark_num_min")

                    if "bookmark_num_max" in _next_request_params:
                        _next_request_params["max_bookmarks"] = _next_request_params.pop("bookmark_num_max")

                else:
                    _next_request_params = None

                # 判断结果
                _new_illustrations: list[Illustration] = []
                for raw_data in _response.illusts:
                    _new_illustration = self._parse_illustration(raw_data)
                    if _new_illustration is None:
                        continue

                    if not (self._illustration_filter(_new_illustration, bookmark_threshold) and self._is_new_illust(
                            _new_illustration.iid)):
                        continue

                    _new_illustrations.append(_new_illustration)

                # 计算得分
                for _new_illustration in _new_illustrations:
                    # 过滤
                    if _new_illustration.bookmark < bookmark_threshold:
                        continue

                    _tags_set = _new_illustration.tags_set
                    if any(_tag in ban_tags for _tag in _tags_set):
                        continue

                    if _new_illustration.artist_id in ban_artists:
                        continue

                    if not nsfw:
                        if _new_illustration.NSFW:
                            continue

                    # tag 匹配
                    _score = 0
                    _multiply_hit_reward = 0
                    for _tag in query_tags:
                        if _tag in _tags_set:
                            _tag_score = tags_sequence_reward_mapping[_tag] * (
                                    tags_weight.get(_tag, 0.5) + artist_weight.get(str(_new_illustration.artist_id),
                                                                                   0.5)) / 2
                            _multiply_hit_reward += multiply_hit_factor
                            _score += _tag_score

                    _multiply_hit_reward = min(max_multiply_hit_reward, _multiply_hit_reward) / max_size + 1
                    _normalized_tags_score = (
                        _score * _multiply_hit_reward / max_tags_score
                        if max_tags_score > 0
                        else 0.0
                    )

                    # noinspection PyTypeChecker
                    _final_score = (
                            bookmark_weight * self.bookmark_score(_new_illustration.bookmark)
                            + tags_score_weight * _normalized_tags_score
                    )
                    _final_score = min(1.0, max(0.0, _final_score))
                    if _final_score < score_threshold:
                        continue

                    _iid = _new_illustration.iid
                    search_score[_iid] = {
                        "score": _final_score,
                        "illustration": _new_illustration
                    }

        async def _online_search_runtime():
            nonlocal workers_now
            workers_now += 1
            try:
                await _online_search_worker()

            except (Exception, asyncio.CancelledError) as E:
                _log.error(f"[PIXIV] 原生画廊搜索爬虫 Worker 故障: {E}")
                _log.error(traceback.format_exc())
                return

            finally:
                workers_now -= 1

        async def _online_search_tasks_group():
            async with asyncio.TaskGroup() as tg:
                # 创建初始任务
                for _ in range(max_workers):
                    tg.create_task(_online_search_runtime())

                await asyncio.sleep(0.1)  # 等待计数器
                while len(search_score) < expect_count and search_combinations:
                    if workers_now < max_workers:
                        tg.create_task(_online_search_runtime())

                    await asyncio.sleep(1)

        if not query_tags:
            return {}

        await self._ensure_api()
        max_workers = _INTERNAL_PARAMS["max_searching_workers"] # 确定一个并发数量
        workers_now = 0

        # 去重，复制
        check_repeat = []
        for tag in query_tags:
            if tag not in check_repeat:
                check_repeat.append(tag)
        query_tags = check_repeat

        if tags_weight is None:
            tags_weight = {}

        if artist_weight is None:
            artist_weight = {}

        if ban_tags is None:
            ban_tags = set()

        if ban_artists is None:
            ban_artists = set()

        # 先排除 R-18 tag，这个 tag 用作判断是否 NSFW
        if "R-18" in query_tags:
            query_tags.remove("R-18")

        # AI 给我滚出去
        if "AI生成" in query_tags:
            query_tags.remove("AI生成")

        # 限制tag数
        max_cal_tags = _INTERNAL_PARAMS["max_cal_tags"]
        if len(query_tags) > max_cal_tags:
            query_tags = query_tags[:max_cal_tags]

        # 如果是色图，则把 R-18 添加到末尾
        if nsfw:
            query_tags.append("R-18")

        # 创建组合
        search_combinations: list[str] = []
        for length in range(1, len(query_tags) + 1):
            for combo in combinations(query_tags, length):
                search_combinations.append(' '.join(combo))

        search_combinations.sort(key=lambda x: len(x.split(' ')), reverse=False)  # 优先搜索精确(tag 多的)
        search_score: dict[str, dict] = {}
        bookmark_threshold = _INTERNAL_PARAMS["min_bookmark_threshold"]

        # 计算 rating
        tags_size = len(query_tags)

        # 构建排序奖励
        tags_sequence_reward_mapping: dict[str, float] = {}
        max_size = max(1, min(max_cal_tags, tags_size))
        reward_factor = _INTERNAL_PARAMS["tags_sequence_reward_factor"]
        for i, tag in enumerate(query_tags, start=0):
            multiply = max_size - min(max_size, i)
            reward_weight = multiply * reward_factor / max_size
            tags_sequence_reward_mapping[tag] = reward_weight

        # 构建多次命中奖励
        multiply_hit_factor = _INTERNAL_PARAMS["multiply_tags_hit_reward_factor"]
        max_multiply_hit_reward = multiply_hit_factor * max_cal_tags
        max_hit_reward = min(
            max_multiply_hit_reward,
            multiply_hit_factor * tags_size,
        ) / max_size + 1
        max_tags_score = (
                sum(tags_sequence_reward_mapping.values())
                * max_hit_reward
        )
        bookmark_weight = _INTERNAL_PARAMS["score_weight"]["bookmark"]
        tags_score_weight = _INTERNAL_PARAMS["score_weight"]["tags"]
        score_threshold = _INTERNAL_PARAMS["online_search_score_threshold"]

        # 超级并发
        await _online_search_tasks_group()

        # 返回结果，不做数量校验，不够也没办法
        final_illustrations = {}
        search_result_sorted = dict(
            sorted(search_score.items(), key=lambda x: (round(x[1]["score"], 4), x[1]["illustration"].add_time),
                   reverse=True))

        # 存储计算缓存
        if isinstance(cal_cache, dict):
            cal_cache.clear()
            cal_cache.update(search_result_sorted)

        for iid, content in search_result_sorted.items():
            illustration = content["illustration"]
            available = await self.check_illustration_visible(illustration)
            if not available:
                continue

            final_illustrations[iid] = content
            if len(final_illustrations) >= expect_count:
                break

        return final_illustrations

    async def pixiv_search(
            self,
            query_tags: list[str] = None,
            query_text: str = "",
            expect_count: int = 1,
            nsfw: bool = False,
            tags_weight: dict[str, float] = None,
            artist_weight: dict[str, float] = None,
            group_available: dict[str, Illust] = None,
            group_used: dict[str, Illust] = None,
            ban_tags: set[str] = None,
            ban_artists: set[int] = None,
            cal_cache: dict = None
    ) -> list[_SearchResult]:
        """
        搜索算法，通过时间与空间绕过P站会员
        :param query_tags: 要搜索的 tags
        :param query_text: 要搜索的自然语言
        :param expect_count: 期望返回的画廊数量
        :param nsfw: 是否包含色图
        :param tags_weight: tag 权重
        :param artist_weight: 画师 权重
        :param group_available: 群可用画廊
        :param group_used: 群已用画廊
        :param ban_tags: tag 黑名单
        :param ban_artists: 画师黑名单
        :param cal_cache: 计算缓存，用于加速
        :return:
        """
        def _commit_results(_new_results: dict):
            for k, v in _new_results.items():
                if k in final_results:
                    if final_results[k]["score"] >= v["score"]:
                        continue

                final_results[k] = v

        if not (query_tags or query_text):
            return []

        if expect_count <= 0:
            return []

        if query_tags is None:
            query_tags = []

        else:
            query_tags = query_tags.copy()

        if query_text:
            text2tags = await self.natural_language_to_tags(query_text)
            query_tags.extend(text2tags)

        # 可分为三种搜索: 本地 -> related -> online
        if cal_cache is None:
            cal_cache = {}

        local_cal_cache = cal_cache.copy()
        final_results = {}

        # 先搜索本地，这个最快最经济
        local_results = await self.pixiv_search_local(
            query_tags,
            expect_count,
            nsfw,
            tags_weight,
            artist_weight,
            group_available,
            group_used,
            ban_tags,
            ban_artists,
            local_cal_cache
        )
        _commit_results({result["iid"]: result for result in local_results})

        if len(final_results) >= expect_count:
            cal_cache.update(local_cal_cache)
            return list(final_results.values())

        # 再进行相关性搜索，尝试绕过原生搜索这一坨
        if group_available is None:
            related_cal_cache = local_cal_cache.copy()

        else:
            related_cal_cache = cal_cache.copy()

        new_expect = max(1, expect_count - len(final_results))
        related_results = await self.pixiv_search_related(
            query_tags,
            new_expect,
            nsfw,
            tags_weight,
            artist_weight,
            ban_tags,
            ban_artists,
            related_cal_cache
        )
        _commit_results({result["iid"]: result for result in related_results})

        if len(final_results) >= expect_count:
            cal_cache.update(local_cal_cache)
            cal_cache.update(related_cal_cache)
            return list(final_results.values())

        # 还不够就尝试原生搜索
        online_cal_cache = cal_cache.copy()
        new_expect = max(1, expect_count - len(final_results))
        online_results = await self.pixiv_search_online(
            query_tags,
            new_expect,
            nsfw,
            tags_weight,
            artist_weight,
            ban_tags,
            ban_artists,
            online_cal_cache
        )
        _commit_results({result["iid"]: result for result in online_results})

        # 更新外部缓存
        cal_cache.update(local_cal_cache)
        cal_cache.update(related_cal_cache)
        cal_cache.update(online_cal_cache)

        # 直接返回，不够也没辙
        return list(final_results.values())

    async def pixiv_search_local(
            self,
            query_tags: list[str],
            expect_count: int = 1,
            nsfw: bool = False,
            tags_weight: dict[str, float] = None,
            artist_weight: dict[str, float] = None,
            group_available: dict[str, Illust] = None,
            group_used: dict[str, Illust] = None,
            ban_tags: set[str] = None,
            ban_artists: set[int] = None,
            cal_cache: dict = None
    ) -> list[_SearchResult]:
        """
        通过 tag 匹配算法，搜索本地合适的画廊
        :param query_tags: 要搜索的 tags
        :param expect_count: 期望返回的画廊数量
        :param nsfw: 是否包含色图
        :param tags_weight: tag 权重
        :param artist_weight: 画师 权重
        :param group_available: 群可用画廊
        :param group_used: 群已用画廊
        :param ban_tags: tag 黑名单
        :param ban_artists: 画师黑名单
        :param cal_cache: 计算缓存，用于加速
        :return:
        """
        if ban_tags is None:
            ban_tags = set()

        if ban_artists is None:
            ban_artists = set()

        # 分成两条路径，一条有加速，一条从0构建
        final_illustrations = {}
        if cal_cache:
            # 有缓存，直接过滤结果就行
            illustrations = []
            if group_available is not None:
                available_illust = set(group_available.keys())

            else:
                available_illust = set()

            for iid, content in cal_cache.items():
                illustration = content["illustration"]
                # 额外走一次过滤器
                if available_illust and iid not in available_illust:
                    continue

                if any(tag in ban_tags for tag in illustration.tags_set):
                    continue

                if illustration.artist_id in ban_artists:
                    continue

                available = await self.check_illustration_visible(illustration)
                if not available:
                    continue

                final_illustrations[iid] = content
                illustrations.append(illustration)
                if len(final_illustrations) >= expect_count:
                    break

            illusts = await self.illustration_to_illust(*illustrations)

        else:
            # 过滤一次 tags
            query_tags = [tag for tag in query_tags if tag not in ban_tags]

            # 建立搜索范围
            search_range = self._get_search_range(group_available)

            # 收集数据
            search_data, illusts = self._get_search_illustrations(search_range)

            # 运行搜索
            final_illustrations = await self._tags_searching(
                query_tags,
                search_data,
                nsfw,
                expect_count,
                tags_weight,
                artist_weight,
                ban_tags,
                ban_artists,
                cal_cache
            )

        # 如果群库没有，执行全库搜索
        if group_available and len(final_illustrations) < expect_count:
            # 建立搜索范围
            search_range = self._get_search_range()

            # 收集数据
            search_data, illusts = self._get_search_illustrations(search_range)

            # 运行搜索
            supplemental_illustrations = await self._tags_searching(
                query_tags,
                search_data,
                nsfw,
                expect_count,
                tags_weight,
                artist_weight,
                ban_tags,
                ban_artists,
                cal_cache,
                group_used
            )
            final_illustrations.update(supplemental_illustrations)

        # 合成结果
        results = []
        for iid, content in final_illustrations.items():
            if iid not in illusts:
                continue

            results.append(_SearchResult(
                iid=iid,
                illustration=content["illustration"],
                illust=illusts[iid],
                score=content["score"]
            ))

        _log.info(f"[PIXIV] 本地搜索获取到 [{len(results)}] 张色图")
        return results

    async def pixiv_search_related(
            self,
            query_tags: list[str],
            expect_count: int = 1,
            nsfw: bool = False,
            tags_weight: dict[str, float] = None,
            artist_weight: dict[str, float] = None,
            ban_tags: set[str] = None,
            ban_artists: set[int] = None,
            cal_cache: dict = None
    ) -> list[_SearchResult]:
        """
        相关画廊方式搜索，获取新画廊
        :param query_tags: 要搜索的 tags
        :param expect_count: 期望返回的画廊数量
        :param nsfw: 是否包含色图
        :param tags_weight: tag 权重
        :param artist_weight: 画师 权重
        :param ban_tags: tag 黑名单
        :param ban_artists: 画师黑名单
        :param cal_cache: 计算缓存，用于加速
        :return:
        """
        if ban_tags is None:
            ban_tags = set()

        # 过滤一次 tags
        query_tags = [tag for tag in query_tags if tag not in ban_tags]

        # 分两条路径
        if cal_cache:
            search_data = cal_cache.copy()

        else:
            # 建立搜索范围
            search_range = self._get_search_range()

            # 收集数据
            search_data, illusts = self._get_search_illustrations(search_range)

            # 采用 5 倍基础画廊，写死
            search_data = await self._tags_searching(
                query_tags,
                search_data,
                nsfw,
                expect_count * 5,
                tags_weight,
                artist_weight,
                ban_tags,
                ban_artists,
                cal_cache
            )

        # 运行算法
        search_results = await self._related_searching(
            query_tags,
            search_data,
            nsfw,
            expect_count,
            tags_weight,
            artist_weight,
            ban_tags,
            ban_artists,
            cal_cache
        )

        # 合成结果并入库
        final_illustrations = {}
        illustrations = []
        for iid, content in search_results.items():
            illustration = content["illustration"]
            final_illustrations[iid] = content
            illustrations.append(illustration)

        illusts = await self.illustration_to_illust(*illustrations)

        results = []
        for iid, content in final_illustrations.items():
            if iid not in illusts:
                continue

            results.append(_SearchResult(
                iid=iid,
                illustration=content["illustration"],
                illust=illusts[iid],
                score=content["score"]
            ))

        _log.info(f"[PIXIV] 相关画廊获取到 [{len(results)}] 张色图")
        return results

    async def pixiv_search_online(
            self,
            query_tags: list[str],
            expect_count: int = 1,
            nsfw: bool = False,
            tags_weight: dict[str, float] = None,
            artist_weight: dict[str, float] = None,
            ban_tags: set[str] = None,
            ban_artists: set[int] = None,
            cal_cache: dict = None
    ) -> list[_SearchResult]:
        """
        Pixiv原生搜索，没会员效果一言难尽
        相关画廊方式搜索，获取新画廊
        :param query_tags: 要搜索的 tags
        :param expect_count: 期望返回的画廊数量
        :param nsfw: 是否包含色图
        :param tags_weight: tag 权重
        :param artist_weight: 画师 权重
        :param ban_tags: tag 黑名单
        :param ban_artists: 画师黑名单
        :param cal_cache: 用一个外部字典获取计算缓存
        :return:
        """
        if ban_tags is None:
            ban_tags = set()

        # 过滤一次 tags
        query_tags = [tag for tag in query_tags if tag not in ban_tags]

        # 运行算法
        search_results = await self._online_searching(
            query_tags,
            nsfw,
            expect_count,
            tags_weight,
            artist_weight,
            ban_tags,
            ban_artists,
            cal_cache
        )

        # 合成结果并入库
        final_illustrations = {}
        illustrations = []
        for iid, content in search_results.items():
            illustration = content["illustration"]
            final_illustrations[iid] = content
            illustrations.append(illustration)

        illusts = await self.illustration_to_illust(*illustrations)

        results = []
        for iid, content in final_illustrations.items():
            if iid not in illusts:
                continue

            results.append(_SearchResult(
                iid=iid,
                illustration=content["illustration"],
                illust=illusts[iid],
                score=content["score"]
            ))

        _log.info(f"[PIXIV] 原生搜索获取到 [{len(results)}] 张色图")
        return results

    async def natural_language_to_tags(
            self,
            prompt: str
    ) -> list[str]:
        def _extract_json(_text: str):
            _text = _text.strip()
            if _text.startswith("```"):
                lines = _text.splitlines()
                if lines and lines[0].strip().startswith("```"):
                    lines = lines[1:]

                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]

                _text = "\n".join(lines).strip()

            start = _text.find("[")
            end = _text.rfind("]")
            if 0 <= start < end:
                return _text[start:end + 1]

            return _text


        if not prompt:
            return []

        if not BASE_CFG.SETU.TagsLLM:
            return []

        try:
            setting = parse_llm_setting(BASE_CFG.SETU.TagsLLM)

        except Exception as E:
            _log.error(f"[PIXIV] 获取文本转 Tag LLM 失败: {E}")
            _log.debug(traceback.format_exc())
            return []

        if setting is None:
            return []

        if len(self._tags_mapping) < 100:
            await self.pixiv_update_trending_tags()

        tags_mapping_sorted = dict(sorted(self._tags_mapping.items(), key=lambda x: x[1], reverse=False))  # 最新靠后
        tags_list = list(tags_mapping_sorted.keys())
        if len(tags_list) > 5000:
            tags_list = tags_list[-5000:]

        # 构造prompt
        user_prompt = (f"# Tags列表\n\n```json\n{json.dumps(tags_list, ensure_ascii=False, indent=2)}\n```\n\n\n"
                       f"# 用户输入\n\n{prompt}")

        setting.keep_alive = False
        setting.system_prompt = await _TEXT2TAG_PROMPT.get_prompt()
        output = []
        async with Chat(config=setting) as chat:
            for _ in range(2):  # 只重试一次，写死
                response = await chat.ask(user_prompt)

                if not response:
                    continue

                if not isinstance(response[0], TextPrompt):
                    continue

                text = response[0].text
                json_text = _extract_json(text)
                try:
                    content = json.loads(json_text)
                    if not isinstance(content, list):
                        continue

                    if any(not isinstance(tag, str) for tag in content):
                        continue

                    output = content
                    break

                except json.JSONDecodeError:
                    continue

        return output

    async def random_local_setu(
            self,
            expect_count: int = 1,
            nsfw: bool = False,
            group_available: dict[str, Illust] = None
    ) -> _PixivResults:
        """返回本地随机色图"""
        expect_count = min(expect_count, 10)
        retry = 0
        big_samples: dict[str, tuple[Illustration, Illust]] = {}
        choices = _PixivResults(
            illustrations=[],
            illusts=[]
        )
        if group_available:
            chunk_data_mapping: dict[int, set[str]] = defaultdict(set)
            ids_mapping = self._ids_mapping.mapping()
            for iid in group_available:
                if iid not in ids_mapping:
                    continue

                chunk_data_mapping[ids_mapping[iid]].add(iid)

            for index, data_range in chunk_data_mapping.items():
                chunk = self.get_or_load_chunk(index)
                for iid in data_range:
                    illustration, illust = chunk.get(iid)
                    if not nsfw:
                        if illustration.NSFW:
                            continue

                    if illustration and illust:
                        big_samples[iid] = (illustration, illust)

            while big_samples and len(choices["illustrations"]) < expect_count and retry < 10:
                retry += 1
                if not big_samples:
                    break

                selected = random.sample(list(big_samples.keys()), min(len(big_samples), expect_count))
                for key in selected:
                    illustration, illust = big_samples.pop(key)
                    if illustration is None:
                        continue

                    available = await self.check_illustration_visible(illustration)
                    if available:
                        illust.pages = group_available[key].pages.copy()
                        choices["illustrations"].append(illustration)
                        choices["illusts"].append(illust)

        else:
            if not self._chunks:
                chunk = self.get_or_load_chunk()
                big_samples.update({iid: chunk.get(iid) for iid in chunk.data.keys()})

            else:
                for i, chunk in enumerate(reversed(self._chunks.values())):
                    if i >= 5:
                        break

                    big_samples.update({iid: chunk.get(iid) for iid in chunk.data.keys()})

            while big_samples and len(choices["illustrations"]) < expect_count and retry < 10:
                retry += 1
                if not big_samples:
                    break

                selected = random.sample(list(big_samples.keys()), min(len(big_samples), expect_count))
                for key in selected:
                    illustration, illust = big_samples.pop(key)
                    if illustration is None:
                        continue

                    if not nsfw:
                        if illustration.NSFW:
                            continue

                    available = await self.check_illustration_visible(illustration)
                    if available:
                        choices["illustrations"].append(illustration)
                        choices["illusts"].append(illust)

        return choices

    # =========================================================
    # 给群适配器用的接口，非直接公开接口
    # =========================================================

    async def transfer_new_illusts(self, chunk_index: int, group_chunks: dict[str, set[str]]) -> dict[str, Illust]:
        """给群适配器传递新的画廊"""
        # 决定要打开几个 chunk
        transfer_chunks_limit = SETTING_CFG.SETU.TransferChunksToGroup  # 默认 3 个

        # 加锁，以免重复请求更新
        async with self._fetch_lock:
            index_list = [
                index
                for index in range(chunk_index, chunk_index + transfer_chunks_limit)
                if (PIXIV_CHUNKS_DIR / f"chunk_{index}.json").is_file()
            ]

            if not index_list:
                index_list.append(self._index_now)

            new_illusts: dict[str, Illust] = {}
            for index in index_list:
                chunk = self.get_or_load_chunk(index)
                iids = [iid for iid in chunk.data if iid not in group_chunks.get(str(index), set())]
                for iid in iids:
                    _, illust = chunk.get(iid)
                    if illust is None:
                        continue

                    new_illusts[iid] = illust

            if len(new_illusts) < 10:
                chunk = self.get_or_load_chunk()
                new_chunks = [chunk]
                await self.pixiv_get_new_illusts()
                new_chunk = self.get_or_load_chunk()
                if new_chunk is not chunk:
                    new_chunks.append(new_chunk)

                for c in new_chunks:
                    iids = [iid for iid in c.data if
                            iid not in group_chunks.get(str(c.index), set()) and iid not in new_illusts]
                    for iid in iids:
                        _, illust = c.get(iid)
                        if illust is None:
                            continue

                        new_illusts[iid] = illust

        return new_illusts

    async def transfer_artist_illusts(self, artist_id: int, group_chunks: dict[str, set[str]]) -> dict[str, Illust]:
        """在大库找找指定画师的作品"""
        # 决定要打开几个 chunk
        transfer_chunks_limit = SETTING_CFG.SETU.TransferChunksToGroup  # 默认 3 个
        async with self._fetch_lock:
            await self.pixiv_get_artist_work(artist_id)
            if len(self._chunks) <= transfer_chunks_limit:
                need_open_chunk = []
                index_now = self._get_lastest_index()
                while len(need_open_chunk) < transfer_chunks_limit and index_now > 0:
                    if index_now not in self._chunks:
                        need_open_chunk.append(index_now)

                    index_now -= 1

                for index in need_open_chunk:
                    self.get_or_load_chunk(index)

            # 遍历搜索
            artist_illusts: dict[str, Illust] = {}
            for chunk in reversed(self._chunks.values()):
                exists = group_chunks.get(str(chunk.index), set())
                for iid, illustration in chunk.items():
                    if illustration.artist_id != artist_id:
                        continue

                    if iid in exists:
                        continue

                    _, illust = chunk.get(iid)
                    if illust is None:
                        continue

                    artist_illusts[iid] = illust

            return artist_illusts

    async def transfer_illust_illusts(
            self,
            illust_id: str,
            count: int,
            include_origin: bool,
            group_chunks: dict[str, set[str]]
    ) -> dict[str, Illust]:
        """获取相似画廊"""
        results: dict[str, Illust] = {}
        if include_origin:
            if illust_id in self._ids_mapping:
                index = self._ids_mapping[illust_id]
                chunk = self.get_or_load_chunk(index)
                _, illust = chunk.get(illust_id)
                if illust is not None:
                    results[illust_id] = illust

            if not results:
                illustration  = await self.pixiv_get_illust_detail(illust_id)
                if illustration is not None:
                    added = await self.add_new_illust(illustration, force=True)
                    if added:
                        illust = added[0][1]
                        results[illust_id] = illust

        if len(results) >= count:
            return results

        # 获取相似的
        related_results = await self.pixiv_get_related(
            illust_id=illust_id,
            new_illusts_expect=count - len(results),
            include_origin=False
        )
        results.update({illust.iid: illust for illust in related_results["illusts"] if
                        illust.iid not in group_chunks.get(str(illust.chunk), set())})
        return results


class Setu:
    """
    维护一个群色图对象，作为群色图的适配器
    """
    def __init__(self, group_id: str, nsfw: bool = False):
        # 基本信息
        self._group_id = group_id
        self.nsfw = nsfw
        self._max_loaded_chunks = SETTING_CFG.SETU.MaxGroupLoadedChunks  # 默认 20 个
        self._history_limit = SETTING_CFG.SETU.ResultHistoryLimit  # 默认 1000 条
        self._save_path = GROUPS_DIR / self._group_id / "setu.json"

        # 标识符
        self._chunk_index_now = 1

        # 数据库，会排除老色图
        self._chunks: dict[str, set[str]] = {}  # 一个数据总表，用于判断色图库获取到哪了，不会有多大空间占用
        self._available_illusts: dict[str, Illust] = {}  # 改为统一格式，好管理
        self._used_illusts: dict[str, Illust] = {}
        self._freeze: dict[str, PixivFreeze] = {}  # freeze_id: freeze
        self._history: dict[str, Illust] = {}
        self._unfreeze_task: asyncio.Task | None = asyncio.create_task(self.unfreeze())

        # 锁
        self._fetch_lock = asyncio.Lock()
        self._query_lock = asyncio.Lock()
        self._data_lock = asyncio.Lock()

        # 权重和黑名单
        self._tags_weight: dict[str, float] = {}
        self._artists_weight: dict[str, float] = {}
        self._ban_tags: set[str] = set()
        self._ban_artists: set[int] = set()
        self._ban_illusts: set[str] = set()
        self._ban_pages: dict[str, list[int]] = {}
        CONFIG_OBSERVER.register(self)

        if self._save_path.is_file():
            try:
                saved_data = json.loads(self._save_path.read_text(encoding="utf-8"))

            except (OSError, json.JSONDecodeError) as exc:
                _log.error(f"{self.logger_prefix()}色图库状态文件加载失败: {exc}")

            else:
                self.load_dict(saved_data)


    def update_config(self):
        self._max_loaded_chunks = SETTING_CFG.SETU.MaxGroupLoadedChunks
        self._history_limit = SETTING_CFG.SETU.ResultHistoryLimit

    def check_chunk_index_now(self):
        """找到一个还有空间的 chunk"""
        chunk_sorted = dict(sorted(self._chunks.items(), key=lambda x: int(x[0]), reverse=False))
        if not chunk_sorted:
            return

        index_now = self._chunk_index_now
        for index, content in chunk_sorted.items():
            if len(content) > 500:
                continue

            index_now = int(index)
            break

        if index_now != self._chunk_index_now:
            # 验证下一个 index，防止因为某些原因导致有一个 chunk 一直填不满
            next_index = index_now + 1
            if str(next_index) in chunk_sorted:
                if len(chunk_sorted[str(next_index)]) > 300:
                    index_now = next_index

        else:
            index_now = int(next(reversed(chunk_sorted.keys()))) + 1

        self._chunk_index_now = index_now

    def logger_prefix(self) -> str:
        return f"[PIXIV] 群[{self._group_id}] "

    # =========================================================
    # 新增画廊
    # =========================================================

    async def get_new_illusts(self):
        """从数据库获取新画廊"""
        SETU = AsyncPixivApi.get_api()
        new_illusts = await SETU.transfer_new_illusts(self._chunk_index_now, self._chunks.copy())
        await self._commit_new_illusts(new_illusts)
        _log.info(f"{self.logger_prefix()}已新增 [{len(new_illusts)}] 张色图")

    async def _check_remains(self):
        async with self._fetch_lock:
            if len(self._available_illusts) < 100:
                await self.get_new_illusts()

    def _clear_expire_chunk(self):
        if len(self._chunks) <= self._max_loaded_chunks or len(self._available_illusts) < 0.5 * 500 * self._max_loaded_chunks:
            return

        # 清理一个最老的chunk
        keys = sorted(self._chunks.keys(), key=lambda x: int(x))
        key = keys[0]
        pop_chunk = self._chunks.pop(key)
        for iid in pop_chunk:
            self._available_illusts.pop(iid, None)

    async def _commit_new_illusts(self, new_illusts: dict[str, Illust]) -> dict[str, Illust]:
        """入库新图，自动更新 index_now"""
        async with self._data_lock:
            self._clear_expire_chunk()
            added_illusts: dict[str, Illust] = {}
            for iid, illust in new_illusts.items():
                if iid in self._chunks.get(str(illust.chunk), set()):
                    continue

                # 过滤
                if iid in self._ban_illusts or illust.artist_id in self._ban_artists:
                    continue

                if any(tag in self._ban_tags for tag in illust.tags):
                    continue

                self._chunks.setdefault(str(illust.chunk), set()).add(iid)
                self._available_illusts[iid] = illust.copy()
                added_illusts[iid] = illust

            # 更新 index
            self.check_chunk_index_now()
            self.persist()
            return added_illusts

    # =========================================================
    # 获取、冻结、使用、还原
    # =========================================================

    async def _get_single_page(self, illust: Illust, check_available: bool = True) -> list[int]:
        select_page = []
        ban = self._ban_pages.get(illust.iid, [])
        available = []
        if check_available:
            available_illust = self._available_illusts.get(illust.iid, None)
            if available_illust is not None:
                available = available_illust.pages

        for page in illust.pages:
            if page in ban:
                continue

            if check_available:
                if page not in available:
                    continue

            select_page.append(page)
            break

        if not select_page and check_available:
            async with self._data_lock:
                available_illust = self._available_illusts.get(illust.iid, None)
                if available_illust is not None:
                    if not available_illust.pages:
                        self._available_illusts.pop(illust.iid, None)

        return select_page

    async def _get_random_setu(self, count: int) -> dict[str, Illust]:
        # 收集所有 illusts
        SETU = AsyncPixivApi.get_api()
        available_mirror: dict[str, Illust] = self._available_illusts.copy()

        # 算分，过滤
        scores = {}
        for k, v in available_mirror.items():
            # 检查是否还有效
            if not SETU.is_valid_illust(k):
                continue

            if v.NSFW:
                if not self.nsfw:
                    continue

            if k in self._ban_illusts:
                continue

            if v.artist_id in self._ban_artists:
                continue

            if any(tag in self._ban_tags for tag in v.tags):
                continue

            scores[k] = self.get_rating(v)

        scores_sorted = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        # 获取可用第一页
        results: dict[str, Illust] = {}
        for iid in scores_sorted:
            if len(results) >= count:
                break

            available_illust = available_mirror[iid]
            get_illust = available_illust.copy()
            select_pages = await self._get_single_page(get_illust)
            if not select_pages:
                continue

            get_illust.pages = select_pages
            page = str(select_pages[0])
            get_illust.attempts = {page: available_illust.attempts.get(page, 0)}
            results[iid] = get_illust

        # 如果结果不足，触发一次补库，但不做额外处理
        if len(results) < count:
            await self.get_new_illusts()

        return results

    async def _get_artist_setu(self, artist_id: int, count: int) -> dict[str, Illust]:
        SETU = AsyncPixivApi.get_api()
        results: dict[str, Illust] = {}

        # 先搜群库的
        available_mirror: dict[str, Illust] = self._available_illusts.copy()
        for iid, illust in available_mirror.items():
            if illust.artist_id != artist_id:
                continue

            if not SETU.is_valid_illust(iid):
                continue

            results[iid] = illust

        # 检查数量是否足够
        if len(results) < count:
            async with self._fetch_lock:
                new_illusts = await SETU.transfer_artist_illusts(artist_id, self._chunks.copy())
                added_illusts = await self._commit_new_illusts(new_illusts)
                results.update(added_illusts)

        # 算分，过滤
        scores: dict[str, float] = {}
        for iid, illust in results.items():
            if illust.NSFW:
                if not self.nsfw:
                    continue

            if iid in self._ban_illusts:
                continue

            if any(tag in self._ban_tags for tag in illust.tags):
                continue

            scores[iid] = self.get_rating(illust)

        scores_sorted = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        final_results: dict[str, Illust] = {}
        for iid in scores_sorted:
            if len(final_results) >= count:
                break

            illust = results[iid].copy()
            select_pages = await self._get_single_page(illust)
            if not select_pages:
                continue

            illust.pages = select_pages
            page = str(select_pages[0])
            illust.attempts = {page: illust.attempts.get(page, 0)}
            final_results[iid] = illust

        return final_results

    async def _get_illust_setu(self, illust_id: str, count: int, include_origin: bool) -> dict[str, Illust]:
        SETU = AsyncPixivApi.get_api()
        available_mirror: dict[str, Illust] = self._available_illusts.copy()
        results: dict[str, Illust] = {}
        final_results: dict[str, Illust] = {}

        # 这里是用来决定搜索链路要不要再搜一次原图
        search_origin = include_origin
        if include_origin:
            if illust_id in available_mirror:
                illust = available_mirror[illust_id]
                if illust.pages and SETU.is_valid_illust(illust_id):
                    results[illust_id] = available_mirror[illust_id].copy()
                    search_origin = False

        if len(results) >= count:
            for iid, illust in results.items():
                illust.pages = [illust.pages[0]]
                final_results[iid] = illust

            return final_results

        async with self._fetch_lock:
            related_results = await SETU.transfer_illust_illusts(
                illust_id=illust_id,
                count=count - len(results),
                include_origin=search_origin,
                group_chunks=self._chunks.copy()
            )
            added_illusts = await self._commit_new_illusts(related_results)
            results.update(added_illusts)
            if include_origin and illust_id in related_results:
                results[illust_id] = related_results[illust_id]

        # 算分，过滤
        scores: dict[str, float] = {}
        for iid, illust in results.items():
            if illust.NSFW:
                if not self.nsfw:
                    continue

            if iid in self._ban_illusts:
                continue

            if illust.artist_id in self._ban_artists:
                continue

            if any(tag in self._ban_tags for tag in illust.tags):
                continue

            scores[iid] = self.get_rating(illust)

        if include_origin:
            if illust_id in results:
                scores.pop(illust_id, None)
                origin = results[illust_id].copy()
                select_pages = await self._get_single_page(origin)
                if not select_pages:
                    select_pages = await self._get_single_page(origin, check_available=False)

                if select_pages:
                    origin.pages = select_pages
                    page = str(select_pages[0])
                    origin.attempts = {page: origin.attempts.get(page, 0)}
                    final_results[illust_id] = origin

        scores_sorted = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        for iid in scores_sorted:
            if len(final_results) >= count:
                break

            illust = results[iid].copy()
            select_pages = await self._get_single_page(illust)
            if not select_pages:
                continue

            illust.pages = select_pages
            page = str(select_pages[0])
            illust.attempts = {page: illust.attempts.get(page, 0)}
            final_results[iid] = illust

        return final_results

    async def _get_query_setu(self, query_tags: list[str], count: int) -> dict[str, Illust]:
        SETU = AsyncPixivApi.get_api()
        results: dict[str, Illust] = {}
        search_results = await SETU.pixiv_search(
            query_tags=query_tags,
            expect_count=count,
            nsfw=self.nsfw,
            tags_weight=self._tags_weight.copy(),
            artist_weight=self._artists_weight.copy(),
            group_available=self._available_illusts.copy(),
            group_used=self._used_illusts.copy(),
            ban_tags=self._ban_tags.copy(),
            ban_artists=self._ban_artists.copy(),
        )
        async with self._fetch_lock:
            search_dict: dict[str, Illust] = {result["iid"]: result["illust"] for result in search_results}
            await self._commit_new_illusts(search_dict)
            results.update(search_dict)

        scores: dict[str, float] = {}

        # 算分，过滤
        for iid, illust in results.items():
            if illust.NSFW:
                if not self.nsfw:
                    continue

            if iid in self._ban_illusts:
                continue

            if illust.artist_id in self._ban_artists:
                continue

            if any(tag in self._ban_tags for tag in illust.tags):
                continue

            scores[iid] = self.get_rating(illust)

        scores_sorted = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        final_results: dict[str, Illust] = {}
        for iid in scores_sorted:
            if len(final_results) >= count:
                break

            illust = results[iid].copy()
            select_pages = await self._get_single_page(illust)
            if not select_pages:
                continue

            illust.pages = select_pages
            page = str(select_pages[0])
            illust.attempts = {page: illust.attempts.get(page, 0)}
            final_results[iid] = illust

        return final_results

    # 超级大入口
    async def get_setu(
            self,
            expect_count: int = 1,
            query_text: str = "",
            illust_id: str = "",
            artist_id: str = "",
            include_related: bool = True,
            include_origin: bool = False
    ) -> PixivPayload | None:
        """
        适配器连接 Agent 的入口，会返回一个 Payload 对象。
        查询优先级按 illust_id -> artist_id -> query_text -> random
        :param expect_count: 要获取的色图数量
        :param query_text: 查询色图自然语言
        :param illust_id: 色图id
        :param artist_id: 画师id
        :param include_related: 若是 illust_id，是否返回相关图片
        :param include_origin: 若是 illust_id，是否返回这个id的自身
        :return:
        """
        await self._check_remains()

        if illust_id:
            if not illust_id.isdigit():
                raise ValueError(f"[PIXIV] 画廊id必须是数字: {illust_id}")

            if not include_related:
                expect_count = 1
                include_origin = True

            return await self.illust_setu(
                illust_id=illust_id,
                count=expect_count,include_origin=include_origin
            )

        if artist_id:
            if not artist_id.isdigit():
                raise ValueError(f"[PIXIV] 画师id必须是数字: {artist_id}")

            return await self.artist_setu(artist_id=int(artist_id), count=expect_count)

        if query_text:
            return await self.query_setu(query=query_text, count=expect_count)

        return await self.random_setu(expect_count)

    @staticmethod
    async def _pixiv_payload_maker(
            *args: Illust,
            expect_count: int,
            payload_type: PayloadType,
            freeze_id: str = "",
            query_text: str = "",
            query_tags: list[str] = None,
    ) -> PixivPayload | None:
        if query_tags is None:
            query_tags = []

        SETU = AsyncPixivApi.get_api()

        # 检查画廊存在
        exist_illustration: dict[str, Illustration] = SETU.illust_to_illustration(*args)
        exist_illusts = {illust.iid: illust.copy() for illust in args if illust.iid in exist_illustration}

        # 确保只有一页
        need_del = []
        for iid, illust in exist_illusts.items():
            if not illust.pages:
                need_del.append(iid)
                continue

            illust.pages = [illust.pages[0]]

        for iid in need_del:
            exist_illustration.pop(iid, None)
            exist_illusts.pop(iid, None)

        if not exist_illustration:
            _log.error(f"[PIXIV] 所有要获取的结果均已丢失，无法获取")
            return None

        # 尝试下载图片，会下载所有页面，尽力而为建立图片缓存
        tasks: dict[str, asyncio.Task] = {}
        for iid, illustration in exist_illustration.items():
            tasks[iid] = asyncio.create_task(SETU.get_image(illustration))

        exist_hashes: dict[str, str] = {}
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        for iid, result in tasks.items():
            if not result.done() or result.cancelled() or result.exception() is not None:
                continue

            hash_dict: dict[str, str] = result.result()
            if str(exist_illusts[iid].pages[0]) not in hash_dict:
                continue

            exist_hashes[iid] = hash_dict[str(exist_illusts[iid].pages[0])]

        # 组装payload
        payload = PixivPayload(
            count=expect_count,
            freeze_id=freeze_id,
            query_text=query_text,
            query_tags=query_tags.copy(),
            payload_type=payload_type,
            illusts=exist_illusts,
            illustrations={k: v.copy() for k, v in exist_illustration.items()},
            hash_names=exist_hashes
        )
        return payload

    async def random_setu(self, count: int = 1) -> PixivPayload | None:
        """并非随机，而是返回最高评分的色图"""
        multiplier = SETTING_CFG.SETU.SubstituteMultiplier  # 默认 4
        expect_count = count
        count = count * multiplier
        async with self._query_lock:
            results = await self._get_random_setu(count)

            # 冻结、返回
            if not results:
                return None

            freeze = await self.freeze(*results.values())
            if freeze is None:
                freeze_id = ""

            else:
                freeze_id = freeze.freeze_id

        return await self._pixiv_payload_maker(
            *results.values(),
            expect_count=expect_count,
            payload_type=PayloadType.RANDOM,
            freeze_id=freeze_id,
        )

    async def artist_setu(self, artist_id: int, count: int = 5) -> PixivPayload | None:
        """找指定画师的色图"""
        multiplier = SETTING_CFG.SETU.SubstituteMultiplier  # 默认 4
        expect_count = count
        count = count * multiplier
        if artist_id in self._ban_artists:
            _log.warning(f"{self.logger_prefix()}获取已被 ban 的画师")
            return None

        async with self._query_lock:
            results = await self._get_artist_setu(artist_id, count)
            lacking = count - len(results)
            if lacking > 0:
                standby = await self._get_random_setu(lacking)
                results.update(standby)

            # 冻结、返回
            if not results:
                return None

            freeze = await self.freeze(*results.values())
            if freeze is None:
                freeze_id = ""

            else:
                freeze_id = freeze.freeze_id

        return await self._pixiv_payload_maker(
            *results.values(),
            expect_count=expect_count,
            payload_type=PayloadType.ARTIST,
            freeze_id=freeze_id
        )

    async def illust_setu(self, illust_id: str, count: int = 5, include_origin: bool = False) -> PixivPayload | None:
        """获取指定ID的画廊，及其相关作品"""
        multiplier = SETTING_CFG.SETU.SubstituteMultiplier  # 默认 4
        expect_count = count
        count = count * multiplier
        if illust_id in self._ban_illusts:
            _log.warning(f"{self.logger_prefix()}获取已被 ban 的画廊")
            return None

        async with self._query_lock:
            results = await self._get_illust_setu(
                illust_id=illust_id,
                count=count,
                include_origin=include_origin
            )
            lacking = count - len(results)
            if lacking > 0:
                standby = await self._get_random_setu(lacking)
                results.update(standby)

            # 冻结、返回
            if not results:
                return None

            freeze = await self.freeze(*results.values())
            if freeze is None:
                freeze_id = ""

            else:
                freeze_id = freeze.freeze_id

        return await self._pixiv_payload_maker(
            *results.values(),
            expect_count=expect_count,
            payload_type=PayloadType.ILLUST,
            freeze_id=freeze_id
        )

    async def query_setu(self, query: str, count: int = 1) -> PixivPayload | None:
        """自然语言查询"""
        multiplier = SETTING_CFG.SETU.SubstituteMultiplier  # 默认 4
        expect_count = count
        count = count * multiplier
        SETU = AsyncPixivApi.get_api()
        tags = await SETU.natural_language_to_tags(query)
        for tag in tags.copy():
            if tag in self._ban_tags:
                tags.remove(tag)

        if not tags:
            _log.warning(f"{self.logger_prefix()}无有效 tags")
            return None

        # 先不上锁，因为搜索太久了
        results = await self._get_query_setu(tags, count)
        async with self._query_lock:
            lacking = count - len(results)
            if lacking > 0:
                standby = await self._get_random_setu(lacking)
                results.update(standby)

            # 冻结、返回
            if not results:
                return None

            freeze = await self.freeze(*results.values())
            if freeze is None:
                freeze_id = ""

            else:
                freeze_id = freeze.freeze_id

        return await self._pixiv_payload_maker(
            *results.values(),
            expect_count=expect_count,
            payload_type=PayloadType.QUERY,
            freeze_id=freeze_id,
            query_tags=tags,
            query_text=query
        )

    # =========================================================
    # 群色图库操作
    # =========================================================

    async def freeze(self, *args: Illust) -> PixivFreeze | None:
        freeze_payload = PixivFreeze()
        async with self._data_lock:
            for illust in args:
                iid = illust.iid
                if not illust.pages:
                    continue

                if iid not in self._available_illusts:
                    continue

                # 移除 available
                available_illust = self._available_illusts[iid]
                available_pages = available_illust.pages
                frozen_illust = illust.copy()
                frozen_illust.pages = list(dict.fromkeys(
                    page for page in frozen_illust.pages
                    if page in available_pages
                ))
                if not frozen_illust.pages:
                    continue

                freeze_payload.illusts[iid] = frozen_illust
                for page in frozen_illust.pages:
                    frozen_illust.attempts[str(page)] = frozen_illust.attempts.setdefault(str(page), 0) + 1  # 记录 attempt
                    available_pages.remove(page)

                if not available_pages:
                    self._available_illusts.pop(iid, None)

            if freeze_payload:
                self._freeze[freeze_payload.freeze_id] = freeze_payload
                if self._unfreeze_task is not None:
                    self._unfreeze_task.cancel()

                self._unfreeze_task = asyncio.create_task(self.unfreeze())
                self.persist()
                return freeze_payload

            else:
                return None

    async def unfreeze(self, freeze_id: str = None):
        """解冻超时未提交的色图，attempt 达到 5 的 page 不再回到 available"""
        timeout = SETTING_CFG.SETU.SendSETUTaskTimeout  # 默认 600 秒
        times_limit = SETTING_CFG.SETU.UnfreezeTimesLimit  # 默认 5 次
        if freeze_id is None:
            await asyncio.sleep(timeout)

        now = time.time()
        async with self._data_lock:
            if freeze_id is not None:
                freeze_ids = [freeze_id] if freeze_id in self._freeze else []

            else:
                freeze_ids = [
                    _freeze_id
                    for _freeze_id, freeze_payload in self._freeze.items()
                    if now - freeze_payload.freeze_time >= timeout
                ]

            if not freeze_ids:
                return

            for _freeze_id in freeze_ids:
                if _freeze_id not in self._freeze:
                    continue

                freeze_payload = self._freeze[_freeze_id]
                for iid, illust in freeze_payload.illusts.items():
                    attempts = illust.attempts
                    available_pages = [
                        page
                        for page in illust.pages
                        if attempts.get(str(page), 0) < times_limit
                    ]
                    if not available_pages:
                        continue

                    valid_attempts = {
                        page: count
                        for page, count in attempts.items()
                        if count < times_limit
                    }

                    if iid not in self._available_illusts:
                        back_illust = illust.copy()
                        back_illust.pages = available_pages
                        back_illust.attempts = valid_attempts
                        self._available_illusts[iid] = back_illust
                        continue

                    available_illust = self._available_illusts[iid]
                    for page in available_pages:
                        if page not in available_illust.pages:
                            available_illust.pages.append(page)

                    available_illust.attempts.update(valid_attempts)

                self._freeze.pop(_freeze_id, None)

            self.persist()

    def _trim_interaction_history_unlocked(self):
        limit = max(1, int(self._history_limit))
        while len(self._history) > limit:
            self._history.pop(next(iter(self._history)))

        used_trimmed = False
        while len(self._used_illusts) > limit:
            self._used_illusts.pop(next(iter(self._used_illusts)))
            used_trimmed = True

        if used_trimmed:
            self._refresh_interaction_weights_unlocked()

    async def mark_used(
            self,
            *,
            message_id: str | None,
            illust: Illust,
            freeze_id: str
    ):
        """原子提交一次成功发送，并从冻结区消费对应页面。"""
        async with self._data_lock:
            iid = illust.iid
            used_illust = self._used_illusts.pop(iid, None)
            if used_illust is None:
                used_illust = illust.copy()

            else:
                for page in illust.pages:
                    if page not in used_illust.pages:
                        used_illust.pages.append(page)

                used_illust.attempts.update(illust.attempts)

            self._used_illusts[iid] = used_illust
            if message_id:
                self._history[message_id] = illust.copy()

            available_illust = self._available_illusts.get(iid)
            if available_illust is not None:
                available_illust.pages = [
                    page for page in available_illust.pages
                    if page not in illust.pages
                ]
                if not available_illust.pages:
                    self._available_illusts.pop(iid, None)

            freeze_payload = self._freeze.get(freeze_id)
            if freeze_payload is not None:
                frozen_illust = freeze_payload.illusts.get(iid)
                if frozen_illust is not None:
                    frozen_illust.pages = [
                        page for page in frozen_illust.pages
                        if page not in illust.pages
                    ]
                    if not frozen_illust.pages:
                        freeze_payload.illusts.pop(iid, None)

                if not freeze_payload:
                    self._freeze.pop(freeze_id, None)

            self._trim_interaction_history_unlocked()
            self.persist()

    async def mark_ban_pages(self, illusts: dict[str, Illust]):
        """ban某页"""
        async with self._data_lock:
            for iid, illust in illusts.items():
                ban_pages = self._ban_pages.setdefault(iid, [])
                for page in illust.pages:
                    if page not in ban_pages:
                        ban_pages.append(page)

            self.persist()

    async def mark_ban_illusts(self, illusts: dict[str, Illust]):
        """ban某画廊"""
        async with self._data_lock:
            self._ban_illusts.update(
                illusts.keys()
            )
            self.persist()

    async def mark_ban_artist(self, illusts: dict[str, Illust]):
        """ban某画师"""
        async with self._data_lock:
            self._ban_artists.update(
                {illust.artist_id for illust in illusts.values()}
            )
            self.persist()

    async def mark_ban_tags(self, *args: str):
        """ban某tag"""
        async with self._data_lock:
            self._ban_tags.update(
                {tag for tag in args if tag}
            )
            self.persist()

    async def mark_unban(
            self,
            *,
            illust_id: str | None = None,
            page: int | None = None,
            artist_id: int | None = None,
            tags: tuple[str, ...] = (),
    ) -> None:
        """根据显式标识解除页面、画廊、画师或标签屏蔽。"""
        if page is not None and illust_id is None:
            raise ValueError("page requires illust_id")

        async with self._data_lock:
            if illust_id is not None:
                if page is None:
                    self._ban_illusts.discard(illust_id)

                else:
                    ban_pages = self._ban_pages.get(illust_id)
                    if ban_pages is not None:
                        self._ban_pages[illust_id] = [
                            banned_page for banned_page in ban_pages
                            if banned_page != page
                        ]
                        if not self._ban_pages[illust_id]:
                            self._ban_pages.pop(illust_id)

            if artist_id is not None:
                self._ban_artists.discard(artist_id)

            self._ban_tags.difference_update(tag for tag in tags if tag)
            self.persist()

    # =========================================================
    # 评分系统
    # =========================================================

    def get_rating(self, illust: Illust) -> float:
        """算分"""
        bookmark = illust.bookmark
        bookmark_score = AsyncPixivApi.bookmark_score(bookmark)
        effective_tags = [tag for tag in illust.tags if tag in self._tags_weight]
        if effective_tags:
            tags_score = sum(self._tags_weight.get(tag, 0.5) for tag in effective_tags) / len(effective_tags)

        else:
            tags_score = 0.5

        artist_score = self._artists_weight.get(str(illust.artist_id), 0.5)
        used_illust = self._used_illusts.get(illust.iid, None)
        used_count = len(used_illust.pages) if used_illust is not None else 0
        penalty = math.sqrt(used_count) * _INTERNAL_PARAMS["used_penalty_factor"]

        bookmark_weight = _INTERNAL_PARAMS["group_score_weight"]["bookmark"]
        artist_weight = _INTERNAL_PARAMS["group_score_weight"]["artist"]
        tags_weight = _INTERNAL_PARAMS["group_score_weight"]["tags"]
        # noinspection PyTypeChecker
        final_score = bookmark_score * bookmark_weight + artist_score * artist_weight + tags_score * tags_weight - penalty
        return min(1.0, max(0.0, final_score))

    # =========================================================
    # 结果互动/响应
    # =========================================================

    @staticmethod
    def _parse_interaction_score(text: str) -> Decimal | None:
        try:
            score = Decimal(text)

        except InvalidOperation:
            return None

        if score.is_nan():
            return None

        if score.is_infinite():
            score = Decimal("10") if score > 0 else Decimal("0")

        score = min(Decimal("10"), max(Decimal("0"), score))
        return score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _refresh_interaction_weights_unlocked(self):
        tag_scores: dict[str, list[float]] = defaultdict(list)
        artist_scores: dict[str, list[float]] = defaultdict(list)
        for illust in self._used_illusts.values():
            scores = [
                score
                for page_scores in illust.scores.values()
                for score in page_scores.values()
            ]
            if not scores:
                continue

            for tag in illust.tags:
                tag_scores[tag].extend(scores)

            artist_scores[str(illust.artist_id)].extend(scores)

        min_samples = _INTERNAL_PARAMS["rating_min_samples"]
        self._tags_weight = {
            tag: sum(scores) / len(scores)
            for tag, scores in tag_scores.items()
            if len(scores) >= min_samples
        }
        self._artists_weight = {
            artist: sum(scores) / len(scores)
            for artist, scores in artist_scores.items()
            if len(scores) >= min_samples
        }

    async def _mark_interaction_score(
            self,
            illust: Illust,
            user_id: str,
            score: Decimal
    ) -> str:
        async with self._data_lock:
            used_illust = self._used_illusts.get(illust.iid)
            if used_illust is None or not illust.pages:
                return "这张色图的互动记录已经过期了喵~"

            page = str(illust.pages[0])
            page_scores = used_illust.scores.setdefault(page, {})
            page_scores[user_id] = float(score / Decimal("10"))
            self._refresh_interaction_weights_unlocked()
            self.persist()

        if score <= 0:
            return "这张色图的评分是... 零蛋！！！"

        elif score >= 10:
            return "这张色图的评分是... 满分！！！"

        else:
            return f"这张色图的评分是... {score:.2f}！"

    @staticmethod
    def _deduplicate_targets(targets: tuple[Setu, ...]) -> tuple[Setu, ...]:
        unique_targets: dict[int, Setu] = {}
        for target in targets:
            unique_targets.setdefault(id(target), target)

        # noinspection PyTypeChecker
        return tuple(unique_targets.values())

    def _resolve_interaction_targets(
            self,
            context: SetuInteractionContext,
            command: CommandArgs
    ) -> tuple[tuple[Setu, ...], str | None]:
        if not command.global_scope:
            return (self,), None

        if not context.allow_global:
            return (), "权限不足，无法执行全局色图库操作"

        targets = self._deduplicate_targets((self, *context.global_targets))
        return targets, None

    @staticmethod
    async def _apply_to_targets(
            targets: tuple[Setu, ...],
            method_name: str,
            *args,
            **kwargs
    ) -> tuple[int, int]:
        results = await asyncio.gather(
            *(
                getattr(target, method_name)(*args, **kwargs)
                for target in targets
            ),
            return_exceptions=True
        )
        failed = sum(isinstance(result, BaseException) for result in results)
        return len(results) - failed, failed

    async def _run_illust_action(
            self,
            context: SetuInteractionContext,
            command: CommandArgs,
            method_name: str,
            action_text: str
    ) -> str:
        targets, error = self._resolve_interaction_targets(context, command)
        if error is not None:
            return error

        illusts = {context.illust.iid: context.illust}
        success, failed = await self._apply_to_targets(targets, method_name, illusts)
        if failed:
            return f"{action_text}：成功 [{success}] 个群，失败 [{failed}] 个群"

        return f"{action_text}：已作用于 [{success}] 个群"

    async def _run_tags_action(
            self,
            context: SetuInteractionContext,
            command: CommandArgs,
            method_name: str,
            action_text: str
    ) -> str:
        targets, error = self._resolve_interaction_targets(context, command)
        if error is not None:
            return error

        tags = tuple(tag.strip() for tag in command.tags if tag.strip())
        if not tags:
            return "至少需要提供一个 tag"

        success, failed = await self._apply_to_targets(targets, method_name, *tags)
        if failed:
            return f"{action_text} {list(tags)}：成功 [{success}] 个群，失败 [{failed}] 个群"

        return f"{action_text} {list(tags)}：已作用于 [{success}] 个群"

    async def _format_ban_state(self, category: str = "all") -> str:
        async with self._data_lock:
            ban_pages = copy.deepcopy(self._ban_pages)
            ban_illusts = self._ban_illusts.copy()
            ban_artists = self._ban_artists.copy()
            ban_tags = self._ban_tags.copy()

        sections = []
        if category in {"all", "page"}:
            pages = [
                f"{iid}: {sorted(page_list)}"
                for iid, page_list in sorted(ban_pages.items())
            ]
            sections.append("屏蔽页面：" + ("；".join(pages) if pages else "无"))

        if category in {"all", "id"}:
            ids = ", ".join(sorted(ban_illusts)) or "无"
            sections.append(f"屏蔽画廊：{ids}")

        if category in {"all", "artist"}:
            artists = ", ".join(str(item) for item in sorted(ban_artists)) or "无"
            sections.append(f"屏蔽画师：{artists}")

        if category in {"all", "tags"}:
            tags = ", ".join(sorted(ban_tags)) or "无"
            sections.append(f"屏蔽标签：{tags}")

        return '\n'.join(sections)

    @_SETU_INTERACTION_COMMANDS.command(name="info")
    async def _interaction_info(
            self,
            context: SetuInteractionContext,
            command: CommandArgs
    ) -> str:
        illust = context.illust
        rating = self.get_rating(illust) * 10
        return '\n'.join((
            illust.title,
            f"画师：{illust.artist_name} ({illust.artist_id})",
            f"标签：{sorted(illust.tags)}",
            f"群内评分：{rating:.2f}",
            f"作品链接：https://www.pixiv.net/artworks/{illust.iid}",
            f"作者链接：https://www.pixiv.net/users/{illust.artist_id}",
        ))

    @_SETU_INTERACTION_COMMANDS.command(name="check-ban")
    async def _interaction_check_ban(self, context: SetuInteractionContext, command: CommandArgs) -> str:
        return await self._format_ban_state()

    @_SETU_INTERACTION_COMMANDS.command(name="check-ban-page")
    async def _interaction_check_ban_page(self, context: SetuInteractionContext, command: CommandArgs) -> str:
        return await self._format_ban_state("page")

    @_SETU_INTERACTION_COMMANDS.command(name="check-ban-id")
    async def _interaction_check_ban_id(self, context: SetuInteractionContext, command: CommandArgs) -> str:
        return await self._format_ban_state("id")

    @_SETU_INTERACTION_COMMANDS.command(name="check-ban-artist")
    async def _interaction_check_ban_artist(self, context: SetuInteractionContext, command: CommandArgs) -> str:
        return await self._format_ban_state("artist")

    @_SETU_INTERACTION_COMMANDS.command(name="check-ban-tags")
    async def _interaction_check_ban_tags(self, context: SetuInteractionContext, command: CommandArgs) -> str:
        return await self._format_ban_state("tags")

    @_SETU_INTERACTION_COMMANDS.command(name="ban-page")
    @_SETU_INTERACTION_COMMANDS.option("--global", dest="global_scope", action="store_true", default=False)
    async def _interaction_ban_page(self, context: SetuInteractionContext, command: CommandArgs) -> str:
        return await self._run_illust_action(context, command, "mark_ban_pages", "已屏蔽当前页面")

    @_SETU_INTERACTION_COMMANDS.command(name="ban-id")
    @_SETU_INTERACTION_COMMANDS.option("--global", dest="global_scope", action="store_true", default=False)
    async def _interaction_ban_id(self, context: SetuInteractionContext, command: CommandArgs) -> str:
        return await self._run_illust_action(context, command, "mark_ban_illusts", "已屏蔽当前画廊")

    @_SETU_INTERACTION_COMMANDS.command(name="ban-artist")
    @_SETU_INTERACTION_COMMANDS.option("--global", dest="global_scope", action="store_true", default=False)
    async def _interaction_ban_artist(self, context: SetuInteractionContext, command: CommandArgs) -> str:
        return await self._run_illust_action(context, command, "mark_ban_artist", "已屏蔽当前画师")

    @_SETU_INTERACTION_COMMANDS.command(name="ban-tags")
    @_SETU_INTERACTION_COMMANDS.option("--global", dest="global_scope", action="store_true", default=False)
    @_SETU_INTERACTION_COMMANDS.argument("tags", nargs="+")
    async def _interaction_ban_tags(self, context: SetuInteractionContext, command: CommandArgs) -> str:
        return await self._run_tags_action(context, command, "mark_ban_tags", "已屏蔽标签")

    @_SETU_INTERACTION_COMMANDS.command(name="unban")
    @_SETU_INTERACTION_COMMANDS.option("--global", dest="global_scope", action="store_true", default=False)
    @_SETU_INTERACTION_COMMANDS.option("-id", dest="illust_id", _type=int, default=None)
    @_SETU_INTERACTION_COMMANDS.option("-artist", dest="artist_id", _type=int, default=None)
    @_SETU_INTERACTION_COMMANDS.option("-page", dest="page", _type=int, default=None)
    @_SETU_INTERACTION_COMMANDS.option("-tags", dest="raw_tags", default=None)
    async def _interaction_unban(
            self,
            context: SetuInteractionContext,
            command: CommandArgs
    ) -> str:
        if command.illust_id is not None and command.illust_id <= 0:
            return "-id 必须是正整数"

        if command.artist_id is not None and command.artist_id <= 0:
            return "-artist 必须是正整数"

        if command.page is not None and command.page < 0:
            return "-page 必须是非负整数"

        if command.page is not None and command.illust_id is None:
            return "使用 -page 时必须同时提供 -id"

        tags = ()
        if command.raw_tags is not None:
            tags = tuple(dict.fromkeys(
                tag.strip()
                for tag in command.raw_tags.replace("，", ",").split(",")
                if tag.strip()
            ))
            if not tags:
                return "-tags 至少需要提供一个 tag"

        if (
                command.illust_id is None
                and command.artist_id is None
                and not tags
        ):
            return "至少需要提供 -id、-artist 或 -tags 中的一项"

        targets, error = self._resolve_interaction_targets(context, command)
        if error is not None:
            return error

        illust_id = (
            str(command.illust_id)
            if command.illust_id is not None
            else None
        )
        success, failed = await self._apply_to_targets(
            targets,
            "mark_unban",
            illust_id=illust_id,
            page=command.page,
            artist_id=command.artist_id,
            tags=tags,
        )
        if failed:
            return f"解除屏蔽：成功 [{success}] 个群，失败 [{failed}] 个群"

        return f"已解除屏蔽：已作用于 [{success}] 个群"

    async def respond_to_result(
            self,
            message_id: str,
            user_id: str,
            text: str,
            *,
            allow_global: bool = False,
            global_targets: tuple[Setu, ...] = ()
    ) -> str | None:
        """处理对已发送色图的回复；返回 None 表示不是色图互动。"""
        async with self._data_lock:
            history_illust = self._history.get(message_id)
            if history_illust is None:
                return None

            history_illust = history_illust.copy()

        text = text.strip()
        if not text:
            return None

        score = self._parse_interaction_score(text)
        invocation = None
        if score is None:
            if not text.startswith(_SETU_INTERACTION_COMMANDS.prefix):
                return None

            try:
                invocation = _SETU_INTERACTION_COMMANDS.parse(text)

            except CommandError as exc:
                return f"色图互动指令错误：{exc}"

        async with self._data_lock:
            if history_illust.iid not in self._used_illusts:
                return "这张色图的互动记录已经过期了喵~"

        if score is not None:
            return await self._mark_interaction_score(history_illust, user_id, score)

        context = SetuInteractionContext(
            illust=history_illust,
            user_id=user_id,
            allow_global=allow_global,
            global_targets=global_targets
        )
        return await invocation.spec.handler(self, context, invocation.args)


    # =========================================================
    # 持久化
    # =========================================================

    def to_dict(self) -> dict:
        dt = datetime.now()
        dump_content = {
            "update": f"{dt:%x_%X}",
            "description": f"此文件为群 [{self._group_id}] 的色图库文件",
            "version": "0.1.0",
        }
        chunks = {k: list(v) for k, v in self._chunks.items()}
        available = {k: v.to_dict() for k, v in self._available_illusts.items()}
        used = {k: v.to_dict() for k, v in self._used_illusts.items()}
        freeze = {k: v.to_dict() for k, v in self._freeze.items()}
        history = {k: v.to_dict() for k, v in self._history.items()}
        tags_weight = self._tags_weight.copy()
        artists_weight = self._artists_weight.copy()
        ban_tags = list(self._ban_tags)
        ban_artists = list(self._ban_artists)
        ban_illusts = list(self._ban_illusts)
        ban_pages = copy.deepcopy(self._ban_pages)
        index_now = self._chunk_index_now

        # noinspection PyTypeChecker
        dump_content["data"] = {
            "chunks": chunks,
            "available": available,
            "used": used,
            "freeze": freeze,
            "history": history,
            "tags_weight": tags_weight,
            "artists_weight": artists_weight,
            "ban_tags": ban_tags,
            "ban_artists": ban_artists,
            "ban_illusts": ban_illusts,
            "ban_pages": ban_pages,
            "index_now": index_now
        }
        return dump_content

    def persist(self):
        """持久化，手动管理，后续要改为 SQLite"""
        atomic_save_json(
            self.to_dict(),
            self._save_path
        )

    def load_dict(self, data: dict):
        if not isinstance(data, dict):
            return

        data = data.get("data", data)
        if not isinstance(data, dict):
            return

        def _load_illusts(raw_data: object) -> dict[str, Illust]:
            if not isinstance(raw_data, dict):
                return {}

            result = {}
            for raw_illust in raw_data.values():
                illust = Illust.from_dict(raw_illust)
                if illust is not None:
                    result[illust.iid] = illust

            return result

        def _load_history(raw_data: object) -> dict[str, Illust]:
            if not isinstance(raw_data, dict):
                return {}

            result = {}
            for message_id, raw_illust in raw_data.items():
                illust = Illust.from_dict(raw_illust)
                if illust is not None and str(message_id):
                    result[str(message_id)] = illust

            return result

        chunks: dict[str, set[str]] = {}
        raw_chunks = data.get("chunks", {})
        if isinstance(raw_chunks, dict):
            for raw_index, raw_iids in raw_chunks.items():
                if not isinstance(raw_iids, (list, set, tuple)):
                    continue

                try:
                    index = str(int(raw_index))

                except (TypeError, ValueError):
                    continue

                chunks[index] = {
                    str(iid)
                    for iid in raw_iids
                    if isinstance(iid, (str, int)) and str(iid)
                }

        freezes: dict[str, PixivFreeze] = {}
        raw_freezes = data.get("freeze", {})
        if isinstance(raw_freezes, dict):
            for raw_freeze in raw_freezes.values():
                freeze = PixivFreeze.from_dict(raw_freeze)
                if freeze is not None:
                    freezes[freeze.freeze_id] = freeze

        def _load_weights(raw_weights: object) -> dict[str, float]:
            if not isinstance(raw_weights, dict):
                return {}

            weights = {}
            for key, value in raw_weights.items():
                if not isinstance(value, (int, float)):
                    continue

                weights[str(key)] = float(value)

            return weights

        def _load_str_set(raw_items: object) -> set[str]:
            if not isinstance(raw_items, (list, set, tuple)):
                return set()

            return {
                str(item)
                for item in raw_items
                if isinstance(item, (str, int)) and str(item)
            }

        ban_artists = set()
        raw_ban_artists = data.get("ban_artists", [])
        if isinstance(raw_ban_artists, (list, set, tuple)):
            for raw_artist in raw_ban_artists:
                try:
                    ban_artists.add(int(raw_artist))

                except (TypeError, ValueError):
                    continue

        ban_pages: dict[str, list[int]] = {}
        raw_ban_pages = data.get("ban_pages", {})
        if isinstance(raw_ban_pages, dict):
            for raw_iid, raw_pages in raw_ban_pages.items():
                if not isinstance(raw_pages, (list, set, tuple)):
                    continue

                pages = []
                for raw_page in raw_pages:
                    try:
                        page = int(raw_page)

                    except (TypeError, ValueError):
                        continue

                    if page not in pages:
                        pages.append(page)

                if pages:
                    ban_pages[str(raw_iid)] = pages

        try:
            index_now = max(1, int(data.get("index_now", self._chunk_index_now)))

        except (TypeError, ValueError):
            index_now = self._chunk_index_now

        self._chunks = chunks
        self._available_illusts = _load_illusts(data.get("available", {}))
        self._used_illusts = _load_illusts(data.get("used", {}))
        self._freeze = freezes
        self._history = _load_history(data.get("history", {}))
        self._tags_weight = _load_weights(data.get("tags_weight", {}))
        self._artists_weight = _load_weights(data.get("artists_weight", {}))
        self._ban_tags = _load_str_set(data.get("ban_tags", []))
        self._ban_artists = ban_artists
        self._ban_illusts = _load_str_set(data.get("ban_illusts", []))
        self._ban_pages = ban_pages
        self._chunk_index_now = index_now
        self._trim_interaction_history_unlocked()
