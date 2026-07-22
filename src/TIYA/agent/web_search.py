from __future__ import annotations

import time
import traceback
import asyncio
import json
import httpx
from typing import Literal
from datetime import datetime
from dataclasses import dataclass, field
from tavily import AsyncTavilyClient
from tavily.errors import TimeoutError
from TIYA.config import BASE_CFG, SETTING_CFG, get_proxy, ROOT_DIR
from TIYA.utils import parse_proxies_to_httpx, atomic_save_json
from TIYA.logger import get_logger
from TIYA.file_cache import get_file_path_async, add_file_async, remove_file_async
from TIYA.memory import get_context_memory_engine, BucketHandle, LLMPresetConfigError

_log = get_logger()
_GLOBAL_LOCK = asyncio.Lock()
_USAGE_STORAGE: None | dict[str, int] = None
_SAVE_DIR = ROOT_DIR / "data" / "websearch"
_SAVE_DIR.mkdir(parents=True, exist_ok=True)
_USAGE_FILE = _SAVE_DIR / "usage.json"
_CACHE_FILE = _SAVE_DIR / "cache.json"
_RESULT_CACHE: dict[str, dict] = {}
_memory_engine = get_context_memory_engine()
_WEBSEARCH_BUCKET: BucketHandle | None


class EmptyTokenError(ValueError):
    """未提供搜索API的Token"""
    ...


@dataclass(slots=True)
class WebResult:
    ok: bool = False
    result_from: Literal["SEARCH", "CACHE"] = "SEARCH"
    query: str = ""
    answer: str = ""
    results: list | dict = field(default_factory=list)
    error_message: str = ""
    usage: int = 0
    create_time: float = field(default_factory=time.time)

    def to_dict(self, trans_date=True):
        dt = datetime.fromtimestamp(self.create_time)
        return {
            "ok": self.ok,
            "result_from": self.result_from,
            "query": self.query,
            "answer": self.answer,
            "results": self.results.copy(),
            "error_message": self.error_message,
            "create_time": f"{dt:%Y-%m-%d %H:%M:%S}" if trans_date else self.create_time
        }

    @classmethod
    def from_dict(cls, data: dict):
        return WebResult(**data)


class Tavily:
    SEARCH_PARAMS = {
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": "basic",
        "auto_parameters": True,
        "include_usage": True,
        "timeout": 120
    }
    EXTRACT_PARAMS = {
        "extract_depth": "basic",
        "include_usage": True,
        "timeout": 120
    }
    MAP_PARAMS = {
        "max_depth": 2,
        "include_usage": True,
        "timeout": 120
    }
    def __init__(self):
        self.client : None | AsyncTavilyClient = None
        self._token = ""
        self._proxies: dict = {}
        self._update_client()
        self._last_online_update = 0

    def _update_client(self):
        token = BASE_CFG.WebSearch.get("TavilyKey", "")
        proxy = BASE_CFG.WebSearch.get("ProxyMode", "")
        if not token:
            return

        proxies = get_proxy(proxy)
        self._token = token
        self._proxies = proxies
        self.client = AsyncTavilyClient(api_key=token, proxies=proxies)

    @staticmethod
    def _parse_search_response(resp: dict) -> WebResult:
        new_result = WebResult()
        if not isinstance(resp, dict):
            return new_result

        results = resp.get("results", [])
        query = resp.get("query", "")
        answer = resp.get("answer", "")
        error_message = resp.get("detail", {}).get("error", "")
        usage = resp.get("usage", {}).get("credits", 0)

        if not (results or answer):
            new_result.error_message = error_message
            new_result.usage = usage
            return new_result

        new_result.query = query
        new_result.answer = answer
        new_result.results = results
        new_result.usage = usage
        new_result.ok = True
        return new_result

    @staticmethod
    def _parse_extract_response(resp: dict, query: str | None = None) -> WebResult:
        new_result = WebResult()
        if not isinstance(resp, dict):
            return new_result

        results = resp.get("results", [])
        error_message = resp.get("detail", {}).get("error", "")
        usage = resp.get("usage", {}).get("credits", 0)

        if not results:
            new_result.error_message = error_message
            new_result.usage = usage
            return new_result

        if query:
            new_result.query = query

        new_result.results = results
        new_result.usage = usage
        new_result.ok = True
        return new_result

    @staticmethod
    def _parse_map_response(resp: dict) -> WebResult:
        new_result = WebResult()
        if not isinstance(resp, dict):
            return new_result

        results = resp.get("results", [])
        error_message = resp.get("detail", {}).get("error", "")
        usage = resp.get("usage", {}).get("credits", 0)

        if not results:
            new_result.error_message = error_message
            new_result.usage = usage
            return new_result

        new_result.results = results
        new_result.usage = usage
        new_result.ok = True
        return new_result

    async def search(self, query: str, extra_params: dict = None):
        if self.client is None:
            raise EmptyTokenError

        if extra_params is None:
            extra_params = {}

        params = self.SEARCH_PARAMS.copy()
        params.update(extra_params)
        response = await self.client.search(query=query, **params)
        return self._parse_search_response(response)

    async def extract(self, urls: list[str] | str, query: str = None, extra_params: dict = None):
        if self.client is None:
            raise EmptyTokenError

        if extra_params is None:
            extra_params = {}

        params = self.EXTRACT_PARAMS.copy()
        params.update(extra_params)
        response = await self.client.extract(urls=urls, query=query, **params)
        return self._parse_extract_response(response, query)

    async def url_map(self, url: str, query: str = None, extra_params: dict = None):
        if self.client is None:
            raise EmptyTokenError

        if extra_params is None:
            extra_params = {}

        params = self.MAP_PARAMS.copy()
        params.update(extra_params)
        response = await self.client.map(url=url, instructions=query, **params)
        return self._parse_map_response(response)

    @staticmethod
    async def fetch_url(url: str, extra_parameters: dict = None) -> WebResult:
        if extra_parameters is None:
            extra_parameters = {}
        proxies = parse_proxies_to_httpx(get_proxy("NecessaryProxy"))
        if "proxy" in extra_parameters:
            proxies = None

        timeout = 120
        if "timeout" in extra_parameters:
            timeout = extra_parameters["timeout"]
            extra_parameters.pop("timeout")

        extra_parameters.setdefault("follow_redirects", True)
        async with httpx.AsyncClient(mounts=proxies, timeout=timeout, **extra_parameters) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()

            except Exception as E:
                return WebResult(
                ok=False,
                error_message=str(E)
            )

            else:
                return WebResult(
                    ok=bool(response.text),
                    results={"headers": dict(response.headers), "text": response.text}
                )

    async def usage(self) -> dict[str, int]:
        if not self._token:
            raise EmptyTokenError

        if not time.time() - self._last_online_update > 300:
            return {}

        self._last_online_update = time.time()
        proxies = parse_proxies_to_httpx(self._proxies)
        header = {"Authorization": f"Bearer {self._token}"}
        async with httpx.AsyncClient(mounts=proxies, headers=header) as cli:
            response = await cli.get(url="https://api.tavily.com/usage", timeout=30)
            if response.status_code != 200:
                raise httpx.HTTPError(str(response.status_code))

            resp: dict = response.json()
            if "detail" in resp:
                raise httpx.HTTPError(f"{resp.get('detail', {}).get('error', '')}")

            usage = resp.get("account", {})
            if not usage:
                raise KeyError

            return {
                "plan_usage": usage["plan_usage"],
                "plan_limit": usage["plan_limit"],
                "paygo_usage": usage["paygo_usage"],
                "paygo_limit": usage["paygo_limit"]
            }


class WebSearch:
    def __init__(self):
        self.tavily = None
        self.result: dict[str, dict] = {
            "SEARCH": {},
            "EXTRACT": {},
            "MAP": {},
            "FETCH": {}
        }
        asyncio.create_task(self.get_tavily())

    async def get_tavily(self):
        self.tavily = await _initiate_module()

    @staticmethod
    async def _update_usage(add_usage: int = 0):
        usage = _USAGE_STORAGE.copy()

        def _get_max_usage(new_dict: dict):
            if not new_dict:
                return

            if not any(k not in new_dict for k in {"plan_usage", "plan_limit", "paygo_usage", "paygo_limit"}):
                if new_dict["plan_usage"] + new_dict["plan_limit"] + new_dict["paygo_usage"] + new_dict["paygo_usage"] < 10:
                    usage["plan_usage"] = new_dict["plan_usage"] + add_usage
                    usage["plan_limit"] = new_dict["plan_limit"]
                    usage["paygo_usage"] = new_dict["paygo_usage"]
                    usage["paygo_limit"] = new_dict["paygo_limit"]
                    return

            usage["plan_usage"] = max(new_dict.get("plan_usage", 0), usage["plan_usage"])
            usage["plan_limit"] = max(new_dict.get("plan_limit", 0), usage["plan_limit"])
            usage["paygo_usage"] = max(new_dict.get("paygo_usage", 0), usage["paygo_usage"])
            usage["paygo_limit"] = max(new_dict.get("paygo_limit", 0), usage["paygo_limit"])

        retry = 0
        online_usage = {}
        while retry < 3:
            try:
                online_usage = await _TAVILY.usage()

            except Exception as E:
                _log.error(E)

            else:
                break

            finally:
                retry += 1

        async with _GLOBAL_LOCK:
            if add_usage > 0:
                plan_usage = usage["plan_usage"] + add_usage
                if plan_usage > usage["plan_limit"]:
                    overusage = plan_usage - usage["plan_limit"]
                    usage["plan_usage"] = usage["plan_limit"]
                    usage["paygo_usage"] += overusage

                else:
                    usage["plan_usage"] = plan_usage

            _get_max_usage(online_usage)
            _USAGE_STORAGE.update(usage)
            atomic_save_json(usage, _USAGE_FILE, indent=4)

    @staticmethod
    def _check_usage_limit() -> bool:
        usage = _USAGE_STORAGE.copy()
        if usage["paygo_limit"] > 0:
            available = usage["paygo_limit"] - usage["paygo_usage"]

        else:
            available = usage["plan_limit"] - usage["plan_usage"]

        if available < 5:
            return False

        else:
            return True

    async def search(self, query: str, include_domains: list[str] = None, purge: bool = False) -> list[dict]:
        """
        Run a Websearch.
        :param query: natural language query content, use english first.
        :param include_domains: websearch in certain domains, only given when explicitly requested or prompted.
        :param purge: default False;  when True, will force run websearch instead of try to get from cache first.
        :return: a list of result dict
        """
        if self.tavily is None:
            self.tavily = await _initiate_module()

        if not purge:
            # CoMe
            try:
                cache_results = await _get_memory(query)

            except (RuntimeError, LLMPresetConfigError):
                _log.debug("记忆系统错误")
                _log.debug(traceback.format_exc())
                pass

            else:
                if cache_results:
                    return [cache_results[0].to_dict()]

            cache_results = await _get_cache(query, "SEARCH")
            if cache_results:
                return [x.to_dict() for x in cache_results]

        if not self._check_usage_limit():
            result = WebResult(error_message="WebSearch out of API usage limit")
            return [result.to_dict()]

        retry = 0
        response = WebResult()
        while not response.ok and retry < 3:
            try:
                response = await self.tavily.search(query, extra_params={"include_domains": include_domains})

            except EmptyTokenError:
                response.error_message = "API Token not set, Can't use WebSearch"
                return [response.to_dict()]

            except TimeoutError:
                response.error_message = "Timeout"

            finally:
                retry += 1

        if response.usage > 0:
            await self._update_usage(response.usage)

        if response.ok:
            # CoMe
            try:
                await _add_memory(query, response)

            except (RuntimeError, LLMPresetConfigError):
                _log.debug("记忆系统错误")
                _log.debug(traceback.format_exc())
                pass

            await _add_cache(query, response, "SEARCH")
            self.result["SEARCH"].setdefault(query, []).append(response)

        return [response.to_dict()]

    async def url_map(self, url: str, purge: bool = False) -> list[dict]:
        """
        find sub url relative to given url.
        :param url: target url.
        :param purge: default False;  when True, will force run websearch instead of try to get from cache first.
        :return: a list of result dict
        """
        if not purge:
            cache_results = await _get_cache(url, "MAP")
            if cache_results:
                return [x.to_dict() for x in cache_results]

        retry = 0
        response = WebResult()
        while not response.ok and retry < 3:
            try:
                response = await self.tavily.url_map(url)

            except EmptyTokenError:
                response.error_message = "API Token not set, Can't use WebSearch"
                return [response.to_dict()]

            except TimeoutError:
                response.error_message = "Timeout"

            finally:
                retry += 1

        if response.usage > 0:
            await self._update_usage(response.usage)

        if response.ok:
            await _add_cache(url, response, "MAP")
            self.result["MAP"].setdefault(url, []).append(response)

        return [response.to_dict()]

    async def extract(self, urls: list[str] | str, query: str, purge: bool = False) -> list[dict]:
        """
        extract (crawl) content from given url, can specify query string for extract certain topic.
        :param urls: string: one url;  urls list: multiple url.
        :param query: natural language query content, use english first;  when given, results will rerank by the query.
        :param purge: default False;  when True, will force run websearch instead of try to get from cache first.
        :return: a list of result dict
        """
        async def extract_one(_url: str) -> list[dict]:
            if not purge:
                cache_results = await _get_cache(_url, "EXTRACT")
                if cache_results:
                    return [x.to_dict() for x in cache_results]

            retry = 0
            response = WebResult()
            while not response.ok and retry < 3:
                try:
                    response = await self.tavily.extract(_url, query=query)

                except EmptyTokenError:
                    response.error_message = "API Token not set, Can't use WebSearch"
                    return [response.to_dict()]

                except TimeoutError:
                    response.error_message = "Timeout"

                finally:
                    retry += 1

            if response.usage > 0:
                await self._update_usage(response.usage)

            if response.ok:
                await _add_cache(_url, response, "EXTRACT")
                self.result["EXTRACT"].setdefault(_url, []).append(response)

            return [response.to_dict()]

        if isinstance(urls, str):
            urls = [urls]

        resp = []
        for url in urls:
            resp.extend(await extract_one(url))

        return resp

    async def raw_crawl(self, url: str, extra_parameters: dict = None,  purge: bool = False) -> list[dict]:
        """
        using local tool fetch url headers and raw HTML text, without any parse.
        only use this tool when `extract` fail, will inject a lot token in context
        :param url: target url.
        :param extra_parameters: params can use in httpx, such as cookies, headers.
        :param purge: default False;  when True, will force fetch url instead of try to get from cache first.
        :return: a list of result dict
        """
        if not purge:
            cache_results = await _get_cache(url, "FETCH")
            if cache_results:
                return [x.to_dict() for x in cache_results]

        retry = 0
        response = WebResult()
        while not response.ok and retry < 3:
            response = await self.tavily.fetch_url(url, extra_parameters)
            retry += 1

        if response.ok:
            await _add_cache(url, response, "FETCH")
            self.result["FETCH"].setdefault(url, []).append(response)

        return [response.to_dict()]

    def to_dict(self) -> dict:
        return {
            "SEARCH": list(self.result["SEARCH"].keys()),
            "EXTRACT": list(self.result["EXTRACT"].keys()),
            "MAP": list(self.result["MAP"].keys()),
            "FETCH": list(self.result["FETCH"].keys())
        }

    async def load_dict(self, data: dict):
        search_cache = data.get("SEARCH", [])
        extract_cache = data.get("EXTRACT", [])
        map_cache = data.get("MAP", [])
        fetch_cache = data.get("FETCH", [])

        async def _load(key: Literal["SEARCH", "MAP", "EXTRACT", "FETCH"], caches: list):
            for k in caches:
                results = await _get_cache(k, key)
                if results:
                    self.result[key][k] = results

        await asyncio.gather(
            asyncio.create_task(_load("SEARCH", search_cache)),
            asyncio.create_task(_load("EXTRACT", extract_cache)),
            asyncio.create_task(_load("MAP", map_cache)),
            asyncio.create_task(_load("FETCH", fetch_cache))
        )



_TAVILY: None | Tavily = None
async def _initiate_module() -> Tavily:
    global _TAVILY
    global _USAGE_STORAGE
    global _WEBSEARCH_BUCKET
    async with _GLOBAL_LOCK:
        _WEBSEARCH_BUCKET = await _memory_engine.set_bucket(
            "[WEBSEARCH]",
            summary="存放WEBSEARCH的缓存",
            summary_locked=True
        )
        if _TAVILY is None:
            _TAVILY = Tavily()

        if _USAGE_STORAGE is None:
            usage = {
                "plan_usage": 0,
                "plan_limit": 0,
                "paygo_usage": 0,
                "paygo_limit": 0
            }
            retry = 0
            while retry < 3:
                try:
                    online_usage = await _TAVILY.usage()

                except Exception as E:
                    _log.error(E)
                    _log.debug(traceback.format_exc())
                    online_usage = {}

                else:
                    break

                finally:
                    retry += 1

            load_content = {}
            if _USAGE_FILE.is_file():
                with _USAGE_FILE.open('r', encoding="utf-8") as f:
                    rc = f.read()

                try:
                    load_content: dict = json.loads(rc)

                except json.JSONDecodeError:
                    pass

            usage["plan_usage"] = max(online_usage.get("plan_usage", 0), load_content.get("plan_usage", 0))
            usage["plan_limit"] = max(online_usage.get("plan_limit", 0), load_content.get("plan_limit", 0))
            usage["paygo_usage"] = max(online_usage.get("paygo_usage", 0), load_content.get("paygo_usage", 0))
            usage["paygo_limit"] = max(online_usage.get("paygo_limit", 0), load_content.get("paygo_limit", 0))
            _USAGE_STORAGE = usage

        return _TAVILY

async def _get_cache(key: str, result_type: Literal["SEARCH", "MAP", "EXTRACT", "FETCH"]) -> list[WebResult]:
    """已改用CoMe，仅作为fallback使用"""
    if not _RESULT_CACHE:
        if _CACHE_FILE.is_file():
            load_content: dict = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            _RESULT_CACHE.update(load_content)

        else:
            _RESULT_CACHE.update(
                {
                    "SEARCH": {},
                    "EXTRACT": {},
                    "MAP": {},
                    "FETCH": {}
                }
            )

        atomic_save_json(_RESULT_CACHE, _CACHE_FILE)

    finds = _RESULT_CACHE[result_type].get(key, [])
    if not finds:
        return []

    now = time.time()
    expire = SETTING_CFG.Agent.WebSearchCacheExpire * 24 * 3600
    result = []
    for item in finds:
        if now - item["create_time"] > expire:
            filename = item["filename"]
            await remove_file_async(filename)
            _RESULT_CACHE[result_type].pop(key, None)
            continue

        filename = item["filename"]
        try:
            path = await get_file_path_async(filename)

        except FileNotFoundError:
            continue

        try:
            load_content: dict = json.loads(path.read_text(encoding="utf-8"))

        except json.JSONDecodeError:
            await remove_file_async(filename)
            _RESULT_CACHE[result_type].pop(key, None)
            continue

        try:
            new_result = WebResult.from_dict(load_content)
            new_result.result_from = "CACHE"

        except Exception as E:
            _log.error(E)
            _log.debug(E)
            await remove_file_async(filename)
            _RESULT_CACHE[result_type].pop(key, None)
            continue

        result.append(new_result)

    atomic_save_json(_RESULT_CACHE, _CACHE_FILE)
    result.sort(key=lambda x: x.create_time, reverse=True)
    result = [x for i, x in enumerate(result) if i < 3]
    return result

async def _add_cache(key: str, data: WebResult, result_type: Literal["SEARCH", "MAP", "EXTRACT", "FETCH"]):
    if not _RESULT_CACHE:
        if _CACHE_FILE.is_file():
            load_content: dict = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            _RESULT_CACHE.update(load_content)

        else:
            _RESULT_CACHE.update(
                {
                    "SEARCH": {},
                    "EXTRACT": {},
                    "MAP": {},
                    "FETCH": {}
                }
            )

        atomic_save_json(_RESULT_CACHE, _CACHE_FILE)

    filename = await add_file_async(json.dumps(data.to_dict(False), ensure_ascii=False).encode())
    info = {
        "create_time": data.create_time,
        "filename": filename
    }
    _RESULT_CACHE[result_type].setdefault(key, []).append(info)
    atomic_save_json(_RESULT_CACHE, _CACHE_FILE)

async def _add_memory(query: str, data: WebResult):
    result = json.dumps(
        {k: v for k, v in data.to_dict().items() if k not in {"ok", "result_from"}},
        ensure_ascii=False, indent=2)

    memory_text = f"WebSearch记忆:  \n\nQUERY: {query}  \n\nRESULT:  \n```json\n{result}\n```"
    await _WEBSEARCH_BUCKET.add_memory(memory_text)

async def _get_memory(query: str) -> list[WebResult]:
    result = await _WEBSEARCH_BUCKET.query(query)
    if not (result.success and result.result_source != "LOCAL" and result.matches):
        return []

    mem_obj = result.matches[0]
    if not mem_obj.score > SETTING_CFG.Agent.WebSearchCoMeGate:  # 默认0.8
        return []

    mem = await _WEBSEARCH_BUCKET.get_memory(mem_obj.key)
    mem_content = mem.content
    answer = result.sub_answer or result.answer
    return [
        WebResult(
            ok=True,
            result_from="CACHE",
            query=query,
            answer=answer,
            results=[mem_content]
        )
    ]
