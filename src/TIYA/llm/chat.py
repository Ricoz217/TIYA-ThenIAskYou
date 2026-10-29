from __future__ import annotations
import asyncio
import copy
import time
import weakref
from typing import Callable, Awaitable, TYPE_CHECKING
from uuid import uuid4
from contextvars import ContextVar
import httpx
from TIYA.config import DATA_DIR, SETTING_CFG
from TIYA.logger import get_logger
from TIYA.utils import parse_proxies_to_httpx
from .usage_compat import ApiPrice, LLMUsage, _GLOBAL_USAGE
from .model import (
    Context,
    SystemPrompt,
    BasePrompt,
    Prompts,
    TextPrompt,
    ToolInput,
    TokenUsage,
    AskTask,
    ChatConfig,
    CallOutcome,
    ContextOverflowError,
)
from .config import Preset, Route, fingerprint, credential
from .gateway import Gateway
from .runtime import Runtime, estimate
from .state import StateStore
from .pricing import PriceCatalog
from .provider import get_provider
from .provider.legacy import LegacyPayloadBuilder

_log = get_logger()
if TYPE_CHECKING:
    from TIYA.logger import Logger, BlockHandle
_CONTEXT_LOCKS = weakref.WeakKeyDictionary()


def context_lock(context):
    per_loop = _CONTEXT_LOCKS.setdefault(context, weakref.WeakKeyDictionary())
    loop = asyncio.get_running_loop()
    if loop not in per_loop:
        per_loop[loop] = asyncio.Lock()
    return per_loop[loop]


class Chat(LegacyPayloadBuilder):
    def __init__(
        self,
        *,
        endpoint: str = "",
        token: str = "",
        model: str = "",
        max_context: int = None,
        auto_compress_rate: float = None,
        system_prompt: str = "",
        api_provider: str = "openai",  # "openai" | "anthropic"
        price: ApiPrice = None,  # 模型单价, price / 1M token
        client_params: dict = None,
        model_params: dict = None,
        context: Context = None,
        keep_alive: bool = False,
        logger: Logger | BlockHandle = None,
        log_off: bool = False,
        usage_record: list[LLMUsage] = None,
        config: ChatConfig = None,
    ):
        if context is None:
            context = Context()
            if system_prompt:
                context.append(SystemPrompt(system_prompt))

        if logger is None:
            logger = _log

        if client_params is None:
            client_params = {}

        if model_params is None:
            model_params = {"max_tokens": 8192, "temperature": 0.7, "stream": True}

        else:
            base_params = {"max_tokens": 8192, "temperature": 0.7, "stream": True}
            base_params.update(model_params)
            model_params = base_params

        if usage_record is None:
            usage_record = []

        # 配置参数
        self._endpoint: str = endpoint
        self._token: str = token
        self._model: str = model
        self._max_context: int | None = max_context
        self._compress_rate: float | None = auto_compress_rate
        self._provider = api_provider
        self._client_params: dict = copy.deepcopy(client_params)
        self._model_params: dict = copy.deepcopy(model_params)
        self._price: ApiPrice = price

        # 数据容器
        self._context: Context = context
        self._image_cache: dict[str, str] = {}  # 用于储存照片的完整base64缓存
        self._queue = asyncio.PriorityQueue()
        self._sequence = 0
        self._preset = None
        self._route_clients = {}
        self._runtime = None
        self._last_outcome = CallOutcome()
        self._outcome_var = ContextVar(f"llm_outcome_{id(self)}", default=None)
        self._ask_tasks: dict[str, AskTask] = {}
        self._tools_mapping: dict[str, Callable | Awaitable] = {}
        self._last_usage: TokenUsage = TokenUsage()
        self._usage_record: list[LLMUsage] = usage_record
        self._context_use: int = 0
        self._payload_discard_prompt_ids: set[str] = set()
        self._payload_append_prompts: list[BasePrompt] = []

        # 状态位和工具
        self._closed = False
        self._id = str(time.time_ns())
        self._keep_alive = keep_alive
        self._running = asyncio.Event()
        self._running.set()
        self._running_task_id = ""
        self._client: httpx.AsyncClient | None = None
        self._client_stale = False
        self._logger = logger
        self.log_off = log_off
        self._worker_loop = asyncio.create_task(self.run())
        self._first_SSE_retry = True
        if self._keep_alive:
            self._update_client()

        if isinstance(config, ChatConfig):
            self.setting(config)

    def setting(self, new_config: ChatConfig):
        if not isinstance(new_config, ChatConfig):
            return

        client_config_changed = False
        if new_config.complete:
            self._preset = copy.deepcopy(new_config.preset)
            self._price = new_config.price
            self._max_context = new_config.max_context
            self._compress_rate = new_config.auto_compress_rate
            client_config_changed |= self._client_params != (
                new_config.client_params or {}
            )
            self._client_params = copy.deepcopy(new_config.client_params or {})
            self._model_params = copy.deepcopy(new_config.model_params or {})
        elif new_config.preset is not None:
            self._preset = copy.deepcopy(new_config.preset)
        if new_config.endpoint is not None:
            client_config_changed |= new_config.endpoint != self._endpoint
            self._endpoint = new_config.endpoint

        if new_config.token is not None:
            client_config_changed |= new_config.token != self._token
            self._token = new_config.token

        if new_config.model is not None:
            if self._context._usage_accumulation:
                self._context._usage_accumulation.clear()

            self._model = new_config.model

        if new_config.system_prompt is not None:
            self._context.append(SystemPrompt(new_config.system_prompt))

        if new_config.api_provider is not None:
            client_config_changed |= new_config.api_provider != self._provider
            self._provider = new_config.api_provider

        if new_config.price is not None:
            if isinstance(new_config.price, ApiPrice):
                self._price = new_config.price

        if new_config.max_context is not None:
            if isinstance(new_config.max_context, int):
                self._max_context = new_config.max_context

        if new_config.auto_compress_rate is not None:
            if isinstance(new_config.auto_compress_rate, (float, int)):
                self._compress_rate = new_config.auto_compress_rate

        if new_config.client_params is not None:
            client_config_changed |= new_config.client_params != self._client_params
            self._client_params = copy.deepcopy(new_config.client_params)

        if new_config.model_params is not None:
            self._model_params = copy.deepcopy(new_config.model_params)

        if new_config.keep_alive is not None:
            self._keep_alive = new_config.keep_alive

        if new_config.logger is not None:
            self._logger = new_config.logger

        if new_config.log_off is not None:
            self.log_off = bool(new_config.log_off)

        if client_config_changed and self._client is not None:
            self._client_stale = True

        if not self.log_off:
            self._logger.info(f"已更新配置。当前模型: [{self._model}]")

    def _update_client(self):
        client_params = self._client_params.copy()
        proxies = client_params.pop("proxies", {})
        proxy_mounts = parse_proxies_to_httpx(proxies)
        client_params.pop("headers", None)
        headers = self._header_constructor()
        self._client = httpx.AsyncClient(
            mounts=proxy_mounts,
            headers=headers,
            **client_params,
        )
        self._client_stale = False

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client_stale and self._client is not None:
            stale_client = self._client
            self._client = None
            try:
                await stale_client.aclose()

            except Exception as exc:
                self._logger.warning(f"关闭过期 LLM 客户端失败: {exc}")

        if self._client is None:
            self._update_client()

        return self._client

    def add_tools(self, tools: Prompts | list[ToolInput] | ToolInput) -> Prompts:
        self._context.append(tools)
        return self._context.tools

    def add_context(self, new_context: Context | Prompts | BasePrompt) -> Context:
        self._context.append(new_context)
        return self._context.copy()

    @property
    def model(self) -> dict:
        """返回模型信息"""
        return {
            "model": self._model,
            "endpoint": self._endpoint,
            "token": self._token,
            "provider": self._provider,
            "parameter": self._model_params,
        }

    @property
    def tools(self) -> Prompts:
        """返回当前工具集"""
        return self._context.tools

    @property
    def context(self) -> Context:
        """返回Context对象的shadow copy"""
        return self._context.copy()

    @property
    def context_use(self) -> float:
        if self._max_context is None or self._max_context <= 0:
            return -1.0

        return self._context_use / self._max_context

    def replace_context(self, new_context: Context):
        """使用外部的Context，让Chat自动管理"""
        self._context = new_context

    async def get_context(self) -> Context:
        """注意，此函数返回的是Context对象本身，不是shadow"""
        return self._context

    def back_to_last_round(self) -> Prompts:
        return self._context.back2last_round()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await asyncio.sleep(0)
        await self.close()

    def _snapshot_preset(self):
        category = {"openai": "chat_completions", "anthropic": "anthropic_api"}.get(
            self._provider.lower(), self._provider.lower()
        )
        if self._preset is not None:
            result = copy.deepcopy(self._preset)
            primary = result.routes[0]
            if (
                self._endpoint != primary.endpoint
                or self._token != credential(primary.settings)
                or category != primary.category
            ):
                # A legacy scalar override cannot implicitly move the old Files binding to another account.
                primary.files_ref = None
                primary.files_settings = None
            primary.category, primary.model = category, self._model
            primary.settings["endpoint"] = self._endpoint
            primary.settings["key"] = self._token
            primary.params = copy.deepcopy(self._model_params)
            primary.client_params = copy.deepcopy(self._client_params)
            primary.max_context = self._max_context
            return result
        route = Route(
            f"{category}:legacy:{self._model}",
            category,
            "legacy",
            self._model,
            {"endpoint": self._endpoint, "key": self._token},
            copy.deepcopy(self._model_params),
            self._max_context,
            "full",
            client_params=copy.deepcopy(self._client_params),
        )
        if self._price is not None:
            route.settings["legacy_price"] = self._price.to_dict()
        return Preset("", [route], self._compress_rate)

    def _header_constructor(self):
        return get_provider(self._snapshot_preset().routes[0]).headers()

    def _service(self):
        if self._runtime is None:
            self._runtime = Runtime(
                StateStore(DATA_DIR / "llm" / "runtime.sqlite3"),
                _GLOBAL_USAGE.ledger,
                PriceCatalog(DATA_DIR / "llm" / "pricing.sqlite3"),
                self._logger,
                self._usage_record,
            )
        return self._runtime

    async def _gateway(self, route):
        primary = self._snapshot_preset().routes[0]
        if (
            route.endpoint == self._endpoint
            and route.settings.get("key") == self._token
            and route.category == primary.category
        ):
            return Gateway(
                await self._ensure_client(), download_options=route.client_params
            )
        key = fingerprint(
            [route.endpoint, get_provider(route).headers(), route.client_params]
        )
        if key not in self._route_clients:
            params = copy.deepcopy(route.client_params)
            proxies = params.pop("proxies", {})
            params.pop("headers", None)
            self._route_clients[key] = httpx.AsyncClient(
                mounts=parse_proxies_to_httpx(proxies),
                headers=get_provider(route).headers(),
                **params,
            )
        return Gateway(self._route_clients[key], download_options=route.client_params)

    async def _tiktoken_prompts(self, prompts):
        return (await asyncio.to_thread(estimate, Context(prompts)))["input"]

    async def _estimate_context_windows(self, new_prompts):
        snapshot = copy.deepcopy(self._context)
        snapshot.append(new_prompts)
        return (await asyncio.to_thread(estimate, snapshot))["input"]

    async def run(self):
        while True:
            _, _, task = await self._queue.get()
            try:
                if task.cancelled or task.event.is_set():
                    continue
                task.handle = asyncio.create_task(self._execute_task(task))
                try:
                    task.response = await task.handle
                except asyncio.CancelledError:
                    if self._closed:
                        return
                except Exception as exc:
                    task.error = exc
                    task.outcome = getattr(
                        exc,
                        "llm_outcome",
                        CallOutcome(
                            "failed",
                            False,
                            remote_state="completed" if task.pending else "unknown",
                            category="commit",
                        ),
                    )
            finally:
                task.event.set()
                self._queue.task_done()
                task = None

    async def post(self, task):
        if task.cancelled or task.event.is_set():
            return None
        preset = self._snapshot_preset()
        self._running_task_id = task.id
        try:
            service = await asyncio.to_thread(self._service)
            pending = await service.execute(
                self._context, task.prompts, preset, self._gateway, task.timeout
            )
            task.pending = pending
            task.outcome = CallOutcome(
                "received",
                False,
                pending.call_id,
                pending.result.request_id,
                "completed",
            )
            response = pending.result.prompts
            response._duration = max(0.0, time.time() - float(task.prompts.time))
            self._last_usage = copy.deepcopy(response.usage)
            self._context_use = response.usage.total_t if response.usage else 0
            return response
        except asyncio.CancelledError as exc:
            task.outcome = getattr(exc, "llm_outcome", CallOutcome("cancelled"))
            raise
        except ContextOverflowError:
            raise
        except Exception as exc:
            task.error = exc
            task.outcome = getattr(
                exc, "llm_outcome", CallOutcome("failed", category=type(exc).__name__)
            )
            self._logger.error(f"LLM 请求停止: {type(exc).__name__}")
            return None

    async def _execute_task(self, task):
        context = self._context
        async with context_lock(context):
            if self._closed or task.cancelled or task.event.is_set():
                return None
            self._running.clear()
            try:
                async with asyncio.timeout(task.timeout):
                    response = await self.post(task)
                if (
                    response is not None
                    and not task.cancelled
                    and not task.event.is_set()
                ):
                    if self._context is not context:
                        raise RuntimeError("Context 已替换，拒绝提交")
                    await self._service().commit_async(
                        context, task.pending, lambda: setattr(task, "committed", True)
                    )
                    task.outcome.status = "completed"
                    task.outcome.committed = True
                return response
            finally:
                if not self._keep_alive:
                    await self._close_clients()
                self._running_task_id = ""
                self._running.set()

    async def post_nowait(self, task):
        try:
            task.handle = asyncio.create_task(self._execute_task(task))
            return await task.handle
        except asyncio.CancelledError:
            if not task.cancelled:
                raise
            return None

    async def ask(
        self, new_ask, *, task_id=None, queue=True, no_wait=False, timeout=None
    ):
        if self._closed or not new_ask:
            return None
        timeout = (
            timeout
            if timeout is not None and timeout > 0
            else SETTING_CFG.LLM.ChatRequestTimeout
        )
        if not isinstance(new_ask, Prompts):
            new_ask = (
                Prompts(TextPrompt("user", new_ask))
                if isinstance(new_ask, str)
                else Prompts(new_ask)
            )
        if (
            not queue
            and not no_wait
            and (
                self._ask_tasks
                or context_lock(self._context).locked()
                or not self._queue.empty()
            )
        ):
            return None
        # Preserve the existing early compression signal without changing another candidate's Context.
        rate = max(
            0.3,
            min(0.9, self._compress_rate if self._compress_rate is not None else 0.7),
        )
        if (
            self._max_context
            and self._last_usage.input_t >= (rate - 0.1) * self._max_context
        ):
            if (
                await self._estimate_context_windows(new_ask)
                >= rate * self._max_context
            ):
                raise ContextOverflowError()
        identifier = task_id or uuid4().hex
        while identifier in self._ask_tasks:
            identifier = uuid4().hex
        task = AskTask(copy.deepcopy(new_ask), timeout, identifier)
        self._ask_tasks[identifier] = task
        self._sequence += 1
        await self._queue.put((0 if no_wait else 1, self._sequence, task))
        try:
            await asyncio.wait_for(task.event.wait(), timeout)
            self._publish_outcome(task)
            if isinstance(task.error, ContextOverflowError):
                raise task.error
            return task.response
        except TimeoutError:
            await self.stop_from_id(identifier)
            self._publish_outcome(task)
            return None
        except asyncio.CancelledError:
            await self.stop_from_id(identifier)
            self._publish_outcome(task)
            raise
        finally:
            self._ask_tasks.pop(identifier, None)

    def _publish_outcome(self, task):
        outcome = task.outcome or CallOutcome(
            "cancelled" if task.cancelled else "failed"
        )
        self._last_outcome = outcome
        self._outcome_var.set(outcome)

    @property
    def last_outcome(self):
        return self._outcome_var.get() or self._last_outcome

    def copy(self, new_context=True):
        config = ChatConfig(
            endpoint=self._endpoint,
            token=self._token,
            model=self._model,
            max_context=self._max_context,
            auto_compress_rate=self._compress_rate,
            api_provider=self._provider,
            price=copy.deepcopy(self._price),
            client_params=copy.deepcopy(self._client_params),
            model_params=copy.deepcopy(self._model_params),
            preset=copy.deepcopy(self._preset),
            complete=True,
            keep_alive=self._keep_alive,
            logger=self._logger,
            log_off=self.log_off,
        )
        context = None if new_context else self._context
        return Chat(
            config=config,
            context=context,
            system_prompt=self._context.system.text
            if new_context and self._context.system
            else "",
            usage_record=self._usage_record.copy(),
        )

    async def clear(self, system=False, tools=False):
        async with context_lock(self._context):
            self._context.clear(system, tools)

    async def stop_from_id(self, task_id):
        task = self._ask_tasks.get(task_id)
        if task is None:
            return
        if task.committed:
            return  # The response has already been atomically committed; cancellation arrived too late.
        task.cancelled = True
        if task.handle and not task.handle.done():
            task.handle.cancel()
            await asyncio.gather(task.handle, return_exceptions=True)
        task.response = None
        task.event.set()

    async def stop_one(self):
        if self._running_task_id:
            await self.stop_from_id(self._running_task_id)

    async def stop_all(self):
        handles = []
        for task in list(self._ask_tasks.values()):
            if task.committed:
                continue
            task.cancelled = True
            task.response = None
            task.event.set()
            if task.handle and not task.handle.done():
                task.handle.cancel()
                handles.append(task.handle)
        await asyncio.gather(*handles, return_exceptions=True)

    async def _close_clients(self):
        clients = list(self._route_clients.values())
        self._route_clients.clear()
        if self._client is not None:
            clients.append(self._client)
        self._client = None
        for client in clients:
            await client.aclose()

    async def close(self):
        self._closed = True
        await self.stop_all()
        self._worker_loop.cancel()
        await asyncio.gather(self._worker_loop, return_exceptions=True)
        await self._close_clients()

    def __del__(self):
        getattr(self, "_image_cache", {}).clear()
