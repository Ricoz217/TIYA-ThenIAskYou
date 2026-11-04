from __future__ import annotations

import asyncio
import copy
import time
from contextvars import ContextVar
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx

from TIYA.config import DATA_DIR, SETTING_CFG
from TIYA.logger import get_logger
from TIYA.utils import fingerprint, parse_proxies_to_httpx

from .config_model import ChatConfig, FilesConfig, LLMConfig, Params, RequestPolicy
from .gateway import Gateway
from .model import (
    AskTask,
    BasePrompt,
    CallOutcome,
    Context,
    ContextOverflowError,
    Prompts,
    SystemPrompt,
    TextPrompt,
    TokenUsage,
    ToolInput,
)
from .pricing import PriceCatalog
from .provider import get_provider
from .runtime import Runtime
from .state import StateStore
from .usage import check_context_window, reported_context_tokens
from .usage_compat import _GLOBAL_USAGE, ApiPrice, LLMUsage
from .workers import run_sync

_log = get_logger()
if TYPE_CHECKING:
    from TIYA.logger import BlockHandle, Logger

class Chat:
    def __init__(
        self,
        *,
        endpoint: str = "",
        token: str = "",
        model: str = "",
        max_context: int | None = None,
        auto_compress_rate: float | None = None,
        system_prompt: str = "",
        api_provider: str = "openai",  # "openai" | "anthropic"
        price: ApiPrice | None = None,  # 模型单价, price / 1M token
        client_params: dict[str, object] | None = None,
        model_params: Params | None = None,
        context: Context | None = None,
        keep_alive: bool = False,
        logger: Logger | BlockHandle | None = None,
        log_off: bool = False,
        usage_record: list[LLMUsage] | None = None,
        config: ChatConfig | None = None,
    ) -> None:
        """创建对话会话，需在运行中的事件循环内调用。

        config 提供有序候选和策略；endpoint/token/model 等参数保留直接单模型入口。
        context 若传入则直接共享，不复制；usage_record 是可选的附加账本订阅者。
        keep_alive 控制请求后是否保留 HTTP 连接，close 不销毁进程共享线程池。
        """
        if context is None:
            context = Context()
            if system_prompt:
                context.append(SystemPrompt(system_prompt))

        if logger is None:
            logger = _log

        if client_params is None:
            client_params = {}

        category = {"openai": "chat_completions", "anthropic": "anthropic_api"}.get(
            api_provider.lower(), api_provider.lower()
        )
        base_params = {"stream": True}
        # 没有这个默认值
        # if category in ("chat_completions", "anthropic_api"):
        #     base_params.update(max_tokens=8192, temperature=0.7)

        base_params.update(model_params or {})

        if usage_record is None:
            usage_record = []

        # One authoritative configuration; flat business reads are primary properties.
        self._config = ChatConfig(
            [
                LLMConfig(
                    category,
                    "legacy",
                    endpoint,
                    token,
                    model,
                    model_params=base_params,
                    client_params=copy.deepcopy(client_params),
                    max_context=max_context,
                    context_history_mode="full",
                    legacy_price=price,
                )
            ],
            auto_compress_rate=auto_compress_rate,
            request_policy=RequestPolicy(),
            keep_alive=keep_alive,
            logger=logger,
            log_off=log_off,
        )

        # 数据容器
        self._context: Context = context
        self._queue = asyncio.PriorityQueue()
        self._sequence = 0
        self._route_clients: dict[str, httpx.AsyncClient] = {}  # 保存使用过/在使用的连接对象
        self._runtime = None
        self._last_outcome = CallOutcome()
        self._outcome_var = ContextVar(f"llm_outcome_{id(self)}", default=None)
        self._ask_tasks: dict[str, AskTask] = {}
        self._last_usage: TokenUsage = TokenUsage()
        self._usage_record: list[LLMUsage] = usage_record
        self._context_use: int = 0
        for previous_round in reversed(context.rounds.values()):
            if previous_round.usage is not None:
                self._last_usage = previous_round.usage.copy()
                self._context_use = self._last_usage.total_t
                break

        self._usage_context_signature: str | None = (
            context.signature if self._last_usage.samples else None
        )
        self._usage_scope: str | None = None  # 存档只有 model；新提交复用实际候选 scope。

        # 状态位和工具
        self._closed = False
        self._id = str(time.time_ns())
        self._config.keep_alive = keep_alive
        self._running = asyncio.Event()
        self._running.set()
        self._running_task_id = ""
        self._client: httpx.AsyncClient | None = None
        self._client_stale = False
        self._config.logger = logger
        self.log_off = log_off
        self._worker_loop = asyncio.create_task(self.run())
        if self._config.keep_alive:
            self._update_client()

        if isinstance(config, ChatConfig):
            self.setting(config)

    def setting(self, new_config: ChatConfig) -> None:
        """Replace candidate configuration, or consume a documented partial update."""
        if not isinstance(new_config, ChatConfig):
            return
        previous = self._config
        old_client = self._client_identity(previous.primary)
        if new_config.candidates is not None:
            updated = copy.deepcopy(new_config)
            updated.request_policy = updated.request_policy or RequestPolicy()

        else:
            updated = previous.copy()
            if new_config.model_params is not None:
                assert updated.candidates is not None
                updated.candidates[0] = copy.deepcopy(previous.primary)
                updated.model_params = new_config.model_params

            if new_config.auto_compress_rate is not None:
                updated.auto_compress_rate = new_config.auto_compress_rate

            if new_config.request_policy is not None:
                updated.request_policy = copy.deepcopy(new_config.request_policy)

        # Session options update only when explicitly supplied, even on preset switches.
        for field in ("keep_alive", "logger", "log_off"):
            value = getattr(new_config, field)
            setattr(
                updated, field, value if value is not None else getattr(previous, field)
            )

        if previous.model != updated.model:
            self._context.replace_usage(TokenUsage())

        updated.refresh_fingerprints()
        self._config = updated
        if new_config.system_prompt is not None:
            self._context.append(SystemPrompt(new_config.system_prompt))

        if self._client is not None and old_client != self._client_identity(
            updated.primary
        ):
            self._client_stale = True

        if self._runtime is not None:
            self._runtime.logger = self._config.logger

        if not self.log_off:
            self._config.logger.info(f"已更新配置。当前模型: [{self._config.model}]")

    def update_config(
        self,
        *,
        model_params: Params | None = None,
        keep_alive: bool | None = None,
        system_prompt: str | None = None,
        logger: Logger | BlockHandle | None = None,
        log_off: bool | None = None,
    ) -> None:
        """局部更新明确给出的字段，保留候选列表；model_params 只替换首选参数。"""
        self.setting(
            ChatConfig(
                model_params=model_params,
                keep_alive=keep_alive,
                system_prompt=system_prompt,
                logger=logger,
                log_off=log_off,
            )
        )

    @property
    def config(self) -> ChatConfig:
        """Return an isolated configuration snapshot; apply changes with setting()."""
        return copy.deepcopy(self._config)

    @property
    def log_off(self) -> bool:
        return bool(self._config.log_off)

    @log_off.setter
    def log_off(self, value: bool):
        self._config.log_off = value

    @staticmethod
    def _client_identity(candidate: LLMConfig) -> str:
        """返回一个候选连接预设的指纹"""
        return candidate.connection_fingerprint(get_provider(candidate).headers())

    def _update_client(self):
        client_params = self._config.client_params.copy()
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
                self._config.logger.warning(f"关闭过期 LLM 客户端失败: {exc}")

        if self._client is None:
            self._update_client()

        return self._client  # type: ignore

    def add_tools(self, tools: Prompts | list[ToolInput] | ToolInput) -> Prompts:
        """向当前 Context 添加可调用工具定义，返回合并后的工具列表，不执行工具。"""
        self._context.append(tools)
        return self._context.tools

    def add_context(self, new_context: Context | Prompts | BasePrompt) -> Context:
        """直接补充业务上下文，返回当前 Context；不进行网络收发。"""
        self._context.append(new_context)
        return self._context.copy()

    @property
    def model(self) -> dict:
        """返回模型信息"""
        return {
            "model": self._config.model,
            "endpoint": self._config.endpoint,
            "token": self._config.token,
            "provider": self._config.api_provider,
            "parameter": self._config.model_params,
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
        if self._config.max_context is None or self._config.max_context <= 0:
            return -1.0

        return self._context_use / self._config.max_context

    def replace_context(self, new_context: Context) -> None:
        """替换业务 Context；在途结果若仍绑定旧对象，提交时会被拒绝。"""
        self._context = new_context

    async def get_context(self) -> Context:
        """直接读取当前业务 Context；不等待在途请求，也不返回独立副本。"""
        return self._context

    def back_to_last_round(self) -> Prompts:
        return self._context.back2last_round()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await asyncio.sleep(0)
        await self.close()

    def _header_constructor(self):
        return get_provider(self._config.primary).headers()

    def _service(self) -> Runtime:
        """懒初始化共享账本和本 Chat 的通信 Runtime；包含同步 SQLite IO。"""
        if self._runtime is None:
            self._runtime = Runtime(
                StateStore(DATA_DIR / "llm" / "runtime.sqlite3"),
                _GLOBAL_USAGE.ledger,
                PriceCatalog(DATA_DIR / "llm" / "pricing.sqlite3"),
                self._config.logger,
                self._usage_record,
            )

        return self._runtime

    async def _gateway(self, route: LLMConfig | FilesConfig) -> Gateway:
        """复用模型/Files 各自的网络连接，由本 Chat 负责关闭。

        Files 请求自带隔离认证头，其连接池只按网络选项复用，不继承模型代理。
        """
        if isinstance(route, FilesConfig):
            key = fingerprint(["files", route.client_params])
            if key not in self._route_clients:
                params = copy.deepcopy(route.client_params)
                proxies = params.pop("proxies", {})
                params.pop("headers", None)
                self._route_clients[key] = httpx.AsyncClient(
                    mounts=parse_proxies_to_httpx(proxies), **params
                )

            return Gateway(self._route_clients[key], download_options=route.client_params)

        if self._client_identity(route) == self._client_identity(self._config.primary):
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

    async def run(self) -> None:
        """队列消费协程，串行处理任务并通知等待者；由 close 取消收尾。"""
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
                            remote_state="completed" if task.pending else "not_sent",
                            category="context_overflow" if isinstance(exc, ContextOverflowError) else "commit",
                        ),
                    )

            finally:
                task.event.set()
                self._queue.task_done()
                task = None  # 只是为了清除引用，让大对象 GC

    async def post(self, task: AskTask) -> Prompts | None:
        """执行请求并把待提交结果挂到 task；此步骤不修改业务 Context。"""
        if task.cancelled or task.event.is_set():
            return None

        # setting() 安装新对象，不原地修改在途请求使用的配置。
        preset = self._config
        self._running_task_id = task.id
        try:
            service = await run_sync(self._service)  # 数据库和其他组件初始化是有一定时间的
            pending = await service.execute(
                self._context,
                task.prompts,
                preset,
                self._gateway,
                task.timeout,
                context_usage=self._current_context_usage(),
                usage_signature=self._usage_context_signature,
                usage_scope=self._usage_scope,
            )
            task.pending = pending
            task.outcome = CallOutcome(
                "received",
                False,
                pending.call_id,
                pending.result.request_id,
                "completed",
            )
            return pending.result.prompts

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
            self._config.logger.error(f"LLM 请求停止: {type(exc).__name__}")
            return None

    async def _execute_task(self, task: AskTask) -> Prompts | None:
        """执行单个队列任务，检查取消/替换后提交；Context 的跨 Chat 并发由业务控制。"""
        context = self._context
        if self._closed or task.cancelled or task.event.is_set():
            return None

        self._running.clear()
        try:
            async with asyncio.timeout(task.timeout):
                response = await self.post(task)

            if response is not None and not task.cancelled and not task.event.is_set():
                if self._context is not context:
                    raise RuntimeError("Context 已替换，拒绝提交")

                await self._service().commit_async(
                    context, task.pending, lambda: self._publish_committed(task, response)
                )

            return response

        except Exception as exc:
            if not task.committed:
                raise

            logger = self._config.logger
            if logger is not None:
                logger.error(f"LLM 结果已提交，后续收尾失败: {type(exc).__name__}")

            return task.response

        finally:
            if not self._config.keep_alive:
                try:
                    await self._close_clients()

                except Exception as exc:
                    if not task.committed:
                        raise

                    logger = self._config.logger
                    if logger is not None:
                        logger.error(f"LLM 结果已提交，连接清理失败: {type(exc).__name__}")

            self._running_task_id = ""
            self._running.set()

    def _publish_committed(self, task: AskTask, response: Prompts) -> None:
        """内存提交回调：同步发布成功结果，不等待检查点和连接清理。"""
        response._duration = max(0.0, time.time() - float(task.prompts.time))
        task.response = response
        task.committed = True
        task.outcome.status = "completed"
        task.outcome.committed = True
        self._last_usage = response.usage.copy() if response.usage is not None else TokenUsage()
        self._context_use = self._last_usage.total_t
        self._usage_context_signature = self._context.signature
        self._usage_scope = task.pending.route.scope
        task.event.set()

    def _current_context_usage(self) -> TokenUsage | None:
        """借用仍对应当前历史的最近计量；模型/连接来源由实际候选再校验。"""
        if self._usage_context_signature is None:
            return None

        if self._context.signature != self._usage_context_signature:
            return None

        return self._last_usage

    def _check_context_window(self) -> None:
        """单候选真实用量的同步快检；未知计量/多候选留给 Runtime 执行时判断。

        不运行 tokenizer、不查路由 CD、不让出执行权。多候选的实际模型只有
        Runtime 选路后才确定，不能用首选身份否定正常备用模型的真实计量。
        """
        if not self._config.candidates or len(self._config.candidates) != 1:
            return

        route = self._config.primary
        tokens = reported_context_tokens(self._current_context_usage(), route, self._usage_scope)
        if tokens is not None:
            check_context_window(tokens, route, self._config.auto_compress_rate)

    async def post_nowait(self, task: AskTask) -> Prompts | None:
        """兼容直接调度入口，沿用统一提交规则；调用方负责与队列任务互斥。"""
        try:
            task.handle = asyncio.create_task(self._execute_task(task))
            return await task.handle

        except asyncio.CancelledError:
            if not task.cancelled:
                raise

            return None

    async def ask(
        self,
        new_ask: str | Prompts | BasePrompt | list[BasePrompt],
        *,
        task_id: str | None = None,
        queue: bool = True,
        no_wait: bool = False,
        timeout: float | None = None,
    ) -> Prompts | None:
        """
        使用new_ask开始新一轮对话请求，并返回响应
        :param new_ask: Prompts对象
        :param task_id: 指定任务ID
        :param queue: 是否进入任务队列等待其他任务。若为False，则有任务在运行时直接跳过，不会保存Prompts
        :param no_wait: 是否不进入任务队列直接开始，若为True，则在这一个任务结束后直接运行
        :param timeout: 超时/秒
        :return: 响应的 Prompts 对象
        """
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

        if not queue and not no_wait:
            if self._ask_tasks:
                return None

            if not self._running.is_set():
                return None

            if not self._queue.empty():
                return None

        self._check_context_window()

        identifier = task_id or uuid4().hex
        while identifier in self._ask_tasks:
            identifier = uuid4().hex

        task = AskTask(copy.deepcopy(new_ask), timeout, identifier)
        self._ask_tasks[identifier] = task
        self._sequence += 1
        self._queue.put_nowait((0 if no_wait else 1, self._sequence, task))
        try:
            await asyncio.wait_for(task.event.wait(), timeout)
            self._publish_outcome(task)
            if isinstance(task.error, ContextOverflowError):
                raise task.error

            return task.response

        except TimeoutError:
            await self.stop_from_id(identifier)
            self._publish_outcome(task)
            return task.response if task.committed else None

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
    def last_outcome(self) -> CallOutcome | None:
        """读取当前调用协程可见的终态，含远端状态、提交状态及账本 ID。"""
        return self._outcome_var.get() or self._last_outcome

    def copy(self, new_context: bool = True) -> Chat:
        """复制配置创建 Chat；new_context=False 共享原 Context，并发由业务控制。"""
        config = self._config.copy()
        config.system_prompt = (
            None  # Copy the current Context system, not an old setting command.
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

    async def clear(self, system: bool = False, tools: bool = False) -> None:
        """清除历史；system/tools 指定是否连同系统提示和工具一起清除。"""
        self._context.clear(system, tools)

    async def stop_from_id(self, task_id: str) -> None:
        """取消指定未提交任务并等待本地收尾；普通断连不保证服务器停止推理。"""
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

    async def stop_one(self) -> None:
        """取消当前正在执行的任务，不取消其他排队任务。"""
        if self._running_task_id:
            await self.stop_from_id(self._running_task_id)

    async def stop_all(self) -> None:
        """取消全部未提交任务并等待在途协程收尾。"""
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
        if self._runtime is not None:
            await self._runtime.media.close_clients(clients)
            return

        for client in clients:
            await client.aclose()

    async def close(self) -> None:
        """停止队列与未提交任务，关闭自有连接；不关闭进程共享 LLM 线程池。"""
        self._closed = True
        await self.stop_all()
        self._worker_loop.cancel()
        await asyncio.gather(self._worker_loop, return_exceptions=True)
        await self._close_clients()
