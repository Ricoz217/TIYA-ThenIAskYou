"""
agent.py
存放了基本的Agent主体框架
"""

from __future__ import annotations
__version__ = "0.4.2"

import copy
import time
import json
import traceback
import inspect
import asyncio
import contextvars
import weakref
from typing import Callable, Awaitable, Any, TYPE_CHECKING, Literal, TypedDict
from pathlib import Path
from hashlib import blake2b
from functools import partial
from datetime import datetime
from TIYA.logger import get_logger, BlockHandle
from TIYA.re_filter import _checkpoint_filter, _date_dir_filter
from TIYA.config import SETTING_CFG, PROMPTS_DIR, BASE_SKILL_DIR
from TIYA.executor import AGENT_EXECUTOR
from TIYA.params_validator import validate_tool_arguments
from TIYA.utils import atomic_save_json, find_path_by_time_sequence, debug
from TIYA.agent.agent_prompt import AgentPrompt
from TIYA.agent.agent_structure import (AgentTask, AgentScheTask, AgentTaskQueue, AgentHistoryManager, AgentContext,
                                        AgentHistory, Trigger, AgentScheTaskManage, _ts_done, SubTaskError,
                                        translate_task_status, CallbackMode, SummaryHint)
from TIYA.agent.agent_structure import HistoryType as HT
from TIYA.agent.agent_runtime import AgentWorkerManager
from TIYA.agent.agent_skill import SkillManager
from TIYA.agent.web_search import WebSearch
from .agent_scheduler import AsyncSchedulerManager, _AGENTS
from TIYA.time_id import next_time_id
from TIYA.LLM_connect import (Chat, ChatConfig, Prompts, Context, ContextOverflowError, ContextEmptyError,
                              ToolCall, TextPrompt, build_tool_payloads, ToolInput, SystemPrompt,
                              ToolResponse, ImagePrompt, parse_llm_setting)


if TYPE_CHECKING:
    from TIYA.logger import Logger, LogHandle


_log = get_logger()

_TOOL_NOT_FOUND = "#缺失函数"
_PARAMS_ERROR = "#参数错误"

class ToolNotExistError(KeyError):
    """工具不存在"""
    ...


class ToolArgumentsError(ValueError):
    """传入参数错误"""
    ...


class _ToolConfig(TypedDict):
    wait: bool
    callback: Literal["ALWAYS", "EXCEPTION", "NEVER", "CALL"]


class BaseAgent:
    """Agent基类，不实现任何业务逻辑"""
    def __init__(
            self,
            host_id: str,  # 群号或者QQ号
            data_path: Path,
            llm_preset: str = "",
            agent_name: str = "",
            system_prompt: SystemPrompt | AgentPrompt | str | Path = None,
            initiate_prompt: AgentPrompt | Path = None,
            control_timeout: float = -1,
            logger: BlockHandle | Logger = None,
    ):
        if logger is None:
            logger = _log

        if control_timeout <= 0:
            control_timeout = SETTING_CFG.Agent.AgentControlTimeout  # 默认15分钟

        # 元数据
        self.id = str(next_time_id())
        self.display_uid = host_id
        self.name = agent_name
        self._data_path = data_path
        self._data_path.mkdir(parents=True, exist_ok=True)
        self.control_timeout = control_timeout

        # 内置prompt
        self._baseline: None | AgentPrompt = None
        self._initiate_prompts: list[AgentPrompt] = []
        self._system_prompt = system_prompt
        self._init_prompt = initiate_prompt

        # 数据结构
        self._history = AgentHistoryManager(self._data_path)
        self._task_queue = AgentTaskQueue()
        self._schedule_tasks = AgentScheTaskManage()
        self._context = AgentContext()
        self._scheduler = AsyncSchedulerManager(self.id, self._schedule_tasks, self._history.schedule_task_file)
        self._skills = SkillManager(self._data_path, self.register_function, self.uninstall_tools_source, self._upload_image)
        self._websearch = WebSearch()

        # 容器
        self._prompts: dict[str, dict[str, AgentPrompt]] = {}
        self._functions: dict[str, Callable[..., Awaitable | Any]] = {}
        self._functions_name_mapping: dict[str, str] = {}  # 函数名与任务名的映射表
        self._functions_source_mapping: dict[str, dict[str, str]] = {}  # 外部函数名与调用函数名的映射
        self._last_call_ids: dict[str, str] = {}  # {call_id: task_id}
        self._merge_calls = []
        self._pending_upload_image: list[Path] = []
        self.tools: dict[str, Callable[..., Awaitable | Any]] = {}  # 要注入LLM的TOOL INPUT

        # 配置
        self.functions_default_setting: dict[str, _ToolConfig] = {
            "create_task": {"wait": True, "callback": "EXCEPTION"},
            "create_schedule_task": {"wait": True, "callback": "EXCEPTION"},
            "cancel_schedule_task": {"wait": True, "callback": "NEVER"},
            "list_tasks_status": {"wait": True, "callback": "CALL"},
            "get_task_data": {"wait": True, "callback": "CALL"},
            "list_external_tools": {"wait": True, "callback": "CALL"},
            "notepad": {"wait": False, "callback": "ALWAYS"},
            "run_external_tool": {"wait": True, "callback": "EXCEPTION"},
            "load_external_tool": {"wait": True, "callback": "CALL"},
        }

        # 请求和任务循环
        self.control = Chat(logger=logger)  # 主控LLM
        self.task_runner = AgentWorkerManager(self._task_queue, self.callback)

        # 状态位和工具
        self.control_retry_limit: int = SETTING_CFG.Agent.AgentControlRetryLimit  # 默认3
        self._initiated = False
        self._suspending = False
        self._react_times = 0
        self._retry_times = 0  # 如果失败过多自动回滚，防止卡死
        self.event = asyncio.Event()
        self.event.set()
        self.suspend_event = asyncio.Event()
        self.suspend_event.set()
        self._persistence_lock = asyncio.Lock()
        self._cb_merge_lock = asyncio.Lock()
        self._logger = logger
        self._status_handle: LogHandle | None = None
        _AGENTS[self.id] = weakref.proxy(self)

        if llm_preset:
            self.modify_agent(llm_preset=llm_preset)

    # =========================================================
    # HOOK 区，存放各种 HOOK
    # =========================================================

    async def before_initiate(self):
        """预留初始化钩子"""
        ...

    async def after_run(self, run_result: Prompts | None):
        """预留运行一轮后的钩子"""
        ...

    # =========================================================
    # 系统初始化区
    # =========================================================

    async def edit_system_prompt(self, prompt: SystemPrompt | AgentPrompt | str = None):
        if not self._baseline:
            self._logger.warning(f"{self._log_prefix()} 未载入基线提示词")
            system_prompt = ""

        else:
            system_prompt = await self._baseline.get_prompt()

        if prompt is None:
            pass

        elif isinstance(prompt, SystemPrompt):
            system_prompt += prompt.text

        elif isinstance(prompt, AgentPrompt):
            system_prompt += await prompt.get_prompt()

        elif isinstance(prompt, str):
            system_prompt += prompt

        else:
            raise TypeError

        if not system_prompt and self.control.context.system is None:
            raise ValueError(f"{self._log_prefix()} 未载入系统提示词")

        new_system = SystemPrompt(system_prompt)
        self.control.add_context(new_system)

    async def add_system_prompt(self, prompt: SystemPrompt | str | AgentPrompt):
        if not self.control.context.system:
            await self.edit_system_prompt()

        system_prompt = self.control.context.system
        if isinstance(prompt, SystemPrompt):
            if not prompt:
                return

            system_prompt.text += prompt.text

        elif isinstance(prompt, str):
            if not prompt:
                return

            system_prompt.text += prompt

        elif isinstance(prompt, AgentPrompt):
            prompt = await prompt.get_prompt()
            if not prompt:
                return

            system_prompt.text += prompt

        else:
            raise TypeError

        new_system = SystemPrompt(system_prompt.text)
        self.control.add_context(new_system)

    async def reload_system_prompt(self):
        system_prompt = self.get_last_version_prompt("system_prompt")
        old_system = self.control.context.system.text
        if system_prompt is None:
            system_prompt = await self._parse_system_prompt()

        if system_prompt is None:
            self._logger.error(f"{self._log_prefix()} 找不到可用的系统提示词")
            return

        await self.edit_system_prompt(system_prompt)
        new_system = self.control.context.system.text
        self._history.reload_context(data={
            "old_system": old_system,
            "new_system": new_system
        })

    def add_initiate_prompts(self, prompt: AgentPrompt | list[AgentPrompt]):
        if isinstance(prompt, list):
            self._initiate_prompts.extend(prompt)

        else:
            self._initiate_prompts.append(prompt)

    def _load_baseline_prompt(self):
        baseline_dir = PROMPTS_DIR / "baseline"
        if not baseline_dir.is_dir():
            raise FileNotFoundError("基线prompt文件丢失")

        baseline = self.register_prompt(
            "基线系统提示词",
            baseline_dir,
            description="基线系统提示词，无业务逻辑，需要配合具体业务系统提示词使用",
            role="system"
        )
        self._baseline = baseline

    def _load_basic_initiate_prompt(self):
        builtin_initiate_dir = PROMPTS_DIR / "base_initiate"
        if not builtin_initiate_dir.is_dir():
            raise FileNotFoundError("内置初始化prompt文件丢失")

        initiate_prompt = self.register_prompt(
            "内置初始化提示词",
            builtin_initiate_dir,
            description="内置初始化提示词，可按需修改",
            role="user"
        )
        self.add_initiate_prompts(initiate_prompt)

    async def _inject_initiate_prompt(self) -> Prompts:
        injection = Prompts()
        final_prompt = ""
        initiate_prompt = ""
        for prompt in self._initiate_prompts:
            initiate_prompt += await prompt.get_prompt()

        skill_prompt = "\n\n已安装 SKILL : \n\n"
        skill_list = self._skills.list_skills()
        for i, skill in enumerate(skill_list, start=1):
            for k, v in skill.items():
                skill_prompt += f"{i}. {k}: {v}\n"

        if initiate_prompt:
            final_prompt += initiate_prompt

        if skill_list:
            final_prompt += skill_prompt

        if final_prompt:
            injection.append(TextPrompt("user", final_prompt))

        return injection

    async def _send_initiate_post(self):
        if not self.control.context.system:
            await self.edit_system_prompt()

        if not self.control.context.has_content:
            injection = await self._inject_initiate_prompt()
            try:
                await asyncio.wait_for(self.run(injection, from_user=False), timeout=SETTING_CFG.Agent.AgentWorkerTimeout)

            except asyncio.TimeoutError:
                self._logger.error(f"{self._log_prefix()} 自动初始化失败")

    async def _parse_system_prompt(self) -> AgentPrompt | str | None:
        if self._system_prompt is None:
            return None

        elif isinstance(self._system_prompt, Path):
            system_prompt = self.register_prompt(
                name="system_prompt",
                path=self._system_prompt,
                description="系统提示词",
                role="system"
            )

        elif isinstance(self._system_prompt, AgentPrompt):
            system_prompt = await self._system_prompt.get_prompt()

        elif isinstance(self._system_prompt, SystemPrompt):
            system_prompt = self._system_prompt.text

        elif isinstance(self._system_prompt, str):
            system_prompt = self._system_prompt

        else:
            raise TypeError(f"{self._log_prefix()}输入系统提示词类型错误")

        return system_prompt

    async def _parse_initiate_prompt(self) -> AgentPrompt | None:
        if self._init_prompt is None:
            return None

        elif isinstance(self._init_prompt, Path):
            init_prompt = self.register_prompt(
                name="initiate_prompt",
                path=self._init_prompt,
                description="初始化提示词",
                role="user"
            )

        elif isinstance(self._init_prompt, AgentPrompt):
            self._prompts.setdefault(self._init_prompt.name, {})[self._init_prompt.version] = self._init_prompt
            init_prompt = self._init_prompt

        else:
            raise TypeError(f"{self._log_prefix()}输入初始化提示词类型错误")

        return init_prompt

    async def _initiate_agent(
            self,
            system_prompt: AgentPrompt = None,
            initiate_prompt : AgentPrompt = None
    ):
        if system_prompt is None:
            system_prompt = await self._parse_system_prompt()

        if initiate_prompt is None:
            initiate_prompt = await self._parse_initiate_prompt()

        await self.before_initiate()
        self._load_internal_tools()
        self._tools_input()
        self._load_baseline_prompt()
        if not initiate_prompt:
            if not self._initiate_prompts:
                self._load_basic_initiate_prompt()

        else:
            self.add_initiate_prompts(initiate_prompt)

        await self.edit_system_prompt(system_prompt)
        self._load_prompt()
        self._skills.initiate()
        await self._load_task()
        self._load_history()
        self._load_context()
        self.control.replace_context(self._context.current_window)
        self._load_checkpoint()
        await self._load_websearch()
        await self._scheduler.start()
        self._update_scheduled_tasks()
        if not self._last_call_ids:
            self._clear_lost_tool_call()

        if self._check_initiate():
            self._initiated = True

        await self._send_initiate_post()

    def _load_internal_tools(self):
        self.register_function(
            self.create_task,
            display_name="创建普通任务"
        ), self.register_function(
            self.create_schedule_task,
            display_name="创建定时任务"
        ), self.register_function(
            self.cancel_schedule_task,
            display_name="取消定时任务"
        ), self.register_function(
            self.list_tasks_status,
            display_name="获取任务列表"
        ), self.register_function(
            self.get_task_data,
            display_name="获取任务数据"
        ), self.register_function(
            self.list_external_tools,
            display_name="获取外部工具"
        ), self.register_function(
            self.load_external_tool,
            display_name="加载外部工具"
        ), self.register_function(
            self.run_external_tool,
            display_name="调用外部工具"
        ), self.register_function(
            self.notepad,
            display_name="记事本"
        ),
        self.register_function(
            self._empty_tool,
            nickname=_TOOL_NOT_FOUND
        ),
        self.register_function(
            self._params_error,
            nickname=_PARAMS_ERROR
        )

        self.tools[self.create_task.__name__] = self.create_task
        self.tools[self.create_schedule_task.__name__] = self.create_schedule_task
        self.tools[self.cancel_schedule_task.__name__] = self.cancel_schedule_task
        self.tools[self.list_tasks_status.__name__] = self.list_tasks_status
        self.tools[self.get_task_data.__name__] = self.get_task_data
        self.tools[self.list_external_tools.__name__] = self.list_external_tools
        self.tools[self.load_external_tool.__name__] = self.load_external_tool
        self.tools[self.run_external_tool.__name__] = self.run_external_tool
        self.tools[self.notepad.__name__] = self.notepad

    def _tools_input(self):
        tools = [ToolInput(tool) for tool in self.tools.values()]
        self.control.add_tools(tools)
        self._context.current_window.extend(tools)

    def _check_initiate(self) -> bool:
        for tool in self.control.context.tools:  # type: ToolInput
            if tool.function_name not in self._functions:
                self._logger.error(f"{self._log_prefix()}有内置工具缺失，初始化失败")
                return False

        return True

    @property
    def running(self) -> bool:
        return not self.event.is_set()

    @property
    def suspending(self) -> bool:
        return self._suspending

    async def start(
            self,
            system_prompt: AgentPrompt = None,
            initiate_prompt: AgentPrompt = None
    ):
        await self._initiate_agent(system_prompt, initiate_prompt)

    async def shutdown(self):
        self._initiated = False
        await self.save_agent_async(save_context_history=True)
        await self.control.close()
        await self._scheduler.shutdown()
        await self.task_runner.shutdown()
        _AGENTS.pop(self.id, None)

    def modify_agent(self, *, name: str = "", llm_preset: str = ""):
        if not (name or llm_preset):
            return

        if name:
            self.name = name

        if llm_preset:
            new_setting = parse_llm_setting(llm_preset)
            new_setting.keep_alive = True
            self.control.setting(new_setting)

    # =========================================================
    # 持久化，存储和读取本地文件还原Agent
    # =========================================================

    def _save_history(self, dt: datetime = None, snapshot: list[AgentHistory] = None):
        if dt is None:
            dt = datetime.now()

        save_path = self._history.history_file
        history = snapshot or list(self._history.history)
        dump_content = {
            "update": f"{dt:%x_%X}",
            "description": "此文件为Agent的操作历史",
            "version": "0.1.0",
            "data": [his.to_dict() for his in history]
        }
        try:
            atomic_save_json(dump_content, save_path)

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 保存历史记录出错: {E}")
            self._logger.debug(traceback.format_exc())

    def _save_task(self, dt: datetime = None):
        if dt is None:
            dt = datetime.now()

        save_path = self._history.task_file
        dump_content = {
            "update": f"{dt:%x_%X}",
            "description": "此文件为Agent任务列表",
            "version": "0.1.0",
            "data": {
                "tasks": self._task_queue.to_dict(),
                "scheduler": self._schedule_tasks.to_dict()
            }
        }
        try:
            atomic_save_json(dump_content, save_path)

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 保存任务列表出错: {E}")
            self._logger.debug(traceback.format_exc())

    def _save_schedule_task(self):
        try:
            self._scheduler.save_task()

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 保存定时任务出错: {E}")
            self._logger.debug(traceback.format_exc())

    def _save_current_context(self, dt: datetime = None):
        if dt is None:
            dt = datetime.now()

        save_path = self._history.current_context_file
        dump_content = {
            "update": f"{dt:%x_%X}",
            "description": "此文件为Agent的当前上下文",
            "version": "0.1.0",
            "data": self._context.current_window.to_dict()
        }
        try:
            atomic_save_json(dump_content, save_path)

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 保存当前上下文出错: {E}")
            self._logger.debug(traceback.format_exc())

    def _save_context_history(self, dt: datetime = None, snapshot: dict[str, Context] = None):
        if dt is None:
            dt = datetime.now()

        context_history = snapshot or self._context.window_history.copy()
        save_path = self._history.context_history_file
        dump_content = {
            "update": f"{dt:%x_%X}",
            "description": "此文件为Agent的历史上下文",
            "version": "0.1.0",
            "data": {key: context.to_dict() for key, context in context_history.items()}
        }
        try:
            atomic_save_json(dump_content, save_path)

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 保存历史上下文出错: {E}")
            self._logger.debug(traceback.format_exc())

    def _save_checkpoint(self, dt: datetime = None, snapshot: dict = None):
        if dt is None:
            dt = datetime.now()

        if snapshot is not None:
            pending_tasks: dict[str, AgentTask] = snapshot["tasks"]
            pending_accomplished: dict[str, AgentTask] = snapshot["accomplished"]
            pending_history: list[AgentHistory] = snapshot["history"]
            pending_context_history: dict[str, Context] = snapshot["context_history"]

        else:
            pending_tasks = self._task_queue.tasks.copy()
            pending_accomplished = self._task_queue.accomplished_task.copy()
            pending_history = list(self._history.history)
            pending_context_history = self._context.window_history.copy()

        tasks = {
            "tasks": {k: v.status.value for k, v in pending_tasks.items()},
            "accomplished": {k: v.status.value for k, v in pending_accomplished.items()}
        }
        history = {
            "history": [h.id for h in pending_history]
        }
        context = {
            "current": self._context.current_window.to_dict(),
            "history": list(pending_context_history.keys())
        }
        dump_content = {
            "update": f"{dt:%x_%X}",
            "description": "此文件为Agent的checkpoint",
            "version": "0.1.0",
            "data": {
                "tasks": tasks,
                "history": history,
                "context": context
            }
        }
        save_path = self._data_path / "checkpoint" / f"{dt:%y_%m_%d}"
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / "checkpoint.json"
        try:
            checkpoint = atomic_save_json(dump_content, save_file, time_info=True)
            if checkpoint is not None:
                self._history.checkpoint.append((str(dt.timestamp()), checkpoint))

        except Exception as E:
            self._logger.error(f"Agent {self.name}[{self.display_uid}]保存CKPT出错: {E}")
            self._logger.debug(traceback.format_exc())

    def _save_websearch(self):
        save_file = self._data_path / "websearch.json"
        atomic_save_json(self._websearch.to_dict(), save_file)

    def save_agent(self, *, save_context_history: bool = False):
        dt = datetime.now()
        self._save_task(dt)
        self._save_schedule_task()
        self._save_history(dt)
        self._save_current_context(dt)
        if save_context_history:
            self._save_context_history(dt)

        self._save_websearch()

    async def save_agent_async(self, *, save_context_history: bool = False):
        async with self._persistence_lock:
            loop = asyncio.get_running_loop()
            save = partial(self.save_agent, save_context_history=save_context_history)

            # noinspection PyTypeChecker
            await loop.run_in_executor(AGENT_EXECUTOR, save)

    async def _save_checkpoint_async(self):
        async with self._persistence_lock:
            snapshot = {
                "tasks": self._task_queue.tasks.copy(),
                "accomplished": self._task_queue.accomplished_task.copy(),
                "history": list(self._history.history),
                "context_history": self._context.window_history.copy()
            }
            loop = asyncio.get_running_loop()
            func = partial(self._save_checkpoint, snapshot=snapshot)

            # noinspection PyTypeChecker
            await loop.run_in_executor(AGENT_EXECUTOR, func)

    async def _load_task(self):
        task_path = self._history.task_file
        if not task_path.is_file():
            return

        try:
            with task_path.open('r', encoding="utf-8") as f:
                load_content: dict = json.loads(f.read())

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 读取任务列表出错: {E}")
            self._logger.debug(traceback.format_exc())
            return

        else:
            await self._task_queue.from_dict(load_content["data"]["tasks"], self._functions)
            self._schedule_tasks.from_dict(load_content["data"]["scheduler"], self._functions)

    # 已弃用
    def _load_schedule_task(self):
        try:
            self._scheduler.load_task()

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 读取定时任务出错: {E}")
            self._logger.debug(traceback.format_exc())

    def _load_history(self):
        save_file = self._history.history_file
        if not save_file.is_file():
            return

        try:
            with save_file.open('r', encoding="utf-8") as f:
                load_content: dict = json.loads(f.read())

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 读取历史列表出错: {E}")
            self._logger.debug(traceback.format_exc())
            return

        else:
            data: list = load_content.get("data", [])
            for his in data:
                new_his = AgentHistory.from_dict(his)
                self._history.history.append(new_his)

    def _load_context(self):
        current_path = self._history.current_context_file
        history_path = self._history.context_history_file
        context_path = self._history.context_file
        has_current = current_path.is_file()
        has_history = history_path.is_file()
        if not (has_current or has_history or context_path.is_file()):
            return

        try:
            legacy_context = None
            if (not has_current or not has_history) and context_path.is_file():
                with context_path.open('r', encoding="utf-8") as f:
                    legacy_context = json.loads(f.read())["data"]

            if has_current:
                with current_path.open('r', encoding="utf-8") as f:
                    current_context = json.loads(f.read())["data"]

            elif legacy_context is not None:
                current_context = legacy_context["current"]

            else:
                current_context = None

            if current_context is not None:
                self._context.current_window = Context.from_dict(current_context, self._functions)

            if has_history:
                with history_path.open('r', encoding="utf-8") as f:
                    history_context = json.loads(f.read())["data"]

            elif legacy_context is not None:
                history_context = legacy_context["history"]

            else:
                history_context = None

            if history_context is not None:
                self._context.window_history = {
                    key: Context.from_dict(context, self._functions)
                    for key, context in history_context.items()
                }

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 读取上下文出错: {E}")
            self._logger.debug(traceback.format_exc())

    def _load_prompt(self):
        ...

    async def _load_websearch(self):
        source = BASE_SKILL_DIR / "websearch"
        self.register_function(
            self._websearch.search,
            source=source,
            source_name="websearch",
            display_name="WebSearch"
        )
        self.register_function(
            self._websearch.extract,
            source=source,
            source_name="websearch",
            display_name="WebCrawl"
        )
        self.register_function(
            self._websearch.url_map,
            source=source,
            source_name="websearch",
            display_name="WebMap"
        )
        self.register_function(
            self._websearch.raw_crawl,
            source=source,
            source_name="websearch",
            display_name="RawCrawl"
        )
        save_file = self._data_path / "websearch.json"
        if not save_file.is_file():
            return

        try:
            data = json.loads(save_file.read_text(encoding="utf-8"))

        except json.JSONDecodeError:
            return

        else:
            await self._websearch.load_dict(data)

    def _load_checkpoint(self):
        ckpt_dir = self._history.checkpoint_path
        newest_dir = find_path_by_time_sequence(ckpt_dir, _date_dir_filter, "%y_%m_%d")
        if not newest_dir:
            return

        newest_dir = newest_dir[0]
        checkpoints = find_path_by_time_sequence(newest_dir, _checkpoint_filter, "%y%m%d_%H%M%S", 5)
        if checkpoints:
            for ckpt in checkpoints:
                match = _checkpoint_filter.search(ckpt.name)
                dt = datetime.strptime(match.group(1), "%y%m%d_%H%M%S")
                self._history.checkpoint.append((str(dt.timestamp()), ckpt))

    def _restore_last_checkpoint(self):
        if not self._initiated:
            return

        if not self._history.checkpoint:
            return
        ckpt = self._history.checkpoint[-1]
        date = ckpt[0]
        path = ckpt[1]
        dt = datetime.fromtimestamp(float(date))
        self._restore_checkpoint(path)
        self._logger.info(f"{self._log_prefix()} 已回退到CKPT: {dt:%Y-%m-%d %H:%M:%S}")

    def _restore_checkpoint(self, checkpoint: Path):
        if not self._initiated:
            return

        if not self._initiated:
            return

        if not checkpoint.is_file():
            return

        try:
            with checkpoint.open('r', encoding="utf-8") as f:
                load_content: dict = json.loads(f.read())

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 读取CKPT出错: {E}")
            self._logger.debug(traceback.format_exc())
            return

        else:
            tasks = load_content["data"].get("tasks", {})
            context = load_content["data"].get("context", {})
            history = load_content["data"].get("history", {})

            # 恢复任务
            self._task_queue.from_checkpoint(tasks)

            # 恢复上下文
            self._context.from_checkpoint(context, self._functions)

            # 恢复历史
            self._history.from_checkpoint(history)

    # =========================================================
    # IO相关，请求、处理响应、TOOLS、PROMPTS
    # =========================================================

    def register_function(
            self,
            function: Callable | Awaitable,
            source: Path = None,
            source_name: str = None,
            *,
            nickname: str = "",
            display_name: str = ""
    ) -> str:
        if nickname:
            function_name = nickname

        else:
            function_name = function.__name__

        if source is not None:
            h = blake2b(function_name.encode(), digest_size=4)
            call_name = function_name + f"_{h.hexdigest()}"

        else:
            call_name = function_name

        if source_name is not None:
            self._functions_source_mapping.setdefault(function_name, {})[source_name] = call_name
            if source_name in self._skills.skills:
                skill = self._skills.skills[source_name]
                skill.tools_name_mapping[call_name] = function_name  # 反向映射

        if call_name in self._functions:
            _log.warning(f"{self._log_prefix()} 函数[{call_name}]重复注册")

        self._functions[call_name] = function
        if display_name:
            self._functions_name_mapping[call_name] = display_name

        return call_name

    def register_prompt(
            self,
            name: str,
            path: Path,
            *,
            description: str = "",
            role: str = "",
            default_input: dict = None
    ) -> AgentPrompt:
        if default_input is None:
            default_input = {}

        new_prompt = AgentPrompt(
            name=name,
            data_path=path,
            description=description,
            role=role,
            input_function=default_input
        )
        if name in self._prompts and new_prompt.version in self._prompts[name]:
            _log.warning(f"{self._log_prefix()} Prompt [{name}], version={new_prompt.version}重复注册")

        self._prompts.setdefault(name, {})[new_prompt.version] = new_prompt
        return new_prompt

    def uninstall_tools_source(self, source_name: str):
        remapping = {}
        for k, v in self._functions_source_mapping.items():
            for n, c in v.items():
                remapping.setdefault(n, []).append({k: c})

        if source_name not in remapping:
            return

        tools_set = remapping[source_name]
        for tool in tools_set:
            k = next(iter(tool))
            c = next(iter(tool.values()))
            self._functions.pop(c, None)
            self._functions_name_mapping.pop(c)
            self._functions_source_mapping[k].pop(source_name, None)
            if not self._functions_source_mapping[k]:
                self._functions_source_mapping.pop(k, None)

    def get_last_version_prompt(self, name: str) -> AgentPrompt | None:
        if name not in self._prompts:
            return None

        versions = self._prompts[name].copy()
        overall = versions.pop("#UNDEFINE", None)
        if versions:
            return versions[max(versions.keys())]

        else:
            return overall

    def get_tool_call_result(self, force_stop=False) -> Prompts:
        result = Prompts()
        if not self._last_call_ids:
            return result

        last_call_mirror = self._last_call_ids.copy()
        lookup_tasks = self._get_task_lookup(True)
        for call_id, task_id in last_call_mirror.items():
            if task_id not in lookup_tasks:
                result.append(
                    ToolResponse(
                        function_name=_TOOL_NOT_FOUND,
                        call_id=call_id,
                        response={
                            "system": {
                                "status": "执行失败",
                                "messages": "tool任务丢失，执行失败"
                            }
                        }
                    )
                )
                continue

            task = lookup_tasks[task_id]
            if force_stop:
                result.append(
                    ToolResponse(
                        function_name=next(iter(task.function)),
                        call_id=call_id,
                        response={
                            "system": {
                                "status": translate_task_status("EXPIRED"),
                                "messages": "已达到REACT自请求上限，强制挂起"
                            }
                        }
                    )
                )

            if task.status not in _ts_done:
                result.append(
                    ToolResponse(
                        function_name=next(iter(task.function)),
                        call_id=call_id,
                        response={
                            "system": {
                                "status": translate_task_status(task.status),
                                "messages": "tool尚未执行完毕"
                            }
                        }
                    )
                )

            elif task.status.value == "FAILED":
                result.append(
                    ToolResponse(
                        function_name=next(iter(task.function)),
                        call_id=call_id,
                        response={
                            "system": {
                                "status": translate_task_status(task.status),
                                "messages": "tool执行失败"
                            }
                        }
                    )
                )

            elif task.status.value == "CANCELED":
                result.append(
                    ToolResponse(
                        function_name=next(iter(task.function)),
                        call_id=call_id,
                        response={
                            "system": {
                                "status": translate_task_status(task.status),
                                "messages": "已取消tool执行"
                            }
                        }
                    )
                )

            elif task.status.value == "TIMEOUT":
                result.append(
                    ToolResponse(
                        function_name=next(iter(task.function)),
                        call_id=call_id,
                        response={
                            "system": {
                                "status": translate_task_status(task.status),
                                "messages": "tool执行超时"
                            }
                        }
                    )
                )

            elif task.status.value == "EXPIRED":
                result.append(
                    ToolResponse(
                        function_name=next(iter(task.function)),
                        call_id=call_id,
                        response={
                            "system": {
                                "status": translate_task_status(task.status),
                                "messages": "tool执行失败"
                            }
                        }
                    )
                )

            elif task.status.value == "EXCEPTION":
                result.append(
                    ToolResponse(
                        function_name=next(iter(task.function)),
                        call_id=call_id,
                        response={
                            "system": {
                                "status": translate_task_status(task.status),
                                "messages": [f"{e.__class__}: {e}" for e in task.exceptions]
                            }
                        }
                    )
                )

            elif task.status.value == "DONE":
                result.append(
                    ToolResponse(
                        function_name=next(iter(task.function)),
                        call_id=call_id,
                        response=task.results.get("result", None)
                    )
                )

            else:
                raise ValueError("未知任务状态")

        return result

    def _detach_tool_calls_for_context_reset(self, reason: str) -> str:
        """解除即将失效的原生 ToolCall 关联，同时保留后台任务。"""
        if not self._last_call_ids:
            return ""

        call_mapping = self._last_call_ids.copy()
        lookup_tasks = self._get_task_lookup(schedule=True)
        detached_tasks = []
        for call_id, task_id in call_mapping.items():
            task = lookup_tasks.get(task_id)
            if task is None:
                tool_name = "unknown"
                status = "LOST"

            else:
                tool_name = next(iter(task.function))
                status = task.status.value

            detached_tasks.append({
                "call_id": call_id,
                "task_id": task_id,
                "tool_name": tool_name,
                "status": status
            })

        self._last_call_ids.clear()
        self._logger.warning(
            f"Context即将变更，已解除 {len(detached_tasks)} 个未结算 ToolCall 的原生响应关联"
        )
        payload = json.dumps(detached_tasks, ensure_ascii=False, indent=2)
        return (
            "# Context 更换时解除的 TOOL CALL\n\n"
            f"原因：{reason}\n\n"
            "以下 ToolCall 的原生 ToolResponse 通道已经失效，但后台任务没有被取消。"
            "不要继续等待旧 ToolResponse；需要结果时请调用 "
            "`get_task_data(task_id, wait=False)`，也可以先调用 `list_tasks_status` 查看状态。\n\n"
            f"```json\n{payload}\n```\n\n---\n\n"
        )

    # =========================================================
    # 主入口
    # =========================================================

    async def run(self, prompts: Prompts | TextPrompt, from_user=True) -> Prompts | None:
        """Agent主请求入口，返回主控返回的原始内容，但已自带处理逻辑"""
        if not self._initiated:
            return None

        if not self.control.context.system:
            await self.edit_system_prompt()

        if self._suspending:
            if not from_user:
                await self.suspend_event.wait()

        await self.event.wait()
        self.event.clear()
        try:
            self._update_status()
            react_limit = SETTING_CFG.Agent.MaxReactTimes  # 默认20
            react_noted = SETTING_CFG.Agent.NoteReactTimes  # 默认15
            if from_user:
                self._suspending = False
                self.suspend_event.set()
                self._react_times = 0

            if self._react_times >= react_noted:
                note = "\n\n> 已达到REACT上限， 立即挂起任务、停止TOOL CALL并等待用户后续指令"
                if isinstance(prompts, TextPrompt):
                    prompts.text += note

                elif isinstance(prompts, Prompts):
                    if prompts and isinstance(prompts.prompts[-1], TextPrompt):
                        prompts.prompts[-1].text += note  # type: ignore

                    else:
                        prompts.append(TextPrompt("user", note))

            if self._retry_times >= 3:
                try:
                    self._context.current_window.back2last_round()

                except ContextEmptyError:
                    self._logger.error(f"{self._log_prefix()} 无法请求，请检查设置")

                else:
                    detached_notice = self._detach_tool_calls_for_context_reset("连续请求失败，自动回退上一轮对话")
                    if detached_notice:
                        prompts = Prompts(TextPrompt("user", detached_notice), prompts)

                    self._logger.warning(f"{self._log_prefix()} 重试达到上限，已自动回退上一轮对话")
                    self._retry_times = 0

            prompts = self.get_tool_call_result(True if self._react_times >= react_limit - 1 else False) + prompts
            for img in self._pending_upload_image:
                prompts.append(ImagePrompt(role="user", image=img))

            self._pending_upload_image.clear()
            if not prompts or self._react_times >= react_limit:
                self.event.set()
                return None

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 回滚上下文失败: {E}")
            self._logger.debug(traceback.format_exc())
            self.event.set()
            return None

        try:
            retry = 0
            response = None
            while response is None and retry <= self.control_retry_limit and not self._suspending:
                retry += 1
                try:
                    _log.debug(str(prompts))
                    response = await self.control.ask(prompts, timeout=self.control_timeout)

                except ContextOverflowError:
                    prompts = await self.compress_context(prompts)

                except Exception as E:
                    self._logger.error(f"{self._log_prefix()} 主控请求错误: {E}")
                    self._logger.debug(traceback.format_exc())

            if response is None:
                if self._suspending:
                    self.event.set()
                    return None

                self._retry_times += 1
                self._logger.error(f"{self._log_prefix()} 请求失败！")
                self.event.set()
                return None

            self._last_call_ids.clear()  # 只有成功才清除call_id
            if from_user:
                self._history.user_ask(prompts.to_dict())

            else:
                self._react_times += 1
                self._history.llm_ask(prompts.to_dict())

        except asyncio.CancelledError:
            self.event.set()
            raise

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 主控请求过程中发生未捕获异常: {E}")
            self._logger.debug(traceback.format_exc())
            self.event.set()
            return None

        try:
            _log.debug(str(response))
            self._history.llm_response(response.to_dict())
            await self.handle_response(response)
            return response

        except Exception as E:
            self._logger.error(f"{self._log_prefix()} 处理LLM响应错误: {E}")
            self._logger.debug(traceback.format_exc())
            return response

        finally:
            self.event.set()
            asyncio.create_task(self.after_run(response))

    async def handle_response(self, response: Prompts):
        text_resp = []
        wait_list: list[AgentTask] = []
        for prompt in response:
            if isinstance(prompt, ToolCall):
                function = self._functions.get(prompt.function_name, None)
                if function is None:
                    new_task = await self.add_task(
                        _TOOL_NOT_FOUND,
                        call_id=prompt.call_id,
                        description="缺失工具的代替任务",
                        pin=True,
                        callback="ALWAYS",
                        wait=True
                    )
                    self._last_call_ids[prompt.call_id] = new_task.id
                    continue

                check = validate_tool_arguments(function, prompt.arguments)
                if not check.ok:
                    new_task = await self.add_task(
                        _PARAMS_ERROR,
                        params={
                            "function_name": prompt.function_name,
                            "errors": [e.to_dict() for e in check.issues]
                        },
                        call_id=prompt.call_id,
                        description="参数校验未通过",
                        pin=True,
                        callback="ALWAYS",
                        wait=True
                    )
                    wait_list.append(new_task)
                    self._last_call_ids[prompt.call_id] = new_task.id
                    continue

                callback = "NEVER"
                wait = False
                name = self._functions_name_mapping.get(prompt.function_name, "")
                if prompt.function_name in self.functions_default_setting:
                    callback = self.functions_default_setting[prompt.function_name]["callback"]
                    wait = self.functions_default_setting[prompt.function_name]["wait"]

                new_task = await self.add_task(
                    function_name=prompt.function_name,
                    name=name,
                    params=prompt.arguments,
                    call_id=prompt.call_id,
                    callback=callback,  # type: ignore
                    wait=wait
                )
                if new_task is not None:
                    self._last_call_ids[prompt.call_id] = new_task.id
                    if wait:
                        wait_list.append(new_task)

            elif isinstance(prompt, TextPrompt):
                text_resp.append(prompt.text)

        if text_resp:
            self._logger.info(f"Agent主控：{'\n'.join(text_resp)}")

        if wait_list:
            await asyncio.gather(*[t.event.wait() for t in wait_list], return_exceptions=True)

        await self.save_agent_async()

    async def callback(self, run: bool = False, call_only: bool = False, call_from: str = None):
        # self._logger.debug(f"来自任务[{call_from}]的callback")
        self._update_status()
        if not call_from:
            return

        lookup_task = self._get_task_lookup(schedule=True)
        if call_from not in lookup_task:
            return

        task = lookup_task[call_from]
        self._history.task_done(task.to_dict(), task.id)
        if run:
            result = task.to_llm()
            result.pop("last_fire_at", None)
            result.pop("next_run_at", None)

            if call_only or task.call_id in self._last_call_ids:
                asyncio.create_task(self._merge_callback({}))
                return

            else:
                asyncio.create_task(self._merge_callback(result))
                return

    # =========================================================
    # 任务相关
    # =========================================================

    async def add_task(
            self,
            function_name: str,
            params: dict = None,
            *,
            name: str = "",
            description: str = "",
            call_id: str = "",
            timeout: float = 0,
            pin: bool = False,
            retry: int = 0,
            dependents: set[AgentTask] = None,
            data: dict = None,
            callback: Literal["ALWAYS", "EXCEPTION", "NEVER", "CALL"] = "NEVER",
            wait: bool = False
    ) -> AgentTask | None:
        function = self._functions.get(function_name, None)
        if function is None:
            return None

        if params is None:
            params = {}

        if pin:
            priority = 0

        else:
            priority = 50

        if retry == 0:
            retry = SETTING_CFG.Agent.AgentTaskRetryLimit  # 默认3次

        if dependents is None:
            dependents = set()

        else:
            dependents = {t.id for t in dependents}

        if data is None:
            data = {}

        new_task = AgentTask(
            function={function_name: function},
            input_params=params.copy(),
            name=name,
            description=description,
            call_id=call_id,
            priority=priority,
            timeout=timeout,
            retry_limit=retry,
            dependents=dependents,
            data=data.copy(),
            callback=CallbackMode(callback),
            is_wait=wait
        )
        await self.task_runner.adjust_worker()
        await self._task_queue.add(new_task)
        self._history.task_create(new_task.to_dict(), new_task.id)
        return new_task

    def _update_scheduled_tasks(self):
        jobs = self._scheduler.list_job()
        expired_jobs = self._schedule_tasks.check_scheduled(jobs)
        if expired_jobs:
            for job_id in expired_jobs:
                self._logger.warning(f"{self._log_prefix()} 定时任务已丢失: {job_id}")
                self._scheduler.remove_job(job_id)

    def add_schedule_task(
            self,
            function_name: str,
            trigger: dict,
            params: dict = None,
            *,
            name: str = "",
            description="",
            timeout: float = 0,
            retry: int = 0,
            data: dict = None,
            callback: Literal["ALWAYS", "EXCEPTION", "NEVER", "CALL"] = "NEVER",
            wait: bool = False
    ) -> AgentScheTask | None:
        function = self._functions.get(function_name, None)
        if function is None:
            return None

        if params is None:
            params = {}

        if retry == 0:
            retry = SETTING_CFG.Agent.AgentTaskRetryLimit  # 默认3次

        if data is None:
            data = {}

        new_task = AgentScheTask(
            function={function_name: function},
            input_params=params,
            name=name,
            description=description,
            timeout=timeout,
            retry_limit=retry,
            data=data.copy(),
            callback=CallbackMode(callback),
            is_wait=wait
        )

        new_task.trigger_spec = Trigger(**trigger)
        self._scheduler.add_job(new_task)
        self._schedule_tasks.tasks[new_task.id] = new_task
        self._update_scheduled_tasks()
        self._history.schedule_task_create(new_task.to_dict(), new_task.id)
        return new_task

    def remove_schedule_task(self, task: AgentScheTask):
        self._scheduler.remove_job(task)
        self._schedule_tasks.tasks.pop(task.id, None)
        self._update_scheduled_tasks()

    def _callback(self, task: AgentScheTask):
        self._history.schedule_task_fire(task.to_dict(), task.id)
        if task.callback is CallbackMode.NEVER:
            asyncio.create_task(self.callback(run=False, call_from=task.id))
            return

        elif task.callback is CallbackMode.EXCEPTION:
            if task.status.value not in {"EXCEPTION", "TIMEOUT", "FAILED"}:
                asyncio.create_task(self.callback(run=False, call_from=task.id))
                return

        elif task.callback is CallbackMode.CALL:
            asyncio.create_task(self.callback(run=True, call_only=True, call_from=task.id))
            return

        asyncio.create_task(self.callback(run=True, call_from=task.id))

    async def _merge_callback(self, result: dict):
        async with self._cb_merge_lock:
            if self._merge_calls:
                self._merge_calls.append(result)
                return

            self._merge_calls.append(result)

        await self.suspend_event.wait()  # 等待手动挂起
        await self.event.wait()
        await asyncio.sleep(0.5)  # 等待一次0.5s，抛出控制权，让排队的用户信息可以发送
        await self.event.wait()
        async with self._cb_merge_lock:
            calls = [c for c in self._merge_calls if c]
            self._merge_calls.clear()

        if not calls:
            if not self._last_call_ids:
                return

            prompt = Prompts()

        else:
            payload = json.dumps(calls, ensure_ascii=False, indent=2, sort_keys=True)
            payload_text = f"来自任务的callback请求:  \n```json\n{payload}\n```"
            prompt = TextPrompt(
                role="user",
                text=payload_text
            )

        try:
            await asyncio.wait_for(self.run(prompt, from_user=False), timeout=SETTING_CFG.Agent.AgentWorkerTimeout)

        except asyncio.TimeoutError:
            self._logger.error(f"{self._log_prefix()} 回调请求超时")

    async def run_schedule_task(self, task_id):
        if not self._initiated:
            return

        if task_id not in self._schedule_tasks.tasks:
            self._logger.warning(f"{self._log_prefix()} 定时任务已丢失: {task_id}")
            return

        task = self._schedule_tasks.tasks[task_id]
        self._logger.debug(f"正在运行定时任务[{task.name if task.name else 'untitled'}]")
        timeout = task.timeout if task.timeout > 0 else self._schedule_tasks.timeout
        function = next(iter(task.function.values()))
        task.attempt = 0
        task.exceptions.clear()
        task.results.clear()
        max_attempts = task.retry_limit if task.retry_limit > 0 else 1
        try:
            while task.attempt < max_attempts:
                token = self._task_queue.task_context.set(task.id)
                try:
                    task.set_running()
                    task.attempt += 1
                    if inspect.iscoroutinefunction(function):
                        result = await asyncio.wait_for(function(**task.input_params), timeout=timeout)

                    else:
                        ctx = contextvars.copy_context()
                        warp_function = partial(function, **task.input_params)
                        loop = asyncio.get_running_loop()
                        # noinspection PyTypeChecker
                        result = await asyncio.wait_for(loop.run_in_executor(AGENT_EXECUTOR, ctx.run, warp_function),
                                                        timeout=timeout)

                except asyncio.TimeoutError:
                    task.set_timeout()

                except asyncio.CancelledError:
                    task.set_cancel()
                    self._callback(task)
                    return

                except Exception as E:
                    _log.error(E)
                    task.set_exception(E)

                else:
                    task.results["result"] = result
                    if task.is_once():
                        task.set_done()

                    else:
                        task.set_ready()

                    self._callback(task)
                    return

                finally:
                    self._task_queue.task_context.reset(token)

            task_name = task.name if task.name else task.id
            self._logger.error(
                f"定时任务[{task_name}]执行失败，已达到重试上限[{max_attempts}]"
            )
            self._callback(task)

        finally:
            task.finish_fire()

    def get_self_id(self) -> str:
        """for agent function"""
        task_id = self._task_queue.task_context.get()
        return task_id

    def get_self_data(self) -> dict:
        """for agent function"""
        task_id = self._task_queue.task_context.get()
        lookup_task = self._get_task_lookup(True)
        task: AgentTask = lookup_task.get(task_id, None)
        if task is None:
            return {}

        return task.data.copy()

    def get_dependent_data(self) -> dict:
        """for agent function"""
        task_id = self._task_queue.task_context.get()
        lookup_task = self._get_task_lookup(True)
        task: AgentTask | None = lookup_task.get(task_id, None)
        if task is None:
            return {}

        dependent_data = {}
        for did in task.dependents:
            d_task: AgentTask | None = lookup_task.get(did, None)
            if d_task is None:
                continue

            task_name = d_task.name if d_task.name else d_task.id
            dependent_data[task_name] = d_task.data.copy()

        return dependent_data

    # =========================================================
    # 指令和辅助函数
    # =========================================================

    def _clear_lost_tool_call(self):
        last_round = self.control.context.last_round
        if not last_round:
            return

        lookup_tasks = self._get_task_lookup(schedule=True)
        for prompt in last_round:
            if isinstance(prompt, ToolCall):
                call_id = prompt.call_id
                for task in reversed(lookup_tasks.values()):
                    if task.call_id == call_id:
                        self._last_call_ids[call_id] = task.id
                        break

                if call_id not in self._last_call_ids:
                    self._last_call_ids[call_id] = "#LOST"

    def tasks2text(self):
        tasks = self.get_tasks_list()
        text = ""
        schedule = tasks.get("schedule", [])
        schedule = [t for t in schedule if t["task_name"] != "untitled"]
        if schedule:
            text += f"{'定时任务':^50}\n"
            for t in schedule:
                text += f"{t['task_name']:<15} run at {t['next_run_at']:>27}\n"

        running = tasks.get("running", [])
        running = [t for t in running if t["task_name"] != "untitled"]
        if schedule and running:
            text += '-' * 50
            text += '\n'

        if running:
            text += f"{'队列中任务':^35}\n"
            for t in running:
                text += f"{t['task_name']:<15} status: {t['status']:>26}\n"

        text.strip('\n')
        return text

    def _update_status(self):
        if not isinstance(self._logger, BlockHandle):
            return

        if self._status_handle is None:
            self._status_handle = self._logger.status()

        self._update_scheduled_tasks()
        self._status_handle.update(self.tasks2text())

    def _log_prefix(self) -> str:
        return  f"Agent {self.name}[{self.display_uid}] "

    def _get_task_lookup(self, schedule: bool = False) -> dict[str, AgentTask | AgentScheTask]:
        lookup = {**self._task_queue.tasks, **self._task_queue.accomplished_task}
        if schedule:
            lookup.update(self._schedule_tasks.tasks)

        return lookup

    @staticmethod
    async def _empty_tool():
        """当LLM调用了丢失的工具时，返回一个提示"""
        return "TOOL NOT FOUND"

    @staticmethod
    async def _params_error(function_name: str, errors: list[dict]):
        return f"函数[{function_name}]参数有误: \n{errors}"

    async def interrupt_control(self):
        self._suspending = True
        self.suspend_event.clear()
        await self.control.stop_one()
        self.event.set()

    def _reverse_call_name(self, call_name: str):
        lookup_mapping = {}
        for source in self._skills.skills.values():
            lookup_mapping.update(source.tools_name_mapping)

        function_name = lookup_mapping.get(call_name, "")

        if not function_name:
            function_name = call_name

        return function_name

    def _get_call_name(self, tool_name: str, source_name: str) -> str:
        call_name = self._functions_source_mapping.get(tool_name, {}).get(source_name, "#NO_MATCH")
        return call_name

    def _summary_mapping(self, key_step: list, data: Any, event_time: float, detail=False) -> dict | str:
        """
        用于压缩context的需要LLM总结部分映射表
        :param key_step:
        :param data:
        :return: hint_dict/#BYPASS/#UNDEFINE
        """
        def _parse_task_event(mode: Literal["CREATE", "FINISH", "SCHE_CREATE", "SCHE_FINISH"]):
            if data["function"] in {"create_task", "create_schedule_task", "run_external_tool", "#缺失函数", "#参数错误"}:
                return "#BYPASS"

            call_name = data["function"]
            params = data['input_params']
            result = data['results'].get('result', None)
            task_status = data["status"]
            event = SummaryHint(
                title="创建工具调用任务",
                information={
                    "task_id": data["id"],
                    "task_name": data["name"],
                    "tool_name": self._reverse_call_name(call_name)
                },
                event_time=event_time
            )
            if mode == "FINISH":
                event.title = "工具执行完毕"

            elif mode == "SCHE_CREATE":
                event.title = "创建定时工具调用"
                mode = "CREATE"

            elif mode == "SCHE_FINISH":
                event.title = "定时任务触发"
                mode = "FINISH"

            if call_name == "list_tasks_status":
                if mode == "CREATE":
                    event.information["introduction"] = f"获取任务列表"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"已获取任务列表:  \n内容:  {result}"

                    else:
                        event.information["introduction"] = f"已获取任务列表，内容省略。"

            elif call_name == "get_task_data":
                if mode == "CREATE":
                    event.information["introduction"] = f"获取任务 [{params['task_id']}] 的详情"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"任务 [{params['task_id']}] 的详情:  \n{result}"

                    else:
                        if isinstance(result, dict):
                            omit = {k: v for k, v in result.items() if
                                    k in {"task_id", "task_name", "tool_name", "status", "last_fire_at", "next_run_at"}}

                        else:
                            omit = None

                        event.information["introduction"] = f"任务 [{params['task_id']}] 的基本信息:  \n{omit}"

            elif call_name == "list_external_tools":
                if mode == "CREATE":
                    event.information["introduction"] = f"获取外部工具列表"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"外部工具列表:  \n{result}"

                    else:
                        event.information["introduction"] = f"已获取外部工具列表，内容略"

            elif call_name == "load_external_tool":
                if mode == "CREATE":
                    event.information["introduction"] = f"获取外部工具[{params['source_name']}.{params['tool_name']}]请求体"

                elif mode == "FINISH":
                    event.information["introduction"] = f"返回外部工具[{params['source_name']}.{params['tool_name']}]请求体"

            elif call_name == "cancel_schedule_task":
                if mode == "CREATE":
                    event.information["introduction"] = f"取消定时任务 [{params['task_id']}]"

                elif mode == "FINISH":
                    event.information["introduction"] = f"已取消定时任务 [{params['task_id']}]"

            elif call_name == "notepad":
                if mode == "CREATE":
                    if detail:
                        event.information["introduction"] = f"创建一个记事本  \n内容: {params['note']}"

                    else:
                        event.information["introduction"] = f"创建一个记事本"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"记事本已触发，返回内容:  \n{params['note']}"

                    else:
                        event.information["introduction"] = f"记事本已触发，返回内容略"

            elif call_name == self._get_call_name("install_skill", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"从路径 <{params['path']}> 安装一个SKILL"

                elif mode == "FINISH":
                    if task_status != "DONE":
                        event.information["introduction"] = f"路径 <{params['path']}> 安装SKILL失败"

                    else:
                        event.information["introduction"] = f"已安装路径 <{params['path']}> 的SKILL"

            elif call_name == self._get_call_name("load_skill", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"加载SKILL <{params['skill_name']}> 的大纲"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"SKILL <{params['skill_name']}> 的大纲:  \n{result}"

                    else:
                        event.information["introduction"] = f"返回SKILL <{params['skill_name']}> 的大纲，内容略"

            elif call_name == self._get_call_name("list_skills", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"列出已安装SKILL的列表"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"已安装SKILL的列表:  \n{result}"

                    else:
                        event.information["introduction"] = f"返回已安装SKILL的列表，内容略"

            elif call_name == self._get_call_name("uninstall_skill", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"卸载SKILL <{params['skill_name']}>"

                elif mode == "FINISH":
                    if task_status != "DONE":
                        event.information["introduction"] = f"卸载SKILL <{params['skill_name']}> 失败"

                    else:
                        event.information["introduction"] = f"已成功卸载SKILL <{params['skill_name']}>"

            elif call_name == self._get_call_name("get_skill_content", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"读取SKILL <{params['skill_name']}> 的内容"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"SKILL <{params['skill_name']}> 的指定内容:  \n{result}"

                    else:
                        event.information["introduction"] = f"返回SKILL <{params['skill_name']}> 指定的内容，详情略"

            elif call_name == self._get_call_name("get_content", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"读取文件 <{params['file']}>"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"文件 <{params['file']}> 的内容:  \n{result}"

                    else:
                        event.information["introduction"] = f"返回文件 <{params['file']}> 的内容，详情略"

            elif call_name == self._get_call_name("set_content", "skill_installer"):
                if mode == "CREATE":
                    if detail:
                        event.information["introduction"] = (f"编辑文件 <{params['path']}>  \n"
                                                             f"变更内容:  \nstart_line: {params.get('start_line', 0)}  \n"
                                                             f"end_line: {params.get('end_line', 0)}  \n{params['content']}")

                    else:
                        event.information["introduction"] = f"编辑文件 <{params['path']}>"

                elif mode == "FINISH":
                    if task_status == "DONE":
                        if detail:
                            event.information["introduction"] = f"成功写入文件 <{params['path']}> ，写入内容:  \n{params['content']}"

                        else:
                            event.information["introduction"] = f"成功写入文件 <{params['path']}> ，内容略"

                    else:
                        event.information["introduction"] = f"写入文件 <{params['path']}> 失败"

            elif call_name == self._get_call_name("remove_path", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"删除文件 <{params['path']}>"

                elif mode == "FINISH":
                    if task_status != "DONE":
                        event.information["introduction"] = f"删除文件 <{params['path']}> 失败"

                    else:
                        event.information["introduction"] = f"已删除文件 <{params['path']}>"

            elif call_name == self._get_call_name("list_dir", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"获取文件目录 <{params['dir_path']}>"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"<{params['dir_path']}> 的目录结构:  \n{result}"

                    else:
                        event.information["introduction"] = f"返回 <{params['dir_path']}> 的目录结构，详情略"

            elif call_name == self._get_call_name("load_tools_from_python_file", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"加载SKILL <{params['skill_name']}> 的附属工具"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"SKILL <{params['skill_name']}> 的附属工具:  \n{result}"

                    else:
                        event.information["introduction"] = f"返回SKILL <{params['skill_name']}> 的附属工具，详情略"

            elif call_name == self._get_call_name("execute_script", "skill_installer"):
                if mode == "CREATE":
                    if detail:
                        event.information["introduction"] = (f"执行SKILL <{params['skill_name']}> 的脚本  \n"
                                                             f"脚本路径: <{params['script_path']}>  \n"
                                                             f"输入参数: <{params.get('arguments', None)}>")

                    else:
                        event.information["introduction"] = f"执行SKILL <{params['skill_name']}> 的脚本"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"脚本执行结果:  \n{result}"

                    else:
                        event.information["introduction"] = f"已执行SKILL <{params['skill_name']}> 的脚本，结果略"

            elif call_name == self._get_call_name("is_skill", "skill_installer"):
                if mode == "CREATE":
                    event.information["introduction"] = f"判断路径 <{params['skill_path']}> 是否SKILL"

                elif mode == "FINISH":
                    if result:
                        event.information["introduction"] = f"路径 <{params['skill_path']}> 是SKILL"

                    else:
                        event.information["introduction"] = f"路径 <{params['skill_path']}> 不是SKILL"

            elif call_name == self._get_call_name("search", "websearch"):
                if mode == "CREATE":
                    event.information[
                        "introduction"] = f"调用websearch查询 <{params['query']}>; 使用 <{params['cache_key']}> 作为缓存索引>"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = (f"websearch查询 <{params['query']}> 的结果:  \n"
                                                             f"缓存索引: <{params['cache_key']}>  \n"
                                                             f"结果: {result}")

                    else:
                        event.information["introduction"] = (f"返回websearch查询 <{params['query']}> 的结果; "
                                                             f"缓存索引: <{params['cache_key']}> ; 详情略")

            elif call_name == self._get_call_name("extract", "websearch"):
                if mode == "CREATE":
                    event.information["introduction"] = f"调用爬虫爬取 <{params['urls']}>"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"爬虫爬取 <{params['urls']}> 的结果:  \n{result}"

                    else:
                        event.information["introduction"] = f"返回爬虫爬取 <{params['urls']}> 的结果，详情略"

            elif call_name == self._get_call_name("url_map", "websearch"):
                if mode == "CREATE":
                    event.information["introduction"] = f"获取链接 <{params['url']}> 的相关链接"

                elif mode == "FINISH":
                    if detail:
                        event.information["introduction"] = f"链接 <{params['url']}> 的相关链接:  \n{result}"

                    else:
                        event.information["introduction"] = f"返回链接 <{params['url']}> 的相关链接，详情略"

            else:
                return "#UNDEFINE"

            return event

        def _parse_prompts(mode: Literal["USER_ASK", "SYSTEM_ASK", "RESPONSE"]):
            resort = []
            prompts: list[dict] = data["prompts"]
            if not prompts:
                return "#BYPASS"

            for prompt in prompts:
                prompt_data: dict = copy.deepcopy(prompt["data"])
                prompt_type = prompt["type"]
                if prompt_type == "TEXT":
                    if not detail:
                        prompt_data.pop("reasoning_content", None)

                    resort.append({"type": "文本信息", "content": prompt_data})

                elif prompt_type == "IMAGE":
                    resort.append({"type": "图片信息", "content": prompt_data})

            for prompt in prompts:
                prompt_data: dict = copy.deepcopy(prompt["data"])
                prompt_type = prompt["type"]
                if prompt_type == "TOOLCALL":
                    resort.append(
                        {"type": "TOOL_CALL", "content": {k: v for k, v in prompt_data.items() if k != "arguments"}})

                elif prompt_type == "TOOL_RESP":
                    resort.append(
                        {"type": "TOOL_RESPONSE", "content": {k: v for k, v in prompt_data.items() if k != "response"}})

            event = SummaryHint(
                title="用户发起对话请求",
                information=resort,
                event_time=event_time
            )

            if mode == "SYSTEM_ASK":
                event.title = "系统发起对话请求(callback)"

            if mode == "RESPONSE":
                event.title = "LLM返回对话响应"

            return event

        def _parse_summary():
            event = SummaryHint(
                title="Context压缩结果",
                information=data,
                event_time=event_time
            )
            return event

        mapping_rules = {
            HT.TASK_CREATE: lambda : _parse_task_event("CREATE"),
            HT.TASK_DONE: lambda : _parse_task_event("FINISH"),
            HT.SCHE_CREATE: lambda : _parse_task_event("SCHE_CREATE"),
            HT.SCHE_FIRE: lambda : _parse_task_event("SCHE_FINISH"),
            HT.USER_ASK: lambda : _parse_prompts("USER_ASK"),
            HT.LLM_ASK: lambda : _parse_prompts("SYSTEM_ASK"),
            HT.LLM_RESP: lambda : _parse_prompts("RESPONSE"),
            HT.COMP_CONTEXT: _parse_summary,
        }

        next_step = mapping_rules
        for key in key_step:
            if key in next_step:
                next_step = mapping_rules[key]
                if callable(next_step):
                    hint = next_step()
                    if isinstance(hint, SummaryHint):
                        return hint.to_dict()

                    return hint

                elif not isinstance(next_step, dict):
                    return "#UNDEFINE"

        return "#UNDEFINE"

    def _upload_image(self, image: Path) -> str:
        """
        添加图片，返回index
        :param image:
        :return:
        """
        if image in self._pending_upload_image:
            return str(self._pending_upload_image.index(image) + 1)

        self._pending_upload_image.append(image)
        return str(len(self._pending_upload_image))

    @staticmethod
    def _decode_compress_context(prompt: Prompts) -> str:
        if not prompt:
            return ""

        summary = prompt.prompts[0]
        if not isinstance(summary, TextPrompt):
            return ""

        try:
            parsed: dict = json.loads(summary.text)

        except json.JSONDecodeError:
            return ""

        else:
            output = ""
            if "dialog_history" not in parsed:
                return ""

            if "context_summary" not in parsed:
                return ""

            dialog_history: list[dict] = parsed["dialog_history"]
            context_summary: str = parsed["context_summary"]
            running_task = parsed.get("running_task", "")
            context_tasks = parsed.get("context_tasks", {})

            if dialog_history:
                output += "# 对话简要\n\n"

            i = 1
            for dialog in dialog_history:
                theme = dialog.get("theme", "")
                if not theme:
                    continue

                result = dialog.get("result", "")
                summary = dialog.get("summary", "")
                tasks = dialog.get("tasks", {})

                output += f"## 对话{i}\n\n"
                output += f"主题: {theme if theme else '无主题'}  \n"
                output += f"结论: {result if result else '无结论'}  \n"
                output += f"总结: {summary if summary else '无总结'}  \n"
                if tasks:
                    output += "关键任务:  \n"
                    for k, v in tasks.items():
                        output += f"- `{k}` : {v}\n"

                output += "\n---\n\n"
                i += 1

            output += "# Context全文总结\n\n"
            output += context_summary if context_summary else "无全文总结"
            output += "\n\n"

            output += "# 当前进行中任务，需要关注的信息\n\n"
            output += running_task if running_task else "无进行中任务"
            output += "\n\n"

            if context_tasks:
                output += "# 需要关注的任务:  \n"
                for k, v in context_tasks.items():
                    output += f"- `{k}` : {v}\n"

                output += '\n'

            return output

    def _get_compress_context_pinned(self) -> str:
        output = "# 外部工具列表\n\n"
        tool_list = json.dumps(self.list_external_tools(), ensure_ascii=False, indent=2)
        output += f"```json\n{tool_list}\n```\n\n"

        def _merge_title_tree(_origin: dict, _new: dict):
            # 全局获取
            if not isinstance(_origin, dict) or not isinstance(_new, dict):
                return

            if not _origin:
                return

            if not _new:
                _origin.clear()
                return

            for k, v in _new.items():
                if not isinstance(v, dict):
                    continue

                if k not in _origin:
                    _origin[k] = copy.deepcopy(v)
                    continue

                if not isinstance(_origin[k], dict):
                    _origin[k] = copy.deepcopy(v)
                    continue

                if not _origin[k]:
                    continue

                if not v:
                    _origin[k] = {}
                    continue

                _merge_title_tree(_origin[k], v)

        def _filter_title_tree(_content_tree: dict, _title_tree: dict | None) -> dict | None:
            if not isinstance(_content_tree, dict) or not isinstance(_title_tree, dict):
                return None

            if not _title_tree:
                return {}

            output = {}
            for k, v in _title_tree.items():
                if k not in _content_tree:
                    continue

                if not isinstance(v, dict):
                    continue

                if not v:
                    output[k] = {}
                    continue

                filtered = _filter_title_tree(_content_tree[k], v)
                if filtered is not None:
                    output[k] = filtered

            return output if output else None

        loaded_skill_content = {}
        loaded_external_tool: dict[str, dict] = {}
        for tc in self.control.context:
            if not isinstance(tc, ToolCall):
                continue

            if tc.function_name == "load_external_tool":
                sn = tc.arguments.get("source_name")
                tn = tc.arguments.get("tool_name")
                helper = self.load_external_tool(sn, tn)
                if isinstance(helper, str):
                    continue

                loaded_external_tool.setdefault(sn, {})[tn] = helper
                continue

            if tc.function_name != "run_external_tool":
                continue

            if tc.arguments.get("source_name", "") != "skill_installer":
                continue

            if tc.arguments.get("tool_name", "") != "get_skill_content":
                continue

            args = tc.arguments.get("arguments", None)
            if not isinstance(args, dict):
                continue

            skill_name = args.get("skill_name", "")
            title_tree = args.get("title_tree", None)
            if not skill_name:
                continue

            if skill_name not in self._skills.skills:
                continue

            skill = self._skills.skills[skill_name]
            title_tree = _filter_title_tree(skill.md_content, title_tree)
            if title_tree is None:
                continue

            if skill_name not in loaded_skill_content:
                loaded_skill_content[skill_name] = title_tree

            else:
                _merge_title_tree(loaded_skill_content[skill_name], title_tree)

        if loaded_external_tool:
            remapping = {k: list(v.values()) for k, v in loaded_external_tool.items()}
            output += "# 已加载外部工具\n\n"
            output += f"```json\n{json.dumps(remapping, ensure_ascii=False, indent=2)}\n```\n\n"

        if loaded_skill_content:
            output += "# 已加载SKILL内容\n\n"
            for key, value in loaded_skill_content.items():
                output += f"## {key}\n\n"
                output += f"```markdown\n{self._skills.get_skill_content(key, value)}\n```\n\n---\n\n"

        return output

    async def compress_context(self, new_prompt: Prompts = None) -> Prompts:
        pinned = self._get_compress_context_pinned()
        suffix = self._detach_tool_calls_for_context_reset("BaseAgent 自动压缩 Context")
        suffix_image = []
        if new_prompt is not None:
            lost_tool_resp: list[ToolResponse] = []
            lost_ask = ""
            for prompt in new_prompt:
                if isinstance(prompt, ToolResponse):
                    lost_tool_resp.append(prompt)

                elif isinstance(prompt, TextPrompt):
                    lost_ask = prompt.text

                elif isinstance(prompt, ImagePrompt):
                    suffix_image.append(prompt)

            if lost_tool_resp:
                suffix += "# 丢失的TOOL RESPONSE\n\n"
                for tr in lost_tool_resp:
                    suffix += f"```\n{tr.to_dict()}\n```\n\n---\n\n"

            if lost_ask:
                suffix += f"# 丢失的对话请求\n\n{lost_ask}\n\n---\n\n"

        window: list[AgentHistory] = []
        for his in reversed(self._history.history):
            window.append(his)
            if his.event_type is HT.COMP_CONTEXT:
                break

        summary_input = []
        get_last = False
        for his in window:
            if not get_last:
                hint = self._summary_mapping([his.event_type], his.event_data, float(his.event_time), True)
                if isinstance(hint, str):
                    continue

                else:
                    summary_input.append(hint)

            else:
                hint = self._summary_mapping([his.event_type], his.event_data, float(his.event_time))
                if isinstance(hint, str):
                    continue

                else:
                    summary_input.append(hint)

            if not get_last and his.event_type is HT.USER_ASK:
                get_last = True

        summary_input.reverse()
        summary_chat = self.control.copy()
        model_params = summary_chat.model.get("parameter", {})
        model_params.setdefault("response_format", {}).update({'type': 'json_object'})
        new_set = ChatConfig(model_params=model_params, keep_alive=False)
        summary_chat.setting(new_set)
        summary_system_prompt = self.get_last_version_prompt("base_summary_context_system")
        if summary_system_prompt is None:
            summary_system_prompt = self.register_prompt(
                "base_summary_context_system",
                PROMPTS_DIR / "base_summary_context_system",
                description="基本Context压缩系统提示词",
                role="system"
            )

        summary_chat.add_context(SystemPrompt(await summary_system_prompt.get_prompt()))
        retry = 0
        parsed_resp = ""
        input_prompt = TextPrompt("user", json.dumps(summary_input, ensure_ascii=False, indent=2))
        while retry < self.control_retry_limit:
            try:
                response = await summary_chat.ask(input_prompt, timeout=SETTING_CFG.Agent.AgentCompressContextTimeout)  # 默认900
                if response is None:
                    continue

            except Exception as E:
                self._logger.error(f"{self._log_prefix()} 自动压缩上下文错误: {E}")
                self._logger.debug(traceback.format_exc())

            else:
                parsed_resp = self._decode_compress_context(response)
                if not parsed_resp:
                    continue

                break

            finally:
                retry += 1

        await summary_chat.close()
        self._history.compress_context(data=parsed_resp if parsed_resp else "无压缩信息")

        # fallback
        # 直接开启新上下文，只保留顶置信息
        await self._save_checkpoint_async()
        compress_tips = (f"> Context已自动压缩，以上为压缩系统自动创建的内容  \n\n"
                         f"# 压缩后重新初始化流程\n\n"
                         f"1. 调用 `load_external_tool` 加载缺失的所需外部工具\n"
                         f"2. 调用 `load_skill` 检查 SKILL 是否缺失所需内容\n"
                         f"3. 调用 `get_skill_content` 加载/更新所需的 SKILL 内容\n"
                         f"4. 继续未完成的任务\n\n\n")
        first_prompt = Prompts([TextPrompt(
            role="user",
            text=pinned + parsed_resp + suffix + compress_tips
        )] + suffix_image)
        new_context = self._context.new()
        self.control.replace_context(new_context)
        await self.save_agent_async(save_context_history=True)
        return first_prompt

    # 测试自动压缩
    async def compress_test(self):
        pinned = self._get_compress_context_pinned()
        window: list[AgentHistory] = []
        for his in reversed(self._history.history):
            window.append(his)
            if his.event_type is HT.COMP_CONTEXT:
                break

        summary_input = []
        get_last = False
        for his in window:
            if not get_last:
                hint = self._summary_mapping([his.event_type], his.event_data, float(his.event_time), True)
                if isinstance(hint, str):
                    continue

                else:
                    summary_input.append(hint)

            else:
                hint = self._summary_mapping([his.event_type], his.event_data, float(his.event_time))
                if isinstance(hint, str):
                    continue

                else:
                    summary_input.append(hint)

            if not get_last and his.event_type is HT.USER_ASK:
                get_last = True

        summary_input.reverse()
        debug(summary_input)
        summary_chat = self.control.copy()
        model_params = summary_chat.model.get("parameter", {})
        model_params.setdefault("response_format", {}).update({'type': 'json_object'})
        new_set = ChatConfig(model_params=model_params, keep_alive=False)
        summary_chat.setting(new_set)
        summary_system_prompt = self.get_last_version_prompt("base_summary_context_system")
        if summary_system_prompt is None:
            summary_system_prompt = self.register_prompt(
                "base_summary_context_system",
                PROMPTS_DIR / "base_summary_context_system",
                description="基本Context压缩系统提示词",
                role="system"
            )

        summary_chat.add_context(SystemPrompt(await summary_system_prompt.get_prompt()))
        retry = 0
        parsed_resp = ""
        input_prompt = TextPrompt("user", json.dumps(summary_input, ensure_ascii=False, indent=2))
        while retry < self.control_retry_limit:
            try:
                response = await summary_chat.ask(input_prompt,
                                                  timeout=SETTING_CFG.Agent.AgentCompressContextTimeout)  # 默认900
                if response is None:
                    continue

            except Exception as E:
                self._logger.error(f"{self._log_prefix()} 自动压缩上下文错误: {E}")
                self._logger.debug(traceback.format_exc())

            else:
                parsed_resp = self._decode_compress_context(response)
                if not parsed_resp:
                    continue

                break

            finally:
                retry += 1

        await summary_chat.close()
        debug(pinned + parsed_resp + "> Context已自动压缩，以上为压缩系统自动创建的内容")

    @staticmethod
    def _check_trigger(spec: str, a: int, b: int) -> str:
        times = []
        if '/' in spec:
            spec = ''.join(spec.split('/')[:-1])

        spec = spec.replace('*', "")
        if not spec:
            return ""

        spec = spec.replace(',', '-')
        times.extend(spec.split('-'))
        if not times:
            return ""

        for t in times:  # type: str
            t = t.strip()
            if not t:
                continue

            if not t.isdigit():
                return "值格式有误"

            if int(t) < a or int(t) > b:
                return f"值超出范围。允许范围{a}-{b}"

        return ""

    def get_tasks_list(self) -> dict:
        result = {
            "schedule": [],
            "running": [],
            "accomplished": []
        }
        # 处理定时任务
        self._update_scheduled_tasks()
        for task_id, data in self._schedule_tasks.pending_task.items():
            job = self._scheduler.check_job(task_id)
            if job is None:
                continue

            data.next_run_at = f"{job.next_run_time:%y-%m-%d %H:%M:%S}" if job.next_run_time else None
            info = data.to_llm()
            info.pop("result", None)
            info.pop("exceptions", None)
            result["schedule"].append(info)

        # 处理计划中任务
        for task_id, data in self._task_queue.tasks.items():
            info = data.to_llm()
            info.pop("result", None)
            info.pop("exceptions", None)
            result["running"].append(info)

        # 处理已完成任务
        for task_id, data in self._task_queue.accomplished_task.items():
            info = data.to_llm()
            info.pop("result", None)
            info.pop("exceptions", None)
            result["accomplished"].append(info)

        return result

    # =========================================================
    # 内置TOOLS
    # =========================================================

    async def create_task(
            self,
            tool_name: str,
            arguments: dict | None = None,
            task_name: str = "",
            callback: Literal["ALWAYS", "EXCEPTION", "NEVER"] = "NEVER"
    ) -> str:
        """
        将一个工具添加进任务队列。
        :param tool_name: 工具的tool_name。
        :param arguments: 输入参数。
        :param task_name: 任务名，不要太长。
        :param callback: 任务完成时是否执行一次LLM请求。 ALWAYS: 无论任务成功还是失败都请求;
        EXCEPTION: 仅在失败或异常时请求; NEVER: 任何情况都不请求。
        :return:成功: 返回任务id; tool不存在: 返回"工具 <tool_name> 不存在"
        """
        if tool_name == "create_task":
            return "禁止在显式创建任务中再显式创建任务"

        if arguments is None:
            arguments = {}

        if not task_name:
            task_name = tool_name

        if tool_name in self.functions_default_setting:
            callback = self.functions_default_setting[tool_name]["callback"]

        new_task = await self.add_task(
            tool_name,
            arguments,
            name=task_name,
            callback=callback
        )
        if new_task is None:
            raise ToolNotExistError(f"Tool {tool_name} not exist")

        return new_task.id

    def create_schedule_task(
            self,
            tool_name: str,
            trigger: dict,
            arguments: dict | None = None,
            task_name: str = "",
            callback: Literal["ALWAYS", "EXCEPTION", "NEVER"] = "NEVER"
    ) -> str:
        """
        将一个工具添加为定时任务。
        :param tool_name: 工具的tool_name。
        :param arguments: 输入参数。
        :param trigger: 采用APScheduler(v3.11)标准的触发器字典，可用键名与类型如下:
        {"trigger_once": str (ISO8601), "year": str (4位数字), "month": str (1-12), "day": str (1-31),
        "week": str (1-53), "day_of_week": str (0-6或mon,tue,wed,thu,fri,sat,sun), "hour": str (0-23),
        "minute": str (0-59), "second": str (0-59), "start_date": str (ISO8601), "end_date": str (ISO8601)}。
        创建一次性定时任务: 只需包含"trigger_once"的字典{"trigger_once": ISO8601标准的时间字符串}。
        创建间隔任务: 示例<每天早上9点30分>{"hour": "9", "minute": "30"}; 示例<每10分钟>{"minute": "*/10"};
        示例<每个工作日下午6点>{"day_of_week": "mon-fri", "hour": "18"}; 示例<每天早上9点和下午5点>{"hour": "9,17"}。
        当"trigger_once"存在时，其他键将不会生效。
        :param task_name: 任务名，不要太长。
        :param callback: 任务完成时是否执行一次LLM请求。 ALWAYS: 无论任务成功还是失败都请求;
        EXCEPTION: 仅在失败或异常时请求; NEVER: 任何情况都不请求。
        :return: 成功: 返回任务id; tool不存在: 返回"工具 <tool_name> 不存在"; trigger格式错误: 返回error_message(string)
        """
        if tool_name == "create_schedule_task":
            return "禁止在定时任务中创建定时任务"

        if arguments is None:
            arguments = {}

        if not isinstance(trigger, dict):
            return "trigger必须是字典dict"

        if not trigger:
            return "trigger为空"

        if not task_name:
            task_name = tool_name

        issues = {}
        now = datetime.now(tz=self._scheduler.tz)
        for key, value in trigger.items():
            if key not in {"trigger_once", "year", "month", "day", "week", "day_of_week", "hour",
                           "minute", "second", "start_date", "end_date"}:
                issues[key] = "意外键名"
                continue

            elif key in {"trigger_once", "start_date", "end_date"}:
                try:
                    dt = datetime.fromisoformat(value)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=self._scheduler.tz)

                except ValueError:
                    issues[key] = f"<{value}> 不是ISO8601格式的字符串"
                    continue

                else:
                    if key == "trigger_once" and dt <= now:
                        issues[key] = f"指定的执行时间已过期，当前时间：{now.isoformat()}"
                        continue

            elif key == "year":
                if value.isdigit():
                    if len(value) != 4:
                        issues[key] = "year的值不是4位数字"
                        continue

                    if int(value) < now.year:
                        issues[key] = f"指定的执行时间已过期，当前时间：{now.isoformat()}"
                        continue

            elif key == "month":
                error = self._check_trigger(value, 1, 12)
                if error:
                    issues[key] = error
                    continue

            elif key == "day":
                error = self._check_trigger(value, 1, 31)
                if error:
                    issues[key] = error
                    continue

            elif key == "week":
                error = self._check_trigger(value, 1, 53)
                if error:
                    issues[key] = error
                    continue

            elif key == "day_of_week":
                temp_value = value.replace('*', '').replace(',', '-')
                values = temp_value.split('-')
                for t in values:  # type: str
                    t = t.strip()
                    if not t:
                        continue

                    if t.isdigit():
                        if int(t) < 0 or int(t) > 6:
                            issues[key] = "值超出范围。允许范围0-6"
                            continue

                    else:
                        if t not in {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}:
                            issues[key] = '星期缩写错误。允许值Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]'
                            continue

            elif key == "hour":
                error = self._check_trigger(value, 0, 23)
                if error:
                    issues[key] = error
                    continue

            elif key in {"minute", "second"}:
                error = self._check_trigger(value, 0, 59)
                if error:
                    issues[key] = error
                    continue

        if issues:
            error = "trigger格式有误：\n"
            for k, v in issues.items():
                error += f"[{k}]: {v}\n"

            return error

        if tool_name in self.functions_default_setting:
            callback = self.functions_default_setting[tool_name]["callback"]

        new_task = self.add_schedule_task(
            tool_name,
            trigger=trigger,
            params=arguments,
            name=task_name,
            callback=callback
        )
        if new_task is None:
            raise ToolNotExistError(f"Tool {tool_name} not exist")

        return new_task.id

    def cancel_schedule_task(
            self,
            task_id: str
    ) -> str:
        """
        取消一个定时任务
        :param task_id: 定时任务id
        :return: 成功: 成功提示
        """
        self._scheduler.remove_job(task_id)
        return f"定时任务[{task_id}]已取消"

    def list_tasks_status(self, list_all: bool = False) -> dict:
        """
        列出当前任务的状态。 默认只列出正在运行的任务、定时任务、新完成的任务(从未在context里出现过)。
        :param list_all: 若为True，则列出所有任务。由于任务列表很长消耗token较多，因此一般为False就好。
        :return: 成功: 一个嵌套字典{"schedule": [], "running": [], "accomplished": []};
        无任务: 空字典
        """
        result = self.get_tasks_list()

        if not list_all:
            result["accomplished"] = [t for t in result["accomplished"] if t["task_id"] in self._task_queue.show_tasks]
            self._task_queue.show_tasks.clear()

        return result

    async def get_task_data(self, task_id: str, wait: bool = False) -> dict:
        """
        获取指定ID的任务结果。
        :param task_id: 任务id。
        :param wait: 是否等待任务完成。若为True，则会等待任务运行完毕才返回结果。对于长耗时任务 **切勿** 等待。
        :return:成功: 一个嵌套字典：{"status": str, "result": Any, "exceptions": List[exception messages]};
        任务未完成，且wait为False: 则返回一个运行中字典{"result": "Task Running"};
        任务不存在: 返回一个不存在字典{"result": "Task Not Found"}
        """
        normal_task_lookup: dict[str, AgentTask] = self._get_task_lookup()
        if task_id in normal_task_lookup:
            task = normal_task_lookup[task_id]
            if task.status not in _ts_done:
                if not wait:
                    result = task.to_llm()
                    result["result"] = "Task Running"
                    return result

                else:
                    await task.event.wait()

            return task.to_llm()

        elif task_id in self._schedule_tasks.tasks:
            job = self._scheduler.check_job(task_id)
            task = self._schedule_tasks.tasks[task_id]
            if job is None:
                result = task.to_llm()
                result["exceptions"] = ["任务未安排或已丢失"]
                result["last_fire_at"] = None
                result["next_run_at"] = None
                return result

            task.next_run_at = f"{job.next_run_time:%y-%m-%d %H:%M:%S}" if job.next_run_time else None
            return task.to_llm()

        else:
            return {"result": "Task Not Found"}

    def list_external_tools(self) -> dict:
        """
        列出所有外部工具，根据需要使用 load_external_tool 获取请求格式，然后使用 run_external_tool 执行。
        :return:成功: 返回带详细说明的外部工具字典。若无外部工具则为空字典
        """
        remapping = {"external tools": {}}
        for k, v in self._functions_source_mapping.items():
            for s, c in v.items():
                if c not in self._functions:
                    continue

                function = self._functions[c]
                description = build_tool_payloads(function)["openai"]["function"]["description"]
                helper = {"source_name": s, "tool_name": k, "description": description}
                remapping["external tools"].setdefault(s, []).append(helper)

        return remapping if remapping["external tools"] else {}

    def load_external_tool(self, source_name: str,tool_name: str) -> dict | str:
        """
        返回指定外部工具的具体调用方法
        :param source_name: 外部工具源名称(MCP/SKILL名字)。
        :param tool_name: 外部工具的tool_name。
        :return: 成功: 返回外部工具请求体字典;
        source不存在: 返回"工具源 <source_name> 不存在";
        tool不存在: 返回"工具 <tool_name> 不存在"
        """
        if tool_name not in self._functions_source_mapping:
            return f"工具 {tool_name} 不存在"

        if source_name not in self._functions_source_mapping[tool_name]:
            return f"工具源 {source_name} 不存在"

        call_name = self._get_call_name(tool_name, source_name)
        if call_name not in self._functions:
            self._logger.error(f"工具 {source_name}: {call_name} 已丢失")
            return f"工具 {call_name} 不存在"

        function = self._functions[call_name]
        instruction = build_tool_payloads(function)["openai"]["function"]
        helper = {"source_name": source_name, "tool_name": tool_name, **instruction}
        return helper

    async def run_external_tool(
            self,
            source_name: str,
            tool_name: str,
            arguments: dict = None,
            callback: Literal["ALWAYS", "EXCEPTION", "NEVER"] = "NEVER"
    ) -> Any | str:
        """
        外部工具执行器，返回调用的外部工具的结果。
        调用任何外部工具前，必须确保已通过 load_external_tool 获取具体的请求格式
        :param source_name: 外部工具源名称(MCP/SKILL名字)。
        :param tool_name: 外部工具的tool_name。
        :param arguments: 输入参数。
        :param callback: 任务完成时是否执行一次LLM请求。 ALWAYS: 无论任务成功还是失败都请求;
        EXCEPTION: 仅在失败或异常时请求; NEVER: 任何情况都不请求。
        :return: 成功: 返回外部工具的结果/异常;
        tool不存在: 抛出"工具 <tool_name> 不存在"异常;
        source不存在: 抛出"工具源 <source_name> 异常"
        """
        if arguments is None:
            arguments = {}

        if tool_name in self._functions_source_mapping:
            if source_name not in self._functions_source_mapping[tool_name]:
                raise ToolNotExistError(f"工具源 {source_name} 不存在")

            call_name = self._get_call_name(tool_name, source_name)
            if call_name not in self._functions:
                raise ToolNotExistError(f"工具 {call_name} 不存在")

        elif tool_name in self._functions:
            call_name = tool_name

        else:
            raise ToolNotExistError(f"工具 {tool_name} 不存在")

        if source_name in self._skills.skills:
            self._skills.skills[source_name].last_use = time.time()

        function = self._functions[call_name]
        check = validate_tool_arguments(function, arguments)
        if not check.ok:
            raise ToolArgumentsError(f"调用外部工具[{tool_name}]传入参数有误: \n{[e.to_dict() for e in check.issues]}")

        wait = self._skills.tools_default_setting.get(source_name, {}).get(tool_name, {}).get("wait", False)
        default_callback = self._skills.tools_default_setting.get(source_name, {}).get(tool_name, {}).get("callback", "")
        callback = default_callback or callback

        # 创建子任务
        lookup_tasks: dict[str, AgentTask] = self._get_task_lookup(True)
        name = self._functions_name_mapping.get(call_name, "")
        self_id = self.get_self_id()
        if not self_id or self_id not in lookup_tasks:
            raise ValueError("父任务id不存在")

        parent_task = lookup_tasks[self_id]
        call_id = parent_task.call_id
        sub_task = await self.add_task(
            call_name,
            arguments,
            name=name,
            call_id=call_id,
            callback=callback,  # type: ignore
            wait=wait
        )

        # 修改call_id
        if call_id not in self._last_call_ids:
            await asyncio.sleep(1)  # 可能还没加入

        if call_id in self._last_call_ids:
            self._last_call_ids[call_id] = sub_task.id

        if sub_task.is_wait:
            await sub_task.event.wait()
            parent_task.exceptions.extend(sub_task.exceptions)
            if sub_task.status.value == "EXCEPTION":
                raise SubTaskError(f"子任务[{sub_task.id}]异常")

            return sub_task.results["result"]

        else:
            return f"已创建外部工具任务[{sub_task.id}]"

    def notepad(self, note: str, callback: Literal["ALWAYS", "EXCEPTION", "NEVER"] = "ALWAYS") -> str:
        """
        搭配定时任务使用，输入一个字符串，返回该字符串，用作记事本。
        :param note: 字符串，用LLM能理解的方式记录。
        :param callback: 任务完成时是否执行一次LLM请求。 ALWAYS: 无论任务成功还是失败都请求;
        EXCEPTION: 仅在失败或异常时请求; NEVER: 任何情况都不请求。
        :return: 成功: 输入的字符串
        """
        task_id = self.get_self_id()
        if task_id in self._schedule_tasks.tasks:
            self._schedule_tasks.tasks[task_id].callback = callback

        return note
