from __future__ import annotations

"""Private-chat controller Agent and its dedicated speaking model."""

import asyncio
import json
import time
import traceback
import weakref
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypedDict,
    cast,
)

import cnlunar
from PIL import Image, UnidentifiedImageError

from TIYA.LLM_connect import (
    Chat,
    Context,
    ContextOverflowError,
    ImagePrompt,
    Prompts,
    SystemPrompt,
    TextPrompt,
    ToolCall,
    ToolResponse,
    parse_llm_setting,
)
from TIYA.agent.agent import BaseAgent
from TIYA.agent.agent_prompt import AgentPrompt
from TIYA.config import (
    BASE_SKILL_DIR,
    PROMPTS_DIR,
    SETTING_CFG,
    get_bot_name,
    get_bot_uid,
)
from TIYA.file_cache import get_file_path_async
from TIYA.image_to_text import image2text
from TIYA.logger import BlockHandle, Logger, get_logger
from TIYA.model.character import get_character
from TIYA.model.message import BaseMsg, MessageManager, PrivateMsg
from TIYA.time_id import next_time_id
from TIYA.utils import timestamp2text

if TYPE_CHECKING:
    from TIYA.aqueue import Aqueue
    from TIYA.auto_fav import AutoFav
    from TIYA.model.chat_notification import ChatNotification, Notice
    from TIYA.model.persona import PrivatePersonaResult


_log = get_logger()
_SPEAK_INTENT_MAPPING = {
    "react": "响应话题",
    "joke": "开玩笑、玩梗",
    "tease": "调侃、互损",
    "irony": "反讽、说反话",
    "answer": "回应问题/要求",
    "ask": "提出问题",
    "opinion": "提出观点/话题",
    "agree": "认同",
    "disagree": "反驳",
    "comfort": "安慰/抚慰/表达友好",
    "refuse": "拒绝",
    "deflect": "回避或转移话题/要求",
    "quarrel": "争吵/谩骂/人身攻击",
    "deescalate": "退避/缓和聊天氛围/主动退让",
    "cool_down": "降低/缓和当前对话的氛围/热度",
    "neutral": "保持当前对话的氛围/热度",
    "heat_up": "增加/升温当前对话的氛围/热度",
}


class PrivateChatAgentHost(Protocol):
    user_id: str
    username: str
    data_path: Path
    message_history: MessageManager
    agent_round_limit: int
    speaker_round_limit: int
    relative_news: dict[str, dict[str, list]]
    NOTICE: ChatNotification
    memory_save_notice: Notice
    FAV: AutoFav
    aqueue: Aqueue
    logger: BlockHandle

    @property
    def participant_aliases(self) -> dict[str, list[str]]: ...

    async def speak(
        self,
        *,
        speak_prompt: str = "",
        working: str = "",
        relative_information: dict[str, str] | None = None,
        relative_memory: dict[str, str] | None = None,
        call_id: str = "",
    ) -> str | None: ...

    async def get_private_summary(self, immediate: bool = False) -> str: ...
    async def get_private_persona(self, immediate: bool = False, force: bool = False) -> PrivatePersonaResult: ...
    async def get_news_by_date(self, date: str | None = None) -> dict: ...
    async def add_memory(self, memory: str) -> dict: ...
    async def query_memory(self, query: str) -> dict: ...
    async def set_alias(self, user_id: str, alias: str) -> None: ...
    def get_old_message(self, count: int | None = None) -> list[PrivateMsg]: ...
    async def get_private_news(self) -> dict[str, list]: ...


class PrivatePostReason(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    USER_MESSAGE = auto()
    REPLY = auto()
    AGENT_INTERNAL = auto()
    INITIATE = auto()
    SAVE_MEMORY = auto()


_FORCE_SPEAK = {PrivatePostReason.USER_MESSAGE, PrivatePostReason.REPLY}
_CHOICE_SPEAK = {PrivatePostReason.AGENT_INTERNAL}
_AVOID_SPEAK = {PrivatePostReason.INITIATE, PrivatePostReason.SAVE_MEMORY}
_NEED_WAIT_PARSE = {PrivatePostReason.USER_MESSAGE}


def post_reason_prompt_maker(reason: PrivatePostReason) -> str:
    """根据触发来源所属的发言分组，生成统一的主控行为提示。"""
    reason_mapping = {
        PrivatePostReason.REPLY: "用户回复了你的消息",
        PrivatePostReason.USER_MESSAGE: "用户发来了新的消息",
        PrivatePostReason.AGENT_INTERNAL: "Agent 内部 React",
        PrivatePostReason.INITIATE: "Agent 初始化",
        PrivatePostReason.SAVE_MEMORY: "保存记忆 React，若已保存记忆可跳过",
    }

    if reason in _FORCE_SPEAK:
        speak_instruction = "应当回应用户"

    elif reason in _CHOICE_SPEAK:
        speak_instruction = "自行判断是否发言，优先不发言"

    elif reason in _AVOID_SPEAK:
        speak_instruction = "请勿发言"

    else:
        raise KeyError("有遗漏的私聊请求原因")

    return f"**本次请求来自 *{reason_mapping[reason]}*，{speak_instruction}**"


@dataclass(slots=True)
class PrivatePostTask:
    reason: PrivatePostReason
    timeout: float
    expired: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(slots=True)
class PrivateAgentPostPayload:
    reason: PrivatePostReason
    messages: list[BaseMsg]
    notice: str = ""
    participant_aliases: dict[str, list[str]] = field(default_factory=dict)
    said_intention: dict[str, dict] = field(default_factory=dict)

    def _parse_intention(self, msg_id: str) -> str:
        intention = self.said_intention.get(msg_id, {})
        if not intention:
            return ""

        output = "本次发言意图: "
        target_message_id = intention.get("target_message_id", "")
        if target_message_id:
            output += f"响应消息 [{target_message_id}]; "

        intent = intention.get("intent", "")
        if intent in _SPEAK_INTENT_MAPPING:
            output += f"发言行为类别: {_SPEAK_INTENT_MAPPING[intent]}; "

        effect = intention.get("effect", "")
        if effect in _SPEAK_INTENT_MAPPING:
            output += f"期望发言效果: {_SPEAK_INTENT_MAPPING[effect]}"

        return output.rstrip("; ")

    async def get_prompt(self) -> str:
        messages: list[dict] = []
        if self.reason in _NEED_WAIT_PARSE:
            tasks = [
                asyncio.create_task(
                    msg.to_llm(
                        wait=True,
                        timeout=SETTING_CFG.Groups.MessageParseTimeout,
                    )
                )
                for msg in self.messages
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for task in tasks:
                if (
                    not task.done()
                    or task.cancelled()
                    or task.exception() is not None
                ):
                    continue

                message = task.result()
                message["user_aliases"] = self.participant_aliases.get(
                    message["user_id"],
                    [],
                )
                intention = self._parse_intention(message["message_id"])
                if intention:
                    message["message_tips"].append(intention)

                messages.append(message)

        else:
            for message in self.messages:
                content = await message.to_llm(wait=False)
                content["user_aliases"] = self.participant_aliases.get(
                    content["user_id"],
                    [],
                )
                intention = self._parse_intention(content["message_id"])
                if intention:
                    content["message_tips"].append(intention)

                messages.append(content)

        now = datetime.now()
        return f"""\
{post_reason_prompt_maker(self.reason)}
**当前时间: {timestamp2text(now.timestamp(), "%y-%m-%d %H:%M:%S %A")}**
**农历: {_format_lunar(now)}**

# 你的身份信息

- QQ号: `{get_bot_uid()}`
- QQ昵称: `{get_bot_name()}`

# 系统通知

{self.notice if self.notice else '**无新通知**'}

# 聊天记录

```json
{json.dumps(messages, ensure_ascii=False, indent=2) if messages else '**无新消息**'}
```
"""


@dataclass(slots=True)
class PrivateSpeakerPostPayload:
    messages: list[BaseMsg]
    speak_prompt: str = ""
    working: str = ""
    relative_information: dict[str, str] = field(default_factory=dict)
    relative_memory: dict[str, str] = field(default_factory=dict)
    fav_list: list[str] = field(default_factory=list)
    participant_aliases: dict[str, list[str]] = field(default_factory=dict)

    async def get_prompt(self, wait: bool = False) -> str:
        messages: list[dict] = []
        if wait:
            tasks = [
                asyncio.create_task(
                    msg.to_llm(
                        wait=True,
                        timeout=SETTING_CFG.Groups.MessageParseTimeout,
                    )
                )
                for msg in self.messages
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for task in tasks:
                if (
                    not task.done()
                    or task.cancelled()
                    or task.exception() is not None
                ):
                    continue

                message = task.result()
                message["user_aliases"] = self.participant_aliases.get(
                    message["user_id"],
                    [],
                )
                messages.append(message)

        else:
            for message in self.messages:
                content = await message.to_llm(wait=False)
                content["user_aliases"] = self.participant_aliases.get(
                    content["user_id"],
                    [],
                )
                messages.append(content)

        fav_text = "# 表情列表\n\n**无可用表情**\n\n"
        if self.fav_list:
            fav_text = "# 表情列表\n\n" + "\n".join(self.fav_list) + "\n\n"

        information_text = ""
        if self.relative_information:
            lines = ["# 相关信息\n"]
            lines.extend(
                f"- `{key}`: {value}"
                for key, value in self.relative_information.items()
            )
            information_text = "\n".join(lines) + "\n\n"

        memory_text = ""
        if self.relative_memory:
            lines = ["# 相关记忆\n"]
            lines.extend(
                f"- `{key}`: {value}"
                for key, value in self.relative_memory.items()
            )
            memory_text = "\n".join(lines) + "\n\n"

        working_text = ""
        if self.working:
            working_text = f"# 主控正在做的事\n\n{self.working}\n\n"

        now = datetime.now()
        return f"""\
**当前时间: {timestamp2text(now.timestamp(), "%y-%m-%d %H:%M:%S %A")}**
**农历: {_format_lunar(now)}**

# Agent 主控的发言方向

{self.speak_prompt or '**无**'}

# 你的身份信息

- QQ号: `{get_bot_uid()}`
- QQ昵称: `{get_bot_name()}`

# 聊天记录

```json
{json.dumps(messages, ensure_ascii=False, indent=2) if messages else '**无新消息**'}
```

{fav_text}{information_text}{memory_text}{working_text}""".rstrip()


class SpeakResponseDict(TypedDict):
    message_sequence: list[str]
    message_content: dict[str, dict]
    to_agent: str
    intention: dict


@dataclass(slots=True)
class PrivateSpeakTask:
    call_id: str
    speak_prompt: str = ""
    working: str = ""
    relative_information: dict[str, str] = field(default_factory=dict)
    relative_memory: dict[str, str] = field(default_factory=dict)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    delivered: asyncio.Event = field(default_factory=asyncio.Event)
    result: SpeakResponseDict | None = None
    exception: BaseException | None = None
    merged_to: str = ""


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    return stripped[start:end + 1] if 0 <= start <= end else stripped


def _speaker_failure_guidance(retryable: bool) -> str:
    if retryable:
        return (
            "这是可重试失败，可根据当前聊天需要再次调用 speak；"
            "如果连续触发相同错误，请停止继续重试并等待下一次触发。"
        )

    return "这是不可重试失败，请勿重试，等待下一次自然触发。"


def _format_lunar(dt: datetime) -> str:
    lunar = cnlunar.Lunar(dt)
    solar_term = f" {lunar.todaySolarTerms}" if lunar.todaySolarTerms != "无" else ""
    festival = f" {lunar.get_legalHolidays()}" if lunar.get_legalHolidays() else ""
    return (
        f"{lunar.lunarYearCn}年{lunar.lunarMonthCn}{lunar.lunarDayCn} "
        f"{lunar.chineseYearZodiac}年{solar_term}{festival}"
    )


class PrivateChatSpeaker:
    def __init__(
        self,
        *,
        host: PrivateChatAgentHost,
        character: str,
        llm_preset: str,
        logger: BlockHandle | Logger | None = None,
    ) -> None:
        # 标识符
        self._initiated = False
        self.last_message = ""
        self.last_failure_reason = ""
        self.last_failure_retryable = False
        self._run_rounds = 0
        self._force_refresh = False
        self._force_clear = False
        self._running = False

        # 锁
        self._request_lock = asyncio.Lock()

        # 数据结构
        self._host = host
        self.character = character
        self.llm_preset = llm_preset
        self.aqueue = host.aqueue
        self.system_prompt = AgentPrompt(
            name="system_prompt",
            data_path=PROMPTS_DIR / "private_speak_system",
            description="私聊发言 LLM 系统提示词",
            role="system",
        )
        self._speak_queue: deque[PrivateSpeakTask] = deque()
        self._allow_key_type = {"reply", "break", "text", "image", "fav"}

        # 工具对象
        self.HISTORY = host.message_history
        self.FAV = host.FAV
        self.logger = logger or _log
        self.LLM = Chat(logger=self.logger)
        if llm_preset:
            setting = parse_llm_setting(llm_preset)
            if setting is not None:
                setting.keep_alive = True
                self.LLM.setting(setting)

    async def initiate(self) -> None:
        if self._initiated:
            return

        self.LLM.add_context(SystemPrompt(text=await self.system_prompt.get_prompt(self.character)))
        self._initiated = True

    async def _refresh_context(self) -> None:
        context = Context()
        context.append(SystemPrompt(text=await self.system_prompt.get_prompt(self.character)))
        self.LLM.replace_context(context)
        if self._force_clear:
            latest = self.HISTORY.get_last_message(1)
            if latest:
                self.last_message = latest[0].msg_id

        self._force_clear = False
        self._force_refresh = False
        self._run_rounds = 0

    def mark_failure(self, reason: str, *, retryable: bool = False) -> None:
        self.last_failure_reason = reason
        self.last_failure_retryable = retryable

    def _build_payload(
        self,
        *,
        speak_prompt: str = "",
        working: str = "",
        relative_information: dict[str, str] | None = None,
        relative_memory: dict[str, str] | None = None,
    ) -> PrivateSpeakerPostPayload:
        """立即取得聊天记录和参数快照，后续调用不得修改该 payload。"""
        if not self.last_message:
            messages = self.HISTORY.get_last_message(
                SETTING_CFG.Agent.SpeakerInitInputMessages
            )

        else:
            messages = self.HISTORY.get_new_message_start_from(self.last_message)

        character = get_character(self.character)
        private_fav_list = list(self.FAV.fav_list.keys())
        character_fav_list = list(character.fav.fav_list.keys())
        return PrivateSpeakerPostPayload(
            messages=list(messages),
            speak_prompt=speak_prompt,
            working=working,
            relative_information=dict(relative_information or {}),
            relative_memory=dict(relative_memory or {}),
            fav_list=private_fav_list + character_fav_list,
            participant_aliases=self._host.participant_aliases,
        )

    async def speak(self, task: PrivateSpeakTask) -> None:
        async with self._request_lock:
            if self._running:
                self._speak_queue.append(task)
                return

            self._running = True
            asyncio.create_task(self._run_task_chain(task))

    async def _run_task_chain(self, task: PrivateSpeakTask) -> None:
        current: PrivateSpeakTask | None = task
        while current is not None:
            await self._run_single_task(current)
            await current.delivered.wait()
            current = await self._take_next_merged_task()

    async def _run_single_task(self, task: PrivateSpeakTask) -> None:
        payload = self._build_payload(
            speak_prompt=task.speak_prompt,
            working=task.working,
            relative_information=task.relative_information,
            relative_memory=task.relative_memory,
        )

        try:
            result = await self._request_payload(payload)
            if result is None:
                reason = self.last_failure_reason or "私聊 Speaker 未返回发言结果"
                task.exception = RuntimeError(reason)

            else:
                task.result = result

        except asyncio.CancelledError as exc:
            self.logger.error("发言被取消")
            task.exception = exc

        except Exception as exc:
            self.logger.error(f"发言异常: {exc}")
            self.logger.debug(traceback.format_exc())
            task.exception = exc

        finally:
            task.event.set()

    async def _request_payload(
        self,
        payload: PrivateSpeakerPostPayload,
    ) -> SpeakResponseDict | None:
        timeout = float(SETTING_CFG.LLM.SpeakerRequestTimeout) + 30
        task = self.aqueue.add_task(self._speak_payload(payload), timeout=timeout)
        await task.wait()
        if task.cancelled or not task.result:
            reason = "私聊 Speaker LLM 任务执行异常或超时"
            if task.exception:
                reason = f"{reason}: {task.exception[-1]}"

            self.mark_failure(reason, retryable=True)
            return None

        result: SpeakResponseDict | None = task.result[0]
        if result is None:
            if not self.last_failure_reason:
                self.mark_failure("私聊 Speaker 未返回发言结果", retryable=True)

            return None

        if not isinstance(result, dict):
            if not self.last_failure_reason:
                self.mark_failure("私聊 Speaker 未返回合法发言结果", retryable=True)

            return None

        # noinspection PyInvalidCast
        return cast(SpeakResponseDict, result)

    async def _take_next_merged_task(self) -> PrivateSpeakTask | None:
        async with self._request_lock:
            if not self._speak_queue:
                self._running = False
                return None

            tasks = list(self._speak_queue)
            self._speak_queue.clear()

        return self._merge_tasks(tasks)

    @staticmethod
    def _merge_tasks(tasks: list[PrivateSpeakTask]) -> PrivateSpeakTask:
        target = tasks[-1]
        speak_prompts: list[str] = []
        information: dict[str, str] = {}
        memory: dict[str, str] = {}
        working = ""

        for task in tasks:
            if task.speak_prompt and task.speak_prompt not in speak_prompts:
                speak_prompts.append(task.speak_prompt)

            if task.working:
                working = task.working

            information.update(task.relative_information)
            memory.update(task.relative_memory)

        for task in tasks[:-1]:
            task.merged_to = target.call_id
            task.event.set()

        target.speak_prompt = "\n\n".join(speak_prompts)
        target.working = working
        target.relative_information = information
        target.relative_memory = memory
        return target

    async def _speak_payload(
        self,
        payload: PrivateSpeakerPostPayload,
    ) -> SpeakResponseDict | None:
        """执行一份不可变的 Speaker payload，只负责请求和解析发言。"""
        await self.initiate()
        self.last_failure_reason = ""
        self.last_failure_retryable = False
        prompt = await payload.get_prompt()
        previous = next(
            (msg for msg in reversed(payload.messages) if msg.done or msg.broken),
            None,
        )
        if (
            self._force_refresh
            or self._force_clear
            or self._run_rounds > self._host.speaker_round_limit
        ):
            await self._refresh_context()

        character = get_character(self.character)
        for _ in range(2):
            try:
                response = await self.LLM.ask(prompt, timeout=SETTING_CFG.LLM.SpeakerRequestTimeout)

            except ContextOverflowError:
                await self._refresh_context()
                response = await self.LLM.ask(prompt, timeout=SETTING_CFG.LLM.SpeakerRequestTimeout)

            if not response or not isinstance(response[0], TextPrompt):
                self.mark_failure("私聊 Speaker 未返回文本响应", retryable=True)
                continue

            try:
                raw = json.loads(_extract_json_object_text(response[0].text))

            except json.JSONDecodeError:
                self.mark_failure("私聊 Speaker 返回内容不是合法 JSON")
                continue

            if not isinstance(raw, dict) or set(raw) != {
                "message_sequence",
                "message_content",
                "to_agent",
                "intention",
            }:
                self.mark_failure("私聊 Speaker 返回字段不完整")
                continue

            sequence = raw["message_sequence"]
            content = raw["message_content"]
            to_agent = raw["to_agent"]
            intention = raw["intention"]
            if (
                not isinstance(sequence, list)
                or not isinstance(content, dict)
                or not isinstance(to_agent, str)
                or not isinstance(intention, dict)
            ):
                self.mark_failure("私聊 Speaker 返回字段类型错误")
                continue

            clean: dict[str, dict] = {}
            invalid = False
            combined_fav = self.FAV | character.fav
            for key in sequence:
                if not isinstance(key, str) or key not in content or key in clean:
                    invalid = True
                    break

                key_type = key.split("_", 1)[0]
                if key_type not in self._allow_key_type:
                    invalid = True
                    break

                if key_type == "fav":
                    title = content[key].get("fav_title", "")
                    hash_name = combined_fav.get_fav(title)
                    if hash_name:
                        clean[key] = {"hash_name": hash_name}

                else:
                    clean[key] = content[key]

            if invalid or not clean:
                self.mark_failure("私聊 Speaker 未返回可发送的消息", retryable=True)
                continue

            self._run_rounds += 1
            if previous is not None:
                self.last_message = previous.msg_id

            return SpeakResponseDict(
                message_sequence=list(clean),
                message_content=clean,
                to_agent=to_agent,
                intention=intention,
            )

        return None

    def force_refresh(self) -> None:
        self._force_refresh = True

    def clear_history(self) -> None:
        self._force_clear = True

    async def shutdown(self) -> None:
        async with self._request_lock:
            while self._speak_queue:
                task = self._speak_queue.popleft()
                task.exception = asyncio.CancelledError()
                task.event.set()
            self._running = False
        await self.LLM.close()

    def switch_model(self, llm_preset: str) -> bool:
        setting = parse_llm_setting(llm_preset)
        if setting is None:
            return False

        self.llm_preset = llm_preset
        self.LLM.setting(setting)
        return True

    @property
    def running(self) -> bool:
        return self._running


class PrivateChatAgent(BaseAgent):
    def __init__(
        self,
        *,
        host: PrivateChatAgentHost,
        llm_preset: str = "",
        agent_name: str = "",
        control_timeout: float = -1,
        logger: BlockHandle | Logger | None = None,
    ) -> None:
        super().__init__(
            host_id=host.user_id,
            data_path=host.data_path,
            llm_preset=llm_preset,
            agent_name=agent_name,
            system_prompt=PROMPTS_DIR / "private_chat_system",
            control_timeout=control_timeout,
            logger=logger or _log,
        )
        self._host = cast(PrivateChatAgentHost, weakref.proxy(host))
        self.last_message = ""
        self._run_rounds = 0
        self._force_refresh = False
        self._force_clear = False
        self._agent_post_queue: asyncio.Queue[PrivatePostTask] = asyncio.Queue()
        self._run_loop: asyncio.Task | None = None
        self._speak_request_generation = 0
        self._force_speak_observers: set[asyncio.Task] = set()
        self._memory_lock = asyncio.Lock()
        self._last_add_memory_time = 0.0
        self.force_save_memory_retry = 0
        self._speak_prompt_times = 0
        self._not_speak_information_times = 0
        self.said_review_count = 0
        self.said_intention: dict[str, dict] = {}
        self._said_review_lock = asyncio.Lock()
        self.functions_default_setting["speak"] = {"wait": False, "callback": "EXCEPTION"}
        self.functions_default_setting["_speaker_to_llm_callback"] = {
            "wait": True,
            "callback": "NEVER",
        }

    def _load_internal_tools(self):
        super()._load_internal_tools()
        call_name = self.register_function(self.speak, display_name="构造私聊发言中")
        self.tools[call_name] = self.speak

    async def before_initiate(self):
        await super().before_initiate()
        self.register_function(
            self._speaker_to_llm_callback,
            display_name="私聊发言 LLM 返回消息",
        )
        self.register_prompt(
            name="refresh_context_prompt",
            path=PROMPTS_DIR / "private_chat_refresh_context",
            description="私聊 Agent 刷新 Context 的 payload",
            role="user",
        )
        source = BASE_SKILL_DIR / "private_chat"
        registrations = (
            (self.get_single_message, "获取一条私聊消息"),
            (self.get_time_interval_messages, "获取时间段内私聊消息"),
            (self.get_image_detail, "获取详细图片内容"),
            (self.get_news_by_date, "获取指定日期新闻"),
            (self.fetch_news, "获取新的新闻"),
            (self.add_memory, "新增私聊记忆"),
            (self.query_memory, "查询私聊关系记忆"),
            (self.get_persona, "获取完整私聊画像"),
            (self.set_alias, "设置私聊参与者爱称"),
            (self.get_fav_list, "获取当前表情列表"),
            (self.add_private_fav, "添加一张私聊表情"),
        )
        for function, display_name in registrations:
            self.register_function(
                function,
                source=source,
                source_name="private_chat",
                display_name=display_name,
            )

        private_setting = self._skills.tools_default_setting.setdefault("private_chat", {})
        private_setting["get_single_message"] = {"wait": True, "callback": "CALL"}
        private_setting["get_time_interval_messages"] = {"wait": True, "callback": "CALL"}
        private_setting["get_image_detail"] = {"wait": False, "callback": "ALWAYS"}
        private_setting["get_news_by_date"] = {"wait": True, "callback": "CALL"}
        private_setting["fetch_news"] = {"wait": False, "callback": "ALWAYS"}
        private_setting["add_memory"] = {"wait": False, "callback": "EXCEPTION"}
        private_setting["query_memory"] = {"wait": False, "callback": "ALWAYS"}
        private_setting["get_persona"] = {"wait": False, "callback": "ALWAYS"}
        private_setting["set_alias"] = {"wait": False, "callback": "EXCEPTION"}
        private_setting["get_fav_list"] = {"wait": True, "callback": "CALL"}
        private_setting["add_private_fav"] = {"wait": False, "callback": "EXCEPTION"}

    async def run_in_queue(
        self,
        reason: PrivatePostReason,
        timeout: float | None = None,
    ) -> None:
        if timeout is None:
            timeout = SETTING_CFG.PrivateChat.AgentPostQueueTimeout

        task = PrivatePostTask(reason=reason, timeout=timeout)
        self._ensure_running_loop()
        await self._agent_post_queue.put(task)

        try:
            await asyncio.wait_for(task.expired.wait(), timeout=timeout)

        except asyncio.TimeoutError:
            task.expired.set()

    def _ensure_running_loop(self) -> None:
        if self._run_loop is None or self._run_loop.done():
            self._run_loop = asyncio.create_task(self._run_with_payload())

    async def _payload_constructor(self, task: PrivatePostTask) -> PrivateAgentPostPayload:
        if not self.last_message:
            messages = self._host.message_history.get_last_message(
                SETTING_CFG.Agent.AgentInitInputMessages
            )

        else:
            messages = self._host.message_history.get_new_message_start_from(self.last_message)

        return PrivateAgentPostPayload(
            reason=task.reason,
            messages=messages,
            notice=self._host.NOTICE.notice2text(),
            participant_aliases=self._host.participant_aliases,
            said_intention=self.said_intention,
        )

    async def after_run(self, run_result: Prompts | None) -> None:
        await super().after_run(run_result)
        asyncio.create_task(self._auto_said_review())
        save_notice = self._host.memory_save_notice
        if not save_notice:
            return

        retry_limit = SETTING_CFG.Agent.ForceSaveMemoryRetry
        if self._host.NOTICE.is_notice_alive(save_notice):
            if self.force_save_memory_retry < retry_limit:
                self.force_save_memory_retry += 1
                # 构造保存记忆 Recat
                # noinspection PyAsyncCall
                self._host.aqueue.add_task(
                    self.run_in_queue(PrivatePostReason.SAVE_MEMORY),
                    timeout=-1,
                )

            else:
                self._host.logger.warning("已达到强制保存记忆 React 重试上限，但主控仍未保存记忆")

    async def _run_with_payload(self) -> None:
        while self._initiated:
            try:
                post_task = await self._agent_post_queue.get()
                if post_task.expired.is_set():
                    continue

                post_task.expired.set()
                if post_task.reason in _FORCE_SPEAK:
                    self._schedule_force_speak_fallback()

                payload = await self._payload_constructor(post_task)
                previous = next(
                    (msg for msg in reversed(payload.messages) if msg.done or msg.broken),
                    None,
                )
                prompt = TextPrompt(role="user", text=await payload.get_prompt())
                response = await self.run(await self._refresh_context_helper(prompt))
                if response is not None:
                    self._run_rounds += 1
                    if previous is not None:
                        self.last_message = previous.msg_id

            except asyncio.CancelledError:
                return

            except Exception as exc:
                self._logger.error(f"私聊主控任务队列异常: {exc}")
                self._logger.debug(traceback.format_exc())

    async def _refresh_context_helper(self, prompt: TextPrompt) -> Prompts:
        prompts = Prompts(prompt)
        if self._force_clear:
            await self.reload_system_prompt()
            self._force_clear = False
            return await self.compress_context(prompts, 0)

        if self._force_refresh or self._run_rounds > self._host.agent_round_limit:
            return await self.compress_context(prompts)

        return prompts

    def _schedule_force_speak_fallback(self) -> None:
        generation = self._speak_request_generation
        timeout = float(SETTING_CFG.PrivateChat.ReplyFallbackTimeout)
        observer = asyncio.create_task(
            self._observe_force_speak_fallback(generation, timeout)
        )
        self._force_speak_observers.add(observer)
        observer.add_done_callback(self._force_speak_observers.discard)

    async def _observe_force_speak_fallback(
        self,
        generation: int,
        timeout: float,
    ) -> None:
        try:
            if timeout > 0:
                await asyncio.sleep(timeout)

            if not self._initiated or self._speak_request_generation != generation:
                return

            # 比较与递增之间没有 await，保证多个到期观察器只会有一个胜出。
            self._speak_request_generation += 1
            await self.add_task(
                function_name="speak",
                description="系统自动创建的强制发言任务",
                call_id=f"system_{next_time_id()}",
                callback="EXCEPTION",
            )

        except asyncio.CancelledError:
            return

        except Exception as exc:
            self._logger.error(f"私聊自动发言保底失败: {exc}")

    async def _cancel_force_speak_observers(self) -> None:
        observers = list(self._force_speak_observers)
        for observer in observers:
            observer.cancel()
        if observers:
            await asyncio.gather(*observers, return_exceptions=True)
        self._force_speak_observers.clear()

    async def handle_response(self, response: Prompts):
        if any(
            isinstance(prompt, ToolCall) and prompt.function_name == "speak"
            for prompt in response
        ):
            self._speak_request_generation += 1

        await super().handle_response(response)

    async def compress_context(
        self,
        new_prompt: Prompts | None = None,
        old_messages_count: int | None = None,
    ) -> Prompts:
        pinned = self._get_compress_context_pinned()
        suffix = self._detach_tool_calls_for_context_reset("PrivateChatAgent 刷新 Context")
        images: list[ImagePrompt] = []
        request: TextPrompt | None = None
        if new_prompt is not None:
            for prompt in new_prompt:
                if isinstance(prompt, TextPrompt):
                    request = prompt

                elif isinstance(prompt, ImagePrompt):
                    images.append(prompt)

                elif isinstance(prompt, ToolResponse):
                    suffix += f"\n丢失的 ToolResponse: {prompt.to_dict()}\n"

        refresh = self.get_last_version_prompt("refresh_context_prompt")
        text = ""
        if refresh is not None:
            text = await refresh.get_prompt(
                user_id=self._host.user_id,
                username=self._host.username,
                private_summary=await self._host.get_private_summary(True),
                old_messages=self._host.get_old_message(old_messages_count),
                news=await self._host.get_private_news(),
                new_request=request,
            )

        self._history.compress_context(data="刷新 Context 窗口")
        await self._save_checkpoint_async()
        self.control.replace_context(self._context.new())
        self._force_refresh = False
        self._run_rounds = 0
        await self.save_agent_async(save_context_history=True)
        return Prompts(TextPrompt(role="user", text=pinned + suffix + text), *images)

    async def speak(
        self,
        *,
        speak_prompt: str = "",
        working: str = "",
        relative_information: dict[str, str] | None = None,
        relative_memory: dict[str, str] | None = None,
    ) -> str | None:
        """
        发言接口，让 BOT 发言一次
        :param speak_prompt: 发言方向或指示、主要提示词，可选。
        :param working: 正在做的事情或任务状态，可选。
        :param relative_information: 相关信息，例如新闻、websearch结果、外部数据，使用 {"title": "content"}输入，可选。
        :param relative_memory: 相关记忆，采用 {"title": "content"}输入，可选。
        :return: 工具执行的任务信息、id
        """
        task_id = self.get_self_id()
        failure_reason = ""
        failure_retryable = False
        if speak_prompt:
            self._speak_prompt_times += 1

        else:
            self._speak_prompt_times = 0

        if not (relative_information or relative_memory):
            self._not_speak_information_times += 1

        else:
            self._not_speak_information_times = 0

        if self._speak_prompt_times > 3:
            self._host.NOTICE.add_notice(
                content="作为主控，你已连续多次传递 `speak_prompt` 干涉发言。**切勿干涉过多**，遵守规则，不要管太多 Speaker 自己能处理的事情",
                title="发言规范",
                priority=35,
                alive_until_self_messages=2,
                exist_ok=True,
            )

        if self._not_speak_information_times > 3:
            self._host.NOTICE.add_notice(
                content="作为主控，你已多次未传递任何相关信息和记忆调用 `speak`，你需要 **积极传递** 信息给 Speaker",
                title="发言规范",
                priority=35,
                alive_until_self_messages=2,
                exist_ok=True,
            )

        try:
            result = await self._host.speak(
                speak_prompt=speak_prompt,
                working=working,
                relative_information=relative_information,
                relative_memory=relative_memory,
                call_id=task_id or "",
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            failure_reason = f"发言链路异常: {exc}"
            failure_retryable = True
            self._logger.error(f"{failure_reason}")
            self._logger.debug(traceback.format_exc())
            result = None

        if result is None:
            if not failure_reason:
                speaker = getattr(self._host, "SPEAKER", None)
                if speaker is not None:
                    failure_reason = getattr(speaker, "last_failure_reason", "")
                    failure_retryable = getattr(
                        speaker,
                        "last_failure_retryable",
                        False,
                    )

            if not failure_reason:
                failure_reason = "发言任务失败，但未记录到具体原因"

            message = (
                f"发言任务[{task_id}]失败。原因: {failure_reason}。"
                f"{_speaker_failure_guidance(failure_retryable)}"
            )
            self._logger.warning(f"{message}")
            asyncio.create_task(
                self.add_task(
                    function_name="_speaker_to_llm_callback",
                    params={"message": message},
                    call_id=task_id,
                    pin=True,
                    callback="ALWAYS",
                    wait=True,
                )
            )
            return message

        if result:
            asyncio.create_task(
                self.add_task(
                    function_name="_speaker_to_llm_callback",
                    params={"message": f"发言任务[{task_id}]返回给主控的信息:\n\n{result}"},
                    call_id=task_id,
                    pin=True,
                    callback="ALWAYS",
                    wait=True,
                )
            )
        return None

    @staticmethod
    def _speaker_to_llm_callback(message: str) -> str:
        """SPEAKER的回调"""
        return message

    @staticmethod
    def _parse_intention(msg_id: str, intention: dict[str, dict]) -> str:
        content = intention.get(msg_id, {})
        if not content:
            return ""

        output = "本次发言意图: "
        target_message_id = content.get("target_message_id", "")
        if target_message_id:
            output += f"响应消息 [{target_message_id}]; "

        intent = content.get("intent", "")
        if intent in _SPEAK_INTENT_MAPPING:
            output += f"发言行为类别: {_SPEAK_INTENT_MAPPING[intent]}; "

        effect = content.get("effect", "")
        if effect in _SPEAK_INTENT_MAPPING:
            output += f"期望发言效果: {_SPEAK_INTENT_MAPPING[effect]}"

        return output.rstrip("; ")

    async def _auto_said_review(self) -> None:
        if self.said_review_count < SETTING_CFG.Agent.SaidReviewPeriod:
            return

        async with self._said_review_lock:
            if self.said_review_count < SETTING_CFG.Agent.SaidReviewPeriod:
                return

            count = self.said_review_count
            self.said_review_count = 0
            intention_mirror = self.said_intention.copy()
            self.said_intention.clear()
            bot_said = self._host.message_history.get_message_from_user(
                user_id=str(get_bot_uid())
            )
            recent_said = bot_said[-count:]

            tasks = [
                asyncio.create_task(
                    message.to_llm(
                        wait=True,
                        timeout=SETTING_CFG.Groups.MessageParseTimeout,
                    )
                )
                for message in recent_said
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            messages: list[dict] = []
            for task in tasks:
                if (
                    not task.done()
                    or task.cancelled()
                    or task.exception() is not None
                ):
                    continue

                message = task.result()
                message["user_aliases"] = self._host.participant_aliases.get(
                    message["user_id"],
                    [],
                )
                intention = self._parse_intention(
                    message["message_id"],
                    intention_mirror,
                )
                if intention:
                    message["message_tips"].append(intention)

                messages.append(message)

            payload = (
                "**本次请求为 *BOT 发言结果审阅任务*。请勿发言**\n\n"
                "# 近期 BOT 发言记录\n\n"
                f"```json\n{json.dumps(messages, ensure_ascii=False, indent=2)}\n```\n\n\n"
                "你需要结合完整聊天记录，根据规则进行审阅、质量评估，并在后续发言任务里，"
                "通过 `relative_information` 把反馈给 Speaker"
            )
            # noinspection PyAsyncCall
            self._host.aqueue.add_task(
                self.run(TextPrompt(role="user", text=payload), from_user=False),
                timeout=-1,
            )

    # =========================================================
    # private_chat SKILL 工具区
    # =========================================================

    async def get_single_message(self, message_id: str) -> dict:
        """
        获取一条聊天消息的详情，会等待所有内容解析完成再返回。
        :param message_id: 消息id(msg_id)。
        :return: 成功: 标准聊天消息字典; 消息不存在/已过期: 空字典
        """
        message = await self._host.message_history.get_message(message_id)
        if message is None:
            return {}
        timeout = SETTING_CFG.Groups.MessageParseTimeout
        return await message.to_llm(wait=True, timeout=timeout)

    async def get_time_interval_messages(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        """
        获取一个时间段内的所有聊天消息，会等待所有消息的所有内容解析完成再返回
        :param start: 起始时间: 标准 `ISO 8601`字符串(无时区)，若不传入则为聊天记录中最早的消息时间。
        :param end: 结束时间: 标准 `ISO 8601`字符串(无时区)，若不传入则为聊天记录中最新的消息时间。
        :return: 成功: 一个标准聊天消息的字典列表; 无消息: 空列表; 失败: 抛出异常
        """
        start_ts = datetime.fromisoformat(start).astimezone().timestamp() if start else None
        end_ts = datetime.fromisoformat(end).astimezone().timestamp() if end else None
        messages = self._host.message_history.get_time_interval_message(start_ts, end_ts)
        timeout = SETTING_CFG.Groups.MessageParseTimeout
        tasks = [
            asyncio.create_task(msg.to_llm(wait=True, timeout=timeout))
            for msg in messages
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return [
            task.result()
            for task in tasks
            if task.done() and not task.cancelled() and task.exception() is None
        ]

    @staticmethod
    async def get_image_detail(image_hash_name: str, query: str, image_name: str = "") -> dict:
        """
        调用一个大型多模态模型去更仔细的识别图片。
        :param image_hash_name: 聊天消息中图片内容的 `hash_name`
        :param query: 想要识别的方向，自然语言描述
        :param image_name: 若图片来自聊天消息，可把图片内容中固定 `image_name`也输入，有助于缓存系统存储
        :return: 成功/失败: 一个识图结果字典
        """
        try:
            image_file = await get_file_path_async(image_hash_name)
            with Image.open(image_file) as image:
                image.verify()

        except (FileNotFoundError, ValueError, UnidentifiedImageError):
            return {"success": False, "answer": "", "message": "图片不存在或已损坏"}

        answer = await image2text(hash_name=image_hash_name, query=query, image_name=image_name)
        return {"success": bool(answer), "answer": answer or "", "message": "" if answer else "识图无结果"}

    async def get_news_by_date(self, date: str | None = None) -> dict:
        """
        获取指定日期的新闻。
        :param date: 日期: 标准 `ISO 8601`字符串(无时区)，不传入则默认为今日。
        :return: 成功: 该日期的新闻字典; 无新闻: 空字典; 失败: 抛出异常
        """
        return await self._host.get_news_by_date(date)

    async def fetch_news(self, topic: str) -> dict:
        """
        调用 WebSearch 去获取指定话题的新闻。
        :param topic: 话题: 自然语言描述，一次只获取一个话题的新闻。尽量使用关键词、名词。
        :return: 成功: 新闻字典; 无新闻: 空字典; 失败: 抛出异常
        """
        results = await self._websearch.search(f"有关 [{topic}] 的新闻或相关信息", purge=True)
        news = [item["results"] for item in results if item.get("ok")]
        if not news:
            return {}

        key = timestamp2text(datetime.now().timestamp(), "%y-%m-%d")
        today_news = self._host.relative_news.setdefault(key, {})
        today_news.setdefault(topic, []).extend(news)
        return {"date": key, "news": {topic: news}}

    async def add_memory(self, memory: str) -> dict:
        """
        新增记忆。
        :param memory: 记忆内容，可以字符串或者自然语言表示，也可以结构化。
        :return: 记忆结果字典
        """
        async with self._memory_lock:
            if time.time() - self._last_add_memory_time < 15:
                return {"success": False, "memory": "", "message": "添加记忆过于频繁"}

            result = await self._host.add_memory(memory)
            self._last_add_memory_time = time.time()
            return result

    async def query_memory(self, query: str) -> dict:
        """
        全局搜索指定内容的记忆。
        :param query: 需要搜索的内容，自然语言表示。
        :return: 成功/失败: 记忆结果字典
        """
        return await self._host.query_memory(query)

    async def get_persona(self) -> dict:
        """
        获取私聊画像
        :return: 成功/失败: 私聊画像字典
        """
        return dict((await self._host.get_private_persona()).to_dict())

    async def set_alias(self, user_id: str, alias: str) -> None:
        """
        设置当前私聊用户或 BOT 的爱称。
        :param user_id: 当前私聊用户或 BOT 的 QQ 号。
        :param alias: 新增的爱称。
        :return: 成功: 无返回值; 参与者不存在或记忆写入失败: 抛出异常
        """
        return await self._host.set_alias(user_id, alias)

    def get_fav_list(self) -> dict:
        """
        获取最新的表情列表，分为私聊表情和 Speaker 当前角色的表情
        :return: 成功: 表情列表结果字典; 无表情: 结果字典，带有空列表; 失败: 抛出异常
        """
        class _FavResult(TypedDict):
            private_fav: list[str]
            character_fav: list[str]

        private_fav = list(self._host.FAV.fav_list.keys())
        speaker = getattr(self._host, "SPEAKER", None)
        character_name = getattr(speaker, "character", "")
        if not character_name:
            return dict(_FavResult(
                private_fav=private_fav,
                character_fav=[],
            ))

        return dict(_FavResult(
            private_fav=private_fav,
            character_fav=list(get_character(character_name).fav.fav_list.keys()),
        ))

    async def add_private_fav(self, hash_name: str, fav_title: str = "") -> dict:
        """
        添加一张私聊表情。
        :param hash_name: 表情图片的 `hash_name`。
        :param fav_title: 表情标题，尽量15字内，若不传入则调用识图模型自动获取。
        :return: 成功: 一个结果字典; 失败: 抛出异常。仅失败会回调。
        """
        class _AddFavResult(TypedDict):
            success: bool
            hash_name: str
            fav_title: str
            fav_type: str
            message: str

        # 让异常直接抛出，避免损坏图片写入表情库。
        image_path = await get_file_path_async(hash_name)
        with Image.open(image_path) as image:
            image.verify()

        title = await self._host.FAV.add_fav(hash_name, fav_title)
        return dict(_AddFavResult(
            success=True,
            hash_name=hash_name,
            fav_title=title,
            fav_type="private",
            message="成功添加私聊表情",
        ))

    async def start(
        self,
        system_prompt: AgentPrompt | None = None,
        initiate_prompt: AgentPrompt | None = None,
    ) -> None:
        await super().start(system_prompt, initiate_prompt)
        self._run_loop = asyncio.create_task(self._run_with_payload())

    async def shutdown(self) -> None:
        await self._cancel_force_speak_observers()
        pending: list[asyncio.Task] = []
        if self._run_loop is not None:
            self._run_loop.cancel()
            pending.append(self._run_loop)

        await super().shutdown()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def clear_history(self) -> None:
        self._force_clear = True

    def force_refresh(self) -> None:
        self._force_refresh = True

    def switch_model(self, llm_preset: str) -> bool:
        try:
            setting = parse_llm_setting(llm_preset)

        except (TypeError, ValueError):
            return False

        if setting is None:
            return False

        self.modify_agent(llm_preset=llm_preset)
        return True

    @property
    def run_queue_empty(self) -> bool:
        return self._agent_post_queue.empty()


__all__ = [
    "PrivateAgentPostPayload",
    "PrivateChatAgent",
    "PrivateChatSpeaker",
    "PrivatePostReason",
    "PrivatePostTask",
    "PrivateSpeakerPostPayload",
    "post_reason_prompt_maker",
]
