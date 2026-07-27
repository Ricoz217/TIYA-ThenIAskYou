from __future__ import annotations
"""
group_chat_agent.py
群聊的主 Agent
"""
__version__ = "0.1.0"

import time
import weakref
import asyncio
import json
import traceback
import cnlunar
from typing import TYPE_CHECKING, Protocol, cast, TypedDict
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from PIL import Image, UnidentifiedImageError

from TIYA.LLM_connect import (
    Chat,
    Context,
    SystemPrompt,
    TextPrompt,
    ToolCall,
    ToolResponse,
    ImagePrompt,
    ContextOverflowError,
    Prompts,
    parse_llm_setting
)
from TIYA.auto_fav import get_autofav
from TIYA.file_cache import add_file_async, check_file_exists, get_file_path_async
from TIYA.time_id import next_time_id
from TIYA.image_to_text import image2text
from TIYA.model.character import get_character
from TIYA.model.message import MessageManager, BaseMsg, GroupMsg
from TIYA.config import PROMPTS_DIR, BASE_SKILL_DIR, SETTING_CFG, get_bot_uid, get_bot_name
from TIYA.logger import get_logger, Logger, BlockHandle
from TIYA.utils import timestamp2text
from .agent import BaseAgent
from .agent_prompt import AgentPrompt

_log = get_logger()
_MEMBER_ROLE_MAPPING = {
    "member": "普通群员",
    "admin": "管理员",
    "owner": "群主",
    "robot": "群机器人"
}
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
    "cool_down": "降低/缓和当前话题的氛围/热度",
    "neutral": "保持当前话题的氛围/热度",
    "heat_up": "增加/升温当前话题的氛围/热度"
}

if TYPE_CHECKING:
    from TIYA.model.chat_notification import ChatNotification
    from TIYA.dialog.group_dialog import GroupMainDialog
    from TIYA.qq_group import Member


class GroupChatAgentHost(Protocol):
    group_id: str
    data_path: Path
    message_history: MessageManager
    NOTICE: ChatNotification


class PostReason(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    CALL_OR_REPLY = auto()
    AGENT_INTERNAL = auto()
    INITIATE = auto()
    RANDOM_SPEAK = auto()
    SAVE_MEMORY = auto()


_FORCE_SPEAK = {PostReason.CALL_OR_REPLY, PostReason.RANDOM_SPEAK}
_CHOICE_SPEAK = {PostReason.AGENT_INTERNAL}
_AVOID_SPEAK = {PostReason.INITIATE, PostReason.SAVE_MEMORY}
_NEED_WAIT_PARSE = {PostReason.CALL_OR_REPLY}
_FORCE_SPEAK_OBSERVATION_SECONDS = 10.0

def post_reason_prompt_maker(reason: PostReason) -> str:
    force_speak = "必须调用 `speak` 发言"
    choice_speak = "自行判断是否发言，优先不发言"
    avoid_speak = "请勿发言"

    reason_mapping = {
        PostReason.CALL_OR_REPLY: "BOT被@到或他人回复了BOT",
        PostReason.AGENT_INTERNAL: "Agent内部React循环",
        PostReason.INITIATE: "Agent初始化",
        PostReason.RANDOM_SPEAK: "BOT概率随机发言",
        PostReason.SAVE_MEMORY: "保存记忆React，若已保存记忆可跳过"
    }

    if reason in _FORCE_SPEAK:
        speak_hit = force_speak

    elif reason in _CHOICE_SPEAK:
        speak_hit = choice_speak

    elif reason in _AVOID_SPEAK:
        speak_hit = avoid_speak

    else:
        raise KeyError("有遗漏的请求原因")

    prompt = f"**本次请求来自 *{reason_mapping[reason]}*，{speak_hit}**"
    return prompt


@dataclass(slots=True)
class PostTask:
    reason: PostReason
    timeout: float
    force_speak: bool = False
    expired: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(slots=True)
class AgentPostPayload:
    reason: PostReason
    messages: list[BaseMsg]
    notice: str
    bot_info: Member | None
    member_alias: dict[str, list[str]] = field(default_factory=dict)
    said_intention: dict[str, dict] = field(default_factory=dict)

    def _parse_intention(self, msg_id: str) -> str:
        output = ""
        intention = self.said_intention.get(msg_id, {})
        if not intention:
            return output

        output = "本次发言意图: "
        if "target_message_id" in intention:
            target_message_id = intention["target_message_id"]
            if target_message_id:
                output += f"响应消息 [{intention['target_message_id']}]; "

        if "target_user_id" in intention:
            target_user_id = intention["target_user_id"]
            if target_user_id:
                output += f"响应群员 [{intention['target_user_id']}]; "

        if "intent" in intention:
            intent = intention["intent"]
            if intent in _SPEAK_INTENT_MAPPING:
                output += f"发言行为类别: {_SPEAK_INTENT_MAPPING[intent]}"

        if "effect" in intention:
            effect = intention["effect"]
            if effect in _SPEAK_INTENT_MAPPING:
                output += f"期望发言效果: {_SPEAK_INTENT_MAPPING[effect]}"

        return output

    async def get_prompt(self):
        if self.reason in _NEED_WAIT_PARSE:
            need_wait = True

        else:
            need_wait = False

        messages_text = []
        if need_wait:
            tasks: list[asyncio.Task] = []
            timeout = SETTING_CFG.Groups.MessageParseTimeout  # 默认15
            for msg in self.messages:
                tasks.append(asyncio.create_task(msg.to_llm(wait=True, timeout=timeout)))

            await asyncio.gather(*tasks, return_exceptions=True)

            for task in tasks:
                if not task.done() or task.cancel() or task.exception() is not None:
                    continue

                to_llm_dict = task.result()
                to_llm_dict["user_aliases"] = self.member_alias.get(to_llm_dict["user_id"], [])
                intention_text = self._parse_intention(to_llm_dict["message_id"])
                if intention_text:
                    to_llm_dict["message_tips"].append(intention_text)

                messages_text.append(to_llm_dict)

        else:
            for msg in self.messages:
                to_llm_dict = await msg.to_llm(wait=False)
                to_llm_dict["user_aliases"] = self.member_alias.get(to_llm_dict["user_id"], [])
                intention_text = self._parse_intention(to_llm_dict["message_id"])
                if intention_text:
                    to_llm_dict["message_tips"].append(intention_text)

                messages_text.append(to_llm_dict)

        messages_json = json.dumps(messages_text, ensure_ascii=False, indent=2)

        nickname = "`无法获取`"
        role = "`无法获取`"
        title = "`无法获取`"
        if self.bot_info is not None:
            if self.bot_info.nickname:
                nickname = f"`{self.bot_info.nickname}`"

            else:
                nickname = ""

            if self.bot_info.title:
                title = f"`{self.bot_info.title}`"

            else:
                title = ""

            role = f"`{_MEMBER_ROLE_MAPPING[self.bot_info.role.value]}`"

        now = datetime.now()
        text = f"""\
{post_reason_prompt_maker(self.reason)}  
**当前时间: {timestamp2text(now.timestamp(), "%y-%m-%d %H:%M:%S %A")}**  
**农历: {_format_lunar(now)}**  


# 你的身份信息

- QQ号: `{get_bot_uid()}`
- QQ昵称: `{get_bot_name()}`
- 群昵称: {nickname}
- 群头衔: {title}
- 群身份: {role}  


# 系统通知

{self.notice if self.notice else "**无新通知**"}  


# 聊天记录

```json
{messages_json if messages_json else "**无新消息**"}
```  
"""
        return text


@dataclass(slots=True)
class SpeakerPostPayload:
    speak_prompt: str = ""
    messages: list[BaseMsg] = field(default_factory=list)
    fav_list: list[str] = field(default_factory=list)
    relative_information: dict[str, str] = field(default_factory=dict)
    relative_memory: dict[str, str] = field(default_factory=dict)
    working: str = "",
    bot_info: Member | None = None
    member_alias: dict[str, list[str]] = field(default_factory=dict)

    async def get_prompt(self, wait=False):
        messages_text = []
        if wait:
            tasks: list[asyncio.Task] = []
            timeout = SETTING_CFG.Groups.MessageParseTimeout  # 默认15
            for msg in self.messages:
                tasks.append(asyncio.create_task(msg.to_llm(wait=True, timeout=timeout)))

            await asyncio.gather(*tasks, return_exceptions=True)

            for task in tasks:
                if not task.done() or task.cancel() or task.exception() is not None:
                    continue

                to_llm_dict = task.result()
                to_llm_dict["user_aliases"] = self.member_alias.get(to_llm_dict["user_id"], [])
                messages_text.append(to_llm_dict)

        else:
            for msg in self.messages:
                to_llm_dict = await msg.to_llm(wait=False)
                to_llm_dict["user_aliases"] = self.member_alias.get(to_llm_dict["user_id"], [])
                messages_text.append(to_llm_dict)

        messages_json = json.dumps(messages_text, ensure_ascii=False, indent=2)

        fav_text = "# 表情列表\n\n**无可用表情**\n\n\n"
        if self.fav_list:
            fav_lines: list[str] = ["# 表情列表\n"]
            fav_lines.extend(self.fav_list)
            fav_lines.append('\n')
            fav_text = '\n'.join(fav_lines)

        information_text = ""
        if self.relative_information:
            information_lines: list[str] = ["# 相关信息\n"]
            for k, v in self.relative_information.items():
                information_lines.append(f"- `{k}`: {v}")

            information_lines.append('\n')
            information_text = '\n'.join(information_lines)

        memory_text = ""
        if self.relative_memory:
            memory_lines: list[str] = ["# 相关记忆\n"]
            for k, v in self.relative_memory.items():
                memory_lines.append(f"- `{k}`: {v}")

            memory_lines.append('\n')
            memory_text = '\n'.join(memory_lines)

        working_text = ""
        if self.working:
            working_text = f"# 主控正在做的事\n\n{self.working}"

        nickname = "`无法获取`"
        role = "`无法获取`"
        title = "`无法获取`"
        if self.bot_info is not None:
            if self.bot_info.nickname:
                nickname = f"`{self.bot_info.nickname}`"

            else:
                nickname = ""

            if self.bot_info.title:
                title = f"`{self.bot_info.title}`"

            else:
                title = ""

            role = f"`{_MEMBER_ROLE_MAPPING[self.bot_info.role.value]}`"

        now = datetime.now()
        text = f"""\
**当前时间: {timestamp2text(now.timestamp(), "%y-%m-%d %H:%M:%S %A")}**  
**农历: {_format_lunar(now)}**  


# Agent主控的指令/发言方向  

{self.speak_prompt if self.speak_prompt else "**无**"}  


# 你的身份信息

- QQ号: `{get_bot_uid()}`
- QQ昵称: `{get_bot_name()}`
- 群昵称: {nickname}
- 群头衔: {title}
- 群身份: {role}  


# 聊天记录

```json
{messages_json if messages_json else "**无新消息**"}
```  


{fav_text}{information_text}{memory_text}{working_text}
"""
        return text.rstrip()


class _SpeakResponseDict(TypedDict):
    message_sequence: list[str]
    message_content: dict[str, dict]
    to_agent: str
    intention: dict


def _extract_json_object_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start:end + 1]

    return text


def _speaker_failure_guidance(retryable: bool) -> str:
    if retryable:
        return (
            "这是可重试失败，可根据当前聊天需要再次调用 speak；"
            "如果连续触发相同错误，可能是系统异常，请停止继续重试并等待下一次触发。"
        )

    return "这是不可重试失败，请勿重试，等待下一次自然触发。"


def _format_lunar(dt: datetime) -> str:
    lunar = cnlunar.Lunar(dt)
    shengxiao = lunar.chineseYearZodiac
    jieqi = f" {lunar.todaySolarTerms}" if lunar.todaySolarTerms != "无" else ""
    festival = f" {lunar.get_legalHolidays()}" if lunar.get_legalHolidays() else ""
    format_text = f"{lunar.lunarYearCn}年{lunar.lunarMonthCn}{lunar.lunarDayCn} {shengxiao}年{jieqi}{festival}"
    return format_text


class GroupChatSpeaker:
    def __init__(
            self,
            *,
            host: GroupMainDialog,
            character: str,
            llm_preset: str = "",
            logger: BlockHandle | Logger = None
    ):
        if logger is None:
            logger = _log

        # 数据存储
        self.last_message: str = ""
        self.last_failure_reason: str = ""
        self.last_failure_retryable: bool = False
        self.character: str = character
        self.llm_preset = llm_preset
        self.system_prompt: AgentPrompt = AgentPrompt(
            name="system_prompt",
            data_path=PROMPTS_DIR / r"group_speak_system",
            description="群聊发言LLM系统提示词",
            role="system"
        )
        self.member_persona = {}

        # 标识符
        self._initiated = False
        self._run_rounds = 0
        self._force_refresh = False
        self._force_clear = False
        self.had_group_persona = False

        # 工具对象
        self._host = host
        self.LLM = Chat()
        self.FAV = get_autofav(self._host.data_path / "fav_index.json")
        self.HISTORY = self._host.message_history

        # helper
        self.logger = logger
        self._allow_key_type = {"reply", "at", "break", "text", "image", "fav"}

        if llm_preset:
            llm_config = parse_llm_setting(llm_preset)
            if llm_config is not None:
                llm_config.keep_alive = True
                llm_config.logger = logger
                self.LLM.setting(new_config=llm_config)

    async def initiate(self):
        if self._initiated:
            return

        self.LLM.add_context(
            SystemPrompt(
                text=await self.system_prompt.get_prompt(self.character)
            )
        )
        self._initiated = True

    async def _refresh_context(self):
        new_context = Context()
        new_context.append(SystemPrompt(
            text=await self.system_prompt.get_prompt(self.character)
        ))
        self.LLM.replace_context(new_context)
        if self._force_clear:
            if self.HISTORY:
                self.last_message = self.HISTORY[-1].msg_id

        self._force_clear = False
        self._force_refresh = False
        self._initiated = True
        self.had_group_persona = False
        self.member_persona.clear()

    async def _refresh_context_helper(self):
        if self._force_refresh or self._force_clear or self._run_rounds > self._host.speaker_round_limit:
            await self._refresh_context()
            self._run_rounds = 0

    def mark_failure(self, reason: str, *, retryable: bool = False):
        self.last_failure_reason = reason
        self.last_failure_retryable = retryable

    async def speak(
            self,
            *,
            speak_prompt: str = "",
            working: str = "",
            relative_information: dict[str, str] = None,
            relative_memory: dict[str, str] = None,
            bot_info: Member | None = None
    ) -> _SpeakResponseDict | None:
        self.last_failure_reason = ""
        self.last_failure_retryable = False
        await self.initiate()
        timeout = SETTING_CFG.LLM.SpeakerRequestTimeout  # 默认180
        if not self.last_message:
            count = SETTING_CFG.Agent.SpeakerInitInputMessages  # 默认50
            new_messages = self.HISTORY.get_last_message(count)

        else:
            new_messages = self.HISTORY.get_new_message_start_from(self.last_message)

        # 获取角色表情列表
        character_fav = None
        try:
            char = get_character(self.character)

        except KeyError:
            char_fav_list = []

        else:
            character_fav = char.fav
            char_fav_list = list(character_fav.fav_list.keys())

        payload = SpeakerPostPayload(
            speak_prompt=speak_prompt,
            messages=new_messages,
            fav_list=list(self.FAV.fav_list.keys()) + char_fav_list,
            relative_information= relative_information,
            relative_memory=relative_memory,
            working=working,
            bot_info=bot_info,
            member_alias=self._host.member_alias
        )
        prompt = await payload.get_prompt()

        # 获取已经解析完成的最后一条消息
        prev_message = None
        if new_messages:
            for msg in new_messages:
                if msg.done or msg.broken:
                    prev_message = msg

                else:
                    break

        await self._refresh_context_helper()
        for _ in range(2):  # 只重试一次，写死
            try:
                response = await self.LLM.ask(prompt, timeout=timeout)

            except ContextOverflowError:
                await self._refresh_context()
                response = await self.LLM.ask(prompt, timeout=timeout)

            if not response:
                self.mark_failure("发言 LLM 未返回响应")
                continue

            result = response[0]
            if not isinstance(result, TextPrompt):
                self.mark_failure(f"发言 LLM 返回了非文本响应: {type(result).__name__}", retryable=True)
                continue

            json_text = _extract_json_object_text(result.text)
            try:
                json_content: _SpeakResponseDict = json.loads(json_text)
                if not isinstance(json_content, dict):
                    self.mark_failure("发言 LLM 返回的 JSON 不是 object", retryable=True)
                    continue

            except json.JSONDecodeError:
                self.mark_failure("发言 LLM 返回内容不是合法 JSON", retryable=True)
                continue

            else:
                # 请求成功了，把游标推到最后一个解析完成的消息处
                self._run_rounds += 1
                if prev_message is not None:
                    self.last_message = prev_message.msg_id

                bad_resp = False
                if set(json_content.keys()) != {"message_sequence", "message_content", "to_agent", "intention"}:
                    self.mark_failure("发言 LLM 返回字段不完整或存在多余字段", retryable=True)
                    continue

                clean_content: dict[str, dict] = {}
                sequence: list[str] = json_content["message_sequence"]
                content: dict[str, dict] = json_content["message_content"]
                to_agent: str = json_content["to_agent"]
                intention: dict = json_content["intention"]
                if not isinstance(sequence, list):
                    self.mark_failure("发言 LLM 返回的 message_sequence 不是 list", retryable=True)
                    continue

                if not isinstance(content, dict):
                    self.mark_failure("发言 LLM 返回的 message_content 不是 dict", retryable=True)
                    continue

                if not isinstance(to_agent, str):
                    self.mark_failure("发言 LLM 返回的 to_agent 不是 str", retryable=True)
                    continue

                if not isinstance(intention, dict):
                    self.mark_failure("发言 LLM 返回的 intention 不是 dict", retryable=True)
                    continue

                combined_fav = (
                    self.FAV | character_fav
                    if character_fav is not None
                    else self.FAV
                )
                for key in sequence:
                    if not isinstance(key, str):
                        self.mark_failure("发言 LLM 返回的 message_sequence 包含非字符串 key", retryable=True)
                        bad_resp = True
                        break

                    if key not in content:
                        self.mark_failure(f"发言 LLM 返回的 message_content 缺少 key: {key}", retryable=True)
                        bad_resp = True
                        break

                    key_type = key.split('_')[0]
                    if key_type not in self._allow_key_type:
                        self.mark_failure(f"发言 LLM 返回了不支持的消息类型: {key_type}", retryable=True)
                        bad_resp = True
                        break

                    if key in clean_content:
                        self.mark_failure(f"发言 LLM 返回了重复的消息 key: {key}", retryable=True)
                        bad_resp = True
                        break

                    if key_type == "fav":
                        fav = content.get(key, {})
                        if not fav:
                            continue

                        fav_hash_name = combined_fav.get_fav(fav.get("fav_title", ""))
                        if not fav_hash_name:
                            continue

                        clean_content[key] = {"hash_name": fav_hash_name}
                        continue

                    clean_content[key] = content[key]

                if bad_resp:
                    continue

                if not clean_content:
                    self.mark_failure("发言 LLM 未返回可发送的消息内容", retryable=True)
                    continue

                return _SpeakResponseDict(
                    message_sequence=list(clean_content.keys()),
                    message_content=clean_content,
                    to_agent=to_agent,
                    intention=intention
                )

        if not self.last_failure_reason:
            self.mark_failure("发言 LLM 未返回可发送的合法消息")

        return None

    def force_refresh(self):
        if self._force_refresh:
            return

        self._force_refresh = True

    def clear_history(self):
        if self._force_clear:
            return

        self._force_clear = True

    async def shutdown(self):
        await self.LLM.close()

    def switch_model(self, llm_preset: str) -> bool:
        """切换模型"""
        try:
            _ = parse_llm_setting(llm_preset)

        except (TypeError, ValueError):
            return False

        if _ is None:
            return False

        self.LLM.setting(parse_llm_setting(llm_preset))
        return True


class GroupChatAgent(BaseAgent):
    def __init__(
            self,
            *,
            host: GroupChatAgentHost,
            llm_preset: str = "",
            agent_name: str = "",
            control_timeout: float = -1,
            logger: BlockHandle | Logger = None
    ):
        if logger is None:
            logger = _log

        super().__init__(
            host_id=host.group_id,
            data_path=host.data_path,
            llm_preset=llm_preset,
            agent_name=agent_name,
            system_prompt=PROMPTS_DIR / r"group_chat_system",
            control_timeout=control_timeout,
            logger=logger
        )
        # 标识符
        self.last_message: str = ""
        self._run_rounds = 0
        self._force_refresh = False
        self._force_clear = False
        self.force_save_memory_retry = 0
        self._speak_prompt_times = 0
        self._not_speak_information_times = 0
        self._last_add_memory_time = time.time()
        self.said_review_count = 0

        # 数据容器
        self._speak_delay_record: dict[str, dict] = {}
        self._speak_request_generation = 0
        self._force_speak_observers: set[asyncio.Task] = set()
        self.said_intention: dict[str, dict] = {}  # 发言意图

        # 队列
        self._agent_post_queue: asyncio.Queue[PostTask] = asyncio.Queue()

        # 工具对象
        # self._host: GroupChatAgentHost = cast(GroupChatAgentHost, weakref.proxy(host))
        self._host: GroupMainDialog = weakref.proxy(host)

        # running_time
        self._run_loop: asyncio.Task | None = None

        # 锁
        self._memory_lock = asyncio.Lock()
        self._said_review_lock = asyncio.Lock()

        # 初始化
        self.functions_default_setting["speak"] = {"wait": False, "callback": "EXCEPTION"}
        self.functions_default_setting["_speaker_to_llm_callback"] = {"wait": True, "callback": "NEVER"}

    # =========================================================
    # Agent 入口
    # =========================================================

    # 新增一个内置工具
    def _load_internal_tools(self):
        super()._load_internal_tools()
        call_name = self.register_function(
            self.speak,
            display_name="构造发言中"
        )
        self.tools[call_name] = self.speak

    # 新增组件
    async def before_initiate(self):
        await super().before_initiate()
        self.register_function(
            self._speaker_to_llm_callback,
            display_name="发言 LLM 返回消息"
        )

        self.register_prompt(
            name="refresh_context_prompt",
            path=PROMPTS_DIR / "group_chat_refresh_context",
            description="群聊 Agent 刷新 context 的 payload",
            role="user"
        )

        # 注册 group_chat SKILL 的 Tools
        source = BASE_SKILL_DIR / "group_chat"
        self.register_function(
            self.get_single_message,
            source=source,
            source_name="group_chat",
            display_name="获取一条聊天消息"
        )
        self.register_function(
            self.get_time_interval_messages,
            source=source,
            source_name="group_chat",
            display_name="获取时间段内聊天消息"
        )
        self.register_function(
            self.get_image_detail,
            source=source,
            source_name="group_chat",
            display_name="获取详细图片内容"
        )
        self.register_function(
            self.mute,
            source=source,
            source_name="group_chat",
            display_name="拉黑他人"
        )
        self.register_function(
            self.unmute,
            source=source,
            source_name="group_chat",
            display_name="解除拉黑他人"
        )
        self.register_function(
            self.get_news_by_date,
            source=source,
            source_name="group_chat",
            display_name="获取指定日期新闻"
        )
        self.register_function(
            self.fetch_news,
            source=source,
            source_name="group_chat",
            display_name="获取新的新闻"
        )
        self.register_function(
            self.add_memory,
            source=source,
            source_name="group_chat",
            display_name="新增记忆"
        )
        self.register_function(
            self.get_memory,
            source=source,
            source_name="group_chat",
            display_name="获取记忆"
        )
        self.register_function(
            self.search_memory,
            source=source,
            source_name="group_chat",
            display_name="查询记忆"
        )
        self.register_function(
            self.get_group_persona,
            source=source,
            source_name="group_chat",
            display_name="获取群画像"
        )
        self.register_function(
            self.get_member_persona,
            source=source,
            source_name="group_chat",
            display_name="获取群员画像"
        )
        self.register_function(
            self.set_alias,
            source=source,
            source_name="group_chat",
            display_name="设置群员爱称"
        )
        self.register_function(
            self.upload_file,
            source=source,
            source_name="group_chat",
            display_name="上传文件"
        )
        self.register_function(
            self.get_fav_list,
            source=source,
            source_name="group_chat",
            display_name="获取当前表情列表"
        )
        self.register_function(
            self.add_group_fav,
            source=source,
            source_name="group_chat",
            display_name="添加一张群表情"
        )
        self.register_function(
            self.setu,
            source=source,
            source_name="group_chat",
            display_name="来点色图"
        )

        group_chat_setting = self._skills.tools_default_setting.setdefault("group_chat", {})
        group_chat_setting["get_single_message"] = {"wait": True, "callback": "CALL"}
        group_chat_setting["get_time_interval_messages"] = {"wait": True, "callback": "CALL"}
        group_chat_setting["get_image_detail"] = {"wait": False, "callback": "ALWAYS"}
        group_chat_setting["mute"] = {"wait": True, "callback": "NEVER"}
        group_chat_setting["unmute"] = {"wait": True, "callback": "NEVER"}
        group_chat_setting["get_news_by_date"] = {"wait": True, "callback": "CALL"}
        group_chat_setting["fetch_news"] = {"wait": False, "callback": "ALWAYS"}
        group_chat_setting["add_memory"] = {"wait": False, "callback": "EXCEPTION"}
        group_chat_setting["get_memory"] = {"wait": False, "callback": "ALWAYS"}
        group_chat_setting["search_memory"] = {"wait": False, "callback": "ALWAYS"}
        group_chat_setting["get_group_persona"] = {"wait": False, "callback": "ALWAYS"}
        group_chat_setting["get_member_persona"] = {"wait": False, "callback": "ALWAYS"}
        group_chat_setting["set_alias"] = {"wait": False, "callback": "EXCEPTION"}
        group_chat_setting["upload_file"] = {"wait": False, "callback": "EXCEPTION"}
        group_chat_setting["get_fav_list"] = {"wait": True, "callback": "CALL"}
        group_chat_setting["add_group_fav"] = {"wait": False, "callback": "EXCEPTION"}
        group_chat_setting["setu"] = {"wait": False, "callback": "EXCEPTION"}

    # 使用运行后钩子
    async def after_run(self, run_result: Prompts | None):
        await super().after_run(run_result)

        # 自行审阅任务
        asyncio.create_task(self._auto_said_review())

        # 检查是否有强制保存提示
        if save_notice := self._host.memory_save_notice:
            retry_limit = SETTING_CFG.Agent.ForceSaveMemoryRetry  # 默认3次
            if self._host.NOTICE.is_notice_alive(save_notice):
                if self.force_save_memory_retry < retry_limit:
                    self.force_save_memory_retry += 1
                    # 构造保存记忆 Recat
                    # noinspection PyAsyncCall
                    self._host.aqueue.add_task(self.run_in_queue(PostReason.SAVE_MEMORY), timeout=-1)

                else:
                    self._host.logger.warning("已达到强制保存记忆 React 重试上限，但主控仍未保存记忆")

    # 覆写压缩流程
    async def compress_context(self, new_prompt: Prompts = None, old_messages_count: int = None) -> Prompts:
        pinned = self._get_compress_context_pinned()
        suffix = self._detach_tool_calls_for_context_reset("GroupChatAgent 刷新 Context")
        suffix_image = []
        group_name = self._host.group_name
        group_id = self._host.group_id
        group_summary = await self._host.get_group_summary(True)
        member_list = self._host.member_list
        active_member_info = await self._host.get_active_member_info(True)
        old_messages = self._host.get_old_message(old_messages_count)
        news = await self._host.get_group_news()
        new_request = None
        if new_prompt is not None:
            lost_tool_resp: list[ToolResponse] = []
            for prompt in new_prompt:
                if isinstance(prompt, ToolResponse):
                    lost_tool_resp.append(prompt)

                elif isinstance(prompt, TextPrompt):
                    new_request = prompt

                elif isinstance(prompt, ImagePrompt):
                    suffix_image.append(prompt)

            if lost_tool_resp:
                suffix += "# 丢失的TOOL RESPONSE\n\n"
                for tr in lost_tool_resp:
                    suffix += f"```\n{tr.to_dict()}\n```\n\n---\n\n"

        refresh_prompt = self.get_last_version_prompt("refresh_context_prompt")
        self._history.compress_context(data="刷新 Context 窗口")
        await self._save_checkpoint_async()
        if refresh_prompt is None:
            text = ""

        else:
            text = await refresh_prompt.get_prompt(
                group_name=group_name,
                group_id=group_id,
                group_summary=group_summary,
                member_list=member_list,
                active_member_info=active_member_info,
                old_messages=old_messages,
                news=news,
                new_request=new_request
            )

        first_prompt = Prompts(
            TextPrompt(
                role="user",
                text=pinned + suffix + text
            ),
            *suffix_image
        )
        new_context = self._context.new()
        self.control.replace_context(new_context)
        await self.save_agent_async(save_context_history=True)
        self._force_refresh = False
        self._run_rounds = 0
        self._speak_delay_record.clear()
        return first_prompt

    # 检查任务循环
    def _ensure_running_loop(self):
        if self._run_loop is None or self._run_loop.done():
            self._run_loop = asyncio.create_task(self._run_with_payload())

    async def run_in_queue(
            self,
            reason: PostReason,
            timeout: float = None
    ):
        """Agent入口，实现一个异步原子运行队列，让自然等待的请求不会在中途退出"""
        self._ensure_running_loop()
        if timeout is None:
            timeout = SETTING_CFG.Groups.AgentPostQueueTimeout  # 默认180s

        new_task = PostTask(
            reason=reason,
            timeout=timeout,
        )
        await self._agent_post_queue.put(new_task)
        # self._logger.debug(f"已存入 Agent 任务")
        try:
            await asyncio.wait_for(new_task.expired.wait(), timeout=timeout)

        except asyncio.TimeoutError:
            new_task.expired.set()

    async def _payload_constructor(self, task: PostTask) -> AgentPostPayload:
        message_history: MessageManager = self._host.message_history
        notification = self._host.NOTICE
        if not self.last_message:
            count = SETTING_CFG.Agent.AgentInitInputMessages  # 默认100条
            new_messages = message_history.get_last_message(count)

        else:
            new_messages = message_history.get_new_message_start_from(self.last_message)

        notices_text = notification.notice2text()
        bot_info = await self._host.get_bot_info()
        return AgentPostPayload(
            reason=task.reason,
            messages=new_messages,
            notice=notices_text,
            bot_info=bot_info,
            member_alias=self._host.member_alias,
            said_intention=self.said_intention
        )

    async def _refresh_context_helper(self, prompt: TextPrompt) -> Prompts:
        prompt = Prompts(prompt)

        # 先判断强制指令
        if self._force_clear:
            await self.reload_system_prompt()
            prompt = await self.compress_context(prompt, 0)
            if self._host.message_history:
                self.last_message = self._host.message_history[-1].msg_id

            self._force_clear = False
            return prompt

        if self._force_refresh or self._run_rounds > self._host.agent_round_limit:
            prompt = await self.compress_context(prompt)
            return prompt

        return prompt

    async def handle_response(self, response: Prompts):
        if any(
                isinstance(prompt, ToolCall) and prompt.function_name == "speak"
                for prompt in response
        ):
            self._speak_request_generation += 1

        await super().handle_response(response)

    def _schedule_force_speak_fallback(self, start_time: float):
        generation = self._speak_request_generation
        observer = asyncio.create_task(
            self._observe_force_speak_fallback(generation, start_time)
        )
        self._force_speak_observers.add(observer)
        observer.add_done_callback(self._force_speak_observers.discard)

    async def _observe_force_speak_fallback(
            self,
            generation: int,
            start_time: float
    ):
        await asyncio.sleep(_FORCE_SPEAK_OBSERVATION_SECONDS)
        if not self._initiated or self._speak_request_generation != generation:
            return

        call_id = f"system_{next_time_id()}"
        await self.add_task(
            function_name="speak",
            description="系统自动创建的强制发言任务",
            call_id=call_id,
            callback="EXCEPTION"
        )
        self._speak_delay_record[call_id] = {"start": start_time}

    async def _cancel_force_speak_observers(self):
        observers = list(self._force_speak_observers)
        for observer in observers:
            observer.cancel()

        if observers:
            await asyncio.gather(*observers, return_exceptions=True)

        self._force_speak_observers.clear()

    async def _run_with_payload(self):
        while self._initiated:
            try:
                post_task = await self._agent_post_queue.get()
                # self._logger.debug(f"已取得 Agent 任务")
                if post_task.expired.is_set():
                    continue

                post_task.expired.set()
                start_time = time.time()
                payload = await self._payload_constructor(post_task)
                prompt = TextPrompt(role="user", text=await payload.get_prompt())
                # 获取最后解析成功的消息
                prev_message = None
                if payload.messages:
                    for msg in payload.messages:
                        if msg.done or msg.broken:
                            prev_message = msg

                        else:
                            break

                prompt = await self._refresh_context_helper(prompt)
                response = await self.run(prompt)

                # 后续检验
                need_speak = False
                if post_task.reason in _FORCE_SPEAK:
                    need_speak = True

                if response is not None:
                    self._run_rounds += 1

                    # commit new message
                    if prev_message is not None:
                        self.last_message = prev_message.msg_id

                    call_id = ""
                    for item in response:
                        if isinstance(item, ToolCall):
                            if item.function_name == "speak":
                                call_id = item.call_id
                                break

                    if call_id:
                        self._speak_delay_record[call_id] = {"start": start_time}

                    else:
                        if need_speak:
                            self._schedule_force_speak_fallback(start_time)

                else:
                    if need_speak:
                        self._schedule_force_speak_fallback(start_time)

            except asyncio.CancelledError:
                return

            except Exception as E:
                self._logger.error(f"群聊主控任务队列发送未捕获异常: {E}")
                self._logger.debug(traceback.format_exc())
                continue

    async def start(
            self,
            system_prompt: AgentPrompt = None,
            initiate_prompt: AgentPrompt = None
    ):
        await super().start(system_prompt, initiate_prompt)
        self._run_loop = asyncio.create_task(self._run_with_payload())

    async def shutdown(self):
        await self._cancel_force_speak_observers()
        await super().shutdown()
        if self._run_loop is not None:
            self._run_loop.cancel()

    # =========================================================
    # 发言接口，以及相应的数据处理。最重要！！
    # =========================================================

    async def speak(
            self,
            *,
            speak_prompt: str = "",
            working: str = "",
            relative_information: dict[str, str] = None,
            relative_memory: dict[str, str] = None
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
                content="作为主控，你已连续多次传递 `speak_prompt` 干涉发言。**切勿干涉过多**，遵守规则，不要管多余的事情",
                title="发言规范",
                priority=35,
                alive_until_self_messages=2,
                exist_ok=True
            )

        if self._not_speak_information_times > 7:
            self._host.NOTICE.add_notice(
                content="作为主控，你已多次未传递任何相关信息和记忆调用 `speak`，你需要 **积极传递** 信息给 Speaker",
                title="发言规范",
                priority=35,
                alive_until_self_messages=2,
                exist_ok=True
            )

        try:
            result = await self._host.speak(
                speak_prompt=speak_prompt,
                working=working,
                relative_information=relative_information,
                relative_memory=relative_memory
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            failure_reason = f"发言链路异常: {exc}"
            failure_retryable = True
            self._logger.error(f"{failure_reason}")
            self._logger.debug(traceback.format_exc())
            result = None

        lookup_tasks = self._get_task_lookup()
        if task_id in lookup_tasks:
            call_id = lookup_tasks[task_id].call_id
            self._speak_delay_record.setdefault(call_id, {})["end"] = time.time()


        if result is None:
            if not failure_reason:
                speaker = getattr(self._host, "SPEAKER", None)
                if speaker is not None:
                    failure_reason = getattr(speaker, "last_failure_reason", "")
                    failure_retryable = getattr(speaker, "last_failure_retryable", False)

            if not failure_reason:
                failure_reason = "发言任务失败，但未记录到具体原因"

            next_action = _speaker_failure_guidance(failure_retryable)
            message = f"发言任务[{task_id}]失败。原因: {failure_reason}。{next_action}"
            self._logger.warning(f"{message}")
            asyncio.create_task(
                self.add_task(
                    function_name="_speaker_to_llm_callback",
                    params={"message": message},
                    call_id=task_id,
                    pin=True,
                    callback="ALWAYS",
                    wait=True
                )
            )
            return message

        else:
            if result:
                asyncio.create_task(
                    self.add_task(
                        function_name="_speaker_to_llm_callback",
                        params={"message": f"发言任务[{task_id}]返回给主控的信息:  \n\n{result}"},
                        call_id=task_id,
                        pin=True,
                        callback="ALWAYS",
                        wait=True
                    )
                )

        return None

    @staticmethod
    def _speaker_to_llm_callback(message: str) -> str:
        """SPEAKER的回调"""
        return message

    @staticmethod
    def _parse_intention(msg_id: str, intention: dict):
        output = ""
        intention = intention.get(msg_id, {})
        if not intention:
            return output

        output = "本次发言意图: "
        if "target_message_id" in intention:
            target_message_id = intention["target_message_id"]
            if target_message_id:
                output += f"响应消息 [{intention['target_message_id']}]; "

        if "target_user_id" in intention:
            target_user_id = intention["target_user_id"]
            if target_user_id:
                output += f"响应群员 [{intention['target_user_id']}]; "

        if "intent" in intention:
            intent = intention["intent"]
            if intent in _SPEAK_INTENT_MAPPING:
                output += f"发言行为类别: {_SPEAK_INTENT_MAPPING[intent]}"

        if "effect" in intention:
            effect = intention["effect"]
            if effect in _SPEAK_INTENT_MAPPING:
                output += f"期望发言效果: {_SPEAK_INTENT_MAPPING[effect]}"

        return output

    async def _auto_said_review(self):
        if self.said_review_count < SETTING_CFG.Agent.SaidReviewPeriod:  # 默认 15 次
            return

        async with self._said_review_lock:
            if self.said_review_count < SETTING_CFG.Agent.SaidReviewPeriod:  # 默认 15 次
                return

            count = self.said_review_count
            self.said_review_count = 0
            intention_mirror = self.said_intention.copy()
            self.said_intention.clear()
            bot_said = self._host.message_history.get_message_from_user(user_id=self._host.bot_id)
            recent_said = []
            for i, msg in enumerate(reversed(bot_said), start=1):
                if i > count:
                    break

                recent_said.append(msg)

            msg_payload = []
            tasks: list[asyncio.Task] = []
            timeout = SETTING_CFG.Groups.MessageParseTimeout  # 默认15
            for msg in reversed(recent_said):
                tasks.append(asyncio.create_task(msg.to_llm(wait=True, timeout=timeout)))

            await asyncio.gather(*tasks, return_exceptions=True)

            for task in tasks:
                if not task.done() or task.cancel() or task.exception() is not None:
                    continue

                to_llm_dict = task.result()
                to_llm_dict["user_aliases"] = self._host.member_alias.get(to_llm_dict["user_id"], [])
                intention_text = self._parse_intention(to_llm_dict["message_id"], intention_mirror)
                if intention_text:
                    to_llm_dict["message_tips"].append(intention_text)

                msg_payload.append(to_llm_dict)

            payload = (f"**本次请求为 *BOT 发言结果审阅任务*。请勿发言**\n\n"
                       f"# 近期 BOT 发言记录\n\n```json\n{json.dumps(msg_payload, ensure_ascii=False, indent=2)}\n```\n\n\n"
                       f"你需要结合完整聊天记录，根据规则进行审阅、质量评估，并在后续发言任务里，"
                       f"通过 `relative_information` 把反馈给 Speaker")
            # noinspection PyAsyncCall
            self._host.aqueue.add_task(self.run(TextPrompt(role="user", text=payload), from_user=False), timeout=-1)

    # =========================================================
    # "group_chat" SKILL 业务逻辑
    # =========================================================

    async def get_single_message(self, message_id: str) -> dict:
        """
        获取一条聊天消息的详情，会等待所有内容解析完成再返回。
        :param message_id: 消息id(msg_id)。
        :return: 成功: 标准聊天消息字典; 消息不存在/已过期: 空字典
        """
        msg = await self._host.message_history.get_message(message_id)
        if msg is None:
            return {}

        timeout = SETTING_CFG.Groups.MessageParseTimeout  # 默认15
        result = await msg.to_llm(wait=True, timeout=timeout)
        return result

    async def get_time_interval_messages(self, start: str = None, end: str = None) -> list[dict]:
        """
        获取一个时间段内的所有聊天消息，会等待所有消息的所有内容解析完成再返回
        :param start: 起始时间: 标准 `ISO 8601`字符串(无时区)，若不传入则为聊天记录中最早的消息时间。
        :param end: 结束时间: 标准 `ISO 8601`字符串(无时区)，若不传入则为聊天记录中最新的消息时间。
        :return: 成功: 一个标准聊天消息的字典列表; 无消息: 空列表; 失败: 抛出异常
        """
        if start is not None:
            try:
                start = datetime.fromisoformat(start).astimezone().timestamp()

            except (TypeError, ValueError):
                raise ValueError("解析起始时间字符串失败")

        if end is not None:
            try:
                end = datetime.fromisoformat(end).astimezone().timestamp()

            except (TypeError, ValueError):
                raise ValueError("解析结束时间字符串失败")

        messages = cast(list[GroupMsg], self._host.message_history.get_time_interval_message(start, end))  # type:
        timeout = SETTING_CFG.Groups.MessageParseTimeout  # 默认15
        output = []
        tasks: list[asyncio.Task] = []
        if not messages:
            return []

        for msg in messages:
            tasks.append(asyncio.create_task(msg.to_llm(True, timeout)))

        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            if not task.done() or task.cancelled() or task.exception() is not None:
                continue

            output.append(task.result())

        return output

    @staticmethod
    async def get_image_detail(image_hash_name: str, query: str, image_name: str = "") -> dict:
        """
        调用一个大型多模态模型去更仔细的识别图片。
        :param image_hash_name: 聊天消息中图片内容的 `hash_name`
        :param query: 想要识别的方向，自然语言描述
        :param image_name: 若图片来自聊天消息，可把图片内容中固定 `image_name`也输入，有助于缓存系统存储
        :return: 成功/失败: 一个识图结果字典
        """
        class _ImageQueryResult(TypedDict):
            success: bool
            image_hash_name: str
            query: str
            answer: str
            message: str

        try:
            image_file = await get_file_path_async(image_hash_name)

        except (FileNotFoundError, ValueError):
            return dict(_ImageQueryResult(
                success=False,
                image_hash_name=image_hash_name,
                query=query,
                answer="",
                message="图片 `hash_name` 不正确或文件不存在/已过期"
            ))

        try:
            with Image.open(image_file) as _image:
                _image.verify()

        except UnidentifiedImageError:
            # noinspection PyUnboundLocalVariable
            return dict(_ImageQueryResult(
                success=False,
                image_hash_name=image_hash_name,
                query=query,
                answer="",
                message="输入的 `hash_name` 指向非图片文件或图片已损坏"
            ))


        response = await image2text(
            hash_name=image_hash_name,
            query=query,
            image_name=image_name
        )
        if response:
            return dict(_ImageQueryResult(
                success=True,
                image_hash_name=image_hash_name,
                query=query,
                answer=response,
                message=""
            ))

        else:
            return dict(_ImageQueryResult(
                success=False,
                image_hash_name=image_hash_name,
                query=query,
                answer="",
                message="识图LLM无返回结果"
            ))

    async def mute(self, user_id: str, duration: float = None) -> dict:
        """
        拉黑他人，拉黑后将拒绝接收此人的消息。
        :param user_id: 要拉黑的人的QQ号(user_id)。
        :param duration: 拉黑时间，单位: 秒。默认 3 小时。
        :return: 成功/失败: 结果字典
        """
        class _MuteResult(TypedDict):
            success: bool
            user_id: str
            uptime: str
            message: str

        result = await self._host.mute(user_id, duration)
        if result is None:
            return dict(_MuteResult(
                success=False,
                user_id=user_id,
                uptime="",
                message="拉黑失败，用户不存在或获取成员列表失败"
            ))

        return dict(_MuteResult(
            success=True,
            user_id=user_id,
            uptime=timestamp2text(result),
            message=f"已拉黑用户[{user_id}]"
        ))

    async def unmute(self, user_id: str) -> None:
        """
        解除拉黑他人。
        :param user_id: 要解除拉黑的人的QQ号(user_id)
        :return: 成功/失败: 无返回内容(None)
        """
        await self._host.unmute(user_id)
        return None

    async def get_news_by_date(self, date: str = None) -> dict:
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
        class _NewsResult(TypedDict):
            date: str
            news: dict[str, list]

        today = datetime.now()
        key = timestamp2text(today.timestamp(), "%y-%m-%d")
        results = await self._websearch.search(f"有关 [{topic}] 的新闻或相关信息", purge=True)
        news = []
        for result in results:
            if not result["ok"]:
                continue

            news.append(result["results"])

        if not news:
            return {}

        today_news = self._host.relative_news.setdefault(key, {})
        topic_exist = today_news.setdefault(topic, [])
        topic_exist.extend(news)
        news_file = self._host.data_path / "news" / f"{key}.json"
        news_file.parent.mkdir(parents=True, exist_ok=True)
        news_file.write_text(json.dumps(today_news, ensure_ascii=False), encoding="utf-8")

        packed = _NewsResult(
            date=timestamp2text(today.timestamp(), "%y-%m-%d"),
            news={topic: news}
        )

        return dict(packed)

    async def add_memory(self, target: str, memory: str) -> dict:
        """
        新增目标对象的记忆。
        :param target: 目标对象: 用户QQ号(user_id)/'GROUP'(群整体记忆)/'BOT'(你自己的记忆)。
        :param memory: 记忆内容，任意字符串，可用自然语言表示，也可以结构化。
        :return: 记忆结果字典
        """
        # 加锁，防止一次性添加多次记忆
        async with self._memory_lock:
            if time.time() - self._last_add_memory_time < 15:
                return {
                    "success": False,
                    "memory": "",
                    "message": "添加记忆过于频繁，若实在有必要请稍后再试"
                }

            result = await self._host.add_memory(target, memory)
            self._last_add_memory_time = time.time()
            return result

    async def get_memory(self, target: str, query: str) -> dict:
        """
        获取指定目标对象的相关记忆。
        :param target: 目标对象: 用户QQ号(user_id)/'GROUP'(群整体记忆)/'BOT'(你自己的记忆)。
        :param query: 想要获取的内容，自然语言表示。
        :return: 成功/失败: 记忆结果字典
        """
        return await self._host.get_memory(target, query)

    async def search_memory(self, query: str) -> dict:
        """
        全局搜索指定内容的记忆。
        :param query: 想要搜索的内容，自然语言表示。
        :return: 成功/失败: 记忆结果字典
        """
        return await self._host.search_memory(query)

    async def get_group_persona(self) -> dict:
        """
        获取群画像。
        :return: 成功/失败: 群画像字典
        """
        result = await self._host.get_group_persona()
        return dict(result.to_dict())

    async def get_member_persona(self, user_id: str) -> dict:
        """
        获取群员用户画像。
        :param user_id: 群员的QQ号(user_id)。
        :return: 成功: 群用户画像字典; 失败: 空字典
        """
        result = await self._host.get_member_persona(user_id)
        if result is None:
            return {}

        return dict(result.to_dict())

    async def set_alias(self, user_id: str, alias: str) -> None:
        """
        给指定群员设置一个爱称
        :param user_id: 群员 QQ 号
        :param alias: 爱称
        :return: 成功: 无返回(None)，新爱称会直接出现在后续聊天记录; 失败: 抛出异常
        """
        return await self._host.set_alias(user_id, alias)

    async def upload_file(self, file: str, upload_filename: str) -> dict:
        """
        将本地文件上传到群文件。
        :param file: 本地文件路径/hash_name。
        :param upload_filename: 上传到群文件的文件名，与本地真实文件名无关，可以不同。
        :return: 成功: 上传结果字典; 失败: 抛出异常
        """
        class _FileUpload(TypedDict):
            success: bool
            message_id: str
            hash_name: str
            upload_filename: str
            message: str

        # 先解析文件
        file_path = self._skills._resolve_path(file)
        if file_path.is_file():
            hash_name = await add_file_async(file_path)

        else:
            hash_name = file
            if not check_file_exists(hash_name):
                raise FileNotFoundError(f"文件[{file}]不存在，无法上传")

        success, response = await self._host.upload_group_file(hash_name, upload_filename)
        if not success:
            raise RuntimeError(f"文件[{file}]上传失败")

        message_id = response.msg_id if response is not None else ""
        message = "文件上传成功"
        if response is None:
            message += "，但群消息记录尚未同步，暂时无法获取 message_id"

        return dict(_FileUpload(
            success=True,
            message_id=message_id,
            hash_name=hash_name,
            upload_filename=upload_filename,
            message=message
        ))

    def get_fav_list(self) -> dict:
        """
        获取最新的表情列表，分为群表情和 Speaker 当前角色的表情
        :return: 成功: 表情列表结果字典; 无表情: 结果字典，带有空列表; 失败: 抛出异常
        """
        class _FavResult(TypedDict):
            group_fav: list[str]
            character_fav: list[str]

        group_fav = list(self._host.FAV.fav_list.keys())
        character_fav = []
        if self._host.GCM is not None:
            character_now = self._host.GCM.character_now
            char = get_character(character_now)
            character_fav = list(char.fav.fav_list.keys())

        return dict(_FavResult(
            group_fav=group_fav,
            character_fav=character_fav
        ))

    async def add_group_fav(self, hash_name: str, fav_title: str = "") -> dict:
        """
        添加一张群表情。
        :param hash_name: 表情图片的 `hash_name`。
        :param fav_title: 表情标题，尽量15字内，若不传入则调用识图模型自动获取。
        :return: 成功: 一个结果字典; 失败: 抛出异常。仅失败会回调
        """
        class _AddFavResult(TypedDict):
            success: bool
            hash_name: str
            fav_title: str
            fav_type: str
            message: str

        # 让异常直接抛出
        image_path = await get_file_path_async(hash_name)
        with Image.open(image_path) as image:
            image.verify()

        title = await self._host.FAV.add_fav(hash_name, fav_title)
        return dict(_AddFavResult(
                success=True,
                hash_name=hash_name,
                fav_title=title,
                fav_type="group",
                message="成功添加群表情"
            ))

    async def setu(self, args: str = "") -> None:
        """
        色图功能，CLI使用方式，仅可在启用色图功能的群使用，否则抛出无权限异常。
        :param args: CLI 形式的附加参数，全部均为可选项，可通过 `--help` 查看使用方法，详细用例读取 `group_chat` SKILL。
        :return: 成功: 静默，不返回内容(None)，发送的色图结果会显示在聊天记录中; `--help`: 通过异常返回指令用法; 失败: 抛出对应异常
        """
        timeout = SETTING_CFG.SETU.SendSETUTaskTimeout
        task = self._host.aqueue.add_task(self._host.setu(args), timeout)
        await task.wait()
        if task.exception:
            raise task.exception[0]

        if task.result:
            return task.result[0]

        else:
            return None

    # =========================================================
    # 外部指令接口
    # =========================================================

    def clear_history(self):
        """强制清空上下文"""
        if self._force_clear:
            return

        self._force_clear = True

    def force_refresh(self):
        """强制刷新上下文窗口"""
        if self._force_refresh:
            return

        self._force_refresh = True

    def get_average_speak_delay(self) -> float | None:
        """返回发言平均延迟"""
        delays = []
        for record in self._speak_delay_record.values():  # type: dict[str, float]
            start = record.get("start", None)
            end = record.get("end", None)
            if start is None or end is None:
                continue

            delays.append(end - start)

        if delays:
            return sum(delays) / len(delays)

        return None

    def switch_model(self, llm_preset: str) -> bool:
        """切换模型"""
        try:
            _ = parse_llm_setting(llm_preset)

        except (TypeError, ValueError):
            return False

        if _ is None:
            return False

        self.modify_agent(llm_preset=llm_preset)
        return True

    @property
    def run_queue_empty(self) -> bool:
        return self._agent_post_queue.empty()
