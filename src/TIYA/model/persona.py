"""
persona.py
记录群画像和用户画像
"""
from __future__ import annotations

__version__ = "0.1.0"

import traceback
import asyncio
import json
from typing import TYPE_CHECKING, Literal, Any, TypedDict, cast
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime

from TIYA.memory import get_context_memory_engine
from TIYA.memory.models import AddResult, QueryResult, MemoryRecord
from TIYA.agent.agent_prompt import AgentPrompt
from TIYA.LLM_connect import TextPrompt
from TIYA.utils import timestamp2text, clean_json_text_from_llm
from TIYA.logger import get_logger
from TIYA.config import PROMPTS_DIR, SETTING_CFG, get_bot_name, get_bot_uid

if TYPE_CHECKING:
    from TIYA.memory import BucketHandle
    from TIYA.qq_group import Member, QQGroup


_memory_engine = get_context_memory_engine()
_log = get_logger()
GROUP_ACTION_TYPE_MAPPING = {
    "ORDINARY": "普通群",
    "GAME": "游戏群",
    "WORK": "工作/项目群",
    "FRIEND": "亲友群",
    "FAMILY": "家庭/家族群",
    "NOTIFICATION": "通知/公告群",
    "STUDY": "学习群",
    "SPORTS": "运动健身群",
    "READING": "读书/文学交流群",
    "TECH": "技术/编程群",
    "FANDOM": "粉丝/追星群",
    "PET": "宠物交流群",
    "PARENTING": "家长/育儿群",
    "NEIGHBOR": "业主/邻里群",
    "SHOPPING": "拼单/团购/二手交易群",
    "TRAVEL": "旅游群",
    "FOOD": "美食/探店群",
    "CARPOOL": "拼车/顺风车群",
    "DATING": "线下交友/脱单群",
    "ANONYMOUS": "匿名/树洞群",
    "CLASS": "班级群",
    "ALUMNI": "校友/同学群",
    "CUSTOMER": "客户/售后群",
    "EMERGENCY": "应急/互助群",
    "INVEST": "投资/理财/股票群",
    "JOB_HUNT": "求职/招聘/内推群",
    "RELIGION": "宗教/信仰群",
}

class GroupActionType(Enum):
    ORDINARY = "ORDINARY"  # 普通群
    GAME = "GAME"  # 游戏群
    WORK = "WORK"  # 工作/项目群
    FRIEND = "FRIEND"  # 亲友群
    FAMILY = "FAMILY"  # 家庭/家族群
    NOTIFICATION = "NOTIFICATION"  # 通知/公告群
    STUDY = "STUDY"  # 学习群
    SPORTS = "SPORTS"  # 运动健身群
    READING = "READING"  # 读书/文学交流群
    TECH = "TECH"  # 技术/编程群
    FANDOM = "FANDOM"  # 粉丝/追星群
    PET = "PET"  # 宠物交流群
    PARENTING = "PARENTING"  # 家长/育儿群
    NEIGHBOR = "NEIGHBOR"  # 业主/邻里群
    SHOPPING = "SHOPPING"  # 拼单/团购/二手交易群
    TRAVEL = "TRAVEL"  # 旅游群
    FOOD = "FOOD"  # 美食/探店群
    CARPOOL = "CARPOOL"  # 拼车/顺风车群
    DATING = "DATING"  # 线下交友/脱单群
    ANONYMOUS = "ANONYMOUS"  # 匿名/树洞群
    CLASS = "CLASS"  # 班级群）
    ALUMNI = "ALUMNI"  # 校友/同学群
    CUSTOMER = "CUSTOMER"  # 客户/售后群
    EMERGENCY = "EMERGENCY"  # 应急/互助群
    INVEST = "INVEST"  # 投资/理财/股票群
    JOB_HUNT = "JOB_HUNT"  # 求职/招聘/内推群
    RELIGION = "RELIGION"  # 宗教/信仰群


class _GroupPersonaResultDict(TypedDict):
    group_id: str
    group_name: str
    group_action_type: str
    group_hobby: str
    group_disgust: str
    group_vip: list[str]
    group_event: str
    group_memes: list[str]
    group_relations: dict[str, str]


@dataclass(slots=True)
class GroupPersonaResult:
    group_id: str
    group_name: str = ""
    group_action_type: GroupActionType = GroupActionType.ORDINARY  # 群类型
    group_hobby: str = ""  # 群爱好
    group_disgust: str = ""  # 群厌恶
    group_vip: list[str] = field(default_factory=list)  # 群重要成员
    group_event: str = ""  # 群近期事件
    group_memes: list[str] = field(default_factory=list)  # 群常用梗
    group_relations: dict[str, str] = field(default_factory=dict)  # 群员关系、联系

    def to_dict(self) -> _GroupPersonaResultDict:
        base_dict = asdict(self)
        base_dict["group_action_type"]: str = self.group_action_type.value

        # noinspection PyTypeChecker
        return base_dict


class _MemberPersonaResultDict(TypedDict):
    user_id: str
    user_name: str
    nickname: str
    title: str
    role: str
    alias: list[str]
    gender: Literal["male", "female", "unknown"]
    age: int
    birthday: str
    city: str
    occupation: str
    hobby: str
    disgust: str
    working: str
    memes: list[str]
    catchphrase: str
    style: str
    relations: dict[str, str]


@dataclass(slots=True)
class MemberPersonaResult:
    user_id: str
    user_name: str = ""
    nickname: str = ""
    title: str = ""
    role: str = "member"
    alias: list[str] = field(default_factory=list) # 群友爱称
    gender: Literal["male", "female", "unknown"] = "unknown"  # 性别
    age: int = 0  # 年龄
    birthday: str = "未知"  # 生日
    city: str = "未知"  # 所在城市
    occupation: str = "未知"  # 职业
    hobby: str = "未知"  # 爱好
    disgust: str = "未知"  # 厌恶
    working: str = "未知"  # 近期在做
    memes: list[str] = field(default_factory=list) # 常用梗
    catchphrase: str = "未知"  # 口头禅
    style: str = "未知"  # 发言风格
    relations: dict[str, str] = field(default_factory=dict)  # 群员关系

    def to_dict(self) -> _MemberPersonaResultDict:

        # noinspection PyTypeChecker
        return asdict(self)


def _persona_string(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    return value if isinstance(value, str) else default


def _persona_string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _persona_string_dict(data: dict[str, Any], key: str) -> dict[str, str]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        return {}
    return {
        str(item_key): item_value
        for item_key, item_value in value.items()
        if isinstance(item_key, str) and isinstance(item_value, str)
    }


@dataclass(slots=True)
class PrivateParticipantPersonaResult:
    """私聊参与者画像；用户和 BOT 使用完全相同的字段结构。"""

    participant_id: str  # 参与者唯一标识，由程序填写，不允许画像模型修改。
    participant_name: str = ""  # 参与者当前名称，例如 QQ 昵称或 BOT 名称。
    alias: list[str] = field(default_factory=list)  # 对方在当前私聊关系中对该参与者的常用称呼。
    gender: Literal["male", "female", "unknown"] = "unknown"  # 性别，可依据稳定上下文合理推测。
    age: int = 0  # 当前周岁年龄，无法合理推测时为 0。
    birthday: str = "未知"  # 生日或有依据的生日范围。
    city: str = "未知"  # 当前主要居住或长期生活的城市。
    occupation: str = "未知"  # 当前职业、身份或主要社会角色。
    personality: str = "未知"  # 较稳定的性格、行为倾向和思考方式。
    values: str = "未知"  # 长期表现出的价值取向、原则和重视的事情。
    hobby: str = "未知"  # 长期感兴趣、喜欢或主动投入的事物。
    disgust: str = "未知"  # 明确厌恶、排斥或长期不喜欢的事物。
    working: str = "未知"  # 近期持续投入的工作、项目、目标或生活事项。
    memes: list[str] = field(default_factory=list)  # 经常使用且能代表表达习惯的梗。
    catchphrase: str = "未知"  # 稳定重复使用的口头禅或标志性表达。
    style: str = "未知"  # 整体语言和交流风格。
    emotional_style: str = "未知"  # 表达和处理情绪的稳定方式。
    relations: dict[str, str] = field(default_factory=dict)  # 与重要人物、组织、项目或概念的联系。

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        participant_id: str,
        participant_name: str,
    ) -> PrivateParticipantPersonaResult:
        """逐字段解析画像；单个字段错误不会使整个参与者画像失效。"""
        source = data if isinstance(data, dict) else {}
        gender = source.get("gender", "unknown")
        if gender not in {"male", "female", "unknown"}:
            gender = "unknown"

        age = source.get("age", 0)
        if not isinstance(age, int) or isinstance(age, bool):
            age = 0

        return cls(
            participant_id=str(participant_id),
            participant_name=str(participant_name),
            alias=_persona_string_list(source, "alias"),
            gender=gender,  # type: ignore
            age=age,
            birthday=_persona_string(source, "birthday", "未知"),
            city=_persona_string(source, "city", "未知"),
            occupation=_persona_string(source, "occupation", "未知"),
            personality=_persona_string(source, "personality", "未知"),
            values=_persona_string(source, "values", "未知"),
            hobby=_persona_string(source, "hobby", "未知"),
            disgust=_persona_string(source, "disgust", "未知"),
            working=_persona_string(source, "working", "未知"),
            memes=_persona_string_list(source, "memes"),
            catchphrase=_persona_string(source, "catchphrase", "未知"),
            style=_persona_string(source, "style", "未知"),
            emotional_style=_persona_string(source, "emotional_style", "未知"),
            relations=_persona_string_dict(source, "relations"),
        )


@dataclass(slots=True)
class PrivateRelationshipPersonaResult:
    """用户与 BOT 在当前一对一私聊中共同形成的关系画像。"""

    relationship_type: str = "未知"  # 关系的主要性质，例如陪伴、协作、咨询或混合关系。
    relationship_stage: str = "初识"  # 当前关系阶段，不使用没有依据的精确亲密度数值。
    relationship_summary: str = ""  # 对关系特点的简短总体描述，不堆积具体记忆。
    shared_hobby: str = "未知"  # 双方共同喜欢、讨论或参与的内容。
    shared_disgust: str = "未知"  # 双方共同排斥、不喜欢或习惯共同吐槽的内容。
    current_event: str = "未知"  # 双方目前共同关注、推进或持续讨论的事情。
    current_state: str = "未知"  # 当前相处氛围和近期关系状态。
    interaction_style: str = "未知"  # 双方实际形成的相处与对话方式。
    interaction_preferences: str = "未知"  # 用户期待的服务方式及双方形成的默认互动习惯。
    shared_memes: list[str] = field(default_factory=list)  # 双方共同理解或反复使用的内部梗。
    routines: list[str] = field(default_factory=list)  # 双方稳定重复的互动习惯。
    milestones: list[str] = field(default_factory=list)  # 对关系发展有代表性的共同经历或节点。
    boundaries: list[str] = field(default_factory=list)  # 双方已表达或长期形成的禁区与相处边界。
    commitments: list[str] = field(default_factory=list)  # 双方仍然有效的承诺，内容需明确承诺主体。
    unfinished_business: list[str] = field(default_factory=list)  # 尚未完成、需要跟进或回访的事项。
    relations: dict[str, str] = field(default_factory=dict)  # 关系与共同人物、项目、事物或概念的联系。

    @classmethod
    def from_dict(cls, data: object) -> PrivateRelationshipPersonaResult:
        """逐字段解析关系画像；无效字段单独回退到默认值。"""
        source = data if isinstance(data, dict) else {}
        return cls(
            relationship_type=_persona_string(source, "relationship_type", "未知"),
            relationship_stage=_persona_string(source, "relationship_stage", "初识"),
            relationship_summary=_persona_string(source, "relationship_summary", ""),
            shared_hobby=_persona_string(source, "shared_hobby", "未知"),
            shared_disgust=_persona_string(source, "shared_disgust", "未知"),
            current_event=_persona_string(source, "current_event", "未知"),
            current_state=_persona_string(source, "current_state", "未知"),
            interaction_style=_persona_string(source, "interaction_style", "未知"),
            interaction_preferences=_persona_string(source, "interaction_preferences", "未知"),
            shared_memes=_persona_string_list(source, "shared_memes"),
            routines=_persona_string_list(source, "routines"),
            milestones=_persona_string_list(source, "milestones"),
            boundaries=_persona_string_list(source, "boundaries"),
            commitments=_persona_string_list(source, "commitments"),
            unfinished_business=_persona_string_list(source, "unfinished_business"),
            relations=_persona_string_dict(source, "relations"),
        )


@dataclass(slots=True)
class PrivatePersonaResult:
    """同一个私聊关系记忆桶生成的用户、BOT 和关系三视角画像。"""

    user: PrivateParticipantPersonaResult  # 当前私聊用户的参与者画像。
    bot: PrivateParticipantPersonaResult  # BOT 在当前用户关系中的动态画像。
    relationship: PrivateRelationshipPersonaResult  # 用户与 BOT 共同形成的关系画像。

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def empty(
        cls,
        *,
        user_id: str,
        user_name: str,
        bot_id: str | None = None,
        bot_name: str | None = None,
    ) -> PrivatePersonaResult:
        """创建尚无有效记忆时使用的完整默认画像。"""
        return cls(
            user=PrivateParticipantPersonaResult(
                participant_id=str(user_id),
                participant_name=str(user_name),
            ),
            bot=PrivateParticipantPersonaResult(
                participant_id=str(bot_id if bot_id is not None else get_bot_uid()),
                participant_name=str(bot_name if bot_name is not None else get_bot_name()),
            ),
            relationship=PrivateRelationshipPersonaResult(),
        )

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        user_id: str,
        user_name: str,
        bot_id: str | None = None,
        bot_name: str | None = None,
    ) -> PrivatePersonaResult:
        """解析模型或持久化数据，未知字段忽略，错误字段独立回退。"""
        source = data if isinstance(data, dict) else {}
        resolved_bot_id = str(bot_id if bot_id is not None else get_bot_uid())
        resolved_bot_name = str(bot_name if bot_name is not None else get_bot_name())
        return cls(
            user=PrivateParticipantPersonaResult.from_dict(
                source.get("user", {}),
                participant_id=str(user_id),
                participant_name=str(user_name),
            ),
            bot=PrivateParticipantPersonaResult.from_dict(
                source.get("bot", {}),
                participant_id=resolved_bot_id,
                participant_name=resolved_bot_name,
            ),
            relationship=PrivateRelationshipPersonaResult.from_dict(
                source.get("relationship", {}),
            ),
        )


class BasePersona:
    def __init__(self, bucket_name: str, summary: str = ""):
        self.bucket_name: str = bucket_name
        self.summary = summary
        self.bucket: BucketHandle | None = None
        self.prompts: dict[str, AgentPrompt] = {}
        self._lock = asyncio.Lock()
        asyncio.create_task(self._initiate_bucket())

    async def _initiate_bucket(self):
        async with self._lock:
            if self.bucket is not None:
                return

            parent = await _memory_engine.set_bucket(
                title="[PERSONA]",
                summary="存放用户、群组画像",
                content="群与用户的画像存储父桶",
                summary_locked=True
            )
            self.bucket = await parent.set_bucket(
                title=self.bucket_name,
                summary=self.summary,
                summary_locked=True if self.summary else False
            )

        memories = await self.bucket.list_memories()
        exist = memories.total_memory_count
        if not exist:
            await self._add_first_memory()

    async def _add_first_memory(self):
        raise NotImplementedError

    async def _ensure_bucket_handle(self):
        if self.bucket is None:
            await self._initiate_bucket()

    def _load_advance_query_system(self):
        raise NotImplementedError

    async def add_memory(self, content: str, include_time: bool = True, include_prefix: bool = True):
        raise NotImplementedError

    async def get_memory(self, query: str, include_time: bool = True) -> QueryResult:
        await self._ensure_bucket_handle()
        now = datetime.now()
        time_info = ""
        if include_time:
            time_info  = f"> 当前时间: {timestamp2text(now.timestamp())}  \n\n"

        command = f"# 查询内容\n\n{query}"

        response = await self.bucket.query(time_info + command)
        return response

    async def get_memory_content(self, mem_id: str) -> MemoryRecord | None:
        await self._ensure_bucket_handle()
        return await self.bucket.get_memory(mem_id)

    async def get_persona(self):
        """获取画像"""
        raise NotImplementedError

    async def is_empty(self) -> bool:
        await self._ensure_bucket_handle()
        if self.bucket is None:
            return True

        memorise_list = await self.bucket.list_memories()
        total = memorise_list.total_memory_count
        if total <= 1:
            return True

        return False

    @property
    def bucker_id(self) -> str:
        if self.bucket is None:
            return ""

        return self.bucket.bucket_id


class BaseGroupPersona(BasePersona):
    def __init__(self, group_id: str, bucket_name: str, summary: str = ""):
        self.group_id = group_id
        self.parent_bucket: BucketHandle | None = None
        self._parent_name = f"[PERSONA]_[PARENT][GROUP]{self.group_id}"
        super().__init__(bucket_name, summary)

    async def _initiate_bucket(self):
        async with self._lock:
            if self.bucket is not None:
                return

            parent = await _memory_engine.set_bucket(
                title="[PERSONA]",
                summary="存放用户、群组画像",
                content="群与用户的画像存储父桶",
                summary_locked=True
            )
            self.parent_bucket = await parent.set_bucket(
                title=self._parent_name,
                summary="群画像的父桶",
                summary_locked=True
            )
            self.bucket = await self.parent_bucket.set_bucket(
                title=self.bucket_name,
                summary=self.summary,
                summary_locked=True if self.summary else False
            )

    def _load_advance_query_system(self):
        search_system = AgentPrompt(
            name="search_system",
            data_path=PROMPTS_DIR / "group_search_memory_system",
            description="全库搜索记忆的系统提示词",
            role="system"
        )
        self.prompts["search_system"] = search_system

    async def search_memory(self, query: str, include_time: bool = True) -> dict:
        await self._ensure_bucket_handle()
        if "search_system" not in self.prompts:
            self._load_advance_query_system()

        system_prompt = await self.prompts["search_system"].get_prompt()
        now = datetime.now()
        time_info = ""
        if include_time:
            time_info = f"> 当前时间: {timestamp2text(now.timestamp())}  \n\n"

        command = f"# 查询内容\n\n{query}"
        content = {}
        for _ in range(2):  # 只重试一次，写死
            try:
                response = await self.parent_bucket.advance_query(
                    system_prompt=system_prompt,
                    command=time_info + command
                )
                if not response:
                    continue

                text_response = response[0]
                if not (hasattr(text_response, "text") and text_response.text):
                    continue

                json_text = clean_json_text_from_llm(text_response.text)
                try:
                    content = json.loads(json_text)

                except json.JSONDecodeError:
                    continue

                else:
                    if not content or not isinstance(content, dict):
                        continue

                    if not content.get("answer", ""):
                        continue

                    break

            except Exception as E:
                _log.error(f"全库搜索记忆发生错误: {E}")
                _log.debug(traceback.format_exc())
                continue

        if not content:
            return {}

        answer = content.get("answer", "")
        matches = content.get("matches", {})
        alias_keys = [
            key
            for key in matches
            if key.startswith("memory_") or key.startswith("bucket_")
        ]
        resolved_aliases = await self.parent_bucket.resolve_aliases(alias_keys) if alias_keys else {}
        clean_matches = []
        for key, memory in matches.items():  # type: str, dict
            try:
                if key.startswith("memory_"):
                    mem_id = resolved_aliases[key]
                    memory_record = await self.parent_bucket.get_memory(mem_id)
                    if memory_record is None:
                        continue

                    bucket_id = memory_record.bucket_id

                elif key.startswith("bucket_"):
                    bucket_id = resolved_aliases[key]
                    mem_id = ""

                else:
                    continue

            except (KeyError, TypeError):
                continue

            else:
                memory["bucket_id"] = bucket_id
                memory["mem_id"] = mem_id
                clean_matches.append(memory)

        return {
            "answer": answer,
            "matches": clean_matches
        }

    @property
    def parent_bucket_id(self) -> str:
        if self.parent_bucket is None:
            return ""

        return self.parent_bucket.bucket_id


class PrivatePersona(BasePersona):
    """一个私聊关系对应一个记忆桶，并从同一来源生成三视角画像。"""

    def __init__(self, user_id: str, user_name: str = ""):
        self.user_id = str(user_id)
        self.user_name = str(user_name)
        super().__init__(
            bucket_name=f"[PERSONA]_[PRIVATE]{self.user_id}",
            summary=f"用户[{self.user_id}]与 BOT 的私聊关系记忆",
        )

    async def _add_first_memory(self):
        """只写入程序能够确定的双方基础身份，不生成推测画像。"""
        await self._ensure_bucket_handle()
        bucket = self.bucket
        if bucket is None:
            raise RuntimeError("私聊关系记忆桶初始化失败")

        await bucket.add_memory(
            "这是一个 QQ 一对一私聊关系的基础信息\n"
            f"用户 QQ号: {self.user_id}\n"
            f"用户 QQ昵称: {self.user_name or '未知'}\n"
            f"BOT QQ号: {get_bot_uid()}\n"
            f"BOT 昵称: {get_bot_name()}"
        )

    def _load_advance_query_system(self):
        self.prompts["persona_system"] = AgentPrompt(
            name="persona_system",
            data_path=PROMPTS_DIR / "private_persona_system",
            description="私聊用户、BOT 与关系三视角画像提示词",
            role="system",
        )

    async def get_persona(self) -> PrivatePersonaResult:
        """一次请求生成用户、BOT 和双方关系画像。"""
        await self._ensure_bucket_handle()
        bucket = self.bucket
        if bucket is None:
            raise RuntimeError("私聊关系记忆桶初始化失败")

        if "persona_system" not in self.prompts:
            self._load_advance_query_system()

        fallback = PrivatePersonaResult.empty(
            user_id=self.user_id,
            user_name=self.user_name,
        )
        system_prompt = await self.prompts["persona_system"].get_prompt()
        for _ in range(SETTING_CFG.Persona.GetPersonaRetry):
            try:
                response = await bucket.advance_query(system_prompt=system_prompt)
                if not response or not hasattr(response[0], "text"):
                    continue

                data = json.loads(clean_json_text_from_llm(response[0].text))
                if not isinstance(data, dict):
                    continue

                return PrivatePersonaResult.from_dict(
                    data,
                    user_id=self.user_id,
                    user_name=self.user_name,
                )
            except (json.JSONDecodeError, TypeError, IndexError) as exc:
                _log.debug(f"私聊关系画像解析失败: {exc}")

            except Exception as exc:
                _log.error(f"私聊关系画像生成失败: {exc}")
                _log.debug(traceback.format_exc())

        return fallback

    async def add_memory(
        self,
        content: str,
        include_time: bool = True,
        include_prefix: bool = True,
    ) -> AddResult:
        """向当前关系桶写入自由记忆，不要求调用方预先选择主体。"""
        await self._ensure_bucket_handle()
        bucket = self.bucket
        if bucket is None:
            raise RuntimeError("私聊关系记忆桶初始化失败")

        time_info = ""
        if include_time:
            time_info = f"> 当前时间: {timestamp2text(datetime.now().timestamp())}  \n\n"

        prefix = ""
        if include_prefix:
            prefix = (
                f"**新增用户[{self.user_id}]与 BOT 的私聊关系记忆**\n\n"
                f"- 用户 QQ昵称: `{self.user_name or '未知'}`\n\n"
            )

        return await bucket.add_memory(f"{time_info}{prefix}# 记忆内容\n\n{content}")

    async def query_memory(self, query: str, include_time: bool = True) -> QueryResult:
        """在当前唯一关系桶中查询记忆。"""
        return await self.get_memory(query, include_time=include_time)


class GroupPersona(BaseGroupPersona):
    def __init__(self, group: QQGroup):
        self.group_name = group.group_name
        bucket_name = f"[PERSONA]_[GROUP]{group.group_id}"
        summary = f"群[{group.group_id}]的群画像记忆"
        super().__init__(group.group_id, bucket_name, summary)

    def _load_advance_query_system(self):
        super()._load_advance_query_system()
        persona_system = AgentPrompt(
            name="persona_system",
            data_path=PROMPTS_DIR / "group_persona_system",
            description="获取群画像的提示词",
            role="system"
        )
        self.prompts["persona_system"] = persona_system

    async def _add_first_memory(self):
        await self._ensure_bucket_handle()
        basic_info = [
            "这是一个QQ群的基本信息  ",
            f"群号: [{self.group_id}]  ",
            f"群名: [{self.group_name}]"
        ]
        await self.bucket.add_memory('\n'.join(basic_info))

    async def get_persona(self) -> GroupPersonaResult:
        await self._ensure_bucket_handle()
        if "persona_system" not in self.prompts:
            self._load_advance_query_system()

        system_prompt = await self.prompts["persona_system"].get_prompt()
        retry_limit: int = SETTING_CFG.Persona.GetPersonaRetry  # 默认1次
        for _ in range(retry_limit):
            try:
                result = await self.parent_bucket.advance_query(
                    system_prompt=system_prompt
                )
                if not result:
                    continue

                json_text = cast(TextPrompt, result.prompts[0])
                if not hasattr(json_text, "text"):
                    continue

                json_text = clean_json_text_from_llm(json_text.text)
                json_dict: dict[str, Any] = json.loads(json_text)

                # 验证结果
                action_type = json_dict.get("group_action_type", "ORDINARY")
                hobby = json_dict.get("group_hobby", "")
                disgust = json_dict.get("group_disgust", "")
                vip = json_dict.get("group_vip", [])
                event = json_dict.get("group_event", "")
                memes = json_dict.get("group_memes", [])
                relations = json_dict.get("group_relations", {})

                if not (isinstance(action_type, str) and action_type in GroupActionType._value2member_map_):
                    continue

                if not isinstance(hobby, str):
                    continue

                if not isinstance(disgust, str):
                    continue

                if not isinstance(vip, list):
                    continue

                if not isinstance(event, str):
                    continue

                if not isinstance(memes, list):
                    continue

                if not isinstance(relations, dict):
                    continue

                persona_result = GroupPersonaResult(
                    group_id=self.group_id,
                    group_name=self.group_name,
                    group_action_type=GroupActionType(action_type),
                    group_hobby=hobby,
                    group_disgust=disgust,
                    group_vip=[member for member in vip if isinstance(member, str)],
                    group_event=event,
                    group_memes=[meme for meme in memes if isinstance(meme, str)],
                    group_relations={k: v for k, v in relations.items() if isinstance(k, str) and isinstance(v, str)}
                )
                return persona_result

            except json.JSONDecodeError as E:
                _log.debug(f"群画像解析失败: {E}")
                continue

            except Exception as E:
                _log.error(E)
                _log.debug(traceback.format_exc())
                continue

        return GroupPersonaResult(
            group_id=self.group_id,
            group_name=self.group_name
        )

    async def add_memory(self, content: str, include_time: bool = True, include_prefix: bool = True) -> AddResult:
        await self._ensure_bucket_handle()
        now = datetime.now()
        time_info = ""
        if include_time:
            time_info = f"> 当前时间: {timestamp2text(now.timestamp())}  \n\n"

        prefix = ""
        if include_prefix:
            prefix = (f"**新增QQ群[{self.group_id}]整体记忆**\n\n"
                      f"# QQ群当前信息\n\n"
                      f"- 群名: `{self.group_name}`\n")

        memory = f"# 记忆内容\n\n{content}"
        response = await self.bucket.add_memory(time_info + prefix + memory)
        return response


class GroupMemberPersona(BaseGroupPersona):
    _role_mapping = {
        "member": "普通群员",
        "owner": "群主",
        "admin": "管理员",
        "robot": "群机器人"
    }

    def __init__(self, member: Member):
        self.user_id = member.user_id
        self.member = member
        bucket_name = f"[PERSONA]_[GROUP]{member.group_id}_[MEMBER]{member.user_id}"
        summary = f"群[{member.group_id}]的群员{member.user_id}的用户画像记忆"
        super().__init__(member.group_id, bucket_name, summary)

    async def _add_first_memory(self):
        basic_info = [
            f"这是QQ群[{self.group_id}]中群员[{self.user_id}]的基本信息",
            f"用户名: {self.member.username}",
            f"群昵称: {self.member.nickname}",
            f"群头衔: {self.member.title}",
            f"群角色: {self._role_mapping[self.member.role.value]}",
            f"进群时间: {timestamp2text(self.member.join_time)}"
        ]
        await self.bucket.add_memory('\n'.join(basic_info))

    def _load_advance_query_system(self):
        persona_system = AgentPrompt(
            name="persona_system",
            data_path=PROMPTS_DIR / "member_persona_system",
            description="获取群员画像的提示词",
            role="system"
        )
        self.prompts["persona_system"] = persona_system

    async def get_persona(self) -> MemberPersonaResult:
        await self._ensure_bucket_handle()
        if "persona_system" not in self.prompts:
            self._load_advance_query_system()

        system_prompt = await self.prompts["persona_system"].get_prompt()
        retry_limit: int = SETTING_CFG.Persona.GetPersonaRetry  # 默认1次
        for _ in range(retry_limit):
            try:
                result = await self.bucket.advance_query(
                    system_prompt=system_prompt
                )
                if not result:
                    continue

                json_text = cast(TextPrompt, result.prompts[0])
                if not hasattr(json_text, "text"):
                    continue

                json_text = clean_json_text_from_llm(json_text.text)
                json_dict: dict[str, Any] = json.loads(json_text)

                # 验证结果
                alias = json_dict.get("alias", [])
                gender = json_dict.get("gender", "unknown")
                age = json_dict.get("age", 0)
                birthday = json_dict.get("birthday", "未知")
                city = json_dict.get("city", "未知")
                occupation = json_dict.get("occupation", "未知")
                hobby = json_dict.get("hobby", "未知")
                disgust = json_dict.get("disgust", "未知")
                working = json_dict.get("working", "未知")
                memes = json_dict.get("memes", [])
                catchphrase = json_dict.get("catchphrase", "未知")
                style = json_dict.get("style", "未知")
                relations = json_dict.get("relations", {})

                if not gender in {"male", "female", "unknown"}:
                    continue

                if not isinstance(age, int):
                    continue

                if any(not isinstance(key, list) for key in [alias, memes]):
                    continue

                if not isinstance(relations, dict):
                    continue

                if any(not isinstance(key, str) for key in
                       [birthday, city, occupation, hobby, disgust, working, catchphrase, style]):
                    continue

                persona_result = MemberPersonaResult(
                    user_id=self.user_id,
                    user_name=self.member.username,
                    nickname=self.member.nickname,
                    title=self.member.title,
                    role=self.member.role.value,
                    alias=alias,
                    gender=gender,
                    age=age,
                    birthday=birthday,
                    city=city,
                    occupation=occupation,
                    hobby=hobby,
                    disgust=disgust,
                    working=working,
                    memes=[meme for meme in memes if isinstance(meme, str)],
                    catchphrase=catchphrase,
                    style=style,
                    relations={k: v for k, v in relations.items() if isinstance(k, str) and isinstance(v, str)}
                )
                return persona_result

            except json.JSONDecodeError as E:
                _log.debug(f"群员画像解析失败: {E}")
                continue

            except Exception as E:
                _log.error(E)
                _log.debug(traceback.format_exc())
                continue

        return  MemberPersonaResult(
            user_id=self.user_id,
            user_name=self.member.username,
            nickname=self.member.nickname,
            title=self.member.title,
            role=self.member.role.value
        )

    async def add_memory(self, content: str, include_time: bool = True, include_prefix: bool = True) -> AddResult:
        await self._ensure_bucket_handle()
        now = datetime.now()
        time_info = ""
        if include_time:
            time_info = f"> 当前时间: {timestamp2text(now.timestamp())}  \n\n"

        prefix = ""
        if include_prefix:
            prefix = (f"**新增QQ群[{self.group_id}]群员[{self.user_id}]记忆**\n\n"
                      f"# 群员当前信息\n\n"
                      f"- 用户名: `{self.member.username}`\n"
                      f"- 群昵称: `{self.member.nickname}`\n"
                      f"- 群头衔: `{self.member.title}`\n"
                      f"- 群身份: `{self._role_mapping[self.member.role.value]}`\n\n\n")

        memory = f"# 记忆内容\n\n{content}"
        response = await self.bucket.add_memory(time_info + prefix + memory)
        return response
