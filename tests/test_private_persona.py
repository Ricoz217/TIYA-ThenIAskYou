from __future__ import annotations

import asyncio
import json
import re
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from TIYA.memory.models import QueryResult
from TIYA.model import persona


class _Bucket:
    def __init__(self, title: str) -> None:
        self.title = title
        self.bucket_id = title
        self.children: dict[str, _Bucket] = {}
        self.memories: list[str] = []
        self.advance_response = None

    async def set_bucket(self, *, title: str, **_kwargs) -> "_Bucket":
        return self.children.setdefault(title, _Bucket(title))

    async def list_memories(self):
        return SimpleNamespace(total_memory_count=len(self.memories))

    async def add_memory(self, content: str):
        self.memories.append(content)
        return SimpleNamespace(success=True, message="ok")

    async def advance_query(self, **_kwargs):
        return self.advance_response

    async def query(self, query: str) -> QueryResult:
        return QueryResult(success=True, answer=query, result_source="LLM")


class _MemoryEngine(_Bucket):
    def __init__(self) -> None:
        super().__init__("ROOT")


def test_private_persona_uses_one_direct_relation_bucket(monkeypatch) -> None:
    async def run() -> None:
        engine = _MemoryEngine()
        monkeypatch.setattr(persona, "_memory_engine", engine)

        current = persona.PrivatePersona("10001", "Alice")
        other = persona.PrivatePersona("10002", "Bob")
        await asyncio.gather(
            current._ensure_bucket_handle(),
            other._ensure_bucket_handle(),
        )

        root = engine.children["[PERSONA]"]
        assert current.bucket is root.children["[PERSONA]_[PRIVATE]10001"]
        assert other.bucket is root.children["[PERSONA]_[PRIVATE]10002"]
        assert not any("[PARENT]" in title for title in root.children)
        assert not current.bucket.children
        assert "Alice" in current.bucket.memories[0]

    asyncio.run(run())


def test_private_persona_result_parses_each_field_independently() -> None:
    result = persona.PrivatePersonaResult.from_dict(
        {
            "user": {
                "alias": ["阿A"],
                "age": "不是数字",
                "hobby": "编程",
                "memes": ["梗一", 2],
                "relations": {"TIYA": "长期项目", "bad": 3},
                "unknown_field": "ignored",
            },
            "bot": {
                "personality": "愿意纠错",
                "gender": "invalid",
            },
            "relationship": {
                "relationship_type": "长期协作",
                "routines": ["审阅后再修改", None],
                "commitments": "错误类型",
            },
        },
        user_id="10001",
        user_name="Alice",
        bot_id="20001",
        bot_name="TIYA",
    )

    assert result.user.participant_id == "10001"
    assert result.user.alias == ["阿A"]
    assert result.user.age == 0
    assert result.user.hobby == "编程"
    assert result.user.memes == ["梗一"]
    assert result.user.relations == {"TIYA": "长期项目"}
    assert result.bot.participant_id == "20001"
    assert result.bot.personality == "愿意纠错"
    assert result.bot.gender == "unknown"
    assert result.relationship.relationship_type == "长期协作"
    assert result.relationship.routines == ["审阅后再修改"]
    assert result.relationship.commitments == []


def test_private_persona_generates_three_views_in_one_request(monkeypatch) -> None:
    async def run() -> None:
        engine = _MemoryEngine()
        monkeypatch.setattr(persona, "_memory_engine", engine)
        monkeypatch.setattr(persona, "get_bot_uid", lambda: "20001")
        monkeypatch.setattr(persona, "get_bot_name", lambda: "TIYA")
        monkeypatch.setattr(
            persona,
            "SETTING_CFG",
            SimpleNamespace(Persona=SimpleNamespace(GetPersonaRetry=1)),
        )

        private = persona.PrivatePersona("10001", "Alice")
        await private._ensure_bucket_handle()
        private.bucket.advance_response = [
            SimpleNamespace(
                text=json.dumps(
                    {
                        "user": {"occupation": "程序员"},
                        "bot": {"style": "先分析再行动"},
                        "relationship": {"current_event": "重构私聊 Agent"},
                    },
                    ensure_ascii=False,
                )
            )
        ]

        result = await private.get_persona()

        assert result.user.occupation == "程序员"
        assert result.bot.style == "先分析再行动"
        assert result.relationship.current_event == "重构私聊 Agent"

    asyncio.run(run())


def test_group_persona_accepts_markdown_fenced_json(monkeypatch) -> None:
    async def run() -> None:
        engine = _MemoryEngine()
        monkeypatch.setattr(persona, "_memory_engine", engine)
        monkeypatch.setattr(
            persona,
            "SETTING_CFG",
            SimpleNamespace(Persona=SimpleNamespace(GetPersonaRetry=1)),
        )

        group = SimpleNamespace(group_id="806869605", group_name="StarRail Omega Project")
        group_persona = persona.GroupPersona(group)
        await group_persona._ensure_bucket_handle()
        group_persona.parent_bucket.advance_response = SimpleNamespace(
            prompts=[
                persona.TextPrompt(
                    "assistant",
                    "```json\n"
                    + json.dumps(
                        {
                            "group_action_type": "GAME",
                            "group_hobby": "CS2",
                            "group_disgust": "草台班子",
                            "group_vip": ["923002166", 123],
                            "group_event": "讨论比赛",
                            "group_memes": ["警钟撅烂", 456],
                            "group_relations": {"923002166": "开发者", "bad": 789},
                        },
                        ensure_ascii=False,
                    )
                    + "\n```",
                )
            ]
        )

        result = await group_persona.get_persona()

        assert result.group_action_type is persona.GroupActionType.GAME
        assert result.group_hobby == "CS2"
        assert result.group_vip == ["923002166"]
        assert result.group_memes == ["警钟撅烂"]
        assert result.group_relations == {"923002166": "开发者"}

    asyncio.run(run())


def test_group_member_persona_accepts_markdown_fenced_json(monkeypatch) -> None:
    async def run() -> None:
        engine = _MemoryEngine()
        monkeypatch.setattr(persona, "_memory_engine", engine)
        monkeypatch.setattr(
            persona,
            "SETTING_CFG",
            SimpleNamespace(Persona=SimpleNamespace(GetPersonaRetry=1)),
        )

        member = SimpleNamespace(
            group_id="806869605",
            user_id="923002166",
            username="Rico",
            nickname="尼尼孩孩",
            title="",
            role=SimpleNamespace(value="member"),
            join_time=0,
        )
        member_persona = persona.GroupMemberPersona(member)
        await member_persona._ensure_bucket_handle()
        member_persona.bucket.advance_response = SimpleNamespace(
            prompts=[
                persona.TextPrompt(
                    "assistant",
                    "```json\n"
                    + json.dumps(
                        {
                            "alias": ["尼尼", 1],
                            "gender": "unknown",
                            "age": 0,
                            "hobby": "写 Agent",
                            "memes": ["喵", 1],
                            "relations": {"布布": "群聊机器人", "bad": None},
                        },
                        ensure_ascii=False,
                    )
                    + "\n```",
                )
            ]
        )

        result = await member_persona.get_persona()

        assert result.alias == ["尼尼", 1]
        assert result.hobby == "写 Agent"
        assert result.memes == ["喵"]
        assert result.relations == {"布布": "群聊机器人"}

    asyncio.run(run())


def test_private_persona_exposes_one_memory_query(monkeypatch) -> None:
    async def run() -> None:
        engine = _MemoryEngine()
        monkeypatch.setattr(persona, "_memory_engine", engine)
        private = persona.PrivatePersona("10001", "Alice")

        result = await private.query_memory("我们在做什么")

        assert result.success
        assert result.answer.endswith("我们在做什么")

    asyncio.run(run())


def test_private_persona_result_serializes_nested_schema() -> None:
    result = persona.PrivatePersonaResult.empty(
        user_id="10001",
        user_name="Alice",
        bot_id="20001",
        bot_name="TIYA",
    ).to_dict()

    assert set(result) == {"user", "bot", "relationship"}
    assert result["user"]["participant_id"] == "10001"
    assert result["bot"]["participant_id"] == "20001"
    assert result["relationship"]["relationship_stage"] == "初识"


def test_private_persona_prompt_schema_matches_result_models() -> None:
    root = Path(__file__).parents[1]
    prompt = (
        root / "data/prompt/private_persona_system/private_persona_system.md"
    ).read_text(encoding="utf-8")
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", prompt, flags=re.DOTALL)
    schema = json.loads(json_blocks[-1])

    participant_fields = {
        item.name
        for item in fields(persona.PrivateParticipantPersonaResult)
        if item.name not in {"participant_id", "participant_name"}
    }
    relationship_fields = {
        item.name for item in fields(persona.PrivateRelationshipPersonaResult)
    }

    assert set(schema) == {"user", "bot", "relationship"}
    assert set(schema["user"]) == participant_fields
    assert set(schema["bot"]) == participant_fields
    assert set(schema["relationship"]) == relationship_fields
