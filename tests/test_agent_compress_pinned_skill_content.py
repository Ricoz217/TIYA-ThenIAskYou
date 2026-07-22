from __future__ import annotations

from types import SimpleNamespace

from TIYA.LLM_connect import ToolCall
from TIYA.agent.agent import BaseAgent


class _FakeSkill:
    def __init__(self) -> None:
        self.md_content = {
            "group_chat": {
                "Usage": {"_content": "usage-body"},
                "16) setu": {"_content": "setu-body"},
            }
        }


class _FakeSkillManager:
    def __init__(self) -> None:
        self.skills = {"group_chat": _FakeSkill()}
        self.requested: list[tuple[str, dict]] = []

    def get_skill_content(self, skill_name: str, title_tree: dict) -> str:
        self.requested.append((skill_name, title_tree))
        return f"CONTENT:{title_tree}"


def _make_agent_with_context(context: list[object]) -> tuple[BaseAgent, _FakeSkillManager]:
    agent = object.__new__(BaseAgent)
    skill_manager = _FakeSkillManager()
    agent.control = SimpleNamespace(context=context)
    agent._skills = skill_manager
    agent.list_external_tools = lambda: {}
    agent.load_external_tool = lambda source_name, tool_name: "工具不存在"
    return agent, skill_manager


def test_compress_pinned_ignores_string_leaf_title_tree() -> None:
    agent, skill_manager = _make_agent_with_context([
        ToolCall(
            "run_external_tool",
            "bad-call",
            {
                "source_name": "skill_installer",
                "tool_name": "get_skill_content",
                "arguments": {
                    "skill_name": "group_chat",
                    "title_tree": {"group_chat": "Usage"},
                },
            },
        )
    ])

    pinned = agent._get_compress_context_pinned()

    assert "已加载SKILL内容" not in pinned
    assert skill_manager.requested == []


def test_compress_pinned_merges_only_valid_title_trees() -> None:
    agent, skill_manager = _make_agent_with_context([
        ToolCall(
            "run_external_tool",
            "valid-usage",
            {
                "source_name": "skill_installer",
                "tool_name": "get_skill_content",
                "arguments": {
                    "skill_name": "group_chat",
                    "title_tree": {"group_chat": {"Usage": {}}},
                },
            },
        ),
        ToolCall(
            "run_external_tool",
            "bad-setu",
            {
                "source_name": "skill_installer",
                "tool_name": "get_skill_content",
                "arguments": {
                    "skill_name": "group_chat",
                    "title_tree": {"group_chat": "16) setu"},
                },
            },
        ),
    ])

    pinned = agent._get_compress_context_pinned()

    assert "已加载SKILL内容" in pinned
    assert "Usage" in pinned
    assert "16) setu" not in pinned
    assert skill_manager.requested == [("group_chat", {"group_chat": {"Usage": {}}})]
