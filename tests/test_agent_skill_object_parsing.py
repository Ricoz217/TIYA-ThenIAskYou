from __future__ import annotations

import sys
from pathlib import Path

from TIYA.agent.agent_skill import SKILL, SkillManager
from TIYA.model.model import SkillToolsFilteredResult


def test_addition_tools_require_config_and_package(tmp_path: Path) -> None:
    skill = SKILL(name="demo", description="demo", root=tmp_path)

    assert not SkillManager._check_skill_addition_tools(skill)

    (tmp_path / "tools.json").write_text("", encoding="utf-8")
    assert not SkillManager._check_skill_addition_tools(skill)

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("", encoding="utf-8")
    assert SkillManager._check_skill_addition_tools(skill)


def test_python_package_is_reused_without_reexecution(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text(
        "def tool() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )

    first, _ = SkillManager._getmember_from_python_package("demo", tools_dir)
    module_name = first["tool"].__module__
    try:
        second, _ = SkillManager._getmember_from_python_package("demo", tools_dir)
        assert second["tool"] is first["tool"]
    finally:
        for name in tuple(sys.modules):
            if name == module_name or name.startswith(module_name + "."):
                sys.modules.pop(name, None)


def test_object_tool_detection_handles_unions_and_builtins() -> None:
    class Result:
        pass

    def ordinary() -> dict:
        return {}

    def optional_object():
        return None

    def mixed_object():
        return 1

    optional_object.__annotations__["return"] = Result | None
    mixed_object.__annotations__["return"] = int | Result

    assert SkillManager._check_object_tool(ordinary, {"Result": Result}) == (
        "ORDINARY",
        [],
    )
    assert SkillManager._check_object_tool(optional_object, {"Result": Result}) == (
        "OBJECT",
        [Result],
    )
    assert SkillManager._check_object_tool(mixed_object, {"Result": Result}) == (
        "OBJECT",
        [Result],
    )
    assert SkillManager._check_object_tool(optional_object, {}) == ("INVALID", [])


def test_object_tool_detection_rejects_nested_object_returns() -> None:
    class Page:
        pass

    def nested_list():
        return []

    def deeply_nested_dict():
        return {}

    def ordinary_nested_dict():
        return {}

    nested_list.__annotations__["return"] = list[Page]
    deeply_nested_dict.__annotations__["return"] = dict[str, list[Page]]
    ordinary_nested_dict.__annotations__["return"] = dict[str, list[int]]

    assert SkillManager._check_object_tool(nested_list, {Page}) == (
        "NESTED",
        set(),
    )
    assert SkillManager._check_object_tool(deeply_nested_dict, {Page}) == (
        "NESTED",
        set(),
    )
    assert SkillManager._check_object_tool(ordinary_nested_dict, {Page}) == (
        "ORDINARY",
        set(),
    )

    result = SkillToolsFilteredResult(
        agent_level="default",
        all_tools={"nested_list": nested_list},
        allowed_tools={"nested_list": nested_list},
    )
    manager = object.__new__(SkillManager)

    manager._parse_object_oriented_tools(result, {Page})

    assert "nested_list" not in result.allowed_tools
    assert result.forbidden_tools["nested_list"] == "NESTED_OBJECT_RETURN"


def test_parse_class_methods_handles_descriptors_and_private_names() -> None:
    class Example:
        value = 1

        def normal(self) -> int:
            return 1

        def _private(self) -> int:
            return 2

        @staticmethod
        def static() -> int:
            return 3

        @classmethod
        def create(cls) -> Example:
            return cls()

        @property
        def property_value(self) -> int:
            return self.value

    methods = SkillManager._parse_class_methods(Example)

    assert set(methods) == {"normal", "static", "create"}
    assert methods["normal"] is Example.__dict__["normal"]
    assert methods["static"] is Example.__dict__["static"].__func__
    assert methods["create"] is Example.__dict__["create"].__func__


def test_recursive_object_method_parsing_stops_on_cycles() -> None:
    class Node:
        def next_node(self):
            return self

    def make_node():
        return Node()

    Node.next_node.__annotations__["return"] = Node
    make_node.__annotations__["return"] = Node
    result = SkillToolsFilteredResult(
        agent_level="default",
        all_tools={"make_node": make_node},
        allowed_tools={"make_node": make_node},
    )
    manager = object.__new__(SkillManager)

    manager._parse_object_oriented_tools(result, {"Node": Node})

    assert set(result.object_methods) == {"Node"}
    assert result.object_methods["Node"].sub_object_methods == {
        "next_node": ["Node"]
    }
