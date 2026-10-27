from __future__ import annotations

import sys
from pathlib import Path

from TIYA.agent.agent_skill import SKILL, SkillManager
from TIYA.model.model import SkillToolsFilteredResult


def test_addition_tools_only_require_python_package(tmp_path: Path) -> None:
    skill = SKILL(name="demo", description="demo", root=tmp_path)

    assert not SkillManager._check_skill_addition_tools(skill)

    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("", encoding="utf-8")
    assert SkillManager._check_skill_addition_tools(skill)


def test_missing_or_empty_tools_config_uses_open_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "tools.json"

    missing = SkillManager._read_skill_tools_config(config_file)
    assert missing.agent_levels == ()
    assert missing.whitelist == ()
    assert missing.blacklist == ()
    assert missing.objects == {}
    assert missing.tools_setting == {}

    config_file.write_text("", encoding="utf-8")
    assert SkillManager._read_skill_tools_config(config_file) == missing


def test_tools_config_rejects_invalid_json_and_field_types(tmp_path: Path) -> None:
    config_file = tmp_path / "tools.json"
    config_file.write_text("{", encoding="utf-8")

    try:
        SkillManager._read_skill_tools_config(config_file)
    except ValueError as exc:
        assert "tools.json" in str(exc)
    else:
        raise AssertionError("invalid JSON should fail")

    config_file.write_text('{"whitelist": "browse"}', encoding="utf-8")
    try:
        SkillManager._read_skill_tools_config(config_file)
    except TypeError as exc:
        assert "whitelist" in str(exc)
    else:
        raise AssertionError("invalid whitelist should fail")

    config_file.write_text('{"agent_levels": "websearch"}', encoding="utf-8")
    try:
        SkillManager._read_skill_tools_config(config_file)
    except TypeError as exc:
        assert "agent_levels" in str(exc)
    else:
        raise AssertionError("invalid agent_levels should fail")


def test_tools_config_reads_optional_object_class_targets(tmp_path: Path) -> None:
    config_file = tmp_path / "tools.json"
    config_file.write_text(
        """
        {
          "agent_levels": ["websearch"],
          "objects": {"BrowserSession": "browser:PlaywrightBrowserSession"},
          "whitelist": ["browse", "BrowserSession.*"],
          "blacklist": ["BrowserSession.close"]
        }
        """,
        encoding="utf-8",
    )

    config = SkillManager._read_skill_tools_config(config_file)

    assert config.agent_levels == ("websearch",)
    assert config.objects == {
        "BrowserSession": "browser:PlaywrightBrowserSession"
    }
    assert config.whitelist == ("browse", "BrowserSession.*")
    assert config.blacklist == ("BrowserSession.close",)


def test_agent_level_filter_skips_package_before_import(tmp_path: Path) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text(
        "raise RuntimeError('tools package should not be imported')\n",
        encoding="utf-8",
    )
    (tmp_path / "tools.json").write_text(
        '{"agent_levels": ["websearch"]}',
        encoding="utf-8",
    )
    skill = SKILL(name="demo", description="demo", root=tmp_path)
    manager = object.__new__(SkillManager)
    manager.agent_level = "main"
    import_attempts = []

    def record_import(*args):
        import_attempts.append(args)
        raise RuntimeError("tools package should not be imported")

    manager._getmember_from_python_package = record_import
    module_name = SkillManager._skill_tools_module_name("demo", tools_dir)

    manager._load_skill_addition_tools(skill)

    assert import_attempts == []
    assert module_name not in sys.modules


def test_filter_defaults_to_public_and_blacklist_wins() -> None:
    def browse() -> str:
        return ""

    def search() -> str:
        return ""

    def _private() -> str:
        return ""

    manager = object.__new__(SkillManager)
    manager.agent_level = "default"
    tools = {"browse": browse, "search": search, "_private": _private}

    default_result = manager._addition_tools_filter([], [], tools)
    assert set(default_result.allowed_tools) == {"browse", "search"}

    empty_whitelist = manager._addition_tools_filter([], [], tools)
    assert set(empty_whitelist.allowed_tools) == {"browse", "search"}

    selected = manager._addition_tools_filter(
        ["browse", "Page.*", "_private"],
        ["browse"],
        {**tools, "Page.click": browse, "Page._secret": browse},
    )
    assert set(selected.allowed_tools) == {"Page.click"}
    assert selected.forbidden_tools["browse"] == "BLACKLIST"
    assert selected.forbidden_tools["search"] == "WHITELIST"
    assert "_private" not in selected.all_tools
    assert "Page._secret" not in selected.all_tools

    denied = manager._addition_tools_filter([], ["*"], tools)
    assert not denied.allowed_tools
    assert set(denied.forbidden_tools) == {"browse", "search"}


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


def test_configured_object_classes_are_loaded_relative_to_tools_package(
        tmp_path: Path
) -> None:
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text(
        "def browse() -> str:\n"
        "    return ''\n",
        encoding="utf-8",
    )
    (tools_dir / "browser.py").write_text(
        "class PlaywrightBrowserSession:\n"
        "    pass\n",
        encoding="utf-8",
    )

    tools, _ = SkillManager._getmember_from_python_package("demo", tools_dir)
    module_name = tools["browse"].__module__
    try:
        classes = SkillManager._load_configured_object_classes(
            "demo",
            tools_dir,
            {"BrowserSession": "browser:PlaywrightBrowserSession"},
        )
        loaded_class = classes["BrowserSession"]
        assert loaded_class.__name__ == "PlaywrightBrowserSession"
        assert loaded_class.__module__ == f"{module_name}.browser"
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

    assert SkillManager._check_object_tool(ordinary, {Result}) == (
        "ORDINARY",
        set(),
    )
    assert SkillManager._check_object_tool(optional_object, {Result}) == (
        "OBJECT",
        {Result},
    )
    assert SkillManager._check_object_tool(mixed_object, {Result}) == (
        "OBJECT",
        {Result},
    )
    assert SkillManager._check_object_tool(optional_object, set()) == ("INVALID", set())


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

    manager._parse_object_oriented_tools(result, {Node})

    assert set(result.object_methods) == {Node}
    assert result.object_methods[Node].sub_object_methods == {
        "next_node": [Node]
    }


def test_return_annotations_are_object_roots_and_member_rules_apply() -> None:
    class Page:
        def click(self) -> str:
            return "clicked"

        def close(self) -> None:
            return None

        def _internal(self) -> str:
            return "secret"

    def browse() -> Page:
        return Page()

    browse.__annotations__["return"] = Page

    result = SkillToolsFilteredResult(
        agent_level="default",
        all_tools={"browse": browse},
        allowed_tools={"browse": browse},
    )
    manager = object.__new__(SkillManager)

    manager._parse_object_oriented_tools(
        result,
        set(),
        whitelist=("browse", "Page.*"),
        blacklist=("Page.close",),
    )

    assert set(result.object_methods) == {Page}
    assert set(result.object_methods[Page].methods) == {"click"}
    assert result.forbidden_tools["Page.close"] == "BLACKLIST"
    assert "Page._internal" not in result.forbidden_tools


def test_explicit_object_class_gets_its_own_public_method_directory() -> None:
    class BrowserSession:
        def close(self) -> None:
            return None

    class PlaywrightBrowserSession(BrowserSession):
        def inspect_dom(self) -> str:
            return ""

    def browse() -> BrowserSession:
        return PlaywrightBrowserSession()

    browse.__annotations__["return"] = BrowserSession
    result = SkillToolsFilteredResult(
        agent_level="default",
        all_tools={"browse": browse},
        allowed_tools={"browse": browse},
    )
    manager = object.__new__(SkillManager)

    manager._parse_object_oriented_tools(
        result,
        {BrowserSession, PlaywrightBrowserSession},
        configured_classes={PlaywrightBrowserSession},
        class_names={
            BrowserSession: "BrowserSession",
            PlaywrightBrowserSession: "BrowserSession",
        },
    )

    assert set(result.object_methods) == {
        BrowserSession,
        PlaywrightBrowserSession,
    }
    assert set(result.object_methods[PlaywrightBrowserSession].methods) == {
        "inspect_dom"
    }
