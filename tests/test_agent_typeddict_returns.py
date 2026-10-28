from __future__ import annotations

from types import SimpleNamespace
from typing import Annotated, NotRequired, Required, TypedDict

import pytest

import TIYA.agent.agent_object_tool as object_tool_module
from TIYA.agent.agent_object_tool import AgentObjectToolManager, ObjectToolReturnTypeError
from TIYA.agent.agent_skill import SkillManager, SkillToolsFilteredResult


class Result(TypedDict):
    title: str
    count: NotRequired[int]


class RecursiveResult(TypedDict):
    children: list[RecursiveResult]


class Page:
    pass


class ObjectFields(TypedDict):
    page: Required[Page]


class NestedResult(ObjectFields):
    children: NotRequired[list[NestedResult]]


@pytest.mark.parametrize("annotation", [
    Result, Result | None, list[Result], dict[str, list[Result]], RecursiveResult,
])
def test_typeddict_returns_remain_ordinary_tools(annotation):
    def tool():
        return {}

    tool.__annotations__["return"] = annotation
    assert SkillManager._check_object_tool(tool, set()) == ("ORDINARY", set())

    result = SkillToolsFilteredResult(
        agent_level="default", allowed_tools={"tool": tool},
    )
    manager = object.__new__(SkillManager)
    manager._parse_object_oriented_tools(result, set())
    assert result.ordinary_tools == {"tool": tool}
    assert not result.object_tools
    assert not result.object_methods


@pytest.mark.parametrize("annotation", [
    NestedResult, list[NestedResult], Annotated[NestedResult | None, "result"],
])
def test_typeddict_fields_still_reject_nested_objects(annotation):
    def tool():
        return {}

    tool.__annotations__["return"] = annotation
    assert SkillManager._check_object_tool(tool, {Page}) == ("NESTED", set())


def test_union_only_registers_the_actual_object_class():
    def tool() -> Page | Result:
        return {}

    assert SkillManager._check_object_tool(tool, {Page}) == ("OBJECT", {Page})


@pytest.mark.parametrize(("value", "annotation", "expected"), [
    ({"title": "hello"}, Result, True),
    ({}, Result, True),
    ({"title": 42}, Result, True),
    ([], Result, False),
    (None, Result | None, True),
    ([{"title": "hello"}], list[Result], True),
    ({"result": {}}, dict[str, Result], True),
])
def test_typeddict_runtime_checks_are_permissive(value, annotation, expected):
    assert AgentObjectToolManager._matches_annotation(value, annotation) is expected


def test_object_union_preserves_typeddict_result_and_checks_serialization(monkeypatch):
    warnings = []
    monkeypatch.setattr(object_tool_module, "_log", SimpleNamespace(warning=warnings.append))
    result = {"title": "hello"}

    def tool() -> Page | Result:
        return result

    manager = AgentObjectToolManager()
    wrapped = manager.register("demo", "tool", tool, {Page}, {Page})

    assert wrapped() is result
    assert not manager._hook_objects
    assert not warnings

    result["title"] = Page()
    with pytest.raises(ObjectToolReturnTypeError):
        wrapped()
