from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import get_type_hints

import pytest

import TIYA.agent.agent_object_tool as object_tool_module
from TIYA.agent.agent_object_tool import (
    AgentObjectToolManager,
    ObjectProxy,
    ObjectToolRegistrationError,
    ObjectToolReturnTypeError,
    ObjectToolAttributeError,
    ObjectToolProxyNotFoundError,
)
from TIYA.LLM_connect import build_tool_payloads


@pytest.fixture
def attribute_object(monkeypatch):
    class Example:
        def __init__(self):
            self.value = 1
            self.reads = 0
            self._secret = "private"
            self.unserializable = object()

        @property
        def title(self):
            self.reads += 1
            return "example"

        @property
        def rejected(self):
            return 1

        @rejected.setter
        def rejected(self, value):
            raise ValueError("setter rejected value")

        @property
        def label(self):
            raise AssertionError("setting label must not read its getter")

        @label.setter
        def label(self, value):
            self._label = value.upper()

        def method(self):
            return None

    clock = [120.0]
    monkeypatch.setattr(object_tool_module.time, "time", lambda: clock[0])
    obj = Example()
    proxy = ObjectProxy(obj, "demo", "create", 100.0, 100.0, 60.0, "obj:demo:create:1")
    manager = AgentObjectToolManager()
    manager._hook_objects[proxy.object_id] = proxy
    return manager, proxy, obj, clock


def test_get_attr_reads_property_once_and_renews_only_on_success(attribute_object):
    manager, proxy, obj, clock = attribute_object

    assert manager.get_attr(proxy.object_id, "title") == "example"
    assert obj.reads == 1
    assert proxy.renewal_time == clock[0]

    clock[0] += 10
    for name in ("_secret", "missing", "unserializable", "method"):
        with pytest.raises(ObjectToolAttributeError):
            manager.get_attr(proxy.object_id, name)
    assert proxy.renewal_time == 120.0


def test_set_attr_changes_existing_attributes_and_preserves_failed_ttl(attribute_object):
    manager, proxy, obj, clock = attribute_object

    result = manager.set_attr(proxy.object_id, "value", {"new": [1, 2]})
    assert result["success"] is True
    assert result["new_value"] == {"new": [1, 2]}
    assert result["message"]
    assert obj.value == {"new": [1, 2]}
    assert proxy.renewal_time == clock[0]

    clock[0] += 10
    for name, value in (("_secret", 2), ("missing", 2), ("method", 2), ("value", object())):
        with pytest.raises(ObjectToolAttributeError):
            manager.set_attr(proxy.object_id, name, value)
    with pytest.raises(AttributeError):
        manager.set_attr(proxy.object_id, "title", "changed")
    with pytest.raises(ValueError, match="setter rejected"):
        manager.set_attr(proxy.object_id, "rejected", 2)
    assert obj.value == {"new": [1, 2]}
    assert not hasattr(obj, "missing")
    assert proxy.renewal_time == 120.0


def test_set_attr_calls_setter_without_reading_property(attribute_object):
    manager, proxy, obj, clock = attribute_object

    result = manager.set_attr(proxy.object_id, "label", "hello")

    assert obj._label == "HELLO"
    assert result["new_value"] == "hello"
    assert proxy.renewal_time == clock[0]


def test_renewal_object_updates_ttl_from_current_time(attribute_object):
    manager, proxy, _, clock = attribute_object

    result = manager.renewal_object(proxy.object_id, 30.0)

    assert proxy.ttl == 30.0
    assert proxy.renewal_time == clock[0]
    assert result == {"object_id": proxy.object_id, "ttl": 30.0, "expires_at": 150.0}

    clock[0] = 140.0
    manager.get_attr(proxy.object_id, "value")
    assert proxy.ttl == 30.0
    assert proxy.renewal_time == 140.0
    assert proxy.expired(170.0)


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan"), True, "60"])
def test_renewal_object_rejects_invalid_ttl_without_mutation(attribute_object, ttl):
    manager, proxy, _, _ = attribute_object
    with pytest.raises((TypeError, ValueError)):
        manager.renewal_object(proxy.object_id, ttl)
    assert proxy.ttl == 60.0
    assert proxy.renewal_time == 100.0


@pytest.mark.parametrize("operation", ["get_attr", "set_attr", "renewal_object"])
def test_object_access_rejects_missing_and_expired_objects(attribute_object, operation):
    manager, proxy, _, clock = attribute_object
    method = getattr(manager, operation)
    args = {"get_attr": ("value",), "set_attr": ("value", 2), "renewal_object": (60.0,)}[operation]
    with pytest.raises(ObjectToolProxyNotFoundError):
        method("missing-id", *args)

    clock[0] = 160.0
    with pytest.raises(ObjectToolAttributeError, match="回收"):
        method(proxy.object_id, *args)
    assert proxy._hook_object is None
    assert proxy.renewal_time == 100.0

    clock[0] = 120.0
    with pytest.raises(ObjectToolAttributeError, match="回收"):
        method(proxy.object_id, *args)


def test_attribute_tools_have_parameter_and_return_descriptions():
    manager = AgentObjectToolManager()
    for method in (manager.get_attr, manager.set_attr, manager.renewal_object):
        schema = build_tool_payloads(method)["openai"]["function"]
        assert schema["description"]
        assert all(item.get("description") for item in schema["parameters"]["properties"].values())
        assert ":return:" in inspect.getdoc(method)


def test_object_proxy_uses_sliding_expiration_time() -> None:
    proxy = ObjectProxy(
        _hook_object=object(),
        source_skill_name="demo",
        source_method_name="create",
        create_time=10.0,
        renewal_time=100.0,
        ttl=60.0,
        object_id="obj:demo:create:1",
    )

    assert not proxy.expired(159.9)
    assert proxy.expired_at(159.9) != "已过期"
    assert proxy.expired(160.0)
    assert proxy.expired_at(160.0) == "已过期"


def test_register_wraps_allowed_concrete_subclass_and_reuses_object_id(
        monkeypatch: pytest.MonkeyPatch
) -> None:
    class Session:
        pass

    class ConcreteSession(Session):
        pass

    session = ConcreteSession()

    async def browse() -> Session:
        return session

    browse.__annotations__["return"] = Session | None
    monkeypatch.setattr(
        object_tool_module,
        "SETTING_CFG",
        SimpleNamespace(Agent=SimpleNamespace(DefaultHookObjectTTL=3600)),
    )

    manager = AgentObjectToolManager()
    wrapped = manager.register(
        "websearch",
        "browse",
        browse,
        declared_object_classes={Session},
        skill_classes={Session, ConcreteSession},
    )

    first_id = asyncio.run(wrapped())
    second_id = asyncio.run(wrapped())

    assert first_id == second_id
    assert inspect.iscoroutinefunction(wrapped)
    assert first_id.startswith("obj:websearch:browse:")
    assert manager._hook_objects[first_id]._hook_object is session
    assert get_type_hints(wrapped)["return"] == str | None
    assert inspect.signature(wrapped).return_annotation == str | None
    assert build_tool_payloads(wrapped)["openai"]["function"]["returns"] == {
        "type": ["string", "null"],
    }


def test_register_allows_serializable_mismatch_but_rejects_illegal_object(
        monkeypatch: pytest.MonkeyPatch
) -> None:
    class Page:
        pass

    class InternalClient:
        pass

    def serializable_mismatch() -> Page:
        return {"usable": True}  # type: ignore[return-value]

    def illegal_object() -> Page:
        return InternalClient()  # type: ignore[return-value]

    serializable_mismatch.__annotations__["return"] = Page
    illegal_object.__annotations__["return"] = Page
    warnings: list[str] = []
    monkeypatch.setattr(
        object_tool_module,
        "_log",
        SimpleNamespace(warning=warnings.append),
    )

    manager = AgentObjectToolManager()
    loose = manager.register(
        "demo",
        "loose",
        serializable_mismatch,
        declared_object_classes={Page},
        skill_classes={Page},
    )
    illegal = manager.register(
        "demo",
        "illegal",
        illegal_object,
        declared_object_classes={Page},
        skill_classes={Page, InternalClient},
    )

    assert loose() == {"usable": True}
    with pytest.raises(ObjectToolReturnTypeError):
        illegal()
    assert len(warnings) == 2


def test_register_is_idempotent_but_rejects_name_collisions() -> None:
    class Page:
        pass

    def first() -> Page:
        return Page()

    def second() -> Page:
        return Page()

    first.__annotations__["return"] = Page
    second.__annotations__["return"] = Page

    manager = AgentObjectToolManager()
    wrapped = manager.register(
        "demo",
        "create",
        first,
        declared_object_classes={Page},
        skill_classes={Page},
    )

    assert manager.register(
        "demo",
        "create",
        first,
        declared_object_classes={Page},
        skill_classes={Page},
    ) is wrapped

    with pytest.raises(ObjectToolRegistrationError):
        manager.register(
            "demo",
            "create",
            second,
            declared_object_classes={Page},
            skill_classes={Page},
        )
