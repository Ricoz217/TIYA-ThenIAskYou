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
)
from TIYA.LLM_connect import build_tool_payloads


def test_object_proxy_uses_sliding_expiration_time() -> None:
    proxy = ObjectProxy(
        hook_object=object(),
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
    assert manager._hook_objects[first_id].hook_object is session
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
