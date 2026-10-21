from __future__ import annotations
"""
model.py
存放各种各样的数据结构
"""
__version__ = "0.1.0"
from typing import Callable, Literal
from dataclasses import dataclass, field

# =========================================================
# agent/agent_skill.py
# =========================================================

@dataclass(slots=True)
class ObjectToolsMethods:
    hook_class: object
    methods: dict[str, Callable] = field(default_factory=dict)
    sub_object_methods: dict[str, list[str]] = field(default_factory=dict)  # 对象方法指向下一个对象 {method: class_name}

    @property
    def name(self) -> str:
        return self.hook_class.__name__

@dataclass(slots=True)
class SkillToolsFilteredResult:
    FORBID_REASON = Literal["WHITELIST", "BLACKLIST", "CLASS_NOT_MATCH"]
    agent_level: str
    all_tools: dict[str, Callable] = field(default_factory=dict)
    allowed_tools: dict[str, Callable] = field(default_factory=dict)
    ordinary_tools: dict[str, Callable] = field(default_factory=dict)
    object_tools: dict[str, Callable] = field(default_factory=dict)  # 面向对象方法
    object_methods: dict[str, ObjectToolsMethods] = field(default_factory=dict)  # 该对象支持的方法
    forbidden_tools: dict[str, FORBID_REASON] = field(default_factory=dict)  # {函数名: 理由}