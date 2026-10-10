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
class SkillToolsFilteredResult:
    FORBID_REASON = Literal["WHITELIST", "BLACKLIST"]
    agent_level: str
    all_tools: dict[str, Callable] = field(default_factory=dict)
    allowed_tools: dict[str, Callable] = field(default_factory=dict)
    forbidden_tools: dict[str, FORBID_REASON] = field(default_factory=dict)  # {函数名: 理由}