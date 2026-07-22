from __future__ import annotations
"""
global_vars.py
存放全局变了，不依赖内部组件
"""
__version__ = "0.1.0"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TIYA.private_chat import PrivateChat
    from TIYA.qq_group import QQGroup

QQ_GROUPS: dict[str, QQGroup] = {}
PRIVATE_CHATS: dict[str, PrivateChat] = {}
