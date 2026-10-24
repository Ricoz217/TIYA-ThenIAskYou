from __future__ import annotations
"""
agent_object_tool.py
对象工具管理器，负责对象工具的对象生命周期管理，以及所有的对象工具 <-> LLM 的转换层
"""
__version__ = "0.1.0"

import time
import inspect

from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Awaitable, Any, TypedDict

from TIYA.time_id import next_time_id
from TIYA.config import SETTING_CFG


@dataclass(slots=True)
class AgentObjectTool:
    skill_name: str
    method_name: str
    origin_method: Callable
    wrapped_method: Callable


@dataclass(slots=True)
class ObjectProxy:
    hook_object: Any
    source_skill_name: str
    source_method_name: str
    create_time: float
    renewal_time: float
    ttl: float
    object_id: str

    def expired_at(self, now: float) -> str:
        """返回自动回收的 提示字符串，主要是 LLM 阅读友好"""
        if now is None:
            now = time.time()

        expire = self.renewal_time + self.ttl
        life_time = now - expire
        if life_time < 0:
            return "已过期"

        elif life_time <= 60:
            return "一分钟内"

        elif life_time <= 3600:
            if life_time == 3600:
                return "1 小时"

            minutes = life_time // 60 + 1
            return f"{minutes} 分钟"

        elif life_time <= 3600 * 6:
            if life_time == 3600 * 6:
                return "6 小时"

            hours = life_time // 3600 + 1
            minutes = (life_time % 3600) // 60 + 1
            return f"{hours} 小时 {minutes} 分钟"

        else:
            return datetime.fromtimestamp(expire).isoformat()

    def expired(self, now: float = None) -> bool:
        """判断是否过期"""
        if now is None:
            now = time.time()

        return now - self.ttl - self.create_time < 0

    def kill(self):
        self.hook_object = None


class _ObjectAttributeSetResult(TypedDict):
    success: bool
    new_value: Any
    message: str


class AgentObjectToolManager:
    """类对象方法管理器"""

    def __init__(self):
        # 数据结构
        self.object_tools: dict[str, dict[str, AgentObjectTool]] = {}  # {skill_name: {method_name: 对象工具封装}}，这个管注册
        self._hook_objects: dict[str, ObjectProxy] = {}  # {object_id: 托管对象}
        self._skill_to_objects_mapping: dict[str, dict[str, list[str]]] = {}  # {skill_name: {method_name: [object_id]}}，这个管运行

    """
    框架逻辑
    """

    def register(self, skill_name: str, func: Callable[..., Any | Awaitable]) -> Callable:
        """注册对象函数、包装函数对象"""
        func_name = func.__name__
        def _convert_result(_result: Any) -> Any | str:
            """一个闭包，对结果转换，若是内置类型则原样返回，若是类则托管，且分配对象 id"""
            if type(_result).__module__ == "builtins":
                return _result

            else:
                object_id = f"{skill_name}:{func_name}:{next_time_id()}"
                while object_id in self._hook_objects:
                    object_id = f"{skill_name}:{func_name}:{next_time_id()}"

                now = time.time()
                self._hook_objects[object_id] = ObjectProxy(
                    hook_object=_result,
                    source_skill_name=skill_name,
                    source_method_name=func_name,
                    create_time=now,
                    renewal_time=now,
                    ttl=SETTING_CFG.Agent.DefaultHookObjectTTL,  # 默认一小时
                    object_id=object_id
                )
                self._skill_to_objects_mapping.setdefault(skill_name, {}).setdefault(func_name, []).append(object_id)
                return object_id

        if inspect.iscoroutinefunction(func):
            async def _async_tool_wrapper(*args, **kwargs) -> Any:
                """异步函数包装器"""
                result = await func(*args, **kwargs)
                return _convert_result(result)

            wrapped = wraps(func)(_async_tool_wrapper)

        else:
            def _sync_tool_wrapper(*args, **kwargs) -> Any:
                """同步函数包装器"""
                result = func(*args, **kwargs)
                return _convert_result(result)

            wrapped = wraps(func)(_sync_tool_wrapper)

        package = AgentObjectTool(
            skill_name=skill_name,
            method_name=func_name,
            origin_method=func,
            wrapped_method=wrapped
        )
        self.object_tools.setdefault(skill_name, {})[func_name] = package
        return wrapped

    """
    对象方法通用接口
    """

    def get_attr(self, object_id: str, attribute: str) -> Any:
        """获取指定对象的属性"""
        ...

    def set_attr(self, object_id: str, attribute: str, value: Any) -> _ObjectAttributeSetResult:
        """设置指定对象的属性"""
        ...

    def get_object_info(self, object_id: str) -> dict:
        """获取指定对象信息"""
        ...

    def get_object_methods(self, object_id: str) -> dict:
        """获取指定对象的所有方法"""
        ...

    def get_object_method_schema(self, object_id: str, method_name: str) -> dict:
        """获取指定对象方法的详细签名，具体使用 schema"""
        ...

    async def object_method_executor(self, object_id: str, method_name: str, arguments: dict) -> Any:
        """执行对象方法"""
        ...

    def kill_object(self, object_id):
        """回收指定对象"""
        ...

    def kill_object_many(self, skill_name: str, method_name: str):
        """批量回收对象"""
        ...