from __future__ import annotations
"""
agent_object_tool.py
对象工具管理器，负责对象工具的对象生命周期管理，以及所有的对象工具 <-> LLM 的转换层
"""
__version__ = "0.1.0"

import time
import inspect
import json

from functools import reduce, wraps
from math import ceil
from operator import or_
from dataclasses import dataclass
from datetime import datetime
from types import UnionType
from typing import (
    Annotated,
    Callable,
    Awaitable,
    Any,
    Literal,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from TIYA.time_id import next_time_id
from TIYA.config import SETTING_CFG
from TIYA.logger import get_logger

_log = get_logger()


class ObjectToolRegistrationError(RuntimeError):
    """对象工具注册信息互相冲突或不完整。"""


class ObjectToolReturnTypeError(TypeError):
    """对象工具返回了既不能托管、也不能发送给 LLM 的值。"""


@dataclass(frozen=True, slots=True)
class ObjectReturnContract:
    origin_annotation: Any
    exposed_annotation: Any
    declared_object_classes: frozenset[type]
    allowed_runtime_classes: frozenset[type]


@dataclass(slots=True)
class AgentObjectTool:
    skill_name: str
    method_name: str
    origin_method: Callable
    wrapped_method: Callable
    return_contract: ObjectReturnContract


@dataclass(slots=True)
class ObjectProxy:
    hook_object: Any
    source_skill_name: str
    source_method_name: str
    create_time: float
    renewal_time: float
    ttl: float
    object_id: str

    def expired_at(self, now: float | None = None) -> str:
        """返回自动回收的 提示字符串，主要是 LLM 阅读友好"""
        if now is None:
            now = time.time()

        expire = self.renewal_time + self.ttl
        remaining = expire - now
        if remaining <= 0:
            return "已过期"

        elif remaining <= 60:
            return "一分钟内"

        elif remaining <= 3600:
            if remaining == 3600:
                return "1 小时"

            minutes = ceil(remaining / 60)
            return f"{minutes} 分钟"

        elif remaining <= 3600 * 6:
            if remaining == 3600 * 6:
                return "6 小时"

            total_minutes = ceil(remaining / 60)
            hours, minutes = divmod(total_minutes, 60)
            if minutes == 0:
                return f"{hours} 小时"

            return f"{hours} 小时 {minutes} 分钟"

        else:
            return datetime.fromtimestamp(expire).isoformat()

    def expired(self, now: float | None = None) -> bool:
        """判断是否过期"""
        if now is None:
            now = time.time()

        return now >= self.renewal_time + self.ttl

    def renew(self, now: float | None = None) -> None:
        """从当前时刻重新开始计算动态 TTL。"""
        self.renewal_time = time.time() if now is None else now

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
        self._skill_to_objects_mapping: dict[str, dict[str, set[str]]] = {}  # {skill_name: {method_name: {object_id}}}，这个管运行
        self._object_identity_mapping: dict[int, str] = {}  # {id(object): object_id}

    """
    框架逻辑
    """

    def register(
            self,
            skill_name: str,
            method_name: str,
            func: Callable[..., Any | Awaitable[Any]],
            declared_object_classes: set[type],
            skill_classes: set[type]
    ) -> Callable:
        """
        注册对象函数、包装函数对象
        :param skill_name: 来源 SKILL 的名字
        :param method_name: 显式传入函数名
        :param func: 函数本体，对象
        :param declared_object_classes:
        :param skill_classes:
        :return:
        """
        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
            raise ObjectToolRegistrationError("对象工具不支持生成器或异步生成器函数")

        registered = self.object_tools.get(skill_name, {}).get(method_name)
        if registered is not None:
            if registered.origin_method is func:
                return registered.wrapped_method

            raise ObjectToolRegistrationError(
                f"对象工具注册冲突: {skill_name}:{method_name}"
            )

        contract = self._build_return_contract(
            func,
            declared_object_classes,
            skill_classes,
        )

        if inspect.iscoroutinefunction(func):
            async def _async_tool_wrapper(*args, **kwargs) -> Any:
                """异步函数包装器"""
                result = await func(*args, **kwargs)
                return self._convert_result(
                    result,
                    skill_name,
                    method_name,
                    contract,
                )

            wrapped = wraps(func)(_async_tool_wrapper)

        else:
            def _sync_tool_wrapper(*args, **kwargs) -> Any:
                """同步函数包装器"""
                result = func(*args, **kwargs)
                return self._convert_result(
                    result,
                    skill_name,
                    method_name,
                    contract,
                )

            wrapped = wraps(func)(_sync_tool_wrapper)

        wrapped.__annotations__ = {
            **func.__annotations__,
            "return": contract.exposed_annotation,
        }
        wrapped.__signature__ = inspect.signature(func).replace(
            return_annotation=contract.exposed_annotation,
        )

        package = AgentObjectTool(
            skill_name=skill_name,
            method_name=method_name,
            origin_method=func,
            wrapped_method=wrapped,
            return_contract=contract,
        )
        self.object_tools.setdefault(skill_name, {})[method_name] = package
        return wrapped

    @staticmethod
    def _build_return_contract(
            func: Callable,
            declared_object_classes: set[type],
            skill_classes: set[type]
    ) -> ObjectReturnContract:
        """构造返回值注解契约，以及允许托管的具体运行时类。"""
        declared = frozenset(declared_object_classes)
        if not declared or not all(inspect.isclass(item) for item in declared):
            raise ObjectToolRegistrationError("对象工具必须声明至少一个合法对象类")

        allowed_runtime_classes = frozenset(
            candidate
            for candidate in skill_classes
            if any(
                AgentObjectToolManager._is_same_or_subclass(candidate, parent)
                for parent in declared
            )
        )
        if not allowed_runtime_classes:
            raise ObjectToolRegistrationError("对象工具没有可托管的运行时类")

        origin_annotation = get_type_hints(func, include_extras=True).get(
            "return",
            inspect.Signature.empty,
        )
        exposed_annotation = AgentObjectToolManager._replace_object_annotation(
            origin_annotation,
            declared,
        )
        return ObjectReturnContract(
            origin_annotation=origin_annotation,
            exposed_annotation=exposed_annotation,
            declared_object_classes=declared,
            allowed_runtime_classes=allowed_runtime_classes,
        )

    @staticmethod
    def _is_same_or_subclass(candidate: type, parent: type) -> bool:
        if candidate is parent:
            return True

        try:
            return issubclass(candidate, parent)

        except TypeError:
            return False

    @staticmethod
    def _replace_object_annotation(
            annotation: Any,
            object_classes: frozenset[type]
    ) -> Any:
        if any(annotation is item for item in object_classes):
            return str

        origin = get_origin(annotation)
        if origin in (UnionType, Union):
            converted = tuple(
                AgentObjectToolManager._replace_object_annotation(item, object_classes)
                for item in get_args(annotation)
            )
            return reduce(or_, converted)

        if origin is Annotated:
            base, *metadata = get_args(annotation)
            converted_base = AgentObjectToolManager._replace_object_annotation(
                base,
                object_classes,
            )
            return Annotated[converted_base, *metadata]

        return annotation

    def _convert_result(
            self,
            result: Any,
            skill_name: str,
            source_name: str,
            contract: ObjectReturnContract
    ) -> Any | str:
        """将合法对象转换成 ID；可序列化的注解偏差仅记录警告。"""
        if type(result) in contract.allowed_runtime_classes:
            return self._store_object(result, skill_name, source_name)

        annotation_matches = self._matches_annotation(
            result,
            contract.origin_annotation,
        )
        if not annotation_matches:
            _log.warning(
                f"对象工具 [{skill_name}:{source_name}] 返回值与注解不一致: "
                f"declared={contract.origin_annotation!r}, actual={type(result)!r}"
            )

        try:
            json.dumps(result, ensure_ascii=False, allow_nan=False)

        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ObjectToolReturnTypeError(
                f"对象工具 [{skill_name}:{source_name}] 返回了无法托管且无法 JSON 序列化的 "
                f"{type(result).__name__}"
            ) from exc

        return result

    def _store_object(
            self,
            result: Any,
            skill_name: str,
            source_name: str
    ) -> str:
        """托管新对象；同一个仍有效的实例复用原 ID 并续期。"""
        now = time.time()
        identity = id(result)
        known_id = self._object_identity_mapping.get(identity)
        if known_id is not None:
            known_proxy = self._hook_objects.get(known_id)
            if (
                    known_proxy is not None
                    and known_proxy.hook_object is result
                    and not known_proxy.expired(now)
            ):
                known_proxy.renew(now)
                return known_id

        object_id = f"obj:{skill_name}:{source_name}:{next_time_id()}"
        while object_id in self._hook_objects:
            object_id = f"obj:{skill_name}:{source_name}:{next_time_id()}"

        self._hook_objects[object_id] = ObjectProxy(
            hook_object=result,
            source_skill_name=skill_name,
            source_method_name=source_name,
            create_time=now,
            renewal_time=now,
            ttl=SETTING_CFG.Agent.DefaultHookObjectTTL,
            object_id=object_id,
        )
        self._object_identity_mapping[identity] = object_id
        self._skill_to_objects_mapping.setdefault(skill_name, {}).setdefault(
            source_name,
            set(),
        ).add(object_id)
        return object_id

    @staticmethod
    def _matches_annotation(value: Any, annotation: Any) -> bool:
        """宽容校验只用于诊断，不作为可序列化普通结果的拒绝条件。"""
        if annotation in (Any, inspect.Signature.empty):
            return True

        origin = get_origin(annotation)
        if origin is Annotated:
            return AgentObjectToolManager._matches_annotation(
                value,
                get_args(annotation)[0],
            )

        if origin in (UnionType, Union):
            return any(
                AgentObjectToolManager._matches_annotation(value, item)
                for item in get_args(annotation)
            )

        if origin is Literal:
            return value in get_args(annotation)

        if annotation is type(None):
            return value is None

        if origin in (list, set, frozenset):
            if type(value) is not origin:
                return False

            arguments = get_args(annotation)
            return not arguments or all(
                AgentObjectToolManager._matches_annotation(item, arguments[0])
                for item in value
            )

        if origin is tuple:
            if type(value) is not tuple:
                return False

            arguments = get_args(annotation)
            if not arguments:
                return True

            if len(arguments) == 2 and arguments[1] is Ellipsis:
                return all(
                    AgentObjectToolManager._matches_annotation(item, arguments[0])
                    for item in value
                )

            return len(value) == len(arguments) and all(
                AgentObjectToolManager._matches_annotation(item, expected)
                for item, expected in zip(value, arguments)
            )

        if origin is dict:
            if type(value) is not dict:
                return False

            arguments = get_args(annotation)
            if len(arguments) != 2:
                return True

            key_type, value_type = arguments
            return all(
                AgentObjectToolManager._matches_annotation(key, key_type)
                and AgentObjectToolManager._matches_annotation(item, value_type)
                for key, item in value.items()
            )

        if inspect.isclass(annotation):
            if annotation in {str, int, float, bool, bytes}:
                return type(value) is annotation

            return isinstance(value, annotation)

        return True

    """
    对象方法通用接口
    """

    def get_attr(self, object_id: str, attribute: str) -> Any:
        """获取指定对象的属性"""
        ...

    def set_attr(self, object_id: str, attribute: str, value: Any) -> _ObjectAttributeSetResult:
        """设置指定对象的属性"""
        ...

    def renewal_object(self, object_id: str, new_ttl: float) -> dict:
        """显式给指定对象续期，设置一个新 TTL，从当前时间开始计算回收时间"""
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
