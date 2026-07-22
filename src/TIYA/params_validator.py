from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Codex"

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Annotated, Literal, Union, get_args, get_origin, get_type_hints


# ============================================================================
# 返回结构
# ============================================================================

@dataclass(slots=True)
class ParamValidationIssue:
    path: str
    message: str

    def to_dict(self):
        return {
            "parameter": self.path,
            "message": self.message
        }


@dataclass(slots=True)
class ParamValidationResult:
    ok: bool
    normalized_args: dict[str, Any] = field(default_factory=dict)
    issues: list[ParamValidationIssue] = field(default_factory=list)


# ============================================================================
# 工具函数：类型校验核心
# ============================================================================

_UNION_TYPE = getattr(__import__("types"), "UnionType", None)


def _is_instance_of_enum_value(value: Any, enum_cls: type[Enum]) -> bool:
    return any(value == member.value for member in enum_cls)


def _check_type(value: Any, annotation: Any, path: str, issues: list[ParamValidationIssue]) -> bool:
    """
    递归检查 value 是否符合 annotation。
    支持：
    - Any / None
    - Annotated
    - Union / Optional
    - Literal
    - Enum
    - list[T] / tuple[T,...] / dict[K,V]
    - 基础类型（str/int/float/bool）
    """
    if annotation is Any:
        return True

    if annotation is inspect.Signature.empty:
        return True

    origin = get_origin(annotation)

    # Annotated[T, ...] -> 只校验 T
    if origin is Annotated:
        base, *_ = get_args(annotation)
        return _check_type(value, base, path, issues)

    # NoneType
    if annotation is type(None):
        if value is None:
            return True
        issues.append(ParamValidationIssue(path, f"期望 None，实际是 {type(value).__name__}"))
        return False

    # Literal[...] 枚举值校验
    if origin is Literal:
        literal_values = get_args(annotation)
        if value in literal_values:
            return True
        issues.append(ParamValidationIssue(path, f"不在允许枚举内，允许值: {list(literal_values)}，实际: {value!r}"))
        return False

    # Union / Optional
    if (_UNION_TYPE is not None and origin is _UNION_TYPE) or origin is Union:
        for sub_ann in get_args(annotation):
            sub_issues: list[ParamValidationIssue] = []
            if _check_type(value, sub_ann, path, sub_issues):
                return True
        issues.append(ParamValidationIssue(path, f"不匹配任一 Union 类型: {annotation}"))
        return False

    # Enum（允许传 member.value）
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        if isinstance(value, annotation) or _is_instance_of_enum_value(value, annotation):
            return True
        enum_values = [m.value for m in annotation]
        issues.append(ParamValidationIssue(path, f"不在 Enum 允许值内: {enum_values}，实际: {value!r}"))
        return False

    # list[T], set[T], tuple[T], dict[K,V]
    if origin in (list, set, frozenset):
        if not isinstance(value, origin):
            issues.append(ParamValidationIssue(path, f"期望 {origin.__name__}，实际 {type(value).__name__}"))
            return False
        args = get_args(annotation)
        if args:
            item_ann = args[0]
            ok = True
            for i, item in enumerate(value):
                if not _check_type(item, item_ann, f"{path}[{i}]", issues):
                    ok = False
            return ok
        return True

    if origin is tuple:
        if not isinstance(value, tuple):
            issues.append(ParamValidationIssue(path, f"期望 tuple，实际 {type(value).__name__}"))
            return False
        args = get_args(annotation)
        if not args:
            return True

        # tuple[T, ...]
        if len(args) == 2 and args[1] is Ellipsis:
            item_ann = args[0]
            ok = True
            for i, item in enumerate(value):
                if not _check_type(item, item_ann, f"{path}[{i}]", issues):
                    ok = False
            return ok

        # tuple[T1, T2, ...]
        if len(value) != len(args):
            issues.append(ParamValidationIssue(path, f"tuple 长度不匹配，期望 {len(args)}，实际 {len(value)}"))
            return False
        ok = True
        for i, (item, item_ann) in enumerate(zip(value, args)):
            if not _check_type(item, item_ann, f"{path}[{i}]", issues):
                ok = False
        return ok

    if origin is dict:
        if not isinstance(value, dict):
            issues.append(ParamValidationIssue(path, f"期望 dict，实际 {type(value).__name__}"))
            return False
        args = get_args(annotation)
        if len(args) == 2:
            key_ann, val_ann = args
            ok = True
            for k, v in value.items():
                if not _check_type(k, key_ann, f"{path}.<key>", issues):
                    ok = False
                if not _check_type(v, val_ann, f"{path}[{k!r}]", issues):
                    ok = False
            return ok
        return True

    # 基础类型
    if inspect.isclass(annotation):
        if isinstance(value, annotation):
            return True
        issues.append(ParamValidationIssue(path, f"类型不匹配，期望 {annotation.__name__}，实际 {type(value).__name__}"))
        return False

    # 兜底：无法识别的 annotation，不强校验
    return True


# ============================================================================
# 对外入口：校验工具参数
# ============================================================================

def validate_tool_arguments(func: Any, parsed_args: dict[str, Any]) -> ParamValidationResult:
    """
    输入：
    - func: 带签名与类型注解的函数
    - parsed_args: LLM 解析出的参数（dict）

    输出：
    - ParamValidationResult
    """
    issues: list[ParamValidationIssue] = []

    if not callable(func):
        return ParamValidationResult(
            ok=False,
            issues=[ParamValidationIssue("func", "func 必须是可调用对象")],
        )

    if not isinstance(parsed_args, dict):
        return ParamValidationResult(
            ok=False,
            issues=[ParamValidationIssue("args", "parsed_args 必须是 dict")],
        )

    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)

    # 1) 禁止 *args/**kwargs 的函数签名（与你当前 schema 生成规则一致）
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            issues.append(ParamValidationIssue(p.name, "不支持 *args / **kwargs 作为 tool 参数"))

    # 2) 检查多余字段
    param_names = set(sig.parameters.keys())
    extra_keys = [k for k in parsed_args.keys() if k not in param_names]
    for k in extra_keys:
        issues.append(ParamValidationIssue(k, "存在未声明参数（additional property）"))

    # 3) 检查缺失必填 + 类型/枚举合法性
    normalized: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        has_value = name in parsed_args

        if not has_value:
            if p.default is inspect.Signature.empty:
                issues.append(ParamValidationIssue(name, "缺少必填参数"))
            else:
                normalized[name] = p.default
            continue

        value = parsed_args[name]
        annotation = hints.get(name, Any)
        if _check_type(value, annotation, name, issues):
            normalized[name] = value

    return ParamValidationResult(ok=(len(issues) == 0), normalized_args=normalized, issues=issues)