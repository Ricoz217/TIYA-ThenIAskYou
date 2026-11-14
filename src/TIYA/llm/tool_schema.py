from __future__ import annotations
import inspect
import types
from enum import Enum
from collections.abc import Callable
from typing import Any, Annotated, Literal, NotRequired, TypedDict, Union, get_args, get_origin, get_type_hints

from .config_model import Params

_UNION_TYPE = getattr(types, "UnionType", None)


class ToolFunction(TypedDict):
    """由 callable 生成的固定工具描述；parameters/returns 为开放 JSON Schema。"""

    name: str
    description: str
    parameters: Params
    returns: NotRequired[Params]


class OpenAITool(TypedDict):
    type: str
    function: ToolFunction


class AnthropicTool(TypedDict):
    name: str
    description: str
    input_schema: Params
    returns: NotRequired[Params]


class ToolPayloads(TypedDict):
    """同一 callable 的两种协议 schema；持有者缓存后按只读数据使用。"""

    openai: OpenAITool
    anthropic: AnthropicTool


def build_tool_payloads(func: Callable) -> ToolPayloads:
    """Build OpenAI/Anthropic tool payloads from a Python callable.

    Works for regular functions and ``async def`` coroutine functions. For tool
    schema generation, only input parameters are used.
    """
    if not callable(func):
        raise TypeError("func must be callable")

    def _annotation_to_schema(_annotation: Any) -> dict[str, Any]:
        origin = get_origin(_annotation)

        if _annotation is Any:
            return {}

        if _annotation is type(None):
            return {"type": "null"}

        if origin is not None:
            if origin is Annotated:
                base, *meta = get_args(_annotation)
                _schema = _annotation_to_schema(base)
                for item in meta:
                    if isinstance(item, str) and item.strip():
                        _schema = {**_schema, "description": item.strip()}
                        break
                return _schema

            if (_UNION_TYPE is not None and origin is _UNION_TYPE) or origin is Union:
                return _union_schema(get_args(_annotation))

            if origin in (list, tuple, set, frozenset):
                args = get_args(_annotation)
                item_schema = _annotation_to_schema(args[0]) if args else {}
                return {"type": "array", "items": item_schema}

            if origin is dict:
                args = get_args(_annotation)
                value_schema = _annotation_to_schema(args[1]) if len(args) == 2 else {}
                return {"type": "object", "additionalProperties": value_schema}

            if origin is Literal:
                values = list(get_args(_annotation))
                if not values:
                    return {}
                literal_types = {type(v) for v in values}
                if len(literal_types) == 1:
                    mapped = _python_type_to_json_type(next(iter(literal_types)))
                    return {"type": mapped, "enum": values}
                return {"enum": values}

        if inspect.isclass(_annotation) and issubclass(_annotation, Enum):
            values = [member.value for member in _annotation]
            enum_type = (
                _python_type_to_json_type(type(values[0])) if values else "string"
            )
            return {"type": enum_type, "enum": values}

        mapped = _python_type_to_json_type(_annotation)
        if mapped:
            return {"type": mapped}

        return {}

    def _union_schema(_args: tuple[Any, ...]) -> dict[str, Any]:
        _schema = [_annotation_to_schema(arg) for arg in _args]

        # Compact Optional[X] into type union when possible.
        if len(_schema) == 2 and any(s.get("type") == "null" for s in _schema):
            non_null = next((s for s in _schema if s.get("type") != "null"), None)
            if non_null and isinstance(non_null.get("type"), str):
                merged = dict(non_null)
                merged["type"] = [non_null["type"], "null"]
                if "enum" in merged and None not in merged["enum"]:
                    merged["enum"] = [*merged["enum"], None]

                return merged

        return {"anyOf": _schema}

    def _python_type_to_json_type(_annotation: Any) -> str | None:
        mapping: dict[Any, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            dict: "object",
            list: "array",
            tuple: "array",
            set: "array",
            frozenset: "array",
            type(None): "null",
        }
        return mapping.get(_annotation)

    def _json_safe_default(_value: Any) -> Any:
        if isinstance(_value, Enum):
            return _value.value
        return _value

    def _return_annotation_to_schema(_annotation: Any) -> dict[str, Any] | None:
        if _annotation in (inspect.Signature.empty, Any):
            return None
        _schema = _annotation_to_schema(_annotation)
        return _schema or None

    def _parse_doc_sections(_doc: str) -> tuple[str, dict[str, str], str]:
        """
        Parse reST-style docstring blocks and split into:
        - global description text
        - per-parameter descriptions from :param
        - return description from :return / :returns
        """
        general_parts: list[str] = []
        param_map: dict[str, list[str]] = {}
        return_parts: list[str] = []

        lines = _doc.splitlines()
        idx = 0
        current_param: str | None = None
        in_return_block = False
        while idx < len(lines):
            stripped = lines[idx].strip()
            if not stripped:
                idx += 1
                continue

            if stripped.startswith(":param "):
                current_param = None
                in_return_block = False

                body = stripped[len(":param ") :]
                if ":" in body:
                    left, first_desc = body.split(":", 1)
                    left = left.strip()
                    param_name = left.split()[-1] if left else ""
                    if param_name:
                        param_map.setdefault(param_name, [])
                        if first_desc.strip():
                            param_map[param_name].append(first_desc.strip())
                        current_param = param_name

                idx += 1
                continue

            if stripped.startswith(":return:") or stripped.startswith(":returns:"):
                current_param = None
                in_return_block = True
                first_desc = stripped.split(":", 2)[-1].strip()
                if first_desc:
                    return_parts.append(first_desc)
                idx += 1
                continue

            if stripped.startswith(":rtype:"):
                current_param = None
                in_return_block = False
                idx += 1
                continue

            # New directive starts; end current param/return capture.
            if stripped.startswith(":") and ":" in stripped[1:]:
                current_param = None
                in_return_block = False
                idx += 1
                continue

            if current_param is not None:
                param_map[current_param].append(stripped)
            elif in_return_block:
                return_parts.append(stripped)
            else:
                general_parts.append(stripped)

            idx += 1

        param_descriptions = {
            key: " ".join(parts).strip() for key, parts in param_map.items() if parts
        }
        return_description = " ".join(return_parts).strip()
        general_description = " ".join(general_parts).strip()
        return general_description, param_descriptions, return_description

    def _schema_to_text(_schema: dict[str, Any]) -> str:
        if not _schema:
            return "any"

        if "anyOf" in _schema:
            return " | ".join(_schema_to_text(part) for part in _schema["anyOf"])

        value_type = _schema.get("type")
        if isinstance(value_type, list):
            return " | ".join(str(item) for item in value_type)
        if value_type == "array":
            return f"array<{_schema_to_text(_schema.get('items', {}))}>"
        if value_type == "object" and "additionalProperties" in _schema:
            return f"object<string, {_schema_to_text(_schema['additionalProperties'])}>"
        if "enum" in _schema:
            return f"{value_type or 'enum'} ({', '.join(map(str, _schema['enum']))})"
        if isinstance(value_type, str):
            return value_type
        return "any"

    # This works for both regular functions and async coroutine functions.
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)

    doc = inspect.getdoc(func) or ""
    doc_description, doc_param_descriptions, doc_return_description = (
        _parse_doc_sections(doc)
    )

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError("Tool schema generation does not support *args or **kwargs")

        annotation = hints.get(name, Any)
        schema = _annotation_to_schema(annotation)
        doc_param_desc = doc_param_descriptions.get(name, "")
        if doc_param_desc:
            existing_desc = (
                schema.get("description", "").strip()
                if isinstance(schema.get("description"), str)
                else ""
            )
            if existing_desc:
                schema["description"] = f"{existing_desc} {doc_param_desc}"
            else:
                schema["description"] = doc_param_desc

        if param.default is inspect.Signature.empty:
            required.append(name)
        else:
            schema["default"] = _json_safe_default(param.default)

        properties[name] = schema

    description = (
        doc_description if doc_description else f"Tool generated from `{func.__name__}`"
    )
    return_annotation = hints.get("return", signature.return_annotation)
    return_schema = _return_annotation_to_schema(return_annotation)
    return_summary = ""
    if return_schema:
        return_summary = _schema_to_text(return_schema)
    if doc_return_description:
        if return_summary:
            return_summary = f"{return_summary}. {doc_return_description}"
        else:
            return_summary = doc_return_description
    if return_summary:
        description = f"{description} Returns: {return_summary}."

    params_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        params_schema["required"] = required

    openai: OpenAITool = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": params_schema,
        },
    }
    if return_schema:
        openai["function"]["returns"] = return_schema

    anthropic: AnthropicTool = {
        "name": func.__name__,
        "description": description,
        "input_schema": params_schema,
    }
    if return_schema:
        anthropic["returns"] = return_schema

    return {"openai": openai, "anthropic": anthropic}
