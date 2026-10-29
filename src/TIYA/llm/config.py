"""Pure resolution of the open Providers tree. No configuration IO or probing."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping


def plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return copy.deepcopy(value)


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            plain(value), sort_keys=True, ensure_ascii=False, default=str
        ).encode()
    ).hexdigest()


PATHS = {
    "chat_completions": "chat/completions",
    "responses_api": "responses",
    "anthropic_api": "messages",
    "files_api": "files",
    "image_generation": "images/generations",
}


def credential(settings: Mapping) -> str:
    value = settings.get("key") if "key" in settings else settings.get("token")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Provider key/token 必须是字符串")
    return value


def endpoint(settings: Mapping, category: str) -> str:
    if settings.get("endpoint"):
        return str(settings["endpoint"])
    if settings.get("base_url"):
        return str(settings["base_url"]).rstrip("/") + "/" + PATHS[category]
    raise ValueError(f"Providers.{category}: 需要 endpoint 或 base_url")


def connection(tree: Mapping, ref: str) -> tuple[str, str, str, dict]:
    parts = ref.split(":", 2)
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError(f"Provider 引用格式错误: {ref}")
    category, name = parts[:2]
    try:
        settings = tree[category][name]
    except (KeyError, TypeError):
        raise ValueError(f"Provider 引用不存在: {category}:{name}") from None
    if not isinstance(settings, Mapping):
        raise ValueError(f"Provider 连接必须是字典: {category}:{name}")
    return category, name, parts[2] if len(parts) == 3 else "", plain(settings)


def resolve_preset(data: Mapping, name: str) -> dict:
    """Read a complete preset snapshot, including bound connection data.

    This returns configuration data, never ChatConfig or live services. Original
    provider references and open fields remain visible. Derived scalar fields at
    the preset root provide compatibility reads of its first candidate.
    """
    if not isinstance(name, str):
        raise TypeError("LLM 预设名必须是字符串")
    version = data.get("schema_version")
    if version in (None, 1):
        entries = data.get("LLM_List", [])
        if not isinstance(entries, list):
            raise ValueError(
                "旧版 LLM_List 必须是列表；字典配置需声明 schema_version: 2"
            )
        old = next(
            (
                plain(e)
                for e in entries
                if isinstance(e, Mapping) and e.get("preset_name") == name
            ),
            None,
        )
        if old is None:
            raise KeyError("LLM 预设不存在")
        category = {"openai": "chat_completions", "anthropic": "anthropic_api"}.get(
            old.get("api_type", "openai")
        )
        if category is None:
            raise ValueError(f"不支持的 api_type: {old.get('api_type')}")
        params = old.get("extra_parameter") or {
            "max_tokens": 8192,
            "temperature": 0.7,
            "stream": True,
        }
        candidate = dict(
            old,
            llm=f"{category}:legacy:{old.get('model', '')}",
            provider_type=category,
            connection_name="legacy",
            llm_params=params,
            context_history_mode="full",
            files_api_config=None,
        )
        return dict(old, providers=[candidate], preset_name=name)
    if version != 2:
        raise ValueError("不支持的 LLM 配置版本")
    presets, tree = data.get("LLM_List"), data.get("Providers")
    if not isinstance(presets, Mapping) or not isinstance(tree, Mapping):
        raise ValueError("v2 的 LLM_List 和 Providers 必须是字典")
    if name not in presets:
        raise KeyError(f"LLM 预设不存在: {name}")
    spec = plain(presets[name])
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("providers"), list)
        or not spec["providers"]
    ):
        raise ValueError(f"LLM_List.{name}.providers 必须是非空列表")
    candidates = []
    for index, candidate in enumerate(spec["providers"]):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("llm"), str):
            raise ValueError(f"LLM_List.{name}.providers[{index}].llm 无效")
        category, conn, model, settings = connection(tree, candidate["llm"])
        if not model:
            raise ValueError(f"模型 ID 缺失: {candidate['llm']}")
        params = plain(settings.get("llm_params", {}))
        overrides = candidate.get("llm_params", {})
        params = {} if params is None else params
        overrides = {} if overrides is None else overrides
        if not isinstance(params, dict) or not isinstance(overrides, dict):
            raise ValueError(f"{name}.providers[{index}].llm_params 必须是字典")
        params.update(overrides)
        settings.update({k: v for k, v in candidate.items() if k != "llm_params"})
        history = settings.get("context_history_mode", "auto")
        if history not in ("auto", "full", "previous_response_id"):
            raise ValueError(f"{name}.providers[{index}].context_history_mode 无效")
        files = None
        if candidate.get("files_api"):
            file_category, file_name, file_model, files = connection(
                tree, candidate["files_api"]
            )
            if file_model:
                raise ValueError(
                    f"Files 引用必须是 分类:连接名: {candidate['files_api']}"
                )
            files.update(candidate.get("files_api_params") or {})
            files.update(
                provider_type=file_category,
                connection_name=file_name,
                endpoint=endpoint(files, file_category),
            )
        settings.update(
            provider_type=category,
            connection_name=conn,
            model=model,
            endpoint=endpoint(settings, category),
            token=credential(settings),
            llm_params=params,
            context_history_mode=history,
            files_api_config=files,
        )
        candidates.append(settings)
    primary = candidates[0]
    spec.update(
        preset_name=name,
        providers=candidates,
        model=primary["model"],
        endpoint=primary["endpoint"],
        token=primary["token"],
        api_type=primary["provider_type"],
        max_context=primary.get("max_context"),
        extra_parameter=plain(primary["llm_params"]),
        price={},
    )
    return spec


def migrate_legacy(data: Mapping) -> dict:
    """Return a new document. Writing/backups are the caller's explicit operation."""
    if data.get("schema_version") == 2:
        return plain(data)
    if data.get("schema_version") not in (None, 1):
        raise ValueError("不支持的 LLM 配置版本")
    result = {k: plain(v) for k, v in data.items() if k != "LLM_List"}
    result.update(schema_version=2, Providers={}, LLM_List={})
    for index, old in enumerate(data.get("LLM_List", [])):
        name = old["preset_name"]
        if name in result["LLM_List"]:
            raise ValueError(f"重复预设名: {name}")
        category = {"openai": "chat_completions", "anthropic": "anthropic_api"}.get(
            old.get("api_type", "openai")
        )
        if category is None:
            raise ValueError(f"不支持迁移的 api_type: {old.get('api_type')}")
        conn = f"legacy_{index}"
        result["Providers"].setdefault(category, {})[conn] = {
            "endpoint": old.get("endpoint", ""),
            "key": old.get("token", ""),
        }
        result["LLM_List"][name] = {
            "auto_compress_rate": old.get("auto_compress_rate"),
            "proxy_mode": old.get("proxy_mode"),
            "providers": [
                {
                    "llm": f"{category}:{conn}:{old['model']}",
                    "llm_params": plain(
                        old.get("extra_parameter")
                        or {"max_tokens": 8192, "temperature": 0.7, "stream": True}
                    ),
                    "max_context": old.get("max_context"),
                    "context_history_mode": "full",
                }
            ],
        }
        # Kept only for the explicit price import, never sent as model parameters.
        if old.get("price"):
            result["LLM_List"][name]["legacy_price"] = plain(old["price"])
    return result
