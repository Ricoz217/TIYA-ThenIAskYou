"""Interpret LLM preset data: references, parameters and protocol connection options."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from TIYA.config import LLMPresetConfig


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


def resolve_preset(preset: LLMPresetConfig) -> dict:
    """Interpret a raw root-config snapshot without changing it or opening a connection."""
    name = preset.preset_name
    spec = plain(preset.data)
    tree = preset.provider_definitions
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
    spec.update(preset_name=name, providers=candidates)
    return spec
