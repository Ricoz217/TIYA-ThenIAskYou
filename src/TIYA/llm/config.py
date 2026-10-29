"""Pure resolution of the open Providers tree. No configuration IO or probing."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return copy.deepcopy(value)


def fingerprint(value: Any) -> str:
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


@dataclass
class Route:
    ref: str
    category: str
    connection: str
    model: str
    settings: dict
    params: dict = field(default_factory=dict)
    max_context: int | None = None
    history_mode: str = "auto"
    files_ref: str | None = None
    files_settings: dict | None = None
    client_params: dict = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return endpoint(self.settings, self.category)

    @property
    def scope(self) -> str:
        # Includes credential fingerprint; credentials themselves never become DB keys.
        location = (
            self.endpoint
            if self.settings.get("endpoint") or self.settings.get("base_url")
            else None
        )
        return fingerprint(
            [
                self.category,
                self.connection,
                self.model,
                location,
                credential(self.settings),
                self.settings.get("headers"),
                {
                    k: self.client_params.get(k)
                    for k in ("headers", "auth", "cookies", "cert")
                },
            ]
        )

    @property
    def revision(self) -> str:
        return fingerprint(
            [
                self.ref,
                self.settings,
                self.params,
                self.files_ref,
                self.files_settings,
                self.client_params,
                self.max_context,
                self.history_mode,
            ]
        )


@dataclass
class Preset:
    name: str
    routes: list[Route]
    auto_compress_rate: float | None = None
    proxy_mode: str | None = None
    options: dict = field(default_factory=dict)

    @property
    def revision(self) -> str:
        return fingerprint([self.name, [r.revision for r in self.routes], self.options])


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


def resolve_preset(data: Mapping, name: str) -> Preset:
    if data.get("schema_version") != 2:
        raise ValueError("需要 schema_version: 2")
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
    routes = []
    for index, candidate in enumerate(spec["providers"]):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("llm"), str):
            raise ValueError(f"LLM_List.{name}.providers[{index}].llm 无效")
        category, conn, model, settings = connection(tree, candidate["llm"])
        if not model:
            raise ValueError(f"模型 ID 缺失: {candidate['llm']}")
        params = plain(settings.get("llm_params") or {})
        overrides = candidate.get("llm_params") or {}
        if not isinstance(params, dict) or not isinstance(overrides, dict):
            raise ValueError(f"{name}.providers[{index}].llm_params 必须是字典")
        params.update(overrides)
        files_ref = candidate.get("files_api")
        files = None
        if files_ref:
            _, _, file_model, files = connection(tree, files_ref)
            if file_model:
                raise ValueError(f"Files 引用必须是 分类:连接名: {files_ref}")
            files.update(candidate.get("files_api_params") or {})
        settings.update(
            {
                k: v
                for k, v in candidate.items()
                if k not in {"llm", "llm_params", "files_api", "files_api_params"}
            }
        )
        history = candidate.get(
            "context_history_mode", settings.get("context_history_mode", "auto")
        )
        if history not in ("auto", "full", "previous_response_id"):
            raise ValueError(f"{name}.providers[{index}].context_history_mode 无效")
        routes.append(
            Route(
                candidate["llm"],
                category,
                conn,
                model,
                settings,
                params,
                candidate.get("max_context", settings.get("max_context")),
                history,
                files_ref,
                files,
                plain(settings.get("client_params") or {}),
            )
        )
    return Preset(
        name,
        routes,
        spec.get("auto_compress_rate"),
        spec.get("proxy_mode"),
        {k: v for k, v in spec.items() if k != "providers"},
    )


def compatibility_view(data: Mapping, name: str) -> dict:
    preset = resolve_preset(data, name)
    route = preset.routes[0]
    return {
        "preset_name": name,
        "model": route.model,
        "endpoint": route.endpoint,
        "token": credential(route.settings),
        "api_type": route.category,
        "max_context": route.max_context,
        "auto_compress_rate": preset.auto_compress_rate,
        "proxy_mode": preset.proxy_mode,
        "extra_parameter": plain(route.params),
        "price": {},
    }


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
                    "llm_params": plain(old.get("extra_parameter") or {}),
                    "max_context": old.get("max_context"),
                    "context_history_mode": "full",
                }
            ],
        }
        # Kept only for the explicit price import, never sent as model parameters.
        if old.get("price"):
            result["LLM_List"][name]["legacy_price"] = plain(old["price"])
    return result
