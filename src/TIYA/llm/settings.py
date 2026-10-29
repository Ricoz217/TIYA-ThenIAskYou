"""Adapt configuration documents to ChatConfig without constructing wire payloads."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from .config import credential, plain, resolve_preset
from .model import ChatConfig
from .usage_compat import ApiPrice


def resolve_chat_config(data: Mapping, preset_name: str) -> ChatConfig:
    """Resolve a fresh configuration; never read globals or modify the document."""
    if not isinstance(preset_name, str):
        raise TypeError("LLM 预设名必须是字符串")
    if data.get("schema_version") == 2:
        preset = resolve_preset(data, preset_name)
        primary = preset.routes[0]
        return ChatConfig(
            endpoint=primary.endpoint,
            token=credential(primary.settings),
            model=primary.model,
            max_context=primary.max_context,
            auto_compress_rate=preset.auto_compress_rate,
            api_provider=primary.category,
            client_params=copy.deepcopy(primary.client_params),
            model_params=copy.deepcopy(primary.params),
            preset=preset,
            complete=True,
        )
    if data.get("schema_version") not in (None, 1):
        raise ValueError("不支持的 LLM 配置版本")
    return _legacy_chat_config(_legacy_entry(data, preset_name), preset_name)


def _legacy_entry(data, preset_name):
    entries = data.get("LLM_List", [])
    if not isinstance(entries, list):
        raise ValueError("旧版 LLM_List 必须是列表；字典配置需声明 schema_version: 2")
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("preset_name") == preset_name:
            return entry
    raise KeyError("LLM 预设不存在")


def _legacy_chat_config(entry, name):
    if not (entry.get("model") and entry.get("endpoint")):
        raise ValueError(f"LLM预设[{name}]未设置endpoint和模型")
    for key in ("endpoint", "token", "model", "api_type"):
        if not isinstance(entry.get(key), str):
            raise TypeError(f"LLM预设[{name}]基本参数类型错误: {key} 必须是字符串")
    for key in ("max_context", "auto_compress_rate"):
        if not isinstance(entry.get(key), (int, float)):
            raise TypeError(f"LLM预设[{name}]context相关参数类型错误: {key} 必须是数字")
    params = entry.get("extra_parameter")
    if params and not isinstance(params, Mapping):
        raise TypeError(f"LLM预设[{name}]额外参数类型错误: extra_parameter 必须是字典")
    price = entry.get("price")
    if price and not isinstance(price, Mapping):
        raise TypeError(f"LLM预设[{name}]价格参数类型错误: price 必须是字典")
    return ChatConfig(
        endpoint=entry["endpoint"],
        token=entry["token"],
        model=entry["model"],
        api_provider=entry["api_type"],
        max_context=entry["max_context"],
        auto_compress_rate=entry["auto_compress_rate"],
        client_params={},
        model_params=plain(
            params or {"max_tokens": 8192, "temperature": 0.7, "stream": True}
        ),
        price=ApiPrice(
            currency=price.get("currency", "CNY"),
            input=float(price.get("input_token", 0)),
            output=float(price.get("output_token", 0)),
            cached=float(price.get("cache_hit", 0)),
        )
        if price
        else None,
        complete=True,
    )


def parse_llm_setting(preset_name: str) -> ChatConfig | None:
    """Historical entry point; both configuration versions use the same Runtime."""
    from TIYA import config

    data = config.LLM_CFG
    setting = resolve_chat_config(data, preset_name)
    if setting.preset is not None:
        mode = setting.preset.proxy_mode
        if mode is not None:
            for route in setting.preset.routes:
                route.client_params["proxies"] = config.get_proxy(mode)
            setting.client_params = copy.deepcopy(
                setting.preset.routes[0].client_params
            )
    else:
        mode = _legacy_entry(data, preset_name).get("proxy_mode")
        if mode:
            setting.client_params["proxies"] = config.get_proxy(mode)
    return setting
