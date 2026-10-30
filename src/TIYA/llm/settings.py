"""Convert preset data to typed Chat configuration; no YAML IO or wire payloads."""

from __future__ import annotations

from collections.abc import Mapping

from TIYA.config import ConfigMap, LLMPresetConfig
from .config import credential, endpoint, plain, resolve_preset
from .config_model import ChatConfig, FilesConfig, LLMConfig, RequestPolicy


def files_config_from_settings(
    settings: Mapping, *, provider_type="files_api", connection_name="direct"
) -> FilesConfig:
    data = plain(settings)
    known = {
        "provider_type",
        "connection_name",
        "endpoint",
        "key",
        "token",
        "headers",
        "client_params",
        "upload_params",
    }
    return FilesConfig(
        data.get("provider_type", provider_type),
        data.get("connection_name", connection_name),
        endpoint(data, data.get("provider_type", provider_type)),
        credential(data),
        headers=plain(data.get("headers") or {}),
        client_params=plain(data.get("client_params") or {}),
        upload_params=plain(data.get("upload_params") or {}),
        provider_options={k: v for k, v in data.items() if k not in known},
    )


def model_config_from_settings(
    settings: Mapping, *, provider_type: str, connection_name: str, model: str
) -> LLMConfig:
    """Bind explicit framework fields; all other connection options belong to Provider."""
    data = plain(settings)
    params = data.get("llm_params") or {}
    if not isinstance(params, dict):
        raise TypeError("llm_params 必须是字典")
    for key in ("headers", "client_params"):
        if data.get(key) is not None and not isinstance(data[key], dict):
            raise TypeError(f"{key} 必须是字典")
    limit = data.get("max_context")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("max_context 必须是正整数")
    retries = data.get("retries")
    if retries is not None and (
        isinstance(retries, bool) or not isinstance(retries, int) or retries < 0
    ):
        raise ValueError("retries 必须是非负整数")
    files = data.get("files_api_config")
    known = {
        "llm",
        "provider_type",
        "connection_name",
        "endpoint",
        "key",
        "token",
        "model",
        "llm_params",
        "headers",
        "client_params",
        "max_context",
        "context_history_mode",
        "files_api",
        "files_api_params",
        "files_api_config",
        "retries",
        "preset_name",
        "auto_compress_rate",
        "proxy_mode",
    }
    return LLMConfig(
        provider_type,
        connection_name,
        endpoint(data, provider_type),
        credential(data),
        model,
        model_params=params,
        headers=plain(data.get("headers") or {}),
        client_params=plain(data.get("client_params") or {}),
        max_context=limit,
        context_history_mode=data.get("context_history_mode", "auto"),
        files=files_config_from_settings(files) if files else None,
        provider_options={k: v for k, v in data.items() if k not in known},
        max_retries=retries,
    )


def chat_config_from_preset(preset: LLMPresetConfig) -> ChatConfig:
    """Consume a get_llm() snapshot, retaining every candidate and its order."""
    preset = resolve_preset(preset)
    candidates = [
        model_config_from_settings(
            entry,
            provider_type=entry["provider_type"],
            connection_name=entry["connection_name"],
            model=entry["model"],
        )
        for entry in preset["providers"]
    ]
    rate = preset.get("auto_compress_rate")
    if rate is not None and (
        isinstance(rate, bool) or not isinstance(rate, (int, float))
    ):
        raise TypeError("auto_compress_rate 必须是数字")
    policy = preset.get("request_policy") or {}
    if not isinstance(policy, Mapping):
        raise TypeError("request_policy 必须是字典")
    return ChatConfig(
        candidates,
        preset_name=preset["preset_name"],
        auto_compress_rate=rate,
        request_policy=RequestPolicy(
            max_attempts=policy.get("max_attempts", preset.get("max_attempts", 6)),
            max_retries_per_candidate=policy.get(
                "max_retries_per_candidate", preset.get("retries", 1)
            ),
            fallback_cooldown_seconds=policy.get(
                "fallback_cooldown_seconds",
                preset.get("fallback_cooldown_seconds", 3600),
            ),
        ),
    )


def resolve_chat_config(data: Mapping, preset_name: str) -> ChatConfig:
    """Convert an explicit raw snapshot without reading configuration globals."""
    return chat_config_from_preset(LLMPresetConfig(preset_name, ConfigMap(plain(data))))


def parse_llm_setting(preset_name: str) -> ChatConfig:
    """Historical entry point: preset lookup, conversion, then proxy-name resolution."""
    from TIYA import config

    preset = config.get_llm(preset_name)
    setting = chat_config_from_preset(preset)
    mode = preset.get("proxy_mode")
    if mode == "":
        for candidate in setting.candidates:
            candidate.client_params.pop("proxies", None)
    elif mode is not None:
        for candidate in setting.candidates:
            candidate.client_params["proxies"] = config.get_proxy(mode)
    return setting
