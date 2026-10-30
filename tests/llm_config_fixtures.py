"""Convenient typed configurations for protocol and recovery scenarios."""

import copy

from TIYA.llm.config import endpoint
from TIYA.llm.config_model import ChatConfig, LLMConfig, RequestPolicy
from TIYA.llm.settings import files_config_from_settings


def candidate(
    ref,
    category,
    connection,
    model,
    settings,
    params=None,
    max_context=None,
    history_mode="auto",
    files_ref=None,
    files_settings=None,
    client_params=None,
):
    settings = copy.deepcopy(settings)
    files = None
    if files_settings:
        kind, name = (files_ref or "files_api:direct").split(":")
        files = files_config_from_settings(
            files_settings, provider_type=kind, connection_name=name
        )
    return LLMConfig(
        category,
        connection,
        endpoint(settings, category)
        if settings.get("endpoint") or settings.get("base_url")
        else "",
        settings.get("key", settings.get("token", "")) or "",
        model,
        model_params=copy.deepcopy(params or {}),
        max_context=max_context,
        context_history_mode=history_mode,
        files=files,
        headers=settings.pop("headers", {}),
        client_params=copy.deepcopy(client_params or {}),
        provider_options={
            k: v for k, v in settings.items() if k not in {"endpoint", "key", "token"}
        },
    )


def configuration(
    name, candidates, auto_compress_rate=None, proxy_mode=None, options=None
):
    options = options or {}
    return ChatConfig(
        candidates,
        preset_name=name,
        auto_compress_rate=auto_compress_rate,
        request_policy=RequestPolicy(
            max_attempts=options.get("max_attempts", 6),
            max_retries_per_candidate=options.get("retries", 1),
            fallback_cooldown_seconds=options.get("fallback_cooldown_seconds", 3600),
        ),
    )
