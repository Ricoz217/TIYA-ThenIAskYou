"""Stable public entry points for the LLM communication services."""

from __future__ import annotations
import copy
import httpx  # Historical monkeypatch/integration seam; the transport uses this same module.
from TIYA.config import get_llm, get_proxy
from TIYA.utils import parse_proxies_to_httpx
from TIYA.LLM_usage import ApiPrice, LLMUsage, _GLOBAL_USAGE
from .llm.model import (
    ContextOverflowError,
    ContextEmptyError,
    PromptType,
    BasePrompt,
    SystemPrompt,
    TextPrompt,
    ImagePrompt,
    ToolInput,
    ToolCall,
    ToolResponse,
    TokenUsage,
    Prompts,
    Context,
    AskTask,
    ChatConfig,
    MediaRef,
    ImageGenerateRequest,
    ImageEditRequest,
    ImageResponseItem,
    ImagesResponse,
    CallOutcome,
)
from .llm.chat import Chat
from .llm.clients import ImagesClient, FilesClient
from .llm.tool_schema import build_tool_payloads
from .llm.legacy_media import (
    _IMAGE_NAME_MAPPING,
    _load_image_hashes,
    handle_unload_image,
)

__version__ = "0.2.0"
__all__ = [
    "Chat",
    "ChatConfig",
    "Context",
    "Prompts",
    "BasePrompt",
    "SystemPrompt",
    "TextPrompt",
    "ImagePrompt",
    "ToolInput",
    "ToolCall",
    "ToolResponse",
    "TokenUsage",
    "AskTask",
    "PromptType",
    "ContextOverflowError",
    "ContextEmptyError",
    "CallOutcome",
    "MediaRef",
    "ImageGenerateRequest",
    "ImageEditRequest",
    "ImageResponseItem",
    "ImagesResponse",
    "ImagesClient",
    "FilesClient",
    "build_tool_payloads",
    "parse_llm_setting",
    "handle_unload_image",
    "ApiPrice",
    "LLMUsage",
    "httpx",
    "parse_proxies_to_httpx",
    "_GLOBAL_USAGE",
    "_IMAGE_NAME_MAPPING",
    "_load_image_hashes",
]


def parse_llm_setting(preset_name: str) -> ChatConfig | None:
    new_setting = ChatConfig(
        client_params={},
        model_params={"max_tokens": 8192, "temperature": 0.7, "stream": True},
    )
    llm = get_llm(preset_name)
    if llm.get("_resolved_preset") is not None:
        preset = copy.deepcopy(llm["_resolved_preset"])
        route = preset.routes[0]
        for candidate in preset.routes:
            if preset.proxy_mode is not None:
                candidate.client_params["proxies"] = get_proxy(preset.proxy_mode)
        return ChatConfig(
            endpoint=route.endpoint,
            token=llm.token,
            model=route.model,
            max_context=route.max_context,
            auto_compress_rate=preset.auto_compress_rate,
            api_provider=route.category,
            client_params=copy.deepcopy(route.client_params),
            model_params=copy.deepcopy(route.params),
            preset=preset,
            complete=True,
        )
    if not (llm.model and llm.endpoint):
        raise ValueError(f"LLM预设[{preset_name}]未设置endpoint和模型")

    if llm.proxy_mode:
        proxy = get_proxy(llm.proxy_mode)
        new_setting.client_params = {"proxies": proxy}

    if llm.price:
        if not isinstance(llm.price, dict):
            raise TypeError(
                f"LLM预设[{preset_name}]价格参数类型错误:\n"
                f"Key <price> should be <dict>, got {type(llm.price)}"
            )

        mirror = llm.price.copy()
        if mirror:
            new_price = ApiPrice(
                currency=mirror.get("currency", "CNY"),
                input=float(mirror.get("input_token", 0.0)),
                output=float(mirror.get("output_token", 0.0)),
                cached=float(mirror.get("cache_hit", 0.0)),
            )
            new_setting.price = new_price

    if llm.extra_parameter:
        if not isinstance(llm.extra_parameter, dict):
            raise TypeError(
                f"LLM预设[{preset_name}]额外参数类型错误:\n"
                f"Key <extra_parameter> should be <dict>, got {type(llm.extra_parameter)}"
            )

        new_setting.model_params = llm.extra_parameter.copy()

    for key in ("endpoint", "token", "model", "api_type"):
        if not isinstance(llm[key], str):
            raise TypeError(
                f"LLM预设[{preset_name}]基本参数类型错误:\n"
                f"Key <{key}> should be <str>, got {type(llm[key])}"
            )

    for key in ("max_context", "auto_compress_rate"):
        if not isinstance(llm[key], (int, float)):
            raise TypeError(
                f"LLM预设[{preset_name}]context相关参数类型错误:\n"
                f"Key <{key}> should be <int | float>, got {type(llm[key])}"
            )

    new_setting.endpoint = llm.endpoint
    new_setting.token = llm.token
    new_setting.model = llm.model
    new_setting.api_provider = llm.api_type
    new_setting.max_context = llm.max_context
    new_setting.auto_compress_rate = llm.auto_compress_rate
    new_setting.complete = True

    return new_setting
