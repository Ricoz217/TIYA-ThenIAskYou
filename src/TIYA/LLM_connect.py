"""Stable public entry points for the LLM communication services."""

from __future__ import annotations
import httpx  # Historical monkeypatch/integration seam; the transport uses this same module.
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
from .llm.settings import parse_llm_setting
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
