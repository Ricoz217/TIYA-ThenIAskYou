"""Pure fingerprints and explicitly labelled local estimates; no media IO."""

from __future__ import annotations
import hashlib
import json
from .config import fingerprint
from .model import (
    ImagePrompt,
    TextPrompt,
    SystemPrompt,
    ToolCall,
    ToolResponse,
    ToolInput,
)
from .tool_schema import build_tool_payloads


def stable(value):
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [stable(v) for v in value]
    if callable(value):
        from .tool_schema import build_tool_payloads

        return build_tool_payloads(value)
    if hasattr(value, "__dict__"):
        return stable(vars(value))
    return value


def prefix(context):
    return [fingerprint(stable(vars(p))) for p in context._messages]


def context_signature(context):
    return fingerprint([prefix(context), [stable(vars(t)) for t in context._tools]])


def estimate(context):
    # Explicit lower bound, never pretend historical context length is a cache hit.
    import tiktoken

    tokenizer = tiktoken.get_encoding("o200k_base")
    total = 0
    for prompt in context:
        if isinstance(prompt, ImagePrompt):
            total += 1440 * max(1, len(prompt.image))
        else:
            if isinstance(prompt, (TextPrompt, SystemPrompt)):
                text = prompt.text + getattr(prompt, "reasoning_content", "")
            elif isinstance(prompt, ToolCall):
                text = json.dumps(
                    [prompt.function_name, prompt.arguments],
                    ensure_ascii=False,
                    default=str,
                )
            elif isinstance(prompt, ToolResponse):
                text = json.dumps(prompt.response, ensure_ascii=False, default=str)
            elif isinstance(prompt, ToolInput):
                text = json.dumps(
                    build_tool_payloads(prompt.function)["openai"],
                    ensure_ascii=False,
                    default=str,
                )
            else:
                text = json.dumps(
                    {
                        k: v
                        for k, v in vars(prompt).items()
                        if k not in ("native", "id", "time")
                    },
                    default=str,
                )
            total += len(tokenizer.encode(text)) + 3
    return {
        "input": total,
        "cached": 0,
        "algorithm": "o200k-image1440-v1",
        "completeness": "lower_bound",
        "assumptions": "images=1440 each; remote output and cached input unknown",
    }
