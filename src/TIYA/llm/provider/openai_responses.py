from __future__ import annotations

import copy
import json
from urllib.parse import quote
from ..gateway import WireRequest

from ..model import (
    SystemPrompt,
    TextPrompt,
    ImagePrompt,
    ToolCall,
    ToolInput,
    ToolResponse,
)
from ..tool_schema import build_tool_payloads
from .base import Provider, ProviderFailure, Result
from .common import image_url, tool_arguments, native_items, attach_native


class Responses(Provider):
    def background_operation(self, response_id, cancel=False):
        url = self.route.endpoint.rstrip("/") + "/" + quote(response_id, safe="")
        if cancel:
            url += "/cancel"
        return WireRequest(
            url,
            "POST" if cancel else "GET",
            self.headers(),
            json={} if cancel else None,
        )

    def accepts_image_files(self):
        return True

    def encode(self, context, media):
        items, tools, covered, instructions = [], [], set(), None
        for prompt in context:
            if prompt.id in covered:
                continue
            native = native_items(prompt, "responses_api", self.route.scope, context)
            if native:
                items.extend(copy.deepcopy(native["items"]))
                covered.update(native["projection_ids"])
            elif isinstance(prompt, ToolInput):
                function = build_tool_payloads(prompt.function)["openai"]["function"]
                function.pop("returns", None)
                tools.append(
                    {
                        "type": "function",
                        **function,
                        "name": prompt.function_name,
                        "strict": False,
                    }
                )
            elif isinstance(prompt, ToolCall):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": prompt.call_id,
                        "name": prompt.function_name,
                        "arguments": json.dumps(prompt.arguments, ensure_ascii=False),
                    }
                )
            elif isinstance(prompt, ToolResponse):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": prompt.call_id,
                        "output": json.dumps(prompt.response, ensure_ascii=False),
                    }
                )
            elif isinstance(prompt, SystemPrompt):
                instructions = prompt.text
            elif isinstance(prompt, TextPrompt):
                items.append(
                    {
                        "role": prompt.role,
                        "content": [
                            {
                                "type": "output_text"
                                if prompt.role == "assistant"
                                else "input_text",
                                "text": prompt.text,
                            }
                        ],
                    }
                )
            elif isinstance(prompt, ImagePrompt):
                content = [
                    {
                        "type": "input_image",
                        **(
                            {"file_id": ref.value}
                            if ref.kind == "remote"
                            else {"image_url": image_url(ref)}
                        ),
                    }
                    for ref in media[prompt.id]
                ]
                items.append({"role": prompt.role, "content": content})
        payload = {
            **copy.deepcopy(self.route.model_params),
            "model": self.route.model,
            "input": items,
        }
        if instructions is not None:
            payload["instructions"] = instructions
        if self.supports_parent and self.route.provider_options.get(
            "responses", {}
        ).get("include_encrypted_reasoning", True):
            includes = payload.setdefault("include", [])
            if "reasoning.encrypted_content" not in includes:
                includes.append("reasoning.encrypted_content")
        # Old callers (Memory/summary) can set response_format; translate only this compatibility field.
        if "response_format" in payload:
            payload.setdefault("text", {})["format"] = payload.pop("response_format")
        if tools:
            payload["tools"] = tools
        return payload

    def decode(self, body):
        self.check_body(body)
        if not isinstance(body.get("output"), list):
            raise ValueError("Responses 响应缺少 output")
        result = Result(
            response_id=body.get("id", ""),
            native=copy.deepcopy(body.get("output", [])),
            reported_model=body.get("model", ""),
            stored=body.get("store"),
            complete=body.get("status", "completed") == "completed",
        )
        for item in result.native:
            if item.get("type") == "message":
                content = item.get("content", [])
                if isinstance(content, dict):
                    content = [content]
                for block in content:
                    if block.get("type") in ("output_text", "text"):
                        result.prompts.append(
                            TextPrompt("assistant", block.get("text", ""))
                        )
                    elif block.get("type") == "refusal":
                        result.prompts.append(
                            TextPrompt("assistant", block.get("refusal", ""))
                        )
            elif item.get("type") == "function_call":
                result.prompts.append(
                    ToolCall(
                        item["name"], item["call_id"], tool_arguments(item["arguments"])
                    )
                )
        if result.native and not result.prompts:
            # Preserve opaque reasoning even when the result has no business text.
            result.prompts.append(TextPrompt("assistant", ""))
        usage = body.get("usage") or {}
        result.usage = {
            target: usage[source]
            for source, target in (
                ("input_tokens", "input"),
                ("output_tokens", "output"),
                ("total_tokens", "total"),
            )
            if usage.get(source) is not None
        }
        if usage.get("input_tokens_details", {}).get("cached_tokens") is not None:
            result.usage["cached"] = usage["input_tokens_details"]["cached_tokens"]
        result.raw_usage = usage
        attach_native(result, "responses_api", self.route.scope)
        return result

    def stream_decoder(self):
        return ResponsesStream(self)

    def usage_result(self, raw):
        return self.decode({"output": [], "usage": raw})


class ResponsesStream:
    def __init__(self, provider):
        self.provider = provider
        self.partial, self.complete = False, False
        self.body = {"output": []}
        self.items = {}
        self.usage = {}

    def feed(self, event):
        self.provider.check_body(event)
        kind = event.get("type", "")
        if kind in ("response.failed", "error"):
            body = event.get("response", event)
            raise ProviderFailure(self.provider.failure(200, body))
        response = event.get("response")
        if response:
            self.body.update(response)
            self.usage.update(response.get("usage") or {})
        if kind == "response.output_item.added":
            item = event["item"]
            if item.get("type") in ("function_call", "reasoning", "message"):
                self.partial = True
        if kind.endswith(".delta") or "partial_image" in kind:
            self.partial = True
        if kind == "response.output_item.done":
            self.partial = True
            self.items[event.get("output_index", len(self.items))] = event["item"]
        if kind == "response.completed" or "output" in event and "type" not in event:
            if not response:
                self.body.update(event)
                self.usage.update(event.get("usage") or {})
            self.complete = True

    def finish(self):
        if not self.body.get("output"):
            self.body["output"] = [self.items[k] for k in sorted(self.items)]
        self.body["usage"] = self.usage
        result = self.provider.decode(self.body)
        result.complete = result.complete and self.complete
        return result
