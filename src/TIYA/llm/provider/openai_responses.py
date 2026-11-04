from __future__ import annotations

import copy
import json
from urllib.parse import quote

from ..config_model import Params
from ..gateway import WireRequest
from ..model import (
    Context,
    ImagePrompt,
    MediaRef,
    SystemPrompt,
    TextPrompt,
    ToolCall,
    ToolInput,
    ToolResponse,
)
from .base import Provider, ProviderFailure, Result, StreamText
from .common import attach_native, image_url, native_items, tool_arguments


class Responses(Provider):
    def background_operation(
        self, response_id: str, cancel: bool = False
    ) -> WireRequest:
        """编码 retrieve/cancel 控制请求；仅适用于已有任务 ID 的后台 Responses。"""
        url = self.route.endpoint.rstrip("/") + "/" + quote(response_id, safe="")
        if cancel:
            url += "/cancel"

        return WireRequest(
            url,
            "POST" if cancel else "GET",
            self.headers(),
            json={} if cancel else None,
        )

    def accepts_image_files(self) -> bool:
        """声明图片 file_id 的协议能力；通用 Files 存在不等于支持图片引用。"""
        return True

    def encode(self, context: Context, media: dict[str, list[MediaRef]]) -> Params:
        """编码 Responses 原生请求，保持历史和图片顺序；不上传或发送。"""
        items, tools, covered, instructions = [], [], set(), None
        message_index = {message.id: message for message in context}
        for prompt in context:
            if prompt.id in covered:
                continue

            native = native_items(prompt, "responses_api", self.route.scope, message_index)
            if native:
                items.extend(copy.deepcopy(native["items"]))
                covered.update(native["projection_ids"])

            elif isinstance(prompt, ToolInput):
                function = dict(prompt.tool_payloads["openai"]["function"])
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

        payload: Params = {
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

    def decode(self, body: Params) -> Result:
        """解析 Responses 文本、工具和原生记录；返回图片交给通信层落盘。"""
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
                        self.refuse(str(block.get("refusal") or ""), body)

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
            # noinspection PyTypeChecker
            result.usage["cached"] = usage["input_tokens_details"]["cached_tokens"]

        result.raw_usage = usage
        attach_native(result, "responses_api", self.route.scope)
        return result

    def stream_decoder(self) -> ResponsesStream:
        """为单次请求创建独立流式解码器，不复用上一轮的输出或 usage。"""
        return ResponsesStream(self)

    def usage_result(self, raw: Params) -> Result:
        """将供应商真实用量转换为统一字段，缺失值保持缺失。"""
        return self.decode({"output": [], "usage": raw})


class ResponsesStream:
    def __init__(self, provider):
        self.provider = provider
        self.refusal_parts: list[str] = []
        self.partial, self.complete = False, False
        self.body: Params = {"output": []}
        self.items = {}
        self.usage = {}

    def feed(self, event: Params) -> list[StreamText]:
        """接收一个原生事件；输出出现后设置 partial，损坏流仍保留已知 usage。"""
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

        if kind == "response.refusal.delta":
            refusal = event.get("delta", "")
            if not isinstance(refusal, str):
                raise ValueError("拒绝增量必须为字符串")

            self.refusal_parts.append(refusal)
            return [StreamText(refusal)]

        if kind == "response.output_text.delta":
            return [StreamText(event.get("delta", ""))]

        if kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            return [StreamText(event.get("delta", ""), reasoning=True)]

        return []

    def finish(self) -> Result:
        """聚合本次流式结果，保留 complete 状态；不补造缺失的结束事件。"""
        if self.refusal_parts:
            self.provider.refuse("".join(self.refusal_parts), {"usage": self.usage})

        if not self.body.get("output"):
            self.body["output"] = [self.items[k] for k in sorted(self.items)]

        self.body["usage"] = self.usage
        result = self.provider.decode(self.body)
        result.complete = result.complete and self.complete
        return result
