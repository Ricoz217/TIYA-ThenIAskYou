from __future__ import annotations

import copy
import json

from ..config_model import Params
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
from ..tool_schema import build_tool_payloads
from .base import Provider, Result, merge_headers
from .common import attach_native, native_items, tool_arguments


class Messages(Provider):
    def accepts_image_files(self) -> bool:
        """声明图片 file_id 的协议能力；通用 Files 存在不等于支持图片引用。"""
        return True

    def headers(self):
        headers = {
            "x-api-key": self.route.token,
            "anthropic-version": self.route.provider_options.get(
                "anthropic_version", "2023-06-01"
            ),
            **self.route.headers,
        }
        if self.route.files and self.route.files.provider_options.get("anthropic_beta"):
            headers.setdefault(
                "anthropic-beta", self.route.files.provider_options["anthropic_beta"]
            )
        return merge_headers(headers, self.route.client_params.get("headers"))

    def encode(self, context: Context, media: dict[str, list[MediaRef]]) -> Params:
        """编码 Anthropic Messages 原生请求，保持历史和图片顺序；不上传或发送。"""
        messages, tools, system, covered = [], [], "", set()
        for prompt in context:
            if prompt.id in covered:
                continue
            native = native_items(prompt, "anthropic_api", self.route.scope, context)
            if native:
                messages.append(
                    {"role": "assistant", "content": copy.deepcopy(native["items"])}
                )
                covered.update(native["projection_ids"])
                continue
            if isinstance(prompt, SystemPrompt):
                system = prompt.text
                continue
            if isinstance(prompt, ToolInput):
                tool = build_tool_payloads(prompt.function)["anthropic"]
                tool["name"] = prompt.function_name
                tool.pop("returns", None)
                tools.append(tool)
                continue
            role = "user" if isinstance(prompt, ToolResponse) else prompt.role
            if isinstance(prompt, ToolCall):
                content = [
                    {
                        "type": "tool_use",
                        "id": prompt.call_id,
                        "name": prompt.function_name,
                        "input": prompt.arguments,
                    }
                ]
            elif isinstance(prompt, ToolResponse):
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": prompt.call_id,
                        "content": json.dumps(prompt.response, ensure_ascii=False),
                    }
                ]
            elif isinstance(prompt, TextPrompt):
                content = [{"type": "text", "text": prompt.text}]
            elif isinstance(prompt, ImagePrompt):
                content = []
                for ref in media[prompt.id]:
                    if ref.kind == "remote":
                        source = {"type": "file", "file_id": ref.value}
                    elif ref.kind == "url":
                        source = {"type": "url", "url": ref.value}
                    else:
                        source = {
                            "type": "base64",
                            "media_type": ref.mime_type,
                            "data": ref.value,
                        }
                    content.append({"type": "image", "source": source})
            else:
                raise ValueError(f"未知 Prompt 类型: {type(prompt)}")
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(content)
            else:
                messages.append({"role": role, "content": content})
        payload = {
            **copy.deepcopy(self.route.model_params),
            "model": self.route.model,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
        return payload

    def decode(self, body: Params) -> Result:
        """解析 Anthropic Messages 文本、工具和原生记录；返回图片交给通信层落盘。"""
        self.check_body(body)
        if not isinstance(body.get("content"), list):
            raise ValueError("Messages 响应缺少 content")
        result = Result(
            response_id=body.get("id", ""),
            native=copy.deepcopy(body.get("content", [])),
            reported_model=body.get("model", ""),
        )
        for block in result.native:
            if block.get("type") == "text":
                result.prompts.append(TextPrompt("assistant", block.get("text", "")))
            elif block.get("type") == "tool_use":
                result.prompts.append(
                    ToolCall(block["name"], block["id"], tool_arguments(block["input"]))
                )
        if result.native and not result.prompts:
            result.prompts.append(TextPrompt("assistant", ""))
        usage = body.get("usage") or {}
        if "input_tokens" in usage:
            result.usage["input"] = (
                usage["input_tokens"]
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )
        for source, target in (
            ("output_tokens", "output"),
            ("cache_read_input_tokens", "cached"),
            ("cache_creation_input_tokens", "cache_write"),
        ):
            if source in usage:
                result.usage[target] = usage[source]
        result.raw_usage = usage
        attach_native(result, "anthropic_api", self.route.scope)
        return result

    def stream_decoder(self) -> MessagesStream:
        """为单次请求创建独立流式解码器，不复用上一轮的输出或 usage。"""
        return MessagesStream(self)

    def usage_result(self, raw: Params) -> Result:
        """将供应商真实用量转换为统一字段，缺失值保持缺失。"""
        return self.decode({"content": [], "usage": raw})


class MessagesStream:
    def __init__(self, provider):
        self.provider = provider
        self.blocks, self.arguments, self.usage = {}, {}, {}
        self.partial, self.complete = False, False
        self.identifier = ""
        self.model = ""

    def feed(self, event: Params) -> None:
        """接收一个原生事件；输出出现后设置 partial，损坏流仍保留已知 usage。"""
        self.provider.check_body(event)
        kind = event.get("type")
        if kind == "message_start":
            self.identifier = event.get("message", {}).get("id", "")
            self.model = event.get("message", {}).get("model", "")
            self.usage.update(event.get("message", {}).get("usage") or {})
        elif kind == "content_block_start":
            block = copy.deepcopy(event["content_block"])
            self.blocks[event["index"]] = block
            self.partial = (
                self.partial or block.get("type") != "text" or bool(block.get("text"))
            )
        elif kind == "content_block_delta":
            self.partial = True
            index, delta = event["index"], event["delta"]
            block = self.blocks[index]
            if delta.get("type") == "input_json_delta":
                self.arguments[index] = (
                    self.arguments.get(index, "") + delta["partial_json"]
                )
            else:
                for key in ("text", "thinking", "signature"):
                    if key in delta:
                        block[key] = block.get(key, "") + delta[key]
        elif kind == "message_delta":
            self.usage.update(event.get("usage") or {})
        elif kind == "message_stop":
            self.complete = True
        elif "content" in event and kind == "message":
            self.final = event
            self.complete = True
            self.partial = True

    def finish(self) -> Result:
        """聚合本次流式结果，保留 complete 状态；不补造缺失的结束事件。"""
        if hasattr(self, "final"):
            return self.provider.decode(self.final)
        for index, arguments in self.arguments.items():
            self.blocks[index]["input"] = tool_arguments(arguments)
        result = self.provider.decode(
            {
                "id": self.identifier,
                "model": self.model,
                "content": [self.blocks[k] for k in sorted(self.blocks)],
                "usage": self.usage,
            }
        )
        result.complete = self.complete
        return result
