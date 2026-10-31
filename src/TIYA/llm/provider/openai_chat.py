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
from .base import Provider, Result
from .common import image_url, tool_arguments


class ChatCompletions(Provider):
    def accepts_image_files(self) -> bool:
        """声明图片 file_id 的协议能力；通用 Files 存在不等于支持图片引用。"""
        return (
            self.route.provider_options.get("chat_completions", {}).get(
                "image_file_format"
            )
            == "deepseek"
        )

    def encode(self, context: Context, media: dict[str, list[MediaRef]]) -> Params:
        """编码 Chat Completions 原生请求，保持历史和图片顺序；不上传或发送。"""
        messages, tools = [], []
        for prompt in context:
            if isinstance(prompt, ToolInput):
                tool = build_tool_payloads(prompt.function)["openai"]
                tool["function"]["name"] = prompt.function_name
                tools.append(tool)
            elif isinstance(prompt, SystemPrompt):
                messages.append({"role": "system", "content": prompt.text})
            elif isinstance(prompt, ToolCall):
                call = {
                    "id": prompt.call_id,
                    "type": "function",
                    "function": {
                        "name": prompt.function_name,
                        "arguments": json.dumps(prompt.arguments, ensure_ascii=False),
                    },
                }
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1].setdefault("tool_calls", []).append(call)
                else:
                    messages.append(
                        {"role": "assistant", "content": None, "tool_calls": [call]}
                    )
            elif isinstance(prompt, ToolResponse):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": prompt.call_id,
                        "content": json.dumps(prompt.response, ensure_ascii=False),
                    }
                )
            elif isinstance(prompt, TextPrompt):
                message = {
                    "role": prompt.role,
                    "content": [{"type": "text", "text": prompt.text}]
                    if prompt.role == "user"
                    else prompt.text,
                }
                if prompt.reasoning_content:
                    message["reasoning_content"] = prompt.reasoning_content
                if (
                    prompt.role == "user"
                    and messages
                    and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                ):
                    messages[-1]["content"].extend(message["content"])
                else:
                    messages.append(message)
            elif isinstance(prompt, ImagePrompt):
                content = []
                for ref in media[prompt.id]:
                    if ref.kind == "remote":
                        if not self.accepts_image_files():
                            raise ValueError("当前 Chat 图片协议不支持 file_id")
                        content.append({"type": "file", "file_id": ref.value})
                    else:
                        content.append(
                            {"type": "image_url", "image_url": {"url": image_url(ref)}}
                        )
                if (
                    messages
                    and messages[-1]["role"] == prompt.role
                    and isinstance(messages[-1]["content"], list)
                ):
                    messages[-1]["content"].extend(content)
                else:
                    messages.append({"role": prompt.role, "content": content})
        payload = {
            **copy.deepcopy(self.route.model_params),
            "model": self.route.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if payload.get("stream"):
            payload.setdefault("stream_options", {})["include_usage"] = True
        return payload

    def decode(self, body: Params) -> Result:
        """解析 Chat Completions 文本、工具和原生记录；返回图片交给通信层落盘。"""
        self.check_body(body)
        if not isinstance(body.get("choices"), list) or not body["choices"]:
            raise ValueError("Chat Completions 响应缺少 choices")
        result = Result(reported_model=body.get("model", ""))
        for choice in body.get("choices", []):
            message = choice.get("message", {})
            content = message.get("content") or message.get("refusal")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") in ("text", "output_text", "refusal"):
                        result.prompts.append(
                            TextPrompt(
                                "assistant", block.get("text", block.get("refusal", ""))
                            )
                        )
                    elif block.get("type") in ("image_url", "image", "output_image"):
                        source = block.get("image_url", block)
                        wire = (
                            {"url": source}
                            if isinstance(source, str)
                            else {
                                k: v
                                for k, v in source.items()
                                if k in ("url", "b64_json")
                            }
                        )
                        if not wire:
                            raise ValueError("图片内容块缺少可获取的图片")
                        image = ImagePrompt("assistant", b"")
                        result.prompts.append(image)
                        result.images.append((image, wire))
                    else:
                        raise ValueError(f"无法投影的 Chat 内容块: {block.get('type')}")
                if message.get("reasoning_content"):
                    for prompt in result.prompts:
                        if isinstance(prompt, TextPrompt):
                            prompt.reasoning_content = message["reasoning_content"]
                            break
            elif content or message.get("reasoning_content"):
                result.prompts.append(
                    TextPrompt(
                        "assistant",
                        content or "",
                        message.get("reasoning_content") or "",
                    )
                )
            for call in message.get("tool_calls", []):
                function = call["function"]
                result.prompts.append(
                    ToolCall(
                        function["name"],
                        call["id"],
                        tool_arguments(function["arguments"]),
                    )
                )
        usage = body.get("usage") or {}
        mapping = {
            "prompt_tokens": "input",
            "completion_tokens": "output",
            "total_tokens": "total",
        }
        result.usage = {
            target: usage[source]
            for source, target in mapping.items()
            if usage.get(source) is not None
        }
        cached = usage.get("prompt_tokens_details", {}).get(
            "cached_tokens", usage.get("prompt_cache_hit_tokens")
        )
        if cached is not None:
            result.usage["cached"] = cached
        result.raw_usage = usage
        result.response_id = body.get("id", "")
        return result

    def stream_decoder(self) -> ChatStream:
        """为单次请求创建独立流式解码器，不复用上一轮的输出或 usage。"""
        return ChatStream(self)

    def usage_result(self, raw: Params) -> Result:
        """将供应商真实用量转换为统一字段，缺失值保持缺失。"""
        return self.decode({"choices": [{"message": {}}], "usage": raw})


class ChatStream:
    def __init__(self, provider):
        self.provider = provider
        self.message = {"content": "", "reasoning_content": ""}
        self.calls = {}
        self.usage, self.partial, self.complete = {}, False, False
        self.model, self.identifier = "", ""

    def feed(self, body: Params) -> None:
        """接收一个原生事件；输出出现后设置 partial，损坏流仍保留已知 usage。"""
        self.provider.check_body(body)
        self.model = body.get("model") or self.model
        self.identifier = body.get("id") or self.identifier
        if "choices" in body and any("message" in c for c in body["choices"]):
            self.final = body
            self.partial = True
            self.complete = True
            return
        self.usage.update(body.get("usage") or {})
        for choice in body.get("choices", []):
            delta = choice.get("delta", {})
            for key in ("content", "reasoning_content"):
                if delta.get(key):
                    self.partial = True
                    if key == "content" and isinstance(delta[key], list):
                        if isinstance(self.message[key], str):
                            self.message[key] = (
                                [{"type": "text", "text": self.message[key]}]
                                if self.message[key]
                                else []
                            )
                        self.message[key].extend(copy.deepcopy(delta[key]))
                    elif key == "content" and isinstance(self.message[key], list):
                        self.message[key].append({"type": "text", "text": delta[key]})
                    else:
                        self.message[key] += delta[key]
            for call in delta.get("tool_calls", []):
                self.partial = True
                target = self.calls.setdefault(
                    call.get("index", 0),
                    {"id": "", "function": {"name": "", "arguments": ""}},
                )
                target["id"] += call.get("id", "")
                for key in ("name", "arguments"):
                    target["function"][key] += call.get("function", {}).get(key, "")
            if choice.get("finish_reason"):
                self.complete = True

    def finish(self) -> Result:
        """聚合本次流式结果，保留 complete 状态；不补造缺失的结束事件。"""
        if hasattr(self, "final"):
            return self.provider.decode(self.final)
        self.message["tool_calls"] = [self.calls[k] for k in sorted(self.calls)]
        result = self.provider.decode(
            {
                "choices": [{"message": self.message}],
                "usage": self.usage,
                "model": self.model,
                "id": self.identifier,
            }
        )
        result.complete = self.complete
        return result
