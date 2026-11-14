from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

from ..config_model import JsonValue, Params
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
from .base import Provider, Result, StreamText
from .common import image_url, tool_arguments


class ChatCompletions(Provider):
    def accepts_image_files(self) -> bool:
        """声明图片 file_id 的协议能力；通用 Files 存在不等于支持图片引用。"""
        options = self.route.provider_options.get("chat_completions", {})
        return isinstance(options, dict) and options.get("image_file_format") == "deepseek"

    def encode(self, context: Context, media: dict[str, list[MediaRef]]) -> Params:
        """编码 Chat Completions 原生请求，保持历史和图片顺序；不上传或发送。"""
        if self.route.model_params.get("n", 1) not in (None, 1):
            raise ValueError("当前仅支持单候选 n=1")

        messages: list[dict] = []
        tools: list[dict] = []
        for prompt in context:
            if isinstance(prompt, ToolInput):
                tool = prompt.tool_payloads["openai"].copy()
                function = tool["function"]
                tool["function"] = {**function, "name": prompt.function_name}
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

        payload = copy.deepcopy(self.route.model_params)
        payload["model"] = self.route.model
        payload["messages"] = list(messages)
        if tools:
            payload["tools"] = list(tools)

        if payload.get("stream"):
            stream_options = payload.setdefault("stream_options", {})
            if not isinstance(stream_options, dict):
                raise ValueError("stream_options 必须为对象")

            stream_options["include_usage"] = True

        return payload

    def decode(self, body: Params) -> Result:
        """解析 Chat Completions 文本、工具和原生记录；返回图片交给通信层落盘。"""
        self.check_body(body)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Chat Completions 响应缺少 choices")

        model = body.get("model", "")
        result = Result(reported_model=model if isinstance(model, str) else "")
        if len(choices) > 1:
            raise ValueError("当前仅支持单候选 n=1")

        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("Chat choice 必须为对象")

            message = choice.get("message", {})
            if not isinstance(message, dict):
                raise ValueError("Chat message 必须为对象")

            if message.get("refusal") or choice.get("finish_reason") == "content_filter":
                self.refuse(str(message.get("refusal") or "服务端内容过滤"), body)

            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        raise ValueError("Chat 内容块必须为对象")

                    if block.get("type") == "refusal":
                        self.refuse(str(block.get("refusal") or block.get("text") or ""), body)

                    if block.get("type") in ("text", "output_text"):
                        result.prompts.append(
                            TextPrompt(
                                "assistant", block.get("text", block.get("refusal", ""))
                            )
                        )

                    elif block.get("type") in ("image_url", "image", "output_image"):
                        source = block.get("image_url", block)
                        if not isinstance(source, (str, dict)):
                            raise ValueError("图片内容块缺少可获取的图片")

                        wire: Params = (
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

            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("tool_calls 必须为列表")

            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("工具调用缺少 function 对象")

                function = call.get("function")
                if not isinstance(function, dict):
                    raise ValueError("工具调用缺少 function 对象")

                arguments = function["arguments"]
                if not isinstance(arguments, (str, dict)):
                    raise ValueError("工具参数必须为 JSON object")

                result.prompts.append(
                    ToolCall(
                        function["name"],
                        call["id"],
                        tool_arguments(arguments),
                    )
                )
        usage = body.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("usage 必须为对象")

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

        details = usage.get("prompt_tokens_details")
        cached = usage.get("prompt_cache_hit_tokens")
        if isinstance(details, dict):
            cached = details.get("cached_tokens", cached)

        if cached is not None:
            result.usage["cached"] = cached

        result.raw_usage = usage
        response_id = body.get("id", "")
        result.response_id = response_id if isinstance(response_id, str) else ""
        return result

    def stream_decoder(self) -> ChatStream:
        """为单次请求创建独立流式解码器，不复用上一轮的输出或 usage。"""
        return ChatStream(self)

    def usage_result(self, raw: Params) -> Result:
        """将供应商真实用量转换为统一字段，缺失值保持缺失。"""
        return self.decode({"choices": [{"message": {}}], "usage": raw})


@dataclass(slots=True)
class _ToolCallFragments:
    """单个工具调用的增量片段，结束后各字段只拼接一次。"""

    ids: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)

    def message(self) -> Params:
        return {
            "id": "".join(self.ids),
            "function": {
                "name": "".join(self.names),
                "arguments": "".join(self.arguments),
            },
        }


class ChatStream:
    """缓存单次 Chat 流的片段；finish 时拼接文本，feed 返回即时日志增量。

    内容块按接收顺序借用，不修改、不深复制。文本和图片交错时只合并相邻文本，
    不把图片前后的文本挪到一起；工具调用按 index 分别聚合。
    """

    def __init__(self, provider: ChatCompletions) -> None:
        self.provider = provider
        self.content_parts: list[str | Params] = []
        self.reasoning_parts: list[str] = []
        self.refusal_parts: list[str] = []
        self.calls: dict[int, _ToolCallFragments] = {}
        self.usage: Params = {}
        self.partial = False
        self.complete = False
        self.model = ""
        self.identifier = ""
        self.final: Params | None = None

    def feed(self, body: Params) -> list[StreamText]:
        """缓存一个事件，不拼接历史片段；usage/partial 在异常前保留已知进度。"""
        self.provider.check_body(body)
        model = body.get("model")
        identifier = body.get("id")
        if isinstance(model, str) and model:
            self.model = model

        if isinstance(identifier, str) and identifier:
            self.identifier = identifier

        usage = body.get("usage")
        if isinstance(usage, dict):
            self.usage.update(usage)

        logs: list[StreamText] = []
        choices = body.get("choices") or []
        if not isinstance(choices, list):
            raise ValueError("流式 choices 必须为列表")

        if any(isinstance(choice, dict) and "message" in choice for choice in choices):
            self.final = body
            self.partial = True
            self.complete = True
            for choice in choices:
                if isinstance(choice, dict):
                    message = choice.get("message", {})
                    if isinstance(message, dict):
                        logs.extend(self._message_logs(message))

            return logs

        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("流式 choice 必须为对象")

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ValueError("流式 delta 必须为对象")

            if choice.get("index", 0) != 0 or len(choices) > 1:
                raise ValueError("当前仅支持单候选 n=1")

            refusal = delta.get("refusal")
            if isinstance(refusal, str) and refusal:
                self.partial = True
                self.refusal_parts.append(refusal)
                logs.append(StreamText(refusal))

            if choice.get("finish_reason") == "content_filter":
                self.partial = True
                self.refusal_parts.append("服务端内容过滤")

            for key in ("reasoning_content", "reasoning"):
                reasoning = delta.get(key)
                if isinstance(reasoning, str) and reasoning:
                    self.partial = True
                    self.reasoning_parts.append(reasoning)
                    logs.append(StreamText(reasoning, reasoning=True))

            content = delta.get("content")
            if content:
                self.partial = True
                if isinstance(content, str):
                    self.content_parts.append(content)

                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            raise ValueError("流式内容块必须为对象")

                        self.content_parts.append(block)

                else:
                    raise ValueError("流式 content 必须为文本或内容块列表")

                logs.extend(self._content_logs(content))

            calls = delta.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("流式 tool_calls 必须为列表")

            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("流式工具调用必须为对象")

                call_index = call.get("index", 0)
                if not isinstance(call_index, int):
                    raise ValueError("流式工具调用 index 必须为整数")

                self.partial = True
                target = self.calls.get(call_index)
                if target is None:
                    target = _ToolCallFragments()
                    self.calls[call_index] = target

                self._collect_call(target, call)

            legacy_call = delta.get("function_call")
            if isinstance(legacy_call, dict) and legacy_call:
                self.partial = True
                target = self.calls.get(-1)
                if target is None:
                    target = _ToolCallFragments(ids=["legacy_function_call"])
                    self.calls[-1] = target

                self._collect_call(target, {"function": legacy_call})

            if choice.get("finish_reason"):
                self.complete = True

        return logs

    @staticmethod
    def _collect_call(target: _ToolCallFragments, call: Params) -> None:
        identifier = call.get("id")
        if isinstance(identifier, str) and identifier:
            target.ids.append(identifier)

        function = call.get("function")
        if not isinstance(function, dict):
            return

        name = function.get("name")
        arguments = function.get("arguments")
        if isinstance(name, str) and name:
            target.names.append(name)

        if isinstance(arguments, str) and arguments:
            target.arguments.append(arguments)

    @staticmethod
    def _content_logs(content: JsonValue) -> list[StreamText]:
        if isinstance(content, str):
            return [StreamText(content)] if content else []

        logs = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue

                if block.get("type") in ("text", "output_text", "refusal"):
                    text = block.get("text", block.get("refusal"))
                    if isinstance(text, str) and text:
                        logs.append(StreamText(text))

        return logs

    def _message_logs(self, message: Params) -> list[StreamText]:
        logs = []
        for key in ("reasoning_content", "reasoning"):
            text = message.get(key)
            if isinstance(text, str) and text:
                logs.append(StreamText(text, reasoning=True))

        logs.extend(self._content_logs(message.get("content") or message.get("refusal") or ""))
        return logs

    def _content(self) -> str | list[JsonValue]:
        """一次线性组装；纯文本返回字符串，图文混合保留内容块顺序。"""
        blocks: list[JsonValue] = []
        text_parts: list[str] = []
        for part in self.content_parts:
            if isinstance(part, str):
                text_parts.append(part)
                continue

            if part.get("type") in ("text", "output_text") and set(part) <= {"type", "text"}:
                text = part.get("text", "")
                if isinstance(text, str):
                    text_parts.append(text)
                    continue

            if text_parts:
                blocks.append({"type": "text", "text": "".join(text_parts)})
                text_parts.clear()

            blocks.append(part)

        if not blocks:
            return "".join(text_parts)

        if text_parts:
            blocks.append({"type": "text", "text": "".join(text_parts)})

        return blocks

    def finish(self) -> Result:
        """组装后复用完整响应解码；保留真实结束状态，不补造结束事件。"""
        if self.final is not None:
            return self.provider.decode(self.final)

        message: Params = {
            "content": self._content(),
            "refusal": "".join(self.refusal_parts),
            "reasoning_content": "".join(self.reasoning_parts),
            "tool_calls": [self.calls[index].message() for index in sorted(self.calls)],
        }
        result = self.provider.decode({
            "choices": [{"message": message}],
            "usage": self.usage,
            "model": self.model,
            "id": self.identifier,
        })
        result.complete = self.complete
        return result
