"""Private adapter retained for callers of the historical payload helper.
Actual requests use the stable Provider/Runtime pipeline.
"""

from __future__ import annotations
import json
from ..model import (
    Context,
    SystemPrompt,
    TextPrompt,
    ImagePrompt,
    ToolCall,
    ToolResponse,
    ToolInput,
)
from ..tool_schema import build_tool_payloads
from ..legacy_media import _load_image_hashes
from TIYA.logger import get_logger

_log = get_logger()


class LegacyPayloadBuilder:
    async def _load_image_from_cache(self, hash_list: list[str]) -> bool:
        return await _load_image_hashes(hash_list, self._image_cache)

    async def _payload_constructor(self, context: Context) -> dict:
        provider = self._provider.lower()
        system_prompt = ""
        messages = []
        tools = []
        user_content = {"role": "user", "content": []}
        assistant_content: dict[str, str | list | None] = {"role": "assistant"}
        normalized_prompts = []
        pending_tool_calls: dict[str, ToolCall] = {}
        orphan_tool_response_ids: set[str] = set()
        closed_tool_call_prompt_ids: set[str] = set()

        def _append_missing_tool_responses(reason: str):
            if not pending_tool_calls:
                return

            for call_id, tool_call in pending_tool_calls.items():
                closed_tool_call_prompt_ids.add(tool_call.id)
                normalized_prompts.append(
                    ToolResponse(
                        function_name=tool_call.function_name,
                        call_id=call_id,
                        response={
                            "system": {"status": "协议已中止", "messages": reason}
                        },
                    )
                )

            _log.error(
                f"检测到 {len(pending_tool_calls)} 个未闭环 ToolCall，"
                f"已在构造 payload 时强制补全响应"
            )
            pending_tool_calls.clear()

        for prompt in context:
            if isinstance(prompt, ToolCall):
                pending_tool_calls[prompt.call_id] = prompt
                normalized_prompts.append(prompt)
                continue

            if isinstance(prompt, ToolResponse):
                if prompt.call_id in pending_tool_calls:
                    pending_tool_calls.pop(prompt.call_id)
                    normalized_prompts.append(prompt)

                else:
                    _append_missing_tool_responses(
                        "检测到不匹配的 ToolResponse，原 ToolCall 已由 payload 兜底关闭"
                    )
                    orphan_tool_response_ids.add(prompt.id)
                    response_text = json.dumps(
                        prompt.response, ensure_ascii=False, default=str
                    )
                    degraded_prompt = TextPrompt(
                        role="user",
                        text=(
                            "# 孤立 TOOL RESPONSE\n\n"
                            "系统检测到该工具结果已失去对应的 ToolCall，"
                            "为避免主控请求失败，已将其降级为普通通知。\n\n"
                            f"- function_name: `{prompt.function_name}`\n"
                            f"- call_id: `{prompt.call_id}`\n"
                            f"- response: `{response_text}`"
                        ),
                    )
                    normalized_prompts.append(degraded_prompt)
                    self._payload_append_prompts.append(degraded_prompt)
                    _log.error(
                        f"检测到孤立 ToolResponse [{prompt.call_id}]，"
                        f"已从 Context 剔除并降级为普通用户通知"
                    )

                continue

            if getattr(prompt, "role", "") == "user":
                _append_missing_tool_responses("后续用户消息到达前仍未取得工具结果")

            normalized_prompts.append(prompt)

        _append_missing_tool_responses("请求发送前仍未取得工具结果")
        discard_prompt_ids = orphan_tool_response_ids | closed_tool_call_prompt_ids
        context.discard_prompt_ids(discard_prompt_ids)
        self._payload_discard_prompt_ids.update(discard_prompt_ids)
        if context is not self._context:
            self._context.discard_prompt_ids(discard_prompt_ids)

        def _commit_user():
            if user_content["content"]:
                messages.append(user_content.copy())
                user_content.clear()
                user_content.update({"role": "user", "content": []})

        def _commit_assistant():
            if "content" in assistant_content or "tool_calls" in assistant_content:
                if "content" not in assistant_content:
                    assistant_content["content"] = None

                messages.append(assistant_content.copy())
                assistant_content.clear()
                assistant_content.update({"role": "assistant"})

        for prompt in normalized_prompts:
            if isinstance(prompt, SystemPrompt):
                if provider == "anthropic":
                    system_prompt = prompt.text

                elif provider == "openai":
                    new_message = {"role": prompt.role, "content": prompt.text}
                    if messages:
                        if messages[0]["role"] == prompt.role:
                            messages[0] = new_message

                        else:
                            messages.insert(0, new_message)

                    else:
                        messages.append(new_message)

                else:
                    raise ValueError(f"使用了不支持的API供应商格式: {provider}")

            elif isinstance(prompt, TextPrompt):
                if prompt.role == "user":
                    _commit_assistant()
                    user_content["content"].append(
                        {"type": "text", "text": prompt.text}
                    )

                else:
                    _commit_user()
                    if provider == "openai":
                        assistant_content["content"] = prompt.text

                        # 新增：thinking 模型需要把上一轮 reasoning_content 原样回传
                        if prompt.reasoning_content:
                            assistant_content["reasoning_content"] = (
                                prompt.reasoning_content
                            )

                    elif provider == "anthropic":
                        if "content" not in assistant_content:
                            assistant_content["content"] = []

                        assistant_content["content"].append(
                            {"type": "text", "text": prompt.text}
                        )

            elif isinstance(prompt, ImagePrompt):
                if prompt.image and not await self._load_image_from_cache(prompt.image):
                    prompt.image.clear()

                if not prompt.image:
                    await prompt.load_image(self._image_cache)

                if not prompt.image or not await self._load_image_from_cache(
                    prompt.image
                ):
                    prompt.image.clear()
                    continue

                if prompt.role == "user":
                    _commit_assistant()
                    if provider == "openai":
                        for hash_name in prompt.image:
                            user_content["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{self._image_cache[hash_name]}"
                                    },
                                }
                            )

                    elif provider == "anthropic":
                        for hash_name in prompt.image:
                            user_content["content"].append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": self._image_cache[hash_name],
                                    },
                                }
                            )

                    else:
                        raise ValueError(f"使用了不支持的API供应商格式: {provider}")

                else:
                    raise RuntimeError("尚未开发生图接口")

            elif isinstance(prompt, ToolCall):
                _commit_user()
                if provider == "openai":
                    if "tool_calls" not in assistant_content:
                        assistant_content["tool_calls"] = []

                    assistant_content["tool_calls"].append(
                        {
                            "id": prompt.call_id,
                            "type": "function",
                            "function": {
                                "name": prompt.function_name,
                                "arguments": json.dumps(
                                    prompt.arguments, ensure_ascii=False
                                ),
                            },
                        }
                    )

                elif provider == "anthropic":
                    if "content" not in assistant_content:
                        assistant_content["content"] = []

                    assistant_content["content"].append(
                        {
                            "type": "tool_use",
                            "id": prompt.call_id,
                            "name": prompt.function_name,
                            "input": prompt.arguments,
                        }
                    )

                else:
                    raise ValueError(f"使用了不支持的API供应商格式: {provider}")

            elif isinstance(prompt, ToolResponse):
                _commit_assistant()
                if provider == "openai":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": prompt.call_id,
                            "content": json.dumps(prompt.response, ensure_ascii=False),
                        }
                    )

                elif provider == "anthropic":
                    user_content["content"].append(
                        {
                            "type": "tool_result",
                            "tool_use_id": prompt.call_id,
                            "content": json.dumps(prompt.response, ensure_ascii=False),
                        }
                    )

                else:
                    raise ValueError(f"使用了不支持的API供应商格式: {provider}")

            elif isinstance(prompt, ToolInput):
                tool_schema = build_tool_payloads(prompt.function)
                tool_schema["openai"]["function"]["name"] = prompt.function_name
                tool_schema["anthropic"]["name"] = prompt.function_name
                if provider == "openai":
                    tools.append(tool_schema["openai"])

                elif provider == "anthropic":
                    tools.append(tool_schema["anthropic"])

                else:
                    raise ValueError(f"使用了不支持的API供应商格式: {provider}")

            else:
                raise ValueError("Prompts对象错误，无法构建payload")

        _commit_assistant()
        _commit_user()

        payload = {**self._model_params, "model": self._model, "messages": messages}

        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            payload["tools"] = tools

        if payload.get("stream", False):
            if provider == "openai":
                payload.setdefault("stream_options", {})  # type: ignore
                payload["stream_options"]["include_usage"] = True  # type: ignore

        # _log.debug(payload)
        # _log.debug(context)
        return payload
