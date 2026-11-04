from __future__ import annotations

import copy
import json

from collections.abc import Mapping
from typing import TypedDict

from ..config_model import Params
from ..model import BasePrompt, Context, MediaRef, Prompts, TextPrompt, ToolCall, ToolResponse
from .base import Result


def normalize(context: Context) -> Context:
    """重建容器并修补工具协议；不修改输入，未修补的消息保持原引用。

    Runtime 在入口负责请求隔离，这里只创建替代消息，不重复深复制整份历史。
    """
    messages: list[BasePrompt] = []
    pending_calls: dict[str, ToolCall] = {}
    original_ids: dict[str, str] = {}

    def complete_pending_calls() -> None:
        """用中止响应补全未闭环调用，确保后续 user 消息不会打断工具协议。"""
        for call_id, pending_call in pending_calls.items():
            messages.append(
                ToolResponse(
                    pending_call.function_name,
                    call_id,
                    {
                        "system": {
                            "status": "协议已中止",
                            "messages": "请求发送前仍未取得工具结果",
                        }
                    },
                )
            )

        pending_calls.clear()

    for source_message in context.message:
        normalized_message = source_message
        if isinstance(source_message, ToolCall):
            pending_calls[source_message.call_id] = source_message

        elif isinstance(source_message, ToolResponse):
            if source_message.call_id in pending_calls:
                pending_calls.pop(source_message.call_id)

            else:
                complete_pending_calls()
                normalized_message = TextPrompt(
                    "user",
                    "# 孤立 TOOL RESPONSE\n\n"
                    + json.dumps(
                        {
                            "function_name": source_message.function_name,
                            "call_id": source_message.call_id,
                            "response": source_message.response,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )

        elif source_message.role == "user":
            complete_pending_calls()

        messages.append(normalized_message)
        original_ids[source_message.id] = normalized_message.id

    complete_pending_calls()

    # 重建规范化后的轮次；未替换的消息及其 native 仍沿用入口快照中的对象。
    rebuilt = Context(messages, context.tools)
    rebuilt.session_id = context.session_id
    rebuilt.replace_usage(context.usage)
    # 轮次重建不能丢掉旧响应的用量。孤立工具响应被转换后，用原消息 ID 找回归属。
    round_for_message: dict[str, Prompts] = {
        message.id: round_prompts
        for round_prompts in rebuilt.rounds.values()
        for message in round_prompts
    }
    for previous_round in context.rounds.values():
        if previous_round.usage is None:
            continue

        for previous_message in reversed(previous_round):
            target = round_for_message.get(original_ids.get(previous_message.id, previous_message.id))
            if target is not None:
                if target.usage is None:
                    target.usage = previous_round.usage.copy()

                else:
                    target.usage += previous_round.usage

                break

    return rebuilt


class NativeRecord(TypedDict):
    """原生内容与消息关联；signatures 是当时的 BasePrompt.signature 快照。"""

    scope: str
    items: list[Params]
    projection_ids: list[str]
    signatures: dict[str, str]


def native_items(
    prompt: BasePrompt,
    protocol: str,
    scope: str,
    message_index: Mapping[str, BasePrompt],
) -> NativeRecord | None:
    """校验并借用原生记录；索引由 encode 构建一次，避免逐记录遍历完整历史。

    返回的记录按只读使用，关联消息必须存在且 signature 与快照一致。
    不探测未显式刷新的原位修改，也不另算投影哈希。
    """
    record: NativeRecord | None = prompt.native.get(protocol)
    if record is None or record["scope"] != scope:
        return None

    signatures = record.get("signatures", {})
    for prompt_id in record["projection_ids"]:
        projected_message = message_index.get(prompt_id)
        if projected_message is None:
            return None

        if signatures.get(prompt_id) != projected_message.signature:
            return None

    return record


def attach_native(result: Result, protocol: str, scope: str) -> None:
    """将原生输出和消息 signature 快照绑定到首条消息，不改变消息签名。

    原生数据另存副本，后续消费 Result.native 不会修改已附着的回放记录。
    """
    if not result.native or not result.prompts.prompts:
        return

    record: NativeRecord = {
        "scope": scope,
        "items": copy.deepcopy(result.native),
        "projection_ids": [message.id for message in result.prompts],
        "signatures": {
            message.id: message.signature for message in result.prompts
        },
    }
    result.prompts[0].native[protocol] = record


def tool_arguments(text: str | Params) -> Params:
    """解析工具参数；只接受 JSON object，其他 JSON 值抛 ValueError。"""
    value = json.loads(text) if isinstance(text, str) else text
    if not isinstance(value, dict):
        raise ValueError("工具参数必须为 JSON object")

    return value


def image_url(ref: MediaRef) -> str:
    """将已准备的 URL/base64 引用转为请求地址；不读取文件或上传。"""
    if not isinstance(ref.value, str):
        raise ValueError("图片 URL/base64 引用必须包含字符串")

    if ref.kind == "url":
        return ref.value

    if ref.kind == "base64":
        return f"data:{ref.mime_type};base64,{ref.value}"

    raise ValueError(f"未准备的图片引用: {ref.kind}")
