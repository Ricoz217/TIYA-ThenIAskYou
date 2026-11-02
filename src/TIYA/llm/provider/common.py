from __future__ import annotations

import copy
import json

from TIYA.utils import fingerprint

from ..model import Context, TextPrompt, ToolCall, ToolResponse, stable


def normalize(context: Context) -> Context:
    """重建容器并修补工具协议；不修改输入，未修补的消息保持原引用。

    Runtime 在入口负责请求隔离，这里只创建替代消息，不重复深复制整份历史。
    """
    messages, pending = [], {}
    original_ids = {}

    def close():
        for call_id, prompt in pending.items():
            messages.append(
                ToolResponse(
                    prompt.function_name,
                    call_id,
                    {
                        "system": {
                            "status": "协议已中止",
                            "messages": "请求发送前仍未取得工具结果",
                        }
                    },
                )
            )
        pending.clear()

    for prompt in context.message:
        original_id = prompt.id
        if isinstance(prompt, ToolCall):
            pending[prompt.call_id] = prompt
        elif isinstance(prompt, ToolResponse):
            if prompt.call_id not in pending:
                close()
                prompt = TextPrompt(
                    "user",
                    "# 孤立 TOOL RESPONSE\n\n"
                    + json.dumps(
                        {
                            "function_name": prompt.function_name,
                            "call_id": prompt.call_id,
                            "response": prompt.response,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
            else:
                pending.pop(prompt.call_id)
        elif prompt.role == "user":
            close()
        messages.append(prompt)
        original_ids[original_id] = prompt.id
    close()
    # Rebuild rounds from the normalized snapshot, retaining native items on each prompt.
    rebuilt = Context(messages, context.tools)
    rebuilt.session_id = context.session_id
    rebuilt.replace_usage(context.usage)
    # 轮次重建不能丢掉旧响应的用量。孤立工具响应被转换后，用原消息 ID 找回归属。
    round_for_message = {
        prompt.id: round_prompts
        for round_prompts in rebuilt.rounds.values()
        for prompt in round_prompts
    }
    for previous in context.rounds.values():
        if previous.usage is None:
            continue

        for prompt in reversed(previous):
            target = round_for_message.get(original_ids.get(prompt.id, prompt.id))
            if target is not None:
                if target.usage is None:
                    target.usage = previous.usage.copy()

                else:
                    target.usage += previous.usage

                break

    return rebuilt


def projection_fingerprint(prompt):
    return fingerprint(stable({k: v for k, v in vars(prompt).items() if k != "native"}))


def native_items(prompt, protocol, scope, context):
    record = getattr(prompt, "native", {}).get(protocol)
    if record and record.get("scope") == scope:
        current = {p.id: p for p in context.message}
        if all(
            pid in current
            and record.get("projection_hashes", {}).get(pid)
            == projection_fingerprint(current[pid])
            for pid in record["projection_ids"]
        ):
            return record
    return None


def attach_native(result, protocol, scope):
    if result.native and result.prompts.prompts:
        result.prompts[0].native[protocol] = {
            "scope": scope,
            "items": copy.deepcopy(result.native),
            "projection_ids": [p.id for p in result.prompts],
            "projection_hashes": {
                p.id: projection_fingerprint(p) for p in result.prompts
            },
        }
        result.prompts.refresh_fingerprints(content_changed=False)


def tool_arguments(text):
    value = json.loads(text) if isinstance(text, str) else text
    if not isinstance(value, dict):
        raise ValueError("工具参数必须为 JSON object")
    return value


def image_url(ref):
    if ref.kind == "url":
        return ref.value
    if ref.kind == "base64":
        return f"data:{ref.mime_type};base64,{ref.value}"
    raise ValueError(f"未准备的图片引用: {ref.kind}")
