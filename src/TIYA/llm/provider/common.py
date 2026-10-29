from __future__ import annotations

import copy
import json

from ..model import Context, TextPrompt, ToolCall, ToolResponse


def normalize(context):
    """Repair a snapshot only; return the complete planned context for atomic commit."""
    result = copy.deepcopy(context)
    messages, pending = [], {}

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

    for prompt in result._messages:
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
    close()
    # Rebuild rounds from the normalized snapshot, retaining native items on each prompt.
    rebuilt = Context(messages, result._tools)
    rebuilt.session_id = context.session_id
    rebuilt._usage_accumulation = result._usage_accumulation
    return rebuilt


def projection_fingerprint(prompt):
    from ..snapshot import stable
    from ..config import fingerprint

    return fingerprint(stable({k: v for k, v in vars(prompt).items() if k != "native"}))


def native_items(prompt, protocol, scope, context):
    record = getattr(prompt, "native", {}).get(protocol)
    if record and record.get("scope") == scope:
        current = {p.id: p for p in context._messages}
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
