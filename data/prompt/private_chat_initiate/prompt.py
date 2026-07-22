from __future__ import annotations

import asyncio
import json

from TIYA.config import get_bot_name, get_bot_uid
from TIYA.model.message import PrivateMsg


async def prompt(
    user_id: str,
    username: str,
    private_summary: str,
    old_messages: list[PrivateMsg],
    news: dict,
) -> str:
    tasks = [asyncio.create_task(msg.to_llm(wait=True, timeout=15)) for msg in old_messages]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    messages = [
        task.result()
        for task in tasks
        if task.done() and not task.cancelled() and task.exception() is None
    ]

    return f"""\
**本次请求来自 Agent 初始化，请勿发言**

1. 调用 `list_external_tools` 检查可用外置工具
2. 调用 `load_external_tool` 加载需要的外部工具
3. 调用 `skill_installer` 的 `load_skill` 加载 `skill_installer` 中关于 SKILL 的推荐使用方式
4. 加载 SKILL `private_chat` 的 **完整** 内容
5. 加载需要的其他已安装 SKILL 信息


# 用户

- QQ号: `{user_id}`
- QQ昵称: `{username or '未知'}`


# BOT

- QQ号: `{get_bot_uid()}`
- QQ昵称: `{get_bot_name()}`


# 私聊完整画像与记忆摘要

{private_summary or '无'}


# 相关新闻

{json.dumps(news, ensure_ascii=False, indent=2) if news else '无'}

# 近期聊天记录

```json
{json.dumps(messages, ensure_ascii=False, indent=2)}
```
"""
