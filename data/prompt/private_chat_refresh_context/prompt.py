from __future__ import annotations

import asyncio
import json

from TIYA.LLM_connect import TextPrompt
from TIYA.config import get_bot_name, get_bot_uid
from TIYA.model.message import PrivateMsg


async def prompt(
    user_id: str,
    username: str,
    private_summary: str,
    old_messages: list[PrivateMsg],
    news: dict,
    new_request: TextPrompt | None,
) -> str:
    new_prompt = ""
    if new_request.text:
        new_prompt = f"\n\n下面是本次请求的 prompt:  \n\n{new_request.text}"

    tasks = [asyncio.create_task(msg.to_llm(wait=True, timeout=15)) for msg in old_messages]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    messages = [
        task.result()
        for task in tasks
        if task.done() and not task.cancelled() and task.exception() is None
    ]
    return f"""\
**Agent 触发了 Context 刷新**
以下是新 Context 初始化信息，你需要根据信息继续群聊  


# 用户
- QQ号: `{user_id}`
- QQ昵称: `{username or '未知'}`


# BOT
- QQ号: `{get_bot_uid()}`
- QQ昵称: `{get_bot_name()}`


# 用户、BOT 与双方关系画像及记忆摘要
{private_summary or '无'}


# 相关新闻
{json.dumps(news, ensure_ascii=False, indent=2) if news else '无'}


# 近期聊天记录
```json
{json.dumps(messages, ensure_ascii=False, indent=2)}
```


> 以上为 Agent Context 刷新自动生成的基本信息  {new_prompt}


# 重新初始化流程

1. 调用 `load_external_tool` 加载缺失的所需外部工具
2. 调用 `load_skill` 检查 SKILL 是否缺失所需内容，尤其是 `private_chat`，必须保证该 SKILL 内容完整
3. 调用 `get_skill_content` 加载/更新所需的 SKILL 内容
4. 调用 `get_tasks_list` 查看当前任务状态
5. 根据信息，继续群聊任务  
"""
