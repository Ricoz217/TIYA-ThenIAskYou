from __future__ import annotations
"""
群聊 Agent 的刷新 context 窗口提示词，按照初始化 prompt 的信息模型，同时附带新请求的内容
"""
__version__ = "0.1.0"

import asyncio
import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from TIYA.qq_group import MemberList
    from TIYA.model.message import GroupMsg
    from TIYA.LLM_connect import TextPrompt

async def prompt(
        group_name: str,
        group_id: str,
        group_summary: str,
        member_list: MemberList,
        active_member_info: list[dict],
        old_messages: list[GroupMsg],
        news: dict[str, Any],
        new_request: TextPrompt | None
) -> str:
    _role_mapping = {
        "member": "普通群员",
        "owner": "群主",
        "admin": "管理员",
        "robot": "群机器人"
    }
    new_prompt = ""
    if new_request is not None:
        new_prompt = f"\n下面是本次请求的 prompt:  \n\n{new_request.text}  "

    if member_list is None:
        return ""

    group_owner = member_list.get_owner
    admins = member_list.get_admins
    robots = member_list.get_robots

    if not admins:
        admins_text = "None"

    else:
        admins_text = ""
        for i, admin in enumerate(admins, start=1):
            admins_text += f"{i}. {str(admin).replace('\n', '; ')}\n"

        admins_text = admins_text.strip('\n')

    if not robots:
        robots_text = "None"

    else:
        robots_text = ""
        for i, robot in enumerate(robots, start=1):
            robots_text += f"{i}. {str(robot).replace('\n', '; ')}\n"

        robots_text = robots_text.strip('\n')

    if not active_member_info:
        active_member_text = "**无**"

    else:
        active_member_text = ""
        for member in active_member_info:
            user_id = member.get("user_id", "")
            member_obj = member_list.get_member(user_id=user_id)
            if not member_obj:
                continue

            member_obj = member_obj[0]
            relations: dict = member.get('relations', {})
            if not relations:
                relations_text = "`未知`"

            else:
                relations_text = "\n"
                for k, v in relations.items():
                    relations_text += f"\t\t- `{k}`: {v}\n"

                relations_text = relations_text.rstrip('\n')

            member_persona = [
                f"- {member_obj.nickname or member_obj.username or user_id}:  ",
                f"\t- QQ号: `{user_id}`",
                f"\t- 身份: `{_role_mapping[member_obj.role.value]}`",
                f"\t- 群友爱称: `{member.get('alias', '无特殊称呼')}`",
                f"\t- 性别: `{member.get('gender', '未知')}`",
                f"\t- 年龄: `{member.get('age', '未知')}`",
                f"\t- 生日: `{member.get('birthday', '未知')}`",
                f"\t- 城市: `{member.get('city', '未知')}`",
                f"\t- 职业: `{member.get('occupation', '未知')}`",
                f"\t- 爱好: `{member.get('hobby', '未知')}`",
                f"\t- 厌恶: `{member.get('disgust', '未知')}`",
                f"\t- 近期在做: `{member.get('working', '未知')}`",
                f"\t- 常用梗: `{member.get('memes', '未知')}`",
                f"\t- 口头禅: `{member.get('catchphrase', '未知')}`",
                f"\t- 发言风格: `{member.get('style', '未知')}`",
                f"\t- 人际关系: {relations_text}",
                "\n"
            ]
            active_member_text += '\n'.join(member_persona)

        active_member_text = active_member_text.strip('\n')

    if not old_messages:
        messages_text = "```\n(空)\n```"

    else:
        to_llm = []
        tasks: list[asyncio.Task] = [asyncio.create_task(msg.to_llm(wait=True, timeout=15)) for msg in old_messages]
        await asyncio.gather(*tasks, return_exceptions=True)
        for msg in tasks:
            if not msg.done() or msg.cancelled() or msg.exception() is not None:
                continue

            to_llm.append(msg.result())

        if not to_llm:
            messages_text = "```\n(空)\n```"

        else:
            messages_text = f"```json\n{json.dumps(to_llm, ensure_ascii=False, indent=2)}\n```"

    if not news:
        news_text = "**无**"

    else:
        news_lines = []
        for title, content in news.items():
            news_lines.append(f"- {title}: {content}\n")

        news_text = '\n'.join(news_lines)

    output_prompt = f"""\
**Agent 触发了 Context 刷新**  
以下是新 Context 初始化信息，你需要根据信息继续群聊  


# 群信息

## 基本信息

群号: `{group_id}`  
群名: `{group_name}`  
群主: `{str(group_owner).replace('\n', "; ")}`  
管理员: `{admins_text}`  
群机器人: `{robots_text}`  


## 其他信息

{group_summary if group_summary else "**无**"}  


## 活跃群员

{active_member_text}  


## 相关新闻、资讯

{news_text}


## 近期聊天记录

{messages_text}  

---

> 以上为 Agent Context 刷新自动生成的基本信息  {new_prompt}    


# 重新初始化流程

1. 调用 `load_external_tool` 加载缺失的所需外部工具
2. 调用 `load_skill` 检查 SKILL 是否缺失所需内容，尤其是 `group_chat`，必须保证该 SKILL 内容完整
3. 调用 `get_skill_content` 加载/更新所需的 SKILL 内容
4. 调用 `get_tasks_list` 查看当前任务状态
5. 根据信息，继续群聊任务  
"""
    return f"{output_prompt}\n\n\n"