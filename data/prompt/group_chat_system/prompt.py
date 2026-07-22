"""
群聊 Agent 主控的系统提示词。负责群聊的主逻辑，但使用另一个LLM进行独立的发言
"""
__version__ = "0.2.0"

from TIYA.agent.agent_prompt import parse_prompt_from_markdown

def prompt():
    return f"\n\n{parse_prompt_from_markdown(r"group_chat_system.md")}"