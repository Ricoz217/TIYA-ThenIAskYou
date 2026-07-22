"""
QQ群的画像系统提示词，用于获取群总结性信息
"""
__version__ = "0.1.0"

from TIYA.agent.agent_prompt import parse_prompt_from_markdown

def prompt():
    return parse_prompt_from_markdown("group_persona_system.md")