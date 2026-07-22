"""
QQ群的全库记忆搜索器(删除用)
"""
__version__ = "0.1.0"

from TIYA.agent.agent_prompt import parse_prompt_from_markdown

def prompt():
    return parse_prompt_from_markdown("group_remove_memory_system.md")