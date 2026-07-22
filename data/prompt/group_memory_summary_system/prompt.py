"""
QQ群的自动记忆总结系统提示词
"""
__version__ = "0.1.0"

from TIYA.agent.agent_prompt import parse_prompt_from_markdown

def prompt():
    return parse_prompt_from_markdown("group_memory_summary_system.md")