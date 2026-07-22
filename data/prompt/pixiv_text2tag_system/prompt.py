"""
色图模块的自然语言转 tag 系统提示词
"""
__version__ = "0.1.0"
from TIYA.agent.agent_prompt import parse_prompt_from_markdown


def prompt():
    return parse_prompt_from_markdown("pixiv_text2tag_system.md")