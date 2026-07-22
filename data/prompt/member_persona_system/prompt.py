"""
群员画像，用于生成群员的身份信息
"""
__version__ = "0.1.0"

from TIYA.agent.agent_prompt import parse_prompt_from_markdown

def prompt():
    return parse_prompt_from_markdown("member_persona_system.md")