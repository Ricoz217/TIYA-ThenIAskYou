"""
群聊发言LLM的系统提示词，只负责发言这一个任务
"""
__version__ = "0.2.0"
from TIYA.agent.agent_prompt import parse_prompt_from_markdown
from TIYA.model.character import get_character

def prompt(character: str):
    character = get_character(character)
    basic = parse_prompt_from_markdown(r"group_speak_system.md")
    char = f"# 人格信息\n\n{character.personality}"
    return basic.replace("# 人格信息", char)
