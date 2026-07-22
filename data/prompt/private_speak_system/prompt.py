from TIYA.agent.agent_prompt import parse_prompt_from_markdown
from TIYA.model.character import get_character


def prompt(character: str) -> str:
    role = get_character(character)
    basic = parse_prompt_from_markdown("private_speak_system.md")
    return basic.replace("# 人格信息", f"# 人格信息\n\n{role.private_personality}")
