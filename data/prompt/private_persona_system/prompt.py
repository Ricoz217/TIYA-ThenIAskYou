from TIYA.agent.agent_prompt import parse_prompt_from_markdown


def prompt() -> str:
    return parse_prompt_from_markdown("private_persona_system.md")
