__version__ = "0.1.0"
from TIYA.agent.agent_prompt import parse_prompt_from_markdown

def prompt():
    return f"{parse_prompt_from_markdown("baseline.md")}\n\n"