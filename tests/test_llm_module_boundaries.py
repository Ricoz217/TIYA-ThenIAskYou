import ast
from pathlib import Path


def test_legacy_exports_are_the_same_classes():
    from TIYA import LLM_connect
    from TIYA.llm import config_model, model, tool_schema

    for name in (
        "Context",
        "Prompts",
        "ImagePrompt",
        "AskTask",
        "TokenUsage",
    ):
        assert getattr(LLM_connect, name) is getattr(model, name)
    assert LLM_connect.ChatConfig is config_model.ChatConfig
    assert LLM_connect.build_tool_payloads is tool_schema.build_tool_payloads


def test_internal_modules_do_not_import_the_facade():
    root = Path(__file__).parents[1] / "src/TIYA/llm"
    assert root.is_dir()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "LLM_connect" not in (node.module or ""), path


def test_deepcopy_preserves_duration_and_isolates_prompt_mutation():
    import copy
    from TIYA.LLM_connect import Context, Prompts, TextPrompt

    prompts = Prompts(TextPrompt("user", "original"), duration=2)
    copied = copy.deepcopy(prompts)
    copied[0].text = "changed"
    assert prompts[0].text == "original"
    assert copied.duration == 2
    context = Context(prompts)
    fork = copy.deepcopy(context)
    fork._messages[0].text = "fork"
    assert context._messages[0].text == "original"


def test_tool_snapshot_keeps_bound_callable_identity():
    import copy
    import threading
    from TIYA.llm.model import Context, ToolInput

    class Owner:
        def __init__(self):
            self.lock = threading.Lock()

        def tool(self, value: str):
            return value

    owner = Owner()
    prompt = ToolInput(owner.tool)
    snapshot = copy.deepcopy(Context(prompt))
    assert snapshot._tools[0].function is prompt.function
