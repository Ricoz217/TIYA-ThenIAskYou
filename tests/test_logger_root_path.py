from __future__ import annotations

import ast
from pathlib import Path

from TIYA.config import LOGS_DIR, ROOT_DIR


def test_logs_dir_is_root_anchored() -> None:
    assert LOGS_DIR == ROOT_DIR / "logs"
    assert LOGS_DIR.is_absolute()


def test_initiate_instance_uses_root_anchored_logs_dir() -> None:
    source = Path("src/TIYA/initiate_instance.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    logger_config_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LoggerConfig"
    ]

    assert logger_config_calls, "initiate_instance.py should configure the global logger"
    assert any(
        keyword.arg == "logs_dir"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "LOGS_DIR"
        for call in logger_config_calls
        for keyword in call.keywords
    )
