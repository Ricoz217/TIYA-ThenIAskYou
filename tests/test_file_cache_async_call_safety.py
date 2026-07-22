import ast
from pathlib import Path


BLOCKING_APIS = {
    "add_file",
    "get_file_metadata",
    "get_file_path",
    "remove_file",
    "remove_fire",
    "renew",
}


def test_async_business_code_does_not_call_blocking_file_cache_apis() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "TIYA"
    violations: list[str] = []

    for source_file in source_root.rglob("*.py"):
        if source_file.name == "file_cache.py":
            continue

        tree = ast.parse(source_file.read_text(encoding="utf-8-sig"))
        imported_names: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "TIYA.file_cache":
                continue

            for imported in node.names:
                if imported.name in BLOCKING_APIS:
                    imported_names[imported.asname or imported.name] = imported.name

        if not imported_names:
            continue

        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue

            api_name = imported_names.get(node.func.id)
            if api_name is None:
                continue

            current: ast.AST = node
            owner: ast.FunctionDef | ast.AsyncFunctionDef | None = None
            while current in parents:
                current = parents[current]
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    owner = current
                    break

            if isinstance(owner, ast.AsyncFunctionDef):
                relative = source_file.relative_to(source_root.parent)
                violations.append(
                    f"{relative}:{node.lineno} {owner.name}() calls {api_name}()"
                )

    assert not violations, "同步文件缓存调用仍位于事件循环内:\n" + "\n".join(violations)
