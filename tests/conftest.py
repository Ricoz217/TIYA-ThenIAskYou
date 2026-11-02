import pytest


@pytest.fixture(autouse=True)
def isolate_llm_accounting(request, tmp_path, monkeypatch):
    """Communication tests must not write simulated charges or files to user storage."""
    if not (
        request.node.path.name.startswith("test_llm_")
        or request.node.path.name == "test_agent_tool_protocol_fallback.py"
    ):
        yield
        return
    import asyncio
    from TIYA import file_cache
    from TIYA.llm import chat, clients
    from TIYA.llm.usage_compat import LLMUsage

    monkeypatch.setattr(chat, "_GLOBAL_USAGE", LLMUsage(tmp_path / "usage.sqlite3"))
    monkeypatch.setattr(chat, "DATA_DIR", tmp_path)
    monkeypatch.setattr(clients, "DATA_DIR", tmp_path)
    cache = file_cache.FileCache(tmp_path / "files")
    monkeypatch.setattr(file_cache, "_DEFAULT_CACHE", cache)
    yield
    asyncio.run(cache.close())
