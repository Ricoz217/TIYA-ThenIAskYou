from pathlib import Path


WEBSEARCH_DIR = (
    Path(__file__).parents[1] / "src" / "TIYA" / "agent_tools" / "websearch"
)


def test_websearch_root_contains_only_skill_tools_and_scripts() -> None:
    assert (WEBSEARCH_DIR / "SKILL.md").is_file()
    assert (WEBSEARCH_DIR / "tools" / "__init__.py").is_file()
    assert (WEBSEARCH_DIR / "scripts").is_dir()
    assert not (WEBSEARCH_DIR / "scripts" / "__init__.py").exists()
    assert list(WEBSEARCH_DIR.glob("*.py")) == []


def test_provider_implementations_live_in_a_non_package_subdirectory() -> None:
    tools_dir = WEBSEARCH_DIR / "tools"
    providers_dir = tools_dir / "providers"

    assert providers_dir.is_dir()
    assert not (providers_dir / "__init__.py").exists()
    assert {path.name for path in providers_dir.glob("*.py")} == {
        "deepseek.py",
        "deepseek_provider.py",
        "crawl4ai_provider.py",
        "httpx_provider.py",
        "playwright_provider.py",
        "tavily_provider.py",
    }
    assert not (tools_dir / "providers.py").exists()
    assert (tools_dir / "provider_protocols.py").is_file()
