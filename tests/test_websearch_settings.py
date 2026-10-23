from pathlib import Path

import pytest

from TIYA.agent_tools.websearch.tools import (
    Capability,
    OptionalProviderDependencyError,
    WebSearchSettings,
    build_websearch_runtime,
)


def test_settings_load_env_file_and_environment_takes_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=file-deepseek\n"
        "TAVILY_API_KEY=\n"
        "TIYA_WEBSEARCH_ENABLE_CRAWL4AI=true\n"
        "TIYA_WEBSEARCH_HEADLESS=false\n"
        "TIYA_WEBSEARCH_BROWSER_CHANNEL=chrome\n"
        "TIYA_WEBSEARCH_REDUCE_AUTOMATION_SIGNALS=true\n"
        "TIYA_WEBSEARCH_UPLOAD_DIR=data/uploads\n"
        "TIYA_WEBSEARCH_PROFILE_DIR=data/profiles\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-deepseek")

    settings = WebSearchSettings.from_env(env_file)

    assert settings.deepseek_api_key == "process-deepseek"
    assert settings.tavily_api_key is None
    assert settings.enable_crawl4ai is True
    assert settings.browser_headless is False
    assert settings.browser_channel == "chrome"
    assert settings.reduce_automation_signals is True
    assert settings.upload_dir == (tmp_path / "data/uploads").resolve()
    assert settings.profile_dir == (tmp_path / "data/profiles").resolve()


def test_runtime_builder_registers_local_and_configured_providers(
    tmp_path: Path,
) -> None:
    settings = WebSearchSettings(
        deepseek_api_key="deepseek-key",
        tavily_api_key="tavily-key",
        download_dir=tmp_path / "downloads",
        profile_dir=tmp_path / "profiles",
        enable_crawl4ai=True,
    )

    runtime = build_websearch_runtime(settings)

    assert runtime.provider_ids(Capability.SEARCH) == ("deepseek", "tavily")
    assert runtime.provider_ids(Capability.FETCH) == (
        "deepseek",
        "tavily",
        "crawl4ai",
        "httpx",
    )
    assert runtime.provider_ids(Capability.MAP) == (
        "tavily",
        "crawl4ai",
        "httpx",
    )
    assert runtime.provider_ids(Capability.DOWNLOAD) == ("httpx",)
    assert runtime.provider_ids(Capability.BROWSE) == ("playwright",)


def test_runtime_builder_configures_real_browser_channel_and_reduced_signals(
    tmp_path: Path,
) -> None:
    settings = WebSearchSettings(
        download_dir=tmp_path / "downloads",
        upload_dir=tmp_path / "uploads",
        profile_dir=tmp_path / "profiles",
        browser_channel="chrome",
        reduce_automation_signals=True,
    )

    runtime = build_websearch_runtime(settings)
    provider = runtime.unregister("playwright")

    assert provider._launch_options == {  # type: ignore[attr-defined]
        "channel": "chrome",
        "ignore_default_args": ["--enable-automation"],
        "args": ["--disable-blink-features=AutomationControlled"],
    }


def test_settings_reject_unsupported_browser_channel() -> None:
    with pytest.raises(ValueError, match="browser_channel"):
        WebSearchSettings(browser_channel="internet-explorer")


def test_runtime_builder_explains_missing_optional_crawl4ai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = WebSearchSettings(
        download_dir=tmp_path / "downloads",
        profile_dir=tmp_path / "profiles",
        enable_crawl4ai=True,
    )
    monkeypatch.setattr(
        "TIYA.agent_tools.websearch.tools.configuration.crawl4ai_available",
        lambda: False,
    )

    with pytest.raises(OptionalProviderDependencyError, match="Crawl4AI"):
        build_websearch_runtime(settings)
