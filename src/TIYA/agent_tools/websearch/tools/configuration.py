"""Environment-backed construction of the provider-neutral web-search runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from .errors import OptionalProviderDependencyError
from .identity import IdentityStore
from .runtime import WebSearchRuntime
from .providers.crawl4ai_provider import (
    Crawl4AIProvider,
    crawl4ai_available,
)
from .providers.deepseek_provider import DeepSeekProvider
from .providers.httpx_provider import HTTPXProvider
from .providers.playwright_provider import PlaywrightProvider
from .providers.tavily_provider import TavilyProvider


_BROWSER_CHANNELS = frozenset(
    {
        "chrome",
        "chrome-beta",
        "chrome-dev",
        "chrome-canary",
        "msedge",
        "msedge-beta",
        "msedge-dev",
        "msedge-canary",
    }
)


@dataclass(frozen=True, slots=True)
class WebSearchSettings:
    """Configuration for the standard local WebSearch Runtime assembly."""

    deepseek_api_key: str | None = None
    tavily_api_key: str | None = None
    download_dir: Path = Path("outputs/websearch/downloads")
    upload_dir: Path = Path("outputs/websearch/uploads")
    profile_dir: Path = Path("data/websearch/profiles")
    browser_type: str = "chromium"
    browser_channel: str | None = None
    browser_headless: bool = True
    reduce_automation_signals: bool = False
    navigation_timeout: float = 30.0
    enable_crawl4ai: bool = False
    crawl4ai_stealth: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "download_dir", Path(self.download_dir).resolve())
        object.__setattr__(self, "upload_dir", Path(self.upload_dir).resolve())
        object.__setattr__(self, "profile_dir", Path(self.profile_dir).resolve())
        if self.browser_type not in {"chromium", "firefox", "webkit"}:
            raise ValueError("browser_type must be chromium, firefox, or webkit")
        if self.browser_channel is not None:
            channel = self.browser_channel.strip().casefold()
            if channel not in _BROWSER_CHANNELS:
                raise ValueError(
                    "browser_channel must name a supported Chrome or Edge channel"
                )
            if self.browser_type != "chromium":
                raise ValueError("browser_channel requires browser_type='chromium'")
            object.__setattr__(self, "browser_channel", channel)
        if self.navigation_timeout <= 0:
            raise ValueError("navigation_timeout must be positive")

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> WebSearchSettings:
        """Load a dotenv file, with process environment values taking precedence."""

        path = Path(env_file).resolve()
        file_values = dotenv_values(path) if path.is_file() else {}

        def value(name: str) -> str | None:
            raw = os.environ.get(name, file_values.get(name))
            if not isinstance(raw, str) or not raw.strip():
                return None
            return raw.strip()

        def boolean(name: str, default: bool) -> bool:
            raw = value(name)
            if raw is None:
                return default
            normalized = raw.casefold()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{name} must be true or false")

        def relative_path(name: str, default: str) -> Path:
            configured = Path(value(name) or default)
            if not configured.is_absolute():
                configured = path.parent / configured
            return configured.resolve()

        timeout = value("TIYA_WEBSEARCH_NAVIGATION_TIMEOUT")
        return cls(
            deepseek_api_key=value("DEEPSEEK_API_KEY"),
            tavily_api_key=value("TAVILY_API_KEY"),
            download_dir=relative_path(
                "TIYA_WEBSEARCH_DOWNLOAD_DIR", "outputs/websearch/downloads"
            ),
            upload_dir=relative_path(
                "TIYA_WEBSEARCH_UPLOAD_DIR", "outputs/websearch/uploads"
            ),
            profile_dir=relative_path(
                "TIYA_WEBSEARCH_PROFILE_DIR", "data/websearch/profiles"
            ),
            browser_type=value("TIYA_WEBSEARCH_BROWSER") or "chromium",
            browser_channel=value("TIYA_WEBSEARCH_BROWSER_CHANNEL"),
            browser_headless=boolean("TIYA_WEBSEARCH_HEADLESS", True),
            reduce_automation_signals=boolean(
                "TIYA_WEBSEARCH_REDUCE_AUTOMATION_SIGNALS", False
            ),
            navigation_timeout=float(timeout) if timeout is not None else 30.0,
            enable_crawl4ai=boolean(
                "TIYA_WEBSEARCH_ENABLE_CRAWL4AI", False
            ),
            crawl4ai_stealth=boolean(
                "TIYA_WEBSEARCH_CRAWL4AI_STEALTH", False
            ),
        )


def build_websearch_runtime(settings: WebSearchSettings) -> WebSearchRuntime:
    """Register standard providers without adding hidden routing or fallback policy."""

    runtime = WebSearchRuntime()
    identities = IdentityStore(settings.profile_dir)
    if settings.deepseek_api_key is not None:
        runtime.register(DeepSeekProvider.from_api_key(settings.deepseek_api_key))
    if settings.tavily_api_key is not None:
        runtime.register(TavilyProvider.from_api_key(settings.tavily_api_key))
    if settings.enable_crawl4ai:
        if not crawl4ai_available():
            raise OptionalProviderDependencyError(
                "Crawl4AI",
                "install tools/providers/requirements-crawl4ai.txt",
            )
        runtime.register(
            Crawl4AIProvider(
                identity_store=identities,
                browser_type=settings.browser_type,
                headless=settings.browser_headless,
                enable_stealth=settings.crawl4ai_stealth,
                page_timeout=settings.navigation_timeout,
            )
        )
    runtime.register(HTTPXProvider(settings.download_dir))
    launch_options: dict[str, object] = {}
    if settings.browser_channel is not None:
        launch_options["channel"] = settings.browser_channel
    if settings.reduce_automation_signals:
        launch_options["ignore_default_args"] = ["--enable-automation"]
        launch_options["args"] = [
            "--disable-blink-features=AutomationControlled"
        ]
    runtime.register(
        PlaywrightProvider(
            browser_type=settings.browser_type,
            headless=settings.browser_headless,
            navigation_timeout=settings.navigation_timeout,
            identity_store=identities,
            allowed_upload_roots=(settings.upload_dir, settings.download_dir),
            launch_options=launch_options,
        )
    )
    return runtime


__all__ = ["WebSearchSettings", "build_websearch_runtime"]
