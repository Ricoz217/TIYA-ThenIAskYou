"""Optional in-process Crawl4AI adapter for rendered fetch and site mapping."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from ..errors import InteractionRequiredError, ProviderRequestError
from ..identity import IdentityStore
from ..models import (
    Capability,
    ContentFormat,
    FetchRequest,
    InteractionKind,
    InteractionRequired,
    Link,
    MapRequest,
    MapResponse,
    MediaKind,
    MediaResource,
    ProviderProfile,
    WebDocument,
)


class _Backend(Protocol):
    async def crawl(
        self,
        url: str,
        *,
        identity_dir: Path | None,
        deep_crawl: Mapping[str, object] | None,
        render_javascript: bool,
    ) -> object: ...

    async def aclose(self) -> None: ...


def crawl4ai_available() -> bool:
    """Return whether the optional Crawl4AI distribution can be imported."""

    return importlib.util.find_spec("crawl4ai") is not None


class Crawl4AIBackend:
    """Lazy bridge to Crawl4AI so the core package has no hard dependency."""

    def __init__(
        self,
        *,
        browser_type: str = "chromium",
        headless: bool = True,
        enable_stealth: bool = False,
        page_timeout: float = 60.0,
    ) -> None:
        if browser_type not in {"chromium", "firefox", "webkit"}:
            raise ValueError("browser_type must be chromium, firefox, or webkit")
        if page_timeout <= 0:
            raise ValueError("page_timeout must be positive")
        self._browser_type = browser_type
        self._headless = headless
        self._enable_stealth = enable_stealth
        self._page_timeout_ms = round(page_timeout * 1000)

    async def crawl(
        self,
        url: str,
        *,
        identity_dir: Path | None,
        deep_crawl: Mapping[str, object] | None,
        render_javascript: bool,
    ) -> object:
        try:
            from crawl4ai import (  # type: ignore[import-untyped]
                AsyncWebCrawler,
                BrowserConfig,
                CrawlerRunConfig,
            )
            from crawl4ai.deep_crawling import (  # type: ignore[import-untyped]
                BFSDeepCrawlStrategy,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Crawl4AI is not installed; install the optional provider requirements"
            ) from exc

        persistent = identity_dir is not None
        browser_config = BrowserConfig(
            browser_type=self._browser_type,
            headless=self._headless,
            enable_stealth=self._enable_stealth,
            use_managed_browser=persistent,
            use_persistent_context=persistent,
            user_data_dir=str(identity_dir) if identity_dir is not None else None,
            verbose=False,
        )
        strategy = None
        if deep_crawl is not None:
            strategy = BFSDeepCrawlStrategy(**deep_crawl)
        run_config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            page_timeout=self._page_timeout_ms,
            process_in_browser=render_javascript,
            stream=False,
            verbose=False,
            log_console=False,
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            if deep_crawl is not None:
                return list(result)
            return result[0]

    async def aclose(self) -> None:
        """Each crawl owns and closes its crawler context."""


class Crawl4AIProvider:
    """Rendered extraction and bounded site traversal using Crawl4AI."""

    provider_id = "crawl4ai"
    profile = ProviderProfile(
        provider_id=provider_id,
        display_name="Crawl4AI",
        description=(
            "Optional local JavaScript crawler for rendered extraction and "
            "bounded multi-page site mapping."
        ),
        capabilities=frozenset({Capability.FETCH, Capability.MAP}),
        strengths=(
            "Extracts Markdown, HTML, links, and media from rendered pages",
            "Maps multiple pages with explicit depth and page-count bounds",
            "Can reuse a persistent identity profile and enable stealth mode",
        ),
        limitations=(
            "Heavy optional dependency with browsers and a large dependency tree",
            "Does not replace interactive browsing for login, OAuth, CAPTCHA, or MFA",
            "Local crawling is slower and less predictable than hosted extraction APIs",
        ),
    )

    def __init__(
        self,
        *,
        backend: _Backend | None = None,
        identity_store: IdentityStore | None = None,
        browser_type: str = "chromium",
        headless: bool = True,
        enable_stealth: bool = False,
        page_timeout: float = 60.0,
    ) -> None:
        self._identity_store = identity_store
        self._backend = backend or Crawl4AIBackend(
            browser_type=browser_type,
            headless=headless,
            enable_stealth=enable_stealth,
            page_timeout=page_timeout,
        )

    async def fetch(self, request: FetchRequest) -> WebDocument:
        """Render and normalize one page without exposing Crawl4AI objects."""

        result = await self._crawl(request, deep_crawl=None)
        return _document(result, request)

    async def map(self, request: MapRequest) -> MapResponse:
        """Traverse a bounded number of pages and return deduplicated links."""

        result = await self._crawl(
            request,
            deep_crawl={
                "max_depth": request.max_depth,
                "max_pages": request.max_results,
                "include_external": request.include_external,
            },
        )
        results = result if isinstance(result, Sequence) else [result]
        links: list[Link] = []
        seen: set[str] = {request.url}
        root_host = urlsplit(request.url).hostname
        query = request.query.casefold() if request.query else None
        for crawl_result in results:
            _ensure_success(
                crawl_result,
                Capability.MAP,
                identity=request.identity,
            )
            candidates = [_result_link(crawl_result, root_host)]
            candidates.extend(_links(crawl_result, root_host))
            for link in candidates:
                if link.url in seen:
                    continue
                if not request.include_external and link.internal is False:
                    continue
                searchable = f"{link.url} {link.text} {link.title}".casefold()
                if query and query not in searchable:
                    continue
                seen.add(link.url)
                links.append(link)
                if len(links) >= request.max_results:
                    break
            if len(links) >= request.max_results:
                break
        return MapResponse(
            root_url=request.url,
            provider=self.provider_id,
            links=tuple(links),
            metadata={"pages_crawled": len(results)},
        )

    async def aclose(self) -> None:
        await self._backend.aclose()

    async def _crawl(
        self,
        request: FetchRequest | MapRequest,
        *,
        deep_crawl: Mapping[str, object] | None,
    ) -> object:
        identity_dir = None
        if request.identity is not None:
            if self._identity_store is None:
                raise ProviderRequestError(
                    provider=self.provider_id,
                    operation=(
                        Capability.FETCH
                        if isinstance(request, FetchRequest)
                        else Capability.MAP
                    ),
                    message=(
                        "an identity was requested but no identity store is configured"
                    ),
                )
            identity_dir = self._identity_store.resolve(
                request.identity, create=True
            )
        try:
            return await self._backend.crawl(
                request.url,
                identity_dir=identity_dir,
                deep_crawl=deep_crawl,
                render_javascript=(
                    request.render_javascript
                    if isinstance(request, FetchRequest)
                    else True
                ),
            )
        except ProviderRequestError:
            raise
        except Exception as exc:
            operation = (
                Capability.FETCH
                if isinstance(request, FetchRequest)
                else Capability.MAP
            )
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=operation,
                message=str(exc),
                retryable=True,
            ) from exc


def _document(result: object, request: FetchRequest) -> WebDocument:
    _ensure_success(result, Capability.FETCH, identity=request.identity)
    html = _string_attr(result, "html")
    cleaned_html = _string_attr(result, "cleaned_html") or html
    markdown = getattr(result, "markdown", None)
    raw_markdown = getattr(markdown, "raw_markdown", None)
    if not isinstance(raw_markdown, str):
        raw_markdown = str(markdown) if markdown is not None else ""

    if request.preferred_format == ContentFormat.HTML:
        content = cleaned_html
    elif request.preferred_format == ContentFormat.TEXT:
        content = BeautifulSoup(cleaned_html, "html.parser").get_text(
            "\n", strip=True
        )
    else:
        content = raw_markdown

    metadata = _mapping_attr(result, "metadata")
    title = metadata.get("title", "")
    if not isinstance(title, str):
        title = ""
    final_url = _optional_http_url(getattr(result, "redirected_url", None))
    result_url = _optional_http_url(getattr(result, "url", None)) or request.url
    visible_url = final_url or result_url
    response_headers = _mapping_attr(result, "response_headers")
    content_type = _header(response_headers, "content-type")
    if content_type is not None:
        content_type = content_type.split(";", 1)[0].strip() or None
    interaction = _detect_interaction(
        html=html,
        url=visible_url,
        title=title,
        identity=request.identity,
    )
    document_metadata: dict[str, object] = dict(metadata)
    stats = getattr(result, "crawl_stats", None)
    if isinstance(stats, Mapping):
        document_metadata["crawl_stats"] = dict(stats)
    return WebDocument(
        url=request.url,
        final_url=visible_url,
        provider=Crawl4AIProvider.provider_id,
        title=title,
        content=content,
        content_format=request.preferred_format,
        status_code=_integer_or_none(getattr(result, "status_code", None)),
        content_type=content_type,
        links=_links(result, urlsplit(visible_url).hostname),
        media=_media(result),
        interaction_required=interaction,
        metadata=document_metadata,
    )


def _ensure_success(
    result: object,
    operation: Capability,
    *,
    identity: Any = None,
) -> None:
    if getattr(result, "success", False) is True:
        return
    message = getattr(result, "error_message", None)
    message_text = message if isinstance(message, str) else ""
    html = _string_attr(result, "html")
    url = _optional_http_url(getattr(result, "url", None))
    if url is not None:
        interaction = _detect_interaction(
            html=f"{html}\n{message_text}",
            url=url,
            title="",
            identity=identity,
        )
        if interaction is not None:
            raise InteractionRequiredError(
                interaction=interaction,
                operation=operation,
                status_code=_integer_or_none(
                    getattr(result, "status_code", None)
                ),
            )
    raise ProviderRequestError(
        provider=Crawl4AIProvider.provider_id,
        operation=operation,
        message=message_text or "crawl failed",
        status_code=_integer_or_none(getattr(result, "status_code", None)),
        retryable=True,
    )


def _links(result: object, current_host: str | None) -> tuple[Link, ...]:
    raw_groups = _mapping_attr(result, "links")
    links: list[Link] = []
    seen: set[str] = set()
    for group, internal in (("internal", True), ("external", False)):
        raw_links = raw_groups.get(group, [])
        if not isinstance(raw_links, Sequence):
            continue
        for raw in raw_links:
            if not isinstance(raw, Mapping):
                continue
            url = _optional_http_url(raw.get("href") or raw.get("url"))
            if url is None or url in seen:
                continue
            seen.add(url)
            links.append(
                Link(
                    url=url,
                    text=_mapping_text(raw, "text"),
                    title=_mapping_text(raw, "title"),
                    internal=(
                        urlsplit(url).hostname == current_host
                        if current_host is not None
                        else internal
                    ),
                    metadata={
                        key: value
                        for key, value in raw.items()
                        if key not in {"href", "url", "text", "title"}
                    },
                )
            )
    return tuple(links)


def _result_link(result: object, root_host: str | None) -> Link:
    url = _optional_http_url(getattr(result, "url", None))
    if url is None:
        raise ProviderRequestError(
            provider=Crawl4AIProvider.provider_id,
            operation=Capability.MAP,
            message="crawl result did not contain a valid URL",
        )
    metadata = _mapping_attr(result, "metadata")
    title = metadata.get("title", "")
    return Link(
        url=url,
        title=title if isinstance(title, str) else "",
        internal=urlsplit(url).hostname == root_host,
    )


def _media(result: object) -> tuple[MediaResource, ...]:
    raw_groups = _mapping_attr(result, "media")
    resources: list[MediaResource] = []
    kinds = {
        "images": MediaKind.IMAGE,
        "videos": MediaKind.VIDEO,
        "audios": MediaKind.AUDIO,
    }
    for group, kind in kinds.items():
        raw_resources = raw_groups.get(group, [])
        if not isinstance(raw_resources, Sequence):
            continue
        for raw in raw_resources:
            if not isinstance(raw, Mapping):
                continue
            url = _optional_http_url(raw.get("src") or raw.get("url"))
            if url is None:
                continue
            mime_type = raw.get("type")
            resources.append(
                MediaResource(
                    url=url,
                    kind=kind,
                    alt=_mapping_text(raw, "alt"),
                    mime_type=mime_type if isinstance(mime_type, str) else None,
                    metadata={
                        key: value
                        for key, value in raw.items()
                        if key not in {"src", "url", "alt", "type"}
                    },
                )
            )
    return tuple(resources)


def _detect_interaction(
    *,
    html: str,
    url: str,
    title: str,
    identity: Any,
) -> InteractionRequired | None:
    searchable = f"{url}\n{title}\n{html}".casefold()
    kind = None
    message = ""
    if any(
        marker in searchable
        for marker in (
            "g-recaptcha",
            "hcaptcha",
            "cf-chl-",
            "turnstile",
            "captcha",
        )
    ):
        kind = InteractionKind.CAPTCHA
        message = "Page appears to require human CAPTCHA completion"
    elif any(
        marker in searchable
        for marker in ("anti-bot", "access denied", "verify you are human")
    ):
        kind = InteractionKind.VERIFICATION
        message = "Page requires human verification before crawling can continue"
    elif "type=\"password\"" in searchable or "type='password'" in searchable:
        kind = InteractionKind.LOGIN
        message = "Page appears to require sign-in"
    elif any(marker in searchable for marker in ("two-factor", "2fa", "mfa")):
        kind = InteractionKind.MFA
        message = "Page appears to require multi-factor authentication"
    elif "/oauth/authorize" in searchable or "oauth consent" in searchable:
        kind = InteractionKind.OAUTH
        message = "Page appears to require OAuth authorization"
    if kind is None:
        return None
    return InteractionRequired(
        kind=kind,
        url=url,
        provider=Crawl4AIProvider.provider_id,
        message=message,
        identity=identity,
    )


def _mapping_attr(value: object, name: str) -> Mapping[str, Any]:
    attribute = getattr(value, name, None)
    return attribute if isinstance(attribute, Mapping) else {}


def _mapping_text(value: Mapping[str, object], name: str) -> str:
    item = value.get(name)
    return item if isinstance(item, str) else ""


def _string_attr(value: object, name: str) -> str:
    item = getattr(value, name, None)
    return item if isinstance(item, str) else ""


def _optional_http_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlsplit(value)
    return value if parsed.scheme in {"http", "https"} and parsed.hostname else None


def _integer_or_none(value: object) -> int | None:
    return value if type(value) is int else None


def _header(headers: Mapping[str, object], name: str) -> str | None:
    for key, value in headers.items():
        if isinstance(key, str) and key.casefold() == name.casefold():
            return value if isinstance(value, str) else None
    return None


__all__ = ["Crawl4AIBackend", "Crawl4AIProvider", "crawl4ai_available"]
