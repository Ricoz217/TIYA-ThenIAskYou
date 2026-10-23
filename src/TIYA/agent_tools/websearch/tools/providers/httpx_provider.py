"""Lightweight, bounded HTTP fetching, mapping, and downloads."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import xml.etree.ElementTree as element_tree
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

from ..errors import ProviderRequestError
from ..models import (
    Capability,
    ContentFormat,
    DownloadArtifact,
    DownloadRequest,
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


_DEFAULT_DOCUMENT_LIMIT = 5 * 1024 * 1024
_DEFAULT_MAP_PAGE_LIMIT = 50
_MAX_REDIRECTS = 10
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_TEXTUAL_APPLICATION_TYPES = {
    "application/json",
    "application/ld+json",
    "application/xhtml+xml",
    "application/xml",
}
_USER_AGENT = "Mozilla/5.0 (compatible; TIYA-WebSearch/1.0)"
_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
_INTERACTION_TEXT_EXCLUDED_PARENTS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
}


class HTTPXProvider:
    """Read ordinary HTTP pages and files without launching a browser."""

    provider_id = "httpx"
    profile = ProviderProfile(
        provider_id=provider_id,
        display_name="HTTPX Direct Web",
        description=(
            "Lightweight direct HTTP fetching, same-origin site mapping, "
            "and bounded binary downloads."
        ),
        capabilities=frozenset(
            {Capability.FETCH, Capability.MAP, Capability.DOWNLOAD}
        ),
        strengths=(
            "Reads static HTML and text without a browser process",
            (
                "Extracts readable content, links, DOM media, inline CSS "
                "backgrounds, and social-card images from known URLs"
            ),
            "Preserves media discovery source, nearby context, links, and dimensions",
            "Maps same-origin links with explicit depth and page limits",
            "Streams binary downloads with size and integrity metadata",
        ),
        limitations=(
            "Does not execute JavaScript or perform page interactions",
            "Does not use browser identities, cookies, or authenticated profiles",
            "Private-network hostnames and obvious local addresses are rejected",
        ),
    )

    def __init__(
        self,
        download_dir: str | Path,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float | httpx.Timeout = 60.0,
        max_document_bytes: int = _DEFAULT_DOCUMENT_LIMIT,
        max_map_pages: int = _DEFAULT_MAP_PAGE_LIMIT,
        allow_private_network: bool = False,
    ) -> None:
        if max_document_bytes <= 0:
            raise ValueError("max_document_bytes must be positive")
        if max_map_pages <= 0:
            raise ValueError("max_map_pages must be positive")
        self._download_dir = Path(download_dir).resolve()
        self._max_document_bytes = max_document_bytes
        self._max_map_pages = max_map_pages
        self._allow_private_network = allow_private_network
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def fetch(self, request: FetchRequest) -> WebDocument:
        """Fetch and normalize one static HTTP document."""

        self._require_static_request(request)
        response, body = await self._read_bounded(
            request.url,
            operation=Capability.FETCH,
            maximum=self._max_document_bytes,
        )
        return _document_from_response(
            request=request,
            response=response,
            body=body,
            provider=self.provider_id,
        )

    async def map(self, request: MapRequest) -> MapResponse:
        """Discover a bounded set of same-origin links by breadth-first fetch."""

        if request.identity is not None:
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.MAP,
                message=(
                    "HTTPX map cannot use a browser identity; choose a browser "
                    "or crawler provider"
                ),
                retryable=False,
            )

        root = _normalize_url(request.url, request.url)
        root_origin = _origin(root)
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        queued = {root}
        fetched: set[str] = set()
        discovered: dict[str, Link] = {}
        failures = 0

        while (
            queue
            and len(fetched) < self._max_map_pages
            and len(discovered) < request.max_results
        ):
            page_url, page_depth = queue.popleft()
            if page_url in fetched:
                continue
            fetched.add(page_url)
            try:
                response, body = await self._read_bounded(
                    page_url,
                    operation=Capability.MAP,
                    maximum=self._max_document_bytes,
                )
                content_type = _content_type(response)
                if not _is_textual(content_type):
                    continue
                links = _links_from_markup(
                    _decode(response, body),
                    str(response.url),
                    content_type=content_type,
                )
                if page_depth == 0:
                    root_origin = _origin(str(response.url))
            except ProviderRequestError:
                if page_url == root:
                    raise
                failures += 1
                continue

            link_depth = page_depth + 1
            if link_depth > request.max_depth:
                continue
            for link in links:
                is_internal = _origin(link.url) == root_origin
                normalized = Link(
                    url=link.url,
                    text=link.text,
                    title=link.title,
                    internal=is_internal,
                    metadata={"depth": link_depth},
                )
                if (
                    (is_internal or request.include_external)
                    and len(discovered) < request.max_results
                    and _matches_query(normalized, request.query)
                ):
                    discovered.setdefault(normalized.url, normalized)
                if (
                    is_internal
                    and link_depth < request.max_depth
                    and normalized.url not in queued
                ):
                    queued.add(normalized.url)
                    queue.append((normalized.url, link_depth))

        return MapResponse(
            root_url=request.url,
            provider=self.provider_id,
            links=tuple(discovered.values()),
            metadata={
                "pages_fetched": len(fetched),
                "fetch_failures": failures,
                "max_depth": request.max_depth,
                "page_limit": self._max_map_pages,
            },
        )

    async def download(self, request: DownloadRequest) -> DownloadArtifact:
        """Download one resource while enforcing the request's size bound."""

        explicit_name = None
        if request.filename is not None:
            explicit_name = _validate_explicit_filename(request.filename)

        output_path: Path | None = None
        output_file: BinaryIO | None = None
        try:
            async with self._stream_response(
                request.url, Capability.DOWNLOAD
            ) as response:
                content_length = _content_length(response)
                if (
                    content_length is not None
                    and content_length > request.max_bytes
                ):
                    raise _size_error(
                        Capability.DOWNLOAD,
                        content_length,
                        request.max_bytes,
                    )

                filename = explicit_name or _filename_from_url(str(response.url))
                await asyncio.to_thread(
                    self._download_dir.mkdir, parents=True, exist_ok=True
                )
                reserved_path, reserved_file = await asyncio.to_thread(
                    _open_unique, self._download_dir, filename
                )
                output_path, output_file = reserved_path, reserved_file

                digest = hashlib.sha256()
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > request.max_bytes:
                        raise _size_error(
                            Capability.DOWNLOAD, size, request.max_bytes
                        )
                    digest.update(chunk)
                    await asyncio.to_thread(reserved_file.write, chunk)

                await asyncio.to_thread(reserved_file.close)
                output_file = None
                metadata: dict[str, object] = {
                    "status_code": response.status_code,
                }
                if content_length is not None:
                    metadata["content_length"] = content_length
                return DownloadArtifact(
                    source_url=request.url,
                    final_url=str(response.url),
                    path=reserved_path,
                    content_type=_content_type(response),
                    size=size,
                    sha256=digest.hexdigest(),
                    provider=self.provider_id,
                    metadata=metadata,
                )
        except ProviderRequestError:
            await _discard(output_path, output_file)
            raise
        except OSError as exc:
            await _discard(output_path, output_file)
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.DOWNLOAD,
                message=f"could not write download: {exc}",
                retryable=False,
            ) from exc

    async def aclose(self) -> None:
        """Close an internally created HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    def _require_static_request(self, request: FetchRequest) -> None:
        if request.render_javascript:
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.FETCH,
                message=(
                    "HTTPX fetch does not render JavaScript; choose a browser "
                    "or crawler provider"
                ),
                retryable=False,
            )
        if request.identity is not None:
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.FETCH,
                message=(
                    "HTTPX fetch cannot use a browser identity; choose a browser "
                    "or crawler provider"
                ),
                retryable=False,
            )

    async def _read_bounded(
        self,
        url: str,
        *,
        operation: Capability,
        maximum: int,
    ) -> tuple[httpx.Response, bytes]:
        async with self._stream_response(url, operation) as response:
            content_length = _content_length(response)
            if content_length is not None and content_length > maximum:
                raise _size_error(operation, content_length, maximum)
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > maximum:
                    raise _size_error(operation, len(body), maximum)
            return response, bytes(body)

    @asynccontextmanager
    async def _stream_response(
        self, url: str, operation: Capability
    ) -> AsyncIterator[httpx.Response]:
        current_url = url
        try:
            for _ in range(_MAX_REDIRECTS + 1):
                _validate_public_target(
                    current_url,
                    provider=self.provider_id,
                    operation=operation,
                    allow_private_network=self._allow_private_network,
                )
                async with self._client.stream(
                    "GET",
                    current_url,
                    follow_redirects=False,
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Accept": (
                            "text/html,application/xhtml+xml,text/plain,"
                            "*/*;q=0.5"
                        ),
                    },
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("Location")
                        if not location:
                            raise ProviderRequestError(
                                provider=self.provider_id,
                                operation=operation,
                                message="redirect response did not include a Location",
                                status_code=response.status_code,
                                retryable=False,
                            )
                        current_url = urljoin(str(response.url), location)
                        continue
                    response.raise_for_status()
                    yield response
                    return
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=operation,
                message=f"more than {_MAX_REDIRECTS} redirects for {url}",
                retryable=False,
            )
        except ProviderRequestError:
            raise
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=operation,
                message=f"HTTP {status_code} for {exc.request.url}",
                status_code=status_code,
                retryable=(
                    status_code in {408, 425, 429} or status_code >= 500
                ),
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=operation,
                message=str(exc),
                retryable=True,
            ) from exc


def _document_from_response(
    *,
    request: FetchRequest,
    response: httpx.Response,
    body: bytes,
    provider: str,
) -> WebDocument:
    content_type = _content_type(response)
    if not _is_textual(content_type):
        raise ProviderRequestError(
            provider=provider,
            operation=Capability.FETCH,
            message=(
                f"content type {content_type or 'unknown'} is not a readable "
                "document; use download for binary content"
            ),
            status_code=response.status_code,
            retryable=False,
        )
    text = _decode(response, body)
    final_url = str(response.url)
    is_html = _is_html(content_type, text)
    title = ""
    links: tuple[Link, ...] = ()
    media: tuple[MediaResource, ...] = ()
    interaction = None
    rendering_recommended = False

    if is_html:
        soup = BeautifulSoup(text, "html.parser")
        title = _page_title(soup)
        has_javascript_shell_evidence = _has_javascript_shell_evidence(soup)
        root = _meaningful_root(soup)
        links = _extract_links(root, final_url)
        media = _extract_media(root, final_url, document=soup)
        interaction = _detect_interaction(
            root=root,
            url=final_url,
            title=title,
            provider=provider,
        )
        plain_text = _plain_text(root)
        rendering_recommended = (
            len(plain_text) < 200 and has_javascript_shell_evidence
        )
        if request.preferred_format is ContentFormat.HTML:
            content = text
        elif request.preferred_format is ContentFormat.TEXT:
            content = plain_text
        else:
            content = _markdown(root, final_url)
    else:
        content = text

    metadata: dict[str, object] = {
        "content_length": len(body),
        "rendering_recommended": rendering_recommended,
    }
    charset = response.encoding
    if charset:
        metadata["encoding"] = charset
    return WebDocument(
        url=request.url,
        final_url=final_url,
        provider=provider,
        title=title,
        content=content,
        content_format=request.preferred_format,
        status_code=response.status_code,
        content_type=content_type,
        links=links,
        media=media,
        interaction_required=interaction,
        metadata=metadata,
    )


def _meaningful_root(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    for selector in ("article", "main", "[role='main']"):
        selected = soup.select_one(selector)
        if selected is not None:
            return selected
    removable = "script, style, noscript, template, svg, nav, footer, aside"
    for tag in soup.select(removable):
        tag.decompose()
    return soup.body or soup


def _page_title(soup: BeautifulSoup) -> str:
    if soup.title is not None:
        title = soup.title.get_text(" ", strip=True)
        if title:
            return title
    heading = soup.find("h1")
    return heading.get_text(" ", strip=True) if heading is not None else ""


def _plain_text(root: Tag | BeautifulSoup) -> str:
    text = root.get_text("\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _markdown(root: Tag | BeautifulSoup, base_url: str) -> str:
    rendered = _render_markdown(root, base_url)
    rendered = re.sub(r"[ \t]+\n", "\n", rendered)
    rendered = re.sub(r"\n[ \t]+", "\n", rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered.strip()


def _render_markdown(node: Any, base_url: str) -> str:
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    if not isinstance(node, (Tag, BeautifulSoup)):
        return ""
    name = node.name.casefold() if isinstance(node.name, str) else ""
    if name in {"script", "style", "noscript", "template", "svg"}:
        return ""
    if name == "img":
        source = _normalize_url(base_url, node.get("src"))
        if not source:
            return ""
        return f"![{node.get('alt', '')}]({source})"
    children = "".join(_render_markdown(child, base_url) for child in node.children)
    children = children.strip() if name not in {"html", "body"} else children
    if name == "a":
        href = _normalize_url(base_url, node.get("href"))
        return f"[{children or href}]({href})" if href else children
    if name == "br":
        return "\n"
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return f"\n\n{'#' * int(name[1])} {children}\n\n"
    if name == "p":
        return f"\n\n{children}\n\n"
    if name == "li":
        return f"\n- {children}"
    if name == "blockquote":
        quoted = "\n".join(f"> {line}" for line in children.splitlines())
        return f"\n\n{quoted}\n\n"
    if name == "pre":
        preformatted = node.get_text("\n", strip=False).strip()
        return f"\n\n```\n{preformatted}\n```\n\n"
    if name in {"article", "main", "section", "div", "ul", "ol"}:
        return f"\n{children}\n"
    return children


def _links_from_markup(
    text: str,
    base_url: str,
    *,
    content_type: str | None,
) -> tuple[Link, ...]:
    parser = "xml" if content_type and "xml" in content_type else "html.parser"
    if parser == "xml":
        links: list[Link] = []
        seen: set[str] = set()
        try:
            root = element_tree.fromstring(text)
        except element_tree.ParseError:
            return ()
        for location in root.iter():
            if location.tag.rsplit("}", 1)[-1].casefold() != "loc":
                continue
            url = _normalize_url(base_url, location.text)
            if url and url not in seen:
                seen.add(url)
                links.append(Link(url=url))
        return tuple(links)
    soup = BeautifulSoup(text, parser)
    return _extract_links(soup, base_url)


def _extract_links(root: Tag | BeautifulSoup, base_url: str) -> tuple[Link, ...]:
    links: list[Link] = []
    seen: set[str] = set()
    base_origin = _origin(base_url)
    for anchor in root.find_all("a", href=True):
        if not isinstance(anchor, Tag):
            continue
        url = _normalize_url(base_url, anchor.get("href"))
        if not url or url in seen:
            continue
        seen.add(url)
        links.append(
            Link(
                url=url,
                text=anchor.get_text(" ", strip=True),
                title=str(anchor.get("title") or ""),
                internal=_origin(url) == base_origin,
            )
        )
    return tuple(links)


def _extract_media(
    root: Tag | BeautifulSoup,
    base_url: str,
    *,
    document: BeautifulSoup | None = None,
) -> tuple[MediaResource, ...]:
    resources: list[MediaResource] = []
    seen: set[str] = set()
    kinds = {
        "img": MediaKind.IMAGE,
        "video": MediaKind.VIDEO,
        "audio": MediaKind.AUDIO,
        "source": MediaKind.OTHER,
    }
    for tag_name, kind in kinds.items():
        for tag in root.find_all(tag_name):
            if not isinstance(tag, Tag):
                continue
            source_attribute = "src"
            raw_url = tag.get(source_attribute)
            if not raw_url:
                source_attribute = "data-src"
                raw_url = tag.get(source_attribute)
            url = _normalize_url(base_url, raw_url)
            if not url or url in seen:
                continue
            seen.add(url)
            mime_type = tag.get("type")
            resolved_kind = _media_kind(tag, kind, mime_type)
            resources.append(
                MediaResource(
                    url=url,
                    kind=resolved_kind,
                    alt=str(
                        tag.get("alt")
                        or tag.get("aria-label")
                        or tag.get("title")
                        or ""
                    ),
                    mime_type=str(mime_type) if mime_type else None,
                    metadata=_element_media_metadata(
                        tag,
                        base_url,
                        source_attribute=source_attribute,
                    ),
                )
            )

    for tag in root.find_all(style=True):
        if not isinstance(tag, Tag):
            continue
        style = tag.get("style")
        if not isinstance(style, str) or "url(" not in style.casefold():
            continue
        for match in _CSS_URL.finditer(style):
            url = _normalize_url(base_url, match.group(2))
            if not url or url in seen:
                continue
            seen.add(url)
            resources.append(
                MediaResource(
                    url=url,
                    kind=MediaKind.IMAGE,
                    alt=str(tag.get("aria-label") or tag.get("title") or ""),
                    metadata=_css_media_metadata(tag),
                )
            )

    if document is not None:
        for tag in document.find_all("meta"):
            if not isinstance(tag, Tag):
                continue
            property_name = str(
                tag.get("property") or tag.get("name") or ""
            ).casefold()
            source = _social_image_source(property_name)
            if source is None:
                continue
            url = _normalize_url(base_url, tag.get("content"))
            if not url or url in seen:
                continue
            seen.add(url)
            resources.append(
                MediaResource(
                    url=url,
                    kind=MediaKind.IMAGE,
                    metadata={
                        "source": source,
                        "tag": "meta",
                        "property": property_name,
                    },
                )
            )
    return tuple(resources)


def _media_kind(
    tag: Tag, default: MediaKind, mime_type: object
) -> MediaKind:
    if default is not MediaKind.OTHER:
        return default
    normalized_type = str(mime_type or "").casefold()
    if normalized_type.startswith("image/"):
        return MediaKind.IMAGE
    if normalized_type.startswith("audio/"):
        return MediaKind.AUDIO
    if normalized_type.startswith("video/"):
        return MediaKind.VIDEO
    parent = tag.parent
    if isinstance(parent, Tag):
        parent_name = parent.name.casefold()
        if parent_name == "picture":
            return MediaKind.IMAGE
        if parent_name == "audio":
            return MediaKind.AUDIO
        if parent_name == "video":
            return MediaKind.VIDEO
    return default


def _element_media_metadata(
    tag: Tag,
    base_url: str,
    *,
    source_attribute: str,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "element",
        "tag": tag.name.casefold(),
        "source_attribute": source_attribute,
    }
    title = tag.get("title")
    if isinstance(title, str) and title.strip():
        metadata["title"] = title.strip()
    context = _media_context(tag)
    if context:
        metadata["context_text"] = context
    anchor = tag.find_parent("a", href=True)
    if isinstance(anchor, Tag):
        link_url = _normalize_url(base_url, anchor.get("href"))
        if link_url:
            metadata["link_url"] = link_url
    for dimension in ("width", "height"):
        value = _positive_integer(tag.get(dimension))
        if value is not None:
            metadata[dimension] = value
    return metadata


def _css_media_metadata(tag: Tag) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "css-background",
        "tag": tag.name.casefold(),
    }
    context = _media_context(tag)
    if context:
        metadata["context_text"] = context
    return metadata


def _media_context(tag: Tag) -> str:
    figure = tag.find_parent("figure")
    if isinstance(figure, Tag):
        caption = figure.find("figcaption")
        if isinstance(caption, Tag):
            text = caption.get_text(" ", strip=True)
            if text:
                return text[:500]
    return ""


def _positive_integer(value: object) -> int | None:
    if not isinstance(value, (str, int)):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _social_image_source(property_name: str) -> str | None:
    if property_name in {"og:image", "og:image:url", "og:image:secure_url"}:
        return "open-graph"
    if property_name in {"twitter:image", "twitter:image:src"}:
        return "twitter-card"
    return None


def _detect_interaction(
    *,
    root: Tag | BeautifulSoup,
    url: str,
    title: str,
    provider: str,
) -> InteractionRequired | None:
    searchable = _interaction_text(root, title)
    path = urlsplit(url).path.casefold()

    captcha_selectors = (
        ".g-recaptcha",
        ".h-captcha",
        ".cf-turnstile",
        "[data-sitekey]",
        "iframe[src*='recaptcha']",
        "iframe[src*='hcaptcha']",
        "iframe[src*='challenges.cloudflare.com']",
        "input[name='g-recaptcha-response']",
        "input[name='h-captcha-response']",
        "input[name='cf-turnstile-response']",
    )
    if any(root.select_one(selector) is not None for selector in captcha_selectors):
        return _interaction(
            InteractionKind.CAPTCHA,
            url,
            provider,
            "Page appears to require human CAPTCHA completion",
        )

    if any(
        phrase in searchable
        for phrase in (
            "verify you are human",
            "confirm you are human",
            "checking your browser before accessing",
            "access denied: human verification required",
        )
    ):
        return _interaction(
            InteractionKind.VERIFICATION,
            url,
            provider,
            "Page requires human verification before reading can continue",
        )

    if root.select_one("input[autocomplete='one-time-code']") is not None or any(
        phrase in searchable
        for phrase in (
            "two-factor authentication required",
            "multi-factor authentication required",
            "enter the code from your authenticator",
            "enter your one-time code",
        )
    ):
        return _interaction(
            InteractionKind.MFA,
            url,
            provider,
            "Page appears to require multi-factor authentication",
        )

    if path.endswith("/oauth/authorize") or "oauth consent" in searchable:
        return _interaction(
            InteractionKind.OAUTH,
            url,
            provider,
            "Page appears to require OAuth authorization",
        )

    if root.select_one("input[type='password']") is not None:
        return _interaction(
            InteractionKind.LOGIN,
            url,
            provider,
            "Page appears to require sign-in",
        )
    return None


def _interaction_text(root: Tag | BeautifulSoup, title: str) -> str:
    parts = [title]
    for value in root.find_all(string=True):
        parent = value.parent
        if isinstance(parent, Tag) and parent.name.casefold() in (
            _INTERACTION_TEXT_EXCLUDED_PARENTS
        ):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts).casefold()


def _interaction(
    kind: InteractionKind,
    url: str,
    provider: str,
    message: str,
) -> InteractionRequired:
    return InteractionRequired(
        kind=kind,
        url=url,
        provider=provider,
        message=message,
    )


def _has_javascript_shell_evidence(soup: BeautifulSoup) -> bool:
    scripts = soup.find_all("script")
    shell_markers = ("__next_data__", "__nuxt__", "webpack", "enable javascript")
    searchable = str(soup).casefold()
    return len(scripts) >= 3 or any(marker in searchable for marker in shell_markers)


def _matches_query(link: Link, query: str | None) -> bool:
    if not query:
        return True
    searchable = f"{link.url}\n{link.text}\n{link.title}".casefold()
    return query.casefold() in searchable


def _decode(response: httpx.Response, body: bytes) -> str:
    encoding = response.encoding or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _content_length(response: httpx.Response) -> int | None:
    raw_value = response.headers.get("Content-Length")
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value >= 0 else None


def _content_type(response: httpx.Response) -> str | None:
    content_type = response.headers.get("Content-Type")
    if content_type is None:
        return None
    return content_type.split(";", 1)[0].strip().casefold() or None


def _is_textual(content_type: str | None) -> bool:
    if content_type is None:
        return True
    return (
        content_type.startswith("text/")
        or content_type in _TEXTUAL_APPLICATION_TYPES
    )


def _is_html(content_type: str | None, text: str) -> bool:
    if content_type in {"text/html", "application/xhtml+xml"}:
        return True
    if content_type is not None and content_type != "text/plain":
        return False
    prefix = text.lstrip()[:256].casefold()
    return prefix.startswith(("<!doctype html", "<html", "<head", "<body"))


def _normalize_url(base_url: str, candidate: object) -> str:
    if not isinstance(candidate, str) or not candidate.strip():
        return ""
    try:
        resolved = urljoin(base_url, candidate.strip())
        parsed = urlsplit(resolved)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    return urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc, parsed.path or "/", parsed.query, "")
    )


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme.casefold(), (parsed.hostname or "").casefold(), parsed.port


def _validate_public_target(
    url: str,
    *,
    provider: str,
    operation: Capability,
    allow_private_network: bool,
) -> None:
    if allow_private_network:
        return
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    blocked_name = (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
    )
    blocked_address = False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        blocked_address = not address.is_global
    if blocked_name or blocked_address:
        raise ProviderRequestError(
            provider=provider,
            operation=operation,
            message=f"private network target is not allowed: {url}",
            retryable=False,
        )


def _validate_explicit_filename(filename: str) -> str:
    if (
        filename != filename.strip()
        or filename in {".", ".."}
        or Path(filename).name != filename
        or _INVALID_FILENAME.search(filename)
        or filename.endswith((".", " "))
        or _is_windows_reserved(filename)
    ):
        raise ProviderRequestError(
            provider=HTTPXProvider.provider_id,
            operation=Capability.DOWNLOAD,
            message="filename must be a safe file name without path components",
            retryable=False,
        )
    return filename


def _filename_from_url(url: str) -> str:
    candidate = unquote(urlsplit(url).path).rsplit("/", 1)[-1]
    candidate = _INVALID_FILENAME.sub("_", candidate).strip(" .")
    if not candidate:
        return "download"
    if _is_windows_reserved(candidate):
        candidate = f"_{candidate}"
    return candidate[:240].rstrip(" .") or "download"


def _is_windows_reserved(filename: str) -> bool:
    return filename.split(".", 1)[0].upper() in _WINDOWS_RESERVED


def _open_unique(directory: Path, filename: str) -> tuple[Path, BinaryIO]:
    base = Path(filename)
    for number in range(10_000):
        suffix = "" if number == 0 else f" ({number})"
        candidate = directory / f"{base.stem}{suffix}{base.suffix}"
        try:
            return candidate, candidate.open("xb")
        except FileExistsError:
            continue
    raise OSError("could not allocate a unique download filename")


async def _discard(path: Path | None, output_file: BinaryIO | None) -> None:
    if output_file is not None:
        await asyncio.to_thread(output_file.close)
    if path is not None:
        try:
            await asyncio.to_thread(path.unlink)
        except FileNotFoundError:
            pass


def _size_error(
    operation: Capability, received: int, maximum: int
) -> ProviderRequestError:
    return ProviderRequestError(
        provider=HTTPXProvider.provider_id,
        operation=operation,
        message=f"resource size {received} bytes exceeds limit {maximum} bytes",
        retryable=False,
    )


__all__ = ["HTTPXProvider"]
