"""Lightweight, bounded HTTP fetching, mapping, and downloads."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import re
import shutil
import xml.etree.ElementTree as element_tree
from collections import deque
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
_DEFAULT_DOWNLOAD_PART_SIZE = 8 * 1024 * 1024
_DEFAULT_DOWNLOAD_RETRY_BACKOFF = 0.25
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
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)
_UNSATISFIED_CONTENT_RANGE = re.compile(r"bytes \*/(\d+)", re.IGNORECASE)
_DOWNLOAD_STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class _DownloadProbe:
    final_url: str
    status_code: int
    content_type: str | None
    total_size: int | None
    range_supported: bool
    validator: str | None
    retry_count: int


@dataclass(frozen=True, slots=True)
class _ByteRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True)
class _DownloadWorkspace:
    target: Path
    state: Path
    parts: tuple[Path, ...]
    assembly: Path
    resumed: bool


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
            (
                "Streams bounded binary downloads with automatic retries, "
                "persistent resume state, concurrent byte ranges, and "
                "integrity metadata"
            ),
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
        download_part_size: int = _DEFAULT_DOWNLOAD_PART_SIZE,
        download_retry_backoff: float = _DEFAULT_DOWNLOAD_RETRY_BACKOFF,
    ) -> None:
        if max_document_bytes <= 0:
            raise ValueError("max_document_bytes must be positive")
        if max_map_pages <= 0:
            raise ValueError("max_map_pages must be positive")
        if download_part_size <= 0:
            raise ValueError("download_part_size must be positive")
        if download_retry_backoff < 0:
            raise ValueError("download_retry_backoff must be non-negative")
        self._download_dir = Path(download_dir).resolve()
        self._max_document_bytes = max_document_bytes
        self._max_map_pages = max_map_pages
        self._allow_private_network = allow_private_network
        self._download_part_size = download_part_size
        self._download_retry_backoff = download_retry_backoff
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
        """Download with bounded retries, persistent resume, and byte ranges."""

        explicit_name = (
            _validate_explicit_filename(request.filename)
            if request.filename is not None
            else None
        )
        probe = await self._probe_download(request)
        if probe.total_size is not None and probe.total_size > request.max_bytes:
            raise _size_error(
                Capability.DOWNLOAD,
                probe.total_size,
                request.max_bytes,
            )

        filename = explicit_name or _filename_from_url(probe.final_url)
        ranges = _download_ranges(
            probe.total_size,
            range_supported=probe.range_supported,
            maximum=request.max_parallel_parts,
            part_size=self._download_part_size,
        )
        state_data = _download_state(
            request=request,
            probe=probe,
            ranges=ranges,
        )
        await asyncio.to_thread(
            self._download_dir.mkdir, parents=True, exist_ok=True
        )
        workspace = await asyncio.to_thread(
            _prepare_download_workspace,
            self._download_dir,
            filename,
            state_data,
            request.resume and probe.validator is not None,
        )

        retry_count = probe.retry_count
        resumed = workspace.resumed
        completed_path: Path | None = None
        status_code = probe.status_code
        content_type = probe.content_type
        try:
            if probe.total_size == 0:
                await asyncio.to_thread(workspace.parts[0].touch, exist_ok=True)
                completed_path = workspace.parts[0]
            elif probe.range_supported and probe.total_size is not None:
                raw_outcomes = await asyncio.gather(
                    *(
                        self._download_range(
                            url=probe.final_url,
                            target=part_path,
                            byte_range=byte_range,
                            total_size=probe.total_size,
                            validator=probe.validator,
                            request=request,
                        )
                        for byte_range, part_path in zip(
                            ranges, workspace.parts, strict=True
                        )
                    ),
                    return_exceptions=True,
                )
                outcomes: list[tuple[bool, int]] = []
                for outcome in raw_outcomes:
                    if isinstance(outcome, BaseException):
                        raise outcome
                    outcomes.append(outcome)
                resumed = resumed or any(item[0] for item in outcomes)
                retry_count += sum(item[1] for item in outcomes)
                status_code = 206
                if len(workspace.parts) == 1:
                    completed_path = workspace.parts[0]
                else:
                    await asyncio.to_thread(
                        _assemble_parts,
                        workspace.parts,
                        workspace.assembly,
                    )
                    completed_path = workspace.assembly
            else:
                full_status, full_content_type, full_retries = (
                    await self._download_full(
                        url=probe.final_url,
                        target=workspace.parts[0],
                        request=request,
                        expected_size=probe.total_size,
                    )
                )
                status_code = full_status
                content_type = full_content_type
                retry_count += full_retries
                completed_path = workspace.parts[0]

            assert completed_path is not None
            size = await asyncio.to_thread(_file_size, completed_path)
            if size > request.max_bytes:
                raise _size_error(Capability.DOWNLOAD, size, request.max_bytes)
            if probe.total_size is not None and size != probe.total_size:
                raise ProviderRequestError(
                    provider=self.provider_id,
                    operation=Capability.DOWNLOAD,
                    message=(
                        f"downloaded size {size} does not match expected "
                        f"size {probe.total_size}"
                    ),
                    retryable=True,
                )
            digest = await asyncio.to_thread(_sha256_file, completed_path)
            await asyncio.to_thread(completed_path.rename, workspace.target)
            await asyncio.to_thread(
                _cleanup_download_workspace,
                workspace,
                workspace.target,
            )
            metadata: dict[str, object] = {
                "status_code": status_code,
                "range_supported": probe.range_supported,
                "parallel_parts": len(ranges),
                "resumed": resumed,
                "retry_count": retry_count,
            }
            if probe.total_size is not None:
                metadata["content_length"] = probe.total_size
            return DownloadArtifact(
                source_url=request.url,
                final_url=probe.final_url,
                path=workspace.target,
                content_type=content_type,
                size=size,
                sha256=digest,
                provider=self.provider_id,
                metadata=metadata,
            )
        except ProviderRequestError as exc:
            if (
                not request.resume
                or not probe.range_supported
                or probe.validator is None
                or not exc.retryable
            ):
                await asyncio.to_thread(_cleanup_download_workspace, workspace)
            raise
        except OSError as exc:
            await asyncio.to_thread(_cleanup_download_workspace, workspace)
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.DOWNLOAD,
                message=f"could not write download: {exc}",
                retryable=False,
            ) from exc

    async def _probe_download(self, request: DownloadRequest) -> _DownloadProbe:
        retries = 0
        while True:
            try:
                async with self._stream_response(
                    request.url,
                    Capability.DOWNLOAD,
                    headers={
                        "Range": "bytes=0-0",
                        "Accept-Encoding": "identity",
                    },
                    allowed_statuses=frozenset({400, 403, 405, 416}),
                ) as response:
                    total_size: int | None
                    range_supported = False
                    if response.status_code == 206:
                        parsed = _parse_content_range(response)
                        if (
                            parsed is None
                            or parsed[0:2] != (0, 0)
                        ):
                            raise ProviderRequestError(
                                provider=self.provider_id,
                                operation=Capability.DOWNLOAD,
                                message="server returned an invalid byte-range probe",
                                status_code=response.status_code,
                                retryable=False,
                            )
                        total_size = parsed[2]
                        range_supported = total_size is not None
                    elif response.status_code == 416:
                        total_size = _unsatisfied_total(response)
                        range_supported = total_size == 0
                        if not range_supported:
                            total_size = None
                    elif response.status_code in {400, 403, 405}:
                        total_size = None
                    else:
                        total_size = _content_length(response)
                    return _DownloadProbe(
                        final_url=str(response.url),
                        status_code=response.status_code,
                        content_type=_content_type(response),
                        total_size=total_size,
                        range_supported=range_supported,
                        validator=_response_validator(response),
                        retry_count=retries,
                    )
            except ProviderRequestError as exc:
                if not exc.retryable or retries >= request.max_retries:
                    raise
                retries += 1
                await self._download_retry_delay(retries)

    async def _download_range(
        self,
        *,
        url: str,
        target: Path,
        byte_range: _ByteRange,
        total_size: int,
        validator: str | None,
        request: DownloadRequest,
    ) -> tuple[bool, int]:
        existing = await asyncio.to_thread(_bounded_part_size, target, byte_range.size)
        resumed = existing > 0
        retries = 0
        while existing < byte_range.size:
            first_byte = byte_range.start + existing
            headers = {
                "Range": f"bytes={first_byte}-{byte_range.end}",
                "Accept-Encoding": "identity",
            }
            if validator is not None:
                headers["If-Range"] = validator
            try:
                async with self._stream_response(
                    url,
                    Capability.DOWNLOAD,
                    headers=headers,
                ) as response:
                    parsed = _parse_content_range(response)
                    expected = (first_byte, byte_range.end, total_size)
                    if response.status_code != 206 or parsed != expected:
                        raise ProviderRequestError(
                            provider=self.provider_id,
                            operation=Capability.DOWNLOAD,
                            message=(
                                "server did not honor the requested byte range; "
                                "the resource may have changed"
                            ),
                            status_code=response.status_code,
                            retryable=False,
                        )
                    returned_validator = _response_validator(response)
                    if (
                        validator is not None
                        and returned_validator is not None
                        and returned_validator != validator
                    ):
                        raise ProviderRequestError(
                            provider=self.provider_id,
                            operation=Capability.DOWNLOAD,
                            message="resource validator changed during download",
                            status_code=response.status_code,
                            retryable=False,
                        )
                    with target.open("ab") as output:
                        async for chunk in response.aiter_bytes():
                            existing += len(chunk)
                            if existing > byte_range.size:
                                raise ProviderRequestError(
                                    provider=self.provider_id,
                                    operation=Capability.DOWNLOAD,
                                    message=(
                                        "server sent bytes beyond the "
                                        "requested range"
                                    ),
                                    retryable=False,
                                )
                            await asyncio.to_thread(output.write, chunk)
                if existing != byte_range.size:
                    raise ProviderRequestError(
                        provider=self.provider_id,
                        operation=Capability.DOWNLOAD,
                        message="server ended a byte range before it was complete",
                        retryable=True,
                    )
            except ProviderRequestError as exc:
                if not exc.retryable or retries >= request.max_retries:
                    raise
                retries += 1
                resumed = existing > 0
                await self._download_retry_delay(retries)
        return resumed, retries

    async def _download_full(
        self,
        *,
        url: str,
        target: Path,
        request: DownloadRequest,
        expected_size: int | None,
    ) -> tuple[int, str | None, int]:
        retries = 0
        while True:
            size = 0
            try:
                async with self._stream_response(
                    url,
                    Capability.DOWNLOAD,
                    headers={"Accept-Encoding": "identity"},
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
                    with target.open("wb") as output:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > request.max_bytes:
                                raise _size_error(
                                    Capability.DOWNLOAD,
                                    size,
                                    request.max_bytes,
                                )
                            await asyncio.to_thread(output.write, chunk)
                    if expected_size is not None and size != expected_size:
                        raise ProviderRequestError(
                            provider=self.provider_id,
                            operation=Capability.DOWNLOAD,
                            message=(
                                f"downloaded size {size} does not match expected "
                                f"size {expected_size}"
                            ),
                            retryable=True,
                        )
                    return response.status_code, _content_type(response), retries
            except ProviderRequestError as exc:
                if not exc.retryable or retries >= request.max_retries:
                    raise
                retries += 1
                await self._download_retry_delay(retries)

    async def _download_retry_delay(self, retry_number: int) -> None:
        delay = self._download_retry_backoff * (2 ** (retry_number - 1))
        if delay:
            await asyncio.sleep(delay)

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
        self,
        url: str,
        operation: Capability,
        *,
        headers: Mapping[str, str] | None = None,
        allowed_statuses: frozenset[int] = frozenset(),
    ) -> AsyncIterator[httpx.Response]:
        current_url = url
        request_headers = {
            "User-Agent": _USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,text/plain,*/*;q=0.5"
            ),
        }
        if headers is not None:
            request_headers.update(headers)
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
                    headers=request_headers,
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
                    if response.status_code not in allowed_statuses:
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


def _parse_content_range(
    response: httpx.Response,
) -> tuple[int, int, int | None] | None:
    raw_value = response.headers.get("Content-Range")
    if raw_value is None:
        return None
    match = _CONTENT_RANGE.fullmatch(raw_value.strip())
    if match is None:
        return None
    total = None if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), total


def _unsatisfied_total(response: httpx.Response) -> int | None:
    raw_value = response.headers.get("Content-Range")
    if raw_value is None:
        return None
    match = _UNSATISFIED_CONTENT_RANGE.fullmatch(raw_value.strip())
    return int(match.group(1)) if match is not None else None


def _response_validator(response: httpx.Response) -> str | None:
    etag = response.headers.get("ETag")
    if etag is not None and not etag.lstrip().casefold().startswith("w/"):
        return etag
    return response.headers.get("Last-Modified")


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


def _download_ranges(
    total_size: int | None,
    *,
    range_supported: bool,
    maximum: int,
    part_size: int,
) -> tuple[_ByteRange, ...]:
    if total_size == 0:
        return (_ByteRange(0, -1),)
    if total_size is None or not range_supported:
        return (_ByteRange(0, (total_size or 1) - 1),)
    part_count = min(maximum, max(1, math.ceil(total_size / part_size)))
    base_size, remainder = divmod(total_size, part_count)
    ranges: list[_ByteRange] = []
    start = 0
    for index in range(part_count):
        size = base_size + (1 if index < remainder else 0)
        end = start + size - 1
        ranges.append(_ByteRange(start, end))
        start = end + 1
    return tuple(ranges)


def _download_state(
    *,
    request: DownloadRequest,
    probe: _DownloadProbe,
    ranges: tuple[_ByteRange, ...],
) -> dict[str, object]:
    return {
        "version": _DOWNLOAD_STATE_VERSION,
        "source_url_hash": _url_identity(request.url),
        "final_url_hash": _url_identity(probe.final_url),
        "total_size": probe.total_size,
        "validator": probe.validator,
        "ranges": [[item.start, item.end] for item in ranges],
    }


def _prepare_download_workspace(
    directory: Path,
    filename: str,
    expected_state: dict[str, object],
    resume: bool,
) -> _DownloadWorkspace:
    base = Path(filename)
    source_url_hash = expected_state["source_url_hash"]
    ranges = expected_state["ranges"]
    if not isinstance(ranges, list):
        raise OSError("download state did not contain byte ranges")
    for number in range(10_000):
        suffix = "" if number == 0 else f" ({number})"
        target = directory / f"{base.stem}{suffix}{base.suffix}"
        workspace = _workspace_for_target(target, len(ranges))
        existing_state = _read_download_state(workspace.state)

        if target.exists():
            if existing_state is not None:
                _cleanup_download_workspace(workspace, target)
            continue
        if existing_state is not None:
            if resume and existing_state == expected_state:
                return _DownloadWorkspace(
                    target=workspace.target,
                    state=workspace.state,
                    parts=workspace.parts,
                    assembly=workspace.assembly,
                    resumed=any(_file_size(path) > 0 for path in workspace.parts),
                )
            if existing_state.get("source_url_hash") == source_url_hash:
                _cleanup_download_artifacts(target)
            else:
                continue
        elif any(directory.glob(f"{target.name}.tiya-*")):
            continue

        try:
            with workspace.state.open("x", encoding="utf-8") as state_file:
                json.dump(expected_state, state_file, ensure_ascii=False)
        except FileExistsError:
            continue
        return workspace
    raise OSError("could not allocate a unique download filename")


def _workspace_for_target(target: Path, part_count: int) -> _DownloadWorkspace:
    parts: tuple[Path, ...]
    if part_count == 1:
        parts = (target.with_name(f"{target.name}.tiya-part"),)
    else:
        parts = tuple(
            target.with_name(f"{target.name}.tiya-part-{index:04d}")
            for index in range(part_count)
        )
    return _DownloadWorkspace(
        target=target,
        state=target.with_name(f"{target.name}.tiya-download.json"),
        parts=parts,
        assembly=target.with_name(f"{target.name}.tiya-assembling"),
        resumed=False,
    )


def _read_download_state(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _bounded_part_size(path: Path, maximum: int) -> int:
    size = _file_size(path)
    if size <= maximum:
        return size
    path.write_bytes(b"")
    return 0


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _assemble_parts(parts: tuple[Path, ...], destination: Path) -> None:
    with destination.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _url_identity(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _cleanup_download_workspace(
    workspace: _DownloadWorkspace,
    keep: Path | None = None,
) -> None:
    for path in (workspace.state, *workspace.parts, workspace.assembly):
        if path == keep:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _cleanup_download_artifacts(target: Path) -> None:
    for path in target.parent.glob(f"{target.name}.tiya-*"):
        try:
            path.unlink()
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
