"""Tavily implementation of the provider-neutral web-search contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from tavily import AsyncTavilyClient  # type: ignore[import-untyped]
from tavily.errors import (  # type: ignore[import-untyped]
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    TimeoutError as TavilyTimeoutError,
    UsageLimitExceededError,
)

from ..errors import InvalidProviderResponseError, ProviderRequestError
from ..models import (
    Capability,
    ContentFormat,
    FetchRequest,
    Link,
    MapRequest,
    MapResponse,
    MediaKind,
    MediaResource,
    ProviderProfile,
    SearchRequest,
    SearchResponse,
    SearchResult,
    WebDocument,
)


class _TavilyClient(Protocol):
    async def search(self, query: str, **kwargs: Any) -> object: ...

    async def extract(self, urls: object, **kwargs: Any) -> object: ...

    async def map(self, url: str, **kwargs: Any) -> object: ...

    async def close(self) -> None: ...


_TAVILY_REQUEST_ERRORS = (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)


class TavilyProvider:
    """Tavily Search, Extract, and Map behind the stable TIYA contract."""

    provider_id = "tavily"
    profile = ProviderProfile(
        provider_id=provider_id,
        display_name="Tavily",
        description=(
            "Search, content extraction, and site mapping through Tavily's "
            "official asynchronous Python client."
        ),
        capabilities=frozenset(
            {Capability.SEARCH, Capability.FETCH, Capability.MAP}
        ),
        strengths=(
            "Search can return ranked snippets and optional page content",
            "Extract supports Markdown or plain text and discovers images",
            "Map discovers site URLs with depth, limit, and instruction controls",
        ),
        limitations=(
            "Does not expose arbitrary HTTP requests or binary downloads",
            "Does not provide a stateful interactive browser session",
            (
                "Short ambiguous or mixed-language queries can drift; inspect "
                "result scores and reformulate instead of trusting rank alone"
            ),
            "Operations consume Tavily API credits and depend on the configured plan",
        ),
    )

    def __init__(self, client: _TavilyClient) -> None:
        self._client = client

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        api_base_url: str | None = None,
        project_id: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> TavilyProvider:
        """Build a provider using Tavily's official asynchronous client."""

        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        client = AsyncTavilyClient(
            api_key=api_key,
            api_base_url=api_base_url,
            project_id=project_id,
            client=http_client,
            client_source="tiya-websearch",
        )
        return cls(client)

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Search the web and normalize Tavily's ranked result objects."""

        try:
            payload = await self._client.search(
                request.query,
                max_results=request.max_results,
                include_answer=False,
                include_raw_content=(
                    "markdown" if request.include_content else False
                ),
                include_favicon=True,
                include_usage=True,
            )
        except TavilyTimeoutError as exc:
            raise _request_error(Capability.SEARCH, exc, retryable=True) from exc
        except _TAVILY_REQUEST_ERRORS as exc:
            raise _request_error(Capability.SEARCH, exc) from exc
        except httpx.HTTPError as exc:
            raise _request_error(Capability.SEARCH, exc, retryable=True) from exc

        data = _mapping(payload, Capability.SEARCH, "response")
        raw_results = _list(data.get("results"), Capability.SEARCH, "results")
        results: list[SearchResult] = []
        for index, raw_result in enumerate(raw_results):
            item = _mapping(
                raw_result,
                Capability.SEARCH,
                f"results[{index}]",
            )
            try:
                url = _required_text(
                    item.get("url"), Capability.SEARCH, f"results[{index}].url"
                )
                results.append(
                    SearchResult(
                        url=url,
                        title=_optional_text(
                            item.get("title"),
                            Capability.SEARCH,
                            f"results[{index}].title",
                        ),
                        snippet=_optional_text(
                            item.get("content"),
                            Capability.SEARCH,
                            f"results[{index}].content",
                        ),
                        content=(
                            _nullable_text(
                                item.get("raw_content"),
                                Capability.SEARCH,
                                f"results[{index}].raw_content",
                            )
                            if request.include_content
                            else None
                        ),
                        site_name=urlsplit(url).hostname,
                        site_icon=_nullable_text(
                            item.get("favicon"),
                            Capability.SEARCH,
                            f"results[{index}].favicon",
                        ),
                        published_at=_datetime(
                            item.get("published_date"),
                            Capability.SEARCH,
                            f"results[{index}].published_date",
                        ),
                        score=_number(
                            item.get("score"),
                            Capability.SEARCH,
                            f"results[{index}].score",
                        ),
                        provider=self.provider_id,
                    )
                )
            except ValueError as exc:
                raise _invalid(Capability.SEARCH, str(exc)) from exc

        return SearchResponse(
            query=request.query,
            provider=self.provider_id,
            results=tuple(results),
            metadata=_metadata(data),
        )

    async def fetch(self, request: FetchRequest) -> WebDocument:
        """Extract one URL as Markdown or plain text."""

        if request.preferred_format == ContentFormat.HTML:
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.FETCH,
                message="Tavily Extract does not return HTML",
                retryable=False,
            )

        format_name = request.preferred_format.value
        try:
            payload = await self._client.extract(
                request.url,
                format=format_name,
                include_images=True,
                include_favicon=True,
                include_usage=True,
            )
        except TavilyTimeoutError as exc:
            raise _request_error(Capability.FETCH, exc, retryable=True) from exc
        except _TAVILY_REQUEST_ERRORS as exc:
            raise _request_error(Capability.FETCH, exc) from exc
        except httpx.HTTPError as exc:
            raise _request_error(Capability.FETCH, exc, retryable=True) from exc

        data = _mapping(payload, Capability.FETCH, "response")
        raw_results = _list(data.get("results"), Capability.FETCH, "results")
        if not raw_results:
            self._raise_extract_failure(data, request.url)
        if len(raw_results) != 1:
            raise _invalid(
                Capability.FETCH,
                "results must contain exactly one extracted document",
            )

        item = _mapping(raw_results[0], Capability.FETCH, "results[0]")
        result_url = _required_text(
            item.get("url"), Capability.FETCH, "results[0].url"
        )
        content = _required_text(
            item.get("raw_content"),
            Capability.FETCH,
            "results[0].raw_content",
            allow_empty=True,
        )
        try:
            media = _images(item.get("images"), Capability.FETCH)
            metadata = _metadata(data)
            favicon = _nullable_text(
                item.get("favicon"), Capability.FETCH, "results[0].favicon"
            )
            if favicon is not None:
                metadata["favicon"] = favicon
            return WebDocument(
                url=request.url,
                final_url=result_url,
                provider=self.provider_id,
                title="",
                content=content,
                content_format=request.preferred_format,
                content_type=(
                    "text/markdown"
                    if request.preferred_format == ContentFormat.MARKDOWN
                    else "text/plain"
                ),
                media=media,
                metadata=metadata,
            )
        except ValueError as exc:
            raise _invalid(Capability.FETCH, str(exc)) from exc

    async def map(self, request: MapRequest) -> MapResponse:
        """Discover URLs under a site root using Tavily Map."""

        try:
            payload = await self._client.map(
                request.url,
                max_depth=request.max_depth,
                limit=request.max_results,
                instructions=request.query,
                allow_external=request.include_external,
                include_usage=True,
            )
        except TavilyTimeoutError as exc:
            raise _request_error(Capability.MAP, exc, retryable=True) from exc
        except _TAVILY_REQUEST_ERRORS as exc:
            raise _request_error(Capability.MAP, exc) from exc
        except httpx.HTTPError as exc:
            raise _request_error(Capability.MAP, exc, retryable=True) from exc

        data = _mapping(payload, Capability.MAP, "response")
        raw_links = _list(data.get("results"), Capability.MAP, "results")
        root_host = urlsplit(request.url).hostname
        links: list[Link] = []
        for index, value in enumerate(raw_links):
            url = _required_text(
                value, Capability.MAP, f"results[{index}]"
            )
            try:
                links.append(
                    Link(
                        url=url,
                        internal=urlsplit(url).hostname == root_host,
                    )
                )
            except ValueError as exc:
                raise _invalid(Capability.MAP, str(exc)) from exc

        metadata = _metadata(data)
        base_url = data.get("base_url")
        if base_url is not None:
            metadata["base_url"] = _required_text(
                base_url, Capability.MAP, "base_url"
            )
        return MapResponse(
            root_url=request.url,
            provider=self.provider_id,
            links=tuple(links),
            metadata=metadata,
        )

    async def aclose(self) -> None:
        """Close the underlying Tavily client and its owned HTTP transport."""

        await self._client.close()

    def _raise_extract_failure(
        self, data: Mapping[str, object], requested_url: str
    ) -> None:
        failures = _list(
            data.get("failed_results", []),
            Capability.FETCH,
            "failed_results",
        )
        if failures:
            failure = _mapping(
                failures[0], Capability.FETCH, "failed_results[0]"
            )
            message = failure.get("error")
            if not isinstance(message, str) or not message:
                message = f"Tavily could not extract {requested_url}"
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.FETCH,
                message=message,
                retryable=False,
            )
        raise _invalid(Capability.FETCH, "results is empty")


def _mapping(
    value: object, operation: Capability, field: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _invalid(operation, f"{field} must be an object")
    return value


def _list(value: object, operation: Capability, field: str) -> list[object]:
    if not isinstance(value, list):
        raise _invalid(operation, f"{field} must be an array")
    return value


def _required_text(
    value: object,
    operation: Capability,
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise _invalid(operation, f"{field} must be {qualifier}")
    return value


def _optional_text(value: object, operation: Capability, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _invalid(operation, f"{field} must be a string or null")
    return value


def _nullable_text(
    value: object, operation: Capability, field: str
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid(operation, f"{field} must be a string or null")
    return value or None


def _number(
    value: object, operation: Capability, field: str
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(operation, f"{field} must be a number or null")
    return float(value)


def _datetime(
    value: object, operation: Capability, field: str
) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise _invalid(operation, f"{field} is not an ISO-8601 datetime") from exc
    else:
        raise _invalid(operation, f"{field} must be a datetime string or null")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _images(value: object, operation: Capability) -> tuple[MediaResource, ...]:
    if value is None:
        return ()
    values = _list(value, operation, "results[0].images")
    resources: list[MediaResource] = []
    for index, raw_image in enumerate(values):
        alt = ""
        if isinstance(raw_image, str):
            url = raw_image
        else:
            image = _mapping(
                raw_image, operation, f"results[0].images[{index}]"
            )
            url = _required_text(
                image.get("url"),
                operation,
                f"results[0].images[{index}].url",
            )
            alt = _optional_text(
                image.get("description", image.get("alt")),
                operation,
                f"results[0].images[{index}].description",
            )
        resources.append(MediaResource(url=url, kind=MediaKind.IMAGE, alt=alt))
    return tuple(resources)


def _metadata(data: Mapping[str, object]) -> dict[str, object]:
    return {
        key: data[key]
        for key in ("response_time", "request_id", "usage")
        if key in data and data[key] is not None
    }


def _request_error(
    operation: Capability, error: Exception, *, retryable: bool = False
) -> ProviderRequestError:
    return ProviderRequestError(
        provider=TavilyProvider.provider_id,
        operation=operation,
        message=str(error),
        retryable=retryable,
    )


def _invalid(
    operation: Capability, message: str
) -> InvalidProviderResponseError:
    return InvalidProviderResponseError(
        provider=TavilyProvider.provider_id,
        operation=operation,
        message=message,
    )


__all__ = ["TavilyProvider"]
