from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from tavily.errors import BadRequestError, TimeoutError as TavilyTimeoutError

from TIYA.agent_tools.websearch.tools import (
    Capability,
    ContentFormat,
    FetchRequest,
    InvalidProviderResponseError,
    MapRequest,
    MediaKind,
    ProviderRequestError,
    SearchFilter,
    SearchFilterMode,
    SearchFilters,
    SearchRequest,
    TavilyProvider,
    WebSearchRuntime,
)


def run(coroutine):
    return asyncio.run(coroutine)


class FakeTavilyClient:
    def __init__(self) -> None:
        self.search_response: object = {"results": []}
        self.extract_response: object = {"results": [], "failed_results": []}
        self.map_response: object = {"base_url": "https://example.com", "results": []}
        self.search_calls: list[tuple[str, dict[str, object]]] = []
        self.extract_calls: list[tuple[object, dict[str, object]]] = []
        self.map_calls: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    async def search(self, query: str, **kwargs: object) -> object:
        self.search_calls.append((query, kwargs))
        return self.search_response

    async def extract(self, urls: object, **kwargs: object) -> object:
        self.extract_calls.append((urls, kwargs))
        return self.extract_response

    async def map(self, url: str, **kwargs: object) -> object:
        self.map_calls.append((url, kwargs))
        return self.map_response

    async def close(self) -> None:
        self.closed = True


def test_profile_declares_only_the_supported_stable_operations() -> None:
    assert TavilyProvider.profile.capabilities == frozenset(
        {Capability.SEARCH, Capability.FETCH, Capability.MAP}
    )
    assert TavilyProvider.profile.search_filter_support == {
        SearchFilter.INCLUDE_DOMAINS: SearchFilterMode.NATIVE,
        SearchFilter.EXCLUDE_DOMAINS: SearchFilterMode.NATIVE,
        SearchFilter.PUBLISHED_AFTER: SearchFilterMode.NATIVE,
        SearchFilter.PUBLISHED_BEFORE: SearchFilterMode.NATIVE,
        SearchFilter.LANGUAGE: SearchFilterMode.NATIVE,
        SearchFilter.COUNTRY: SearchFilterMode.NATIVE,
        SearchFilter.SAFE_SEARCH: SearchFilterMode.NATIVE,
    }


def test_search_normalizes_results_and_preserves_response_metadata() -> None:
    client = FakeTavilyClient()
    client.search_response = {
        "query": "latest Python release",
        "results": [
            {
                "title": "Python 3.14.1",
                "url": "https://www.python.org/downloads/release/python-3141/",
                "content": "Python 3.14.1 is the latest maintenance release.",
                "raw_content": "# Python 3.14.1\n\nRelease notes.",
                "score": 0.97,
                "published_date": "2026-10-01T08:30:00Z",
                "favicon": "https://www.python.org/favicon.ico",
            }
        ],
        "response_time": 0.42,
        "request_id": "request-1",
        "usage": {"credits": 1},
    }
    provider = TavilyProvider(client)

    response = run(
        provider.search(
            SearchRequest(
                "latest Python release",
                max_results=4,
                include_content=True,
            )
        )
    )

    assert client.search_calls == [
        (
            "latest Python release",
            {
                "max_results": 4,
                "include_answer": False,
                "include_raw_content": "markdown",
                "include_favicon": True,
                "include_usage": True,
            },
        )
    ]
    result = response.results[0]
    assert response.provider == "tavily"
    assert response.metadata == {
        "response_time": 0.42,
        "request_id": "request-1",
        "usage": {"credits": 1},
    }
    assert result.title == "Python 3.14.1"
    assert result.snippet.startswith("Python 3.14.1")
    assert result.summary == ""
    assert result.content == "# Python 3.14.1\n\nRelease notes."
    assert result.score == 0.97
    assert result.site_icon == "https://www.python.org/favicon.ico"
    assert result.published_at == datetime(
        2026, 10, 1, 8, 30, tzinfo=timezone.utc
    )


def test_search_omits_raw_content_when_not_requested() -> None:
    client = FakeTavilyClient()
    client.search_response = {
        "results": [
            {
                "title": "Example",
                "url": "https://example.com/result",
                "content": "Snippet",
                "raw_content": "Should not escape the adapter",
            }
        ]
    }

    response = run(TavilyProvider(client).search(SearchRequest("query")))

    assert client.search_calls[0][1]["include_raw_content"] is False
    assert response.results[0].content is None


def test_search_accepts_http_date_from_live_tavily_response() -> None:
    client = FakeTavilyClient()
    client.search_response = {
        "results": [
            {
                "title": "Python release",
                "url": "https://www.python.org/downloads/",
                "content": "Release information",
                "published_date": "Wed, 21 Oct 2026 09:00:00 GMT",
            }
        ]
    }

    response = run(TavilyProvider(client).search(SearchRequest("query")))

    assert response.results[0].published_at == datetime(
        2026, 10, 21, 9, 0, tzinfo=timezone.utc
    )


def test_search_maps_stable_filters_to_tavily_native_parameters() -> None:
    client = FakeTavilyClient()
    provider = TavilyProvider(client)

    run(
        provider.search(
            SearchRequest(
                "LLM inference research",
                filters=SearchFilters(
                    include_domains=("arxiv.org", "openreview.net"),
                    exclude_domains=("example.com",),
                    published_after="2026-01-02T08:00:00+08:00",
                    published_before="2026-10-23T00:00:00Z",
                    language="en",
                    country="united states",
                    safe_search=True,
                ),
            )
        )
    )

    assert client.search_calls == [
        (
            "LLM inference research",
            {
                "max_results": 10,
                "include_answer": False,
                "include_raw_content": False,
                "include_favicon": True,
                "include_usage": True,
                "include_domains": ["arxiv.org", "openreview.net"],
                "include_domains_mode": "restrict",
                "exclude_domains": ["example.com"],
                "start_date": "2026-01-02",
                "end_date": "2026-10-23",
                "include_published_date": True,
                "filter_by_published_date": True,
                "language": "en",
                "filter_by_language": True,
                "country": "united states",
                "safe_search": True,
            },
        )
    ]


def test_fetch_normalizes_extracted_content_images_and_favicon() -> None:
    client = FakeTavilyClient()
    client.extract_response = {
        "results": [
            {
                "url": "https://example.com/article",
                "raw_content": "# Article\n\nBody.",
                "images": [
                    "https://example.com/hero.jpg",
                    {
                        "url": "https://cdn.example.com/chart.png",
                        "description": "A chart",
                    },
                ],
                "favicon": "https://example.com/favicon.ico",
            }
        ],
        "failed_results": [],
        "response_time": 0.8,
        "request_id": "extract-1",
    }
    provider = TavilyProvider(client)

    document = run(
        provider.fetch(
            FetchRequest(
                "https://example.com/article",
                preferred_format=ContentFormat.MARKDOWN,
            )
        )
    )

    assert client.extract_calls == [
        (
            "https://example.com/article",
            {
                "format": "markdown",
                "include_images": True,
                "include_favicon": True,
                "include_usage": True,
            },
        )
    ]
    assert document.final_url == "https://example.com/article"
    assert document.title == ""
    assert document.content == "# Article\n\nBody."
    assert document.content_format == ContentFormat.MARKDOWN
    assert document.content_type == "text/markdown"
    assert tuple(resource.url for resource in document.media) == (
        "https://example.com/hero.jpg",
        "https://cdn.example.com/chart.png",
    )
    assert all(resource.kind == MediaKind.IMAGE for resource in document.media)
    assert document.media[1].alt == "A chart"
    assert document.metadata["favicon"] == "https://example.com/favicon.ico"
    assert document.metadata["request_id"] == "extract-1"


def test_fetch_supports_plain_text_and_rejects_html() -> None:
    client = FakeTavilyClient()
    client.extract_response = {
        "results": [
            {"url": "https://example.com", "raw_content": "Plain text"}
        ],
        "failed_results": [],
    }
    provider = TavilyProvider(client)

    document = run(
        provider.fetch(
            FetchRequest(
                "https://example.com", preferred_format=ContentFormat.TEXT
            )
        )
    )
    assert document.content_format == ContentFormat.TEXT
    assert document.content_type == "text/plain"

    with pytest.raises(ProviderRequestError, match="HTML") as caught:
        run(
            provider.fetch(
                FetchRequest(
                    "https://example.com", preferred_format=ContentFormat.HTML
                )
            )
        )
    assert caught.value.retryable is False
    assert len(client.extract_calls) == 1


def test_fetch_surfaces_provider_reported_failure() -> None:
    client = FakeTavilyClient()
    client.extract_response = {
        "results": [],
        "failed_results": [
            {"url": "https://example.com/private", "error": "Access denied"}
        ],
    }

    with pytest.raises(ProviderRequestError, match="Access denied") as caught:
        run(
            TavilyProvider(client).fetch(
                FetchRequest("https://example.com/private")
            )
        )

    assert caught.value.operation == Capability.FETCH
    assert caught.value.retryable is False


def test_map_normalizes_links_and_forwards_mapping_constraints() -> None:
    client = FakeTavilyClient()
    client.map_response = {
        "base_url": "https://docs.example.com",
        "results": [
            "https://docs.example.com/guide",
            "https://docs.example.com/api",
        ],
        "response_time": 1.2,
        "request_id": "map-1",
        "usage": {"credits": 1},
    }
    provider = TavilyProvider(client)

    response = run(
        provider.map(
            MapRequest(
                "https://docs.example.com",
                query="API reference pages",
                max_depth=2,
                max_results=20,
                include_external=True,
            )
        )
    )

    assert client.map_calls == [
        (
            "https://docs.example.com",
            {
                "max_depth": 2,
                "limit": 20,
                "instructions": "API reference pages",
                "allow_external": True,
                "include_usage": True,
            },
        )
    ]
    assert response.root_url == "https://docs.example.com"
    assert tuple(link.url for link in response.links) == (
        "https://docs.example.com/guide",
        "https://docs.example.com/api",
    )
    assert response.metadata["request_id"] == "map-1"


@pytest.mark.parametrize(
    ("operation", "response"),
    [
        ("search", {"results": "not-a-list"}),
        ("search", {"results": [{}]}),
        ("extract", {"results": [{"url": "https://example.com"}]}),
        ("map", {"results": [42]}),
    ],
)
def test_malformed_provider_response_raises_stable_error(
    operation: str, response: object
) -> None:
    client = FakeTavilyClient()
    setattr(client, f"{operation}_response", response)
    provider = TavilyProvider(client)

    if operation == "search":
        call = provider.search(SearchRequest("query"))
    elif operation == "extract":
        call = provider.fetch(FetchRequest("https://example.com"))
    else:
        call = provider.map(MapRequest("https://example.com"))

    with pytest.raises(InvalidProviderResponseError):
        run(call)


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (BadRequestError("bad query"), False),
        (TavilyTimeoutError(30), True),
    ],
)
def test_sdk_errors_are_mapped_to_stable_contract_errors(
    error: Exception, retryable: bool
) -> None:
    class FailingClient(FakeTavilyClient):
        async def search(self, query: str, **kwargs: object) -> object:
            raise error

    with pytest.raises(ProviderRequestError) as caught:
        run(TavilyProvider(FailingClient()).search(SearchRequest("query")))

    assert caught.value.provider == "tavily"
    assert caught.value.operation == Capability.SEARCH
    assert caught.value.retryable is retryable


def test_runtime_dispatches_tavily_and_closes_the_sdk_client() -> None:
    async def scenario() -> tuple[str, bool]:
        client = FakeTavilyClient()
        provider = TavilyProvider(client)
        async with WebSearchRuntime() as runtime:
            runtime.register(provider)
            response = await runtime.search(SearchRequest("query"))
        return response.provider, client.closed

    assert run(scenario()) == ("tavily", True)
