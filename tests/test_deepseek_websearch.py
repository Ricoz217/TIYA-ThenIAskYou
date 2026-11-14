from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from TIYA.agent_tools.websearch.tools import (
    DeepSeekAPIError,
    DeepSeekFetchResponse as FetchResponse,
    DeepSeekResponseError,
    DeepSeekSearchResponse as SearchResponse,
    DeepSeekSearchResult as SearchResult,
    DeepSeekWebClient,
    DeepSeekProvider,
    SearchFilter,
    SearchFilterMode,
    SearchFilters,
    SearchRequest,
    UnsupportedSearchFilterError,
)


API_KEY = "test-deepseek-key"


def run(coroutine):
    return asyncio.run(coroutine)


def test_search_sends_official_payload_and_returns_typed_results() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        seen["payload"] = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "rerank_score": 2.4,
                        "title": "DeepSeek Search API",
                        "url": "https://api-docs.deepseek.com/guides/search/",
                        "snippet": "A short summary.",
                        "summary": "A longer, model-oriented summary.",
                        "content": "# Search API",
                        "site_name": "DeepSeek API Docs",
                        "site_icon": "https://api-docs.deepseek.com/favicon.ico",
                        "date": "2026-10-19T05:41:50Z",
                        "authority": 4,
                    }
                ]
            },
        )

    async def scenario() -> SearchResponse:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = DeepSeekWebClient(API_KEY, http_client=http_client)
            return await client.search(
                "DeepSeek Search API",
                count=5,
                original_question="DeepSeek 官方搜索接口怎么用？",
                localization="203.0.113.10",
                need_content=True,
            )

    response = run(scenario())

    assert isinstance(response, SearchResponse)
    assert response.results == (
        SearchResult(
            rerank_score=2.4,
            title="DeepSeek Search API",
            url="https://api-docs.deepseek.com/guides/search/",
            snippet="A short summary.",
            summary="A longer, model-oriented summary.",
            content="# Search API",
            site_name="DeepSeek API Docs",
            site_icon="https://api-docs.deepseek.com/favicon.ico",
            date=datetime(2026, 10, 19, 5, 41, 50, tzinfo=timezone.utc),
            authority=4,
        ),
    )
    request = seen["request"]
    assert isinstance(request, httpx.Request)
    assert request.method == "POST"
    assert str(request.url) == "https://api.deepseek.com/v1/tools/search"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert seen["payload"] == {
        "query": "DeepSeek Search API",
        "count": 5,
        "ori_question": "DeepSeek 官方搜索接口怎么用？",
        "localization": "203.0.113.10",
        "need_content": True,
    }


def test_search_uses_documented_defaults_and_accepts_empty_results() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(__import__("json").loads(request.content))
        return httpx.Response(200, json={"results": []})

    async def scenario() -> SearchResponse:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            return await DeepSeekWebClient(
                API_KEY, http_client=http_client
            ).search("nothing here")

    response = run(scenario())

    assert response.results == ()
    assert payloads == [
        {"query": "nothing here", "count": 10, "need_content": False}
    ]


def test_provider_rejects_filters_it_cannot_honor_without_calling_api() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"results": []})

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            provider = DeepSeekProvider(
                DeepSeekWebClient(API_KEY, http_client=http_client)
            )
            await provider.search(
                SearchRequest(
                    "query",
                    filters=SearchFilters(country="japan", safe_search=True),
                )
            )

    with pytest.raises(UnsupportedSearchFilterError) as caught:
        run(scenario())

    assert caught.value.filters == (
        SearchFilter.COUNTRY,
        SearchFilter.SAFE_SEARCH,
    )
    assert calls == 0


def test_provider_compiles_stable_filters_to_documented_query_operators() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(__import__("json").loads(request.content))
        return httpx.Response(200, json={"results": []})

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            provider = DeepSeekProvider(
                DeepSeekWebClient(API_KEY, http_client=http_client)
            )
            await provider.search(
                SearchRequest(
                    "model inference",
                    filters=SearchFilters(
                        include_domains=("arxiv.org", "openreview.net"),
                        exclude_domains=("spam.example",),
                        published_after="2026-01-02T08:00:00+08:00",
                        published_before="2026-10-23T00:00:00Z",
                        language="en",
                    ),
                )
            )

    run(scenario())

    assert DeepSeekProvider.profile.search_filter_support == {
        SearchFilter.INCLUDE_DOMAINS: SearchFilterMode.QUERY_OPERATOR,
        SearchFilter.EXCLUDE_DOMAINS: SearchFilterMode.QUERY_OPERATOR,
        SearchFilter.PUBLISHED_AFTER: SearchFilterMode.QUERY_OPERATOR,
        SearchFilter.PUBLISHED_BEFORE: SearchFilterMode.QUERY_OPERATOR,
        SearchFilter.LANGUAGE: SearchFilterMode.QUERY_OPERATOR,
    }
    assert payloads == [
        {
            "query": (
                "model inference site:arxiv.org OR site:openreview.net "
                "-site:spam.example lang:en after:2026-01-02 "
                "before:2026-10-23"
            ),
            "count": 10,
            "need_content": False,
        }
    ]


def test_fetch_returns_title_and_markdown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://proxy.example/api/v1/tools/fetch"
        assert __import__("json").loads(request.content) == {
            "url": "https://example.com/article"
        }
        return httpx.Response(
            200,
            json={"title": "Example", "content": "# Example\n\nArticle."},
        )

    async def scenario() -> FetchResponse:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = DeepSeekWebClient(
                API_KEY,
                base_url="https://proxy.example/api/v1/tools/",
                http_client=http_client,
            )
            return await client.fetch("https://example.com/article")

    assert run(scenario()) == FetchResponse(
        title="Example",
        content="# Example\n\nArticle.",
    )


def test_api_error_exposes_official_error_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "'count' must be between 1 and 50",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_request_error",
                }
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await DeepSeekWebClient(API_KEY, http_client=http_client).search("q")

    with pytest.raises(DeepSeekAPIError) as caught:
        run(scenario())

    assert caught.value.status_code == 400
    assert caught.value.message == "'count' must be between 1 and 50"
    assert caught.value.error_type == "invalid_request_error"
    assert caught.value.code == "invalid_request_error"
    assert caught.value.param is None


@pytest.mark.parametrize(
    ("response_payload", "method", "argument"),
    [
        ({}, "search", "q"),
        ({"results": [{}]}, "search", "q"),
        ({"title": "missing content"}, "fetch", "https://example.com"),
    ],
)
def test_malformed_success_response_raises_adapter_error(
    response_payload: dict[str, object], method: str, argument: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_payload)

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = DeepSeekWebClient(API_KEY, http_client=http_client)
            await getattr(client, method)(argument)

    with pytest.raises(DeepSeekResponseError):
        run(scenario())


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda client: client.search(""), "query"),
        (lambda client: client.search("x" * 129), "query"),
        (lambda client: client.search("valid", count=0), "count"),
        (lambda client: client.search("valid", count=51), "count"),
        (lambda client: client.fetch("ftp://example.com"), "url"),
        (lambda client: client.fetch("https://user:pass@example.com"), "url"),
    ],
)
def test_invalid_request_is_rejected_before_dispatch(call, message: str) -> None:
    async def scenario() -> None:
        client = DeepSeekWebClient(API_KEY)
        try:
            await call(client)
        finally:
            await client.aclose()

    with pytest.raises(ValueError, match=message):
        run(scenario())


def test_client_requires_a_non_empty_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        DeepSeekWebClient("  ")
