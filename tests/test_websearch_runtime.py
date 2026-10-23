from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from TIYA.agent_tools.websearch.tools import (
    BrowseRequest,
    BrowserPage,
    BrowserSession,
    Capability,
    ContentFormat,
    DeepSeekProvider,
    DeepSeekWebClient,
    DownloadArtifact,
    DownloadRequest,
    FetchRequest,
    Link,
    MapRequest,
    MapResponse,
    ProviderProfile,
    ProviderAlreadyRegisteredError,
    ProviderNotFoundError,
    ProviderRequestError,
    ProviderSelectionError,
    SearchRequest,
    SearchResponse,
    SearchResult,
    WebDocument,
    WebSearchRuntime,
)


def run(coroutine):
    return asyncio.run(coroutine)


class FakeSearchProvider:
    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        self.profile = ProviderProfile(
            provider_id=provider_id,
            display_name=provider_id.title(),
            description="Test search provider",
            capabilities=frozenset({Capability.SEARCH}),
            strengths=("predictable",),
            limitations=("test only",),
        )
        self.requests: list[SearchRequest] = []
        self.close_count = 0

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        return SearchResponse(
            query=request.query,
            provider=self.provider_id,
            results=(
                SearchResult(
                    url="https://example.com/result",
                    title="Result",
                    provider=self.provider_id,
                ),
            ),
        )

    async def aclose(self) -> None:
        self.close_count += 1


class FakeAllProvider(FakeSearchProvider):
    def __init__(self, provider_id: str) -> None:
        super().__init__(provider_id)
        self.profile = ProviderProfile(
            provider_id=provider_id,
            display_name=provider_id.title(),
            description="Test provider for every stable operation",
            capabilities=frozenset(Capability),
        )

    async def fetch(self, request: FetchRequest) -> WebDocument:
        return WebDocument(
            url=request.url,
            provider=self.provider_id,
            title="Fetched",
            content="# Fetched",
            content_format=ContentFormat.MARKDOWN,
        )

    async def map(self, request: MapRequest) -> MapResponse:
        return MapResponse(
            root_url=request.url,
            provider=self.provider_id,
            links=(Link(url=f"{request.url}/child", text="Child"),),
        )

    async def download(self, request: DownloadRequest) -> DownloadArtifact:
        return DownloadArtifact(
            source_url=request.url,
            final_url=request.url,
            path=Path("download.bin"),
            content_type="application/octet-stream",
            size=8,
            sha256="0" * 64,
            provider=self.provider_id,
        )

    async def browse(self, request: BrowseRequest) -> BrowserSession:
        return FakeBrowserSession(self.provider_id, request.url)


class FakeBrowserSession(BrowserSession):
    def __init__(self, provider_id: str, url: str) -> None:
        self.provider_id = provider_id
        self.session_id = "fake-session"
        self.url = url

    async def execute(self, action):
        return await self.snapshot()

    async def snapshot(self) -> BrowserPage:
        return BrowserPage(
            session_id=self.session_id,
            revision=0,
            document=WebDocument(
                url=self.url,
                provider=self.provider_id,
                title="Browser",
                content="Browser page",
                content_format=ContentFormat.TEXT,
            ),
        )

    async def screenshot(self, *, full_page: bool = False):
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class InvalidSyncProvider:
    provider_id = "sync"
    profile = ProviderProfile(
        provider_id=provider_id,
        display_name="Sync",
        description="Invalid synchronous test provider",
        capabilities=frozenset({Capability.SEARCH}),
    )

    def search(self, request: SearchRequest) -> SearchResponse:
        raise AssertionError("must not be called")


def test_runtime_delegates_to_the_only_registered_provider() -> None:
    async def scenario() -> tuple[SearchResponse, FakeSearchProvider]:
        provider = FakeSearchProvider("primary")
        runtime = WebSearchRuntime()
        runtime.register(provider)
        response = await runtime.search(SearchRequest("latest information"))
        return response, provider

    response, provider = run(scenario())

    assert response.provider == "primary"
    assert response.results[0].title == "Result"
    assert provider.requests == [SearchRequest("latest information")]


def test_runtime_uses_default_or_explicit_provider_without_hidden_fallback() -> None:
    async def scenario() -> tuple[str, str]:
        first = FakeSearchProvider("first")
        second = FakeSearchProvider("second")
        runtime = WebSearchRuntime()
        runtime.register(first)
        runtime.register(second, default=True)

        default_result = await runtime.search(SearchRequest("q"))
        explicit_result = await runtime.search(
            SearchRequest("q"), provider="first"
        )
        return default_result.provider, explicit_result.provider

    assert run(scenario()) == ("second", "first")


def test_runtime_rejects_ambiguous_or_unknown_provider_selection() -> None:
    runtime = WebSearchRuntime()
    runtime.register(FakeSearchProvider("one"))
    runtime.register(FakeSearchProvider("two"))

    with pytest.raises(ProviderSelectionError, match="default"):
        run(runtime.search(SearchRequest("q")))
    with pytest.raises(ProviderNotFoundError, match="missing"):
        run(runtime.search(SearchRequest("q"), provider="missing"))
    with pytest.raises(ProviderNotFoundError, match="map"):
        run(runtime.map(MapRequest("https://example.com")))


def test_runtime_rejects_duplicate_provider_ids_per_capability() -> None:
    runtime = WebSearchRuntime()
    runtime.register(FakeSearchProvider("same"))

    with pytest.raises(ProviderAlreadyRegisteredError, match="same"):
        runtime.register(FakeSearchProvider("same"))


def test_runtime_rejects_provider_methods_that_are_not_async() -> None:
    with pytest.raises(ValueError, match="async search"):
        WebSearchRuntime().register(InvalidSyncProvider())


def test_runtime_dispatches_every_contract_and_closes_provider_once() -> None:
    async def scenario() -> tuple[object, object, object, object, int]:
        provider = FakeAllProvider("all")
        runtime = WebSearchRuntime()
        runtime.register(provider)

        async with runtime:
            document = await runtime.fetch(FetchRequest("https://example.com"))
            site_map = await runtime.map(MapRequest("https://example.com"))
            artifact = await runtime.download(
                DownloadRequest("https://example.com/file.bin")
            )
            browser = await runtime.browse(BrowseRequest("https://example.com"))
        return document, site_map, artifact, browser, provider.close_count

    document, site_map, artifact, browser, close_count = run(scenario())

    assert isinstance(document, WebDocument)
    assert isinstance(site_map, MapResponse)
    assert isinstance(artifact, DownloadArtifact)
    assert isinstance(browser, BrowserSession)
    assert close_count == 1


def test_runtime_can_change_and_remove_a_default_provider() -> None:
    runtime = WebSearchRuntime()
    runtime.register(FakeSearchProvider("one"), default=True)
    runtime.register(FakeSearchProvider("two"))

    runtime.set_default(Capability.SEARCH, "two")
    assert run(runtime.search(SearchRequest("q"))).provider == "two"

    removed = runtime.unregister("two")
    assert removed.provider_id == "two"
    assert run(runtime.search(SearchRequest("q"))).provider == "one"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SearchRequest(""),
        lambda: SearchRequest("query", max_results=0),
        lambda: FetchRequest("ftp://example.com"),
        lambda: MapRequest("https://example.com", max_depth=-1),
        lambda: DownloadRequest("https://example.com/file", max_bytes=0),
        lambda: BrowseRequest("ftp://example.com"),
    ],
)
def test_contract_models_reject_invalid_requests(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_contract_models_freeze_sequences_and_metadata() -> None:
    options = {"vendor": "value"}
    response = SearchResponse(
        query="q",
        provider="p",
        results=[SearchResult(url="https://example.com", provider="p")],
        metadata=options,
    )
    options["vendor"] = "changed"

    assert isinstance(response.results, tuple)
    assert response.metadata["vendor"] == "value"
    with pytest.raises(TypeError):
        response.metadata["new"] = "not allowed"  # type: ignore[index]


def test_runtime_exposes_configured_provider_profiles() -> None:
    runtime = WebSearchRuntime()
    provider = FakeAllProvider("all")
    runtime.register(provider)

    assert runtime.provider_profiles() == (provider.profile,)
    assert provider.profile.capabilities == frozenset(
        {
            Capability.SEARCH,
            Capability.FETCH,
            Capability.MAP,
            Capability.DOWNLOAD,
            Capability.BROWSE,
        }
    )


def test_deepseek_provider_normalizes_search_and_fetch_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "rerank_score": 2.7,
                            "title": "Official result",
                            "url": "https://example.com/result",
                            "snippet": "Short",
                            "summary": "Long",
                            "content": "Body",
                            "site_name": "Example",
                            "site_icon": None,
                            "date": "2026-10-23T00:00:00Z",
                            "authority": 4,
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"title": "Page", "content": "# Page\n\nContent"},
        )

    async def scenario() -> tuple[SearchResponse, WebDocument]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = DeepSeekWebClient("key", http_client=http_client)
            provider = DeepSeekProvider(client)
            runtime = WebSearchRuntime()
            runtime.register(provider)
            result = await runtime.search(
                SearchRequest(
                    "query",
                    max_results=5,
                    original_question="question",
                    include_content=True,
                )
            )
            document = await runtime.fetch(
                FetchRequest("https://example.com/result")
            )
            return result, document

    result, document = run(scenario())

    assert DeepSeekProvider.profile.capabilities == frozenset(
        {Capability.SEARCH, Capability.FETCH}
    )
    assert result.provider == "deepseek"
    assert result.results[0] == SearchResult(
        url="https://example.com/result",
        title="Official result",
        snippet="Short",
        summary="Long",
        content="Body",
        site_name="Example",
        published_at=result.results[0].published_at,
        score=2.7,
        authority=4,
        provider="deepseek",
    )
    assert document == WebDocument(
        url="https://example.com/result",
        provider="deepseek",
        title="Page",
        content="# Page\n\nContent",
        content_format=ContentFormat.MARKDOWN,
        status_code=200,
        fetched_at=document.fetched_at,
    )


def test_deepseek_provider_maps_api_failures_to_contract_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "rate limited",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": "rate_limit_error",
                }
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            provider = DeepSeekProvider(
                DeepSeekWebClient("key", http_client=http_client)
            )
            await provider.search(SearchRequest("query"))

    with pytest.raises(ProviderRequestError) as caught:
        run(scenario())

    assert caught.value.provider == "deepseek"
    assert caught.value.operation == Capability.SEARCH
    assert caught.value.status_code == 429
    assert caught.value.retryable is True
