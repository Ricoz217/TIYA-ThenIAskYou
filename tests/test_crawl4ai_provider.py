from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from TIYA.agent_tools.websearch.tools import (
    Capability,
    ContentFormat,
    Crawl4AIProvider,
    FetchRequest,
    IdentityRef,
    IdentityStore,
    InteractionKind,
    InteractionRequiredError,
    MapRequest,
    ProviderRequestError,
)


def run(coroutine):
    return asyncio.run(coroutine)


class FakeBackend:
    def __init__(self, results: object) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []
        self.close_count = 0

    async def crawl(self, url: str, **kwargs: object) -> object:
        self.calls.append({"url": url, **kwargs})
        return self.results

    async def aclose(self) -> None:
        self.close_count += 1


def crawl_result(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "url": "https://example.com/start",
        "redirected_url": "https://example.com/final",
        "success": True,
        "html": "<html><title>Example</title><body>Hello</body></html>",
        "cleaned_html": "<main>Hello</main>",
        "markdown": SimpleNamespace(raw_markdown="# Example\n\nHello"),
        "metadata": {"title": "Example"},
        "links": {
            "internal": [
                {"href": "https://example.com/docs", "text": "Docs"}
            ],
            "external": [
                {"href": "https://outside.test", "text": "Outside"}
            ],
        },
        "media": {
            "images": [
                {
                    "src": "https://example.com/hero.png",
                    "alt": "Hero",
                    "type": "image/png",
                }
            ],
            "videos": [],
            "audios": [],
        },
        "status_code": 200,
        "response_headers": {"content-type": "text/html; charset=utf-8"},
        "error_message": None,
        "crawl_stats": {"attempts": 1},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_crawl4ai_fetch_normalizes_document_and_reuses_identity(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(crawl_result())
    store = IdentityStore(tmp_path / "profiles")
    provider = Crawl4AIProvider(
        backend=backend,
        identity_store=store,
        headless=True,
        enable_stealth=True,
    )

    document = run(
        provider.fetch(
            FetchRequest(
                "https://example.com/start",
                preferred_format=ContentFormat.MARKDOWN,
                identity=IdentityRef("example-user"),
            )
        )
    )

    assert provider.profile.capabilities == frozenset(
        {Capability.FETCH, Capability.MAP}
    )
    assert document.title == "Example"
    assert document.content == "# Example\n\nHello"
    assert document.final_url == "https://example.com/final"
    assert document.links[0].internal is True
    assert document.media[0].mime_type == "image/png"
    assert backend.calls[0]["identity_dir"] == store.resolve(
        IdentityRef("example-user")
    )
    assert backend.calls[0]["deep_crawl"] is None


def test_crawl4ai_marks_pages_that_need_human_interaction() -> None:
    backend = FakeBackend(
        crawl_result(
            url="https://example.com/login",
            redirected_url=None,
            html='<form><input type="password" name="password"></form>',
            cleaned_html="<form>Sign in</form>",
            markdown=SimpleNamespace(raw_markdown="Sign in"),
            metadata={"title": "Sign in"},
        )
    )
    provider = Crawl4AIProvider(backend=backend)

    document = run(provider.fetch(FetchRequest("https://example.com/login")))

    assert document.interaction_required is not None
    assert document.interaction_required.kind == InteractionKind.LOGIN
    assert document.interaction_required.resumable is True


def test_crawl4ai_map_uses_bounded_deep_crawl_and_deduplicates_links() -> None:
    backend = FakeBackend(
        [
            crawl_result(redirected_url=None),
            crawl_result(
                url="https://example.com/docs",
                redirected_url=None,
                metadata={"title": "Docs"},
            ),
        ]
    )
    provider = Crawl4AIProvider(backend=backend)

    response = run(
        provider.map(
            MapRequest(
                "https://example.com/start",
                max_depth=2,
                max_results=3,
                include_external=False,
            )
        )
    )

    assert [link.url for link in response.links] == [
        "https://example.com/docs"
    ]
    deep_crawl = backend.calls[0]["deep_crawl"]
    assert deep_crawl == {
        "max_depth": 2,
        "max_pages": 3,
        "include_external": False,
    }


def test_crawl4ai_failure_maps_to_provider_error() -> None:
    backend = FakeBackend(
        crawl_result(success=False, error_message="navigation failed")
    )

    with pytest.raises(ProviderRequestError, match="navigation failed"):
        run(
            Crawl4AIProvider(backend=backend).fetch(
                FetchRequest("https://example.com")
            )
        )


def test_crawl4ai_antibot_failure_requests_human_verification() -> None:
    backend = FakeBackend(
        crawl_result(
            success=False,
            error_message="Blocked by anti-bot protection",
            status_code=503,
        )
    )

    with pytest.raises(InteractionRequiredError) as caught:
        run(
            Crawl4AIProvider(backend=backend).fetch(
                FetchRequest("https://example.com")
            )
        )

    assert caught.value.interaction.kind == InteractionKind.VERIFICATION
    assert caught.value.status_code == 503
