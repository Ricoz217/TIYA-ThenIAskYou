"""Provider-neutral wrapper around the raw DeepSeek Search/Fetch client."""

from __future__ import annotations

import httpx

from ..errors import InvalidProviderResponseError, ProviderRequestError
from ..models import (
    Capability,
    ContentFormat,
    FetchRequest,
    ProviderProfile,
    SearchRequest,
    SearchResponse,
    SearchResult,
    WebDocument,
)
from .deepseek import (
    DeepSeekAPIError,
    DeepSeekResponseError,
    DeepSeekWebClient,
)


class DeepSeekProvider:
    """DeepSeek implementation of the Search and Fetch contracts."""

    provider_id = "deepseek"
    profile = ProviderProfile(
        provider_id=provider_id,
        display_name="DeepSeek Web Search",
        description="Official DeepSeek Search and Fetch API adapter.",
        capabilities=frozenset({Capability.SEARCH, Capability.FETCH}),
        strengths=(
            "Search results include snippets and provider-generated summaries",
            "Fetch returns normalized Markdown",
        ),
        limitations=(
            "No site mapping, binary download, or interactive browsing",
            "No arbitrary HTTP methods or browser actions",
        ),
    )

    def __init__(self, client: DeepSeekWebClient) -> None:
        self._client = client

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com/v1/tools",
        timeout: float | httpx.Timeout = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> DeepSeekProvider:
        """Build a provider and its raw client from connection settings."""

        return cls(
            DeepSeekWebClient(
                api_key,
                base_url=base_url,
                timeout=timeout,
                http_client=http_client,
            )
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Run a DeepSeek search and normalize its provider-specific schema."""

        try:
            response = await self._client.search(
                request.query,
                count=min(request.max_results, 50),
                original_question=request.original_question,
                localization=request.localization or "",
                need_content=request.include_content,
            )
        except DeepSeekAPIError as exc:
            raise _api_error(Capability.SEARCH, exc) from exc
        except DeepSeekResponseError as exc:
            raise InvalidProviderResponseError(
                provider=self.provider_id,
                operation=Capability.SEARCH,
                message=str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise _transport_error(Capability.SEARCH, exc) from exc

        return SearchResponse(
            query=request.query,
            provider=self.provider_id,
            results=tuple(
                SearchResult(
                    url=item.url,
                    title=item.title,
                    snippet=item.snippet,
                    summary=item.summary,
                    content=item.content or None,
                    site_name=item.site_name,
                    site_icon=item.site_icon,
                    published_at=item.date,
                    score=item.rerank_score,
                    authority=item.authority,
                    provider=self.provider_id,
                )
                for item in response.results
            ),
        )

    async def fetch(self, request: FetchRequest) -> WebDocument:
        """Fetch Markdown through DeepSeek and normalize it as a document."""

        try:
            response = await self._client.fetch(request.url)
        except DeepSeekAPIError as exc:
            raise _api_error(Capability.FETCH, exc) from exc
        except DeepSeekResponseError as exc:
            raise InvalidProviderResponseError(
                provider=self.provider_id,
                operation=Capability.FETCH,
                message=str(exc),
            ) from exc
        except httpx.HTTPError as exc:
            raise _transport_error(Capability.FETCH, exc) from exc

        return WebDocument(
            url=request.url,
            provider=self.provider_id,
            title=response.title,
            content=response.content,
            content_format=ContentFormat.MARKDOWN,
            status_code=200,
        )

    async def aclose(self) -> None:
        """Close the owned raw client, if it owns its HTTP transport."""

        await self._client.aclose()


def _api_error(
    operation: Capability,
    error: DeepSeekAPIError,
) -> ProviderRequestError:
    return ProviderRequestError(
        provider=DeepSeekProvider.provider_id,
        operation=operation,
        message=error.message,
        status_code=error.status_code,
        retryable=(error.status_code in {408, 429} or error.status_code >= 500),
    )


def _transport_error(
    operation: Capability,
    error: httpx.HTTPError,
) -> ProviderRequestError:
    return ProviderRequestError(
        provider=DeepSeekProvider.provider_id,
        operation=operation,
        message=str(error),
        retryable=True,
    )


__all__ = ["DeepSeekProvider"]

