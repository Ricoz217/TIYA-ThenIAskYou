"""Structural contracts implemented by web-search providers."""

from __future__ import annotations

from typing import Protocol

from .browser import BrowserSession
from .models import (
    BrowseRequest,
    DownloadArtifact,
    DownloadRequest,
    FetchRequest,
    MapRequest,
    MapResponse,
    ProviderProfile,
    SearchRequest,
    SearchResponse,
    WebDocument,
)


class Provider(Protocol):
    """Metadata common to every configured provider."""

    provider_id: str
    profile: ProviderProfile


class SearchProvider(Provider, Protocol):
    """Provider capable of discovering web resources."""

    provider_id: str

    async def search(self, request: SearchRequest) -> SearchResponse: ...


class FetchProvider(Provider, Protocol):
    """Provider capable of reading one web document."""

    provider_id: str

    async def fetch(self, request: FetchRequest) -> WebDocument: ...


class MapProvider(Provider, Protocol):
    """Provider capable of discovering links below one root URL."""

    provider_id: str

    async def map(self, request: MapRequest) -> MapResponse: ...


class DownloadProvider(Provider, Protocol):
    """Provider capable of materializing a bounded remote resource."""

    provider_id: str

    async def download(self, request: DownloadRequest) -> DownloadArtifact: ...


class BrowseProvider(Provider, Protocol):
    """Provider capable of opening a stateful browser session."""

    async def browse(self, request: BrowseRequest) -> BrowserSession: ...


class AsyncCloseableProvider(Protocol):
    """Optional lifecycle contract recognized by the runtime."""

    async def aclose(self) -> None: ...


__all__ = [
    "AsyncCloseableProvider",
    "BrowseProvider",
    "DownloadProvider",
    "FetchProvider",
    "MapProvider",
    "Provider",
    "SearchProvider",
]

