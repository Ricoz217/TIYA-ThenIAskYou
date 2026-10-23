"""Provider registration and deterministic asynchronous dispatch."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, Self, cast

from .browser import BrowserSession
from .errors import (
    ProviderAlreadyRegisteredError,
    ProviderNotFoundError,
    ProviderSelectionError,
)
from .models import (
    BrowseRequest,
    Capability,
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
from .provider_protocols import (
    BrowseProvider,
    DownloadProvider,
    FetchProvider,
    MapProvider,
    Provider,
    SearchProvider,
)


_METHOD_BY_CAPABILITY: dict[Capability, str] = {
    Capability.SEARCH: "search",
    Capability.FETCH: "fetch",
    Capability.MAP: "map",
    Capability.DOWNLOAD: "download",
    Capability.BROWSE: "browse",
}


class WebSearchRuntime:
    """Thin, policy-free dispatcher for the five stable web operations.

    A provider is registered once with a profile declaring all supported
    capabilities. The runtime never retries or falls back implicitly.
    """

    def __init__(self) -> None:
        self._providers: dict[str, object] = {}
        self._profiles: dict[str, ProviderProfile] = {}
        self._by_capability: dict[Capability, dict[str, object]] = {
            capability: {} for capability in Capability
        }
        self._defaults: dict[Capability, str] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def register(self, provider: Provider, *, default: bool = False) -> None:
        """Register one provider for every capability declared in its profile."""

        provider_id = getattr(provider, "provider_id", None)
        profile = getattr(provider, "profile", None)
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        if not isinstance(profile, ProviderProfile):
            raise ValueError("provider must expose a ProviderProfile")
        if profile.provider_id != provider_id:
            raise ValueError("provider_id must match profile.provider_id")
        if provider_id in self._providers:
            first_capability = next(iter(profile.capabilities))
            raise ProviderAlreadyRegisteredError(first_capability, provider_id)

        for capability in profile.capabilities:
            method_name = _METHOD_BY_CAPABILITY[capability]
            method = getattr(provider, method_name, None)
            if not inspect.iscoroutinefunction(method):
                raise ValueError(
                    f"provider {provider_id!r} declares {capability.value} "
                    f"but has no async {method_name} method"
                )

        self._providers[provider_id] = provider
        self._profiles[provider_id] = profile
        for capability in profile.capabilities:
            self._by_capability[capability][provider_id] = provider
            if default:
                self._defaults[capability] = provider_id

    def set_default(self, capability: Capability, provider: str) -> None:
        """Select a registered provider as the default for one capability."""

        if provider not in self._by_capability[capability]:
            raise ProviderNotFoundError(capability, provider)
        self._defaults[capability] = provider

    def unregister(self, provider: str) -> object:
        """Remove and return a provider from all of its capabilities."""

        try:
            removed = self._providers.pop(provider)
            profile = self._profiles.pop(provider)
        except KeyError as exc:
            raise ProviderNotFoundError(provider=provider) from exc
        for capability in profile.capabilities:
            self._by_capability[capability].pop(provider, None)
            if self._defaults.get(capability) == provider:
                self._defaults.pop(capability, None)
        return removed

    def provider_ids(self, capability: Capability) -> tuple[str, ...]:
        """Return configured providers for one operation, in registration order."""

        return tuple(self._by_capability[capability])

    def provider_profiles(self) -> tuple[ProviderProfile, ...]:
        """Return the configured provider list exposed to higher layers."""

        return tuple(self._profiles.values())

    async def search(
        self, request: SearchRequest, *, provider: str | None = None
    ) -> SearchResponse:
        selected = cast(
            SearchProvider, self._resolve(Capability.SEARCH, provider)
        )
        return await selected.search(request)

    async def fetch(
        self, request: FetchRequest, *, provider: str | None = None
    ) -> WebDocument:
        selected = cast(FetchProvider, self._resolve(Capability.FETCH, provider))
        return await selected.fetch(request)

    async def map(
        self, request: MapRequest, *, provider: str | None = None
    ) -> MapResponse:
        selected = cast(MapProvider, self._resolve(Capability.MAP, provider))
        return await selected.map(request)

    async def download(
        self, request: DownloadRequest, *, provider: str | None = None
    ) -> DownloadArtifact:
        selected = cast(
            DownloadProvider, self._resolve(Capability.DOWNLOAD, provider)
        )
        return await selected.download(request)

    async def browse(
        self, request: BrowseRequest, *, provider: str | None = None
    ) -> BrowserSession:
        selected = cast(BrowseProvider, self._resolve(Capability.BROWSE, provider))
        return await selected.browse(request)

    async def aclose(self) -> None:
        """Close every registered provider once when it exposes ``aclose``."""

        for provider in self._providers.values():
            close = getattr(provider, "aclose", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await cast(Awaitable[Any], result)

    def _resolve(self, capability: Capability, provider: str | None) -> object:
        registrations = self._by_capability[capability]
        if provider is not None:
            try:
                return registrations[provider]
            except KeyError as exc:
                raise ProviderNotFoundError(capability, provider) from exc

        default = self._defaults.get(capability)
        if default is not None and default in registrations:
            return registrations[default]
        if not registrations:
            raise ProviderNotFoundError(capability)
        if len(registrations) == 1:
            return next(iter(registrations.values()))
        raise ProviderSelectionError(capability)


__all__ = ["WebSearchRuntime"]
