"""Stable errors exposed by the web-search tools package."""

from __future__ import annotations

from .models import Capability, InteractionRequired


class WebSearchError(Exception):
    """Base class for provider-neutral web-search failures."""


class OptionalProviderDependencyError(WebSearchError):
    """An explicitly enabled provider is missing its optional dependency."""

    def __init__(self, provider: str, install_hint: str) -> None:
        self.provider = provider
        self.install_hint = install_hint
        super().__init__(
            f"{provider} optional dependencies are not installed; {install_hint}"
        )


class ProviderAlreadyRegisteredError(WebSearchError):
    """A capability already has a provider with the same identifier."""

    def __init__(self, capability: Capability, provider: str) -> None:
        self.capability = capability
        self.provider = provider
        super().__init__(
            f"provider {provider!r} is already registered for {capability.value}"
        )


class ProviderNotFoundError(WebSearchError):
    """The requested capability or provider has not been registered."""

    def __init__(
        self,
        capability: Capability | None = None,
        provider: str | None = None,
    ) -> None:
        self.capability = capability
        self.provider = provider
        if capability is None and provider is not None:
            message = f"provider {provider!r} is not registered"
        elif capability is None:
            message = "provider is not registered"
        elif provider is None:
            message = f"no provider is registered for {capability.value}"
        else:
            message = (
                f"provider {provider!r} is not registered for {capability.value}"
            )
        super().__init__(message)


class ProviderSelectionError(WebSearchError):
    """Several providers exist but no deterministic selection was configured."""

    def __init__(self, capability: Capability) -> None:
        self.capability = capability
        super().__init__(
            f"multiple providers are registered for {capability.value}; "
            "select a provider or configure a default"
        )


class ProviderRequestError(WebSearchError):
    """A selected provider could not complete an operation."""

    def __init__(
        self,
        *,
        provider: str,
        operation: Capability,
        message: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        status = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"{provider} {operation.value} failed{status}: {message}")


class InteractionRequiredError(ProviderRequestError):
    """A provider reached a page that needs human-assisted interaction."""

    def __init__(
        self,
        *,
        interaction: InteractionRequired,
        operation: Capability,
        status_code: int | None = None,
    ) -> None:
        self.interaction = interaction
        super().__init__(
            provider=interaction.provider,
            operation=operation,
            message=interaction.message,
            status_code=status_code,
            retryable=False,
        )


class InvalidProviderResponseError(WebSearchError):
    """A provider returned data that could not satisfy the contract."""

    def __init__(
        self,
        *,
        provider: str,
        operation: Capability,
        message: str,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.message = message
        super().__init__(
            f"{provider} returned an invalid {operation.value} response: {message}"
        )


__all__ = [
    "InvalidProviderResponseError",
    "InteractionRequiredError",
    "OptionalProviderDependencyError",
    "ProviderAlreadyRegisteredError",
    "ProviderNotFoundError",
    "ProviderRequestError",
    "ProviderSelectionError",
    "WebSearchError",
]

