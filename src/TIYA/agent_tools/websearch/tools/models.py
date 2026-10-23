"""Provider-neutral request and result objects for web-search tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit


_PROFILE_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


class InteractionKind(StrEnum):
    """Kinds of browser interaction that may require a person."""

    LOGIN = "login"
    OAUTH = "oauth"
    CAPTCHA = "captcha"
    MFA = "mfa"
    CONSENT = "consent"
    VERIFICATION = "verification"


class Capability(StrEnum):
    """Independent capabilities that providers may implement."""

    SEARCH = "search"
    FETCH = "fetch"
    MAP = "map"
    DOWNLOAD = "download"
    BROWSE = "browse"


class ContentFormat(StrEnum):
    """Representation used by a fetched document."""

    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"


class MediaKind(StrEnum):
    """Broad media categories discovered in a document."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class IdentityRef:
    """Opaque reference to a reusable local browser identity profile."""

    profile_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.profile_id, str)
            or not self.profile_id
            or self.profile_id in {".", ".."}
            or any(character not in _PROFILE_ID_CHARS for character in self.profile_id)
        ):
            raise ValueError(
                "profile_id may contain only letters, numbers, dot, "
                "underscore, and hyphen"
            )


@dataclass(frozen=True, slots=True)
class InteractionRequired:
    """Structured signal that automated reading needs a human-assisted step."""

    kind: InteractionKind
    url: str
    provider: str
    message: str
    identity: IdentityRef | None = None
    resumable: bool = True

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        _validate_provider_id(self.provider)
        if not self.message.strip():
            raise ValueError("message must not be empty")


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Human-readable capabilities and trade-offs of one configured provider."""

    provider_id: str
    display_name: str
    description: str
    capabilities: frozenset[Capability]
    strengths: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_provider_id(self.provider_id)
        if not self.display_name.strip():
            raise ValueError("display_name must not be empty")
        if not self.description.strip():
            raise ValueError("description must not be empty")
        capabilities = frozenset(self.capabilities)
        if not capabilities:
            raise ValueError("capabilities must not be empty")
        if not all(isinstance(item, Capability) for item in capabilities):
            raise ValueError("capabilities must contain Capability values")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "strengths",
            _freeze_texts(self.strengths, "strengths"),
        )
        object.__setattr__(
            self,
            "limitations",
            _freeze_texts(self.limitations, "limitations"),
        )


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """A provider-neutral web search request."""

    query: str
    max_results: int = 10
    original_question: str | None = None
    include_content: bool = False
    localization: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if type(self.max_results) is not int or self.max_results <= 0:
            raise ValueError("max_results must be a positive integer")
        if self.original_question is not None and not isinstance(
            self.original_question, str
        ):
            raise ValueError("original_question must be a string or None")
        if not isinstance(self.include_content, bool):
            raise ValueError("include_content must be a boolean")
        if self.localization is not None and not isinstance(self.localization, str):
            raise ValueError("localization must be a string or None")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One normalized result returned by a search provider."""

    url: str
    title: str = ""
    snippet: str = ""
    summary: str = ""
    content: str | None = None
    site_name: str | None = None
    site_icon: str | None = None
    published_at: datetime | None = None
    score: float | None = None
    authority: int | None = None
    provider: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Ordered search results produced by one provider."""

    query: str
    provider: str
    results: tuple[SearchResult, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query:
            raise ValueError("query must not be empty")
        _validate_provider_id(self.provider)
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """Request one web document from a provider."""

    url: str
    preferred_format: ContentFormat = ContentFormat.MARKDOWN
    render_javascript: bool = False
    identity: IdentityRef | None = None

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        if not isinstance(self.render_javascript, bool):
            raise ValueError("render_javascript must be a boolean")


@dataclass(frozen=True, slots=True)
class Link:
    """A navigable link discovered in a web document."""

    url: str
    text: str = ""
    title: str = ""
    internal: bool | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class MediaResource:
    """A media resource referenced by a web document."""

    url: str
    kind: MediaKind = MediaKind.OTHER
    alt: str = ""
    mime_type: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class WebDocument:
    """Normalized content and resources obtained from one URL."""

    url: str
    provider: str
    title: str
    content: str
    content_format: ContentFormat
    final_url: str | None = None
    status_code: int | None = None
    content_type: str | None = None
    links: tuple[Link, ...] = ()
    media: tuple[MediaResource, ...] = ()
    interaction_required: InteractionRequired | None = None
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        if self.final_url is not None:
            _validate_http_url(self.final_url, "final_url")
        _validate_provider_id(self.provider)
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "media", tuple(self.media))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class MapRequest:
    """Discover links below a root URL without requiring their content."""

    url: str
    query: str | None = None
    max_depth: int = 1
    max_results: int = 100
    include_external: bool = False
    identity: IdentityRef | None = None

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if type(self.max_results) is not int or self.max_results <= 0:
            raise ValueError("max_results must be a positive integer")


@dataclass(frozen=True, slots=True)
class MapResponse:
    """Links discovered beneath a root URL."""

    root_url: str
    provider: str
    links: tuple[Link, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_http_url(self.root_url, "root_url")
        _validate_provider_id(self.provider)
        object.__setattr__(self, "links", tuple(self.links))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    """Request a bounded binary download."""

    url: str
    filename: str | None = None
    max_bytes: int = 50 * 1024 * 1024

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if self.filename is not None and not self.filename:
            raise ValueError("filename must not be empty")


@dataclass(frozen=True, slots=True)
class BrowseRequest:
    """Open a stateful browser session at one HTTP(S) URL."""

    url: str
    identity: IdentityRef | None = None
    visible: bool | None = None

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        if self.visible is not None and not isinstance(self.visible, bool):
            raise ValueError("visible must be a boolean or None")


@dataclass(frozen=True, slots=True)
class DownloadArtifact:
    """Metadata for a file materialized by a download provider."""

    source_url: str
    final_url: str
    path: Path
    content_type: str | None
    size: int
    sha256: str
    provider: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_http_url(self.source_url, "source_url")
        _validate_http_url(self.final_url, "final_url")
        _validate_provider_id(self.provider)
        if self.size < 0:
            raise ValueError("size must be non-negative")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("extension options and metadata must be mappings")
    return MappingProxyType(dict(value))


def _freeze_texts(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    frozen = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in frozen):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return frozen


def _validate_provider_id(provider_id: str) -> None:
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise ValueError("provider must be a non-empty string")


def _validate_http_url(url: str, field_name: str) -> None:
    if not isinstance(url, str) or not url:
        raise ValueError(f"{field_name} must be a non-empty HTTP(S) URL")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError(f"{field_name} must be a valid HTTP(S) URL")


__all__ = [
    "Capability",
    "BrowseRequest",
    "ContentFormat",
    "DownloadArtifact",
    "DownloadRequest",
    "FetchRequest",
    "IdentityRef",
    "InteractionKind",
    "InteractionRequired",
    "Link",
    "MapRequest",
    "MapResponse",
    "MediaKind",
    "MediaResource",
    "ProviderProfile",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "WebDocument",
]

