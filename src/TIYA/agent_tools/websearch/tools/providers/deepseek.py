"""Async adapter for DeepSeek's Search and Fetch JSON APIs.

Official API guide: https://api-docs.deepseek.com/guides/search/
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self

import httpx


DEFAULT_BASE_URL = "https://api.deepseek.com/v1/tools"
DEFAULT_SEARCH_COUNT = 10
MAX_QUERY_LENGTH = 128
MAX_RESULT_COUNT = 50
MAX_URL_LENGTH = 8192


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One result returned by DeepSeek's Search API."""

    rerank_score: float
    title: str
    url: str
    snippet: str
    summary: str
    content: str
    site_name: str | None
    site_icon: str | None
    date: datetime | None
    authority: int


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """A successful Search API response, preserving provider result order."""

    results: tuple[SearchResult, ...]


@dataclass(frozen=True, slots=True)
class FetchResponse:
    """A successfully fetched page."""

    title: str
    content: str


class DeepSeekWebError(Exception):
    """Base class for errors raised by this adapter."""


class DeepSeekAPIError(DeepSeekWebError):
    """An error response returned by the DeepSeek API."""

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        error_type: str | None = None,
        code: str | None = None,
        param: object = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param
        super().__init__(f"DeepSeek API returned HTTP {status_code}: {message}")


class DeepSeekResponseError(DeepSeekWebError):
    """A successful DeepSeek response did not match the documented schema."""


class DeepSeekWebClient:
    """Reusable async client for DeepSeek's Search and Fetch endpoints.

    When ``http_client`` is supplied, its lifecycle remains the caller's
    responsibility. Otherwise this object owns an ``httpx.AsyncClient`` and can
    be used as an asynchronous context manager.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("api_key must not be empty")

        normalized_base_url = _validate_base_url(base_url)
        self._api_key = normalized_key
        self._base_url = normalized_base_url
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the internally created HTTP client, if any."""

        if self._owns_http_client:
            await self._http_client.aclose()

    async def search(
        self,
        query: str,
        *,
        count: int = DEFAULT_SEARCH_COUNT,
        original_question: str | None = None,
        localization: str = "",
        need_content: bool = False,
    ) -> SearchResponse:
        """Search the web and return ranked, structured results.

        ``query`` may include DeepSeek's documented operators such as ``site:``,
        ``lang:``, ``after:``, quoted phrases, and exclusion operators.
        """

        _validate_search_request(
            query=query,
            count=count,
            original_question=original_question,
            localization=localization,
            need_content=need_content,
        )
        payload: dict[str, object] = {
            "query": query,
            "count": count,
            "need_content": need_content,
        }
        if original_question is not None:
            payload["ori_question"] = original_question
        if localization:
            payload["localization"] = localization

        response_payload = await self._post("search", payload)
        return _parse_search_response(response_payload)

    async def fetch(self, url: str) -> FetchResponse:
        """Fetch one HTTP(S) page and return its Markdown content."""

        _validate_fetch_url(url)
        response_payload = await self._post("fetch", {"url": url})
        return _parse_fetch_response(response_payload)

    async def _post(
        self,
        operation: str,
        payload: dict[str, object],
    ) -> object:
        response = await self._http_client.post(
            f"{self._base_url}/{operation}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
        if response.is_error or response.is_redirect:
            raise _parse_api_error(response)

        try:
            return response.json()
        except ValueError as exc:
            raise DeepSeekResponseError(
                f"DeepSeek {operation} response was not valid JSON"
            ) from exc


def _validate_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    try:
        parsed = httpx.URL(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError("base_url must be a valid HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.host:
        raise ValueError("base_url must be a valid HTTP(S) URL")
    return normalized


def _validate_search_request(
    *,
    query: str,
    count: int,
    original_question: str | None,
    localization: str,
    need_content: bool,
) -> None:
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
        raise ValueError("query must contain 1 to 128 Unicode characters")
    if type(count) is not int or not 1 <= count <= MAX_RESULT_COUNT:
        raise ValueError("count must be an integer between 1 and 50")
    if original_question is not None and (
        not isinstance(original_question, str)
        or not 1 <= len(original_question) <= MAX_QUERY_LENGTH
    ):
        raise ValueError("original_question must contain 1 to 128 characters")
    if not isinstance(localization, str):
        raise ValueError("localization must be a string")
    if not isinstance(need_content, bool):
        raise ValueError("need_content must be a boolean")


def _validate_fetch_url(url: str) -> None:
    if not isinstance(url, str) or not url or len(url) > MAX_URL_LENGTH:
        raise ValueError("url must be a non-empty string of at most 8192 characters")
    try:
        parsed = httpx.URL(url)
    except (TypeError, ValueError) as exc:
        raise ValueError("url must be an HTTP(S) URL with a host") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.host
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "url must be an HTTP(S) URL with a host and no username or password"
        )


def _parse_search_response(payload: object) -> SearchResponse:
    data = _require_mapping(payload, "Search")
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise DeepSeekResponseError(
            "DeepSeek Search response must contain a results array"
        )

    results: list[SearchResult] = []
    for index, raw_result in enumerate(raw_results):
        item = _require_mapping(raw_result, f"Search result {index}")
        try:
            rerank_score = item["rerank_score"]
            title = item["title"]
            url = item["url"]
            snippet = item["snippet"]
            summary = item["summary"]
            content = item["content"]
            site_name = item["site_name"]
            site_icon = item["site_icon"]
            raw_date = item["date"]
            authority = item["authority"]
        except KeyError as exc:
            raise DeepSeekResponseError(
                f"DeepSeek Search result {index} is missing {exc.args[0]!r}"
            ) from exc

        if not _is_number(rerank_score):
            raise _invalid_result(index, "rerank_score")
        for field_name, value in (
            ("title", title),
            ("url", url),
            ("snippet", snippet),
            ("summary", summary),
            ("content", content),
        ):
            if not isinstance(value, str):
                raise _invalid_result(index, field_name)
        if site_name is not None and not isinstance(site_name, str):
            raise _invalid_result(index, "site_name")
        if site_icon is not None and not isinstance(site_icon, str):
            raise _invalid_result(index, "site_icon")
        if type(authority) is not int:
            raise _invalid_result(index, "authority")

        results.append(
            SearchResult(
                rerank_score=float(rerank_score),
                title=title,
                url=url,
                snippet=snippet,
                summary=summary,
                content=content,
                site_name=site_name,
                site_icon=site_icon,
                date=_parse_date(raw_date, index),
                authority=authority,
            )
        )
    return SearchResponse(results=tuple(results))


def _parse_fetch_response(payload: object) -> FetchResponse:
    data = _require_mapping(payload, "Fetch")
    title = data.get("title")
    content = data.get("content")
    if not isinstance(title, str) or not isinstance(content, str):
        raise DeepSeekResponseError(
            "DeepSeek Fetch response must contain string title and content fields"
        )
    return FetchResponse(title=title, content=content)


def _parse_api_error(response: httpx.Response) -> DeepSeekAPIError:
    fallback_message = response.reason_phrase or "request failed"
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return DeepSeekAPIError(
            status_code=response.status_code,
            message=fallback_message,
        )

    error = payload["error"]
    message = error.get("message")
    error_type = error.get("type")
    code = error.get("code")
    return DeepSeekAPIError(
        status_code=response.status_code,
        message=message if isinstance(message, str) else fallback_message,
        error_type=error_type if isinstance(error_type, str) else None,
        code=code if isinstance(code, str) else None,
        param=error.get("param"),
    )


def _require_mapping(payload: object, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DeepSeekResponseError(f"DeepSeek {label} response must be an object")
    return payload


def _parse_date(value: object, result_index: int) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid_result(result_index, "date")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _invalid_result(result_index, "date") from exc
    if parsed.tzinfo is None:
        raise _invalid_result(result_index, "date")
    return parsed


def _invalid_result(index: int, field_name: str) -> DeepSeekResponseError:
    return DeepSeekResponseError(
        f"DeepSeek Search result {index} has an invalid {field_name!r} field"
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "DeepSeekAPIError",
    "DeepSeekResponseError",
    "DeepSeekWebClient",
    "DeepSeekWebError",
    "FetchResponse",
    "SearchResponse",
    "SearchResult",
]
