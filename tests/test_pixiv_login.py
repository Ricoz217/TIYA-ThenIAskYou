from __future__ import annotations

import asyncio

import pytest

from TIYA.setu.remote import (
    _RemoteLoginSession,
    RemoteLoginUnavailable,
    _extract_pixiv_code,
    _validate_public_base_url,
)
from TIYA.setu.public_url import (
    _extract_global_ip,
    _format_public_base_url,
    select_public_bind_host,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://app-api.pixiv.net/web/v1/users/auth/pixiv/callback"
            "?state=state-value&code=auth-code",
            "auth-code",
        ),
        ("pixiv://account/login?code=deep-link-code", "deep-link-code"),
        ("https://accounts.pixiv.net/post-redirect?code=ignored", None),
        ("https://example.com/callback?code=ignored", None),
        (
            "https://app-api.pixiv.net/web/v1/users/auth/pixiv/callback"
            "?code=",
            None,
        ),
    ],
)
def test_extract_pixiv_code_only_accepts_known_callbacks(
        url: str,
        expected: str | None,
) -> None:
    assert _extract_pixiv_code(url) == expected


def test_public_login_url_requires_https() -> None:
    with pytest.raises(RemoteLoginUnavailable):
        _validate_public_base_url("http://login.example.com/pixiv")


def test_auto_discovered_public_url_can_explicitly_allow_http() -> None:
    assert _validate_public_base_url(
        "http://203.0.113.10:11452",
        allow_insecure_http=True,
    ) == "http://203.0.113.10:11452"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8765",
        "http://localhost:8765/login",
        "https://login.example.com/pixiv",
    ],
)
def test_public_login_url_accepts_https_or_loopback_http(url: str) -> None:
    assert _validate_public_base_url(url) == url.rstrip("/")


def test_tls_certificate_and_key_must_be_configured_together() -> None:
    with pytest.raises(RemoteLoginUnavailable):
        _RemoteLoginSession._create_ssl_context("certificate.pem", None)


@pytest.mark.parametrize(
    ("address", "port", "expected"),
    [
        ("203.0.113.10", 11452, "http://203.0.113.10:11452"),
        (
            "2001:db8::1234",
            11452,
            "http://[2001:db8::1234]:11452",
        ),
    ],
)
def test_format_public_base_url(
        address: str,
        port: int,
        expected: str,
) -> None:
    assert _format_public_base_url(address, port) == expected


def test_auto_discovered_ipv6_listens_on_ipv6() -> None:
    assert select_public_bind_host(
        "http://[2001:db8::1234]:11452",
        "0.0.0.0",
        auto_discovered=True,
    ) == "::"


@pytest.mark.parametrize(
    ("payload", "version", "expected"),
    [
        ("8.8.4.4\n", 4, "8.8.4.4"),
        (
            "<html><body>Current IP Address: 8.8.8.8</body></html>",
            4,
            "8.8.8.8",
        ),
        ('{"data":{"ip":"1.1.1.1"}}', 4, "1.1.1.1"),
        ('{"ip":"192.168.1.10"}', 4, None),
        ("2001:4860:4860::8888", 6, "2001:4860:4860::8888"),
    ],
)
def test_extract_global_ip_from_common_response_formats(
        payload: str,
        version: int,
        expected: str | None,
) -> None:
    assert _extract_global_ip(payload, version) == expected


def test_repeated_submission_click_is_suppressed() -> None:
    asyncio.run(_test_repeated_submission_click_is_suppressed())


async def _test_repeated_submission_click_is_suppressed() -> None:
    session = _RemoteLoginSession(
        login_url="https://app-api.pixiv.net/web/v1/login?test=1",
        public_base_url="http://127.0.0.1:8765",
        bind_host="127.0.0.1",
        bind_port=8765,
        proxy=None,
        locale="zh-HK",
        tls_certfile=None,
        tls_keyfile=None,
    )
    target = {
        "activation": True,
        "submission": True,
        "signature": "BUTTON|login|submit",
    }

    assert session._should_suppress_click(target, 100, 200) is False
    assert session._should_suppress_click(target, 100, 200) is True


def test_enter_is_suppressed_after_submission_click() -> None:
    asyncio.run(_test_enter_is_suppressed_after_submission_click())


async def _test_enter_is_suppressed_after_submission_click() -> None:
    session = _RemoteLoginSession(
        login_url="https://app-api.pixiv.net/web/v1/login?test=1",
        public_base_url="http://127.0.0.1:8765",
        bind_host="127.0.0.1",
        bind_port=8765,
        proxy=None,
        locale="zh-HK",
        tls_certfile=None,
        tls_keyfile=None,
    )
    target = {
        "activation": True,
        "submission": True,
        "signature": "BUTTON|login|submit",
    }
    session._should_suppress_click(target, 100, 200)

    class _CDP:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def send(self, method: str, payload: dict) -> None:
            self.calls.append((method, payload))

    cdp = _CDP()
    session.cdp = cdp
    await session._handle_input('{"type": "key", "key": "Enter"}')

    assert cdp.calls == []


def test_first_remote_client_permanently_claims_session() -> None:
    asyncio.run(_test_first_remote_client_permanently_claims_session())


async def _test_first_remote_client_permanently_claims_session() -> None:
    session = _RemoteLoginSession(
        login_url="https://app-api.pixiv.net/web/v1/login?test=1",
        public_base_url="http://127.0.0.1:8765",
        bind_host="127.0.0.1",
        bind_port=8765,
        proxy=None,
        locale="zh-HK",
        tls_certfile=None,
        tls_keyfile=None,
    )

    assert session._claim_client("first-browser") is True
    assert session._claim_client("first-browser") is True
    assert session._claim_client("second-browser") is False
