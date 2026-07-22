from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pixivpy_async import PixivError

import TIYA.setu.setu as setu_module
from TIYA.setu.setu import MyAppPixivAPI


class _LoginConfig(dict):
    def __getattr__(self, key: str):
        return self.get(key)

    def __setattr__(self, key: str, value):
        self[key] = value


@pytest.fixture(autouse=True)
def _login_config(monkeypatch) -> None:
    monkeypatch.setitem(
        setu_module.BASE_CFG,
        "SETU",
        _LoginConfig(
            PixivLoginPublicUrl="",
            PixivLoginAutoDiscoverIP=True,
            PixivLoginBindHost="0.0.0.0",
            PixivLoginBindPort=8765,
            PixivLoginPublicPort=0,
            PixivLoginIPv4Endpoints=None,
            PixivLoginIPv6Endpoints=None,
            PixivLoginLocale="zh-HK",
            PixivLoginTLSCert="",
            PixivLoginTLSKey="",
            ProxyMode="NecessaryProxy",
        ),
    )
    monkeypatch.setitem(
        setu_module.SETTING_CFG,
        "SETU",
        SimpleNamespace(TokenResponseTimeout=60),
    )


def test_remote_login_is_preferred_when_configured(monkeypatch) -> None:
    asyncio.run(_test_remote_login_is_preferred_when_configured(monkeypatch))


async def _test_remote_login_is_preferred_when_configured(monkeypatch) -> None:
    api = object.__new__(MyAppPixivAPI)
    remote_capture = AsyncMock(return_value="remote-code")
    manual_capture = AsyncMock(return_value="manual-code")
    monkeypatch.setattr(setu_module, "capture_pixiv_code", remote_capture)
    monkeypatch.setattr(api, "_request_manual_login_code", manual_capture)
    monkeypatch.setattr(
        setu_module.BASE_CFG.SETU,
        "PixivLoginPublicUrl",
        "https://login.example.com/pixiv",
    )

    code = await api._obtain_authorization_code(
        "https://app-api.pixiv.net/web/v1/login?challenge=value",
        "manual instructions",
    )

    assert code == "remote-code"
    remote_capture.assert_awaited_once()
    assert remote_capture.await_args.kwargs["tls_certfile"] == ""
    assert remote_capture.await_args.kwargs["tls_keyfile"] == ""
    assert remote_capture.await_args.kwargs["locale"] == "zh-HK"
    manual_capture.assert_not_awaited()


def test_remote_login_failure_falls_back_to_manual(monkeypatch) -> None:
    asyncio.run(_test_remote_login_failure_falls_back_to_manual(monkeypatch))


async def _test_remote_login_failure_falls_back_to_manual(monkeypatch) -> None:
    api = object.__new__(MyAppPixivAPI)
    remote_capture = AsyncMock(
        side_effect=setu_module.RemoteLoginUnavailable("browser unavailable")
    )
    manual_capture = AsyncMock(return_value="manual-code")
    monkeypatch.setattr(setu_module, "capture_pixiv_code", remote_capture)
    monkeypatch.setattr(api, "_request_manual_login_code", manual_capture)
    monkeypatch.setattr(
        setu_module.BASE_CFG.SETU,
        "PixivLoginPublicUrl",
        "https://login.example.com/pixiv",
    )

    code = await api._obtain_authorization_code(
        "https://app-api.pixiv.net/web/v1/login?challenge=value",
        "manual instructions",
    )

    assert code == "manual-code"
    manual_capture.assert_awaited_once_with("manual instructions")


def test_empty_public_url_uses_manual_login(monkeypatch) -> None:
    asyncio.run(_test_empty_public_url_uses_manual_login(monkeypatch))


async def _test_empty_public_url_uses_manual_login(monkeypatch) -> None:
    api = object.__new__(MyAppPixivAPI)
    remote_capture = AsyncMock(return_value="remote-code")
    public_url_discovery = AsyncMock(
        side_effect=setu_module.RemoteLoginUnavailable("discovery unavailable")
    )
    manual_capture = AsyncMock(return_value="manual-code")
    monkeypatch.setattr(setu_module, "capture_pixiv_code", remote_capture)
    monkeypatch.setattr(
        setu_module,
        "discover_public_login_url",
        public_url_discovery,
    )
    monkeypatch.setattr(api, "_request_manual_login_code", manual_capture)
    monkeypatch.setattr(
        setu_module.BASE_CFG.SETU,
        "PixivLoginPublicUrl",
        "",
    )

    code = await api._obtain_authorization_code(
        "https://app-api.pixiv.net/web/v1/login?challenge=value",
        "manual instructions",
    )

    assert code == "manual-code"
    remote_capture.assert_not_awaited()


def test_empty_public_url_uses_discovered_ip(monkeypatch) -> None:
    asyncio.run(_test_empty_public_url_uses_discovered_ip(monkeypatch))


async def _test_empty_public_url_uses_discovered_ip(monkeypatch) -> None:
    api = object.__new__(MyAppPixivAPI)
    remote_capture = AsyncMock(return_value="remote-code")
    public_url_discovery = AsyncMock(
        return_value="http://203.0.113.10:11452"
    )
    manual_capture = AsyncMock(return_value="manual-code")
    monkeypatch.setattr(setu_module, "capture_pixiv_code", remote_capture)
    monkeypatch.setattr(
        setu_module,
        "discover_public_login_url",
        public_url_discovery,
    )
    monkeypatch.setattr(api, "_request_manual_login_code", manual_capture)
    monkeypatch.setattr(
        setu_module.BASE_CFG.SETU,
        "PixivLoginPublicUrl",
        "",
    )
    monkeypatch.setattr(
        setu_module.BASE_CFG.SETU,
        "PixivLoginPublicPort",
        11452,
    )

    code = await api._obtain_authorization_code(
        "https://app-api.pixiv.net/web/v1/login?challenge=value",
        "manual instructions",
    )

    assert code == "remote-code"
    public_url_discovery.assert_awaited_once()
    assert public_url_discovery.await_args.args == (11452,)
    assert (
        remote_capture.await_args.kwargs["public_base_url"]
        == "http://203.0.113.10:11452"
    )
    assert remote_capture.await_args.kwargs["bind_host"] == "0.0.0.0"
    assert remote_capture.await_args.kwargs["allow_insecure_public_url"] is True
    manual_capture.assert_not_awaited()


def test_remote_login_cancel_does_not_fall_back(monkeypatch) -> None:
    asyncio.run(_test_remote_login_cancel_does_not_fall_back(monkeypatch))


async def _test_remote_login_cancel_does_not_fall_back(monkeypatch) -> None:
    api = object.__new__(MyAppPixivAPI)
    remote_capture = AsyncMock(
        side_effect=setu_module.RemoteLoginCancelled("cancelled")
    )
    manual_capture = AsyncMock(return_value="manual-code")
    monkeypatch.setattr(setu_module, "capture_pixiv_code", remote_capture)
    monkeypatch.setattr(api, "_request_manual_login_code", manual_capture)
    monkeypatch.setattr(
        setu_module.BASE_CFG.SETU,
        "PixivLoginPublicUrl",
        "https://login.example.com/pixiv",
    )

    with pytest.raises(PixivError, match="Pixiv"):
        await api._obtain_authorization_code(
            "https://app-api.pixiv.net/web/v1/login?challenge=value",
            "manual instructions",
        )

    manual_capture.assert_not_awaited()


def test_pending_login_rejects_second_login_without_inspecting_exception(
        monkeypatch,
) -> None:
    asyncio.run(
        _test_pending_login_rejects_second_login_without_inspecting_exception(
            monkeypatch
        )
    )


async def _test_pending_login_rejects_second_login_without_inspecting_exception(
        monkeypatch,
) -> None:
    api = object.__new__(MyAppPixivAPI)
    started = asyncio.Event()
    release = asyncio.Event()

    async def token_login() -> dict:
        started.set()
        await release.wait()
        return {"ok": True}

    monkeypatch.setattr(api, "_token_login", token_login)
    monkeypatch.setattr(setu_module, "_LOGIN_TASK", None)

    first = asyncio.create_task(api.process_token_login())
    await started.wait()
    assert await api.process_token_login() is None
    release.set()
    assert await first == {"ok": True}
