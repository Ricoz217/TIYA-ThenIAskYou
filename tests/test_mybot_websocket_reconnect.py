from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest
from websockets.exceptions import ConnectionClosedError

with patch.dict(os.environ, {"LOG_FORMAT": "%(message)s"}):
    import TIYA.mybot as mybot


@pytest.mark.parametrize(
    "connection_error",
    [
        ConnectionClosedError(None, None),
        OSError("connection lost"),
    ],
)
def test_custom_websocket_reconnects_after_network_error(
        monkeypatch,
        connection_error: Exception,
) -> None:
    attempts = 0

    async def connect(_websocket) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise connection_error
        raise asyncio.CancelledError

    sleep = AsyncMock()
    monkeypatch.setattr(mybot.Websocket, "ws_connect", connect)
    monkeypatch.setattr(mybot.asyncio, "sleep", sleep)

    websocket = mybot.CustomWebsocket(object())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(websocket.ws_connect())

    assert attempts == 2
    sleep.assert_awaited_once_with(3)


def test_custom_websocket_does_not_retry_cancellation(monkeypatch) -> None:
    connect = AsyncMock(side_effect=asyncio.CancelledError)
    sleep = AsyncMock()
    monkeypatch.setattr(mybot.Websocket, "ws_connect", connect)
    monkeypatch.setattr(mybot.asyncio, "sleep", sleep)

    websocket = mybot.CustomWebsocket(object())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(websocket.ws_connect())

    connect.assert_awaited_once_with()
    sleep.assert_not_awaited()


def test_custom_websocket_does_not_hide_programming_errors(monkeypatch) -> None:
    connect = AsyncMock(side_effect=RuntimeError("broken handler"))
    sleep = AsyncMock()
    monkeypatch.setattr(mybot.Websocket, "ws_connect", connect)
    monkeypatch.setattr(mybot.asyncio, "sleep", sleep)

    websocket = mybot.CustomWebsocket(object())
    with pytest.raises(RuntimeError, match="broken handler"):
        asyncio.run(websocket.ws_connect())

    connect.assert_awaited_once_with()
    sleep.assert_not_awaited()
