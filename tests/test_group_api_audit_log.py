import asyncio

from ncatbot.core.message import MessageChain

from TIYA.api import group_api


class _LoggerSpy:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.debugs: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def debug(self, message: str) -> None:
        self.debugs.append(message)


class _FailingApi:
    async def post_group_msg(self, group_id, rtf):
        raise RuntimeError(f"send failed for {group_id}: {rtf}")


class _FailingBot:
    api = _FailingApi()


class _NotOkApi:
    async def post_group_msg(self, group_id, rtf):
        return {
            "status": "failed",
            "retcode": 100,
            "message": "blocked",
        }


class _NotOkBot:
    api = _NotOkApi()


def test_say_logs_sdk_send_exception(monkeypatch) -> None:
    logger = _LoggerSpy()
    monkeypatch.setattr(group_api, "_log", logger)
    monkeypatch.setattr(group_api, "get_bot", lambda: _FailingBot())

    result = asyncio.run(group_api.say("group-1", MessageChain("hello")))

    assert result["say"] is None
    assert logger.errors
    assert "group-1" in logger.errors[0]
    assert "send failed" in logger.errors[0]
    assert logger.debugs


def test_say_logs_not_ok_response(monkeypatch) -> None:
    logger = _LoggerSpy()
    monkeypatch.setattr(group_api, "_log", logger)
    monkeypatch.setattr(group_api, "get_bot", lambda: _NotOkBot())

    result = asyncio.run(group_api.say("group-1", MessageChain("hello")))

    assert result["say"]["status"] == "failed"
    assert logger.errors
    assert "retcode" in logger.errors[0]
