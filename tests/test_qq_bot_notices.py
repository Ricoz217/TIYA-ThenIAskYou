import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

with patch.dict(os.environ, {"LOG_FORMAT": "%(message)s"}):
    import TIYA.QQ_bot as qq_bot


def test_group_ban_notice_forces_member_list_refresh(monkeypatch) -> None:
    group = SimpleNamespace(update_member=AsyncMock())
    monkeypatch.setattr(qq_bot, "QQ_GROUPS", {"123456": group})

    asyncio.run(qq_bot._refresh_group_members_for_ban_notice({
        "notice_type": "group_ban",
        "group_id": 123456,
        "sub_type": "ban",
        "user_id": 654321,
        "duration": 600,
    }))

    group.update_member.assert_awaited_once_with(force=True)


def test_group_unban_notice_also_forces_member_list_refresh(monkeypatch) -> None:
    group = SimpleNamespace(update_member=AsyncMock())
    monkeypatch.setattr(qq_bot, "QQ_GROUPS", {"123456": group})

    asyncio.run(qq_bot._refresh_group_members_for_ban_notice({
        "notice_type": "group_ban",
        "group_id": "123456",
        "sub_type": "lift_ban",
        "user_id": "654321",
        "duration": 0,
    }))

    group.update_member.assert_awaited_once_with(force=True)


def test_group_ban_notice_ignores_groups_not_loaded(monkeypatch) -> None:
    group = SimpleNamespace(update_member=AsyncMock())
    monkeypatch.setattr(qq_bot, "QQ_GROUPS", {"other-group": group})

    asyncio.run(qq_bot._refresh_group_members_for_ban_notice({
        "notice_type": "group_ban",
        "group_id": "123456",
    }))

    group.update_member.assert_not_awaited()


@pytest.mark.parametrize("sub_type", ["invite", "approve"])
def test_self_group_increase_refreshes_groups_after_delay(monkeypatch, sub_type: str) -> None:
    sleep = AsyncMock()
    reload_groups = AsyncMock()
    monkeypatch.setattr(qq_bot.asyncio, "sleep", sleep)
    monkeypatch.setattr(qq_bot, "hot_reload", reload_groups)

    refreshed = asyncio.run(qq_bot._refresh_groups_for_self_increase_notice({
        "notice_type": "group_increase",
        "sub_type": sub_type,
        "group_id": 123456,
        "user_id": 3955071851,
        "self_id": 3955071851,
    }))

    assert refreshed is True
    sleep.assert_awaited_once_with(10.0)
    reload_groups.assert_awaited_once_with()


def test_group_increase_for_other_member_does_not_refresh_groups(monkeypatch) -> None:
    sleep = AsyncMock()
    reload_groups = AsyncMock()
    monkeypatch.setattr(qq_bot.asyncio, "sleep", sleep)
    monkeypatch.setattr(qq_bot, "hot_reload", reload_groups)

    refreshed = asyncio.run(qq_bot._refresh_groups_for_self_increase_notice({
        "notice_type": "group_increase",
        "sub_type": "approve",
        "group_id": 123456,
        "user_id": 654321,
        "self_id": 3955071851,
    }))

    assert refreshed is False
    sleep.assert_not_awaited()
    reload_groups.assert_not_awaited()


def test_self_group_increase_logs_refresh_failure(monkeypatch) -> None:
    logger = SimpleNamespace(error=Mock(), debug=Mock())
    monkeypatch.setattr(qq_bot.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(qq_bot, "hot_reload", AsyncMock(side_effect=RuntimeError("refresh failed")))
    monkeypatch.setattr(qq_bot, "_log", logger)

    refreshed = asyncio.run(qq_bot._refresh_groups_for_self_increase_notice({
        "notice_type": "group_increase",
        "sub_type": "invite",
        "group_id": "123456",
        "user_id": "3955071851",
        "self_id": "3955071851",
    }))

    assert refreshed is False
    logger.error.assert_called_once()
    assert "123456" in logger.error.call_args.args[0]
    assert "refresh failed" in logger.error.call_args.args[0]
