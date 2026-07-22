from __future__ import annotations

import asyncio
import types
from functools import wraps

import pytest

import TIYA.dialog.group_dialog as group_dialog
import TIYA.setu.setu as setu_module
from TIYA.command import CommandPermission
from TIYA.model.message import GroupMsg, ReplyMsg, TextMsg
from TIYA.setu.setu import Illust, PixivFreeze, Setu


def _async_test(func):
    @wraps(func)
    def runner(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return runner


def _illust(
        illust_id: int = 101,
        *,
        pages: list[int] | None = None,
        tags: set[str] | None = None,
) -> Illust:
    return Illust(
        id=illust_id,
        artist_id=456,
        artist_name="artist",
        title=f"title-{illust_id}",
        chunk=1,
        pages=pages or [0],
        tags=tags or {"tag-a", "tag-b"},
        bookmark=3_000,
        NSFW=True,
    )


def _setu(*, history_limit: int = 1_000) -> tuple[Setu, list[bool]]:
    setu = object.__new__(Setu)
    setu._group_id = "group"
    setu.nsfw = True
    setu._max_loaded_chunks = 20
    setu._history_limit = history_limit
    setu._chunk_index_now = 1
    setu._chunks = {}
    setu._fetch_lock = asyncio.Lock()
    setu._query_lock = asyncio.Lock()
    setu._data_lock = asyncio.Lock()
    setu._available_illusts = {}
    setu._used_illusts = {}
    setu._history = {}
    setu._freeze = {}
    setu._unfreeze_task = None
    setu._tags_weight = {}
    setu._artists_weight = {}
    setu._ban_tags = set()
    setu._ban_artists = set()
    setu._ban_illusts = set()
    setu._ban_pages = {}
    persisted: list[bool] = []
    setu.persist = lambda: persisted.append(True)
    return setu, persisted


def _add_sent_result(setu: Setu, message_id: str = "message-1") -> Illust:
    illust = _illust()
    setu._used_illusts[illust.iid] = illust.copy()
    setu._history[message_id] = illust.copy()
    return illust


@_async_test
async def test_score_is_clamped_quantized_and_overwrites_user_vote() -> None:
    setu, _ = _setu()
    illust = _add_sent_result(setu)

    high = await setu.respond_to_result("message-1", "user", "114514")
    tiny = await setu.respond_to_result("message-1", "user", "0.005")
    low = await setu.respond_to_result("message-1", "user", "-1919810")

    assert high
    assert tiny
    assert low
    assert setu._used_illusts[illust.iid].scores == {"0": {"user": 0.0}}


@_async_test
async def test_five_scores_rebuild_tag_and_artist_weights() -> None:
    setu, _ = _setu()
    illust = _add_sent_result(setu)

    for index, score in enumerate((2, 4, 6, 8, 10), start=1):
        await setu.respond_to_result("message-1", f"user-{index}", str(score))

    assert setu._tags_weight == pytest.approx({"tag-a": 0.6, "tag-b": 0.6})
    assert setu._artists_weight == pytest.approx({str(illust.artist_id): 0.6})


@_async_test
async def test_flat_commands_accept_arbitrary_tags_and_check_state() -> None:
    setu, _ = _setu()
    illust = _add_sent_result(setu)

    await setu.respond_to_result("message-1", "user", "--ban-id")
    await setu.respond_to_result(
        "message-1",
        "user",
        '--ban-tags "never seen tag" another-tag',
    )
    checked = await setu.respond_to_result("message-1", "user", "--check-ban")
    await setu.respond_to_result(
        "message-1",
        "user",
        '--unban -tags "never seen tag"',
    )

    assert illust.iid in setu._ban_illusts
    assert "never seen tag" not in setu._ban_tags
    assert "another-tag" in setu._ban_tags
    assert illust.iid in checked
    assert "another-tag" in checked


@_async_test
async def test_flat_page_artist_and_info_commands() -> None:
    setu, _ = _setu()
    illust = _add_sent_result(setu)

    info = await setu.respond_to_result("message-1", "user", "--info")
    await setu.respond_to_result("message-1", "user", "--ban-page")
    await setu.respond_to_result("message-1", "user", "--ban-artist")
    pages = await setu.respond_to_result("message-1", "user", "--check-ban-page")
    artists = await setu.respond_to_result("message-1", "user", "--check-ban-artist")
    await setu.respond_to_result(
        "message-1",
        "user",
        f"--unban -id {illust.iid} -page 0 -artist {illust.artist_id}",
    )

    assert f"artworks/{illust.iid}" in info
    assert f"users/{illust.artist_id}" in info
    assert illust.iid in pages
    assert str(illust.artist_id) in artists
    assert setu._ban_pages == {}
    assert setu._ban_artists == set()


@_async_test
async def test_unban_uses_explicit_targets_and_supports_both_commas() -> None:
    setu, persisted = _setu()
    replied_illust = _add_sent_result(setu)
    setu._ban_illusts = {replied_illust.iid, "999"}
    setu._ban_pages = {"888": [0, 1]}
    setu._ban_artists = {456, 777}
    setu._ban_tags = {"tag-a", "tag-b", "tag-c"}

    response = await setu.respond_to_result(
        "message-1",
        "user",
        "--unban -id 888 -page 0 -artist 777 -tags tag-a，tag-b,tag-a",
    )
    await setu.respond_to_result("message-1", "user", "--unban -id 999")

    assert "已解除" in response
    assert setu._ban_illusts == {replied_illust.iid}
    assert setu._ban_pages == {"888": [1]}
    assert setu._ban_artists == {456}
    assert setu._ban_tags == {"tag-c"}
    assert persisted == [True, True]


@_async_test
async def test_unban_rejects_page_without_illust_id() -> None:
    setu, persisted = _setu()
    _add_sent_result(setu)
    setu._ban_pages = {"101": [0]}

    response = await setu.respond_to_result(
        "message-1",
        "user",
        "--unban -page 0",
    )

    assert "-page" in response
    assert "-id" in response
    assert setu._ban_pages == {"101": [0]}
    assert persisted == []


@_async_test
async def test_global_option_has_an_independent_permission_gate() -> None:
    source, _ = _setu()
    target, _ = _setu()
    illust = _add_sent_result(source)

    denied = await source.respond_to_result(
        "message-1",
        "user",
        "--ban-id --global",
        allow_global=False,
        global_targets=(source, target),
    )

    assert "权限" in denied
    assert source._ban_illusts == set()
    assert target._ban_illusts == set()

    allowed = await source.respond_to_result(
        "message-1",
        "admin",
        "--ban-id --global",
        allow_global=True,
        global_targets=(source, target),
    )

    assert "2" in allowed
    assert source._ban_illusts == {illust.iid}
    assert target._ban_illusts == {illust.iid}

    unbanned = await source.respond_to_result(
        "message-1",
        "admin",
        f"--unban -id {illust.iid} --global",
        allow_global=True,
        global_targets=(source, target),
    )

    assert "2" in unbanned
    assert source._ban_illusts == set()
    assert target._ban_illusts == set()


@_async_test
async def test_non_interaction_reply_propagates_and_expired_command_is_handled() -> None:
    setu, _ = _setu()
    illust = _illust()
    setu._history["message-1"] = illust

    assert await setu.respond_to_result("message-1", "user", "真好看") is None

    expired = await setu.respond_to_result("message-1", "user", "8")
    unknown = await setu.respond_to_result("message-1", "user", "--unknown")

    assert "过期" in expired
    assert "错误" in unknown


@_async_test
async def test_mark_used_commits_history_used_and_freeze_together() -> None:
    setu, persisted = _setu(history_limit=1)
    frozen = _illust(pages=[0, 1])
    setu._available_illusts[frozen.iid] = frozen.copy()
    setu._freeze["freeze-1"] = PixivFreeze(
        illusts={frozen.iid: frozen.copy()},
        freeze_id="freeze-1",
    )

    await setu.mark_used(
        message_id="message-1",
        illust=_illust(pages=[0]),
        freeze_id="freeze-1",
    )
    await setu.mark_used(
        message_id="message-2",
        illust=_illust(pages=[1]),
        freeze_id="freeze-1",
    )

    assert setu._used_illusts[frozen.iid].pages == [0, 1]
    assert frozen.iid not in setu._available_illusts
    assert list(setu._history) == ["message-2"]
    assert setu._history["message-2"].pages == [1]
    assert "freeze-1" not in setu._freeze
    assert persisted == [True, True]


@_async_test
async def test_query_search_keeps_only_group_available_pages(monkeypatch) -> None:
    setu, _ = _setu()
    candidate = _illust(pages=[0, 1])
    available = candidate.copy()
    available.pages = [1]
    used = candidate.copy()
    used.pages = [0]
    setu._chunks = {"1": {candidate.iid}}
    setu._available_illusts[candidate.iid] = available
    setu._used_illusts[candidate.iid] = used

    class _FakeApi:
        async def pixiv_search(self, **kwargs):
            return [{"iid": candidate.iid, "illust": candidate.copy()}]

    monkeypatch.setattr(
        setu_module.AsyncPixivApi,
        "get_api",
        lambda: _FakeApi(),
    )

    results = await setu._get_query_setu(["tag-a"], 1)

    assert results[candidate.iid].pages == [1]


@_async_test
async def test_freeze_rejects_page_missing_from_available() -> None:
    setu, _ = _setu()
    available = _illust(pages=[1])
    setu._available_illusts[available.iid] = available

    frozen = await setu.freeze(_illust(pages=[0]))

    assert frozen is None
    assert setu._available_illusts[available.iid].pages == [1]


@_async_test
async def test_random_setu_continues_when_nothing_can_be_frozen(
        monkeypatch,
) -> None:
    setu, _ = _setu()
    available = _illust(pages=[1])
    setu._available_illusts[available.iid] = available
    stale = _illust(pages=[0])
    payload = object()
    maker_calls = []

    async def get_stale_result(count: int):
        return {stale.iid: stale}

    async def payload_maker(*args, **kwargs):
        maker_calls.append((args, kwargs))
        return payload

    setu._get_random_setu = get_stale_result
    setu._pixiv_payload_maker = payload_maker
    monkeypatch.setattr(
        setu_module,
        "SETTING_CFG",
        types.SimpleNamespace(
            SETU=types.SimpleNamespace(
                SubstituteMultiplier=4,
                SendSETUTaskTimeout=600,
                UnfreezeTimesLimit=5,
            )
        ),
    )

    result = await setu.random_setu()

    assert result is payload
    assert len(maker_calls) == 1
    args, kwargs = maker_calls[0]
    assert args == (stale,)
    assert kwargs["freeze_id"] == ""


def test_used_page_receives_a_rating_penalty() -> None:
    setu, _ = _setu()
    illust = _illust()
    unused_score = setu.get_rating(illust)
    setu._used_illusts[illust.iid] = illust.copy()

    used_score = setu.get_rating(illust)

    assert used_score < unused_score


def test_illust_scores_round_trip() -> None:
    illust = _illust()
    illust.scores = {"0": {"user": 0.889}}

    restored = Illust.from_dict(illust.to_dict())

    assert restored is not None
    assert restored.scores == illust.scores


@_async_test
async def test_used_history_eviction_rebuilds_rating_weights() -> None:
    setu, _ = _setu(history_limit=1)
    old = _illust(101)
    old.scores = {
        "0": {f"user-{index}": 1.0 for index in range(5)}
    }
    setu._used_illusts[old.iid] = old
    setu._refresh_interaction_weights_unlocked()
    assert setu._tags_weight

    await setu.mark_used(
        message_id="message-2",
        illust=_illust(202),
        freeze_id="",
    )

    assert list(setu._used_illusts) == ["202"]
    assert setu._tags_weight == {}
    assert setu._artists_weight == {}


@_async_test
async def test_setu_trigger_passes_global_permission_and_targets(monkeypatch) -> None:
    calls = []

    class FakeSetu:
        async def respond_to_result(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "done"

    class FakeCommandDialog:
        async def get_user_permission(self, user_id: str):
            return CommandPermission.BOT_ADMIN

    dialog = object.__new__(group_dialog.GroupMainDialog)
    dialog.SETU = FakeSetu()
    dialog.host = types.SimpleNamespace(command_dialog=FakeCommandDialog())
    dialog.message_history = types.SimpleNamespace(bot_msg_id={"setu-message"})
    sent = []

    async def say(message):
        sent.append(message)
        return []

    dialog.say = say
    other_setu = FakeSetu()
    monkeypatch.setattr(
        group_dialog,
        "QQ_GROUPS",
        {
            "one": types.SimpleNamespace(main_dialog=types.SimpleNamespace(SETU=dialog.SETU)),
            "two": types.SimpleNamespace(main_dialog=types.SimpleNamespace(SETU=other_setu)),
        },
    )
    message = GroupMsg(
        group_id="group",
        msg_id="reply-message",
        user_id="admin",
        username="Admin",
        nickname="",
        content=[ReplyMsg(msg_id="setu-message"), TextMsg(text="--ban-id --global")],
    )

    handled = await dialog._setu_trigger(message)

    assert handled is True
    assert calls[0][0] == ("setu-message", "admin", "--ban-id --global")
    assert calls[0][1]["allow_global"] is True
    assert calls[0][1]["global_targets"] == (dialog.SETU, other_setu)
    assert len(sent) == 1


@_async_test
async def test_setu_trigger_schedules_pixiv_artwork_and_user_links(monkeypatch) -> None:
    queued = []
    setu_calls = []
    sent = []

    class FakeQueue:
        def add_task(self, coroutine, *, timeout):
            queued.append((coroutine, timeout))

    dialog = object.__new__(group_dialog.GroupMainDialog)
    dialog.host = types.SimpleNamespace(
        group_config=types.SimpleNamespace(setu=True),
        aqueue=FakeQueue(),
    )

    async def setu(args: str):
        setu_calls.append(args)

    async def say(message):
        sent.append(message)
        return []

    dialog.setu = setu
    dialog.say = say
    monkeypatch.setattr(
        group_dialog,
        "BASE_CFG",
        types.SimpleNamespace(Module=types.SimpleNamespace(setu=True)),
    )
    monkeypatch.setattr(
        group_dialog,
        "SETTING_CFG",
        types.SimpleNamespace(
            SETU=types.SimpleNamespace(SendSETUTaskTimeout=600),
        ),
    )
    message = GroupMsg(
        group_id="group",
        msg_id="message-with-links",
        user_id="user",
        username="User",
        nickname="",
        content=[TextMsg(
            text=(
                "https://www.pixiv.net/artworks/123?share=1 "
                "pixiv.net/users/456 "
                "https://www.pixiv.net/artworks/123"
            )
        )],
    )

    handled = await dialog._setu_trigger(message)
    await asyncio.gather(*(coroutine for coroutine, _ in queued))

    assert handled is True
    assert setu_calls == [
        "--count 5 --illust 123 --origin",
        "--count 5 --artist 456",
    ]
    assert [timeout for _, timeout in queued] == [600, 600]
    assert len(sent) == 1


@_async_test
async def test_setu_trigger_ignores_pixiv_text_inside_another_domain(monkeypatch) -> None:
    class FakeQueue:
        def add_task(self, coroutine, *, timeout):
            coroutine.close()
            raise AssertionError("unexpected setu task")

    dialog = object.__new__(group_dialog.GroupMainDialog)
    dialog.host = types.SimpleNamespace(
        group_config=types.SimpleNamespace(setu=True),
        aqueue=FakeQueue(),
    )
    monkeypatch.setattr(
        group_dialog,
        "BASE_CFG",
        types.SimpleNamespace(Module=types.SimpleNamespace(setu=True)),
    )
    message = GroupMsg(
        group_id="group",
        msg_id="not-pixiv",
        user_id="user",
        username="User",
        nickname="",
        content=[TextMsg(text="https://example.com/pixiv.net/artworks/123")],
    )

    assert await dialog._setu_trigger(message) is False
