from __future__ import annotations

import asyncio
import types
from functools import wraps

import TIYA.setu.setu as setu_module
from TIYA.setu.setu import Illust, PixivFreeze, Setu


def _async_test(func):
    @wraps(func)
    def runner(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return runner


def _illust(illust_id: int, *, chunk: int) -> Illust:
    return Illust(
        id=illust_id,
        artist_id=456,
        artist_name="artist",
        title=f"title-{illust_id}",
        chunk=chunk,
        pages=[0, 1],
        tags={"tag", "tag-2"},
        bookmark=3_000,
        NSFW=True,
        attempts={"0": 2},
    )


@_async_test
async def test_setu_restores_persisted_group_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(setu_module, "GROUPS_DIR", tmp_path)
    monkeypatch.setattr(
        setu_module,
        "SETTING_CFG",
        types.SimpleNamespace(
            SETU=types.SimpleNamespace(
                MaxGroupLoadedChunks=20,
                ResultHistoryLimit=1_000,
                SendSETUTaskTimeout=600,
                UnfreezeTimesLimit=5,
            )
        ),
    )

    original = Setu("group-1", nsfw=True)
    available = _illust(101, chunk=2)
    used = _illust(102, chunk=2)
    history = _illust(103, chunk=3)
    frozen = _illust(104, chunk=3)

    original._chunk_index_now = 3
    original._chunks = {"2": {available.iid, used.iid}, "3": {history.iid, frozen.iid}}
    original._available_illusts = {available.iid: available}
    original._used_illusts = {used.iid: used}
    history.scores = {"0": {"user": 0.75}}
    original._history = {"message-103": history}
    original._freeze = {
        "freeze-1": PixivFreeze(
            illusts={frozen.iid: frozen},
            freeze_id="freeze-1",
            freeze_time=123.0,
        )
    }
    original._tags_weight = {"tag": 0.8}
    original._artists_weight = {"456": 0.9}
    original._ban_tags = {"ban-tag"}
    original._ban_artists = {789}
    original._ban_illusts = {"999"}
    original._ban_pages = {"101": [1]}
    original.persist()

    restored = Setu("group-1")

    assert restored._chunk_index_now == 3
    assert restored._chunks == original._chunks
    assert restored._available_illusts[available.iid].attempts == {"0": 2}
    assert restored._used_illusts[used.iid].title == used.title
    assert restored._history["message-103"].chunk == 3
    assert restored._history["message-103"].scores == {"0": {"user": 0.75}}
    assert restored._freeze["freeze-1"].illusts[frozen.iid].attempts == {"0": 2}
    assert restored._tags_weight == {"tag": 0.8}
    assert restored._artists_weight == {"456": 0.9}
    assert restored._ban_tags == {"ban-tag"}
    assert restored._ban_artists == {789}
    assert restored._ban_illusts == {"999"}
    assert restored._ban_pages == {"101": [1]}


@_async_test
async def test_setu_load_dict_ignores_malformed_entries(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(setu_module, "GROUPS_DIR", tmp_path)
    monkeypatch.setattr(
        setu_module,
        "SETTING_CFG",
        types.SimpleNamespace(
            SETU=types.SimpleNamespace(
                MaxGroupLoadedChunks=20,
                ResultHistoryLimit=1_000,
                SendSETUTaskTimeout=600,
                UnfreezeTimesLimit=5,
            )
        ),
    )
    setu = Setu("group-2")

    setu.load_dict(
        {
            "data": {
                "chunks": {"bad": "not-a-list", "2": [101, None]},
                "available": {"broken": {}},
                "used": [],
                "freeze": {"broken": "not-a-dict"},
                "ban_artists": None,
                "ban_pages": {"101": ["bad-page", 1, 1]},
                "index_now": "bad-index",
            }
        }
    )

    assert setu._chunks == {"2": {"101"}}
    assert setu._available_illusts == {}
    assert setu._used_illusts == {}
    assert setu._freeze == {}
    assert setu._ban_artists == set()
    assert setu._ban_pages == {"101": [1]}
    assert setu._chunk_index_now == 1
