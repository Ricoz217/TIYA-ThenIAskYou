import asyncio

from TIYA.setu.setu import Setu


def _setu() -> tuple[Setu, list[bool]]:
    setu = object.__new__(Setu)
    setu._data_lock = asyncio.Lock()
    setu._ban_pages = {}
    setu._ban_illusts = set()
    setu._ban_artists = set()
    setu._ban_tags = set()
    persisted: list[bool] = []
    setu.persist = lambda: persisted.append(True)
    return setu, persisted


def test_mark_unban_combines_explicit_targets_in_one_persist() -> None:
    setu, persisted = _setu()
    setu._ban_pages = {
        "101": [0, 1, 2],
        "102": [1],
    }
    setu._ban_illusts = {"101", "102"}
    setu._ban_artists = {456, 789}
    setu._ban_tags = {"tag-a", "tag-b"}

    asyncio.run(setu.mark_unban(
        illust_id="101",
        page=1,
        artist_id=456,
        tags=("tag-a", "missing"),
    ))

    assert setu._ban_pages == {"101": [0, 2], "102": [1]}
    assert setu._ban_illusts == {"101", "102"}
    assert setu._ban_artists == {789}
    assert setu._ban_tags == {"tag-b"}
    assert persisted == [True]


def test_mark_unban_illust_id_is_idempotent() -> None:
    setu, persisted = _setu()
    setu._ban_illusts = {"101", "102"}

    asyncio.run(setu.mark_unban(illust_id="101"))

    assert setu._ban_illusts == {"102"}
    assert persisted == [True]
