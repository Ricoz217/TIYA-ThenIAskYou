from __future__ import annotations

import asyncio

import pytest

from TIYA.setu.setu import (
    AsyncPixivApi,
    Illustration,
    IllustType,
    PixivStorageChunk,
)


def _illustration(
        illust_id: int,
        *,
        tags: list[str],
        bookmark: int,
) -> Illustration:
    return Illustration(
        id=illust_id,
        title=str(illust_id),
        type=IllustType.ILLUST,
        caption="",
        restrict=0,
        user={
            "id": 100,
            "name": "artist",
            "is_followed": False,
        },
        tags=[
            {
                "name": tag,
                "translated_name": None,
            }
            for tag in tags
        ],
        page_count=1,
        sanity_level=2,
        total_view=1,
        total_bookmarks=bookmark,
        illust_ai_type=0,
        img_urls={"0": "https://example.com/image.jpg"},
        is_bookmarked=False,
        NSFW=False,
        visible=True,
    )


def _api() -> AsyncPixivApi:
    api = object.__new__(AsyncPixivApi)

    async def always_available(_: Illustration) -> bool:
        return True

    api.check_illustration_visible = always_available
    return api


def test_result_without_tag_match_is_filtered_even_with_full_bookmark_score() -> None:
    api = _api()
    illustration = _illustration(1, tags=["other"], bookmark=10_000)

    result = asyncio.run(
        api._tags_searching(
            ["target"],
            {"1": illustration},
        )
    )

    assert api.bookmark_score(illustration.bookmark) == pytest.approx(1.0)
    assert result == {}


def test_maximum_tag_and_bookmark_score_reaches_one() -> None:
    api = _api()
    illustration = _illustration(
        1,
        tags=["first", "second"],
        bookmark=10_000,
    )

    result = asyncio.run(
        api._tags_searching(
            ["first", "second"],
            {"1": illustration},
            tags_weight={"first": 1.0, "second": 1.0},
            artist_weight={"100": 1.0},
        )
    )

    assert list(result) == ["1"]
    assert result["1"]["illustration"] is illustration
    assert result["1"]["score"] == pytest.approx(1.0)


def test_tag_match_is_a_positive_reward_over_bookmark_weight() -> None:
    api = _api()
    matched = _illustration(1, tags=["target"], bookmark=10_000)
    unmatched = _illustration(2, tags=["other"], bookmark=10_000)

    result = asyncio.run(
        api._tags_searching(
            ["target"],
            {"1": matched, "2": unmatched},
            expect_count=2,
        )
    )

    assert list(result) == ["1"]
    assert result["1"]["illustration"] is matched
    assert result["1"]["score"] > 0.6


def test_empty_search_data_returns_empty_mapping() -> None:
    api = _api()

    result = asyncio.run(api._tags_searching(["target"], {}))

    assert result == {}


def test_loaded_chunk_search_range_keeps_complete_illust_ids() -> None:
    api = _api()
    api._chunks = {
        1: PixivStorageChunk(
            index=1,
            data={
                "123456": _illustration(123456, tags=["target"], bookmark=10_000),
                "789012": _illustration(789012, tags=["target"], bookmark=10_000),
            },
        )
    }

    assert api._get_search_range() == {1: {"123456", "789012"}}


def test_local_search_keeps_group_results_when_supplemental_range_is_disjoint() -> None:
    api = _api()
    group_illustration = _illustration(101, tags=["target"], bookmark=10_000)
    supplemental_illustration = _illustration(202, tags=["target"], bookmark=10_000)
    group_chunk = PixivStorageChunk(index=1)
    supplemental_chunk = PixivStorageChunk(index=2)
    _, group_illust = group_chunk.to_obj(group_illustration)
    _, supplemental_illust = supplemental_chunk.to_obj(supplemental_illustration)

    api._get_search_range = lambda available=None: (
        {1: {"101"}} if available is not None else {2: {"202"}}
    )
    api._get_search_illustrations = lambda search_range: (
        ({"101": group_illustration}, {"101": group_illust})
        if 1 in search_range
        else ({"202": supplemental_illustration}, {"202": supplemental_illust})
    )

    results = asyncio.run(
        api.pixiv_search_local(
            ["target"], expect_count=3, group_available={"101": group_illust}
        )
    )

    assert {result["iid"] for result in results} == {"101", "202"}
