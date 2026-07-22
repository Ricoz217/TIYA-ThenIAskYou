from __future__ import annotations

import json

import TIYA.setu.setu as setu_module
from TIYA.setu.setu import (
    Illustration,
    IllustType,
    PixivStorageChunk,
)


def _illustration(
        illust_id: int = 123,
        *,
        title: str = "test",
) -> Illustration:
    return Illustration(
        id=illust_id,
        title=title,
        type=IllustType.ILLUST,
        caption="caption",
        restrict=0,
        user={
            "id": 456,
            "name": "artist",
            "is_followed": False,
        },
        tags=[
            {
                "name": "tag",
                "translated_name": None,
            }
        ],
        page_count=1,
        sanity_level=2,
        total_view=100,
        total_bookmarks=20,
        illust_ai_type=0,
        img_urls={"0": "https://example.com/image.jpg"},
        is_bookmarked=False,
        NSFW=False,
        visible=True,
    )


def test_chunk_keeps_illustration_objects_in_memory(
        tmp_path,
        monkeypatch,
) -> None:
    monkeypatch.setattr(setu_module, "PIXIV_CHUNKS_DIR", tmp_path)
    chunk = PixivStorageChunk(index=1, loaded=True)
    illustration = _illustration()

    stored, summary = chunk.add(illustration)

    assert chunk.data[illustration.iid] is illustration
    assert stored is illustration
    assert summary.id == illustration.id


def test_chunk_serializes_illustrations_only_at_persistence_boundary(
        tmp_path,
        monkeypatch,
) -> None:
    monkeypatch.setattr(setu_module, "PIXIV_CHUNKS_DIR", tmp_path)
    chunk = PixivStorageChunk(index=2, loaded=True)
    illustration = _illustration()

    chunk.add(illustration)

    saved = json.loads(
        (tmp_path / "chunk_2.json").read_text(encoding="utf-8")
    )
    assert isinstance(saved["data"][illustration.iid], dict)
    assert saved["data"][illustration.iid]["type"] == "illust"


def test_chunk_load_restores_illustration_objects(
        tmp_path,
        monkeypatch,
) -> None:
    monkeypatch.setattr(setu_module, "PIXIV_CHUNKS_DIR", tmp_path)
    illustration = _illustration()
    payload = {
        "index": 3,
        "last_update": 123.0,
        "data": {
            illustration.iid: illustration.to_dict(),
        },
    }
    (tmp_path / "chunk_3.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    chunk = PixivStorageChunk(index=3)
    chunk.load()
    loaded, summary = chunk.get(illustration.iid)

    assert isinstance(chunk.data[illustration.iid], Illustration)
    assert loaded is chunk.data[illustration.iid]
    assert loaded == illustration
    assert summary.id == illustration.id


def test_chunk_replaces_existing_illustration_object(
        tmp_path,
        monkeypatch,
) -> None:
    monkeypatch.setattr(setu_module, "PIXIV_CHUNKS_DIR", tmp_path)
    chunk = PixivStorageChunk(index=4, loaded=True)
    original = _illustration(title="old")
    updated = _illustration(title="new")
    chunk.add(original)

    stored, _ = chunk.add(updated)

    assert chunk.data[updated.iid] is updated
    assert stored is updated
    assert stored.title == "new"
