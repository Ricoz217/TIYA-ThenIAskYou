from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import TIYA.auto_fav as auto_fav_module
from TIYA.auto_fav import AutoFav


def _make_fav(monkeypatch, index_file, entries) -> AutoFav:
    monkeypatch.setattr(
        auto_fav_module,
        "get_file_path_async",
        AsyncMock(side_effect=lambda hash_name: index_file.parent / hash_name),
    )
    monkeypatch.setattr(auto_fav_module, "set_fav_title", lambda *_: None)
    monkeypatch.setattr(
        auto_fav_module,
        "SETTING_CFG",
        SimpleNamespace(Groups=SimpleNamespace(AutoFavExpire=0)),
    )
    fav = AutoFav(index_file)
    asyncio.run(fav.add_favs(entries))
    return fav


def test_get_fav_removes_missing_hash_from_every_title_and_persists(
        monkeypatch,
        tmp_path,
) -> None:
    hash_name = "a" * 32
    index_file = tmp_path / "fav_index.json"
    fav = _make_fav(
        monkeypatch,
        index_file,
        [(hash_name, "title-a"), (hash_name, "title-b")],
    )
    monkeypatch.setattr(auto_fav_module, "check_file_exists", lambda _: False)

    assert fav.get_fav("title-a") == ""
    assert fav.fav_list == {}
    assert AutoFav(index_file).fav_list == {}


def test_get_fav_keeps_existing_hash(monkeypatch, tmp_path) -> None:
    hash_name = "b" * 32
    fav = _make_fav(
        monkeypatch,
        tmp_path / "fav_index.json",
        [(hash_name, "title")],
    )
    monkeypatch.setattr(auto_fav_module, "check_file_exists", lambda _: True)

    assert fav.get_fav("title") == hash_name
    assert fav.fav_list == {"title": {hash_name}}


def test_combined_fav_view_removes_missing_hash_from_source_indexes(
        monkeypatch,
        tmp_path,
) -> None:
    hash_name = "c" * 32
    first = _make_fav(
        monkeypatch,
        tmp_path / "first.json",
        [(hash_name, "title")],
    )
    second = _make_fav(
        monkeypatch,
        tmp_path / "second.json",
        [(hash_name, "title")],
    )
    monkeypatch.setattr(auto_fav_module, "check_file_exists", lambda _: False)

    combined = first | second

    assert combined.get_fav("title") == ""
    assert first.fav_list == {}
    assert second.fav_list == {}
