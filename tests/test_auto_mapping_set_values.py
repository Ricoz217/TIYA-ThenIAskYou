from __future__ import annotations

import json

import pytest

from TIYA.utils import AutoMapping


def test_set_value_type_round_trips_first_level_values(tmp_path):
    storage_path = tmp_path / "mapping.json"
    mapping = AutoMapping[set[str]](
        storage_path,
        persist_period="UPDATE",
        value_type=set,
    )

    mapping["title"] = {"hash_1", "hash_2"}

    saved = json.loads(storage_path.read_text(encoding="utf-8"))
    assert sorted(saved["data"]["title"]["data"]) == ["hash_1", "hash_2"]

    loaded = AutoMapping[set[str]](
        storage_path,
        persist_period="UPDATE",
        value_type=set,
    )
    assert loaded["title"] == {"hash_1", "hash_2"}


def test_set_value_type_does_not_convert_nested_sets(tmp_path):
    mapping = AutoMapping[set[object]](
        tmp_path / "mapping.json",
        persist_period="UPDATE",
        value_type=set,
    )

    with pytest.raises(TypeError, match="not JSON serializable"):
        mapping["title"] = {("nested", frozenset({"value"}))}


def test_default_value_type_does_not_enable_set_conversion(tmp_path):
    mapping = AutoMapping(tmp_path / "mapping.json", persist_period="UPDATE")

    with pytest.raises(TypeError, match="not JSON serializable"):
        mapping["title"] = {"hash"}


def test_touch_persists_mutated_set(tmp_path):
    storage_path = tmp_path / "mapping.json"
    mapping = AutoMapping[set[str]](
        storage_path,
        persist_period="UPDATE",
        value_type=set,
    )
    values = mapping.setdefault("title", set())

    values.add("hash")
    mapping.touch("title")

    loaded = AutoMapping[set[str]](
        storage_path,
        persist_period="UPDATE",
        value_type=set,
    )
    assert loaded["title"] == {"hash"}


def test_set_value_type_applies_to_defaults_and_bulk_updates(tmp_path):
    mapping = AutoMapping[set[str]](
        tmp_path / "mapping.json",
        value_type=set,
        default={"default": ["hash_1"]},
    )

    mapping.update_from_dict({"updated": ["hash_2"]})

    assert mapping["default"] == {"hash_1"}
    assert mapping["updated"] == {"hash_2"}
