from __future__ import annotations

from pathlib import Path

import pytest

from TIYA.model import character


def _write_prompt(path: Path, *, description: str, content: str) -> None:
    path.write_text(
        f"---\ndescription: {description}\n---\n{content}\n",
        encoding="utf-8",
    )


def test_character_directory_loads_group_and_private_prompts(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "characters"
    role = root / "抹布"
    role.mkdir(parents=True)
    _write_prompt(role / "group.md", description="group", content="群聊人格")
    _write_prompt(role / "private.md", description="private", content="私聊人格")
    monkeypatch.setattr(character, "CHARACTER_DIR", root)
    character._GLOBAL_CHARACTER.clear()

    loaded = character.add_character("抹布")
    current = character.get_character("抹布")

    assert loaded is current
    assert current.personality.strip() == "群聊人格"
    assert current.private_personality.strip() == "私聊人格"
    assert current.group_metadata["description"] == "group"
    assert current.private_metadata["description"] == "private"
    assert current.fav._mapping._save_path == role / "fav_index.json"


def test_incomplete_character_directory_is_rejected(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "characters"
    role = root / "broken"
    role.mkdir(parents=True)
    _write_prompt(role / "group.md", description="group", content="群聊人格")
    monkeypatch.setattr(character, "CHARACTER_DIR", root)
    character._GLOBAL_CHARACTER.clear()

    with pytest.raises(FileNotFoundError):
        character.add_character("broken")

    assert "broken" not in character._GLOBAL_CHARACTER
