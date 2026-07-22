from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

import TIYA.agent.group_chat_agent as group_chat_agent_module
import TIYA.auto_fav as auto_fav_module
from TIYA.LLM_connect import TextPrompt
from TIYA.agent.group_chat_agent import GroupChatAgent, GroupChatSpeaker
from TIYA.auto_fav import AutoFav


class DummyHistory:
    def get_last_message(self, count: int):
        return []

    def get_new_message_start_from(self, msg_id: str):
        return []


class DummyLLM:
    def __init__(self, response):
        self.response = response

    async def ask(self, prompt: str, timeout: float):
        return self.response


class DummyFav:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    @property
    def fav_list(self):
        return {title: {hash_name} for title, hash_name in self.mapping.items()}

    def get_fav(self, title: str) -> str:
        return self.mapping.get(title, "")

    def __or__(self, other: DummyFav) -> DummyFav:
        return DummyFav(self.mapping | other.mapping)


def _set_auto_fav_config(monkeypatch) -> None:
    monkeypatch.setattr(
        auto_fav_module,
        "SETTING_CFG",
        SimpleNamespace(Groups=SimpleNamespace(AutoFavExpire=0)),
    )


def test_speaker_can_resolve_character_fav(monkeypatch, tmp_path: Path):
    asyncio.run(_run_speaker_character_fav_case(monkeypatch, tmp_path))


async def _run_speaker_character_fav_case(monkeypatch, tmp_path: Path):
    character_fav = DummyFav({"角色表情": "character_hash"})
    monkeypatch.setattr(
        group_chat_agent_module,
        "get_character",
        lambda _: SimpleNamespace(fav=character_fav),
    )
    monkeypatch.setattr(
        group_chat_agent_module,
        "SETTING_CFG",
        SimpleNamespace(
            LLM=SimpleNamespace(SpeakerRequestTimeout=1),
            Agent=SimpleNamespace(SpeakerInitInputMessages=50),
        ),
    )

    host = SimpleNamespace(
        data_path=tmp_path,
        message_history=DummyHistory(),
        speaker_round_limit=10,
        member_alias={},
    )
    speaker = GroupChatSpeaker(host=host, character="测试角色")
    speaker._initiated = True
    speaker.FAV = DummyFav({})
    speaker.LLM = DummyLLM(
        [
            TextPrompt(
                "assistant",
                """{
  "message_sequence": ["fav_1"],
  "message_content": {
    "fav_1": {
      "fav_title": "角色表情"
    }
  },
  "to_agent": "",
  "intention": {}
}""",
            )
        ]
    )

    result = await speaker.speak(speak_prompt="发一个角色表情")

    assert result is not None
    assert result["message_content"]["fav_1"] == {"hash_name": "character_hash"}


def test_manual_fav_title_is_trimmed(monkeypatch, tmp_path: Path):
    async def run():
        monkeypatch.setattr(
            auto_fav_module,
            "get_file_path_async",
            AsyncMock(return_value=tmp_path / "image.png"),
        )
        monkeypatch.setattr(auto_fav_module, "set_fav_title", lambda *_: None)
        _set_auto_fav_config(monkeypatch)
        fav = AutoFav(tmp_path / "fav.json")

        title = await fav.add_fav("hash_name", "  标题  ")

        assert title == "标题"
        assert "标题" in fav.fav_list
        loaded = AutoFav(tmp_path / "fav.json")
        assert loaded.fav_list["标题"] == {"hash_name"}

    asyncio.run(run())


def test_manual_fav_title_rejects_over_30_characters(monkeypatch, tmp_path: Path):
    async def run():
        monkeypatch.setattr(
            auto_fav_module,
            "get_file_path_async",
            AsyncMock(return_value=tmp_path / "image.png"),
        )
        monkeypatch.setattr(auto_fav_module, "set_fav_title", lambda *_: None)
        _set_auto_fav_config(monkeypatch)
        fav = AutoFav(tmp_path / "fav.json")

        with pytest.raises(ValueError, match="30"):
            await fav.add_fav("hash_name", "长" * 31)

    asyncio.run(run())


def test_add_group_fav_rejects_truncated_image(monkeypatch, tmp_path: Path):
    async def run():
        valid_image = tmp_path / "valid.png"
        truncated_image = tmp_path / "truncated.png"
        Image.new("RGB", (16, 16), "red").save(valid_image)
        truncated_image.write_bytes(valid_image.read_bytes()[:60])

        monkeypatch.setattr(
            group_chat_agent_module,
            "get_file_path_async",
            AsyncMock(return_value=truncated_image),
        )
        dummy_agent = SimpleNamespace(
            _host=SimpleNamespace(
                FAV=SimpleNamespace(
                    add_fav=lambda *args, **kwargs: pytest.fail(
                        "损坏图片不应进入收藏写入流程"
                    )
                )
            )
        )

        with pytest.raises(OSError):
            await GroupChatAgent.add_group_fav(dummy_agent, "hash_name", "标题")

    asyncio.run(run())
