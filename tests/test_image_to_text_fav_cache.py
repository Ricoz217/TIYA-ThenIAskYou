from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import TIYA.auto_fav as auto_fav_module
import TIYA.image_to_text as image_to_text_module
import TIYA.model.message as message_module
from TIYA.auto_fav import AutoFav
from TIYA.model.message import ImgMsg


class MemoryCache(dict):
    def touch(self, key: str):
        return None

    def remove(self, key: str):
        self.pop(key, None)


def _set_auto_fav_config(monkeypatch) -> None:
    monkeypatch.setattr(
        auto_fav_module,
        "SETTING_CFG",
        SimpleNamespace(Groups=SimpleNamespace(AutoFavExpire=0)),
    )


class DummyPrompt:
    def __init__(self, *args, **kwargs):
        pass

    async def get_prompt(self, *args, **kwargs):
        return "prompt"


class DummyContext:
    def append(self, prompt):
        return None


class DummyTextPrompt:
    def __init__(self, role: str, text: str):
        self.text = text


class DummyImagePrompt:
    def __init__(self, *args, **kwargs):
        pass


class DummyPrompts:
    def __init__(self, *prompts):
        self.prompts = list(prompts)


class DummyChat:
    def __init__(self, *args, **kwargs):
        pass

    def setting(self, setting):
        return None

    def replace_context(self, context):
        return None

    async def ask(self, prompt, timeout: float):
        return DummyPrompts(
            DummyTextPrompt("assistant", '{"title": "傲娇推眼镜"}')
        )

    async def close(self):
        return None


def test_normal_lookup_can_prefer_existing_fav_title(monkeypatch):
    async def run():
        cache = MemoryCache({
            "hash_name": {
                "fav_title": "傲娇推眼镜",
                "description": "一段很长的完整图片描述",
                "query": {},
                "time": time.time(),
            }
        })
        monkeypatch.setattr(image_to_text_module, "_IMAGE_CACHE", cache)
        monkeypatch.setattr(
            image_to_text_module,
            "SETTING_CFG",
            SimpleNamespace(ImageRecognize=SimpleNamespace(CacheExpire=30)),
        )

        result = await image_to_text_module.image2text(
            "hash_name",
            mode="NORMAL",
            prefer_fav=True,
        )

        assert result == "傲娇推眼镜"

    asyncio.run(run())


def test_fav_mode_parses_llm_json_before_caching(monkeypatch, tmp_path: Path):
    async def run():
        cache = MemoryCache()
        monkeypatch.setattr(image_to_text_module, "_IMAGE_CACHE", cache)
        monkeypatch.setattr(
            image_to_text_module,
            "get_file_path_async",
            AsyncMock(return_value=tmp_path / "image.png"),
        )
        monkeypatch.setattr(image_to_text_module, "AgentPrompt", DummyPrompt)
        monkeypatch.setattr(image_to_text_module, "Context", DummyContext)
        monkeypatch.setattr(image_to_text_module, "SystemPrompt", DummyPrompt)
        monkeypatch.setattr(image_to_text_module, "TextPrompt", DummyTextPrompt)
        monkeypatch.setattr(image_to_text_module, "ImagePrompt", DummyImagePrompt)
        monkeypatch.setattr(image_to_text_module, "Prompts", DummyPrompts)
        monkeypatch.setattr(image_to_text_module, "Chat", DummyChat)
        monkeypatch.setattr(image_to_text_module, "parse_llm_setting", lambda _: object())
        monkeypatch.setattr(
            image_to_text_module,
            "BASE_CFG",
            SimpleNamespace(Agents=SimpleNamespace(ImageModel="test")),
        )
        monkeypatch.setattr(
            image_to_text_module,
            "SETTING_CFG",
            SimpleNamespace(
                ImageRecognize=SimpleNamespace(VLMRetryLimit=1, VLMTimeout=1),
            ),
        )

        result = await image_to_text_module.image2text("hash_name", mode="FAV")

        assert result == "傲娇推眼镜"
        assert cache["hash_name"]["fav_title"] == "傲娇推眼镜"

    asyncio.run(run())


def test_fav_title_cache_only_accepts_parsed_short_title(monkeypatch):
    cache = MemoryCache()
    monkeypatch.setattr(image_to_text_module, "_IMAGE_CACHE", cache)

    image_to_text_module.set_fav_title("hash_name", '{"title": "傲娇推眼镜"}')

    assert image_to_text_module.is_fav("hash_name")
    assert cache["hash_name"]["fav_title"] == "傲娇推眼镜"


def test_img_message_reads_fav_without_creating_it(monkeypatch):
    async def run():
        calls = []

        async def fake_image2text(hash_name: str, **kwargs):
            calls.append((hash_name, kwargs))
            return "普通完整描述"

        monkeypatch.setattr(message_module, "image2text", fake_image2text)
        monkeypatch.setattr(message_module, "is_fav", lambda _: False)
        monkeypatch.setattr(
            message_module,
            "SETTING_CFG",
            SimpleNamespace(Common=SimpleNamespace(FileCacheExpire=30)),
        )

        image = ImgMsg(name="image.jpg", hash_name="hash_name")
        await image.process()

        assert image.type == "IMAGE"
        assert image.description == "普通完整描述"
        assert calls == [
            (
                "hash_name",
                {
                    "image_name": "image.jpg",
                    "mode": "NORMAL",
                    "prefer_fav": True,
                },
            )
        ]

    asyncio.run(run())


def test_auto_fav_registers_manual_title_in_shared_cache(monkeypatch, tmp_path: Path):
    async def run():
        registered = []
        monkeypatch.setattr(
            auto_fav_module,
            "get_file_path_async",
            AsyncMock(return_value=tmp_path / "image.png"),
        )
        monkeypatch.setattr(
            auto_fav_module,
            "set_fav_title",
            lambda hash_name, title: registered.append((hash_name, title)),
        )
        _set_auto_fav_config(monkeypatch)

        fav = AutoFav(tmp_path / "fav.json")
        await fav.add_fav("hash_name", "  傲娇推眼镜  ")

        assert registered == [("hash_name", "傲娇推眼镜")]

    asyncio.run(run())


def test_auto_fav_rehydrates_persisted_titles_into_shared_cache(monkeypatch, tmp_path: Path):
    async def create_index():
        monkeypatch.setattr(
            auto_fav_module,
            "get_file_path_async",
            AsyncMock(return_value=tmp_path / "image.png"),
        )
        monkeypatch.setattr(auto_fav_module, "set_fav_title", lambda *_: None)
        _set_auto_fav_config(monkeypatch)
        fav = AutoFav(tmp_path / "fav.json")
        await fav.add_fav("hash_name", "傲娇推眼镜")

    asyncio.run(create_index())

    registered = []
    monkeypatch.setattr(
        auto_fav_module,
        "set_fav_title",
        lambda hash_name, title: registered.append((hash_name, title)),
    )

    AutoFav(tmp_path / "fav.json")

    assert registered == [("hash_name", "傲娇推眼镜")]


def test_auto_generated_fav_title_is_already_parsed(monkeypatch, tmp_path: Path):
    async def run():
        monkeypatch.setattr(
            auto_fav_module,
            "get_file_path_async",
            AsyncMock(return_value=tmp_path / "image.png"),
        )
        monkeypatch.setattr(auto_fav_module, "set_fav_title", lambda *_: None)
        monkeypatch.setattr(
            auto_fav_module,
            "image2text",
            lambda *args, **kwargs: asyncio.sleep(0, result="傲娇推眼镜"),
        )
        _set_auto_fav_config(monkeypatch)

        fav = AutoFav(tmp_path / "fav.json")
        title = await fav.add_fav("hash_name")

        assert title == "傲娇推眼镜"
        assert fav.fav_list["傲娇推眼镜"] == {"hash_name"}

    asyncio.run(run())
