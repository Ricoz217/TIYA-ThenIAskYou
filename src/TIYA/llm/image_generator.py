from __future__ import annotations

import json
import asyncio

from typing import TYPE_CHECKING
from pathlib import Path

from .model import (
    TextPrompt,
    ImagePrompt,
    Prompts
)
from .config_model import Params, ChatConfig
from .usage_compat import LLMUsage
from .settings import parse_llm_setting
from TIYA.logger import get_logger
from TIYA.config import debug_mode  # 演示用，实际不需要

_log = get_logger()
if TYPE_CHECKING:
    from TIYA.logger import BlockHandle, Logger


class ImageGen:
    def __init__(
        self,
        *,
        endpoint: str = "",
        token: str = "",
        model: str = "",
        client_params: dict[str, object] | None = None,
        model_params: Params | None = None,
        keep_alive: bool = False,
        logger: Logger | BlockHandle | None = None,
        log_off: bool = False,
        usage_record: list[LLMUsage] | None = None,
        config: ChatConfig | None = None,
    ):
        ...

    async def generate(
        self,
        prompt: str | TextPrompt,
        *args: bytes | Path | ImagePrompt,
        background: str = "auto",
        complexity: str = None,
        frame_count: int = None,
        frame_duration_ms: int = None,
        moderation: str = "low",
        n: int = 1,
        output_compression: int = None,
        output_format: str = "png",
        partial_images: int = None,
        quality: str = "auto",
        size: str = None,
        stream: bool = False,
        **kwargs
    ) -> Prompts:
        """
        生图接口，若显式传入参数则优先使用，否则使用对象属性中的参数
        :param prompt: 文字 prompt
        :param args: 图片，支持多张或者不传。严格按传入顺序发给模型。自动文件名，如果输入是字节流则自行分配一个，哈希或者递增数字都行
        :param background:
        :param complexity:
        :param frame_count:
        :param frame_duration_ms:
        :param moderation:
        :param n:
        :param output_compression:
        :param output_format:
        :param partial_images:
        :param quality:
        :param size:
        :param stream:
        :param kwargs:
        :return:
        """
        ...

    async def edit(
        self,
        prompt: str | TextPrompt,
        base_image: bytes | Path | ImagePrompt,
        *args: bytes | Path | ImagePrompt,
        mask: bytes | Path | ImagePrompt = None,
        markup: bytes | Path | ImagePrompt = None,
        background: str = "auto",
        complexity: str = None,
        frame_count: int = None,
        frame_duration_ms: int = None,
        input_fidelity: str = None,
        moderation: str = "low",
        n: int = 1,
        output_compression: int = None,
        output_format: str = "png",
        partial_images: int = None,
        quality: str = "auto",
        size: str = None,
        stream: bool = False,
        **kwargs
    ) -> Prompts:
        """
        编辑接口，规则同上
        :param prompt: 文字 prompt
        :param base_image: 必须单独指定原图(要修改的图)
        :param args: 额外图片素材，严格按顺序发送
        :param mask:
        :param markup:
        :param background:
        :param complexity:
        :param frame_count:
        :param frame_duration_ms:
        :param input_fidelity:
        :param moderation:
        :param n:
        :param output_compression:
        :param output_format:
        :param partial_images:
        :param quality:
        :param size:
        :param stream:
        :param kwargs:
        :return:
        """
        ...

# 用法示例
async def generate_example():
    preset = "OPENAI_IMAGE_2.5"  # 这个预设我填好了
    config = parse_llm_setting(preset)
    client = ImageGen(config=config)

    # 生图
    kokona_root = Path(r"D:\Python\TIYA_ThenIAskYou_2026\outputs\websearch\kokona_reference_pack")
    profile_file = kokona_root / "profile.json"
    profile = json.loads(profile_file.read_text(encoding="utf-8"))
    character = ImagePrompt(
        role="user",
        image=kokona_root / "images" / "kokona-default.png",
        name="character"
    )
    front = ImagePrompt(
        role="user",
        image=kokona_root / "images" / "kokona-front.png",
        name="front"
    )
    back = ImagePrompt(
        role="user",
        image=kokona_root / "images" / "kokona-back.png",
        name="back"
    )
    detail = ImagePrompt(
        role="user",
        image=kokona_root / "images" / "kokona-character-sheet.jpg",
        name="detail"
    )

    prompt = f"""\
以下是卡拉彼丘的 `心夏` 的角色资料以及参考图。你需要根据这些心夏，生成几个心夏的日常图片，题材不限姿势不限构图不限  

# 角色资料
```text
{json.dumps(profile, ensure_ascii=False, indent=2)}
```


# 参考图

- `character`: 角色立绘
- `front`: 正视图
- `back`: 背视图
- `detail`: 一些设计细节

"""

    response = await client.generate(
        prompt,
        character,
        front,
        back,
        detail
    )

if __name__ == '__main__':
    debug_mode()
    asyncio.run(generate_example())