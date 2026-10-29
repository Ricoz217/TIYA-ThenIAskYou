from __future__ import annotations
import asyncio
import base64
from pathlib import Path
from TIYA.auto_mapping import AutoMapping
from TIYA.config import DATA_DIR
from TIYA.logger import get_logger
from TIYA.file_cache import add_file_async, get_file_path_async
from TIYA.utils import resize_image_to_base64
from TIYA.executor import GLOBAL_EXECUTOR

_log = get_logger()
_MAPPING_FILE = DATA_DIR / "llm_connect" / "image_name_mapping.json"
_IMAGE_NAME_MAPPING: AutoMapping[list[str]] = AutoMapping(_MAPPING_FILE, expire_day=14)


async def _load_image_hashes(hash_list: list[str], cache: dict[str, str]) -> bool:
    """将哈希名图片转换成可传输的 base64，存入缓存，反回一个布尔值，代表是否能发送"""
    loaded_images: dict[str, str] = {}
    for hash_name in hash_list:
        if hash_name in cache:
            continue

        try:
            file_path = await get_file_path_async(hash_name)
            with file_path.open("rb") as file:
                loaded_images[hash_name] = base64.b64encode(file.read()).decode()

        except (FileNotFoundError, OSError):
            return False

    cache.update(loaded_images)
    return True


async def handle_unload_image(image: str | Path | bytes, cache: dict) -> list[str]:
    """从未加载的图片输入，转换成可传输的 base64 列表并存入缓存，然后返回处理后的单个文件的 hash_name 列表"""
    if isinstance(image, str):
        image = base64.b64decode(image)

    # 压缩图片
    loop = asyncio.get_running_loop()
    try:
        compressed_image = await loop.run_in_executor(
            GLOBAL_EXECUTOR, resize_image_to_base64, image
        )

    except Exception as E:
        _log.error(E)
        return []

    else:
        tasks: list[tuple[asyncio.Task[str], str]] = []
        hash_name_list = []
        for b64_str in compressed_image:
            tasks.append(
                (asyncio.create_task(add_file_async(b64_str, "IMAGE")), b64_str)
            )

        await asyncio.gather(*[task[0] for task in tasks], return_exceptions=True)
        for task_union in tasks:
            task = task_union[0]
            if not task.done() or task.cancelled() or task.exception() is not None:
                continue

            hash_name = task.result()
            cache[hash_name] = task_union[1]
            hash_name_list.append(hash_name)

        return hash_name_list
