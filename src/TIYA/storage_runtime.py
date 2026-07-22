from __future__ import annotations

"""Lifecycle orchestration for file storage and bundled resources."""

import traceback
from typing import Any

from TIYA.builtin_resources import ensure_builtin_resources
from TIYA.file_cache import close_file_cache, start_file_cache


async def start_storage_runtime(logger: Any) -> None:
    """Start file storage, then install optional repository resources."""
    await start_file_cache()
    try:
        result = await ensure_builtin_resources()

    except Exception as error:
        logger.error(f"内置资源初始化失败，将继续启动 BOT: {error}")
        logger.debug(traceback.format_exc())
        return

    logger.info(
        "内置资源初始化完成: "
        f"导入[{result.imported}]，跳过[{result.skipped}]，"
        f"新建索引[{result.created_indexes}]"
    )


async def close_storage_runtime() -> None:
    """Close file storage after every runtime consumer has stopped."""
    await close_file_cache()


__all__ = ["close_storage_runtime", "start_storage_runtime"]
