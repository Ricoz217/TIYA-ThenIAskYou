"""ContextMemory 包入口。

这是一个面向 Python 的分层上下文记忆引擎，提供：
- 记忆写入与查询（含递归桶检索与重排）
- 桶管理（创建、切换、分桶、压缩、优化）
- 文件/目录批量导入与证据关联
- 统计、清理与存储维护能力

常用入口：
- `get_context_memory_engine(...)`：获取全局单例引擎
- `ContextMemoryConfig`：构建配置对象
- `ContextMemoryEngineV3` / `BucketHandle`：直接调用引擎与桶级接口
"""

__version__ = "0.3.3"
__data_version__ = 4

from .engine import (
    BucketHandle,
    ContextMemoryConfig,
    ContextMemoryEngineV3,
    ContextMemorySystem,
    __data_version__ as engine_data_version,
    get_context_memory_engine,
)
from .llm_pipeline import LLMPresetConfigError
from .models import BucketContextUsage, ListMemoriesResult, MemoryIndexItem


if int(engine_data_version) != int(__data_version__):
    raise RuntimeError(
        f"memory __data_version__ mismatch: package={__data_version__}, engine={engine_data_version}"
    )


__all__ = [
    "__data_version__",
    "ContextMemoryEngineV3",
    "ContextMemorySystem",
    "ContextMemoryConfig",
    "BucketHandle",
    "get_context_memory_engine",
    "LLMPresetConfigError",
    "MemoryIndexItem",
    "ListMemoriesResult",
    "BucketContextUsage",
]
