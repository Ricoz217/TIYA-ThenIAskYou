from __future__ import annotations
"""
initiate_instance.py
用于初始化一些全局单例，并进行配置
"""
__version__ = "0.1.0"
__all__ = []

import os

from TIYA.memory import get_context_memory_engine, ContextMemoryConfig
from TIYA.logger import get_logger, LoggerConfig
from TIYA.config import DATA_DIR, LOGS_DIR

os.environ.setdefault("LOG_FILE_PATH", str(LOGS_DIR))

__memory_config = ContextMemoryConfig(
    base_dir=DATA_DIR / "memory",
    max_bucket_depth=6
)
__memory = get_context_memory_engine(config=__memory_config)
__log_config = LoggerConfig(
    logs_dir=LOGS_DIR,
    status_idle_text="等待 Token 用量数据...",
    worker_empty_text="暂无新消息",
    worker_area_empty_text="暂无活跃群聊"
)
__log = get_logger(config=__log_config)

