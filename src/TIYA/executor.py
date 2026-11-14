"""
executor.py
存放全局线程池
"""

from concurrent.futures import ThreadPoolExecutor

GLOBAL_EXECUTOR = ThreadPoolExecutor()
AGENT_EXECUTOR = ThreadPoolExecutor()
FILE_CACHE_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="file-cache",
)
AUTOMAPPING_EXECUTOR = ThreadPoolExecutor(
    max_workers=32,
    thread_name_prefix="automapping",
)

# LLM 的 SQLite、token 估算及图片编解码共用此池；不占用 asyncio 默认池。
# 调整这里的 max_workers 即可限制整个进程的 LLM 同步工作并发数。
LLM_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")


def shutdown_all() -> None:
    """业务协程停止后由主线程调用，等待各池收尾，包括 LLM 账本写入。"""
    LLM_EXECUTOR.shutdown()
    GLOBAL_EXECUTOR.shutdown()
    AGENT_EXECUTOR.shutdown()
    FILE_CACHE_EXECUTOR.shutdown()
    AUTOMAPPING_EXECUTOR.shutdown(wait=False, cancel_futures=True)
