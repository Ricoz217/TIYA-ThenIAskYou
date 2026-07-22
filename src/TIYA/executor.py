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

def shutdown_all():
    GLOBAL_EXECUTOR.shutdown()
    AGENT_EXECUTOR.shutdown()
    FILE_CACHE_EXECUTOR.shutdown()
