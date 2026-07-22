from __future__ import annotations

import asyncio

from TIYA.executor import FILE_CACHE_EXECUTOR
from TIYA.file_cache import FileCache


def test_file_cache_executor_has_startup_headroom() -> None:
    assert FILE_CACHE_EXECUTOR._max_workers >= 8


def test_concurrent_same_hash_operations_do_not_deadlock(tmp_path) -> None:
    async def run_test() -> None:
        cache = FileCache(tmp_path)
        payload = b"same-content" * 1024

        hashes = await asyncio.wait_for(
            asyncio.gather(*(cache.add_file_async(payload) for _ in range(24))),
            timeout=10,
        )

        assert len(set(hashes)) == 1
        hash_name = hashes[0]
        paths = await asyncio.wait_for(
            asyncio.gather(*(cache.get_file_path_async(hash_name) for _ in range(24))),
            timeout=10,
        )
        assert all(path.read_bytes() == payload for path in paths)

    asyncio.run(run_test())


def test_cleanup_and_access_race_always_completes(tmp_path) -> None:
    async def run_test() -> None:
        cache = FileCache(tmp_path, expire_days=0)
        hash_name = await cache.add_file_async(b"cleanup-race")
        loop = asyncio.get_running_loop()

        cleanup = loop.run_in_executor(FILE_CACHE_EXECUTOR, cache.cleanup_once)
        access = cache.get_file_path_async(hash_name)
        results = await asyncio.wait_for(
            asyncio.gather(cleanup, access, return_exceptions=True),
            timeout=10,
        )

        assert len(results) == 2
        assert not any(isinstance(result, TimeoutError) for result in results)

    asyncio.run(run_test())
