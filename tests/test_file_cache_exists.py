import asyncio

import pytest

from TIYA.file_cache import FileCache


def test_check_file_exists_only_checks_physical_file(tmp_path) -> None:
    cache = FileCache(tmp_path)
    hash_name = "a" * 32
    target = tmp_path / hash_name[:2] / hash_name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test")

    assert cache.check_file_exists(hash_name)
    assert not cache.check_file_exists("b" * 32)
    assert not cache.check_file_exists("not-a-hash")
    assert not cache.database_path.exists()


def test_async_interfaces_preserve_results_and_exceptions(tmp_path) -> None:
    async def run_test() -> None:
        cache = FileCache(tmp_path)
        hash_name = await cache.add_file_async(b"test", note="async")

        path = await cache.get_file_path_async(hash_name)
        metadata = await cache.get_file_metadata_async(hash_name)

        assert path.read_bytes() == b"test"
        assert metadata is not None
        assert metadata.note == "async"

        with pytest.raises(FileNotFoundError):
            await cache.get_file_path_async("b" * 32)

        with pytest.raises(ValueError):
            await cache.get_file_path_async("not-a-hash")

    asyncio.run(run_test())
