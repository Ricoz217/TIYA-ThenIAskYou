from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from TIYA.file_cache import FileCache, FileRetention


def _hash(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _write_bundle(root: Path, favorites: dict[str, list[bytes]]) -> dict[str, list[str]]:
    files_dir = root / "character" / "抹布" / "files"
    files_dir.mkdir(parents=True)
    manifest_favorites: dict[str, list[str]] = {}
    for title, payloads in favorites.items():
        hashes = []
        for payload in payloads:
            hash_name = _hash(payload)
            (files_dir / hash_name).write_bytes(payload)
            hashes.append(hash_name)
        manifest_favorites[title] = hashes

    manifest = {
        "version": 1,
        "character": "抹布",
        "favorites": manifest_favorites,
    }
    (files_dir.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_favorites


def _write_runtime_index(path: Path, favorites: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "update": 100.0,
            "description": "此文件为映射表持久化数据",
            "version": "0.1.0",
            "data": {
                title: {"update": 100.0, "data": hashes}
                for title, hashes in favorites.items()
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )


def test_new_character_index_is_created_and_default_files_become_permanent(
        tmp_path: Path,
) -> None:
    from TIYA.builtin_resources import ensure_builtin_resources

    async def run_test() -> None:
        resource_root = tmp_path / "builtin_resources"
        character_root = tmp_path / "character"
        expected = _write_bundle(resource_root, {
            "开心": [b"happy"],
            "生气": [b"angry", b"very-angry"],
        })
        cache = FileCache(tmp_path / "cache", expire_days=30)

        result = await ensure_builtin_resources(
            resource_root=resource_root,
            character_root=character_root,
            cache=cache,
        )

        assert result.imported == 3
        assert result.skipped == 0
        index = json.loads(
            (character_root / "抹布" / "fav_index.json").read_text(encoding="utf-8")
        )
        assert {
            title: content["data"]
            for title, content in index["data"].items()
        } == expected
        assert all(content["update"] > 100 for content in index["data"].values())
        for hash_name in {item for values in expected.values() for item in values}:
            metadata = await cache.get_file_metadata_async(hash_name)
            assert metadata is not None
            assert metadata.retention is FileRetention.PERMANENT

        await cache.close()

    asyncio.run(run_test())


def test_existing_user_index_is_not_overwritten_and_removed_defaults_stay_removed(
        tmp_path: Path,
) -> None:
    from TIYA.builtin_resources import ensure_builtin_resources

    async def run_test() -> None:
        resource_root = tmp_path / "builtin_resources"
        character_root = tmp_path / "character"
        defaults = _write_bundle(resource_root, {
            "保留默认": [b"keep"],
            "用户已删除": [b"removed"],
        })
        custom_hash = "f" * 32
        index_path = character_root / "抹布" / "fav_index.json"
        _write_runtime_index(index_path, {
            "保留默认": defaults["保留默认"],
            "用户自己的": [custom_hash],
        })
        original_index = index_path.read_bytes()
        cache = FileCache(tmp_path / "cache", expire_days=30)

        result = await ensure_builtin_resources(
            resource_root=resource_root,
            character_root=character_root,
            cache=cache,
        )

        assert result.imported == 1
        assert index_path.read_bytes() == original_index
        assert cache.check_file_exists(defaults["保留默认"][0])
        assert not cache.check_file_exists(defaults["用户已删除"][0])
        assert not cache.check_file_exists(custom_hash)
        await cache.close()

    asyncio.run(run_test())


def test_second_start_does_not_rewrite_already_permanent_files(tmp_path: Path) -> None:
    from TIYA.builtin_resources import ensure_builtin_resources

    async def run_test() -> None:
        resource_root = tmp_path / "builtin_resources"
        character_root = tmp_path / "character"
        _write_bundle(resource_root, {"默认": [b"same-file"]})
        cache = FileCache(tmp_path / "cache", expire_days=30)

        first = await ensure_builtin_resources(
            resource_root=resource_root,
            character_root=character_root,
            cache=cache,
        )
        second = await ensure_builtin_resources(
            resource_root=resource_root,
            character_root=character_root,
            cache=cache,
        )

        assert first.imported == 1
        assert second.imported == 0
        assert second.skipped == 1
        await cache.close()

    asyncio.run(run_test())


def test_corrupted_builtin_file_is_rejected(tmp_path: Path) -> None:
    from TIYA.builtin_resources import ensure_builtin_resources

    async def run_test() -> None:
        resource_root = tmp_path / "builtin_resources"
        character_root = tmp_path / "character"
        defaults = _write_bundle(resource_root, {"默认": [b"expected"]})
        hash_name = defaults["默认"][0]
        source = resource_root / "character" / "抹布" / "files" / hash_name
        source.write_bytes(b"corrupted")
        cache = FileCache(tmp_path / "cache", expire_days=30)

        with pytest.raises(RuntimeError, match="哈希"):
            await ensure_builtin_resources(
                resource_root=resource_root,
                character_root=character_root,
                cache=cache,
            )

        assert not (character_root / "抹布" / "fav_index.json").exists()
        await cache.close()

    asyncio.run(run_test())


def test_repository_builtin_pack_installs_into_isolated_cache(tmp_path: Path) -> None:
    from TIYA.builtin_resources import ensure_builtin_resources

    async def run_test() -> None:
        resource_root = Path(__file__).resolve().parents[1] / "data" / "builtin_resources"
        cache = FileCache(tmp_path / "cache", expire_days=30)

        result = await ensure_builtin_resources(
            resource_root=resource_root,
            character_root=tmp_path / "character",
            cache=cache,
        )

        assert result.imported == 15
        assert result.created_indexes == 1
        assert sum(
            1
            for path in (tmp_path / "cache").rglob("*")
            if path.is_file() and len(path.name) == 32
        ) == 15
        await cache.close()

    asyncio.run(run_test())
