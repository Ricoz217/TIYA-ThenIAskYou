from __future__ import annotations

"""Install repository-provided resources into writable runtime storage."""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from TIYA.config import BUILTIN_RESOURCES_DIR, CHARACTER_DIR
from TIYA.file_cache import (
    FileCache,
    FileInfo,
    FileRetention,
    add_file_async,
    check_file_exists,
    get_file_metadata_async,
)
from TIYA.utils import atomic_save_json


_HASH_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class _Cache(Protocol):
    def check_file_exists(self, hash_name: str) -> bool: ...

    async def get_file_metadata_async(self, hash_name: str) -> FileInfo | None: ...

    async def add_file_async(
            self,
            file: Path,
            file_type: str = "file",
            note: str = "",
            *,
            retention: FileRetention | str = FileRetention.EXPIRING,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class BuiltinResourceResult:
    imported: int = 0
    skipped: int = 0
    created_indexes: int = 0


class _DefaultCache:
    """Adapt the module-level file-cache API to the cache protocol."""

    @staticmethod
    def check_file_exists(hash_name: str) -> bool:
        return check_file_exists(hash_name)

    @staticmethod
    async def get_file_metadata_async(hash_name: str) -> FileInfo | None:
        return await get_file_metadata_async(hash_name)

    @staticmethod
    async def add_file_async(
            file: Path,
            file_type: str = "file",
            note: str = "",
            *,
            retention: FileRetention | str = FileRetention.EXPIRING,
    ) -> str:
        return await add_file_async(
            file,
            file_type,
            note,
            retention=retention,
        )


def _load_manifest(manifest_path: Path) -> tuple[str, dict[str, list[str]]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"内置资源清单格式错误: {manifest_path}")

    character = raw.get("character")
    favorites = raw.get("favorites")
    if not isinstance(character, str) or not character.strip():
        raise ValueError(f"内置资源清单缺少人格名称: {manifest_path}")
    if not isinstance(favorites, dict):
        raise ValueError(f"内置资源清单缺少表情映射: {manifest_path}")

    normalized: dict[str, list[str]] = {}
    for title, hashes in favorites.items():
        if not isinstance(title, str) or not isinstance(hashes, list):
            raise ValueError(f"内置资源清单表情映射错误: {manifest_path}")

        normalized_hashes: list[str] = []
        for hash_name in hashes:
            if not isinstance(hash_name, str):
                raise ValueError(f"内置资源清单包含非法哈希: {manifest_path}")

            hash_name = hash_name.lower()
            if not _HASH_PATTERN.fullmatch(hash_name):
                raise ValueError(f"内置资源清单包含非法哈希: {hash_name!r}")
            normalized_hashes.append(hash_name)

        normalized[title] = normalized_hashes

    return character.strip(), normalized


def _new_favorite_index(favorites: dict[str, list[str]]) -> dict:
    now = time.time()
    return {
        "update": now,
        "description": "此文件为映射表持久化数据",
        "version": "0.1.0",
        "data": {
            title: {"update": now, "data": list(hashes)}
            for title, hashes in favorites.items()
        },
    }


def _read_referenced_hashes(index_path: Path) -> set[str]:
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), dict):
        raise ValueError(f"人格表情索引格式错误: {index_path}")

    result: set[str] = set()
    for content in raw["data"].values():
        if not isinstance(content, dict):
            continue
        hashes = content.get("data")
        if not isinstance(hashes, list):
            continue
        result.update(
            hash_name.lower()
            for hash_name in hashes
            if isinstance(hash_name, str) and _HASH_PATTERN.fullmatch(hash_name)
        )
    return result


async def ensure_builtin_resources(
        *,
        resource_root: Path = BUILTIN_RESOURCES_DIR,
        character_root: Path = CHARACTER_DIR,
        cache: _Cache | FileCache | None = None,
) -> BuiltinResourceResult:
    """Install only the built-in favorites still selected by each user index."""
    resource_root = Path(resource_root)
    character_root = Path(character_root)
    character_resources = resource_root / "character"
    if not character_resources.is_dir():
        raise FileNotFoundError(f"内置人格资源目录不存在: {character_resources}")

    active_cache: _Cache = cache if cache is not None else _DefaultCache()
    imported = 0
    skipped = 0
    created_indexes = 0
    manifests = sorted(character_resources.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"内置人格资源清单不存在: {character_resources}")

    for manifest_path in manifests:
        character, favorites = _load_manifest(manifest_path)
        if manifest_path.parent.name != character:
            raise ValueError(
                f"内置资源目录与人格名称不一致: {manifest_path.parent.name!r} != {character!r}"
            )

        default_hashes = {
            hash_name
            for hashes in favorites.values()
            for hash_name in hashes
        }
        index_path = character_root / character / "fav_index.json"
        index_exists = index_path.is_file()
        referenced = (
            _read_referenced_hashes(index_path)
            if index_exists
            else default_hashes
        )
        selected_hashes = sorted(default_hashes.intersection(referenced))
        files_dir = manifest_path.parent / "files"
        for hash_name in selected_hashes:
            metadata = None
            if active_cache.check_file_exists(hash_name):
                metadata = await active_cache.get_file_metadata_async(hash_name)
            if metadata is not None and metadata.retention is FileRetention.PERMANENT:
                skipped += 1
                continue

            source_path = files_dir / hash_name
            if not source_path.is_file():
                raise FileNotFoundError(f"内置表情文件不存在: {source_path}")

            actual_hash = await active_cache.add_file_async(
                source_path,
                file_type="image",
                note=f"内置人格[{character}]默认表情",
                retention=FileRetention.PERMANENT,
            )
            if actual_hash != hash_name:
                raise RuntimeError(
                    f"内置表情文件哈希校验失败: 期望[{hash_name}]，实际[{actual_hash}]"
                )
            imported += 1

        if not index_exists:
            atomic_save_json(_new_favorite_index(favorites), index_path)
            created_indexes += 1

    return BuiltinResourceResult(
        imported=imported,
        skipped=skipped,
        created_indexes=created_indexes,
    )


__all__ = ["BuiltinResourceResult", "ensure_builtin_resources"]
