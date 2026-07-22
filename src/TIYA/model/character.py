from __future__ import annotations
"""Load and manage BOT characters for group and private chat scenes."""
__version__ = "0.3.0"

import shutil
import frontmatter
from threading import Lock
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from TIYA.auto_fav import get_autofav, AutoFav
from TIYA.config import GROUPS_CFG, CHARACTER_DIR
from TIYA.logger import get_logger


_log = get_logger()
_CREATE_LOCK = Lock()


@dataclass(slots=True)
class GroupCharacterManager:
    group_id: str
    use_global: bool = True
    character_now: str = "抹布"
    use_character: set[str] = field(default_factory=set)
    ban_character: set[str] = field(default_factory=set)


@dataclass(slots=True)
class Character:
    group_metadata: dict[str, Any]
    private_metadata: dict[str, Any]
    group_personality: str
    private_personality: str
    fav: AutoFav

    @property
    def metadata(self) -> dict[str, Any]:
        """Backward-compatible metadata view for group-chat callers."""
        return self.group_metadata

    @property
    def personality(self) -> str:
        """Backward-compatible group-chat personality text."""
        return self.group_personality

    def to_dict(self):
        return {
            "group_metadata": self.group_metadata.copy(),
            "private_metadata": self.private_metadata.copy(),
            "group_personality": self.group_personality,
            "private_personality": self.private_personality,
        }


_GLOBAL_CHARACTER: dict[str, Character] = {}
GROUP_CHARACTER_MANAGER: dict[str, GroupCharacterManager] = {}

def initiate_groups_character_manager():
    """全局初始化"""
    groups: dict = GROUPS_CFG.Groups.copy()
    groups.pop("114514", None)
    groups.pop("Group_Default_Setting", None)

    GROUP_CHARACTER_MANAGER.clear()
    for k, v in groups.items():
        GROUP_CHARACTER_MANAGER[k] = GroupCharacterManager(
            group_id=k,
            use_global=v.get("use_global_characters", True),
            character_now=v.get("character", v.get("default_character", "抹布")),
            use_character=v.get("use_character", set()),
            ban_character=v.get("ban_character", set())
        )

    _GLOBAL_CHARACTER.clear()
    if not CHARACTER_DIR.is_dir():
        return

    for character_dir in CHARACTER_DIR.iterdir():
        if (
            character_dir.is_dir()
            and not character_dir.name.startswith(".")
            and character_dir.name != "fav"
        ):
            try:
                add_character(character_dir.name)
            except (FileNotFoundError, ValueError) as exc:
                _log.error(f"人格[{character_dir.name}]加载失败: {exc}")

def _load_single_group(group_id: str):
    if group_id not in GROUPS_CFG.Groups:
        _log.error(f"群[{group_id}] 配置暂未生成")
        raise KeyError(f"群[{group_id}] 配置暂未生成")

    data = GROUPS_CFG.Groups[group_id]
    GROUP_CHARACTER_MANAGER[group_id] = GroupCharacterManager(
        group_id=group_id,
        use_global=data.get("use_global_characters", True),
        character_now=data.get("character", data.get("default_character", "抹布")),
        use_character=data.get("use_character", set()),
        ban_character=data.get("ban_character", set())
    )

def get_character_list(group_id: str) -> list[str]:
    """获取人格列表"""
    if group_id not in GROUP_CHARACTER_MANAGER:
        try:
            _load_single_group(group_id)

        except KeyError:
            return []

    GCM = GROUP_CHARACTER_MANAGER[group_id]
    if GCM.use_global:
        return [c for c in _GLOBAL_CHARACTER if c not in GCM.ban_character]

    else:
        return [c for c in GCM.use_character if c in _GLOBAL_CHARACTER]

def get_group_character_manager(group_id: str) -> GroupCharacterManager | None:
    if group_id not in GROUP_CHARACTER_MANAGER:
        try:
            _load_single_group(group_id)

        except KeyError:
            return None

    return GROUP_CHARACTER_MANAGER[group_id]

def get_character(cha_name: str) -> Character:
    """获取人格内容"""
    if cha_name not in _GLOBAL_CHARACTER:
        _log.error(f"人格[{cha_name}] 不存在")
        raise KeyError(f"人格[{cha_name}] 不存在")

    return _GLOBAL_CHARACTER[cha_name]

def get_group_character_now(group_id: str) -> Character:
    GCM = get_group_character_manager(group_id)
    if GCM is None:
        raise KeyError(f"群[{group_id}] 尚未写入配置")

    character_now = GCM.character_now
    return get_character(character_now)

def add_character(
        cha_name: str,
        *,
        cha_file: Path = None,
        update: bool = False,
) -> Character:
    """新增人格"""
    with _CREATE_LOCK:
        if cha_name in _GLOBAL_CHARACTER and not update:
            raise KeyError(f"人格 [{cha_name}] 已存在")

        target_dir = CHARACTER_DIR / cha_name
        if isinstance(cha_file, Path):
            if not cha_file.is_dir():
                raise ValueError("人格路径必须是包含 group.md/private.md 的目录")

            if cha_file.name != cha_name:
                raise ValueError("人格名字必须与目录名相同")

            if cha_file.resolve() != target_dir.resolve():
                shutil.copytree(cha_file, target_dir, dirs_exist_ok=update)

        group_file = target_dir / "group.md"
        private_file = target_dir / "private.md"
        missing = [path.name for path in (group_file, private_file) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"人格[{cha_name}] 缺少文件: {', '.join(missing)}"
            )

        fav_index_file = target_dir / "fav_index.json"
        group_character = frontmatter.load(group_file)
        private_character = frontmatter.load(private_file)
        old_cha = _GLOBAL_CHARACTER.get(cha_name, None)
        if old_cha is None:
            new_cha = Character(
                group_metadata=dict(group_character.metadata),
                private_metadata=dict(private_character.metadata),
                group_personality=group_character.content,
                private_personality=private_character.content,
                fav=get_autofav(fav_index_file)
            )
            _GLOBAL_CHARACTER[cha_name] = new_cha
            return new_cha

        else:
            old_cha.group_metadata = dict(group_character.metadata)
            old_cha.private_metadata = dict(private_character.metadata)
            old_cha.group_personality = group_character.content
            old_cha.private_personality = private_character.content
            old_cha.fav = get_autofav(fav_index_file)
            return old_cha
