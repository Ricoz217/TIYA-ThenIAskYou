from __future__ import annotations

"""Strict, reusable cleanup planning and execution for TIYA runtime data."""

import asyncio
import re
import stat
import threading
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path


_USER_ID_PATTERN = re.compile(r"^\d+$")
_CHECKPOINT_DATE_PATTERN = re.compile(r"^\d{2}_\d{2}_\d{2}$")
_CHECKPOINT_FILE_PATTERN = re.compile(
    r"^checkpoint_(\d{6}_\d{6})\.json$"
)
_LOG_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BOT_LOG_PATTERN = re.compile(
    r"^bot_\d{8}\.log(?:\.\d{4}-\d{2}-\d{2})?$"
)
_NAMED_LOG_FILES = frozenset({"app.log", "debug.log", "error.log"})
_DEFAULT_SERVICE: StorageCleanupService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()
_AUTO_CLEANUP_INITIAL_DELAY = 10 * 60.0
_AUTO_CLEANUP_INTERVAL = 3 * 24 * 60 * 60.0


class CleanupCategory(str, Enum):
    GROUP_CHECKPOINT = "group_checkpoint"
    PRIVATE_CHECKPOINT = "private_checkpoint"
    LOG = "log"


@dataclass(frozen=True, slots=True)
class CleanupPolicy:
    checkpoint_retention_days: int = 3
    log_retention_days: int = 7
    keep_latest_checkpoints: int = 5

    def __post_init__(self) -> None:
        if self.checkpoint_retention_days <= 0:
            raise ValueError("checkpoint_retention_days 必须大于 0")
        if self.log_retention_days <= 0:
            raise ValueError("log_retention_days 必须大于 0")
        if self.keep_latest_checkpoints < 0:
            raise ValueError("keep_latest_checkpoints 不能小于 0")


@dataclass(frozen=True, slots=True)
class CleanupStats:
    files: int = 0
    bytes: int = 0


@dataclass(frozen=True, slots=True)
class CleanupBreakdown:
    group_checkpoints: CleanupStats = field(default_factory=CleanupStats)
    private_checkpoints: CleanupStats = field(default_factory=CleanupStats)
    logs: CleanupStats = field(default_factory=CleanupStats)

    @property
    def total(self) -> CleanupStats:
        return CleanupStats(
            files=(
                self.group_checkpoints.files
                + self.private_checkpoints.files
                + self.logs.files
            ),
            bytes=(
                self.group_checkpoints.bytes
                + self.private_checkpoints.bytes
                + self.logs.bytes
            ),
        )

    @property
    def files(self) -> int:
        return self.total.files

    @property
    def bytes(self) -> int:
        return self.total.bytes


@dataclass(frozen=True, slots=True)
class _CleanupCandidate:
    path: Path
    category: CleanupCategory
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    created_at: datetime
    checkpoint_cutoff: datetime
    log_cutoff: datetime
    summary: CleanupBreakdown
    _root_fingerprint: tuple[Path, Path, Path] = field(repr=False)
    _candidates: tuple[_CleanupCandidate, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class CleanupResult:
    deleted: CleanupBreakdown
    skipped_files: int = 0
    failed_files: int = 0


class StorageCleanupService:
    """Build and execute conservative cleanup plans under canonical roots."""

    def __init__(
            self,
            *,
            groups_root: Path,
            private_root: Path,
            logs_root: Path,
            policy: CleanupPolicy | None = None,
            clock: Callable[[], datetime] = datetime.now,
            error_handler: Callable[[Path, BaseException], None] | None = None,
    ) -> None:
        raw_roots = (
            Path(groups_root),
            Path(private_root),
            Path(logs_root),
        )
        if any(root.is_symlink() for root in raw_roots if root.exists()):
            raise ValueError("清理根目录不能是符号链接")

        self.groups_root = raw_roots[0].resolve(strict=False)
        self.private_root = raw_roots[1].resolve(strict=False)
        self.logs_root = raw_roots[2].resolve(strict=False)
        self.policy = policy or CleanupPolicy()
        self._clock = clock
        self._error_handler = error_handler
        self._lock = asyncio.Lock()
        self._validate_roots()

    async def scan(self) -> CleanupPlan:
        """Scan eligible files without mutating the filesystem."""
        async with self._lock:
            return await asyncio.to_thread(self._scan_sync)

    async def execute(self, plan: CleanupPlan) -> CleanupResult:
        """Execute an earlier plan after revalidating every candidate."""
        async with self._lock:
            return await asyncio.to_thread(self._execute_sync, plan)

    async def cleanup(self) -> CleanupResult:
        """Scan and execute under one lock for unattended cleanup."""
        async with self._lock:
            plan = await asyncio.to_thread(self._scan_sync)
            return await asyncio.to_thread(self._execute_sync, plan)

    def _validate_roots(self) -> None:
        roots = (self.groups_root, self.private_root, self.logs_root)
        data_root = self.groups_root.parent
        valid_layout = (
            self.groups_root.name == "groups_data"
            and self.private_root.name == "private_chats_data"
            and self.logs_root.name == "logs"
            and data_root.name == "data"
            and self.private_root.parent == data_root
            and self.logs_root.parent == data_root.parent
            and len(set(roots)) == 3
        )
        if not valid_layout:
            raise ValueError("清理根目录不符合 TIYA 目录结构")

    def _scan_sync(self) -> CleanupPlan:
        now = self._clock()
        checkpoint_cutoff = now - timedelta(
            days=self.policy.checkpoint_retention_days
        )
        log_cutoff = now - timedelta(days=self.policy.log_retention_days)

        candidates = []
        candidates.extend(self._scan_checkpoint_root(
            self.groups_root,
            CleanupCategory.GROUP_CHECKPOINT,
            checkpoint_cutoff,
        ))
        candidates.extend(self._scan_checkpoint_root(
            self.private_root,
            CleanupCategory.PRIVATE_CHECKPOINT,
            checkpoint_cutoff,
        ))
        candidates.extend(self._scan_logs(log_cutoff))
        packed = tuple(candidates)
        return CleanupPlan(
            created_at=now,
            checkpoint_cutoff=checkpoint_cutoff,
            log_cutoff=log_cutoff,
            summary=self._breakdown(packed),
            _root_fingerprint=self._root_fingerprint,
            _candidates=packed,
        )

    def _scan_checkpoint_root(
            self,
            root: Path,
            category: CleanupCategory,
            cutoff: datetime,
    ) -> list[_CleanupCandidate]:
        if not root.is_dir() or root.is_symlink():
            return []

        candidates: list[_CleanupCandidate] = []
        for user_dir in self._safe_children(root):
            if (
                not _USER_ID_PATTERN.fullmatch(user_dir.name)
                or not user_dir.is_dir()
                or user_dir.is_symlink()
            ):
                continue

            checkpoint_root = user_dir / "checkpoint"
            if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
                continue

            checkpoints: list[tuple[datetime, _CleanupCandidate]] = []
            for date_dir in self._safe_children(checkpoint_root):
                if (
                    not _CHECKPOINT_DATE_PATTERN.fullmatch(date_dir.name)
                    or not date_dir.is_dir()
                    or date_dir.is_symlink()
                ):
                    continue

                try:
                    directory_date = datetime.strptime(
                        date_dir.name,
                        "%y_%m_%d",
                    ).date()
                except ValueError:
                    continue

                for path in self._safe_children(date_dir):
                    match = _CHECKPOINT_FILE_PATTERN.fullmatch(path.name)
                    if match is None:
                        continue

                    try:
                        created_at = datetime.strptime(
                            match.group(1),
                            "%y%m%d_%H%M%S",
                        )
                    except ValueError:
                        continue

                    if created_at.date() != directory_date:
                        continue

                    candidate = self._candidate(path, root, category)
                    if candidate is not None:
                        checkpoints.append((created_at, candidate))

            checkpoints.sort(key=lambda item: item[0], reverse=True)
            keep = self.policy.keep_latest_checkpoints
            for created_at, candidate in checkpoints[keep:]:
                if created_at < cutoff:
                    candidates.append(candidate)

        return candidates

    def _scan_logs(self, cutoff: datetime) -> list[_CleanupCandidate]:
        root = self.logs_root
        if not root.is_dir() or root.is_symlink():
            return []

        candidates: list[_CleanupCandidate] = []
        cutoff_timestamp = cutoff.timestamp()
        for path in self._safe_children(root):
            if path.is_symlink():
                continue

            if path.is_file() and self._valid_root_log_name(path.name):
                candidate = self._candidate(path, root, CleanupCategory.LOG)
                if candidate is not None and candidate.mtime_ns < cutoff_timestamp * 1e9:
                    candidates.append(candidate)
                continue

            if (
                not path.is_dir()
                or not _LOG_DATE_PATTERN.fullmatch(path.name)
            ):
                continue

            try:
                datetime.strptime(path.name, "%Y-%m-%d")
            except ValueError:
                continue

            for log_file in self._safe_children(path):
                if log_file.name not in _NAMED_LOG_FILES:
                    continue

                candidate = self._candidate(
                    log_file,
                    root,
                    CleanupCategory.LOG,
                )
                if candidate is not None and candidate.mtime_ns < cutoff_timestamp * 1e9:
                    candidates.append(candidate)

        return candidates

    def _execute_sync(self, plan: CleanupPlan) -> CleanupResult:
        if plan._root_fingerprint != self._root_fingerprint:
            raise ValueError("清理计划不属于当前清理服务")

        deleted: list[_CleanupCandidate] = []
        deleted_parents: set[Path] = set()
        skipped = 0
        failed = 0
        for candidate in plan._candidates:
            root = self._root_for(candidate.category)
            if not self._valid_candidate_structure(candidate.path, candidate.category):
                skipped += 1
                continue

            try:
                current = self._candidate(candidate.path, root, candidate.category)
            except OSError as exc:
                failed += 1
                self._report_error(candidate.path, exc)
                continue

            if current is None:
                skipped += 1
                continue
            if (
                current.size != candidate.size
                or current.mtime_ns != candidate.mtime_ns
            ):
                skipped += 1
                continue

            try:
                candidate.path.unlink()
            except FileNotFoundError:
                skipped += 1
            except OSError as exc:
                failed += 1
                self._report_error(candidate.path, exc)
            else:
                deleted.append(candidate)
                deleted_parents.add(candidate.path.parent)

        self._remove_empty_date_dirs(deleted_parents)
        return CleanupResult(
            deleted=self._breakdown(tuple(deleted)),
            skipped_files=skipped,
            failed_files=failed,
        )

    def _candidate(
            self,
            path: Path,
            root: Path,
            category: CleanupCategory,
    ) -> _CleanupCandidate | None:
        if path.is_symlink():
            return None

        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                return None

            file_stat = resolved.stat(follow_symlinks=False)
        except (FileNotFoundError, OSError):
            return None

        if not stat.S_ISREG(file_stat.st_mode):
            return None

        return _CleanupCandidate(
            path=resolved,
            category=category,
            size=file_stat.st_size,
            mtime_ns=file_stat.st_mtime_ns,
        )

    def _valid_candidate_structure(
            self,
            path: Path,
            category: CleanupCategory,
    ) -> bool:
        root = self._root_for(category)
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError):
            return False

        if category is CleanupCategory.LOG:
            parts = relative.parts
            if len(parts) == 1:
                return self._valid_root_log_name(parts[0])
            if len(parts) == 2:
                return (
                    _LOG_DATE_PATTERN.fullmatch(parts[0]) is not None
                    and parts[1] in _NAMED_LOG_FILES
                )
            return False

        parts = relative.parts
        if len(parts) != 4:
            return False
        user_id, checkpoint, date_name, filename = parts
        match = _CHECKPOINT_FILE_PATTERN.fullmatch(filename)
        if (
            not _USER_ID_PATTERN.fullmatch(user_id)
            or checkpoint != "checkpoint"
            or not _CHECKPOINT_DATE_PATTERN.fullmatch(date_name)
            or match is None
        ):
            return False

        try:
            return (
                datetime.strptime(match.group(1), "%y%m%d_%H%M%S").date()
                == datetime.strptime(date_name, "%y_%m_%d").date()
            )
        except ValueError:
            return False

    def _remove_empty_date_dirs(self, parents: set[Path]) -> None:
        for parent in parents:
            if parent.is_symlink():
                continue

            valid_checkpoint_dir = (
                _CHECKPOINT_DATE_PATTERN.fullmatch(parent.name) is not None
                and parent.parent.name == "checkpoint"
                and (
                    parent.is_relative_to(self.groups_root)
                    or parent.is_relative_to(self.private_root)
                )
            )
            valid_log_dir = (
                _LOG_DATE_PATTERN.fullmatch(parent.name) is not None
                and parent.parent == self.logs_root
            )
            if not (valid_checkpoint_dir or valid_log_dir):
                continue

            try:
                parent.rmdir()
            except (FileNotFoundError, OSError):
                continue

    @staticmethod
    def _safe_children(path: Path) -> tuple[Path, ...]:
        try:
            return tuple(path.iterdir())
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return ()

    @staticmethod
    def _valid_root_log_name(name: str) -> bool:
        return name in _NAMED_LOG_FILES or _BOT_LOG_PATTERN.fullmatch(name) is not None

    @property
    def _root_fingerprint(self) -> tuple[Path, Path, Path]:
        return self.groups_root, self.private_root, self.logs_root

    def _root_for(self, category: CleanupCategory) -> Path:
        if category is CleanupCategory.GROUP_CHECKPOINT:
            return self.groups_root
        if category is CleanupCategory.PRIVATE_CHECKPOINT:
            return self.private_root
        return self.logs_root

    @staticmethod
    def _breakdown(candidates: tuple[_CleanupCandidate, ...]) -> CleanupBreakdown:
        counters = {
            category: [0, 0]
            for category in CleanupCategory
        }
        for candidate in candidates:
            counter = counters[candidate.category]
            counter[0] += 1
            counter[1] += candidate.size

        return CleanupBreakdown(
            group_checkpoints=CleanupStats(
                files=counters[CleanupCategory.GROUP_CHECKPOINT][0],
                bytes=counters[CleanupCategory.GROUP_CHECKPOINT][1],
            ),
            private_checkpoints=CleanupStats(
                files=counters[CleanupCategory.PRIVATE_CHECKPOINT][0],
                bytes=counters[CleanupCategory.PRIVATE_CHECKPOINT][1],
            ),
            logs=CleanupStats(
                files=counters[CleanupCategory.LOG][0],
                bytes=counters[CleanupCategory.LOG][1],
            ),
        )

    def _report_error(self, path: Path, exc: BaseException) -> None:
        if self._error_handler is not None:
            self._error_handler(path, exc)


def get_storage_cleanup_service() -> StorageCleanupService:
    """Return the lazy process-wide cleaner used by manual and scheduled jobs."""
    global _DEFAULT_SERVICE

    service = _DEFAULT_SERVICE
    if service is not None:
        return service

    with _DEFAULT_SERVICE_LOCK:
        service = _DEFAULT_SERVICE
        if service is not None:
            return service

        from TIYA.config import GROUPS_DIR, LOGS_DIR, PRIVATE_CHATS_DIR

        def report_error(path: Path, exc: BaseException) -> None:
            from TIYA.logger import get_logger

            get_logger().error(f"清理文件失败: path=[{path}], error=[{exc}]")

        service = StorageCleanupService(
            groups_root=GROUPS_DIR,
            private_root=PRIVATE_CHATS_DIR,
            logs_root=LOGS_DIR,
            error_handler=report_error,
        )
        _DEFAULT_SERVICE = service
        return service


def format_storage_size(size: int) -> str:
    """Format byte counts for compact cleanup summaries."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


async def run_auto_storage_cleanup(
        *,
        logger,
        cleaner=None,
        initial_delay: float = _AUTO_CLEANUP_INITIAL_DELAY,
        interval: float = _AUTO_CLEANUP_INTERVAL,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Run the standard cleanup policy after startup and every three days."""
    if initial_delay < 0:
        raise ValueError("initial_delay 不能小于 0")
    if interval <= 0:
        raise ValueError("interval 必须大于 0")

    await sleep(initial_delay)
    service = cleaner or get_storage_cleanup_service()

    while True:
        try:
            result = await service.cleanup()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"自动存储清理失败: {exc}")
            logger.debug(traceback.format_exc())
        else:
            deleted = result.deleted
            if deleted.files or result.skipped_files or result.failed_files:
                logger.info(
                    f"自动存储清理完成：删除 {deleted.files} 个文件，"
                    f"释放 {format_storage_size(deleted.bytes)}，"
                    f"跳过 {result.skipped_files} 个，"
                    f"失败 {result.failed_files} 个"
                )
            else:
                logger.debug("自动存储清理完成：没有可清理的过期数据")

        await sleep(interval)
