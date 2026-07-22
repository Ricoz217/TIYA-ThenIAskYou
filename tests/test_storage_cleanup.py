from __future__ import annotations

import os
import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from TIYA.storage_cleanup import (
    CleanupPolicy,
    CleanupBreakdown,
    CleanupResult,
    StorageCleanupService,
    get_storage_cleanup_service,
    run_auto_storage_cleanup,
)


NOW = datetime(2026, 7, 18, 12, 0, 0)


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    groups_root = data_root / "groups_data"
    private_root = data_root / "private_chats_data"
    logs_root = tmp_path / "logs"
    groups_root.mkdir(parents=True)
    private_root.mkdir(parents=True)
    logs_root.mkdir()
    return groups_root, private_root, logs_root


def _checkpoint(
        root: Path,
        user_id: str,
        created_at: datetime,
        *,
        size: int = 10,
) -> Path:
    date_dir = root / user_id / "checkpoint" / created_at.strftime("%y_%m_%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / f"checkpoint_{created_at:%y%m%d_%H%M%S}.json"
    path.write_bytes(b"x" * size)
    return path


def _log(path: Path, modified_at: datetime, *, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    timestamp = modified_at.timestamp()
    os.utime(path, (timestamp, timestamp))
    return path


def _service(tmp_path: Path) -> tuple[StorageCleanupService, tuple[Path, Path, Path]]:
    roots = _roots(tmp_path)
    service = StorageCleanupService(
        groups_root=roots[0],
        private_root=roots[1],
        logs_root=roots[2],
        policy=CleanupPolicy(
            checkpoint_retention_days=3,
            log_retention_days=7,
            keep_latest_checkpoints=5,
        ),
        clock=lambda: NOW,
    )
    return service, roots


def _populate(service_roots: tuple[Path, Path, Path]) -> dict[str, list[Path] | Path]:
    groups_root, private_root, logs_root = service_roots

    old_group = [
        _checkpoint(groups_root, "100", datetime(2026, 7, 10, 1, 0, second), size=20)
        for second in range(4)
    ]
    recent_group = [
        _checkpoint(groups_root, "100", datetime(2026, 7, 17, 1, 0, second), size=30)
        for second in range(5)
    ]
    old_private = [
        _checkpoint(private_root, "200", datetime(2026, 7, 9, 1, 0, 0), size=40)
    ]
    recent_private = [
        _checkpoint(private_root, "200", datetime(2026, 7, 18, 1, 0, second), size=50)
        for second in range(5)
    ]

    root_level = groups_root / "100" / "checkpoint" / "checkpoint_260701_000000.json"
    root_level.write_bytes(b"must stay")
    wrong_user = _checkpoint(groups_root, "not-a-user", datetime(2026, 7, 1), size=60)
    wrong_name = groups_root / "100" / "checkpoint" / "26_07_01" / "notes.json"
    wrong_name.parent.mkdir(parents=True, exist_ok=True)
    wrong_name.write_bytes(b"must stay")

    old_log_dir = logs_root / "2026-07-01"
    old_logs = [
        _log(old_log_dir / name, datetime(2026, 7, 1, 23, 0), size=70)
        for name in ("app.log", "debug.log", "error.log")
    ]
    old_logs.append(
        _log(
            logs_root / "bot_20260701.log.2026-07-01",
            datetime(2026, 7, 1, 23, 0),
            size=80,
        )
    )
    old_logs.append(_log(logs_root / "app.log", datetime(2026, 7, 1), size=90))
    recent_log = _log(
        logs_root / "2026-07-18" / "app.log",
        datetime(2026, 7, 18, 1, 0),
        size=100,
    )
    arbitrary_log = _log(logs_root / "user-backup.zip", datetime(2026, 6, 1), size=110)

    return {
        "old_group": old_group,
        "recent_group": recent_group,
        "old_private": old_private,
        "recent_private": recent_private,
        "root_level": root_level,
        "wrong_user": wrong_user,
        "wrong_name": wrong_name,
        "old_logs": old_logs,
        "recent_log": recent_log,
        "arbitrary_log": arbitrary_log,
        "old_log_dir": old_log_dir,
    }


def test_scan_is_non_destructive_and_only_counts_strict_matches(tmp_path: Path) -> None:
    async def run_test() -> None:
        service, roots = _service(tmp_path)
        paths = _populate(roots)

        plan = await service.scan()

        assert plan.summary.group_checkpoints.files == 4
        assert plan.summary.group_checkpoints.bytes == 80
        assert plan.summary.private_checkpoints.files == 1
        assert plan.summary.private_checkpoints.bytes == 40
        assert plan.summary.logs.files == 5
        assert plan.summary.logs.bytes == 380
        assert plan.summary.total.files == 10
        assert plan.summary.total.bytes == 500
        assert all(path.exists() for path in paths["old_group"])
        assert all(path.exists() for path in paths["old_logs"])

    import asyncio

    asyncio.run(run_test())


def test_execute_deletes_plan_and_preserves_structure_and_latest_five(
        tmp_path: Path,
) -> None:
    async def run_test() -> None:
        service, roots = _service(tmp_path)
        paths = _populate(roots)
        plan = await service.scan()

        result = await service.execute(plan)

        assert result.deleted == plan.summary
        assert result.skipped_files == 0
        assert result.failed_files == 0
        assert not any(path.exists() for path in paths["old_group"])
        assert all(path.exists() for path in paths["recent_group"])
        assert not any(path.exists() for path in paths["old_private"])
        assert all(path.exists() for path in paths["recent_private"])
        assert not any(path.exists() for path in paths["old_logs"])
        assert paths["recent_log"].exists()
        assert paths["root_level"].exists()
        assert paths["wrong_user"].exists()
        assert paths["wrong_name"].exists()
        assert paths["arbitrary_log"].exists()
        assert not paths["old_log_dir"].exists()
        assert not (roots[0] / "100" / "checkpoint" / "26_07_10").exists()
        assert roots[0].is_dir()
        assert roots[1].is_dir()
        assert roots[2].is_dir()

    import asyncio

    asyncio.run(run_test())


def test_execute_skips_file_changed_after_scan(tmp_path: Path) -> None:
    async def run_test() -> None:
        service, roots = _service(tmp_path)
        paths = _populate(roots)
        plan = await service.scan()
        changed = paths["old_group"][0]
        changed.write_bytes(b"changed after scan")

        result = await service.execute(plan)

        assert changed.exists()
        assert result.skipped_files == 1
        assert result.failed_files == 0
        assert result.deleted.files == plan.summary.total.files - 1
        assert result.deleted.bytes == plan.summary.total.bytes - 20

    import asyncio

    asyncio.run(run_test())


def test_execute_reports_one_failure_and_continues(
        tmp_path: Path,
        monkeypatch,
) -> None:
    async def run_test() -> None:
        roots = _roots(tmp_path)
        errors: list[tuple[Path, BaseException]] = []
        service = StorageCleanupService(
            groups_root=roots[0],
            private_root=roots[1],
            logs_root=roots[2],
            clock=lambda: NOW,
            error_handler=lambda path, exc: errors.append((path, exc)),
        )
        paths = _populate(roots)
        plan = await service.scan()
        blocked = paths["old_group"][0]
        original_unlink = Path.unlink

        def guarded_unlink(path: Path, *args, **kwargs):
            if path == blocked:
                raise PermissionError("file is in use")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", guarded_unlink)

        result = await service.execute(plan)

        assert blocked.exists()
        assert result.deleted.files == 9
        assert result.failed_files == 1
        assert result.skipped_files == 0
        assert len(errors) == 1
        assert errors[0][0] == blocked

    import asyncio

    asyncio.run(run_test())


def test_plan_cannot_execute_against_different_roots(tmp_path: Path) -> None:
    async def run_test() -> None:
        first, first_roots = _service(tmp_path / "first")
        second, _ = _service(tmp_path / "second")
        paths = _populate(first_roots)
        plan = await first.scan()

        with pytest.raises(ValueError, match="不属于"):
            await second.execute(plan)

        assert all(path.exists() for path in paths["old_group"])

    import asyncio

    asyncio.run(run_test())


def test_cleanup_convenience_method_scans_and_executes(tmp_path: Path) -> None:
    async def run_test() -> None:
        service, roots = _service(tmp_path)
        paths = _populate(roots)

        result = await service.cleanup()

        assert result.deleted.files == 10
        assert not any(path.exists() for path in paths["old_group"])

    import asyncio

    asyncio.run(run_test())


def test_service_rejects_misconfigured_or_overlapping_roots(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    groups_root = data_root / "groups_data"
    private_root = data_root / "private_chats_data"
    logs_root = tmp_path / "logs"

    with pytest.raises(ValueError, match="目录结构"):
        StorageCleanupService(
            groups_root=tmp_path / "someone" / "groups_data",
            private_root=private_root,
            logs_root=logs_root,
        )

    with pytest.raises(ValueError, match="目录结构"):
        StorageCleanupService(
            groups_root=groups_root,
            private_root=private_root,
            logs_root=data_root / "logs",
        )


def test_scan_ignores_symlink_even_when_name_matches(
        tmp_path: Path,
        monkeypatch,
) -> None:
    async def run_test() -> None:
        service, roots = _service(tmp_path)
        link = (
            roots[0]
            / "100"
            / "checkpoint"
            / "26_07_01"
            / "checkpoint_260701_000000.json"
        )
        link.parent.mkdir(parents=True)
        link.write_bytes(b"represented symlink")
        original_is_symlink = Path.is_symlink

        def fake_is_symlink(path: Path) -> bool:
            return path == link or original_is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

        plan = await service.scan()
        result = await service.execute(plan)

        assert plan.summary.total.files == 0
        assert result.deleted.files == 0
        assert link.exists()

    import asyncio

    asyncio.run(run_test())


def test_policy_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError):
        CleanupPolicy(checkpoint_retention_days=0)

    with pytest.raises(ValueError):
        CleanupPolicy(log_retention_days=-1)

    with pytest.raises(ValueError):
        CleanupPolicy(keep_latest_checkpoints=-1)


def test_default_service_is_lazy_and_process_shared(tmp_path: Path, monkeypatch) -> None:
    import TIYA.config as config
    import TIYA.storage_cleanup as storage_cleanup

    groups_root, private_root, logs_root = _roots(tmp_path)
    monkeypatch.setattr(config, "GROUPS_DIR", groups_root)
    monkeypatch.setattr(config, "PRIVATE_CHATS_DIR", private_root)
    monkeypatch.setattr(config, "LOGS_DIR", logs_root)
    monkeypatch.setattr(storage_cleanup, "_DEFAULT_SERVICE", None)

    first = get_storage_cleanup_service()
    second = get_storage_cleanup_service()

    assert first is second
    assert first.groups_root == groups_root
    assert first.private_root == private_root
    assert first.logs_root == logs_root


def test_auto_cleanup_waits_ten_minutes_then_runs_every_three_days() -> None:
    async def run_test() -> None:
        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)
            if len(delays) == 2:
                raise asyncio.CancelledError

        deleted = CleanupBreakdown()
        cleaner = SimpleNamespace(
            cleanup=AsyncMock(return_value=CleanupResult(deleted=deleted)),
        )
        logger = Mock()

        with pytest.raises(asyncio.CancelledError):
            await run_auto_storage_cleanup(
                logger=logger,
                cleaner=cleaner,
                sleep=fake_sleep,
            )

        assert delays == [600.0, 3 * 24 * 60 * 60]
        cleaner.cleanup.assert_awaited_once_with()
        logger.debug.assert_called_once_with(
            "自动存储清理完成：没有可清理的过期数据"
        )
        logger.info.assert_not_called()

    asyncio.run(run_test())


def test_auto_cleanup_logs_summary_without_exposing_paths() -> None:
    async def run_test() -> None:
        sleep_count = 0

        async def fake_sleep(_delay: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count == 2:
                raise asyncio.CancelledError

        deleted = CleanupBreakdown(
            logs=SimpleNamespace(files=2, bytes=2 * 1024 ** 3),
        )
        cleaner = SimpleNamespace(
            cleanup=AsyncMock(return_value=CleanupResult(
                deleted=deleted,
                skipped_files=1,
                failed_files=0,
            )),
        )
        logger = Mock()

        with pytest.raises(asyncio.CancelledError):
            await run_auto_storage_cleanup(
                logger=logger,
                cleaner=cleaner,
                sleep=fake_sleep,
            )

        message = logger.info.call_args.args[0]
        assert message == (
            "自动存储清理完成：删除 2 个文件，释放 2.00 GB，"
            "跳过 1 个，失败 0 个"
        )
        assert "checkpoint_" not in message
        assert "\\" not in message

    asyncio.run(run_test())


def test_auto_cleanup_logs_failure_and_keeps_schedule_alive() -> None:
    async def run_test() -> None:
        sleep_count = 0

        async def fake_sleep(_delay: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count == 2:
                raise asyncio.CancelledError

        cleaner = SimpleNamespace(
            cleanup=AsyncMock(side_effect=RuntimeError("scan failed")),
        )
        logger = Mock()

        with pytest.raises(asyncio.CancelledError):
            await run_auto_storage_cleanup(
                logger=logger,
                cleaner=cleaner,
                sleep=fake_sleep,
            )

        cleaner.cleanup.assert_awaited_once_with()
        logger.error.assert_called_once_with("自动存储清理失败: scan failed")
        logger.debug.assert_called_once()

    asyncio.run(run_test())
