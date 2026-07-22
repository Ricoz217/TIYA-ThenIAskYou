from __future__ import annotations

import time
from pathlib import Path

from TIYA.logger import Logger, LoggerConfig


def _make_logger(tmp_path: Path) -> Logger:
    return Logger(
        LoggerConfig(
            logs_dir=tmp_path,
            use_console=False,
            typing_pause_enabled=False,
            worker_activity_window_sec=0.1,
            maintenance_interval_sec=60.0,
        )
    )


def _expire_block(logger: Logger, block_id: str) -> None:
    with logger._state_lock:
        block = logger._state.blocks[block_id]
        block.activity.clear()
        block.created_at = time.monotonic() - 1.0
        logger._maybe_reorder_and_close_workers(time.monotonic())
        assert block_id not in logger._state.blocks


def test_worker_title_survives_info_wakeup_after_timeout(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.create_block("worker-1", "Custom worker title")
    logger.flush()

    _expire_block(logger, handle.block_id)
    handle.info("awake")
    logger.flush()

    with logger._state_lock:
        assert logger._state.blocks[handle.block_id].title == "Custom worker title"
    logger.shutdown()


def test_worker_title_survives_status_wakeup_after_timeout(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.create_block("worker-2", "Status worker title")
    logger.flush()

    _expire_block(logger, handle.block_id)
    handle.status("awake")
    logger.flush()

    with logger._state_lock:
        assert logger._state.blocks[handle.block_id].title == "Status worker title"
    logger.shutdown()


def test_explicit_remove_forgets_worker_title(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.create_block("worker-3", "Disposable title")
    logger.flush()

    handle.remove()
    logger.flush()
    handle.info("recreated after explicit removal")
    logger.flush()

    with logger._state_lock:
        assert logger._state.blocks[handle.block_id].title == handle.block_id
    logger.shutdown()


def test_recreating_worker_updates_remembered_title(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.create_block("worker-4", "Old title")
    logger.flush()
    logger.create_block(handle.block_id, "New title")
    logger.flush()

    _expire_block(logger, handle.block_id)
    handle.info("awake")
    logger.flush()

    with logger._state_lock:
        assert logger._state.blocks[handle.block_id].title == "New title"
    logger.shutdown()
