from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from TIYA.logger import Logger, LoggerConfig


def _make_logger(tmp_path: Path, **overrides) -> Logger:
    config = LoggerConfig(
        logs_dir=tmp_path,
        use_console=False,
        typing_pause_enabled=False,
        **overrides,
    )
    return Logger(config)


def _log_text(root: Path, name: str = "app.log") -> str:
    path = root / datetime.now().strftime("%Y-%m-%d") / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_write_is_buffered_until_close_and_persisted_once(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.info("STREAM_START:", stream=True)
    assert handle is not None

    try:
        handle.write("part-one")
        handle.write("|part-two")
        logger.flush()
        assert "STREAM_START:" not in _log_text(tmp_path)
        with logger._state_lock:
            visible = logger._state.message_stream_visible[handle._stream_id]
            assert visible.text == "STREAM_START:part-one|part-two"

        handle.close()
        logger.flush()
        text = _log_text(tmp_path)
        assert text.count("STREAM_START:part-one|part-two") == 1
        assert text.count("[INFO]") == 1
    finally:
        logger.shutdown()


def test_context_manager_exception_still_persists_partial_stream(
    tmp_path: Path,
) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.info("CONTEXT_STREAM:", stream=True)
    assert handle is not None

    try:
        try:
            with handle:
                handle.write("partial-content")
                raise RuntimeError("stop stream")
        except RuntimeError:
            pass

        logger.flush()
        assert "CONTEXT_STREAM:partial-content" in _log_text(tmp_path)
    finally:
        logger.shutdown()


def test_write_flush_creates_checkpoint_without_close_duplicate(
    tmp_path: Path,
) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.info("CHECKPOINT:", stream=True)
    assert handle is not None

    try:
        handle.write("one", flush=True)
        assert _log_text(tmp_path).count("CHECKPOINT:one") == 1

        handle.close()
        logger.flush()
        assert _log_text(tmp_path).count("CHECKPOINT:one") == 1
    finally:
        logger.shutdown()


def test_write_after_checkpoint_persists_new_final_snapshot(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.info("CHECKPOINT_DIRTY:", stream=True)
    assert handle is not None

    try:
        handle.write("one", flush=True)
        handle.write("|two")
        handle.close()
        logger.flush()

        text = _log_text(tmp_path)
        assert text.count("CHECKPOINT_DIRTY:one") == 2
        assert text.count("CHECKPOINT_DIRTY:one|two") == 1
    finally:
        logger.shutdown()


def test_update_persists_each_snapshot_without_close_duplicate(
    tmp_path: Path,
) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.info("INITIAL_SHOULD_NOT_PERSIST", stream=True)
    assert handle is not None

    try:
        handle.update("UPDATE_SNAPSHOT_ONE")
        handle.update("UPDATE_SNAPSHOT_TWO")
        handle.close()
        logger.flush()

        text = _log_text(tmp_path)
        assert "INITIAL_SHOULD_NOT_PERSIST" not in text
        assert text.count("UPDATE_SNAPSHOT_ONE") == 1
        assert text.count("UPDATE_SNAPSHOT_TWO") == 1
    finally:
        logger.shutdown()


def test_mixed_update_and_write_persists_only_complete_states(
    tmp_path: Path,
) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.info("MIXED_INITIAL", stream=True)
    assert handle is not None

    try:
        handle.write("|discarded-by-update")
        handle.update("MIXED_REPLACEMENT")
        handle.write("|tail")
        handle.close()
        logger.flush()

        text = _log_text(tmp_path)
        assert "MIXED_INITIAL" not in text
        assert "discarded-by-update" not in text
        assert text.count("MIXED_REPLACEMENT") == 2
        assert text.count("MIXED_REPLACEMENT|tail") == 1
    finally:
        logger.shutdown()


def test_file_buffer_is_not_limited_by_ui_stream_tail(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path, stream_tail_chars=32)
    long_text = "LONG_STREAM_START:" + ("x" * 256) + ":LONG_STREAM_END"
    handle = logger.info("", stream=True)
    assert handle is not None

    try:
        handle.write(long_text)
        handle.close()
        logger.flush()
        assert long_text in _log_text(tmp_path)
    finally:
        logger.shutdown()


def test_block_expiry_does_not_discard_pending_file_stream(
    tmp_path: Path,
) -> None:
    logger = _make_logger(
        tmp_path,
        worker_activity_window_sec=0.1,
        maintenance_interval_sec=60.0,
    )
    block = logger.create_block("stream-worker", "Stream Worker")
    handle = block.info("BLOCK_STREAM:", stream=True)
    assert handle is not None

    try:
        handle.write("before-expiry")
        logger.flush()
        with logger._state_lock:
            state = logger._state.blocks[block.block_id]
            state.activity.clear()
            state.created_at = time.monotonic() - 1.0
            logger._maybe_reorder_and_close_workers(time.monotonic())
            assert block.block_id not in logger._state.blocks

        handle.write("|after-expiry")
        handle.close()
        logger.flush()
        assert "BLOCK_STREAM:before-expiry|after-expiry" in _log_text(tmp_path)
    finally:
        logger.shutdown()


def test_shutdown_finalizes_unclosed_stream(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    handle = logger.debug("SHUTDOWN_STREAM:", stream=True)
    assert handle is not None
    handle.write("final-content")

    logger.shutdown()

    debug_text = _log_text(tmp_path, "debug.log")
    assert debug_text.count("SHUTDOWN_STREAM:final-content") == 1


def test_stream_levels_keep_existing_file_routing(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    debug_handle = logger.debug("DEBUG_STREAM:", stream=True)
    error_handle = logger.error("ERROR_STREAM:", stream=True)
    assert debug_handle is not None
    assert error_handle is not None

    try:
        debug_handle.write("debug-content")
        error_handle.write("error-content")
        debug_handle.close()
        error_handle.close()
        logger.flush()

        app_text = _log_text(tmp_path)
        debug_text = _log_text(tmp_path, "debug.log")
        error_text = _log_text(tmp_path, "error.log")
        assert "DEBUG_STREAM:debug-content" not in app_text
        assert "DEBUG_STREAM:debug-content" in debug_text
        assert "ERROR_STREAM:error-content" in app_text
        assert "ERROR_STREAM:error-content" in error_text
    finally:
        logger.shutdown()


def test_status_streams_do_not_persist(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    block = logger.create_block("status-worker", "Status Worker")
    global_status = logger.status("GLOBAL_STATUS_INITIAL")
    worker_status = block.status("WORKER_STATUS_INITIAL")

    try:
        global_status.update("GLOBAL_STATUS_UPDATE")
        worker_status.update("WORKER_STATUS_UPDATE")
        global_status.close()
        worker_status.close()
        logger.flush()

        app_text = _log_text(tmp_path)
        assert "GLOBAL_STATUS_INITIAL" not in app_text
        assert "GLOBAL_STATUS_UPDATE" not in app_text
        assert "WORKER_STATUS_INITIAL" not in app_text
        assert "WORKER_STATUS_UPDATE" not in app_text
    finally:
        logger.shutdown()
