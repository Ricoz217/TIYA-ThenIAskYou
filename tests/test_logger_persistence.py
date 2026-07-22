from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import TIYA.logger as logger_module
from TIYA.logger import Logger, LoggerConfig, _Event


def _dated_app_log(root: Path, day: str) -> Path:
    return root / day / "app.log"


class _FrozenDateTime:
    current = datetime(2026, 6, 9, 8, 0, 0)

    @classmethod
    def now(cls):
        return cls.current


def test_full_event_queue_accounts_for_discarded_item() -> None:
    logger = object.__new__(Logger)
    logger._events = queue.Queue(maxsize=1)

    logger._events.put_nowait(_Event(kind="old", payload={}))
    logger._enqueue(_Event(kind="new", payload={}))

    assert logger._events.qsize() == 1
    assert logger._events.unfinished_tasks == 1


def test_full_file_queue_accounts_for_discarded_item() -> None:
    logger = object.__new__(Logger)
    logger._file_events = queue.Queue(maxsize=1)

    logger._file_events.put_nowait(("app", "old"))
    logger._file_enqueue("app", "new")

    assert logger._file_events.qsize() == 1
    assert logger._file_events.unfinished_tasks == 1


def test_shutdown_waits_for_records_created_by_pending_events(
    tmp_path: Path,
) -> None:
    logger = Logger(
        LoggerConfig(
            logs_dir=tmp_path,
            use_console=False,
            typing_pause_enabled=False,
        )
    )
    logger._state_lock.acquire()
    logger.info("queued-before-shutdown")
    shutdown_thread = threading.Thread(
        target=logger.shutdown,
        kwargs={"timeout": 1.0},
        daemon=True,
    )

    try:
        shutdown_thread.start()
        time.sleep(0.2)
    finally:
        logger._state_lock.release()

    shutdown_thread.join(timeout=2.0)
    shutdown_completed = not shutdown_thread.is_alive()
    app_log = _dated_app_log(tmp_path, datetime.now().strftime("%Y-%m-%d"))
    record_persisted = (
        app_log.exists()
        and "queued-before-shutdown" in app_log.read_text(encoding="utf-8")
    )
    if not shutdown_completed:
        while True:
            try:
                logger._file_events.get_nowait()
            except queue.Empty:
                break
            else:
                logger._file_events.task_done()
        shutdown_thread.join(timeout=1.0)

    assert shutdown_completed
    assert record_persisted


def test_writer_continues_after_one_record_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logger = object.__new__(Logger)
    logger.config = LoggerConfig(logs_dir=tmp_path)
    logger._file_events = queue.Queue()
    logger._shutdown = threading.Event()
    logger._writer_shutdown = threading.Event()
    original_open = Path.open
    failed_once = False
    opened_app_streams = []

    def fail_first_file_set_open(path: Path, *args, **kwargs):
        nonlocal failed_once
        if path.name == "debug.log" and not failed_once:
            failed_once = True
            raise OSError("simulated log write failure")
        stream = original_open(path, *args, **kwargs)
        if path.name == "app.log":
            opened_app_streams.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", fail_first_file_set_open)

    logger._file_events.put(("app", "will-fail"))
    logger._file_events.put(("app", "will-persist"))
    logger._shutdown.set()
    logger._writer_shutdown.set()
    writer_thread = threading.Thread(target=logger._writer_loop, daemon=True)
    writer_thread.start()
    writer_thread.join(timeout=1.0)

    assert failed_once
    assert all(stream.closed for stream in opened_app_streams)
    assert not writer_thread.is_alive()
    assert logger._file_events.unfinished_tasks == 0
    app_log = _dated_app_log(tmp_path, datetime.now().strftime("%Y-%m-%d"))
    assert "will-persist" in app_log.read_text(encoding="utf-8")


def test_writer_rolls_files_by_current_date(tmp_path: Path, monkeypatch) -> None:
    _FrozenDateTime.current = datetime(2026, 6, 9, 8, 0, 0)
    monkeypatch.setattr(logger_module, "datetime", _FrozenDateTime)
    logger = Logger(
        LoggerConfig(
            logs_dir=tmp_path,
            use_console=False,
            typing_pause_enabled=False,
        )
    )

    logger.info("first-day")
    logger.flush()
    _FrozenDateTime.current = datetime(2026, 6, 10, 0, 0, 1)
    logger.info("second-day")
    logger.flush()
    logger.shutdown()

    first_log = _dated_app_log(tmp_path, "2026-06-09")
    second_log = _dated_app_log(tmp_path, "2026-06-10")
    assert "first-day" in first_log.read_text(encoding="utf-8")
    assert "second-day" not in first_log.read_text(encoding="utf-8")
    assert "second-day" in second_log.read_text(encoding="utf-8")
    assert not (tmp_path / "app.log").exists()
