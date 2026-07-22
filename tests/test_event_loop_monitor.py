from __future__ import annotations

import asyncio
import time

from TIYA.event_loop_monitor import EventLoopLagMonitor


class RecordingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)


def test_monitor_does_not_report_when_event_loop_is_responsive() -> None:
    async def run_test() -> None:
        logger = RecordingLogger()
        monitor = EventLoopLagMonitor(
            logger=logger,
            threshold=0.05,
            probe_interval=0.005,
        )

        await monitor.start()
        await asyncio.sleep(0.08)
        await monitor.close()

        assert logger.warning_messages == []
        assert not monitor.is_alive

    asyncio.run(run_test())


def test_monitor_reports_blocking_stack_and_recovery_once() -> None:
    async def run_test() -> None:
        logger = RecordingLogger()
        monitor = EventLoopLagMonitor(
            logger=logger,
            threshold=0.03,
            probe_interval=0.005,
        )

        await monitor.start()
        await asyncio.sleep(0.02)
        time.sleep(0.09)
        await asyncio.sleep(0.03)
        await monitor.close()

        blocked = [
            message
            for message in logger.warning_messages
            if "[EVENT_LOOP_BLOCKED]" in message
        ]
        recovered = [
            message
            for message in logger.warning_messages
            if "[EVENT_LOOP_RECOVERED]" in message
        ]
        assert len(blocked) == 1
        assert len(recovered) == 1
        assert "test_monitor_reports_blocking_stack_and_recovery_once" in blocked[0]
        assert "threshold=0.030s" in blocked[0]
        assert "total_delay=" in recovered[0]

    asyncio.run(run_test())


def test_monitor_close_is_idempotent() -> None:
    async def run_test() -> None:
        logger = RecordingLogger()
        monitor = EventLoopLagMonitor(
            logger=logger,
            threshold=0.05,
            probe_interval=0.005,
        )

        await monitor.start()
        await monitor.close()
        await monitor.close()

        assert not monitor.is_alive

    asyncio.run(run_test())


def test_monitor_can_be_signalled_from_a_non_async_shutdown_path() -> None:
    async def run_test() -> None:
        logger = RecordingLogger()
        monitor = EventLoopLagMonitor(
            logger=logger,
            threshold=0.05,
            probe_interval=0.005,
        )

        await monitor.start()
        monitor.stop()
        await monitor.close()

        assert not monitor.is_alive

    asyncio.run(run_test())
