"""Rich-first logging module with layered terminal UI and file routing."""

from __future__ import annotations

import builtins
import inspect
import json
import math
import os
import queue
import ctypes
import select
import subprocess
import sys
import threading
import time
import traceback
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Set, TextIO
from uuid import uuid4

__all__ = [
    "LevelOptions",
    "LoggerConfig",
    "LogHandle",
    "BlockHandle",
    "Logger",
    "get_logger",
]

_LEVELS = ("STATUS_UI", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL")
_LEVEL_RANK = {"STATUS_UI": 5, "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "FATAL": 40}


@dataclass(slots=True)
class LevelOptions:
    """Per-level display options."""

    show_level: bool = True
    show_time: bool = True
    show_location: bool = False
    show_stack: bool = False


@dataclass(slots=True)
class LoggerConfig:
    """Logger settings for console, rendering, and file output."""

    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    queue_maxsize: int = 10_000
    use_console: bool = True
    renderer_backend: str = "rich"  # "rich" | "prompt_toolkit" | "plain"
    use_rich: bool = True
    auto_install_rich: bool = True
    new_window: bool = False
    use_background_color: bool = False
    background_color: str = "#1f2f4a"
    status_height: int = 3
    min_block_height: int = 4
    min_message_height: int = 6
    history_limit: int = 0
    block_history_limit: int = 400
    include_debug_in_app_log: bool = False
    render_failures_before_fallback: int = 6
    rolling_min_level: str = "DEBUG"
    render_min_interval_sec: float = 0.08
    maintenance_interval_sec: float = 1.0
    worker_activity_window_sec: float = 300.0
    worker_sort_cooldown_sec: float = 30.0
    worker_sort_ratio_threshold: float = 0.1
    worker_sort_min_delta: int = 2
    worker_status_history_limit: int = 80
    stream_tail_chars: int = 4000
    live_reserved_bottom_lines: int = 0
    status_idle_text: str = "STATUS: waiting for updates..."
    worker_empty_text: str = "No messages"
    worker_area_empty_text: str = "[worker-area] no active workers"
    typing_pause_enabled: bool = True
    typing_pause_timeout_sec: float = 15.0
    typing_pause_poll_interval_sec: float = 0.05
    typing_pause_debug: bool = False
    level_options: dict[str, LevelOptions] = field(
        default_factory=lambda: {
            "STATUS_UI": LevelOptions(show_level=False, show_time=False, show_location=False, show_stack=False),
            "DEBUG": LevelOptions(show_time=True, show_location=True, show_stack=False),
            "INFO": LevelOptions(show_time=True, show_location=False, show_stack=False),
            "WARNING": LevelOptions(show_time=True, show_location=True, show_stack=False),
            "ERROR": LevelOptions(show_time=True, show_location=True, show_stack=True),
            "FATAL": LevelOptions(show_time=True, show_location=True, show_stack=True),
        }
    )

    def level(self, name: str) -> LevelOptions:
        return self.level_options.get(name, LevelOptions())

    def resolved_new_window(self) -> bool:
        """Return effective new_window mode."""
        return bool(self.new_window)


@dataclass(slots=True)
class _DisplayLine:
    level: str
    text: str
    timestamp: str
    location: str | None
    stack: str | None
    show_level: bool = True
    show_time: bool = True
    show_location: bool = False
    stream_id: str | None = None


@dataclass(slots=True)
class _BlockState:
    title: str
    lines: Deque[_DisplayLine]
    streams: dict[str, _DisplayLine] = field(default_factory=dict)
    stream_visible: dict[str, _DisplayLine] = field(default_factory=dict)
    status_lines: Deque[_DisplayLine] = field(default_factory=deque)
    status_streams: dict[str, _DisplayLine] = field(default_factory=dict)
    active_content_streams: Set[str] = field(default_factory=set)
    activity: Deque[float] = field(default_factory=deque)
    created_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class _ConsoleState:
    status_line: _DisplayLine | None = None
    blocks: "OrderedDict[str, _BlockState]" = field(default_factory=OrderedDict)
    messages: Deque[_DisplayLine] = field(default_factory=deque)
    message_streams: dict[str, _DisplayLine] = field(default_factory=dict)
    message_stream_visible: dict[str, _DisplayLine] = field(default_factory=dict)


@dataclass(slots=True)
class _Event:
    kind: str
    payload: dict[str, Any]


@dataclass
class _PendingFileStream:
    metadata: _DisplayLine
    parts: list[str]
    dirty: bool = True


class LogHandle:
    """Handle for streaming append/update operations.

    A handle points to a specific stream target (status, rolling message, or block).
    Use ``write`` for append-style streaming and ``update`` for replace-style updates.
    """

    def __init__(
        self,
        logger: "Logger",
        stream_id: str,
        target: str,
        block_id: str | None = None,
        level: str = "INFO",
    ):
        self._logger = logger
        self._stream_id = stream_id
        self._target = target
        self._block_id = block_id
        self._level = level
        self._closed = False

    def write(
        self,
        *values: Any,
        sep: str = " ",
        end: str = "",
        flush: bool = False,
    ) -> None:
        """Append content to the current stream target."""
        if self._closed:
            return
        self._logger._stream_write(
            stream_id=self._stream_id,
            target=self._target,
            block_id=self._block_id,
            values=values,
            sep=sep,
            end=end,
            flush=flush,
            replace=False,
            level=self._level,
        )

    def update(
        self,
        *values: Any,
        sep: str = " ",
        end: str = "",
        flush: bool = False,
    ) -> None:
        """Replace the current stream line for this handle."""
        if self._closed:
            return
        self._logger._stream_write(
            stream_id=self._stream_id,
            target=self._target,
            block_id=self._block_id,
            values=values,
            sep=sep,
            end=end,
            flush=flush,
            replace=True,
            level=self._level,
        )

    def close(self) -> None:
        """Close this stream handle and stop future writes."""
        if self._closed:
            return
        self._closed = True
        self._logger._enqueue(
            _Event(
                kind="stream_close",
                payload={"stream_id": self._stream_id, "target": self._target, "block_id": self._block_id},
            )
        )

    def __enter__(self) -> "LogHandle":
        """Return this handle for ``with`` statements."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Always close stream handle when leaving ``with`` scope."""
        self.close()
        return False


class BlockHandle:
    """Handle tied to a specific worker block region.

    All logging methods route messages into the bound block_id.
    """

    def __init__(self, logger: "Logger", block_id: str):
        self._logger = logger
        self.block_id = block_id

    def log(self, *values: Any, level: str = "INFO", **kwargs: Any) -> LogHandle | None:
        """Log into this block with an explicit level."""
        kwargs["block_id"] = self.block_id
        return self._logger.log(level, *values, **kwargs)

    def info(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log an INFO message into this block."""
        kwargs["block_id"] = self.block_id
        return self._logger.info(*values, **kwargs)

    def debug(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log a DEBUG message into this block."""
        kwargs["block_id"] = self.block_id
        return self._logger.debug(*values, **kwargs)

    def warning(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log a WARNING message into this block."""
        kwargs["block_id"] = self.block_id
        return self._logger.warning(*values, **kwargs)

    def error(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log an ERROR message into this block."""
        kwargs["block_id"] = self.block_id
        return self._logger.error(*values, **kwargs)

    def fatal(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log a FATAL message into this block."""
        kwargs["block_id"] = self.block_id
        return self._logger.fatal(*values, **kwargs)

    def remove(self) -> None:
        """Remove this block from the UI layout."""
        self._logger.remove_block(self.block_id)

    def status(
        self,
        *values: Any,
        sep: str = " ",
        end: str = "",
        flush: bool = False,
        level: str = "STATUS_UI",
    ) -> LogHandle:
        """Update this block's right-side status area and return a stream handle."""
        return self._logger.worker_status(
            self.block_id, *values, sep=sep, end=end, flush=flush, level=level
        )

    def update_status(
        self,
        *values: Any,
        sep: str = " ",
        end: str = "",
        flush: bool = False,
        level: str = "STATUS_UI",
    ) -> LogHandle:
        """Alias for status() to keep API symmetry with update_block()."""
        return self.status(*values, sep=sep, end=end, flush=flush, level=level)


class _BaseRenderer:
    def start(self) -> None:
        return None

    def render(self, state: _ConsoleState) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        return None

    def needs_periodic_refresh(self) -> bool:
        return False


class _NullRenderer(_BaseRenderer):
    def render(self, state: _ConsoleState) -> None:  # noqa: ARG002
        return None


class _PlainRenderer(_BaseRenderer):
    def __init__(self, stream: TextIO | None = None):
        self._stream = stream or sys.stdout
        self._status = ""
        self._last_render = ""

    def _format_line(self, line: _DisplayLine) -> str:
        return _format_prefix(
            line.level,
            line.timestamp,
            line.location,
            line.show_level,
            line.show_time,
            line.show_location,
        ) + line.text

    def render(self, state: _ConsoleState) -> None:
        status_line = self._format_line(state.status_line) if state.status_line else ""
        if status_line != self._status:
            self._status = status_line
            builtins.print(f"\rSTATUS: {self._status}", end="", file=self._stream, flush=True)

        lines: list[str] = []
        for block_id, block in state.blocks.items():
            lines.append(f"\n[{block_id}] {block.title}")
            for line in list(block.lines)[-3:]:
                lines.append(f"  {self._format_line(line)}")
        for line in list(state.messages)[-8:]:
            lines.append(self._format_line(line))
        output = "\n".join(lines)
        if output and output != self._last_render:
            builtins.print(f"\n{output}", file=self._stream, flush=True)
            self._last_render = output


class _PromptToolkitRenderer(_BaseRenderer):
    """Prompt-toolkit based renderer backend.

    This renderer owns terminal drawing using prompt_toolkit Application while
    avoiding stdin capture (DummyInput), so external input managers can remain
    in control of command reads.
    """

    def __init__(self, config: LoggerConfig):
        self._config = config
        self._app: Any | None = None
        self._app_thread: threading.Thread | None = None
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._render_lock = threading.Lock()

        self._status_text = ""
        self._messages_text = ""
        self._blocks_text = ""
        self._stream_text = ""

        self._status_control: Any | None = None
        self._messages_control: Any | None = None
        self._blocks_control: Any | None = None
        self._stream_control: Any | None = None
        self._pipe_input_cm: Any | None = None
        self._pipe_input: Any | None = None

    def _format_line(self, line: _DisplayLine) -> str:
        parts: list[str] = []
        if line.show_level:
            parts.append(f"[{line.level}]")
        if line.show_time:
            parts.append(line.timestamp)
        if line.show_location and line.location:
            parts.append(line.location)
        prefix = (" ".join(parts) + " ") if parts else ""
        text = prefix + line.text
        if line.stack:
            text += f"\n{line.stack}"
        return text

    def _style_text(self, text: str) -> str:
        if not self._config.use_background_color:
            return text
        # prompt_toolkit supports inline ANSI colors.
        return f"\x1b[48;2;31;47;74m{text}\x1b[0m"

    def _build_blocks_text(self, state: _ConsoleState) -> str:
        if not state.blocks:
            return self._config.worker_area_empty_text
        chunks: list[str] = []
        for block_id, block in state.blocks.items():
            chunks.append(f"+-- {block.title} <{block_id}> " + "-" * 24)
            left_lines = [self._format_line(line) for line in list(block.lines)[-8:]]
            status_lines = [self._format_line(line) for line in list(block.status_lines)[-8:] if line.text.strip()]
            if status_lines:
                max_rows = max(len(left_lines), len(status_lines))
                left_lines = left_lines + [""] * (max_rows - len(left_lines))
                status_lines = status_lines + [""] * (max_rows - len(status_lines))
                for left, right in zip(left_lines, status_lines):
                    chunks.append(f"| {left}".ljust(90) + " | " + right)
            else:
                for line in left_lines:
                    chunks.append(f"| {line}")
            chunks.append("+" + "-" * 64)
        return "\n".join(chunks)

    def _build_messages_text(self, state: _ConsoleState) -> str:
        return "\n".join(self._format_line(line) for line in list(state.messages)[-120:])

    def _build_stream_text(self, state: _ConsoleState) -> str:
        if not state.message_stream_visible:
            return ""
        latest = list(state.message_stream_visible.values())[-1]
        return self._format_line(latest)

    def _build_status_text(self, state: _ConsoleState) -> str:
        if state.status_line is None:
            return self._config.status_idle_text
        return self._format_line(state.status_line)

    def start(self) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        self._status_control = FormattedTextControl(text="")
        self._messages_control = FormattedTextControl(text="")
        self._blocks_control = FormattedTextControl(text="")
        self._stream_control = FormattedTextControl(text="")

        root = HSplit(
            [
                Window(content=self._messages_control, wrap_lines=True),
                Window(height=1, content=FormattedTextControl(text=lambda: "-" * 120)),
                Window(content=self._stream_control, wrap_lines=True),
                Window(height=1, content=FormattedTextControl(text=lambda: "-" * 120)),
                Window(content=self._blocks_control, wrap_lines=True),
                Window(height=1, content=FormattedTextControl(text=lambda: "-" * 120)),
                Window(height=1, content=self._status_control, wrap_lines=True),
                Window(height=1, content=FormattedTextControl(text="")),
            ]
        )

        self._pipe_input_cm = create_pipe_input()
        self._pipe_input = self._pipe_input_cm.__enter__()

        self._app = Application(
            layout=Layout(root),
            full_screen=False,
            input=self._pipe_input,
            mouse_support=False,
        )

        def _run_app() -> None:
            self._started.set()
            try:
                self._app.run()
            finally:
                self._stopped.set()

        self._app_thread = threading.Thread(target=_run_app, name="logger-ptk-ui", daemon=True)
        self._app_thread.start()
        self._started.wait(timeout=0.5)

    def stop(self) -> None:
        app = self._app
        if app is not None:
            try:
                app.exit()
            except Exception:
                pass
        if self._app_thread is not None:
            self._app_thread.join(timeout=1.0)
        if self._pipe_input_cm is not None:
            try:
                self._pipe_input_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._pipe_input_cm = None
            self._pipe_input = None

    def render(self, state: _ConsoleState) -> None:
        if self._app is None:
            return
        with self._render_lock:
            status_text = self._build_status_text(state)
            messages_text = self._build_messages_text(state)
            blocks_text = self._build_blocks_text(state)
            stream_text = self._build_stream_text(state)

            changed = (
                status_text != self._status_text
                or messages_text != self._messages_text
                or blocks_text != self._blocks_text
                or stream_text != self._stream_text
            )
            if not changed:
                return

            self._status_text = status_text
            self._messages_text = messages_text
            self._blocks_text = blocks_text
            self._stream_text = stream_text

            if self._status_control is not None:
                self._status_control.text = self._style_text(self._status_text)
            if self._messages_control is not None:
                self._messages_control.text = self._style_text(self._messages_text)
            if self._blocks_control is not None:
                self._blocks_control.text = self._style_text(self._blocks_text)
            if self._stream_control is not None:
                self._stream_control.text = self._style_text(self._stream_text)
            self._app.invalidate()


class _RichRenderer(_BaseRenderer):
    def __init__(self, config: LoggerConfig):
        from rich.console import Console
        from rich.live import Live

        self._config = config
        self._bg_style = f"on {self._config.background_color}" if self._config.use_background_color else ""
        self._reserved_bottom_lines = max(0, int(self._config.live_reserved_bottom_lines))
        if self._bg_style:
            self._console = Console(style=self._bg_style)
        else:
            self._console = Console()
        self._live = Live(
            console=self._console,
            auto_refresh=False,
            screen=False,
            transient=False,
            vertical_overflow="visible",
        )
        self._last_message_count = 0
        self._last_tail_marker: tuple[Any, ...] | None = None
        self._last_stream_markers: dict[str, tuple[Any, ...]] = {}

    def _style(self, fg: str) -> str:
        return f"{fg} {self._bg_style}".strip()

    def _bg_or_none(self) -> str:
        return self._bg_style or "none"

    def start(self) -> None:
        self._live.start()

    def stop(self) -> None:
        self._live.stop()

    def _build_line(self, line: _DisplayLine):
        from rich.text import Text

        level_style = {
            "STATUS_UI": "white",
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "bold red",
            "FATAL": "bold red",
        }.get(line.level, "white")
        text = Text(style=self._bg_style if self._bg_style else None, no_wrap=False, overflow="fold")
        if line.show_level:
            text.append(f"[{line.level}] ", style=self._style(level_style))
        if line.show_time:
            text.append(f"{line.timestamp} ", style=self._style("dim"))
        if line.show_location and line.location:
            text.append(f"{line.location} ", style=self._style("magenta"))
        text.append(line.text, style=self._bg_style if self._bg_style else None)
        if line.stack:
            text.append("\n")
            text.append(line.stack, style=self._style("red"))
        return text

    def _build_wrapped_line(self, line: _DisplayLine, width: int):
        from rich.console import Group

        txt = self._build_line(line)
        wrapped = txt.wrap(self._console, max(8, width), overflow="fold", no_wrap=False)
        return Group(*wrapped)

    def _wrap_line_segments(self, line: _DisplayLine, width: int):
        txt = self._build_line(line)
        return txt.wrap(self._console, max(8, width), overflow="fold", no_wrap=False)

    def _tail_segments(self, lines: list[_DisplayLine], width: int, rows: int):
        if rows <= 0:
            return []
        captured: list[Any] = []
        for line in reversed(lines):
            segments = self._wrap_line_segments(line, width)
            for seg in reversed(segments):
                captured.append(seg)
                if len(captured) >= rows:
                    return list(reversed(captured))
        return list(reversed(captured))

    def _line_marker(self, line: _DisplayLine) -> tuple[Any, ...]:
        return (line.level, line.timestamp, line.location, line.text, line.stack)

    def _print_rolling_line(self, line: _DisplayLine, width: int) -> None:
        """Print a rolling-history line as full-width background to avoid black gaps."""
        from rich.align import Align

        self._console.print(
            Align.left(
                self._build_wrapped_line(line, width),
                width=width,
                style=self._bg_or_none(),
                pad=True,
            ),
            soft_wrap=True,
            overflow="fold",
            crop=False,
        )

    def needs_periodic_refresh(self) -> bool:
        return False

    def render(self, state: _ConsoleState) -> None:
        from rich import box
        from rich.align import Align
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        # Append-only message history: each rolling message is printed once.
        total_messages = len(state.messages)
        terminal_width = max(40, self._console.size.width - 1)
        if total_messages > self._last_message_count:
            new_lines = list(state.messages)[self._last_message_count :]
            for line in new_lines:
                self._print_rolling_line(line, terminal_width)
                if line.stream_id:
                    self._last_stream_markers[line.stream_id] = self._line_marker(line)
            self._last_message_count = total_messages
            tail = state.messages[-1]
            self._last_tail_marker = (tail.level, tail.timestamp, tail.location, tail.text, tail.stack)
        elif total_messages and total_messages == self._last_message_count:
            # Stream updates may replace previous visible entry and append a new one, keeping len unchanged.
            tail = state.messages[-1]
            marker = (tail.level, tail.timestamp, tail.location, tail.text, tail.stack)
            if marker != self._last_tail_marker:
                self._print_rolling_line(tail, terminal_width)
                self._last_tail_marker = marker
                if tail.stream_id:
                    self._last_stream_markers[tail.stream_id] = marker
        elif total_messages < self._last_message_count:
            # History was reset or truncated.
            self._last_message_count = total_messages
            if total_messages:
                tail = state.messages[-1]
                self._last_tail_marker = (tail.level, tail.timestamp, tail.location, tail.text, tail.stack)
            else:
                self._last_tail_marker = None

        terminal_height = max(16, self._console.size.height)
        body_height = terminal_height - 2

        status_renderable = (
            self._build_line(state.status_line)
            if state.status_line
            else Text(self._config.status_idle_text, style=self._style("bold white"))
        )
        separator = Text("-" * terminal_width, style=self._style("bright_cyan"))

        block_count = len(state.blocks)
        max_block_total_height = max(3, (terminal_height * 2) // 3)
        block_total_height = min(body_height, max_block_total_height)
        block_panels: list[Any] = []
        if block_count:
            per_block_cap = max(3, block_total_height // block_count)
            heights: list[int] = []
            ordered_items = list(state.blocks.items())
            for _, block in ordered_items:
                left_need = max(1, len(block.lines))
                if block.active_content_streams:
                    # Prioritize active stream visibility inside worker area.
                    left_need = max(left_need, per_block_cap - 1)
                status_has_content = any(line.text.strip() for line in block.status_lines)
                right_need = max(1, len(block.status_lines)) if status_has_content else 0
                demand = max(left_need, right_need if right_need else left_need) + 2
                heights.append(min(per_block_cap, max(3, demand)))

            # If total demand overflows worker area budget, shrink tallest blocks first.
            while sum(heights) > block_total_height:
                idx = max(range(len(heights)), key=lambda i: heights[i])
                if heights[idx] <= 3:
                    break
                heights[idx] -= 1

            for idx, (block_id, block) in enumerate(ordered_items):
                block_height = heights[idx]
                visible = max(1, block_height - 2)
                panel_inner_width = max(20, terminal_width - 4)
                left_lines_all = list(block.lines)
                status_lines = [line for line in block.status_lines if line.text.strip()]
                if status_lines:
                    right_col_width = max(20, min(80, panel_inner_width // 3))
                    left_col_width = max(12, panel_inner_width - right_col_width - 1)
                    active_lines: list[_DisplayLine] = []
                    normal_lines: list[_DisplayLine] = left_lines_all
                    if block.active_content_streams and block.stream_visible:
                        active_stream_ids = set(block.stream_visible.keys())
                        active_lines = [ln for ln in left_lines_all if ln.stream_id in active_stream_ids]
                        normal_lines = [ln for ln in left_lines_all if ln.stream_id not in active_stream_ids]
                    active_tail = self._tail_segments(active_lines, left_col_width, visible)
                    normal_room = max(0, visible - len(active_tail))
                    normal_tail = self._tail_segments(normal_lines, left_col_width, normal_room)
                    left_segments = normal_tail + active_tail

                    if left_segments:
                        left_text = Group(*left_segments)
                    else:
                        left_text = Text(self._config.worker_empty_text, style=self._style("dim"))
                    right_lines = status_lines[-visible:]
                    right_text = Group(*[self._build_wrapped_line(line, right_col_width) for line in right_lines])
                    inner = Table.grid(expand=True)
                    inner.add_column(ratio=3, no_wrap=False, overflow="fold")
                    inner.add_column(width=1)
                    inner.add_column(ratio=1, no_wrap=False, overflow="fold")
                    inner.add_row(
                        left_text,
                        Text("|", style=self._style("bright_cyan")),
                        right_text,
                    )
                    block_body: Any = inner
                else:
                    active_lines = []
                    normal_lines = left_lines_all
                    if block.active_content_streams and block.stream_visible:
                        active_stream_ids = set(block.stream_visible.keys())
                        active_lines = [ln for ln in left_lines_all if ln.stream_id in active_stream_ids]
                        normal_lines = [ln for ln in left_lines_all if ln.stream_id not in active_stream_ids]
                    active_tail = self._tail_segments(active_lines, panel_inner_width, visible)
                    normal_room = max(0, visible - len(active_tail))
                    normal_tail = self._tail_segments(normal_lines, panel_inner_width, normal_room)
                    left_segments = normal_tail + active_tail
                    if left_segments:
                        left_text = Group(*left_segments)
                    else:
                        left_text = Text(self._config.worker_empty_text, style=self._style("dim"))
                    block_body = left_text

                block_panels.append(
                    Panel(
                        block_body,
                        title=f"{block.title} <{block_id}>",
                        border_style="cyan",
                        box=box.ASCII,
                        height=block_height,
                        style=self._bg_or_none(),
                    )
                )

        blocks_renderable: Any = (
            Group(*block_panels, fit=False)
            if block_panels
            else Text(self._config.worker_area_empty_text, style=self._style("dim"))
        )

        stream_renderable: Any
        if state.message_stream_visible:
            stream_lines = list(state.message_stream_visible.values())[-1:]
            stream_renderable = Group(*[self._build_wrapped_line(line, terminal_width) for line in stream_lines])
        else:
            stream_renderable = Text(" " * terminal_width, style=self._bg_or_none())

        # Reserve one native terminal line for user input below the live region.
        input_line = Text(" " * terminal_width, style=self._bg_or_none())
        root = Group(stream_renderable, blocks_renderable, separator, status_renderable, input_line, fit=False)
        wrapped = Align.left(
            root,
            style=self._bg_or_none(),
            width=self._console.size.width,
            height=max(1, self._console.size.height - self._reserved_bottom_lines),
            pad=False,
        )
        self._live.update(wrapped, refresh=True)


class _SnapshotFileRenderer(_BaseRenderer):
    """Renderer proxy: writes state snapshots to a file consumed by monitor process."""

    def __init__(self, snapshot_file: Path):
        self._snapshot_file = snapshot_file
        self._snapshot_file.parent.mkdir(parents=True, exist_ok=True)

    def _write_payload(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        # Windows can briefly lock files across processes; retry before surfacing errors.
        last_exc: Exception | None = None
        for _ in range(5):
            try:
                with self._snapshot_file.open("w", encoding="utf-8") as fp:
                    fp.write(data)
                return
            except OSError as exc:
                last_exc = exc
                time.sleep(0.01)
        if last_exc is not None:
            raise last_exc

    def render(self, state: _ConsoleState) -> None:
        payload = {
            "shutdown": False,
            "state": _state_to_dict(state),
        }
        self._write_payload(payload)

    def stop(self) -> None:
        self._write_payload({"shutdown": True, "state": None})


def _state_to_dict(state: _ConsoleState) -> dict[str, Any]:
    return {
        "status": _line_to_dict(state.status_line) if state.status_line else None,
        "blocks": [
            {
                "id": block_id,
                "title": block.title,
                "lines": [_line_to_dict(line) for line in block.lines],
                "status_lines": [_line_to_dict(line) for line in block.status_lines],
            }
            for block_id, block in state.blocks.items()
        ],
        "messages": [_line_to_dict(line) for line in state.messages],
        "message_stream_visible": {sid: _line_to_dict(line) for sid, line in state.message_stream_visible.items()},
    }


def _state_from_dict(data: dict[str, Any]) -> _ConsoleState:
    status_data = data.get("status")
    status = _line_from_dict(status_data) if status_data else None
    state = _ConsoleState(status_line=status, messages=deque())
    for block_data in data.get("blocks", []):
        lines = deque((_line_from_dict(item) for item in block_data.get("lines", [])), maxlen=400)
        status_lines = deque((_line_from_dict(item) for item in block_data.get("status_lines", [])), maxlen=80)
        state.blocks[block_data["id"]] = _BlockState(
            title=block_data.get("title", block_data["id"]),
            lines=lines,
            status_lines=status_lines,
        )
    for item in data.get("messages", []):
        state.messages.append(_line_from_dict(item))
    for sid, item in data.get("message_stream_visible", {}).items():
        state.message_stream_visible[sid] = _line_from_dict(item)
    return state


def _line_to_dict(line: _DisplayLine) -> dict[str, Any]:
    return {
        "level": line.level,
        "text": line.text,
        "timestamp": line.timestamp,
        "location": line.location,
        "stack": line.stack,
        "show_level": line.show_level,
        "show_time": line.show_time,
        "show_location": line.show_location,
        "stream_id": line.stream_id,
    }


def _line_from_dict(data: dict[str, Any]) -> _DisplayLine:
    return _DisplayLine(
        level=data.get("level", "INFO"),
        text=data.get("text", ""),
        timestamp=data.get("timestamp", ""),
        location=data.get("location"),
        stack=data.get("stack"),
        show_level=bool(data.get("show_level", True)),
        show_time=bool(data.get("show_time", True)),
        show_location=bool(data.get("show_location", False)),
        stream_id=data.get("stream_id"),
    )


def _copy_line(line: _DisplayLine, *, text: str | None = None) -> _DisplayLine:
    return _DisplayLine(
        level=line.level,
        text=line.text if text is None else text,
        timestamp=line.timestamp,
        location=line.location,
        stack=line.stack,
        show_level=line.show_level,
        show_time=line.show_time,
        show_location=line.show_location,
        stream_id=line.stream_id,
    )


def _trim_stream_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _new_message_deque(limit: int) -> Deque[_DisplayLine]:
    if limit and limit > 0:
        return deque(maxlen=limit)
    return deque()


def _current_log_dir(logs_dir: Path) -> Path:
    return logs_dir / datetime.now().strftime("%Y-%m-%d")


def _format_prefix(
    level: str,
    timestamp: str,
    location: str | None,
    show_level: bool,
    show_time: bool,
    show_location: bool,
) -> str:
    parts: list[str] = []
    if show_level:
        parts.append(f"[{level}]")
    if show_time:
        parts.append(timestamp)
    if show_location and location:
        parts.append(location)
    return (" ".join(parts) + " ") if parts else ""


class Logger:
    """Threaded logger with layered rendering and multi-file routing.

    Public methods are synchronous from the caller perspective (queue submit),
    while rendering and file writes run in background threads.
    """

    def __init__(self, config: LoggerConfig | None = None):
        self.config = config or LoggerConfig()
        self._events: "queue.Queue[_Event]" = queue.Queue(maxsize=self.config.queue_maxsize)
        self._file_events: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=self.config.queue_maxsize)
        self._state = _ConsoleState(messages=_new_message_deque(self.config.history_limit))
        self._block_titles: dict[str, str] = {}
        self._pending_file_streams: dict[str, _PendingFileStream] = {}
        self._state_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._writer_shutdown = threading.Event()
        self._monitor_process: subprocess.Popen[str] | None = None
        self._render_failure_count = 0
        self._last_render_at = 0.0
        self._render_pending = threading.Event()
        self._render_paused = threading.Event()
        self._typing_pause_lock = threading.Lock()
        self._typing_pause_deadline = 0.0
        self._typing_debug_last_at = 0.0
        self._typing_hit_count = 0
        self._last_worker_reorder_at = 0.0
        self._render_lock = threading.Lock()
        self._renderer = self._build_renderer()
        self._writer_thread = threading.Thread(target=self._writer_loop, name="logger-writer", daemon=True)
        self._worker_thread = threading.Thread(target=self._worker_loop, name="logger-worker", daemon=True)
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop, name="logger-maintenance", daemon=True
        )
        self._typing_pause_thread: threading.Thread | None = None
        self._prepare_logs_dir()
        try:
            self._renderer.start()
        except Exception as exc:
            self._fallback_notice(f"renderer start failed ({exc!r}), using plain renderer.")
            self._renderer = _PlainRenderer()
            self._renderer.start()
        self._writer_thread.start()
        self._worker_thread.start()
        self._maintenance_thread.start()
        self._sync_typing_pause_thread()

    def update_config(self, config: LoggerConfig) -> None:
        """Apply a new runtime configuration to this logger instance."""
        if config is None:
            return
        with self._state_lock:
            self.config = config
            self._state.messages = deque(self._state.messages, maxlen=self._history_maxlen())
            for block in self._state.blocks.values():
                block.lines = deque(block.lines, maxlen=self.config.block_history_limit)
                block.status_lines = deque(block.status_lines, maxlen=self.config.worker_status_history_limit)
        self._prepare_logs_dir()
        self._restart_renderer()
        self._sync_typing_pause_thread()
        with self._state_lock:
            snapshot = self._clone_state()
        self._render_pending.set()
        self._try_render(snapshot, respect_throttle=False)

    def _history_maxlen(self) -> int | None:
        return self.config.history_limit if self.config.history_limit > 0 else None

    def _restart_renderer(self) -> None:
        with self._render_lock:
            old = self._renderer
            try:
                old.stop()
            except Exception:
                pass
            self._renderer = self._build_renderer()
            try:
                self._renderer.start()
            except Exception as exc:
                self._fallback_notice(f"renderer restart failed ({exc!r}), using plain renderer.")
                self._renderer = _PlainRenderer()
                self._renderer.start()

    def _sync_typing_pause_thread(self) -> None:
        should_run = bool(self.config.typing_pause_enabled and self.config.use_console)
        thread = self._typing_pause_thread
        if should_run and (thread is None or not thread.is_alive()):
            self._typing_pause_thread = threading.Thread(
                target=self._typing_pause_loop, name="logger-typing-pause", daemon=True
            )
            self._typing_pause_thread.start()
            self._typing_debug("typing pause thread started")

    def _build_renderer(self) -> _BaseRenderer:
        if not self.config.use_console:
            return _NullRenderer()
        backend = (self.config.renderer_backend or "rich").strip().lower()

        if backend == "plain":
            return _PlainRenderer()
        if backend == "prompt_toolkit":
            try:
                return _PromptToolkitRenderer(self.config)
            except Exception:
                self._fallback_notice("prompt_toolkit renderer failed to start, using plain renderer.")
                return _PlainRenderer()

        rich_imported = self._ensure_rich() if self.config.use_rich else False

        if self.config.resolved_new_window():
            if rich_imported:
                proxy = self._start_new_window_monitor()
                if proxy is not None:
                    return proxy
            self._fallback_notice("new_window requested but unavailable, using current terminal.")

        if rich_imported:
            try:
                return _RichRenderer(self.config)
            except Exception:
                self._fallback_notice("rich renderer failed to start, using plain renderer.")
        return _PlainRenderer()

    def _fallback_notice(self, msg: str) -> None:
        builtins.print(f"[logger] {msg}", file=sys.stderr, flush=True)

    def _ensure_rich(self) -> bool:
        try:
            __import__("rich")
            return True
        except ImportError:
            if not self.config.auto_install_rich:
                return False
        python_exe = sys.executable or "python"
        cmd = [python_exe, "-m", "pip", "install", "rich>=13.7.0"]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            __import__("rich")
            return True
        except Exception:
            return False

    def _start_new_window_monitor(self) -> _BaseRenderer | None:
        if os.name != "nt":
            return None
        project_root = Path(__file__).resolve().parents[2]
        snapshot_file = (self.config.logs_dir / ".logger_monitor_state.json").resolve()
        logger_file = Path(__file__).resolve()
        python_exe = sys.executable or "python"
        monitor_code = (
            "import importlib.util, sys; "
            f"spec=importlib.util.spec_from_file_location('logger_single', r'{logger_file}'); "
            "mod=importlib.util.module_from_spec(spec); "
            "sys.modules['logger_single']=mod; "
            "spec.loader.exec_module(mod); "
            f"mod._run_monitor_from_file(r'{snapshot_file}')"
        )
        try:
            self._monitor_process = subprocess.Popen(
                [
                    python_exe,
                    "-u",
                    "-c",
                    monitor_code,
                ],
                cwd=str(project_root),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            # Health-check: if monitor exits immediately, treat as failed startup.
            time.sleep(0.2)
            if self._monitor_process.poll() is not None:
                return None
            return _SnapshotFileRenderer(snapshot_file)
        except Exception:
            return None

    def _prepare_logs_dir(self) -> None:
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

    def create_block(self, block_id: str, title: str | None = None) -> BlockHandle:
        """Create a worker block and return a handle bound to it."""
        if not block_id:
            raise ValueError("block_id is required.")
        self._enqueue(_Event(kind="block_create", payload={"block_id": block_id, "title": title or block_id}))
        return BlockHandle(self, block_id)

    def remove_block(self, block_id: str) -> None:
        """Remove an existing worker block by block_id."""
        self._enqueue(_Event(kind="block_remove", payload={"block_id": block_id}))

    def status(
        self,
        *values: Any,
        sep: str = " ",
        end: str = "",
        flush: bool = False,
        level: str = "STATUS_UI",
    ) -> LogHandle:
        """Update status area and return a handle for real-time status updates."""
        stream_id = uuid4().hex
        self._stream_write(
            stream_id=stream_id,
            target="status",
            block_id=None,
            values=values,
            sep=sep,
            end=end,
            flush=flush,
            replace=True,
            level=level,
        )
        return LogHandle(self, stream_id=stream_id, target="status", level=level)

    def print(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """print-compatible shortcut routed to INFO."""
        return self.info(*values, **kwargs)

    def debug(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log a DEBUG message."""
        return self.log("DEBUG", *values, **kwargs)

    def info(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log an INFO message."""
        return self.log("INFO", *values, **kwargs)

    def warning(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log a WARNING message."""
        return self.log("WARNING", *values, **kwargs)

    def error(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log an ERROR message (same behavior as FATAL)."""
        return self.log("ERROR", *values, **kwargs)

    def fatal(self, *values: Any, **kwargs: Any) -> LogHandle | None:
        """Log a FATAL message (default behavior: record only, no exception raised)."""
        return self.log("FATAL", *values, **kwargs)

    def log(
        self,
        level: str,
        *values: Any,
        sep: str = " ",
        end: str = "\n",
        flush: bool = False,
        file: TextIO | None = None,
        block_id: str | None = None,
        stream: bool = False,
        show_time: bool | None = None,
        show_location: bool | None = None,
        show_stack: bool | None = None,
        exc: BaseException | None = None,
        exc_info: bool = False,
    ) -> LogHandle | None:
        """Unified logging entry point.

        Supports print-like arguments, optional block routing, and streaming mode.
        When ``stream=True`` this returns a ``LogHandle`` for incremental updates.
        """
        level = level.upper()
        if level not in _LEVELS:
            raise ValueError(f"unsupported level: {level}")
        if file is not None and file not in (sys.stdout, sys.stderr):
            builtins.print(*values, sep=sep, end=end, flush=flush, file=file)
            return None

        stream_id = uuid4().hex if stream else None
        payload = self._build_record(
            level=level,
            values=values,
            sep=sep,
            end=end,
            block_id=block_id,
            stream_id=stream_id,
            replace=False,
            show_time=show_time,
            show_location=show_location,
            show_stack=show_stack,
            exc=exc,
            exc_info=exc_info,
        )
        if stream and stream_id:
            payload["is_stream"] = True
            payload["target"] = "block" if block_id else "message"
            payload["checkpoint"] = flush
        self._enqueue(_Event(kind="log", payload=payload))
        if flush:
            self.flush()
        if stream and stream_id:
            target = "block" if block_id else "message"
            return LogHandle(self, stream_id=stream_id, target=target, block_id=block_id, level=level)
        return None

    def update_block(
        self,
        block_id: str,
        *values: Any,
        sep: str = " ",
        end: str = "",
        flush: bool = False,
        level: str = "INFO",
    ) -> LogHandle:
        """Create a replace-style stream handle for a block."""
        stream_id = uuid4().hex
        self._stream_write(
            stream_id=stream_id,
            target="block",
            block_id=block_id,
            values=values,
            sep=sep,
            end=end,
            flush=flush,
            replace=True,
            level=level,
        )
        return LogHandle(self, stream_id=stream_id, target="block", block_id=block_id, level=level)

    def worker_status(
        self,
        block_id: str,
        *values: Any,
        sep: str = " ",
        end: str = "",
        flush: bool = False,
        level: str = "STATUS_UI",
    ) -> LogHandle:
        """Update a worker's right-side status area and return a stream handle."""
        if not block_id:
            raise ValueError("block_id is required.")
        stream_id = uuid4().hex
        self._stream_write(
            stream_id=stream_id,
            target="block_status",
            block_id=block_id,
            values=values,
            sep=sep,
            end=end,
            flush=flush,
            replace=True,
            level=level,
        )
        return LogHandle(self, stream_id=stream_id, target="block_status", block_id=block_id, level=level)

    def update_worker_status(
        self,
        block_id: str,
        *values: Any,
        sep: str = " ",
        end: str = "",
        flush: bool = False,
        level: str = "STATUS_UI",
    ) -> LogHandle:
        """Alias for worker_status() to keep naming consistent with update_* APIs."""
        return self.worker_status(block_id, *values, sep=sep, end=end, flush=flush, level=level)

    def _stream_write(
        self,
        *,
        stream_id: str,
        target: str,
        block_id: str | None,
        values: tuple[Any, ...],
        sep: str,
        end: str,
        flush: bool,
        replace: bool,
        level: str = "INFO",
    ) -> None:
        if target == "block_status" and not block_id:
            raise ValueError("block_id is required for block_status target.")
        payload = self._build_record(
            level=level.upper(),
            values=values,
            sep=sep,
            end=end,
            block_id=block_id,
            stream_id=stream_id,
            replace=replace,
            show_time=None,
            show_location=None,
            show_stack=False,
            exc=None,
            exc_info=False,
        )
        payload["target"] = target
        payload["checkpoint"] = flush
        self._enqueue(_Event(kind="stream_write", payload=payload))
        if flush:
            self.flush()

    def _build_record(
        self,
        *,
        level: str,
        values: tuple[Any, ...],
        sep: str,
        end: str,
        block_id: str | None,
        stream_id: str | None,
        replace: bool,
        show_time: bool | None,
        show_location: bool | None,
        show_stack: bool | None,
        exc: BaseException | None,
        exc_info: bool,
    ) -> dict[str, Any]:
        if level not in _LEVELS:
            level = "INFO"
        options = self.config.level(level)
        use_level = options.show_level
        use_time = options.show_time if show_time is None else show_time
        use_location = options.show_location if show_location is None else show_location
        want_stack = options.show_stack if show_stack is None else show_stack
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location = self._caller_location() if use_location else None
        stack: str | None = None
        if exc is not None:
            stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        elif exc_info and want_stack:
            stack = "".join(traceback.format_stack(limit=10))

        text = sep.join(str(v) for v in values) + end
        return {
            "level": level,
            "text": text,
            "timestamp": timestamp,
            "location": location,
            "stack": stack if want_stack or exc is not None else None,
            "block_id": block_id,
            "stream_id": stream_id,
            "replace": replace,
            "show_level": use_level,
            "show_time": use_time,
            "show_location": use_location,
        }

    def _caller_location(self) -> str | None:
        for frame in inspect.stack(context=0)[2:]:
            module_name = frame.frame.f_globals.get("__name__", "")
            if module_name != __name__:
                return f"{Path(frame.filename).name}:{frame.lineno}"
        return None

    def _enqueue(self, event: _Event) -> None:
        try:
            self._events.put_nowait(event)
        except queue.Full:
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            else:
                self._events.task_done()
            try:
                self._events.put_nowait(event)
            except queue.Full:
                return

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set() or not self._events.empty():
            try:
                event = self._events.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with self._state_lock:
                    self._apply_event(event)
                    self._maybe_reorder_and_close_workers(time.monotonic())
                    snapshot = self._clone_state()
                self._try_render(snapshot, respect_throttle=True)
            finally:
                self._events.task_done()

    def _try_render(self, snapshot: _ConsoleState, *, respect_throttle: bool) -> None:
        if self._render_paused.is_set():
            self._render_pending.set()
            return
        now = time.monotonic()
        min_interval = max(0.0, self.config.render_min_interval_sec)
        if respect_throttle and min_interval > 0 and (now - self._last_render_at) < min_interval:
            self._render_pending.set()
            return
        with self._render_lock:
            try:
                self._renderer.render(snapshot)
                self._last_render_at = time.monotonic()
                self._render_failure_count = 0
                self._render_pending.clear()
            except Exception as exc:
                self._render_pending.set()
                self._render_failure_count += 1
                threshold = max(1, self.config.render_failures_before_fallback)
                if self._render_failure_count < threshold:
                    self._fallback_notice(
                        f"renderer error ({self._render_failure_count}/{threshold}): {exc!r}; retrying."
                    )
                else:
                    self._fallback_notice(
                        f"renderer failed {self._render_failure_count} times; switching to plain renderer."
                    )
                    self._renderer = _PlainRenderer()
                    try:
                        self._renderer.start()
                    except Exception:
                        self._renderer = _NullRenderer()

    def _maintenance_loop(self) -> None:
        last_maintenance_at = time.monotonic()
        while not self._shutdown.is_set():
            interval = max(0.1, self.config.maintenance_interval_sec)
            tick = min(0.05, interval)
            self._shutdown.wait(tick)
            if self._shutdown.is_set():
                break
            snapshot: _ConsoleState | None = None
            force_render = False
            now = time.monotonic()
            with self._state_lock:
                changed = False
                if (now - last_maintenance_at) >= interval:
                    changed = self._maybe_reorder_and_close_workers(now)
                    last_maintenance_at = now
                self._maybe_resume_render_from_timeout(now)
                pending = self._render_pending.is_set()
                if pending:
                    force_render = True
                if changed or pending or self._renderer.needs_periodic_refresh():
                    snapshot = self._clone_state()
            if snapshot is not None:
                self._try_render(snapshot, respect_throttle=not force_render)

    def _maybe_resume_render_from_timeout(self, now: float | None = None) -> None:
        if not self.config.typing_pause_enabled:
            return
        if not self._render_paused.is_set():
            return
        cur = time.monotonic() if now is None else now
        with self._typing_pause_lock:
            deadline = self._typing_pause_deadline
        if deadline > 0 and cur >= deadline:
            self._render_paused.clear()
            self._render_pending.set()
            self._typing_debug("auto resume on timeout")

    def note_input_activity(self) -> None:
        """Pause rendering for a short typing window."""
        if not self.config.typing_pause_enabled:
            return
        timeout = max(0.1, float(self.config.typing_pause_timeout_sec))
        with self._typing_pause_lock:
            self._typing_pause_deadline = time.monotonic() + timeout
        self._typing_hit_count += 1
        self._typing_debug(
            f"input activity detected hit={self._typing_hit_count} paused={self._render_paused.is_set()}",
            throttle_sec=0.2,
        )
        self._render_paused.set()

    def note_input_submit(self) -> None:
        """Resume rendering immediately after Enter/submit."""
        if not self.config.typing_pause_enabled:
            return
        with self._typing_pause_lock:
            self._typing_pause_deadline = 0.0
        self._render_paused.clear()
        self._render_pending.set()
        self._typing_debug("resume on submit")

    def _typing_pause_loop(self) -> None:
        poll = max(0.01, float(self.config.typing_pause_poll_interval_sec))
        if os.name == "nt":
            self._typing_pause_loop_windows(poll)
            return
        self._typing_pause_loop_posix(poll)

    def _typing_pause_loop_windows(self, poll: float) -> None:
        key_event_detector = self._build_windows_key_event_detector()
        self._typing_debug("windows key detector ready")
        while not self._shutdown.is_set():
            try:
                if key_event_detector():
                    self.note_input_activity()
            except Exception:
                pass
            self._shutdown.wait(poll)

    def _build_windows_key_event_detector(self):
        """Return a callable that peeks keyboard events without consuming input."""
        # Prefer Win32 console event peeking; fallback to msvcrt.kbhit.
        if os.name != "nt":
            return lambda: False
        try:
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
            kernel32.GetStdHandle.restype = wintypes.HANDLE
            kernel32.PeekConsoleInputW.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
            ]
            kernel32.PeekConsoleInputW.restype = wintypes.BOOL

            STD_INPUT_HANDLE = wintypes.DWORD(-10).value
            handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
            if handle in (0, -1):
                raise OSError("invalid console input handle")

            class _KEY_EVENT_RECORD(ctypes.Structure):
                _fields_ = [
                    ("bKeyDown", ctypes.c_int),
                    ("wRepeatCount", ctypes.c_ushort),
                    ("wVirtualKeyCode", ctypes.c_ushort),
                    ("wVirtualScanCode", ctypes.c_ushort),
                    ("UnicodeChar", ctypes.c_wchar),
                    ("dwControlKeyState", ctypes.c_uint),
                ]

            class _EVENT_UNION(ctypes.Union):
                _fields_ = [("KeyEvent", _KEY_EVENT_RECORD), ("Raw", ctypes.c_byte * 16)]

            class _INPUT_RECORD(ctypes.Structure):
                _fields_ = [("EventType", ctypes.c_ushort), ("Event", _EVENT_UNION)]

            KEY_EVENT = 0x0001
            records = (_INPUT_RECORD * 32)()
            count = wintypes.DWORD(0)

            def _peek() -> bool:
                ok = kernel32.PeekConsoleInputW(
                    handle,
                    ctypes.byref(records),
                    wintypes.DWORD(len(records)),
                    ctypes.byref(count),
                )
                if not ok:
                    return False
                total = int(count.value)
                if total <= 0:
                    return False
                for i in range(total):
                    rec = records[i]
                    if rec.EventType == KEY_EVENT and bool(rec.Event.KeyEvent.bKeyDown):
                        return True
                return False

            return _peek
        except Exception:
            try:
                import msvcrt  # type: ignore
            except Exception:
                return lambda: False
            self._typing_debug("fallback detector: msvcrt.kbhit")
            return msvcrt.kbhit

    def _typing_debug(self, msg: str, *, throttle_sec: float = 0.0) -> None:
        if not self.config.typing_pause_debug:
            return
        now = time.monotonic()
        if throttle_sec > 0 and (now - self._typing_debug_last_at) < throttle_sec:
            return
        self._typing_debug_last_at = now
        line = f"[typing-pause] {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} {msg}"
        self._file_enqueue("debug", line)

    def _typing_pause_loop_posix(self, poll: float) -> None:
        while not self._shutdown.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    self.note_input_activity()
            except Exception:
                # Non-tty stdin or unsupported environment.
                return
            self._shutdown.wait(poll)

    def _apply_event(self, event: _Event) -> None:
        if event.kind == "block_create":
            block_id = event.payload["block_id"]
            title = event.payload["title"]
            self._block_titles[block_id] = title
            self._state.blocks[block_id] = _BlockState(
                title=title,
                lines=deque(maxlen=self.config.block_history_limit),
                status_lines=deque(maxlen=self.config.worker_status_history_limit),
            )
            return
        if event.kind == "block_remove":
            block_id = event.payload["block_id"]
            self._state.blocks.pop(block_id, None)
            self._block_titles.pop(block_id, None)
            return
        if event.kind in {"log", "stream_write"}:
            is_stream = event.kind == "stream_write" or bool(event.payload.get("is_stream"))
            self._apply_record(event.payload, is_stream)
            return
        if event.kind == "stream_finalize_all":
            self._finalize_all_file_streams()
            return
        if event.kind == "stream_close":
            stream_id = event.payload["stream_id"]
            target = event.payload["target"]
            block_id = event.payload.get("block_id")
            if target in {"message", "block"}:
                self._finalize_file_stream(stream_id)
            if target == "status":
                return
            if target == "block" and block_id and block_id in self._state.blocks:
                block = self._state.blocks[block_id]
                block.streams.pop(stream_id, None)
                block.stream_visible.pop(stream_id, None)
                block.active_content_streams.discard(stream_id)
                return
            if target == "block_status" and block_id and block_id in self._state.blocks:
                self._state.blocks[block_id].status_streams.pop(stream_id, None)
                return
            self._state.message_stream_visible.pop(stream_id, None)
            self._state.message_streams.pop(stream_id, None)
            return

    def _prune_block_activity(self, block: _BlockState, now: float) -> None:
        window = max(0.1, self.config.worker_activity_window_sec)
        cutoff = now - window
        while block.activity and block.activity[0] < cutoff:
            block.activity.popleft()

    def _maybe_reorder_and_close_workers(self, now: float) -> bool:
        if not self._state.blocks:
            return False

        window = max(0.1, self.config.worker_activity_window_sec)
        stale_ids: list[str] = []
        for block_id, block in self._state.blocks.items():
            self._prune_block_activity(block, now)
            last_activity = block.activity[-1] if block.activity else block.created_at
            if (now - last_activity) > window:
                stale_ids.append(block_id)
        changed = False
        for block_id in stale_ids:
            if self._state.blocks.pop(block_id, None) is not None:
                changed = True

        if len(self._state.blocks) < 2:
            return changed

        # Keep visual stability while any worker has an active content stream.
        if any(block.active_content_streams for block in self._state.blocks.values()):
            return changed

        cooldown = max(0.0, self.config.worker_sort_cooldown_sec)
        if (now - self._last_worker_reorder_at) < cooldown:
            return changed

        current_ids = list(self._state.blocks.keys())
        counts = {block_id: len(block.activity) for block_id, block in self._state.blocks.items()}
        desired_ids = sorted(current_ids, key=lambda block_id: (-counts[block_id], current_ids.index(block_id)))
        if desired_ids == current_ids:
            return changed

        desired_idx = {block_id: idx for idx, block_id in enumerate(desired_ids)}

        should_reorder = False
        ratio = max(0.0, self.config.worker_sort_ratio_threshold)
        for upper_idx, upper_id in enumerate(current_ids):
            upper_count = counts[upper_id]
            for lower_idx in range(upper_idx + 1, len(current_ids)):
                lower_id = current_ids[lower_idx]
                if desired_idx[lower_id] < desired_idx[upper_id]:
                    lower_count = counts[lower_id]
                    delta = lower_count - upper_count
                    threshold = max(
                        self.config.worker_sort_min_delta,
                        math.ceil(ratio * max(1, upper_count)),
                    )
                    if delta > threshold:
                        should_reorder = True
                        break
            if should_reorder:
                break

        if not should_reorder:
            return changed

        reordered: "OrderedDict[str, _BlockState]" = OrderedDict()
        for block_id in desired_ids:
            reordered[block_id] = self._state.blocks[block_id]
        self._state.blocks = reordered
        self._last_worker_reorder_at = now
        changed = True
        return changed

    def _apply_record(self, payload: dict[str, Any], is_stream: bool) -> None:
        line = _DisplayLine(
            level=payload["level"],
            text=payload["text"].rstrip("\n"),
            timestamp=payload["timestamp"],
            location=payload["location"],
            stack=payload.get("stack"),
            show_level=bool(payload.get("show_level", True)),
            show_time=bool(payload.get("show_time", True)),
            show_location=bool(payload.get("show_location", False)),
            stream_id=payload.get("stream_id"),
        )
        target = payload.get("target")
        block_id = payload.get("block_id")
        stream_id = payload.get("stream_id")
        replace = bool(payload.get("replace", False))
        checkpoint = bool(payload.get("checkpoint", False))
        skip_file_emit = False

        if target == "status":
            self._state.status_line = line
            skip_file_emit = True
        elif target == "block_status" and block_id:
            block = self._state.blocks.get(block_id)
            if not block:
                block = _BlockState(
                    title=self._block_titles.get(block_id, block_id),
                    lines=deque(maxlen=self.config.block_history_limit),
                    status_lines=deque(maxlen=self.config.worker_status_history_limit),
                )
                self._state.blocks[block_id] = block
            self._apply_to_block_status(block, line, stream_id=stream_id, replace=replace, is_stream=is_stream)
            skip_file_emit = True
        elif block_id:
            block = self._state.blocks.get(block_id)
            if not block:
                block = _BlockState(
                    title=self._block_titles.get(block_id, block_id),
                    lines=deque(maxlen=self.config.block_history_limit),
                    status_lines=deque(maxlen=self.config.worker_status_history_limit),
                )
                self._state.blocks[block_id] = block
            self._apply_to_block(block, line, stream_id=stream_id, replace=replace, is_stream=is_stream)
        else:
            self._apply_to_messages(line, stream_id=stream_id, replace=replace, is_stream=is_stream)

        if skip_file_emit:
            return
        if is_stream and stream_id and target in {"message", "block"}:
            self._apply_to_file_stream(
                stream_id=stream_id,
                line=line,
                replace=replace,
                checkpoint=checkpoint,
            )
            return
        self._emit_file_lines(line)

    def _apply_to_block(
        self,
        block: _BlockState,
        line: _DisplayLine,
        *,
        stream_id: str | None,
        replace: bool,
        is_stream: bool,
    ) -> None:
        block.activity.append(time.monotonic())
        if is_stream and stream_id:
            block.active_content_streams.add(stream_id)
            if replace or stream_id not in block.streams:
                current = _copy_line(line)
                block.streams[stream_id] = current
            else:
                current = block.streams[stream_id]
                current.text += line.text
            current.text = _trim_stream_text(current.text, max_chars=self.config.stream_tail_chars)
            # Keep only one visible live snapshot per stream to avoid duplicate "active" lines.
            previous_visible = block.stream_visible.get(stream_id)
            if previous_visible is not None:
                try:
                    block.lines.remove(previous_visible)
                except ValueError:
                    pass
            snapshot = _copy_line(current)
            block.lines.append(snapshot)
            block.stream_visible[stream_id] = snapshot
            return
        block.lines.append(line)

    def _apply_to_block_status(
        self,
        block: _BlockState,
        line: _DisplayLine,
        *,
        stream_id: str | None,
        replace: bool,
        is_stream: bool,
    ) -> None:
        block.activity.append(time.monotonic())
        if is_stream and stream_id:
            existing = block.status_streams.get(stream_id)
            if existing is None:
                block.status_streams[stream_id] = line
                block.status_lines.append(line)
                return
            if replace:
                existing.level = line.level
                existing.text = line.text
                existing.timestamp = line.timestamp
                existing.location = line.location
                existing.stack = line.stack
                existing.show_level = line.show_level
                existing.show_time = line.show_time
                existing.show_location = line.show_location
                existing.stream_id = line.stream_id
                return
            existing.text += line.text
            return
        block.status_lines.append(line)

    def _apply_to_messages(
        self,
        line: _DisplayLine,
        *,
        stream_id: str | None,
        replace: bool,
        is_stream: bool,
    ) -> None:
        if is_stream and stream_id:
            if replace or stream_id not in self._state.message_streams:
                current = _copy_line(line)
                self._state.message_streams[stream_id] = current
            else:
                current = self._state.message_streams[stream_id]
                current.text += line.text
            current.text = _trim_stream_text(current.text, max_chars=self.config.stream_tail_chars)
            snapshot = _copy_line(current)
            self._state.message_stream_visible[stream_id] = snapshot
            return
        threshold = _LEVEL_RANK.get(self.config.rolling_min_level.upper(), _LEVEL_RANK["DEBUG"])
        current_level = _LEVEL_RANK.get(line.level, _LEVEL_RANK["INFO"])
        if current_level < threshold:
            return
        self._state.messages.append(line)

    def _apply_to_file_stream(
        self,
        *,
        stream_id: str,
        line: _DisplayLine,
        replace: bool,
        checkpoint: bool,
    ) -> None:
        state = self._pending_file_streams.get(stream_id)
        if state is None or replace:
            state = _PendingFileStream(
                metadata=_copy_line(line, text=""),
                parts=[line.text],
            )
            self._pending_file_streams[stream_id] = state
        else:
            state.parts.append(line.text)
            state.dirty = True

        if replace or checkpoint:
            self._emit_file_stream_state(state)

    def _emit_file_stream_state(self, state: _PendingFileStream) -> None:
        text = "".join(state.parts)
        self._emit_file_lines(_copy_line(state.metadata, text=text))
        state.parts = [text]
        state.dirty = False

    def _finalize_file_stream(self, stream_id: str) -> None:
        state = self._pending_file_streams.pop(stream_id, None)
        if state is not None and state.dirty:
            self._emit_file_stream_state(state)

    def _finalize_all_file_streams(self) -> None:
        for stream_id in list(self._pending_file_streams):
            self._finalize_file_stream(stream_id)

    def _emit_file_lines(self, line: _DisplayLine) -> None:
        prefix = _format_prefix(
            level=line.level,
            timestamp=line.timestamp,
            location=line.location,
            show_level=line.show_level,
            show_time=line.show_time,
            show_location=line.show_location,
        )
        full_line = prefix + line.text
        if line.stack:
            full_line += "\n" + line.stack.rstrip("\n")
        level = line.level

        if level == "DEBUG":
            self._file_enqueue("debug", full_line)
            if self.config.include_debug_in_app_log:
                self._file_enqueue("app", full_line)
            return

        self._file_enqueue("app", full_line)
        if level in {"WARNING", "ERROR", "FATAL"}:
            self._file_enqueue("error", full_line)

    def _file_enqueue(self, target: str, line: str) -> None:
        try:
            self._file_events.put_nowait((target, line))
        except queue.Full:
            try:
                self._file_events.get_nowait()
            except queue.Empty:
                pass
            else:
                self._file_events.task_done()
            try:
                self._file_events.put_nowait((target, line))
            except queue.Full:
                return

    def _writer_loop(self) -> None:
        files: dict[str, TextIO] = {}
        active_dir: Path | None = None

        def _close_files() -> None:
            nonlocal files
            for stream in files.values():
                try:
                    stream.close()
                except Exception:
                    pass
            files = {}

        def _ensure_files() -> None:
            nonlocal active_dir, files
            current_dir = _current_log_dir(self.config.logs_dir).resolve()
            if active_dir == current_dir and files:
                return
            _close_files()
            current_dir.mkdir(parents=True, exist_ok=True)
            opened: dict[str, TextIO] = {}
            try:
                opened["app"] = (current_dir / "app.log").open("a", encoding="utf-8")
                opened["debug"] = (current_dir / "debug.log").open("a", encoding="utf-8")
                opened["error"] = (current_dir / "error.log").open("a", encoding="utf-8")
            except Exception:
                for stream in opened.values():
                    try:
                        stream.close()
                    except Exception:
                        pass
                raise
            files = opened
            active_dir = current_dir

        try:
            while not self._writer_shutdown.is_set() or not self._file_events.empty():
                try:
                    target, line = self._file_events.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    _ensure_files()
                    stream = files[target]
                    stream.write(line.rstrip("\n") + "\n")
                    stream.flush()
                except Exception as exc:
                    _close_files()
                    active_dir = None
                    self._fallback_notice(f"failed to persist log record ({exc!r}); record dropped.")
                finally:
                    self._file_events.task_done()
        finally:
            _close_files()

    def _clone_state(self) -> _ConsoleState:
        status = self._state.status_line
        clone = _ConsoleState(status_line=status, messages=_new_message_deque(self.config.history_limit))
        clone.messages.extend(self._state.messages)
        blocks: "OrderedDict[str, _BlockState]" = OrderedDict()
        for block_id, block in self._state.blocks.items():
            stream_visible: dict[str, _DisplayLine] = {}
            for sid, vis in block.stream_visible.items():
                stream_visible[sid] = _copy_line(vis)
            blocks[block_id] = _BlockState(
                title=block.title,
                lines=deque(block.lines, maxlen=self.config.block_history_limit),
                streams=dict(block.streams),
                stream_visible=stream_visible,
                status_lines=deque(block.status_lines, maxlen=self.config.worker_status_history_limit),
                status_streams=dict(block.status_streams),
                active_content_streams=set(block.active_content_streams),
            )
        clone.blocks = blocks
        clone.message_streams = dict(self._state.message_streams)
        clone.message_stream_visible = {sid: _copy_line(line) for sid, line in self._state.message_stream_visible.items()}
        return clone

    def flush(self) -> None:
        """Block until event queue and file queue are fully drained."""
        self._events.join()
        self._file_events.join()

    def shutdown(self, timeout: float = 3.0) -> None:
        """Gracefully stop threads and renderer."""
        if self._shutdown.is_set():
            return
        self._events.put(_Event(kind="stream_finalize_all", payload={}))
        self._shutdown.set()
        self.flush()
        self._worker_thread.join(timeout=timeout)
        self._maintenance_thread.join(timeout=timeout)
        if self._typing_pause_thread is not None:
            self._typing_pause_thread.join(timeout=timeout)
        self._writer_shutdown.set()
        self._writer_thread.join(timeout=timeout)
        self._renderer.stop()
        if self._monitor_process is not None:
            try:
                self._monitor_process.wait(timeout=1.0)
            except Exception:
                pass

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.shutdown()


def _run_monitor_from_file(snapshot_path: str) -> None:
    """Run monitor loop in a separate console process."""
    path = Path(snapshot_path)
    renderer = _RichRenderer(
        LoggerConfig(
            use_console=True,
            use_rich=True,
            auto_install_rich=False,
            new_window=False,
            status_height=3,
            min_block_height=4,
            min_message_height=6,
        )
    )
    renderer.start()
    last_mtime = -1.0
    idle_deadline = time.time() + 1800.0
    try:
        while time.time() < idle_deadline:
            if not path.exists():
                time.sleep(0.05)
                continue

            stat = path.stat()
            mtime = stat.st_mtime
            if mtime <= last_mtime:
                time.sleep(0.05)
                continue
            last_mtime = mtime

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                time.sleep(0.05)
                continue

            if payload.get("shutdown"):
                break

            state_data = payload.get("state")
            if isinstance(state_data, dict):
                renderer.render(_state_from_dict(state_data))
                idle_deadline = time.time() + 1800.0
    finally:
        renderer.stop()


_DEFAULT_LOGGER: Logger | None = None
_DEFAULT_LOCK = threading.Lock()


def get_logger(config: LoggerConfig | None = None) -> Logger:
    """Return process-global logger instance."""
    global _DEFAULT_LOGGER
    with _DEFAULT_LOCK:
        if _DEFAULT_LOGGER is None:
            _DEFAULT_LOGGER = Logger(config=config)
        elif config is not None:
            _DEFAULT_LOGGER.update_config(config)
        return _DEFAULT_LOGGER
