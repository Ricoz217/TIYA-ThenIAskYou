from __future__ import annotations

import json
from pathlib import Path

import pytest

from TIYA.time_id import TimeBasedIdGenerator, get_global_time_id_generator


class FrozenClock:
    def __init__(self, values: list[int]) -> None:
        self._values = values
        self._index = 0

    def __call__(self) -> int:
        if self._index < len(self._values):
            value = self._values[self._index]
            self._index += 1
            return value
        return self._values[-1]


class AdvancingClock:
    def __init__(self, start_ms: int, step_ms: int) -> None:
        self._current_ms = start_ms
        self._step_ms = step_ms

    def __call__(self) -> int:
        current_ms = self._current_ms
        self._current_ms += self._step_ms
        return current_ms


def test_ids_are_unique_and_monotonic_when_time_does_not_move(tmp_path: Path) -> None:
    state_file = tmp_path / "time_id_state.json"
    clock = FrozenClock([1_700_000_000_000] * 100)
    gen = TimeBasedIdGenerator(state_file=state_file, time_ms_fn=clock)

    ids = [gen.next_id() for _ in range(5000)]

    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_burst_persists_one_time_window_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_file = tmp_path / "time_id_state.json"
    gen = TimeBasedIdGenerator(
        state_file=state_file,
        time_ms_fn=FrozenClock([1_700_000_000_000]),
    )
    original_replace = __import__("os").replace
    replace_calls = 0

    def counting_replace(src, dst):
        nonlocal replace_calls
        replace_calls += 1
        return original_replace(src, dst)

    monkeypatch.setattr("TIYA.time_id.os.replace", counting_replace)
    ids = [gen.next_id() for _ in range(10_000)]

    assert len(ids) == len(set(ids))
    assert replace_calls == 1


def test_steady_traffic_persists_once_per_minute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_file = tmp_path / "time_id_state.json"
    gen = TimeBasedIdGenerator(
        state_file=state_file,
        # 10 ms between IDs models 100 IDs per second for 100 seconds.
        time_ms_fn=AdvancingClock(1_700_000_000_000, 10),
    )
    original_replace = __import__("os").replace
    replace_calls = 0

    def counting_replace(src, dst):
        nonlocal replace_calls
        replace_calls += 1
        return original_replace(src, dst)

    monkeypatch.setattr("TIYA.time_id.os.replace", counting_replace)
    ids = [gen.next_id() for _ in range(10_000)]

    assert ids == sorted(ids)
    assert replace_calls == 2


def test_restart_does_not_conflict_with_history(tmp_path: Path) -> None:
    state_file = tmp_path / "time_id_state.json"

    first = TimeBasedIdGenerator(
        state_file=state_file,
        time_ms_fn=FrozenClock([1_700_000_000_000, 1_700_000_000_000]),
    )
    old_ids = {first.next_id(), first.next_id()}

    second = TimeBasedIdGenerator(
        state_file=state_file,
        time_ms_fn=FrozenClock([1_700_000_000_000]),
    )
    new_id = second.next_id()

    assert new_id not in old_ids
    assert new_id > max(old_ids)


def test_restarts_do_not_accumulate_reservation_window(tmp_path: Path) -> None:
    state_file = tmp_path / "time_id_state.json"
    now_ms = 1_800_000_000_000

    for _ in range(10):
        gen = TimeBasedIdGenerator(
            state_file=state_file,
            time_ms_fn=FrozenClock([now_ms]),
        )
        gen.next_id()

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted["last_ms"] <= now_ms + 60_010


def test_global_generator_is_singleton(tmp_path: Path) -> None:
    a = get_global_time_id_generator(state_file=tmp_path / "singleton_state.json")
    b = get_global_time_id_generator(state_file=tmp_path / "another_state.json")

    assert a is b


def test_next_id_survives_persist_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_file = tmp_path / "time_id_state.json"
    gen = TimeBasedIdGenerator(
        state_file=state_file,
        time_ms_fn=FrozenClock([1_700_000_000_000] * 10),
    )

    def always_fail_replace(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("TIYA.time_id.os.replace", always_fail_replace)
    ids = [gen.next_id() for _ in range(3)]

    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    assert gen._state.reserved_through_id is None


def test_next_id_survives_state_directory_error(tmp_path: Path) -> None:
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("occupied", encoding="utf-8")
    gen = TimeBasedIdGenerator(
        state_file=blocking_file / "time_id_state.json",
        time_ms_fn=FrozenClock([1_700_000_000_000]),
    )

    generated_id = gen.next_id()

    assert isinstance(generated_id, int)


def test_save_state_retries_on_transient_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_file = tmp_path / "time_id_state.json"
    gen = TimeBasedIdGenerator(
        state_file=state_file,
        time_ms_fn=FrozenClock([1_700_000_000_000]),
    )

    original_replace = __import__("os").replace
    calls = {"count": 0}

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("denied")
        return original_replace(src, dst)

    monkeypatch.setattr("TIYA.time_id.os.replace", flaky_replace)
    gen.next_id()

    assert calls["count"] == 3
    assert state_file.exists()

def test_default_state_file_is_data_root_anchored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from TIYA.config import DATA_DIR

    monkeypatch.chdir(tmp_path)
    gen = TimeBasedIdGenerator()

    assert gen._state_file == DATA_DIR / "time_id_state.json"
    assert "cache" not in gen._state_file.parts

