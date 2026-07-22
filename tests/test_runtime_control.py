import asyncio
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import TIYA.runtime_control as runtime_control

from TIYA.runtime_control import (
    build_restart_command,
    clear_hot_reload_handler,
    hot_reload,
    pause_before_exit,
    request_restart,
    set_hot_reload_handler,
    save_groups,
    spawn_restart_process,
    start_new_groups,
    shutdown_groups,
)


class _FakeGroup:
    def __init__(self, *, fail_save: bool = False, fail_shutdown: bool = False):
        self.fail_save = fail_save
        self.fail_shutdown = fail_shutdown
        self.saved = False
        self.closed = False

    async def save(self):
        self.saved = True
        if self.fail_save:
            raise RuntimeError("save failed")

    async def shutdown(self):
        self.closed = True
        if self.fail_shutdown:
            raise RuntimeError("shutdown failed")


class _FakeStream:
    def __init__(self, interactive: bool):
        self.interactive = interactive

    def isatty(self):
        return self.interactive


class _GroupConfig:
    def __init__(self, *, chat_model: str, agent_model: str):
        self.chat_model = chat_model
        self.agent_model = agent_model


class _StartableGroup:
    def __init__(self, group_id: str):
        self.group_id = group_id
        self.started = False

    async def start(self):
        self.started = True


class _CoordinatedStartableGroup:
    active_starts = 0
    max_active_starts = 0
    both_started = None
    release = None

    def __init__(self, group_id: str):
        self.group_id = group_id
        self.started = False

    async def start(self):
        type(self).active_starts += 1
        type(self).max_active_starts = max(
            type(self).max_active_starts,
            type(self).active_starts,
        )
        if type(self).active_starts >= 2:
            type(self).both_started.set()

        await type(self).release.wait()
        self.started = True
        type(self).active_starts -= 1


class RuntimeControlTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        clear_hot_reload_handler()

    async def test_hot_reload_calls_registered_async_handler(self):
        result = object()
        handler = Mock(return_value=asyncio.sleep(0, result=result))
        set_hot_reload_handler(handler)

        self.assertIs(await hot_reload(), result)
        handler.assert_called_once_with()

    async def test_hot_reload_rejects_calls_before_runtime_registration(self):
        clear_hot_reload_handler()

        with self.assertRaisesRegex(RuntimeError, "尚未就绪"):
            await hot_reload()

    async def test_hot_reload_serializes_concurrent_calls(self):
        active_calls = 0
        max_active_calls = 0

        async def handler():
            nonlocal active_calls, max_active_calls
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            await asyncio.sleep(0)
            active_calls -= 1
            return object()

        set_hot_reload_handler(handler)

        await asyncio.gather(hot_reload(), hot_reload())

        self.assertEqual(max_active_calls, 1)

    async def test_restart_request_publishes_only_fixed_terminal_command(self):
        hub = Mock()
        with unittest.mock.patch(
            "TIYA.runtime_control.get_input_hub",
            return_value=hub,
        ):
            request_restart()

        hub.publish.assert_called_once_with("restart()")

    async def test_save_groups_attempts_every_group(self):
        good = _FakeGroup()
        bad = _FakeGroup(fail_save=True)

        errors = await save_groups([bad, good])

        self.assertTrue(bad.saved)
        self.assertTrue(good.saved)
        self.assertEqual(len(errors), 1)

    async def test_shutdown_groups_attempts_every_group(self):
        good = _FakeGroup()
        bad = _FakeGroup(fail_shutdown=True)

        errors = await shutdown_groups([bad, good])

        self.assertTrue(bad.closed)
        self.assertTrue(good.closed)
        self.assertEqual(len(errors), 1)

    async def test_shutdown_groups_respects_timeout(self):
        class _SlowGroup:
            async def shutdown(self):
                await asyncio.sleep(1)

        errors = await shutdown_groups([_SlowGroup()], timeout=0.01)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TimeoutError)

    async def test_start_new_groups_starts_only_new_configured_groups(self):
        existing_group = _StartableGroup("100")
        groups = {"100": existing_group}
        group_configs = {
            "300": _GroupConfig(chat_model="chat", agent_model="agent"),
        }
        initialized = []
        created = []

        def initialize_config(group_list):
            initialized.append(group_list)

        def make_group(group_data, group_config):
            group = _StartableGroup(str(group_data["group_id"]))
            groups[group.group_id] = group
            created.append((group_data, group_config, group))
            return group

        result = await start_new_groups(
            [
                {"group_id": 100, "group_name": "existing"},
                {"group_id": 200, "group_name": "skipped"},
                {"group_id": 300, "group_name": "new"},
            ],
            groups=groups,
            skipped_groups={"200"},
            group_configs=group_configs,
            initialize_config=initialize_config,
            make_group=make_group,
        )

        self.assertEqual(initialized, [{"300": "new"}])
        self.assertEqual(result.started, ["300"])
        self.assertEqual(result.pending_config, [])
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0][2].started)

    async def test_start_new_groups_starts_configured_groups_concurrently(self):
        groups = {}
        group_configs = {
            "300": _GroupConfig(chat_model="chat", agent_model="agent"),
            "400": _GroupConfig(chat_model="chat", agent_model="agent"),
        }
        _CoordinatedStartableGroup.active_starts = 0
        _CoordinatedStartableGroup.max_active_starts = 0
        _CoordinatedStartableGroup.both_started = asyncio.Event()
        _CoordinatedStartableGroup.release = asyncio.Event()

        def make_group(group_data, group_config):
            group = _CoordinatedStartableGroup(str(group_data["group_id"]))
            groups[group.group_id] = group
            return group

        start_task = asyncio.create_task(
            start_new_groups(
                [
                    {"group_id": 300, "group_name": "new-300"},
                    {"group_id": 400, "group_name": "new-400"},
                ],
                groups=groups,
                skipped_groups=set(),
                group_configs=group_configs,
                initialize_config=Mock(),
                make_group=make_group,
            )
        )

        try:
            await asyncio.wait_for(
                _CoordinatedStartableGroup.both_started.wait(),
                timeout=0.1,
            )
        finally:
            _CoordinatedStartableGroup.release.set()

        result = await start_task

        self.assertEqual(result.started, ["300", "400"])
        self.assertEqual(_CoordinatedStartableGroup.max_active_starts, 2)
        self.assertTrue(groups["300"].started)
        self.assertTrue(groups["400"].started)

    async def test_start_new_groups_keeps_unconfigured_groups_pending(self):
        groups = {}
        group_configs = {}

        def initialize_config(group_list):
            for group_id in group_list:
                group_configs[group_id] = _GroupConfig(
                    chat_model="YourModelPresetName",
                    agent_model="YourModelPresetName",
                )

        result = await start_new_groups(
            [{"group_id": 300, "group_name": "new"}],
            groups=groups,
            skipped_groups=set(),
            group_configs=group_configs,
            initialize_config=initialize_config,
            make_group=Mock(),
            strict=False,
        )

        self.assertEqual(result.started, [])
        self.assertEqual(result.pending_config, ["300"])
        self.assertEqual(groups, {})

    async def test_start_new_groups_rejects_unconfigured_groups_in_strict_mode(self):
        group_configs = {
            "300": _GroupConfig(
                chat_model="YourModelPresetName",
                agent_model="agent",
            ),
        }

        with self.assertRaisesRegex(RuntimeError, "300"):
            await start_new_groups(
                [{"group_id": 300, "group_name": "new"}],
                groups={},
                skipped_groups=set(),
                group_configs=group_configs,
                initialize_config=Mock(),
                make_group=Mock(),
                strict=True,
            )

    def test_pause_before_exit_waits_in_interactive_terminal(self):
        input_func = Mock()

        pause_before_exit("boom", stream=_FakeStream(True), input_func=input_func)

        input_func.assert_called_once()

    def test_pause_before_exit_skips_non_interactive_stream(self):
        input_func = Mock()

        pause_before_exit("boom", stream=_FakeStream(False), input_func=input_func)

        input_func.assert_not_called()

    def test_build_restart_command_uses_current_interpreter_and_module(self):
        command = build_restart_command(
            executable=r"D:\Python\project\.venv\Scripts\python.exe"
        )

        self.assertEqual(
            command,
            [
                r"D:\Python\project\.venv\Scripts\python.exe",
                "-m",
                "TIYA.QQ_bot",
            ],
        )

    def test_spawn_restart_process_uses_new_console_on_windows(self):
        popen = Mock()
        executable = r"D:\Python\project\.venv\Scripts\python.exe"
        cwd = Path(r"D:\Python\project")
        creation_flag = getattr(subprocess, "CREATE_NEW_CONSOLE", object())

        with patch.object(
                runtime_control.subprocess,
                "CREATE_NEW_CONSOLE",
                creation_flag,
                create=True,
        ):
            spawn_restart_process(
                executable=executable,
                cwd=cwd,
                platform_name="nt",
                popen=popen,
            )

        popen.assert_called_once()
        args, kwargs = popen.call_args
        self.assertEqual(args[0], [executable, "-m", "TIYA.QQ_bot"])
        self.assertEqual(kwargs["cwd"], str(cwd))
        self.assertIs(kwargs["creationflags"], creation_flag)
        self.assertFalse(kwargs["close_fds"])
        self.assertEqual(kwargs["env"]["TIYA_RESTARTED"], "1")

    def test_spawn_restart_process_starts_new_session_elsewhere(self):
        popen = Mock()
        cwd = Path("/tmp/project")

        spawn_restart_process(
            executable="/usr/bin/python3",
            cwd=cwd,
            platform_name="posix",
            popen=popen,
        )

        _, kwargs = popen.call_args
        self.assertTrue(kwargs["start_new_session"])
        self.assertNotIn("creationflags", kwargs)
        self.assertEqual(kwargs["env"]["TIYA_RESTARTED"], "1")


if __name__ == "__main__":
    unittest.main()
