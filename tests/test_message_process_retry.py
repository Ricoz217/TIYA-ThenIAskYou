import asyncio

import pytest

from TIYA.config import ConfigMap, DEFAULT_SETTING
from TIYA.model import message as message_module
from TIYA.model.message import (
    BaseMsg,
    GroupMsg,
    ImgMsg,
    MessageManager,
    PrivateMsg,
)


@pytest.fixture(autouse=True)
def _message_runtime_settings(monkeypatch):
    if "Message" not in message_module.SETTING_CFG:
        monkeypatch.setitem(
            message_module.SETTING_CFG,
            "Message",
            ConfigMap(DEFAULT_SETTING["Message"]),
        )


def _base_kwargs(message_id: str) -> dict[str, str]:
    return {
        "msg_id": message_id,
        "user_id": "user-1",
        "username": "User",
        "nickname": "",
    }


def test_broken_messages_do_not_process_content(monkeypatch) -> None:
    async def run_test() -> None:
        process_calls = 0

        async def fake_process(self, *args, **kwargs) -> None:
            nonlocal process_calls
            process_calls += 1

        monkeypatch.setattr(ImgMsg, "process", fake_process)
        group_message = GroupMsg(
            **_base_kwargs("group-broken"),
            group_id="group-1",
            content=[ImgMsg(name="small.png")],
            broken=True,
        )
        private_message = PrivateMsg(
            **_base_kwargs("private-broken"),
            content=[ImgMsg(name="small.png")],
            broken=True,
        )

        await asyncio.gather(group_message.process(), private_message.process())

        assert process_calls == 0
        assert not group_message.done
        assert not private_message.done

    asyncio.run(run_test())


def test_retry_discards_an_already_broken_message(monkeypatch) -> None:
    async def run_test() -> None:
        monkeypatch.setitem(message_module.SETTING_CFG.Message, "ProcessRetry", 10)
        manager = MessageManager(max_message=10)
        message = GroupMsg(
            **_base_kwargs("broken-retry"),
            group_id="group-1",
            broken=True,
        )
        manager._message_dict[message.msg_id] = message
        manager._process_fails[message.msg_id] = 1

        await manager._message_process_retry(None)

        assert message.msg_id not in manager._process_fails

    asyncio.run(run_test())


def test_retry_discards_a_message_missing_from_history(monkeypatch) -> None:
    async def run_test() -> None:
        monkeypatch.setitem(message_module.SETTING_CFG.Message, "ProcessRetry", 10)
        manager = MessageManager(max_message=10)
        manager._process_fails["missing"] = 1

        await manager._message_process_retry(None)

        assert "missing" not in manager._process_fails

    asyncio.run(run_test())


def test_successful_processing_clears_a_stale_failure() -> None:
    async def run_test() -> None:
        manager = MessageManager(max_message=10)
        message = GroupMsg(
            **_base_kwargs("successful"),
            group_id="group-1",
        )
        manager._process_fails[message.msg_id] = 1

        await manager._message_process_helper(message, None)

        assert message.done
        assert message.msg_id not in manager._process_fails

    asyncio.run(run_test())


def test_untracked_processing_does_not_create_a_retry_entry() -> None:
    async def run_test() -> None:
        class IncompleteMessage(BaseMsg):
            async def process(self, **kwargs) -> None:
                return

        manager = MessageManager(max_message=10)
        message = IncompleteMessage(**_base_kwargs("untracked"))

        await manager._message_process_helper(message, None, track_failure=False)

        assert message.msg_id not in manager._process_fails

    asyncio.run(run_test())


def test_retry_limit_marks_message_broken_and_stops_processing(monkeypatch) -> None:
    async def run_test() -> None:
        monkeypatch.setitem(message_module.SETTING_CFG.Message, "ProcessRetry", 1)
        process_calls = 0

        async def fake_process(self, *args, **kwargs) -> None:
            nonlocal process_calls
            process_calls += 1

        monkeypatch.setattr(ImgMsg, "process", fake_process)
        manager = MessageManager(max_message=10)
        message = GroupMsg(
            **_base_kwargs("retry-limit"),
            group_id="group-1",
            content=[ImgMsg(name="small.png")],
        )
        manager._message_dict[message.msg_id] = message

        await manager._message_process_helper(message, None)
        await manager._message_process_retry(None)
        await manager._message_process_retry(None)
        await manager._message_process_helper(message, None)

        assert process_calls == 2
        assert message.broken
        assert message.msg_id not in manager._process_fails

    asyncio.run(run_test())


def test_concurrent_retries_process_each_message_only_once(monkeypatch) -> None:
    async def run_test() -> None:
        monkeypatch.setitem(message_module.SETTING_CFG.Message, "ProcessRetry", 10)
        started = asyncio.Event()
        release = asyncio.Event()
        process_calls = 0

        class BlockingMessage(BaseMsg):
            async def process(self, **kwargs) -> None:
                nonlocal process_calls
                process_calls += 1
                started.set()
                await release.wait()

        manager = MessageManager(max_message=10)
        message = BlockingMessage(**_base_kwargs("singleflight"))
        manager._message_dict[message.msg_id] = message
        manager._process_fails[message.msg_id] = 0

        retries = [
            asyncio.create_task(manager._message_process_retry(None))
            for _ in range(5)
        ]
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(0)

        assert process_calls == 1

        release.set()
        await asyncio.gather(*retries)

        assert process_calls == 1
        assert manager._process_fails[message.msg_id] == 1

    asyncio.run(run_test())


def test_singleflight_is_shared_by_managers() -> None:
    async def run_test() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        process_calls = 0

        class BlockingMessage(BaseMsg):
            async def process(self, **kwargs) -> None:
                nonlocal process_calls
                process_calls += 1
                started.set()
                await release.wait()

        message = BlockingMessage(**_base_kwargs("shared-singleflight"))
        first_manager = MessageManager(max_message=10)
        second_manager = MessageManager(max_message=10)
        helpers = [
            asyncio.create_task(first_manager._message_process_helper(message, None)),
            asyncio.create_task(second_manager._message_process_helper(message, None)),
        ]
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(0)

        assert process_calls == 1

        release.set()
        await asyncio.gather(*helpers)

        assert process_calls == 1
        assert sum(first_manager._process_fails.values()) + sum(
            second_manager._process_fails.values()
        ) == 1

    asyncio.run(run_test())
