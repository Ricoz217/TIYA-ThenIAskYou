from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from TIYA.agent.group_chat_agent import PostReason, _speaker_failure_guidance
from TIYA.dialog import group_dialog


class _FakeAQueue:
    def __init__(self) -> None:
        self.calls = []

    def add_task(self, coroutine, timeout):
        self.calls.append((coroutine, timeout))
        coroutine.close()


class _FakeAgent:
    run_queue_empty = True

    async def run_in_queue(self, reason: PostReason):
        return reason


class GroupSpeakerBackpressureTests(unittest.IsolatedAsyncioTestCase):
    def test_non_retryable_speaker_failure_guidance_forbids_retry(self) -> None:
        self.assertIn("请勿重试", _speaker_failure_guidance(False))

    async def test_random_speak_is_skipped_while_speaker_is_busy(self) -> None:
        dialog = object.__new__(group_dialog.GroupMainDialog)
        dialog.speaker_busy = True
        dialog.SPEAK_PROBABILITY = SimpleNamespace(
            get_probability=lambda: 1.0,
            wake=lambda: None,
        )
        dialog.BOT_COMMUNITY = SimpleNamespace(
            score=lambda _msg_id: asyncio.sleep(
                0,
                result=SimpleNamespace(continuity=0.0, interest=0.0),
            )
        )
        fake_aqueue = _FakeAQueue()
        dialog.host = SimpleNamespace(
            group_config=SimpleNamespace(speak_rate_max=1.0),
            logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
            aqueue=fake_aqueue,
        )
        dialog.AGENT = _FakeAgent()
        dialog.is_reply_or_at = lambda _msg: False

        with patch.object(group_dialog.random, "random", return_value=0.0):
            await dialog._agent_chat(SimpleNamespace(msg_id="message-1"))

        self.assertEqual(fake_aqueue.calls, [])

    async def test_speaker_lock_queue_timeout_is_non_retryable(self) -> None:
        dialog = object.__new__(group_dialog.GroupMainDialog)
        dialog._speak_lock = asyncio.Lock()
        dialog.speaker_busy = False
        dialog._speaker_task_count = 0
        failures = []
        speaker_calls = 0

        class _FakeSpeaker:
            def mark_failure(self, reason: str, *, retryable: bool = False) -> None:
                failures.append((reason, retryable))

            async def speak(self, **_kwargs):
                nonlocal speaker_calls
                speaker_calls += 1
                return None

        dialog.SPEAKER = _FakeSpeaker()
        await dialog._speak_lock.acquire()
        with patch.object(
            group_dialog,
            "SETTING_CFG",
            SimpleNamespace(LLM=SimpleNamespace(SpeakerRequestTimeout=0.01)),
        ):
            speak_task = asyncio.create_task(dialog.speak())
            await asyncio.sleep(0.03)

            self.assertTrue(dialog.speaker_busy)

            dialog._speak_lock.release()
            result = await speak_task

        self.assertIsNone(result)
        self.assertEqual(speaker_calls, 0)
        self.assertEqual(
            failures,
            [("Speaker 队列等待超时，发言内容已过期，任务已结束，无需重试", False)],
        )
        self.assertFalse(dialog.speaker_busy)
