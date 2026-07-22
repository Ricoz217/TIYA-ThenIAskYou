from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from TIYA.dialog import group_dialog
from TIYA.model import persona


class _Prompt:
    async def get_prompt(self) -> str:
        return "system"


class _BatchBucket:
    def __init__(self, response: dict, resolved: dict[str, str]) -> None:
        self.response = [SimpleNamespace(text=json.dumps(response, ensure_ascii=False))]
        self.resolved = resolved
        self.resolve_aliases = AsyncMock(side_effect=self._resolve_aliases)
        self.resolve_alias = AsyncMock(side_effect=AssertionError("single alias API used"))
        self.get_memory = AsyncMock(
            side_effect=lambda mem_id: SimpleNamespace(bucket_id=f"owner:{mem_id}", kind="memory")
        )

    async def _resolve_aliases(self, aliases, **_kwargs) -> dict[str, str]:
        return {alias: self.resolved[alias] for alias in aliases if alias in self.resolved}

    async def advance_query(self, **_kwargs):
        return self.response


def test_group_persona_search_resolves_matches_once_before_loop() -> None:
    async def _run() -> None:
        bucket = _BatchBucket(
            {
                "answer": "found",
                "matches": {
                    "memory_1": {"score": 0.9},
                    "memory_999": {"score": 0.8},
                    "bucket_1": {"score": 0.7},
                    "ignored": {"score": 0.6},
                },
            },
            {"memory_1": "mem_real", "bucket_1": "bucket_real"},
        )
        target = object.__new__(persona.BaseGroupPersona)
        target.parent_bucket = bucket
        target.prompts = {"search_system": _Prompt()}
        target._ensure_bucket_handle = AsyncMock()

        result = await persona.BaseGroupPersona.search_memory(target, "cache", include_time=False)

        bucket.resolve_aliases.assert_awaited_once_with(
            ["memory_1", "memory_999", "bucket_1"]
        )
        bucket.resolve_alias.assert_not_awaited()
        assert result == {
            "answer": "found",
            "matches": [
                {"score": 0.9, "bucket_id": "owner:mem_real", "mem_id": "mem_real"},
                {"score": 0.7, "bucket_id": "bucket_real", "mem_id": ""},
            ],
        }

    asyncio.run(_run())


def test_remove_memory_candidates_resolve_once_and_keep_scoring_semantics() -> None:
    async def _run() -> None:
        bucket_handle = _BatchBucket(
            {
                "memory_1": 0.9,
                "memory_999": 0.8,
                "bucket_1": 0.7,
                "ignored": 0.6,
                "memory_2": "bad-score",
            },
            {"memory_1": "mem_real"},
        )
        bucket = SimpleNamespace(
            bucket=bucket_handle,
            _ensure_bucket_handle=AsyncMock(),
        )
        dialog = object.__new__(group_dialog.GroupMainDialog)
        dialog.GROUP_PERSONA = bucket
        dialog._ensure_member_list = AsyncMock(return_value=object())
        dialog.say = AsyncMock()

        with patch.object(group_dialog, "AgentPrompt", return_value=_Prompt()):
            result = await group_dialog.GroupMainDialog.get_remove_memory_list(
                dialog,
                target="group",
                query="cache",
                count=50,
                threshold=0.5,
            )

        bucket_handle.resolve_aliases.assert_awaited_once_with(["memory_1", "memory_999"])
        bucket_handle.resolve_alias.assert_not_awaited()
        assert result["bucket"] is bucket
        assert result["memories"] == {"mem_real": 0.9}
        dialog.say.assert_awaited_once()

    asyncio.run(_run())
