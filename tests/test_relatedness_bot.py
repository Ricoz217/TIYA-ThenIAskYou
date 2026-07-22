import asyncio
import inspect
import json
from pathlib import Path

import pytest

from TIYA.relatedness import (
    BotCommunity,
    GroupRelatedness,
    MemberCommunityConfig,
    MessageInput,
    close_relatedness_runtime,
)
from TIYA.relatedness.storage import load_member_state


def _message(
    msg_id: str,
    text: str,
    *,
    user: str,
    timestamp: float,
) -> MessageInput:
    return MessageInput(
        msg_id=msg_id,
        group_id="group",
        user_id=user,
        timestamp=timestamp,
        text=text,
    )


def test_bot_state_survives_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        await relatedness.ingest(
            _message("trigger", "本地模型性能", user="alice", timestamp=1.0)
        )
        await relatedness.ingest(
            _message(
                "bot-output",
                "本地模型可以减少对象分配优化性能",
                user="bot",
                timestamp=2.0,
            )
        )

        first = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
            config=MemberCommunityConfig(interest_min_topic_size=1),
        )
        await first.start()
        await relatedness.refresh_analytics()
        await first.score("bot-output")
        await first.record_output(
            "bot-output",
        )
        await first.close()

        restored = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
            config=MemberCommunityConfig(interest_min_topic_size=1),
        )
        await restored.start()
        await relatedness.ingest(
            _message("query", "本地模型怎么优化性能", user="bob", timestamp=20.0)
        )

        score = await restored.score("query")

        assert score.interest > 0
        assert score.continuity > 0
        await restored.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_tiny_unfiltered_topic_does_not_create_interest(tmp_path: Path) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        await relatedness.ingest(
            _message("trigger", "量子纠缠实验", user="alice", timestamp=1.0)
        )
        await relatedness.ingest(
            _message("bot-output", "这个实验挺有意思", user="bot", timestamp=2.0)
        )
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
        )
        await relatedness.refresh_analytics()
        await bot.record_output("bot-output")
        await relatedness.ingest(
            _message("query", "量子纠缠实验", user="bob", timestamp=20.0)
        )

        score = await bot.score("query")

        assert score.interest == 0.0
        assert score.continuity > 0.0
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_record_output_is_required_to_seed_bot_continuity(tmp_path: Path) -> None:
    async def run_case(case_name: str, *, record_output: bool) -> float:
        relatedness = GroupRelatedness(
            group_id=case_name,
            data_path=tmp_path / case_name / "group",
        )
        await relatedness.start(())
        await relatedness.ingest(
            MessageInput(
                msg_id=f"{case_name}-trigger",
                group_id=case_name,
                user_id="alice",
                timestamp=1.0,
                text="继续讨论数据库索引",
            )
        )
        await relatedness.ingest(
            MessageInput(
                msg_id=f"{case_name}-output",
                group_id=case_name,
                user_id="bot",
                timestamp=2.0,
                text="索引需要结合查询模式",
            )
        )
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / case_name / "bot",
            config=MemberCommunityConfig(),
        )
        if record_output:
            await bot.record_output(f"{case_name}-output")
        await relatedness.ingest(
            MessageInput(
                msg_id=f"{case_name}-query",
                group_id=case_name,
                user_id="alice",
                timestamp=20.0,
                text="数据库索引查询模式",
            )
        )
        score = await bot.score(f"{case_name}-query")
        await bot.close()
        await relatedness.close()
        return score.continuity

    async def scenario() -> None:
        recorded_score = await run_case("recorded", record_output=True)
        automatic_score = await run_case("automatic", record_output=False)

        assert recorded_score > 0.0
        assert automatic_score == 0.0
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_record_output_requires_existing_bot_message(tmp_path: Path) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        await relatedness.ingest(
            _message("bot-output", "消息", user="bot", timestamp=2.0)
        )
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
        )
        await bot.start()

        try:
            await bot.record_output("missing")
        except ValueError as error:
            assert "member message" in str(error)
        else:
            raise AssertionError("missing BOT output must be rejected")

        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_output_content_not_dispatch_trigger_becomes_anchor(tmp_path: Path) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
        )
        await bot.start()
        await relatedness.ingest(
            _message("dispatch", "量子纠缠实验", user="alice", timestamp=1.0)
        )
        await relatedness.ingest(
            _message("bot-output", "数据库复合索引", user="bot", timestamp=2.0)
        )
        await bot.record_output(
            "bot-output",
        )
        await relatedness.ingest(
            _message("dispatch-topic", "量子纠缠实验", user="bob", timestamp=20.0)
        )
        await relatedness.ingest(
            _message("output-topic", "数据库复合索引", user="carol", timestamp=21.0)
        )

        dispatch_score = await bot.score("dispatch-topic")
        output_score = await bot.score("output-topic")

        assert output_score.continuity > dispatch_score.continuity
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_record_output_api_contains_no_dispatch_metadata() -> None:
    parameters = inspect.signature(BotCommunity.record_output).parameters

    assert tuple(parameters) == ("self", "message_id")


def test_corrupt_bot_state_recovers_as_empty_model(tmp_path: Path) -> None:
    async def scenario() -> None:
        state_directory = tmp_path / "bot"
        state_directory.mkdir()
        (state_directory / "member_community_state.json").write_text(
            "{broken",
            encoding="utf-8",
        )
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        await relatedness.ingest(
            _message("query", "普通消息", user="alice", timestamp=1.0)
        )
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=state_directory,
        )

        score = await bot.score("query")

        assert score.interest == 0.0
        assert score.continuity == 0.0
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_member_state_before_automatic_interest_is_rejected(tmp_path: Path) -> None:
    state_path = tmp_path / "member_community_state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 2,
                "group_id": "group",
                "user_id": "bot",
                "saved_at": 10.0,
                "interest_anchors": [
                    {"message_id": "dispatch", "weight": 1.0, "updated_at": 1.0}
                ],
                "tracks": [],
            }
        ),
        encoding="utf-8",
    )

    state = load_member_state(state_path)

    assert state.group_id == ""
    assert state.topic_interests == ()
