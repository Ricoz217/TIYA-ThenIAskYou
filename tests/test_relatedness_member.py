import asyncio
import json
from dataclasses import fields
from pathlib import Path

import pytest

from TIYA.relatedness import (
    BotCommunity,
    GroupRelatedness,
    MemberAffinityScore,
    MessageInput,
    close_relatedness_runtime,
)
from TIYA.relatedness.member import (
    MemberCommunity,
    _TopicObservation,
)
from TIYA.relatedness.models import (
    MessageReference,
    RelatednessConfig,
    TopicInfo,
    TopicSnapshot,
)
from TIYA.relatedness.storage import load_member_state


def _message(
    msg_id: str,
    text: str,
    *,
    user: str,
    timestamp: float,
    reply_to: str | None = None,
    mentions: frozenset[str] = frozenset(),
) -> MessageInput:
    return MessageInput(
        msg_id=msg_id,
        group_id="group",
        user_id=user,
        timestamp=timestamp,
        text=text,
        reply_to=reply_to,
        mention_ids=mentions,
    )


def test_public_member_score_contains_only_interest_and_continuity() -> None:
    assert [field.name for field in fields(MemberAffinityScore)] == [
        "interest",
        "continuity",
    ]


def test_recorded_bot_output_creates_a_chain_for_natural_messages(tmp_path: Path) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
        )
        await bot.start()
        await relatedness.ingest(_message(
            "bot-output", "python asyncio task", user="bot", timestamp=1.0,
        ))
        await bot.record_output("bot-output")
        await relatedness.ingest(_message(
            "source", "python asyncio", user="alice", timestamp=20.0,
        ))

        result = await bot.score("source")

        assert result.continuity > 0.25
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_reply_to_bot_creates_a_new_chain_and_relays_can_continue_it(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
        )
        await bot.start()
        await relatedness.ingest(_message(
            "bot-output", "alpha", user="bot", timestamp=1.0,
        ))
        await bot.record_output("bot-output")
        await relatedness.ingest(_message(
            "relay", "bravo", user="alice", timestamp=20.0,
            reply_to="bot-output",
        ))
        reply_score = await bot.score("relay")
        await relatedness.ingest(_message(
            "source", "charlie", user="bob", timestamp=40.0,
            reply_to="relay",
        ))

        result = await bot.score("source")

        assert reply_score.continuity == 1.0
        assert result.continuity > 0.70
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_complete_group_context_edge_can_maintain_a_dialogue_chain(tmp_path: Path) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
        )
        await bot.start()
        await relatedness.ingest(_message(
            "bot-output", "alpha", user="bot", timestamp=1.0,
        ))
        await bot.record_output("bot-output")
        await relatedness.ingest(_message(
            "source", "zulu", user="alice", timestamp=2.0,
        ))

        result = await bot.score("source")

        assert 0.15 <= result.continuity < 0.25
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_bot_interest_and_continuity_are_independent(tmp_path: Path) -> None:
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
            _message("trigger", "python asyncio 性能优化", user="alice", timestamp=1.0)
        )
        await relatedness.ingest(
            _message(
                "bot-output",
                "异步任务确实需要控制并发",
                user="bot",
                timestamp=2.0,
            )
        )
        await bot.record_output(
            "bot-output",
        )

        await relatedness.ingest(
            _message("related", "asyncio 并发怎么优化", user="alice", timestamp=20.0)
        )
        related = await bot.score("related")
        repeated = await bot.score("related")

        await relatedness.ingest(
            _message(
                "same-user-unrelated",
                "今晚吃红烧肉",
                user="alice",
                timestamp=20.5,
            )
        )
        same_user_unrelated = await bot.score("same-user-unrelated")

        await relatedness.ingest(
            _message("unrelated", "明天去海边", user="bob", timestamp=21.0)
        )
        unrelated = await bot.score("unrelated")

        assert related.continuity > unrelated.continuity
        assert related.continuity > 0.20
        assert repeated == related
        assert same_user_unrelated.continuity < 0.25
        assert unrelated.continuity < 0.25

        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_member_topic_profile_updates_incrementally(tmp_path: Path) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        for index in range(6):
            await relatedness.ingest(_message(
                f"topic-{index}",
                "python asyncio eventloop",
                user="bot" if index < 3 else f"u{index}",
                timestamp=float(index + 1),
            ))
        await relatedness.refresh_topics()
        await relatedness.ingest(_message(
            "query-1", "python asyncio", user="alice", timestamp=10.0,
        ))
        await relatedness.ingest(_message(
            "query-2", "python eventloop", user="alice", timestamp=11.0,
        ))
        calls: list[tuple[str, ...]] = []
        original = relatedness.get_topic_affinities

        async def recording(message_ids: tuple[str, ...]):
            calls.append(message_ids)
            return await original(message_ids)

        relatedness.get_topic_affinities = recording  # type: ignore[method-assign]
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
        )

        await bot.score("query-1")
        await bot.score("query-2")

        assert len(calls[0]) == 4
        assert calls[1] == ("query-2",)
        await relatedness.refresh_topics()
        await relatedness.ingest(_message(
            "query-3", "python task", user="alice", timestamp=12.0,
        ))
        await bot.score("query-3")
        assert len(calls[2]) == 4
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_time_only_penalizes_existing_continuity_evidence(tmp_path: Path) -> None:
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
            _message("bot-output", "数据库索引要看查询模式", user="bot", timestamp=1.0)
        )
        await bot.record_output("bot-output")

        await relatedness.ingest(
            _message("related", "这个查询模式该怎么建索引", user="alice", timestamp=30.0)
        )
        related = await bot.score("related")
        await relatedness.ingest(
            _message("unrelated", "晚饭吃红烧肉", user="bob", timestamp=31.0)
        )
        unrelated = await bot.score("unrelated")
        await relatedness.ingest(
            _message("faded", "查询模式和索引继续说", user="alice", timestamp=301.0)
        )
        faded = await bot.score("faded")

        assert related.continuity > 0.35
        assert unrelated.continuity < 0.15
        assert faded.continuity < related.continuity
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_repeated_bot_outputs_join_the_same_active_chain(tmp_path: Path) -> None:
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
            _message("output-1", "继续讨论 asyncio 并发", user="bot", timestamp=1.0)
        )
        await bot.record_output("output-1")
        await relatedness.ingest(
            _message("output-2", "asyncio 并发确实要限流", user="bot", timestamp=2.0)
        )
        await bot.record_output("output-2")

        await relatedness.ingest(
            _message("query", "asyncio 并发", user="alice", timestamp=3.0)
        )
        result = await bot.score("query")

        assert result.continuity > 0.0
        assert bot.member._continuity.active_chain is not None
        assert [
            node.message_id
            for node in bot.member._continuity.active_chain.nodes
        ][:2] == ["output-1", "output-2"]
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_dialogue_chains_are_restored_after_member_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(group_id="group", data_path=tmp_path / "group")
        await relatedness.start(())
        bot_path = tmp_path / "bot"
        first = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=bot_path,
        )
        await relatedness.ingest(
            _message("output", "python asyncio", user="bot", timestamp=1.0)
        )
        await first.record_output("output")
        await first.close()

        restored = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=bot_path,
        )
        await restored.start()
        await relatedness.ingest(
            _message("source", "python asyncio", user="alice", timestamp=2.0)
        )

        result = await restored.score("source")

        assert result.continuity > 0.0
        assert restored.member._continuity.active_chain is not None
        assert restored.member._continuity.active_chain.nodes[0].message_id == "output"
        await restored.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_member_state_v5_keeps_interest_and_starts_with_empty_chains(
    tmp_path: Path,
) -> None:
    path = tmp_path / "member.json"
    path.write_text(json.dumps({
        "version": 5,
        "group_id": "group",
        "user_id": "bot",
        "saved_at": 1.0,
        "topic_interests": [{
            "topic_id": "topic",
            "weight": 0.4,
            "updated_at": 1.0,
            "source_version": 2,
        }],
    }), encoding="utf-8")

    state = load_member_state(path)

    assert state.topic_interests[0].topic_id == "topic"
    assert state.dialogue_chains == ()


def test_continuity_is_zero_after_the_hard_time_window(tmp_path: Path) -> None:
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
            _message("trigger", "讨论数据库索引", user="alice", timestamp=1.0)
        )
        await relatedness.ingest(
            _message("bot-output", "索引要看查询模式", user="bot", timestamp=2.0)
        )
        await bot.record_output(
            "bot-output",
        )
        await relatedness.ingest(
            _message("late", "数据库索引怎么建", user="alice", timestamp=3_602.0)
        )

        result = await bot.score("late")

        assert result.continuity == 0.0
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_consecutive_low_contributions_close_fast_moving_conversation(
    tmp_path: Path,
) -> None:
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
            _message("trigger", "继续聊编译器", user="alice", timestamp=1.0)
        )
        await relatedness.ingest(
            _message("bot-output", "编译器后端挺有意思", user="bot", timestamp=2.0)
        )
        await bot.record_output(
            "bot-output",
        )
        for index in range(13):
            await relatedness.ingest(
                _message(
                    f"noise-{index}",
                    f"无关刷屏 {index}",
                    user=f"u{index}",
                    timestamp=3.0 + index,
                )
            )
            await bot.score(f"noise-{index}")
        await relatedness.ingest(
            _message("after-gap", "编译器后端继续", user="alice", timestamp=20.0)
        )

        result = await bot.score("after-gap")

        assert result.continuity == 0.0
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())


def test_member_configuration_rejects_invalid_continuity_limits() -> None:
    from TIYA.relatedness import MemberCommunityConfig

    with pytest.raises(ValueError, match="member community limits"):
        MemberCommunityConfig(continuity_chain_capacity=0)

    with pytest.raises(ValueError, match="between zero and one"):
        MemberCommunityConfig(continuity_propagation_decay=1.1)

    with pytest.raises(ValueError, match="durations"):
        MemberCommunityConfig(continuity_time_window_seconds=0.0)

    with pytest.raises(ValueError, match="interest topic limit"):
        MemberCommunityConfig(interest_topic_limit=0)


class _InterestRelatedness:
    group_id = "group"
    config = RelatednessConfig()

    def __init__(self, topics: TopicSnapshot) -> None:
        self._topics = topics

    def get_topics(self) -> TopicSnapshot:
        return self._topics

    def _register_audit_member(self, *args: object, **kwargs: object) -> None:
        pass


def _interest_test_member(tmp_path: Path) -> MemberCommunity:
    topic = TopicInfo(
        topic_id="topic",
        members=tuple((f"m-{index}", 1.0) for index in range(5)),
        representative_ids=(),
        last_active=10.0,
    )
    relatedness = _InterestRelatedness(
        TopicSnapshot(source_version=1, topics=(topic,))
    )
    return MemberCommunity(
        relatedness=relatedness,  # type: ignore[arg-type]
        user_id="bot",
        data_path=tmp_path,
    )


def _topic_observations(count: int) -> tuple[_TopicObservation, ...]:
    return tuple(
        _TopicObservation(
            reference=MessageReference(
                message_id=f"bot-{index}",
                user_id="bot",
                timestamp=10.0,
                sequence=index,
                content_key=("same-output",),
            ),
            affinities=(("topic", 0.5),),
        )
        for index in range(count)
    )


def test_topic_interest_preserves_absolute_evidence_strength(tmp_path: Path) -> None:
    member = _interest_test_member(tmp_path)

    score, _ = member._topic_interest_score(
        10.0,
        (("topic", 0.5),),
        _topic_observations(1),
    )
    weight = member.get_interest_topics()[0].weight

    assert weight == pytest.approx(0.20)
    assert score == pytest.approx(0.10)


def test_repeated_member_outputs_accumulate_instead_of_being_penalized(
    tmp_path: Path,
) -> None:
    one = _interest_test_member(tmp_path / "one")
    repeated = _interest_test_member(tmp_path / "repeated")

    one._topic_interest_score(10.0, (("topic", 0.5),), _topic_observations(1))
    repeated._topic_interest_score(
        10.0,
        (("topic", 0.5),),
        _topic_observations(4),
    )
    one_weight = one.get_interest_topics()[0].weight
    repeated_weight = repeated.get_interest_topics()[0].weight

    assert repeated_weight == pytest.approx(1.0 - (1.0 - 0.20) ** 4)
    assert repeated_weight > one_weight


def test_bot_interest_is_learned_from_recent_topic_distribution(tmp_path: Path) -> None:
    async def scenario() -> None:
        relatedness = GroupRelatedness(
            group_id="group",
            data_path=tmp_path / "group",
        )
        await relatedness.start(())
        for index in range(6):
            await relatedness.ingest(_message(
                f"python-{index}",
                "python asyncio eventloop",
                user="bot" if index < 3 else f"python-user-{index}",
                timestamp=float(index + 1),
            ))
        for index in range(6):
            await relatedness.ingest(_message(
                f"dinner-{index}",
                "晚饭 烧烤 火锅",
                user=f"dinner-user-{index}",
                timestamp=float(index + 20),
            ))
        await relatedness.refresh_analytics()
        await relatedness.ingest(_message(
            "python-query",
            "python asyncio 并发",
            user="alice",
            timestamp=40.0,
        ))
        await relatedness.ingest(_message(
            "dinner-query",
            "晚饭吃火锅",
            user="alice",
            timestamp=41.0,
        ))
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
        )

        python_score = await bot.score("python-query")
        interests = bot.get_interest_topics()
        dinner_score = await bot.score("dinner-query")

        assert python_score.interest > 0.15
        assert dinner_score.interest < python_score.interest * 0.5
        assert interests
        assert interests[0].weight > 0.5
        await bot.close()
        await relatedness.close()
        await close_relatedness_runtime()

    asyncio.run(scenario())
