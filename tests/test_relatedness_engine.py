from pathlib import Path
import asyncio

from TIYA.model.message import (
    AtMsg,
    FileMsg,
    GroupMsg,
    ImgMsg,
    RawMsg,
    ReplyMsg,
    SetuMsg,
    TextMsg,
)
from TIYA.relatedness import GroupRelatedness, NewWordConfig, RelatednessConfig
from TIYA.relatedness.models import StoredRelatednessState
from TIYA.relatedness.storage import save_state


def _message(
    msg_id: str,
    text: str,
    *,
    user_id: str = "user",
    reply_to: str | None = None,
) -> GroupMsg:
    content = []
    if reply_to:
        content.append(ReplyMsg(msg_id=reply_to))
    content.append(TextMsg(text=text))
    return GroupMsg(
        msg_id=msg_id,
        group_id="group",
        user_id=user_id,
        username=user_id,
        nickname="",
        time=float(msg_id) if msg_id.isdigit() else 1.0,
        content=content,
    )


def test_ingest_and_query_related_messages(tmp_path: Path) -> None:
    async def run() -> None:
        engine = GroupRelatedness(
            group_id="group",
            data_path=tmp_path,
            config=RelatednessConfig(maintenance_message_interval=1000),
        )
        await engine.start([])
        await engine.ingest(_message("1", "python asyncio 优化"))
        result = await engine.ingest(_message("2", "完全不同", reply_to="1"))

        related = await engine.get_related("2", explain=True)

        assert result.related[0].message_id == "1"
        assert related[0].details is not None
        assert related[0].details.reply > 0
        await engine.close()

    asyncio.run(run())


def test_topic_refresh_does_not_build_hotword_snapshot(tmp_path: Path) -> None:
    async def run() -> None:
        engine = GroupRelatedness(group_id="group", data_path=tmp_path)
        await engine.start([])
        await engine.ingest(_message("1", "python asyncio", user_id="u1"))
        await engine.ingest(_message("2", "python task", user_id="u2"))

        await engine.refresh_topics()

        assert engine.get_topics().source_version == 2
        assert engine.get_topics().topics
        assert engine.get_analytics().hotwords == ()
        assert engine.get_analytics().hot_sentences == ()
        await engine.close()

    asyncio.run(run())


def test_ingest_updates_live_topic_compensation(tmp_path: Path) -> None:
    async def run() -> None:
        engine = GroupRelatedness(group_id="group", data_path=tmp_path)
        await engine.start([])
        for index in range(4):
            await engine.ingest(
                _message(str(index + 1), "python asyncio", user_id=f"u{index}")
            )
        await engine.refresh_topics()

        result = await engine.ingest(
            _message("5", "python asyncio task", user_id="new-user")
        )

        topics = engine.get_topics()
        assert result.topic_affinities
        assert topics.source_version == 4
        assert topics.live_version == 5
        assert topics.compensations[0].message_id == "5"
        assert topics.activity
        cached = dict(await engine.get_topic_affinities(("5",)))
        assert cached["5"] == result.topic_affinities

        await engine.refresh_topics()
        refreshed = engine.get_topics()
        assert refreshed.source_version == 5
        assert refreshed.compensations == ()
        assert refreshed.activity
        await engine.close()

    asyncio.run(run())


def test_duplicate_message_is_idempotent(tmp_path: Path) -> None:
    async def run() -> None:
        engine = GroupRelatedness(group_id="group", data_path=tmp_path)
        await engine.start([])
        first = await engine.ingest(_message("1", "first"))
        second = await engine.ingest(_message("1", "changed"))

        assert second.version == first.version
        assert engine.message_count == 1
        await engine.close()

    asyncio.run(run())


def test_manual_new_word_refresh_updates_and_persists_dynamic_lexicon(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        engine = GroupRelatedness(
            group_id="group",
            data_path=tmp_path,
            config=RelatednessConfig(maintenance_message_interval=1000),
        )
        await engine.start([])
        docs = (
            _message("1", "\u661f\u6d41\u732b\u732b\u4eca\u5929\u5f88\u70ed\u95f9", user_id="a"),
            _message("2", "\u6211\u4e5f\u770b\u5230\u661f\u6d41\u732b\u732b\u8fd9\u4e2a\u6897", user_id="b"),
            _message("3", "\u7fa4\u91cc\u53c8\u5728\u804a\u661f\u6d41\u732b\u732b", user_id="c"),
            _message("4", "\u661f\u6d41\u732b\u732b\u771f\u7684\u5f88\u597d\u7b11", user_id="d"),
        )
        for message in docs:
            await engine.ingest(message)

        result = await engine.refresh_new_words(
            config=NewWordConfig(
                min_frequency=3,
                min_documents=3,
                min_users=2,
                min_npmi=0.05,
                min_boundary_entropy=0.0,
                min_promoted_score=0.0,
                enable_dynamic_stopwords=False,
                load_project_user_dict=False,
                use_project_stopwords=False,
            )
        )
        await engine.save()

        assert result.promoted
        assert "\u661f\u6d41\u732b\u732b" in engine.dynamic_terms
        assert any(
            item.term == "\u661f\u6d41\u732b\u732b"
            and item.expires_at > item.promoted_at
            for item in engine.dynamic_new_words
        )
        await engine.close()

        restored = GroupRelatedness(
            group_id="group",
            data_path=tmp_path,
            config=RelatednessConfig(maintenance_message_interval=1000),
        )
        await restored.start([])

        assert "\u661f\u6d41\u732b\u732b" in restored.dynamic_terms
        await restored.close()

    asyncio.run(run())


def test_state_round_trip_restores_live_topic_compensation(tmp_path: Path) -> None:
    async def run() -> None:
        messages = [
            _message(str(index + 1), "python asyncio", user_id=f"u{index}")
            for index in range(4)
        ]
        live = _message("5", "python asyncio task", user_id="new-user")
        engine = GroupRelatedness(group_id="group", data_path=tmp_path)
        await engine.start([])
        for message in messages:
            await engine.ingest(message)
        await engine.refresh_topics()
        await engine.ingest(live)
        await engine.save()
        await engine.close()

        restored = GroupRelatedness(group_id="group", data_path=tmp_path)
        await restored.start((*messages, live))

        topics = restored.get_topics()
        assert topics.compensations[0].message_id == "5"
        assert topics.activity
        await restored.close()

    asyncio.run(run())


def test_group_message_adapter_extracts_mentions() -> None:
    message = _message("1", "hello")
    message.content.insert(0, AtMsg(user_id="target"))

    adapted = GroupRelatedness.adapt_message(message)

    assert adapted.mention_ids == frozenset({"target"})
    assert adapted.text == "hello"


def test_group_message_adapter_extracts_structured_media_content() -> None:
    image = ImgMsg(
        name="cat.png",
        hash_name="image-hash",
        description="一只猫举着写有加班的牌子",
    )
    message = GroupMsg(
        msg_id="media",
        group_id="group",
        user_id="user",
        username="",
        nickname="",
        content=[
            ReplyMsg(msg_id="origin"),
            AtMsg(user_id="target"),
            image,
            FileMsg(filename="群聊计划.pdf", hash_name="file-hash"),
            SetuMsg(
                illust_id="42",
                page=0,
                hash_name="setu-hash",
                artist_id=7,
                artist_name="画师",
                type="illust",
                title="夏日",
                caption="海边插画",
                tags=["蓝天", "海浪"],
            ),
            RawMsg(type="face", content={"id": "66"}),
        ],
    )

    adapted = GroupRelatedness.adapt_message(message)

    assert adapted.text == ""
    assert "一只猫" in adapted.semantic_text
    assert "群聊计划.pdf" in adapted.semantic_text
    assert "夏日" in adapted.semantic_text
    assert adapted.media_ids == frozenset(
        {"image:image-hash", "setu:42:0", "face:66"}
    )
    assert adapted.reply_to == "origin"
    assert adapted.mention_ids == frozenset({"target"})


def test_pending_image_enriches_graph_without_reingest(tmp_path: Path) -> None:
    async def run() -> None:
        config = RelatednessConfig(
            maintenance_message_interval=1000,
            media_enrichment_timeout=1.0,
        )
        engine = GroupRelatedness(
            group_id="group",
            data_path=tmp_path,
            config=config,
        )
        await engine.start([])
        await engine.ingest(
            GroupMsg(
                msg_id="known",
                group_id="group",
                user_id="first",
                username="",
                nickname="",
                content=[
                    ImgMsg(
                        name="known.png",
                        hash_name="same-hash",
                        description="猫猫表情包",
                    )
                ],
            )
        )
        pending = ImgMsg(name="pending.png")
        await engine.ingest(
            GroupMsg(
                msg_id="pending",
                group_id="group",
                user_id="second",
                username="",
                nickname="",
                content=[pending],
            )
        )

        before = await engine.get_score("known", "pending")
        assert before is not None

        pending.hash_name = "same-hash"
        pending.description = "猫猫表情包"
        pending.event.set()
        for _ in range(20):
            await asyncio.sleep(0.01)
            after = await engine.get_score("known", "pending")
            if after is not None and after.text > before.text:
                break

        assert after is not None
        assert after.text > before.text
        await engine.close()

    asyncio.run(run())


def test_maintenance_dynamic_stopword_applies_to_future_messages(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        config = RelatednessConfig(
            maintenance_message_interval=1000,
            dynamic_stopword_min_documents=5,
        )
        engine = GroupRelatedness(
            group_id="group",
            data_path=tmp_path,
            config=config,
        )
        await engine.start([])
        for index in range(10):
            await engine.ingest(
                _message(
                    str(index + 1),
                    f"noise topic{index}",
                    user_id=f"u{index}",
                )
            )
        await engine.refresh_analytics()

        assert "noise" in engine.get_analytics().dynamic_stopwords

        await engine.ingest(
            _message("11", "noise final", user_id="last-user")
        )
        snapshot = await engine._runtime.run_realtime(
            engine._state.snapshot,
            enqueue_timeout=None,
        )
        newest = next(node for node in snapshot.nodes if node.msg_id == "11")
        assert "noise" not in newest.lexical_terms
        await engine.close()

    asyncio.run(run())


def test_legacy_dynamic_terms_are_not_loaded_as_new_words(tmp_path: Path) -> None:
    async def run() -> None:
        save_state(
            tmp_path / "relatedness_state.json",
            StoredRelatednessState(dynamic_terms=("legacy-fragment",)),
        )
        engine = GroupRelatedness(group_id="group", data_path=tmp_path)
        await engine.start([])

        assert engine.dynamic_terms == ()
        await engine.close()

    asyncio.run(run())


def test_analytics_refresh_cannot_replace_manual_new_word_lexicon(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        engine = GroupRelatedness(
            group_id="group",
            data_path=tmp_path,
            config=RelatednessConfig(maintenance_message_interval=1000),
        )
        await engine.start([])
        docs = (
            _message("1", "\u661f\u6d41\u732b\u732b\u4eca\u5929\u5f88\u70ed\u95f9", user_id="a"),
            _message("2", "\u6211\u4e5f\u770b\u5230\u661f\u6d41\u732b\u732b\u8fd9\u4e2a\u6897", user_id="b"),
            _message("3", "\u7fa4\u91cc\u53c8\u5728\u804a\u661f\u6d41\u732b\u732b", user_id="c"),
            _message("4", "\u661f\u6d41\u732b\u732b\u771f\u7684\u5f88\u597d\u7b11", user_id="d"),
        )
        for message in docs:
            await engine.ingest(message)
        await engine.refresh_new_words(
            config=NewWordConfig(
                min_frequency=3,
                min_documents=3,
                min_users=2,
                min_npmi=0.05,
                min_boundary_entropy=0.0,
                min_promoted_score=0.0,
                enable_dynamic_stopwords=False,
                load_project_user_dict=False,
                use_project_stopwords=False,
            )
        )
        expected = engine.dynamic_terms

        await engine.refresh_analytics()

        assert "\u661f\u6d41\u732b\u732b" in expected
        assert engine.dynamic_terms == expected
        assert engine.get_analytics().dynamic_terms == ()
        await engine.close()

    asyncio.run(run())
