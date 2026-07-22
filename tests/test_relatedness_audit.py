import asyncio
import json
from pathlib import Path

from TIYA.relatedness import (
    BotCommunity,
    GroupRelatedness,
    MemberCommunityConfig,
    MemberAffinityScore,
    MessageInput,
)
from TIYA.relatedness.audit import RelatednessAudit
from TIYA.relatedness.runtime import RelatednessRuntime


def _message(
    message_id: str,
    text: str,
    *,
    user_id: str,
    timestamp: float,
) -> MessageInput:
    return MessageInput(
        msg_id=message_id,
        group_id="group",
        user_id=user_id,
        timestamp=timestamp,
        text=text,
    )


def test_public_api_automatically_writes_complete_audit(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = RelatednessRuntime()
        relatedness = GroupRelatedness(
            group_id="group",
            data_path=tmp_path / "group",
            runtime=runtime,
        )
        bot = BotCommunity(
            relatedness=relatedness,
            bot_id="bot",
            data_path=tmp_path / "bot",
            config=MemberCommunityConfig(interest_min_topic_size=1),
        )
        await relatedness.start(())
        await bot.start()

        output_ingest = await relatedness.ingest(
            _message("bot-output", "异步任务需要控制并发", user_id="bot", timestamp=1.0)
        )
        await relatedness.refresh_analytics()
        assert await bot.record_output("bot-output") is None
        ingest = await relatedness.ingest(
            _message("incoming", "asyncio 并发应该怎么控制", user_id="alice", timestamp=20.0)
        )

        score = await bot.score("incoming")
        cached = await bot.score("incoming")

        assert score.interest > 0.0
        assert score.continuity > 0.0
        assert cached == score
        assert output_ingest.features is not None
        assert ingest.related
        assert ingest.related[0].details is not None
        assert ingest.features is not None
        assert "asyncio" in ingest.features.lexical_terms
        assert ingest.text_candidates

        await bot.close()
        await relatedness.close()

        audit_files = tuple((tmp_path / "group" / "audit").glob("*.jsonl"))
        assert len(audit_files) == 1
        records = [
            json.loads(line)
            for line in audit_files[0].read_text(encoding="utf-8").splitlines()
        ]
        events = [record["event"] for record in records]
        assert events == [
            "session_start",
            "message_ingested",
            "member_output_recorded",
            "message_ingested",
            "member_scored",
            "member_scored",
            "session_end",
        ]
        start = records[0]["payload"]
        assert start["members"]["bot"]["role"] == "bot"
        assert start["relatedness_config"]
        ingested = records[3]["payload"]
        assert ingested["message"]["msg_id"] == "incoming"
        assert "asyncio" in ingested["result"]["features"]["lexical_terms"]
        scored = records[4]["payload"]
        assert scored["audit"]["cache_hit"] is False
        assert scored["audit"]["interest_evidence"]
        assert scored["audit"]["local_continuity"]["path_ids"]
        assert records[5]["payload"]["audit"]["cache_hit"] is True
        await runtime.close()

    asyncio.run(scenario())


def test_relatedness_audit_write_failure_never_blocks_shutdown(
    tmp_path: Path,
) -> None:
    class BrokenRuntime:
        @staticmethod
        async def run_maintenance(*_args, **_kwargs):
            raise OSError("disk unavailable")

    async def scenario() -> None:
        audit = RelatednessAudit(
            data_path=tmp_path,
            group_id="group",
            bot_id="bot",
            runtime=BrokenRuntime(),  # type: ignore[arg-type]
            session_id="broken",
            batch_size=1,
            flush_interval=0.01,
        )
        await audit.start()
        await asyncio.wait_for(audit.close(), timeout=1.0)

        assert audit.write_error == "OSError: disk unavailable"

    asyncio.run(scenario())


def test_relatedness_audit_writes_ordered_ndjson_and_drains_on_close(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runtime = RelatednessRuntime()
        audit = RelatednessAudit(
            data_path=tmp_path,
            group_id="group",
            bot_id="bot",
            runtime=runtime,
            session_id="session-test",
            queue_limit=8,
            batch_size=2,
            flush_interval=0.01,
        )
        await audit.start({"member_config": {"max_tracks": 4}})
        assert audit.emit("message_ingested", {
            "message": _message("m1", "测试", user_id="alice", timestamp=1.0),
            "score": MemberAffinityScore(interest=0.25, continuity=0.5),
        })
        assert audit.emit("member_scored", {"message_id": "m1"})
        await audit.close()

        records = [
            json.loads(line)
            for line in audit.path.read_text(encoding="utf-8").splitlines()
        ]
        assert [record["event"] for record in records] == [
            "session_start",
            "message_ingested",
            "member_scored",
            "session_end",
        ]
        assert [record["sequence"] for record in records] == [1, 2, 3, 4]
        assert all(record["schema_version"] == 1 for record in records)
        assert all(record["session_id"] == "session-test" for record in records)
        assert records[1]["payload"]["message"]["msg_id"] == "m1"
        assert records[1]["payload"]["score"]["continuity"] == 0.5
        assert records[-1]["payload"]["dropped_events"] == 0
        await runtime.close()

    asyncio.run(scenario())


def test_relatedness_audit_drops_instead_of_waiting_when_queue_is_full(
    tmp_path: Path,
) -> None:
    runtime = RelatednessRuntime()
    audit = RelatednessAudit(
        data_path=tmp_path,
        group_id="group",
        bot_id="bot",
        runtime=runtime,
        session_id="not-started",
        queue_limit=1,
    )

    assert audit.emit("first", {}) is True
    assert audit.emit("second", {}) is False
    assert audit.dropped_events == 1

    asyncio.run(runtime.close())
