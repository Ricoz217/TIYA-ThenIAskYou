from pathlib import Path
import asyncio

from TIYA.model.message import GroupMsg, TextMsg
from TIYA.relatedness import GroupRelatedness, RelatednessConfig


def test_history_rebuild_and_snapshot_refresh(tmp_path: Path) -> None:
    async def run() -> None:
        history = [
            GroupMsg(
                msg_id=str(index),
                group_id="group",
                user_id=f"u{index % 2}",
                username="",
                nickname="",
                time=float(index),
                content=[TextMsg(text="共同话题 python" if index < 3 else "晚饭做菜")],
            )
            for index in range(1, 6)
        ]
        engine = GroupRelatedness(
            group_id="group",
            data_path=tmp_path,
            config=RelatednessConfig(
                maintenance_message_interval=1000,
                topic_min_edge=0.05,
            ),
        )

        await engine.start(history)
        old_snapshot = engine.get_analytics()
        await engine.refresh_analytics()
        new_snapshot = engine.get_analytics()

        assert engine.message_count == 5
        assert old_snapshot.source_version <= new_snapshot.source_version
        assert new_snapshot.published_at >= old_snapshot.published_at
        await engine.close()

    asyncio.run(run())
