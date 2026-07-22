from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .member import MemberCommunity
from .models import MemberAffinityScore, MemberCommunityConfig, TopicInterest


if TYPE_CHECKING:
    from .engine import GroupRelatedness


class BotCommunity:
    """BOT specialization of a member model without upstream trigger side effects."""

    __slots__ = ("member",)

    def __init__(
        self,
        *,
        relatedness: GroupRelatedness,
        bot_id: str,
        data_path: str | Path,
        config: MemberCommunityConfig | None = None,
    ) -> None:
        self.member = MemberCommunity(
            relatedness=relatedness,
            user_id=bot_id,
            data_path=data_path,
            config=config,
        )
        relatedness._register_audit_member(
            bot_id,
            self.member.config,
            role="bot",
        )

    async def start(self) -> None:
        await self.member.start()

    async def score(self, message_id: str) -> MemberAffinityScore:
        return await self.member.score(message_id)

    def get_interest_topics(self) -> tuple[TopicInterest, ...]:
        return self.member.get_interest_topics()

    async def record_output(
        self,
        message_id: str,
    ) -> None:
        """Record actual BOT content without importing dispatch metadata."""
        await self.member.record_member_message(message_id)

    async def save(self) -> None:
        await self.member.save()

    async def close(self) -> None:
        await self.member.close()
