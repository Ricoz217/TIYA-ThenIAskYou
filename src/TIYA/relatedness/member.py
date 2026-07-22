from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .continuity import DialogueChainModel
from .models import (
    TopicInterestAudit,
    TopicInterest,
    MemberAffinityScore,
    MemberCommunityConfig,
    MemberScoreAudit,
    MessageReference,
    StoredMemberCommunityState,
)
from .storage import load_member_state, save_member_state


if TYPE_CHECKING:
    from .engine import GroupRelatedness


@dataclass(frozen=True, slots=True)
class _TopicObservation:
    reference: MessageReference
    affinities: tuple[tuple[str, float], ...]


def _accumulate_bounded(current: float, evidence: float) -> float:
    bounded = min(1.0, max(0.0, evidence))
    return 1.0 - (1.0 - current) * (1.0 - bounded)


class MemberCommunity:
    """Small per-member affinity model backed by one group relation graph."""

    __slots__ = (
        "relatedness",
        "group_id",
        "user_id",
        "data_path",
        "config",
        "_topic_interests",
        "_topic_observations",
        "_topic_source_version",
        "_score_cache",
        "_continuity",
        "_continuity_lock",
        "_lock",
        "_started",
        "_closed",
    )

    def __init__(
        self,
        *,
        relatedness: GroupRelatedness,
        user_id: str,
        data_path: str | Path,
        config: MemberCommunityConfig | None = None,
    ) -> None:
        if not user_id:
            raise ValueError("member user id is required")
        self.relatedness = relatedness
        self.group_id = relatedness.group_id
        self.user_id = str(user_id)
        self.data_path = Path(data_path)
        self.config = config or MemberCommunityConfig()
        self._topic_interests: dict[str, TopicInterest] = {}
        self._topic_observations: dict[str, _TopicObservation] = {}
        self._topic_source_version = -1
        self._score_cache: OrderedDict[str, MemberScoreAudit] = OrderedDict()
        self._continuity = DialogueChainModel(
            self.config,
            self.relatedness.config,
        )
        self._continuity_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self.relatedness._register_audit_member(
            self.user_id,
            self.config,
            role="member",
        )

    @property
    def state_path(self) -> Path:
        return self.data_path / "member_community_state.json"

    def get_interest_topics(self) -> tuple[TopicInterest, ...]:
        return tuple(sorted(
            self._topic_interests.values(),
            key=lambda item: (-item.weight, item.topic_id),
        ))

    async def start(self) -> None:
        chains_loaded = False
        async with self._lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("member community is closed")
            stored = await self.relatedness._runtime.run_maintenance(
                f"{self.group_id}:{self.user_id}:member-load",
                load_member_state,
                self.state_path,
            )
            if (
                stored.group_id == self.group_id
                and stored.user_id == self.user_id
            ):
                self._topic_interests = {
                    item.topic_id: item for item in stored.topic_interests
                }
                self._continuity.replace_chains(stored.dialogue_chains)
                chains_loaded = bool(stored.dialogue_chains)
            self._started = True
        if chains_loaded:
            message_ids = self._continuity.message_ids
            references = await self.relatedness.get_message_references(
                message_ids,
                limit=max(1, len(message_ids)),
            )
            latest = await self.relatedness.get_message_references(limit=1)
            timestamp = latest[-1].timestamp if latest else stored.saved_at
            async with self._continuity_lock:
                self._continuity.reconcile_ids(
                    frozenset(item.message_id for item in references),
                    timestamp,
                )

    async def score(self, message_id: str) -> MemberAffinityScore:
        started = time.perf_counter()
        audit = await self._score_with_audit(message_id)
        self.relatedness._emit_audit("member_scored", {
            "member_id": self.user_id,
            "role": self.relatedness._audit_role(self.user_id),
            "message_id": message_id,
            "audit": audit,
            "total_ms": (time.perf_counter() - started) * 1_000.0,
        })
        return audit.score

    async def _score_with_audit(self, message_id: str) -> MemberScoreAudit:
        await self.start()
        async with self._lock:
            cached = self._score_cache.get(message_id)
            if cached is not None:
                return replace(cached, cache_hit=True)
        async with self._continuity_lock:
            local_snapshot = await self.relatedness.get_continuity_evidence(
                message_id,
                self._continuity.message_ids,
            )
            source = next(
                (node for node in local_snapshot.nodes if node.msg_id == message_id),
                None,
            )
            continuity = (
                self._continuity.process(
                    local_snapshot,
                    source_id=message_id,
                    member_id=self.user_id,
                )
                if source is not None
                else None
            )
        if source is None:
            result = MemberScoreAudit(
                source_id=message_id,
                source_user_id="",
                source_timestamp=0.0,
                source_sequence=-1,
                score=MemberAffinityScore(interest=0.0, continuity=0.0),
                evidence_missing=True,
                interest_topic_count=len(self._topic_interests),
            )
            async with self._lock:
                self._cache_score(message_id, result)
            return result
        references = await self.relatedness.get_message_references(
            user_id=self.user_id,
            limit=self.config.interest_message_window,
        )
        references = tuple(
            item for item in references
            if item.message_id != message_id
            and 0 <= source.timestamp - item.timestamp
            <= self.config.interest_time_window_seconds
        )
        source_affinities, observations = await self._sync_topic_observations(
            message_id,
            references,
        )
        assert continuity is not None
        async with self._lock:
            interest, interest_items = self._topic_interest_score(
                source.timestamp,
                source_affinities,
                observations,
            )
            score = MemberAffinityScore(
                interest=min(1.0, max(0.0, interest)),
                continuity=continuity.score,
            )
            result = MemberScoreAudit(
                source_id=message_id,
                source_user_id=source.user_id,
                source_timestamp=source.timestamp,
                source_sequence=source.sequence,
                score=score,
                interest_topic_count=len(self._topic_interests),
                interest_evidence=interest_items,
                local_continuity=continuity,
            )
            self._cache_score(message_id, result)
            return result

    async def _sync_topic_observations(
        self,
        source_id: str,
        references: tuple[MessageReference, ...],
    ) -> tuple[
        tuple[tuple[str, float], ...],
        tuple[_TopicObservation, ...],
    ]:
        topics = self.relatedness.get_topics()
        source_version = topics.source_version
        async with self._lock:
            rebuild = source_version != self._topic_source_version
            known_ids = set() if rebuild else set(self._topic_observations)
        pending = tuple(
            reference
            for reference in references
            if reference.message_id not in known_ids
        )
        affinity_rows = await self.relatedness.get_topic_affinities(
            (source_id, *(item.message_id for item in pending))
        )
        affinities = dict(affinity_rows)

        async with self._lock:
            if rebuild:
                self._topic_observations.clear()
            for reference in pending:
                self._topic_observations[reference.message_id] = _TopicObservation(
                    reference=reference,
                    affinities=affinities.get(reference.message_id, ()),
                )
            active_ids = {reference.message_id for reference in references}
            self._topic_observations = {
                message_id: observation
                for message_id, observation in self._topic_observations.items()
                if message_id in active_ids
            }
            self._topic_source_version = source_version
            observations = tuple(
                self._topic_observations[reference.message_id]
                for reference in references
                if reference.message_id in self._topic_observations
            )
        return affinities.get(source_id, ()), observations

    async def record_member_message(
        self,
        message_id: str,
    ) -> None:
        started = time.perf_counter()
        await self.start()
        async with self._continuity_lock:
            snapshot = await self.relatedness.get_continuity_evidence(
                message_id,
                self._continuity.message_ids,
            )
            continuity = self._continuity.record_member_output(
                snapshot,
                source_id=message_id,
                member_id=self.user_id,
            )

        self.relatedness._emit_audit("member_output_recorded", {
            "member_id": self.user_id,
            "role": self.relatedness._audit_role(self.user_id),
            "message_id": message_id,
            "continuity_source": "dialogue_chains",
            "continuity": continuity,
            "dialogue_chain_count": len(self._continuity.chains),
            "interest_topic_count": len(self._topic_interests),
            "total_ms": (time.perf_counter() - started) * 1_000.0,
        })

    async def save(self) -> None:
        await self.start()
        async with self._continuity_lock:
            dialogue_chains = self._continuity.chains
        async with self._lock:
            state = StoredMemberCommunityState(
                group_id=self.group_id,
                user_id=self.user_id,
                saved_at=time.time(),
                topic_interests=tuple(self._topic_interests.values()),
                dialogue_chains=dialogue_chains,
            )
        await self.relatedness._runtime.run_maintenance(
            f"{self.group_id}:{self.user_id}:member-save",
            save_member_state,
            self.state_path,
            state,
        )

    async def close(self) -> None:
        if self._closed:
            return
        await self.save()
        self._closed = True

    def _topic_interest_score(
        self,
        timestamp: float,
        source_affinities: tuple[tuple[str, float], ...],
        observations: tuple[_TopicObservation, ...],
    ) -> tuple[float, tuple[TopicInterestAudit, ...]]:
        topics = {
            topic.topic_id: topic
            for topic in self.relatedness.get_topics().topics
            if len(topic.members) >= self.config.interest_min_topic_size
        }
        raw: dict[str, float] = {}
        for observation in reversed(observations):
            reference = observation.reference
            age = max(0.0, timestamp - reference.timestamp)
            decay = math.exp(-age / self.config.interest_time_decay_seconds)
            for topic_id, similarity in observation.affinities:
                if similarity < self.config.interest_min_topic_similarity:
                    continue
                topic = topics.get(topic_id)
                if topic is None:
                    continue
                quality = (
                    self.config.interest_quality_base_weight
                    + self.config.interest_cohesion_weight * topic.cohesion
                    + self.config.interest_user_diversity_weight * topic.user_diversity
                    + self.config.interest_repeat_weight * topic.repeat_score
                    + self.config.interest_time_concentration_weight
                    * topic.time_concentration
                )
                raw[topic_id] = _accumulate_bounded(
                    raw.get(topic_id, 0.0),
                    similarity * decay * quality,
                )

        if raw:
            selected = sorted(raw.items(), key=lambda item: (-item[1], item[0]))[
                :self.config.interest_topic_limit
            ]
            self._topic_interests = {
                topic_id: TopicInterest(
                    topic_id=topic_id,
                    weight=value,
                    updated_at=timestamp,
                    source_version=self.relatedness.get_topics().source_version,
                )
                for topic_id, value in selected
            }
        else:
            active = set(topics)
            self._topic_interests = {
                topic_id: value
                for topic_id, value in self._topic_interests.items()
                if topic_id in active
            }

        source = dict(source_affinities)
        items: list[TopicInterestAudit] = []
        score = 0.0
        for topic_id, interest in self._topic_interests.items():
            similarity = source.get(topic_id, 0.0)
            topic = topics.get(topic_id)
            quality = topic.cohesion if topic is not None else 0.0
            contribution = interest.weight * similarity
            score = _accumulate_bounded(score, contribution)
            items.append(TopicInterestAudit(
                topic_id=topic_id,
                profile_weight=interest.weight,
                source_similarity=similarity,
                quality=quality,
                contribution=contribution,
            ))
        return score, tuple(items)

    def _cache_score(
        self,
        message_id: str,
        score: MemberScoreAudit,
    ) -> None:
        self._score_cache[message_id] = score
        self._score_cache.move_to_end(message_id)
        while len(self._score_cache) > 64:
            self._score_cache.popitem(last=False)
