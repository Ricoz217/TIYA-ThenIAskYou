from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from TIYA.model.message import GroupMsg, ImgMsg

from .analytics import build_analytics
from .audit import RelatednessAudit
from .graph import MessageGraph
from .models import (
    AnalyticsKind,
    AnalyticsSnapshot,
    DynamicLexiconTerm,
    IngestResult,
    MemberCommunityConfig,
    MemberEvidenceSnapshot,
    MessageInput,
    MessageReference,
    RelatedMatch,
    RelatednessConfig,
    RelatednessScore,
    StoredRelatednessState,
    TopicSnapshot,
)
from .new_words import (
    NewWordConfig,
    NewWordDiscoveryResult,
    NewWordDocument,
    discover_new_words,
)
from .runtime import (
    RelatednessRuntime,
    RuntimeOverloadedError,
    get_relatedness_runtime,
)
from .storage import load_state, save_state
from .text import DynamicLexicon, TextIndex, TextProcessor
from .topics import (
    apply_topic_compensation,
    build_topics,
    message_topic_affinities,
)


if TYPE_CHECKING:
    from TIYA.model.message import BaseMsg


_STOPWORDS_PATH = Path(__file__).parents[3] / "data" / "dictionary" / "stopwords.json"
_BASE_NOISE_STOPWORDS = frozenset(
    {
        "www", "b23", "http", "https", "net", "pixiv", "artworks",
        "users", "bilibili", "com", "video", "share", "source", "copy",
        "web", "amp", "title", "artist", "tags", "图片", "内容", "标题",
        "总结",
    }
)


def _load_stopwords() -> set[str]:
    try:
        data = json.loads(_STOPWORDS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {
        str(item) for item in data if isinstance(item, str)
    } | set(_BASE_NOISE_STOPWORDS)


class _GroupComputationState:
    __slots__ = (
        "config",
        "base_stopwords",
        "dynamic_stopwords",
        "processor",
        "lexicon",
        "index",
        "graph",
    )

    def __init__(
        self,
        config: RelatednessConfig,
        dynamic_terms: tuple[str, ...] = (),
        dynamic_stopwords: tuple[str, ...] = (),
    ):
        self.config = config
        self.base_stopwords = frozenset(_load_stopwords())
        self.dynamic_stopwords = frozenset(dynamic_stopwords)
        self.processor = TextProcessor(
            config=config,
            stopwords=set(self.base_stopwords | self.dynamic_stopwords),
        )
        self.lexicon = DynamicLexicon(dynamic_terms, version=1 if dynamic_terms else 0)
        self.index = TextIndex(config=config)
        self.graph = MessageGraph(config)

    def ingest(
        self,
        message: MessageInput,
        topics: TopicSnapshot | None = None,
    ) -> IngestResult:
        if self.graph.contains(message.msg_id):
            return IngestResult(
                message_id=message.msg_id,
                related=self.graph.related(message.msg_id, limit=10, explain=True),
                version=self.graph.version,
            )
        features = self.processor.extract(message, self.lexicon)
        candidates = self.index.search(features)
        evicted = self.graph.add(message, features, candidates)
        self.index.add(message.msg_id, features)
        for msg_id in evicted:
            self.index.remove(msg_id)
        topic_affinities = (
            message_topic_affinities(features.term_weights, topics)
            if topics is not None and topics.topics
            else ()
        )
        return IngestResult(
            message_id=message.msg_id,
            related=self.graph.related(message.msg_id, limit=10, explain=True),
            version=self.graph.version,
            features=features,
            text_candidates=candidates,
            evicted_message_ids=tuple(evicted),
            topic_affinities=topic_affinities,
        )

    def enrich(
        self,
        message: MessageInput,
        topics: TopicSnapshot,
    ) -> tuple[
        bool,
        tuple[tuple[str, float], ...],
        tuple[tuple[str, float], ...],
        int,
    ]:
        features = self.processor.extract(message, self.lexicon)
        self.index.remove(message.msg_id)
        candidates = self.index.search(features)
        changed = self.graph.enrich(message, features, candidates)
        self.index.add(message.msg_id, features)
        affinities = (
            message_topic_affinities(features.term_weights, topics)
            if changed and topics.topics
            else ()
        )
        return changed, features.term_weights, affinities, self.graph.version

    def related(
        self,
        message_id: str,
        limit: int,
        explain: bool,
    ) -> tuple[RelatedMatch, ...]:
        return self.graph.related(message_id, limit=limit, explain=explain)

    def score(self, source_id: str, target_id: str) -> RelatednessScore | None:
        return self.graph.score(source_id, target_id)

    def anchor_evidence(
        self,
        source_id: str,
        anchor_ids: tuple[str, ...],
    ) -> MemberEvidenceSnapshot | None:
        return self.graph.anchor_evidence(source_id, anchor_ids)

    def message_references(
        self,
        message_ids: tuple[str, ...],
        user_id: str | None,
        limit: int,
    ) -> tuple[MessageReference, ...]:
        return self.graph.message_references(message_ids, user_id, limit)

    def snapshot(self):
        return self.graph.snapshot()

    def continuity_snapshot(
        self,
        source_id: str,
        window_seconds: float,
        message_limit: int,
    ):
        return self.graph.local_snapshot(
            source_id,
            window_seconds,
            message_limit,
        )

    def continuity_evidence_snapshot(
        self,
        source_id: str,
        target_ids: tuple[str, ...],
    ):
        return self.graph.continuity_evidence_snapshot(
            source_id,
            target_ids,
            include_reply_interval=True,
        )

    def topic_affinities(
        self,
        message_ids: tuple[str, ...],
        topics: TopicSnapshot,
    ) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        compensated = {
            item.message_id: item.affinities
            for item in topics.compensations
        }
        result = []
        for message_id in dict.fromkeys(message_ids):
            if message_id in compensated:
                result.append((message_id, compensated[message_id]))
                continue
            weights = self.graph.feature_weights(message_id)
            if weights is not None:
                result.append((message_id, message_topic_affinities(weights, topics)))
        return tuple(result)

    def set_dynamic_terms(self, terms: tuple[str, ...]) -> None:
        if terms == self.lexicon.terms:
            return
        self.lexicon = DynamicLexicon(terms, version=self.lexicon.version + 1)

    def set_dynamic_stopwords(
        self,
        dynamic_stopwords: tuple[str, ...],
    ) -> None:
        stopwords = frozenset(dynamic_stopwords)
        if stopwords == self.dynamic_stopwords:
            return
        self.dynamic_stopwords = stopwords
        self.processor = TextProcessor(
            config=self.config,
            stopwords=set(self.base_stopwords | self.dynamic_stopwords),
        )

    def dynamic_terms(self) -> tuple[str, ...]:
        return self.lexicon.terms


class GroupRelatedness:
    def __init__(
        self,
        *,
        group_id: str,
        data_path: str | Path,
        config: RelatednessConfig | None = None,
        runtime: RelatednessRuntime | None = None,
    ):
        self.group_id = str(group_id)
        self.data_path = Path(data_path)
        self.config = config or RelatednessConfig()
        self._runtime = runtime or get_relatedness_runtime(
            realtime_workers=self.config.realtime_workers,
            maintenance_workers=self.config.maintenance_workers,
            queue_limit=self.config.runtime_queue_limit,
        )
        self._state = _GroupComputationState(self.config)
        self._analytics = AnalyticsSnapshot.empty()
        self._topics = TopicSnapshot.empty()
        self._dynamic_new_words: tuple[DynamicLexiconTerm, ...] = ()
        self._operation_lock = asyncio.Lock()
        self._maintenance_task: asyncio.Task[None] | None = None
        self._enrichment_tasks: dict[str, asyncio.Task[None]] = {}
        self._maintenance_rerun = False
        self._messages_since_maintenance = 0
        self._last_maintenance = time.monotonic()
        self._audit = RelatednessAudit(
            data_path=self.data_path / "audit",
            group_id=self.group_id,
            bot_id="",
            runtime=self._runtime,
        )
        self._audit_members: dict[str, dict[str, object]] = {}
        self._started = False
        self._closed = False

    @property
    def state_path(self) -> Path:
        return self.data_path / "relatedness_state.json"

    def _register_audit_member(
        self,
        user_id: str,
        config: MemberCommunityConfig,
        *,
        role: str,
    ) -> None:
        member_id = str(user_id)
        self._audit_members[member_id] = {
            "role": role,
            "config": asdict(config),
        }
        if role == "bot":
            self._audit.set_bot_id(member_id)
        if self._started:
            self._emit_audit("member_registered", {
                "member_id": member_id,
                **self._audit_members[member_id],
            })

    def _audit_role(self, user_id: str) -> str:
        member = self._audit_members.get(str(user_id), {})
        return str(member.get("role", "member"))

    def _emit_audit(self, event: str, payload: dict[str, object]) -> None:
        self._audit.emit(event, payload)

    @property
    def message_count(self) -> int:
        return len(self._state.graph)

    @property
    def dynamic_terms(self) -> tuple[str, ...]:
        return self._state.dynamic_terms()

    @property
    def dynamic_new_words(self) -> tuple[DynamicLexiconTerm, ...]:
        return self._dynamic_new_words

    def _active_dynamic_new_words(
        self,
        terms: tuple[DynamicLexiconTerm, ...],
        *,
        now: float,
    ) -> tuple[DynamicLexiconTerm, ...]:
        return tuple(
            term
            for term in terms
            if term.term and term.expires_at > now
        )

    def _active_dynamic_term_texts(
        self,
        stored: StoredRelatednessState,
        *,
        now: float,
    ) -> tuple[str, ...]:
        active = self._active_dynamic_new_words(
            stored.dynamic_new_words,
            now=now,
        )
        return tuple(item.term for item in active)

    def _merge_dynamic_new_words(
        self,
        previous: tuple[DynamicLexiconTerm, ...],
        discovery: NewWordDiscoveryResult,
        *,
        now: float,
    ) -> tuple[DynamicLexiconTerm, ...]:
        expires_at = now + self.config.dynamic_new_word_ttl_seconds
        current = {
            item.term: item
            for item in self._active_dynamic_new_words(previous, now=now)
        }
        for candidate in discovery.promoted:
            previous_item = current.get(candidate.term)
            current[candidate.term] = DynamicLexiconTerm(
                term=candidate.term,
                score=candidate.score,
                frequency=candidate.frequency,
                user_count=candidate.user_count,
                first_seen=(
                    min(previous_item.first_seen, candidate.first_seen)
                    if previous_item is not None
                    else candidate.first_seen
                ),
                last_seen=max(
                    previous_item.last_seen if previous_item is not None else 0.0,
                    candidate.last_seen,
                ),
                promoted_at=now,
                expires_at=expires_at,
                hit_count=previous_item.hit_count if previous_item is not None else 0,
            )
        return tuple(
            sorted(
                current.values(),
                key=lambda item: (-item.expires_at, -item.score, item.term),
            )
        )

    @staticmethod
    def _new_word_documents_from_snapshot(
        snapshot,
    ) -> tuple[NewWordDocument, ...]:
        return tuple(
            NewWordDocument(
                msg_id=node.msg_id,
                user_id=node.user_id,
                timestamp=node.timestamp,
                text=node.text,
            )
            for node in snapshot.nodes
            if node.text.strip()
        )

    @classmethod
    def _new_word_documents_from_history(
        cls,
        history: tuple[GroupMsg | MessageInput, ...],
        group_id: str,
    ) -> tuple[NewWordDocument, ...]:
        documents: list[NewWordDocument] = []
        for message in history:
            adapted = (
                message
                if isinstance(message, MessageInput)
                else cls.adapt_message(message)
            )
            if adapted.group_id != group_id or not adapted.text.strip():
                continue
            documents.append(
                NewWordDocument(
                    msg_id=adapted.msg_id,
                    user_id=adapted.user_id,
                    timestamp=adapted.timestamp,
                    text=adapted.text,
                )
            )
        return tuple(documents)

    @staticmethod
    def _new_word_config(
        base: RelatednessConfig,
        config: NewWordConfig | None,
    ) -> NewWordConfig:
        return config or NewWordConfig(max_documents=base.new_word_document_limit)

    @staticmethod
    def adapt_message(message: BaseMsg) -> MessageInput:
        if not isinstance(message, GroupMsg):
            raise TypeError("GroupRelatedness only accepts GroupMsg")
        evidence = message.to_relative_cal()
        return MessageInput(
            msg_id=message.msg_id,
            group_id=message.group_id,
            user_id=message.user_id,
            timestamp=float(message.time),
            text=evidence.text,
            semantic_text=evidence.semantic_text,
            media_ids=frozenset(evidence.media_ids),
            reply_to=evidence.reply_to,
            mention_ids=evidence.mention_ids,
        )

    async def start(self, history: Iterable[GroupMsg]) -> None:
        async with self._operation_lock:
            if self._started:
                return
            stored = await self._runtime.run_maintenance(
                f"{self.group_id}:load",
                load_state,
                self.state_path,
            )
            self._analytics = replace(stored.analytics, dynamic_terms=())
            self._topics = stored.topics
            now = time.time()
            self._dynamic_new_words = self._active_dynamic_new_words(
                stored.dynamic_new_words,
                now=now,
            )
            dynamic_terms = self._active_dynamic_term_texts(stored, now=now)
            messages = tuple(
                self.adapt_message(message)
                for message in history
                if isinstance(message, GroupMsg)
                and message.group_id == self.group_id
            )

            def rebuild() -> None:
                self._state = _GroupComputationState(
                    self.config,
                    dynamic_terms,
                    stored.analytics.dynamic_stopwords,
                )
                for item in messages[-self.config.message_window:]:
                    self._state.ingest(item)

            await self._runtime.run_realtime(rebuild, enqueue_timeout=None)
            self._analytics = replace(self._analytics, source_version=0)
            self._topics = replace(
                self._topics,
                source_version=0,
                live_version=self._state.graph.version,
            )
            self._started = True
            await self._audit.start({
                "relatedness_config": asdict(self.config),
                "members": dict(self._audit_members),
                "dynamic_terms": self._state.dynamic_terms(),
                "dynamic_stopwords": self._analytics.dynamic_stopwords,
                "history_message_count": len(messages),
                "graph_message_count": len(self._state.graph),
            })

    async def ingest(self, message: GroupMsg | MessageInput) -> IngestResult:
        started = time.perf_counter()
        if self._closed:
            raise RuntimeError("GroupRelatedness is closed")
        if not self._started:
            await self.start(())
        adapted = (
            message
            if isinstance(message, MessageInput)
            else self.adapt_message(message)
        )
        if adapted.group_id != self.group_id:
            raise ValueError("message belongs to a different group")
        async with self._operation_lock:
            topics = self._topics
            try:
                result = await self._runtime.run_realtime(
                    self._state.ingest,
                    adapted,
                    topics,
                    enqueue_timeout=self.config.realtime_enqueue_timeout,
                )
            except RuntimeOverloadedError:
                result = await self._runtime.run_realtime(
                    self._state.ingest,
                    adapted,
                    topics,
                    enqueue_timeout=None,
                )
                result = replace(result, degraded=True)
            if result.features is not None and topics.topics:
                self._topics = apply_topic_compensation(
                    topics,
                    message_id=adapted.msg_id,
                    timestamp=adapted.timestamp,
                    graph_version=result.version,
                    vector_weights=result.features.term_weights,
                    affinities=result.topic_affinities,
                    config=self.config,
                )
        self._messages_since_maintenance += 1
        if (
            isinstance(message, GroupMsg)
            and message.to_relative_cal().pending_media
        ):
            self._schedule_enrichment(message)
        if self._maintenance_due():
            self._schedule_maintenance()
        self._emit_audit("message_ingested", {
            "message": adapted,
            "result": result,
            "input_type": type(message).__name__,
            "graph_message_count": len(self._state.graph),
            "total_ms": (time.perf_counter() - started) * 1_000.0,
        })
        return result

    def _schedule_enrichment(self, message: GroupMsg) -> None:
        if self._closed or self.config.media_enrichment_timeout <= 0:
            return
        current = self._enrichment_tasks.get(message.msg_id)
        if current is not None and not current.done():
            return
        pending_images = tuple(
            item
            for item in message.content
            if isinstance(item, ImgMsg) and not item.description
        )
        if not pending_images:
            return
        task = asyncio.create_task(
            self._wait_for_media_enrichment(message, pending_images)
        )
        self._enrichment_tasks[message.msg_id] = task

        def remove_task(completed: asyncio.Task[None]) -> None:
            if self._enrichment_tasks.get(message.msg_id) is completed:
                self._enrichment_tasks.pop(message.msg_id, None)

        task.add_done_callback(remove_task)

    async def _wait_for_media_enrichment(
        self,
        message: GroupMsg,
        images: tuple[ImgMsg, ...],
    ) -> None:
        try:
            async with asyncio.timeout(self.config.media_enrichment_timeout):
                await asyncio.gather(*(image.wait() for image in images))
        except TimeoutError:
            return
        if self._closed:
            return
        adapted = self.adapt_message(message)
        async with self._operation_lock:
            topics = self._topics
            changed, vector_weights, affinities, version = (
                await self._runtime.run_realtime(
                    self._state.enrich,
                    adapted,
                    topics,
                    enqueue_timeout=None,
                )
            )
            if changed and topics.topics:
                self._topics = apply_topic_compensation(
                    topics,
                    message_id=adapted.msg_id,
                    timestamp=adapted.timestamp,
                    graph_version=version,
                    vector_weights=vector_weights,
                    affinities=affinities,
                    config=self.config,
                )

    def _maintenance_due(self) -> bool:
        return (
            self._messages_since_maintenance
            >= self.config.maintenance_message_interval
            or time.monotonic() - self._last_maintenance
            >= self.config.maintenance_seconds
        )

    def _schedule_maintenance(self) -> None:
        if self._closed:
            return
        if self._maintenance_task is not None and not self._maintenance_task.done():
            self._maintenance_rerun = True
            return
        self._maintenance_task = asyncio.create_task(
            self._run_scheduled_maintenance()
        )

    async def _run_scheduled_maintenance(self) -> None:
        while not self._closed:
            self._maintenance_rerun = False
            await self.refresh_topics()
            if not self._maintenance_rerun:
                return

    async def get_related(
        self,
        message_id: str,
        *,
        limit: int = 10,
        explain: bool = False,
    ) -> tuple[RelatedMatch, ...]:
        if limit <= 0:
            return ()
        async with self._operation_lock:
            return await self._runtime.run_realtime(
                self._state.related,
                message_id,
                limit,
                explain,
                enqueue_timeout=None,
            )

    async def get_score(
        self,
        source_id: str,
        target_id: str,
    ) -> RelatednessScore | None:
        async with self._operation_lock:
            return await self._runtime.run_realtime(
                self._state.score,
                source_id,
                target_id,
                enqueue_timeout=None,
            )

    async def get_anchor_evidence(
        self,
        source_id: str,
        anchor_ids: tuple[str, ...],
    ) -> MemberEvidenceSnapshot | None:
        async with self._operation_lock:
            return await self._runtime.run_realtime(
                self._state.anchor_evidence,
                source_id,
                anchor_ids,
                enqueue_timeout=None,
            )

    async def get_continuity_snapshot(
        self,
        source_id: str,
        *,
        window_seconds: float,
        message_limit: int,
    ):
        async with self._operation_lock:
            return await self._runtime.run_realtime(
                self._state.continuity_snapshot,
                source_id,
                window_seconds,
                message_limit,
                enqueue_timeout=None,
            )

    async def get_continuity_evidence(
        self,
        source_id: str,
        target_ids: tuple[str, ...] = (),
    ):
        async with self._operation_lock:
            return await self._runtime.run_realtime(
                self._state.continuity_evidence_snapshot,
                source_id,
                target_ids,
                enqueue_timeout=None,
            )

    async def get_message_references(
        self,
        message_ids: tuple[str, ...] = (),
        *,
        user_id: str | None = None,
        limit: int = 32,
    ) -> tuple[MessageReference, ...]:
        if limit <= 0:
            return ()
        async with self._operation_lock:
            return await self._runtime.run_realtime(
                self._state.message_references,
                message_ids,
                user_id,
                limit,
                enqueue_timeout=None,
            )

    def get_analytics(self) -> AnalyticsSnapshot:
        return self._analytics

    def get_topics(self) -> TopicSnapshot:
        return self._topics

    async def get_topic_affinities(
        self,
        message_ids: tuple[str, ...],
    ) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        if not message_ids:
            return ()
        topics = self._topics
        async with self._operation_lock:
            return await self._runtime.run_realtime(
                self._state.topic_affinities,
                message_ids,
                topics,
                enqueue_timeout=None,
            )

    async def refresh_topics(self) -> None:
        if self._closed:
            return
        async with self._operation_lock:
            graph_snapshot = await self._runtime.run_realtime(
                self._state.snapshot,
                enqueue_timeout=None,
            )
            previous = self._topics

        def calculate_topics() -> TopicSnapshot:
            return build_topics(
                graph_snapshot,
                self.config,
                previous=previous,
            )

        topics = await self._runtime.run_maintenance(
            f"{self.group_id}:topics",
            calculate_topics,
        )
        async with self._operation_lock:
            if topics.source_version < self._topics.source_version:
                return
            current = self._topics
            pending = tuple(
                item
                for item in current.compensations
                if item.graph_version > topics.source_version
            )
            for item in pending:
                topics = apply_topic_compensation(
                    topics,
                    message_id=item.message_id,
                    timestamp=item.timestamp,
                    graph_version=item.graph_version,
                    vector_weights=item.vector_weights,
                    config=self.config,
                )
            self._topics = topics
            self._messages_since_maintenance = 0
            self._last_maintenance = time.monotonic()

    async def refresh_analytics(
        self,
        kind: AnalyticsKind = AnalyticsKind.ALL,
    ) -> None:
        if self._closed:
            return
        if kind is AnalyticsKind.TOPICS:
            await self.refresh_topics()
            return
        if kind is AnalyticsKind.NEW_WORDS:
            await self.refresh_new_words()
            return
        if kind is AnalyticsKind.ALL:
            await self.refresh_topics()
        async with self._operation_lock:
            graph_snapshot = await self._runtime.run_realtime(
                self._state.snapshot,
                enqueue_timeout=None,
            )
            topics = self._topics

        def calculate():
            return build_analytics(
                graph_snapshot,
                topics,
                self.config,
                kind,
            )

        analytics = await self._runtime.run_maintenance(
            f"{self.group_id}:{kind.value}",
            calculate,
        )
        if analytics.source_version < self._analytics.source_version:
            return
        async with self._operation_lock:
            await self._runtime.run_realtime(
                self._state.set_dynamic_stopwords,
                analytics.dynamic_stopwords,
                enqueue_timeout=None,
            )
        self._analytics = analytics

    async def refresh_new_words(
        self,
        history: Iterable[GroupMsg | MessageInput] | None = None,
        *,
        config: NewWordConfig | None = None,
    ) -> NewWordDiscoveryResult:
        if self._closed:
            raise RuntimeError("GroupRelatedness is closed")
        if not self._started:
            await self.start(())
        effective_config = self._new_word_config(self.config, config)
        if history is None:
            async with self._operation_lock:
                graph_snapshot = await self._runtime.run_realtime(
                    self._state.snapshot,
                    enqueue_timeout=None,
                )
            documents = self._new_word_documents_from_snapshot(graph_snapshot)
        else:
            history_items = tuple(history)
            documents = await self._runtime.run_maintenance(
                f"{self.group_id}:new_word_documents",
                self._new_word_documents_from_history,
                history_items,
                self.group_id,
            )

        def calculate() -> NewWordDiscoveryResult:
            return discover_new_words(documents, effective_config)

        discovery = await self._runtime.run_maintenance(
            f"{self.group_id}:new_words",
            calculate,
        )
        now = time.time()
        dynamic_new_words = self._merge_dynamic_new_words(
            self._dynamic_new_words,
            discovery,
            now=now,
        )
        active_terms = tuple(item.term for item in dynamic_new_words)
        async with self._operation_lock:
            self._dynamic_new_words = dynamic_new_words
            await self._runtime.run_realtime(
                self._state.set_dynamic_terms,
                active_terms,
                enqueue_timeout=None,
            )
        self._emit_audit("new_words_refreshed", {
            "documents": discovery.documents,
            "candidate_count": discovery.candidate_count,
            "promoted": len(discovery.promoted),
            "candidate": len(discovery.candidates),
            "known": len(discovery.known),
            "active_terms": len(active_terms),
            "elapsed_ms": discovery.elapsed_ms,
        })
        return discovery

    async def save(self) -> None:
        async with self._operation_lock:
            now = time.time()
            self._dynamic_new_words = self._active_dynamic_new_words(
                self._dynamic_new_words,
                now=now,
            )
            terms = await self._runtime.run_realtime(
                self._state.dynamic_terms,
                enqueue_timeout=None,
            )
            analytics = self._analytics
            topics = self._topics
        stored = StoredRelatednessState(
            dynamic_terms=terms,
            dynamic_new_words=self._dynamic_new_words,
            analytics=analytics,
            topics=topics,
        )
        await self._runtime.run_maintenance(
            f"{self.group_id}:save",
            save_state,
            self.state_path,
            stored,
        )

    async def close(self) -> None:
        if self._closed:
            return
        task = self._maintenance_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        enrichment_tasks = tuple(self._enrichment_tasks.values())
        for enrichment_task in enrichment_tasks:
            enrichment_task.cancel()
        if enrichment_tasks:
            await asyncio.gather(*enrichment_tasks, return_exceptions=True)
        await self.save()
        self._closed = True
        await self._audit.close()
