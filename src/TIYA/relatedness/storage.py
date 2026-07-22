from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import (
    AnalyticsSnapshot,
    DialogueChain,
    DialogueChainNode,
    DynamicLexiconTerm,
    DynamicTerm,
    DynamicTermStatus,
    HotSentence,
    Hotword,
    TopicActivity,
    TopicCompensation,
    TopicInterest,
    StoredRelatednessState,
    StoredMemberCommunityState,
    TopicInfo,
    TopicSnapshot,
    TopicTransition,
)


STATE_VERSION = 3
MEMBER_STATE_VERSION = 6


def _topic_to_dict(topic: TopicInfo) -> dict[str, Any]:
    return {
        "topic_id": topic.topic_id,
        "members": [list(item) for item in topic.members],
        "representative_ids": list(topic.representative_ids),
        "last_active": topic.last_active,
        "centroid": [list(item) for item in topic.centroid],
        "centroid_norm": topic.centroid_norm,
        "cohesion": topic.cohesion,
        "user_diversity": topic.user_diversity,
        "repeat_score": topic.repeat_score,
        "time_concentration": topic.time_concentration,
    }


def _topic_from_dict(item: dict[str, Any]) -> TopicInfo:
    return TopicInfo(
        topic_id=str(item["topic_id"]),
        members=tuple(
            (str(msg_id), float(score))
            for msg_id, score in item.get("members", ())
        ),
        representative_ids=tuple(
            str(value) for value in item.get("representative_ids", ())
        ),
        last_active=float(item.get("last_active", 0.0)),
        centroid=tuple(
            (str(term), float(value))
            for term, value in item.get("centroid", ())
        ),
        centroid_norm=float(item.get("centroid_norm", 0.0)),
        cohesion=float(item.get("cohesion", 0.0)),
        user_diversity=float(item.get("user_diversity", 0.0)),
        repeat_score=float(item.get("repeat_score", 0.0)),
        time_concentration=float(item.get("time_concentration", 0.0)),
    )


def _topic_snapshot_to_dict(snapshot: TopicSnapshot) -> dict[str, Any]:
    return {
        "source_version": snapshot.source_version,
        "live_version": snapshot.live_version,
        "topics": [_topic_to_dict(topic) for topic in snapshot.topics],
        "memberships": [
            [message_id, [list(item) for item in memberships]]
            for message_id, memberships in snapshot.memberships
        ],
        "transitions": [
            {
                "source_topic_id": item.source_topic_id,
                "target_topic_id": item.target_topic_id,
                "weight": item.weight,
                "shared_members": item.shared_members,
            }
            for item in snapshot.transitions
        ],
        "compensations": [
            {
                "message_id": item.message_id,
                "timestamp": item.timestamp,
                "graph_version": item.graph_version,
                "vector_weights": [list(value) for value in item.vector_weights],
                "affinities": [list(value) for value in item.affinities],
                "memberships": [list(value) for value in item.memberships],
            }
            for item in snapshot.compensations
        ],
        "activity": [
            {
                "topic_id": item.topic_id,
                "score": item.score,
                "inherited_score": item.inherited_score,
                "message_count": item.message_count,
                "last_active": item.last_active,
                "updated_at": item.updated_at,
            }
            for item in snapshot.activity
        ],
    }


def _topic_snapshot_from_dict(data: dict[str, Any]) -> TopicSnapshot:
    source_version = int(data.get("source_version", 0))
    return TopicSnapshot(
        source_version=source_version,
        live_version=int(data.get("live_version", source_version)),
        topics=tuple(
            _topic_from_dict(item) for item in data.get("topics", ())
        ),
        memberships=tuple(
            (
                str(message_id),
                tuple(
                    (str(topic_id), float(score))
                    for topic_id, score in memberships
                ),
            )
            for message_id, memberships in data.get("memberships", ())
        ),
        transitions=tuple(
            TopicTransition(
                source_topic_id=str(item["source_topic_id"]),
                target_topic_id=str(item["target_topic_id"]),
                weight=float(item["weight"]),
                shared_members=int(item["shared_members"]),
            )
            for item in data.get("transitions", ())
        ),
        compensations=tuple(
            TopicCompensation(
                message_id=str(item["message_id"]),
                timestamp=float(item["timestamp"]),
                graph_version=int(item["graph_version"]),
                vector_weights=tuple(
                    (str(term), float(value))
                    for term, value in item.get("vector_weights", ())
                ),
                affinities=tuple(
                    (str(topic_id), float(score))
                    for topic_id, score in item.get("affinities", ())
                ),
                memberships=tuple(
                    (str(topic_id), float(score))
                    for topic_id, score in item.get("memberships", ())
                ),
            )
            for item in data.get("compensations", ())
        ),
        activity=tuple(
            TopicActivity(
                topic_id=str(item["topic_id"]),
                score=float(item["score"]),
                inherited_score=float(item.get("inherited_score", 0.0)),
                message_count=int(item.get("message_count", 0)),
                last_active=float(item.get("last_active", 0.0)),
                updated_at=float(item.get("updated_at", 0.0)),
            )
            for item in data.get("activity", ())
        ),
    )


def _analytics_to_dict(snapshot: AnalyticsSnapshot) -> dict[str, Any]:
    return {
        "source_version": snapshot.source_version,
        "published_at": snapshot.published_at,
        "topics": [_topic_to_dict(topic) for topic in snapshot.topics],
        "memberships": [
            [msg_id, [list(item) for item in memberships]]
            for msg_id, memberships in snapshot.memberships
        ],
        "hotwords": [
            {
                "term": item.term,
                "score": item.score,
                "document_count": item.document_count,
                "user_count": item.user_count,
                "textrank": item.textrank,
                "specificity": item.specificity,
                "frequency": item.frequency,
                "user_diversity": item.user_diversity,
                "topic_specificity": item.topic_specificity,
                "sentence_centrality": item.sentence_centrality,
                "source_quality": item.source_quality,
            }
            for item in snapshot.hotwords
        ],
        "hot_sentences": [
            {
                "message_id": item.message_id,
                "text": item.text,
                "score": item.score,
                "pagerank": item.pagerank,
                "weighted_degree": item.weighted_degree,
                "topic_confidence": item.topic_confidence,
                "user_diversity": item.user_diversity,
                "recency": item.recency,
            }
            for item in snapshot.hot_sentences
        ],
        "dynamic_terms": [
            {
                "term": item.term,
                "score": item.score,
                "frequency": item.frequency,
                "user_count": item.user_count,
                "status": item.status.value,
            }
            for item in snapshot.dynamic_terms
        ],
        "dynamic_stopwords": list(snapshot.dynamic_stopwords),
    }


def _analytics_from_dict(data: dict[str, Any]) -> AnalyticsSnapshot:
    topics = tuple(_topic_from_dict(item) for item in data.get("topics", ()))
    memberships = tuple(
        (
            str(msg_id),
            tuple((str(topic_id), float(score)) for topic_id, score in values),
        )
        for msg_id, values in data.get("memberships", ())
    )
    hotwords = tuple(
        Hotword(
            term=str(item["term"]),
            score=float(item["score"]),
            document_count=int(item["document_count"]),
            user_count=int(item["user_count"]),
            textrank=float(item.get("textrank", 0.0)),
            specificity=float(item.get("specificity", 0.0)),
            frequency=float(item.get("frequency", 0.0)),
            user_diversity=float(item.get("user_diversity", 0.0)),
            topic_specificity=float(item.get("topic_specificity", 0.0)),
            sentence_centrality=float(item.get("sentence_centrality", 0.0)),
            source_quality=float(item.get("source_quality", 0.0)),
        )
        for item in data.get("hotwords", ())
    )
    hot_sentences = tuple(
        HotSentence(
            message_id=str(item["message_id"]),
            text=str(item["text"]),
            score=float(item["score"]),
            pagerank=float(item.get("pagerank", 0.0)),
            weighted_degree=float(item.get("weighted_degree", 0.0)),
            topic_confidence=float(item.get("topic_confidence", 0.0)),
            user_diversity=float(item.get("user_diversity", 0.0)),
            recency=float(item.get("recency", 0.0)),
        )
        for item in data.get("hot_sentences", ())
    )
    dynamic_terms = tuple(
        DynamicTerm(
            term=str(item["term"]),
            score=float(item["score"]),
            frequency=int(item["frequency"]),
            user_count=int(item["user_count"]),
            status=DynamicTermStatus(item.get("status", "promoted")),
        )
        for item in data.get("dynamic_terms", ())
    )
    return AnalyticsSnapshot(
        source_version=int(data.get("source_version", 0)),
        published_at=float(data.get("published_at", 0.0)),
        topics=topics,
        memberships=memberships,
        hotwords=hotwords,
        hot_sentences=hot_sentences,
        dynamic_terms=dynamic_terms,
        dynamic_stopwords=tuple(
            str(item) for item in data.get("dynamic_stopwords", ())
        ),
    )


def _dynamic_lexicon_term_to_dict(item: DynamicLexiconTerm) -> dict[str, Any]:
    return {
        "term": item.term,
        "score": item.score,
        "frequency": item.frequency,
        "user_count": item.user_count,
        "first_seen": item.first_seen,
        "last_seen": item.last_seen,
        "promoted_at": item.promoted_at,
        "expires_at": item.expires_at,
        "hit_count": item.hit_count,
    }


def _dynamic_lexicon_term_from_dict(data: dict[str, Any]) -> DynamicLexiconTerm:
    return DynamicLexiconTerm(
        term=str(data["term"]),
        score=float(data.get("score", 0.0)),
        frequency=int(data.get("frequency", 0)),
        user_count=int(data.get("user_count", 0)),
        first_seen=float(data.get("first_seen", 0.0)),
        last_seen=float(data.get("last_seen", 0.0)),
        promoted_at=float(data.get("promoted_at", 0.0)),
        expires_at=float(data.get("expires_at", 0.0)),
        hit_count=int(data.get("hit_count", 0)),
    )


def load_state(path: Path) -> StoredRelatednessState:
    if not path.is_file():
        return StoredRelatednessState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        version = int(data.get("version", 0))
        if version not in (1, 2, STATE_VERSION):
            return StoredRelatednessState()
        analytics = _analytics_from_dict(data.get("analytics", {}))
        topics = (
            _topic_snapshot_from_dict(data.get("topics", {}))
            if version >= 2
            else TopicSnapshot(
                source_version=analytics.source_version,
                live_version=analytics.source_version,
                topics=analytics.topics,
                memberships=analytics.memberships,
            )
        )
        return StoredRelatednessState(
            dynamic_terms=tuple(
                str(term) for term in data.get("dynamic_terms", ())
            ),
            dynamic_new_words=tuple(
                _dynamic_lexicon_term_from_dict(item)
                for item in data.get("dynamic_new_words", ())
                if isinstance(item, dict) and item.get("term")
            ),
            analytics=analytics,
            topics=topics,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return StoredRelatednessState()


def save_state(path: Path, state: StoredRelatednessState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STATE_VERSION,
        "dynamic_terms": list(state.dynamic_terms),
        "dynamic_new_words": [
            _dynamic_lexicon_term_to_dict(item)
            for item in state.dynamic_new_words
        ],
        "analytics": _analytics_to_dict(state.analytics),
        "topics": _topic_snapshot_to_dict(state.topics),
    }
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def load_member_state(path: Path) -> StoredMemberCommunityState:
    if not path.is_file():
        return StoredMemberCommunityState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        version = int(data.get("version", 0))
        if version not in (5, MEMBER_STATE_VERSION):
            return StoredMemberCommunityState()
        topic_interests = tuple(
            TopicInterest(
                topic_id=str(item["topic_id"]),
                weight=float(item["weight"]),
                updated_at=float(item["updated_at"]),
                source_version=int(item.get("source_version", 0)),
            )
            for item in data.get("topic_interests", ())
        )
        dialogue_chains = ()
        if version >= 6:
            parsed_chains = tuple(
                DialogueChain(
                    chain_id=str(item["chain_id"]),
                    active=bool(item.get("active", False)),
                    nodes=tuple(
                        DialogueChainNode(
                            message_id=str(node["message_id"]),
                            weight=min(1.0, max(0.0, float(node["weight"]))),
                            timestamp=float(node["timestamp"]),
                        )
                        for node in item.get("nodes", ())
                    ),
                    progress=max(0, int(item.get("progress", 0))),
                    low_streak=max(0, int(item.get("low_streak", 0))),
                )
                for item in data.get("dialogue_chains", ())
            )
            active_seen = False
            normalized: list[DialogueChain] = []
            for chain in reversed(parsed_chains):
                active = chain.active and not active_seen
                active_seen = active_seen or active
                normalized.append(DialogueChain(
                    chain_id=chain.chain_id,
                    active=active,
                    nodes=chain.nodes,
                    progress=chain.progress,
                    low_streak=chain.low_streak,
                ))
            dialogue_chains = tuple(reversed(normalized))
        return StoredMemberCommunityState(
            group_id=str(data.get("group_id", "")),
            user_id=str(data.get("user_id", "")),
            saved_at=float(data.get("saved_at", 0.0)),
            topic_interests=topic_interests,
            dialogue_chains=dialogue_chains,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return StoredMemberCommunityState()


def save_member_state(path: Path, state: StoredMemberCommunityState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MEMBER_STATE_VERSION,
        "group_id": state.group_id,
        "user_id": state.user_id,
        "saved_at": state.saved_at,
        "topic_interests": [
            {
                "topic_id": item.topic_id,
                "weight": item.weight,
                "updated_at": item.updated_at,
                "source_version": item.source_version,
            }
            for item in state.topic_interests
        ],
        "dialogue_chains": [
            {
                "chain_id": chain.chain_id,
                "active": chain.active,
                "progress": chain.progress,
                "low_streak": chain.low_streak,
                "nodes": [
                    {
                        "message_id": node.message_id,
                        "weight": node.weight,
                        "timestamp": node.timestamp,
                    }
                    for node in chain.nodes
                ],
            }
            for chain in state.dialogue_chains
        ],
    }
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
