from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum


class AnalyticsKind(StrEnum):
    ALL = "all"
    TOPICS = "topics"
    HOTWORDS = "hotwords"
    NEW_WORDS = "new_words"


class DynamicTermStatus(StrEnum):
    OBSERVED = "observed"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    STALE = "stale"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class RelatednessConfig:
    message_window: int = 1000
    text_candidate_limit: int = 64
    text_edge_limit: int = 32
    context_edge_limit: int = 8
    mention_message_limit: int = 3
    lexical_weight: float = 0.65
    subword_weight: float = 0.35
    semantic_text_weight: float = 0.35
    semantic_text_max_chars: int = 200
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    reply_weight: float = 0.8
    reply_direct_factor: float = 0.35
    mention_weight: float = 0.2
    text_weight: float = 0.55
    context_weight: float = 0.1
    same_user_weight: float = 0.02
    related_min_score: float = 0.25
    enable_relation_propagation: bool = False
    propagation_restart: float = 0.35
    propagation_decay: float = 0.65
    propagation_score_weight: float = 1.4
    propagation_depth: int = 3
    propagation_node_limit: int = 128
    propagation_min_energy: float = 0.005
    topic_membership_limit: int = 3
    topic_min_edge: float = 0.15
    topic_resolution: float = 0.7
    topic_louvain_restarts: int = 4
    topic_overlap_ratio: float = 0.45
    topic_expire_seconds: float = 86400.0
    topic_inheritance_min_weight: float = 0.05
    topic_dynamic_min_similarity: float = 0.15
    topic_dynamic_activity_decay_seconds: float = 900.0
    topic_dynamic_inheritance_factor: float = 0.30
    topic_compensation_limit: int = 100
    hotword_limit: int = 30
    hot_sentence_limit: int = 10
    hotword_min_documents: int = 2
    hotword_min_chars: int = 2
    hotword_min_ascii_letters: int = 3
    hotword_max_chars: int = 24
    hotword_allow_numeric: bool = False
    hotword_allow_semantic_only_ascii: bool = False
    hotword_verb_score_factor: float = 0.6
    hotword_textrank_rounds: int = 20
    hot_sentence_min_terms: int = 3
    hot_sentence_pagerank_rounds: int = 20
    dynamic_stopword_min_documents: int = 5
    dynamic_stopword_doc_ratio: float = 0.08
    dynamic_stopword_user_ratio: float = 0.25
    dynamic_stopword_max_specificity: float = 0.80
    dynamic_stopword_time_uniformity: float = 0.15
    dynamic_stopword_time_max_specificity: float = 0.80
    dynamic_term_limit: int = 500
    new_word_document_limit: int = 5000
    new_word_min_frequency: int = 8
    new_word_min_users: int = 3
    dynamic_new_word_ttl_seconds: float = 1_209_600.0
    maintenance_message_interval: int = 50
    maintenance_seconds: float = 900.0
    realtime_workers: int = 1
    maintenance_workers: int = 1
    runtime_queue_limit: int = 128
    realtime_enqueue_timeout: float = 0.25
    media_enrichment_timeout: float = 180.0

    def __post_init__(self) -> None:
        positive_ints = (
            self.message_window,
            self.text_candidate_limit,
            self.text_edge_limit,
            self.context_edge_limit,
            self.propagation_depth,
            self.propagation_node_limit,
            self.topic_membership_limit,
            self.topic_compensation_limit,
            self.hotword_limit,
            self.hot_sentence_limit,
            self.hotword_min_documents,
            self.hotword_min_chars,
            self.hotword_min_ascii_letters,
            self.hotword_max_chars,
            self.hotword_textrank_rounds,
            self.hot_sentence_min_terms,
            self.hot_sentence_pagerank_rounds,
            self.dynamic_stopword_min_documents,
            self.dynamic_term_limit,
            self.new_word_document_limit,
            self.semantic_text_max_chars,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("relatedness limits must be positive")
        if self.dynamic_new_word_ttl_seconds <= 0:
            raise ValueError("dynamic new word ttl must be positive")
        if (
            self.hotword_min_chars > self.hotword_max_chars
            or self.hotword_min_ascii_letters > self.hotword_max_chars
        ):
            raise ValueError("hotword shape limits must be consistent")
        if any(
            value < 0
            for value in (
                self.lexical_weight,
                self.subword_weight,
                self.semantic_text_weight,
            )
        ):
            raise ValueError("text feature weights cannot be negative")
        if self.lexical_weight + self.subword_weight <= 0:
            raise ValueError("at least one text feature block must be enabled")
        if self.topic_resolution <= 0:
            raise ValueError("topic resolution must be positive")
        if self.topic_louvain_restarts <= 0:
            raise ValueError("topic_louvain_restarts must be positive")
        if not 0.0 <= self.topic_overlap_ratio <= 1.0:
            raise ValueError("topic overlap ratio must be between zero and one")
        if self.topic_min_edge < 0:
            raise ValueError("topic edge threshold cannot be negative")
        if not 0.0 <= self.topic_inheritance_min_weight <= 1.0:
            raise ValueError("topic inheritance threshold must be between zero and one")
        if not 0.0 <= self.topic_dynamic_min_similarity <= 1.0:
            raise ValueError("topic dynamic threshold must be between zero and one")
        if self.topic_dynamic_activity_decay_seconds <= 0.0:
            raise ValueError("topic dynamic decay must be positive")
        if not 0.0 <= self.topic_dynamic_inheritance_factor <= 1.0:
            raise ValueError("topic dynamic inheritance must be between zero and one")
        if not 0.0 <= self.reply_direct_factor <= 1.0:
            raise ValueError("reply direct factor must be between zero and one")
        if self.propagation_score_weight < 0:
            raise ValueError("propagation score weight cannot be negative")
        if not 0.0 <= self.hotword_verb_score_factor <= 1.0:
            raise ValueError("hotword verb score factor must be between zero and one")
        if any(
            not 0.0 <= value <= 1.0
            for value in (
                self.dynamic_stopword_doc_ratio,
                self.dynamic_stopword_user_ratio,
                self.dynamic_stopword_max_specificity,
                self.dynamic_stopword_time_uniformity,
                self.dynamic_stopword_time_max_specificity,
            )
        ):
            raise ValueError("dynamic stopword ratios must be between zero and one")


@dataclass(frozen=True, slots=True)
class MemberCommunityConfig:
    interest_message_window: int = 64
    interest_topic_limit: int = 5
    interest_min_topic_size: int = 5
    interest_time_window_seconds: float = 604_800.0
    interest_time_decay_seconds: float = 86_400.0
    interest_min_topic_similarity: float = 0.05
    interest_quality_base_weight: float = 0.40
    interest_cohesion_weight: float = 0.20
    interest_user_diversity_weight: float = 0.15
    interest_repeat_weight: float = 0.15
    interest_time_concentration_weight: float = 0.10
    continuity_chain_capacity: int = 20
    continuity_propagation_decay: float = 0.95
    continuity_min_link_strength: float = 0.15
    continuity_admission_threshold: float = 0.15
    continuity_time_soft_seconds: float = 300.0
    continuity_time_window_seconds: float = 900.0
    continuity_time_soft_floor: float = 0.80

    @property
    def continuity_static_progress_limit(self) -> int:
        return self.continuity_chain_capacity * 2

    @property
    def continuity_low_streak_limit(self) -> int:
        return max(1, math.ceil(self.continuity_chain_capacity * 0.5))

    def __post_init__(self) -> None:
        limits = (
            self.interest_message_window,
            self.interest_min_topic_size,
            self.continuity_chain_capacity,
        )
        if any(value <= 0 for value in limits):
            raise ValueError("member community limits must be positive")
        if self.interest_topic_limit <= 0:
            raise ValueError("interest topic limit must be positive")
        positive_seconds = (
            self.interest_time_window_seconds,
            self.interest_time_decay_seconds,
            self.continuity_time_soft_seconds,
            self.continuity_time_window_seconds,
        )
        if any(value <= 0 for value in positive_seconds):
            raise ValueError("member community durations must be positive")
        ratios = (
            self.interest_min_topic_similarity,
            self.interest_quality_base_weight,
            self.interest_cohesion_weight,
            self.interest_user_diversity_weight,
            self.interest_repeat_weight,
            self.interest_time_concentration_weight,
            self.continuity_propagation_decay,
            self.continuity_min_link_strength,
            self.continuity_admission_threshold,
            self.continuity_time_soft_floor,
        )
        if any(not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("member community weights must be between zero and one")
        if self.continuity_time_soft_seconds >= self.continuity_time_window_seconds:
            raise ValueError("continuity soft time must precede the hard window")
        quality_total = (
            self.interest_quality_base_weight
            + self.interest_cohesion_weight
            + self.interest_user_diversity_weight
            + self.interest_repeat_weight
            + self.interest_time_concentration_weight
        )
        if not math.isclose(quality_total, 1.0):
            raise ValueError("member interest quality weights must sum to one")


@dataclass(frozen=True, slots=True)
class MemberAffinityScore:
    interest: float
    continuity: float


@dataclass(frozen=True, slots=True)
class DialogueChainNode:
    message_id: str
    weight: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class DialogueChain:
    chain_id: str
    active: bool
    nodes: tuple[DialogueChainNode, ...] = ()
    progress: int = 0
    low_streak: int = 0


@dataclass(frozen=True, slots=True)
class ContinuityCandidateAudit:
    message_id: str
    relation_source: str
    base_similarity: float
    parent_chain_weight: float
    time_penalty: float
    candidate: float


@dataclass(frozen=True, slots=True)
class ContinuityChainAudit:
    chain_id: str
    active: bool
    contribution: float
    candidates: tuple[ContinuityCandidateAudit, ...] = ()


@dataclass(frozen=True, slots=True)
class LocalContinuityScore:
    source_id: str
    score: float = 0.0
    seed_id: str | None = None
    path_ids: tuple[str, ...] = ()
    edge_strengths: tuple[float, ...] = ()
    direct_strength: float = 0.0
    path_strength: float = 0.0
    elapsed_seconds: float = 0.0
    temporal: float = 0.0
    winning_chain_id: str | None = None
    chains: tuple[ContinuityChainAudit, ...] = ()
    admitted: bool = False
    removed_chain_ids: tuple[str, ...] = ()

    @property
    def propagated(self) -> bool:
        return len(self.path_ids) > 2


@dataclass(frozen=True, slots=True)
class TopicInterestAudit:
    topic_id: str
    profile_weight: float
    source_similarity: float
    quality: float
    contribution: float


@dataclass(frozen=True, slots=True)
class MemberScoreAudit:
    source_id: str
    source_user_id: str
    source_timestamp: float
    source_sequence: int
    score: MemberAffinityScore
    cache_hit: bool = False
    evidence_missing: bool = False
    interest_topic_count: int = 0
    interest_evidence: tuple[TopicInterestAudit, ...] = ()
    local_continuity: LocalContinuityScore | None = None


@dataclass(frozen=True, slots=True)
class MessageInput:
    msg_id: str
    group_id: str
    user_id: str
    timestamp: float
    text: str
    semantic_text: str = ""
    media_ids: frozenset[str] = field(default_factory=frozenset)
    reply_to: str | None = None
    mention_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.msg_id or not self.group_id or not self.user_id:
            raise ValueError("message, group and user ids are required")


@dataclass(frozen=True, slots=True)
class TextFeatures:
    normalized_text: str
    lexical_terms: tuple[str, ...]
    lexical_weights: tuple[float, ...]
    subword_terms: tuple[str, ...]
    subword_weights: tuple[float, ...]
    lexicon_version: int = 0
    primary_lexical_terms: tuple[str, ...] = ()
    semantic_lexical_terms: tuple[str, ...] = ()
    lexical_pos: tuple[tuple[str, str], ...] = ()

    @property
    def term_weights(self) -> tuple[tuple[str, float], ...]:
        lexical = tuple(
            (f"WORD:{term}", weight)
            for term, weight in zip(self.lexical_terms, self.lexical_weights, strict=True)
        )
        subword = tuple(
            (term, weight)
            for term, weight in zip(self.subword_terms, self.subword_weights, strict=True)
        )
        return lexical + subword


@dataclass(frozen=True, slots=True)
class RelatednessScore:
    text: float = 0.0
    reply: float = 0.0
    mention: float = 0.0
    context: float = 0.0
    same_user: float = 0.0
    time: float = 0.0
    propagation: float = 0.0
    final: float = 0.0


@dataclass(frozen=True, slots=True)
class MessageReference:
    message_id: str
    user_id: str
    timestamp: float
    sequence: int
    reply_to: str | None = None
    mention_ids: frozenset[str] = field(default_factory=frozenset)
    content_key: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnchorEvidence:
    message_id: str
    user_id: str
    timestamp: float
    sequence: int
    message_gap: int
    relation: RelatednessScore


@dataclass(frozen=True, slots=True)
class MemberEvidenceSnapshot:
    source_id: str
    source_user_id: str
    source_timestamp: float
    source_sequence: int
    anchors: tuple[AnchorEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class RelatedMatch:
    message_id: str
    score: float
    details: RelatednessScore | None = None


@dataclass(frozen=True, slots=True)
class IngestResult:
    message_id: str
    related: tuple[RelatedMatch, ...]
    version: int
    degraded: bool = False
    features: TextFeatures | None = None
    text_candidates: tuple[tuple[str, float], ...] = ()
    evicted_message_ids: tuple[str, ...] = ()
    topic_affinities: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class MessageSnapshot:
    msg_id: str
    user_id: str
    timestamp: float
    text: str
    lexical_terms: tuple[str, ...] = ()
    semantic_text: str = ""
    media_ids: frozenset[str] = field(default_factory=frozenset)
    primary_lexical_terms: tuple[str, ...] = ()
    semantic_lexical_terms: tuple[str, ...] = ()
    lexical_pos: tuple[tuple[str, str], ...] = ()
    vector_weights: tuple[tuple[str, float], ...] = ()
    sequence: int = -1
    reply_to: str | None = None
    mention_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class GraphEdgeSnapshot:
    source_id: str
    target_id: str
    weight: float
    text: float | None = None
    reply: float | None = None
    mention: float | None = None
    context: float | None = None
    same_user: float | None = None
    time: float | None = None

@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    version: int
    nodes: tuple[MessageSnapshot, ...]
    edges: tuple[GraphEdgeSnapshot, ...]


@dataclass(frozen=True, slots=True)
class TopicInfo:
    topic_id: str
    members: tuple[tuple[str, float], ...]
    representative_ids: tuple[str, ...]
    last_active: float
    centroid: tuple[tuple[str, float], ...] = ()
    centroid_norm: float = 0.0
    cohesion: float = 0.0
    user_diversity: float = 0.0
    repeat_score: float = 0.0
    time_concentration: float = 0.0


@dataclass(frozen=True, slots=True)
class TopicTransition:
    source_topic_id: str
    target_topic_id: str
    weight: float
    shared_members: int


@dataclass(frozen=True, slots=True)
class TopicCompensation:
    message_id: str
    timestamp: float
    graph_version: int
    vector_weights: tuple[tuple[str, float], ...]
    affinities: tuple[tuple[str, float], ...]
    memberships: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class TopicActivity:
    topic_id: str
    score: float
    inherited_score: float
    message_count: int
    last_active: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class TopicSnapshot:
    source_version: int
    live_version: int = 0
    topics: tuple[TopicInfo, ...] = ()
    memberships: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    transitions: tuple[TopicTransition, ...] = ()
    compensations: tuple[TopicCompensation, ...] = ()
    activity: tuple[TopicActivity, ...] = ()

    @classmethod
    def empty(cls, version: int = 0) -> TopicSnapshot:
        return cls(source_version=version, live_version=version)


@dataclass(frozen=True, slots=True)
class Hotword:
    term: str
    score: float
    document_count: int
    user_count: int
    textrank: float = 0.0
    specificity: float = 0.0
    frequency: float = 0.0
    user_diversity: float = 0.0
    topic_specificity: float = 0.0
    sentence_centrality: float = 0.0
    source_quality: float = 0.0
    part_of_speech: str = "n"


@dataclass(frozen=True, slots=True)
class HotSentence:
    message_id: str
    text: str
    score: float
    pagerank: float = 0.0
    weighted_degree: float = 0.0
    topic_confidence: float = 0.0
    user_diversity: float = 0.0
    recency: float = 0.0


@dataclass(frozen=True, slots=True)
class DynamicTerm:
    term: str
    score: float
    frequency: int
    user_count: int
    status: DynamicTermStatus = DynamicTermStatus.PROMOTED


@dataclass(frozen=True, slots=True)
class DynamicLexiconTerm:
    term: str
    score: float
    frequency: int
    user_count: int
    first_seen: float
    last_seen: float
    promoted_at: float
    expires_at: float
    hit_count: int = 0


@dataclass(frozen=True, slots=True)
class AnalyticsSnapshot:
    source_version: int
    published_at: float
    topics: tuple[TopicInfo, ...] = ()
    memberships: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    hotwords: tuple[Hotword, ...] = ()
    hot_sentences: tuple[HotSentence, ...] = ()
    dynamic_terms: tuple[DynamicTerm, ...] = ()
    dynamic_stopwords: tuple[str, ...] = ()

    @classmethod
    def empty(cls, version: int = 0) -> AnalyticsSnapshot:
        return cls(source_version=version, published_at=time.time())


@dataclass(frozen=True, slots=True)
class StoredRelatednessState:
    dynamic_terms: tuple[str, ...] = ()
    dynamic_new_words: tuple[DynamicLexiconTerm, ...] = ()
    analytics: AnalyticsSnapshot = field(default_factory=AnalyticsSnapshot.empty)
    topics: TopicSnapshot = field(default_factory=TopicSnapshot.empty)


@dataclass(frozen=True, slots=True)
class TopicInterest:
    topic_id: str
    weight: float
    updated_at: float
    source_version: int


@dataclass(frozen=True, slots=True)
class StoredMemberCommunityState:
    group_id: str = ""
    user_id: str = ""
    saved_at: float = 0.0
    topic_interests: tuple[TopicInterest, ...] = ()
    dialogue_chains: tuple[DialogueChain, ...] = ()
