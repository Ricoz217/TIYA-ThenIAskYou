from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

from .models import (
    AnalyticsKind,
    AnalyticsSnapshot,
    GraphSnapshot,
    HotSentence,
    Hotword,
    MessageSnapshot,
    RelatednessConfig,
    TopicSnapshot,
)


def _node_key(node: MessageSnapshot) -> tuple[str, ...]:
    if node.media_ids:
        return ("media", *sorted(node.media_ids))
    content = (node.text or node.semantic_text).strip().casefold()
    return ("content", content) if content else ("message", node.msg_id)


def _deduplicated_nodes(
    snapshot: GraphSnapshot,
) -> tuple[MessageSnapshot, ...]:
    unique: dict[tuple[str, ...], MessageSnapshot] = {}
    for node in snapshot.nodes:
        unique.setdefault(_node_key(node), node)
    return tuple(unique.values())


def _term_sequence(node: MessageSnapshot) -> tuple[tuple[str, float], ...]:
    primary = node.primary_lexical_terms
    semantic = node.semantic_lexical_terms
    if not primary and not semantic:
        primary = node.lexical_terms
    sequence = [(term, 1.0) for term in primary]
    sequence.extend((term, 0.35) for term in semantic)
    return tuple(sequence)


def _valid_hotword(term: str, config: RelatednessConfig) -> bool:
    stripped = term.strip()
    if not config.hotword_min_chars <= len(stripped) <= config.hotword_max_chars:
        return False
    if stripped.isnumeric():
        return config.hotword_allow_numeric
    if stripped.isascii():
        letter_count = sum(character.isalpha() for character in stripped)
        if letter_count < config.hotword_min_ascii_letters:
            return False
    return True


def _valid_hotword_pos(part_of_speech: str) -> bool:
    return part_of_speech.startswith(("n", "v"))


def _pagerank(
    node_ids: tuple[str, ...],
    adjacency: dict[str, dict[str, float]],
    rounds: int,
) -> dict[str, float]:
    if not node_ids:
        return {}
    damping = 0.85
    base = (1.0 - damping) / len(node_ids)
    scores = {node_id: 1.0 / len(node_ids) for node_id in node_ids}
    strengths = {
        node_id: sum(adjacency.get(node_id, {}).values())
        for node_id in node_ids
    }
    for _ in range(rounds):
        updated = {}
        for node_id in node_ids:
            contribution = 0.0
            for neighbor_id, weight in adjacency.get(node_id, {}).items():
                strength = strengths.get(neighbor_id, 0.0)
                if strength > 0:
                    contribution += scores[neighbor_id] * weight / strength
            updated[node_id] = base + damping * contribution
        scores = updated
    maximum = max(scores.values(), default=1.0) or 1.0
    return {node_id: score / maximum for node_id, score in scores.items()}


def _dynamic_stopwords(
    nodes,
    config: RelatednessConfig,
) -> tuple[str, ...]:
    document_count: Counter[str] = Counter()
    users: dict[str, set[str]] = defaultdict(set)
    occurrence_times: dict[str, list[float]] = defaultdict(list)
    ordered_nodes = tuple(sorted(nodes, key=lambda node: node.timestamp))
    for node in ordered_nodes:
        terms = {
            term
            for term, _ in _term_sequence(node)
            if _valid_hotword(term, config)
        }
        for term in terms:
            document_count[term] += 1
            users[term].add(node.user_id)
            occurrence_times[term].append(node.timestamp)
    total_documents = len(ordered_nodes)
    total_users = len({node.user_id for node in ordered_nodes})
    if total_documents < config.dynamic_stopword_min_documents:
        return ()
    maximum_idf = math.log((total_documents + 1.0) / 1.5) or 1.0
    window_start = ordered_nodes[0].timestamp
    window_end = ordered_nodes[-1].timestamp
    window_span = max(0.0, window_end - window_start)
    eligible_counts = sorted(
        {
            count
            for count in document_count.values()
            if count >= config.dynamic_stopword_min_documents
        },
        reverse=True,
    )
    frequency_ranks = {
        count: index / max(1, len(eligible_counts))
        for index, count in enumerate(eligible_counts)
    }

    def temporal_uniformity(term: str, count: int) -> float:
        times = occurrence_times[term]
        if len(times) < 2 or window_span <= 0:
            return 1.0
        gaps = [
            later - earlier
            for earlier, later in zip(times, times[1:])
        ]
        gap_total = sum(gaps)
        if gap_total <= 0:
            gap_gini = 1.0
            normalized_cv = 1.0
        else:
            ordered_gaps = sorted(gaps)
            gap_count = len(ordered_gaps)
            gap_gini = sum(
                (2 * index - gap_count - 1) * gap
                for index, gap in enumerate(ordered_gaps, start=1)
            ) / (gap_count * gap_total)
            mean_gap = gap_total / gap_count
            variance = sum(
                (gap - mean_gap) ** 2 for gap in gaps
            ) / gap_count
            cv = math.sqrt(variance) / mean_gap if mean_gap > 0 else 0.0
            normalized_cv = cv / (1.0 + cv)
        coverage_penalty = 1.0 - min(1.0, (times[-1] - times[0]) / window_span)
        frequency_rank = frequency_ranks.get(count, 1.0)
        return (
            0.45 * gap_gini
            + 0.20 * normalized_cv
            + 0.25 * coverage_penalty
            + 0.10 * frequency_rank
        )

    result = []
    for term, count in document_count.items():
        if count < config.dynamic_stopword_min_documents:
            continue
        doc_ratio = count / total_documents
        user_ratio = len(users[term]) / max(1, total_users)
        idf = math.log((total_documents + 1.0) / (count + 0.5))
        specificity = max(0.0, min(1.0, idf / maximum_idf))
        broadly_distributed = (
            doc_ratio >= config.dynamic_stopword_doc_ratio
            or user_ratio >= config.dynamic_stopword_user_ratio
        )
        uniformly_distributed = (
            temporal_uniformity(term, count)
            <= config.dynamic_stopword_time_uniformity
            and specificity <= config.dynamic_stopword_time_max_specificity
        )
        if (
            broadly_distributed
            and specificity <= config.dynamic_stopword_max_specificity
        ) or uniformly_distributed:
            result.append(term)
    return tuple(sorted(result))


def _sentence_scores(
    snapshot: GraphSnapshot,
    topics: TopicSnapshot,
    config: RelatednessConfig,
) -> tuple[HotSentence, ...]:
    representative_by_key: dict[tuple[str, ...], MessageSnapshot] = {}
    for node in snapshot.nodes:
        key = _node_key(node)
        current = representative_by_key.get(key)
        if current is None or node.timestamp > current.timestamp:
            representative_by_key[key] = node
    representative_id = {
        node.msg_id: representative_by_key[_node_key(node)].msg_id
        for node in snapshot.nodes
    }
    nodes = {
        node.msg_id: node for node in representative_by_key.values()
    }
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in snapshot.edges:
        source_id = representative_id[edge.source_id]
        target_id = representative_id[edge.target_id]
        if source_id == target_id:
            continue
        adjacency[source_id][target_id] = max(
            adjacency[source_id].get(target_id, 0.0),
            edge.weight,
        )
        adjacency[target_id][source_id] = max(
            adjacency[target_id].get(source_id, 0.0),
            edge.weight,
        )
    node_ids = tuple(nodes)
    rank = _pagerank(node_ids, adjacency, config.hot_sentence_pagerank_rounds)
    degree = {
        node_id: sum(adjacency.get(node_id, {}).values())
        for node_id in node_ids
    }
    max_degree = max(degree.values(), default=1.0) or 1.0
    memberships = {message_id: values for message_id, values in topics.memberships}
    topic_confidences: defaultdict[str, float] = defaultdict(float)
    for message_id, values in memberships.items():
        topic_confidences[representative_id.get(message_id, message_id)] = max(
            topic_confidences[representative_id.get(message_id, message_id)],
            max((score for _, score in values), default=0.0),
        )
    latest = max((node.timestamp for node in snapshot.nodes), default=0.0)
    earliest = min((node.timestamp for node in snapshot.nodes), default=latest)
    span = max(1.0, latest - earliest)
    best_by_content: dict[tuple[str, ...], HotSentence] = {}
    for node_id, node in nodes.items():
        terms = {
            term
            for term, _ in _term_sequence(node)
            if _valid_hotword(term, config)
        }
        if len(terms) < config.hot_sentence_min_terms:
            continue
        neighbor_users = {
            nodes[neighbor_id].user_id
            for neighbor_id in adjacency.get(node_id, ())
            if neighbor_id in nodes
        }
        user_diversity = 1.0 - 1.0 / (1.0 + len(neighbor_users))
        topic_confidence = topic_confidences[node_id]
        recency = max(0.0, 1.0 - (latest - node.timestamp) / span)
        pagerank_score = rank.get(node_id, 0.0)
        degree_score = degree.get(node_id, 0.0) / max_degree
        score = (
            0.35 * pagerank_score
            + 0.25 * degree_score
            + 0.20 * topic_confidence
            + 0.10 * user_diversity
            + 0.10 * recency
        )
        sentence = HotSentence(
            message_id=node_id,
            text=node.text or node.semantic_text,
            score=score,
            pagerank=pagerank_score,
            weighted_degree=degree_score,
            topic_confidence=topic_confidence,
            user_diversity=user_diversity,
            recency=recency,
        )
        best_by_content[_node_key(node)] = sentence
    return tuple(
        sorted(
            best_by_content.values(),
            key=lambda item: (-item.score, item.message_id),
        )[:config.hot_sentence_limit]
    )


def _hotwords(
    snapshot: GraphSnapshot,
    topics: TopicSnapshot,
    sentences: tuple[HotSentence, ...],
    dynamic_stopwords: tuple[str, ...],
    config: RelatednessConfig,
) -> tuple[Hotword, ...]:
    nodes = _deduplicated_nodes(snapshot)
    blocked = set(dynamic_stopwords)
    document_count: Counter[str] = Counter()
    weighted_frequency: defaultdict[str, float] = defaultdict(float)
    primary_documents: Counter[str] = Counter()
    semantic_documents: Counter[str] = Counter()
    users: dict[str, set[str]] = defaultdict(set)
    cooccurrence: dict[str, dict[str, float]] = defaultdict(dict)
    message_topics = {
        message_id: tuple(topic_id for topic_id, _ in memberships)
        for message_id, memberships in topics.memberships
    }
    topic_frequency: dict[str, Counter[str]] = defaultdict(Counter)
    term_pos: dict[str, str] = {}
    for node in nodes:
        node_pos = dict(node.lexical_pos)
        sequence = [
            (term, weight)
            for term, weight in _term_sequence(node)
            if _valid_hotword(term, config) and term not in blocked
            and _valid_hotword_pos(node_pos.get(term, "n"))
        ]
        per_document: dict[str, float] = {}
        for term, weight in sequence:
            per_document[term] = max(per_document.get(term, 0.0), weight)
        for term, weight in per_document.items():
            term_pos.setdefault(term, node_pos.get(term, "n"))
            document_count[term] += 1
            weighted_frequency[term] += weight
            users[term].add(node.user_id)
            for topic_id in message_topics.get(node.msg_id, ()):
                topic_frequency[term][topic_id] += 1
        primary_terms = set(node.primary_lexical_terms)
        semantic_terms = set(node.semantic_lexical_terms)
        if not primary_terms and not semantic_terms:
            primary_terms = set(node.lexical_terms)
        primary_documents.update(primary_terms)
        semantic_documents.update(semantic_terms)
        for index, (term, term_weight) in enumerate(sequence):
            for other, other_weight in sequence[index + 1:index + 4]:
                if term == other:
                    continue
                weight = min(term_weight, other_weight)
                cooccurrence[term][other] = cooccurrence[term].get(other, 0.0) + weight
                cooccurrence[other][term] = cooccurrence[other].get(term, 0.0) + weight

    candidates = tuple(
        term
        for term, count in document_count.items()
        if count >= config.hotword_min_documents
        and (
            config.hotword_allow_semantic_only_ascii
            or not (term.isascii() and primary_documents[term] == 0)
        )
    )
    rank = _pagerank(candidates, cooccurrence, config.hotword_textrank_rounds)
    sentence_factor: defaultdict[str, float] = defaultdict(float)
    nodes_by_id = {node.msg_id: node for node in snapshot.nodes}
    for position, sentence in enumerate(sentences, start=1):
        sentence_node = nodes_by_id.get(sentence.message_id)
        if sentence_node is None:
            continue
        factor = 1.0 / position
        for term, _ in _term_sequence(sentence_node):
            sentence_factor[term] += factor
    max_sentence_factor = max(sentence_factor.values(), default=1.0) or 1.0
    total_documents = max(1, len(nodes))
    maximum_idf = math.log((total_documents + 1.0) / 1.5) or 1.0
    maximum_frequency = math.log1p(max(document_count.values(), default=1)) or 1.0
    maximum_users = math.log1p(max((len(value) for value in users.values()), default=1)) or 1.0
    result = []
    for term in candidates:
        count = document_count[term]
        idf = math.log((total_documents + 1.0) / (count + 0.5))
        specificity = max(0.0, min(1.0, idf / maximum_idf))
        frequency_score = math.log1p(weighted_frequency[term]) / maximum_frequency
        user_score = math.log1p(len(users[term])) / maximum_users
        per_topic = topic_frequency.get(term)
        topic_specificity = (
            max(per_topic.values()) / sum(per_topic.values())
            if per_topic
            else 0.5
        )
        centrality = sentence_factor[term] / max_sentence_factor
        primary_count = primary_documents[term]
        semantic_count = semantic_documents[term]
        source_quality = (
            (
                primary_count
                + config.semantic_text_weight * semantic_count
            )
            / (primary_count + semantic_count)
            if primary_count + semantic_count
            else 0.0
        )
        textrank_score = rank.get(term, 0.0) * source_quality
        score = (
            0.25 * textrank_score
            + 0.20 * specificity
            + 0.20 * frequency_score
            + 0.15 * user_score
            + 0.10 * topic_specificity
            + 0.05 * centrality
            + 0.05 * source_quality
        )
        part_of_speech = term_pos.get(term, "n")
        if part_of_speech.startswith("v"):
            score *= config.hotword_verb_score_factor
        result.append(
            Hotword(
                term=term,
                score=score,
                document_count=count,
                user_count=len(users[term]),
                textrank=textrank_score,
                specificity=specificity,
                frequency=frequency_score,
                user_diversity=user_score,
                topic_specificity=topic_specificity,
                sentence_centrality=centrality,
                source_quality=source_quality,
                part_of_speech=part_of_speech,
            )
        )
    result.sort(key=lambda item: (-item.score, item.term))
    return tuple(result[:config.hotword_limit])


def build_analytics(
    snapshot: GraphSnapshot,
    topics: TopicSnapshot,
    config: RelatednessConfig,
    kind: AnalyticsKind = AnalyticsKind.ALL,
) -> AnalyticsSnapshot:
    if not snapshot.nodes:
        return AnalyticsSnapshot(
            source_version=snapshot.version,
            published_at=time.time(),
            topics=topics.topics,
            memberships=topics.memberships,
        )

    needs_hotwords = kind in (AnalyticsKind.ALL, AnalyticsKind.HOTWORDS)
    dynamic_stopwords = (
        _dynamic_stopwords(snapshot.nodes, config)
        if needs_hotwords
        else ()
    )
    sentences = (
        _sentence_scores(snapshot, topics, config)
        if needs_hotwords
        else ()
    )
    hotwords = (
        _hotwords(
            snapshot,
            topics,
            sentences,
            dynamic_stopwords,
            config,
        )
        if needs_hotwords
        else ()
    )

    return AnalyticsSnapshot(
        source_version=snapshot.version,
        published_at=time.time(),
        topics=topics.topics,
        memberships=topics.memberships,
        hotwords=hotwords,
        hot_sentences=sentences,
        dynamic_stopwords=dynamic_stopwords,
    )
