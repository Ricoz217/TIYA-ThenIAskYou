from __future__ import annotations

import heapq
import math
from collections import OrderedDict

from .models import (
    AnchorEvidence,
    GraphEdgeSnapshot,
    GraphSnapshot,
    MemberEvidenceSnapshot,
    MessageInput,
    MessageReference,
    MessageSnapshot,
    RelatedMatch,
    RelatednessConfig,
    RelatednessScore,
    TextFeatures,
)


def _normalize_weight(weight: float) -> float:
    return min(1.0, max(0.0, weight))


def _cosine_similarity(source: TextFeatures, target: TextFeatures) -> float:
    source_weights = dict(source.term_weights)
    target_weights = dict(target.term_weights)
    if not source_weights or not target_weights:
        return 0.0
    if len(source_weights) > len(target_weights):
        source_weights, target_weights = target_weights, source_weights
    dot_product = sum(
        weight * target_weights.get(term, 0.0)
        for term, weight in source_weights.items()
    )
    if dot_product <= 0:
        return 0.0
    source_norm = math.sqrt(sum(weight * weight for weight in source_weights.values()))
    target_norm = math.sqrt(sum(weight * weight for weight in target_weights.values()))
    denominator = source_norm * target_norm
    return min(1.0, dot_product / denominator) if denominator > 0 else 0.0


class MessageGraph:
    __slots__ = (
        "config",
        "_nodes",
        "_features",
        "_adjacency",
        "version",
    )

    def __init__(self, config: RelatednessConfig | None = None):
        self.config = config or RelatednessConfig()
        self._nodes: OrderedDict[str, MessageInput] = OrderedDict()
        self._features: dict[str, TextFeatures] = {}
        self._adjacency: dict[str, dict[str, RelatednessScore]] = {}
        self.version = 0

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def contains(self, msg_id: str) -> bool:
        return msg_id in self._nodes

    def feature_weights(self, msg_id: str) -> tuple[tuple[str, float], ...] | None:
        features = self._features.get(msg_id)
        return features.term_weights if features is not None else None

    def _edge_score(
        self,
        source: MessageInput,
        target: MessageInput,
        *,
        text: float = 0.0,
        reply: float = 0.0,
        mention: float = 0.0,
        context: float = 0.0,
    ) -> RelatednessScore:
        distance = max(0.0, abs(source.timestamp - target.timestamp))
        time_score = math.exp(-distance / 600.0) * 0.1
        same_user = self.config.same_user_weight if source.user_id == target.user_id else 0.0
        final = text + reply + mention + context + same_user + time_score
        return RelatednessScore(
            text=text,
            reply=reply,
            mention=mention,
            context=context,
            same_user=same_user,
            time=time_score,
            final=final,
        )

    def _merge_edge(self, source_id: str, target_id: str, incoming: RelatednessScore) -> None:
        if source_id == target_id or target_id not in self._nodes:
            return
        existing = self._adjacency.setdefault(source_id, {}).get(target_id)
        if existing is None:
            merged = incoming
        else:
            source = self._nodes[source_id]
            target = self._nodes[target_id]
            merged = self._edge_score(
                source,
                target,
                text=max(existing.text, incoming.text),
                reply=max(existing.reply, incoming.reply),
                mention=max(existing.mention, incoming.mention),
                context=max(existing.context, incoming.context),
            )
        self._adjacency.setdefault(source_id, {})[target_id] = merged
        self._adjacency.setdefault(target_id, {})[source_id] = merged

    def _direct_result_score(self, edge: RelatednessScore) -> RelatednessScore:
        final = _normalize_weight(
            edge.text
            + edge.reply * self.config.reply_direct_factor
            + edge.mention
            + edge.context
            + edge.same_user
            + edge.time
        )
        return RelatednessScore(
            text=edge.text,
            reply=edge.reply,
            mention=edge.mention,
            context=edge.context,
            same_user=edge.same_user,
            time=edge.time,
            final=final,
        )

    def _add_text_edges(
        self,
        message: MessageInput,
        candidates: tuple[tuple[str, float], ...],
    ) -> None:
        source_features = self._features[message.msg_id]
        for target_id, _ in candidates[:self.config.text_edge_limit]:
            if target_id == message.msg_id or target_id not in self._nodes:
                continue
            similarity = _cosine_similarity(
                source_features,
                self._features[target_id],
            )
            if similarity <= 0:
                continue
            score = self._edge_score(
                message,
                self._nodes[target_id],
                text=self.config.text_weight * similarity,
            )
            self._merge_edge(message.msg_id, target_id, score)

    def add(
        self,
        message: MessageInput,
        features: TextFeatures,
        text_candidates: tuple[tuple[str, float], ...] = (),
    ) -> tuple[str, ...]:
        if message.msg_id in self._nodes:
            return ()
        previous_ids = tuple(self._nodes)
        self._nodes[message.msg_id] = message
        self._features[message.msg_id] = features
        self._adjacency.setdefault(message.msg_id, {})

        self._add_text_edges(message, text_candidates)

        for offset, target_id in enumerate(
            reversed(previous_ids[-self.config.context_edge_limit:]),
            start=1,
        ):
            context = self.config.context_weight * math.exp(-(offset - 1) / 4.0)
            score = self._edge_score(message, self._nodes[target_id], context=context)
            self._merge_edge(message.msg_id, target_id, score)

        if message.reply_to and message.reply_to in self._nodes:
            target = self._nodes[message.reply_to]
            score = self._edge_score(message, target, reply=self.config.reply_weight)
            self._merge_edge(message.msg_id, message.reply_to, score)

        if message.mention_ids:
            matches = 0
            for target_id in reversed(previous_ids):
                target = self._nodes[target_id]
                if target.user_id not in message.mention_ids:
                    continue
                score = self._edge_score(message, target, mention=self.config.mention_weight)
                self._merge_edge(message.msg_id, target_id, score)
                matches += 1
                if matches >= self.config.mention_message_limit:
                    break

        evicted: list[str] = []
        while len(self._nodes) > self.config.message_window:
            evicted_id, _ = self._nodes.popitem(last=False)
            evicted.append(evicted_id)
            self._features.pop(evicted_id, None)
            neighbors = self._adjacency.pop(evicted_id, {})
            for neighbor_id in neighbors:
                self._adjacency.get(neighbor_id, {}).pop(evicted_id, None)
        self.version += 1
        return tuple(evicted)

    def _clear_text_evidence(self, message_id: str) -> None:
        source = self._nodes[message_id]
        for target_id, edge in tuple(self._adjacency.get(message_id, {}).items()):
            refreshed = self._edge_score(
                source,
                self._nodes[target_id],
                reply=edge.reply,
                mention=edge.mention,
                context=edge.context,
            )
            self._adjacency[message_id][target_id] = refreshed
            self._adjacency[target_id][message_id] = refreshed

    def enrich(
        self,
        message: MessageInput,
        features: TextFeatures,
        text_candidates: tuple[tuple[str, float], ...] = (),
    ) -> bool:
        previous = self._nodes.get(message.msg_id)
        if previous is None:
            return False
        self._nodes[message.msg_id] = message
        self._features[message.msg_id] = features
        self._clear_text_evidence(message.msg_id)
        self._add_text_edges(message, text_candidates)
        self.version += 1
        return True

    def score(self, source_id: str, target_id: str) -> RelatednessScore | None:
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        edge = self._adjacency.get(source_id, {}).get(target_id)
        return self._direct_result_score(edge) if edge is not None else None

    def _propagated_scores(self, msg_id: str) -> dict[str, float]:
        if not self.config.enable_relation_propagation:
            return {}
        propagated_scores: dict[str, float] = {}
        queue: list[tuple[float, int, str, frozenset[str]]] = [
            (-1.0, 0, msg_id, frozenset((msg_id,)))
        ]
        visited = 0
        while queue and visited < self.config.propagation_node_limit:
            negative_energy, depth, node_id, path = heapq.heappop(queue)
            energy = -negative_energy
            visited += 1
            if depth >= self.config.propagation_depth:
                continue
            for neighbor_id, edge in self._adjacency.get(node_id, {}).items():
                if neighbor_id in path:
                    continue
                propagated = (
                    energy
                    * _normalize_weight(edge.final)
                    * self.config.propagation_decay
                )
                if propagated < self.config.propagation_min_energy:
                    continue
                next_depth = depth + 1
                if (
                    next_depth >= 2
                    and propagated > propagated_scores.get(neighbor_id, 0.0)
                ):
                    propagated_scores[neighbor_id] = propagated
                if next_depth < self.config.propagation_depth:
                    heapq.heappush(
                        queue,
                        (
                            -propagated,
                            next_depth,
                            neighbor_id,
                            path | frozenset((neighbor_id,)),
                        ),
                    )
        propagated_scores.pop(msg_id, None)
        return propagated_scores

    def _relation_details(
        self,
        source_id: str,
        target_id: str,
        propagation: float,
    ) -> RelatednessScore:
        direct = self.score(source_id, target_id)
        weighted_propagation = min(
            1.0,
            propagation * self.config.propagation_score_weight,
        )
        if direct is None:
            return RelatednessScore(
                propagation=weighted_propagation,
                final=weighted_propagation,
            )
        return RelatednessScore(
            text=direct.text,
            reply=direct.reply,
            mention=direct.mention,
            context=direct.context,
            same_user=direct.same_user,
            time=direct.time,
            propagation=weighted_propagation,
            final=_normalize_weight(direct.final + weighted_propagation),
        )

    def related(
        self,
        msg_id: str,
        *,
        limit: int = 10,
        explain: bool = False,
    ) -> tuple[RelatedMatch, ...]:
        if msg_id not in self._nodes or limit <= 0:
            return ()
        propagated_scores = self._propagated_scores(msg_id)

        matches: list[RelatedMatch] = []
        candidate_ids = set(self._adjacency.get(msg_id, ())) | set(propagated_scores)
        for target_id in candidate_ids:
            details = self._relation_details(
                msg_id,
                target_id,
                propagated_scores.get(target_id, 0.0),
            )
            if details.final < self.config.related_min_score:
                continue
            matches.append(
                RelatedMatch(
                    message_id=target_id,
                    score=details.final,
                    details=details if explain else None,
                )
            )
        matches.sort(key=lambda item: (-item.score, item.message_id))
        return tuple(matches[:limit])

    def anchor_evidence(
        self,
        source_id: str,
        anchor_ids: tuple[str, ...],
    ) -> MemberEvidenceSnapshot | None:
        source = self._nodes.get(source_id)
        if source is None:
            return None
        positions = {
            message_id: index for index, message_id in enumerate(self._nodes)
        }
        source_sequence = positions[source_id]
        propagated_scores = self._propagated_scores(source_id)
        anchors: list[AnchorEvidence] = []
        for anchor_id in dict.fromkeys(anchor_ids):
            anchor = self._nodes.get(anchor_id)
            if anchor is None or anchor_id == source_id:
                continue
            anchor_sequence = positions[anchor_id]
            anchors.append(
                AnchorEvidence(
                    message_id=anchor_id,
                    user_id=anchor.user_id,
                    timestamp=anchor.timestamp,
                    sequence=anchor_sequence,
                    message_gap=max(
                        0,
                        abs(source_sequence - anchor_sequence) - 1,
                    ),
                    relation=self._relation_details(
                        source_id,
                        anchor_id,
                        propagated_scores.get(anchor_id, 0.0),
                    ),
                )
            )
        return MemberEvidenceSnapshot(
            source_id=source_id,
            source_user_id=source.user_id,
            source_timestamp=source.timestamp,
            source_sequence=source_sequence,
            anchors=tuple(anchors),
        )

    def message_references(
        self,
        message_ids: tuple[str, ...] = (),
        user_id: str | None = None,
        limit: int = 32,
    ) -> tuple[MessageReference, ...]:
        if limit <= 0:
            return ()
        positions = {
            message_id: index for index, message_id in enumerate(self._nodes)
        }
        if message_ids:
            selected = [
                message_id
                for message_id in dict.fromkeys(message_ids)
                if message_id in self._nodes
            ][:limit]
        else:
            selected = [
                message_id
                for message_id, message in reversed(self._nodes.items())
                if user_id is None or message.user_id == user_id
            ][:limit]
            selected.reverse()
        return tuple(
            MessageReference(
                message_id=message_id,
                user_id=self._nodes[message_id].user_id,
                timestamp=self._nodes[message_id].timestamp,
                sequence=positions[message_id],
                reply_to=self._nodes[message_id].reply_to,
                mention_ids=self._nodes[message_id].mention_ids,
                content_key=(
                    tuple(sorted(self._nodes[message_id].media_ids))
                    or (self._nodes[message_id].text.casefold(),)
                ),
            )
            for message_id in selected
        )

    def snapshot(self) -> GraphSnapshot:
        return self._snapshot(tuple(self._nodes))

    def local_snapshot(
        self,
        source_id: str,
        window_seconds: float,
        message_limit: int,
    ) -> GraphSnapshot:
        source = self._nodes.get(source_id)
        if source is None or window_seconds <= 0.0 or message_limit <= 0:
            return GraphSnapshot(version=self.version, nodes=(), edges=())
        node_ids = tuple(self._nodes)
        source_index = node_ids.index(source_id)
        eligible = tuple(
            message_id
            for message_id in node_ids[:source_index + 1]
            if 0.0 <= source.timestamp - self._nodes[message_id].timestamp
            <= window_seconds
        )
        return self._snapshot(eligible[-message_limit:])

    def continuity_evidence_snapshot(
        self,
        source_id: str,
        target_ids: tuple[str, ...] = (),
        *,
        include_reply_interval: bool = True,
    ) -> GraphSnapshot:
        source = self._nodes.get(source_id)
        if source is None:
            return GraphSnapshot(version=self.version, nodes=(), edges=())
        selected = {
            message_id
            for message_id in target_ids
            if message_id in self._nodes
        }
        selected.add(source_id)
        node_ids = tuple(self._nodes)
        if (
            include_reply_interval
            and source.reply_to in self._nodes
        ):
            source_index = node_ids.index(source_id)
            target_index = node_ids.index(source.reply_to)
            if target_index < source_index:
                selected.update(node_ids[target_index:source_index + 1])
        ordered = tuple(
            message_id for message_id in node_ids if message_id in selected
        )
        return self._snapshot(ordered)

    def _snapshot(self, message_ids: tuple[str, ...]) -> GraphSnapshot:
        selected = set(message_ids)
        positions = {
            message_id: index for index, message_id in enumerate(self._nodes)
        }
        nodes = tuple(
            MessageSnapshot(
                msg_id=msg_id,
                user_id=message.user_id,
                timestamp=message.timestamp,
                text=message.text,
                semantic_text=message.semantic_text,
                media_ids=message.media_ids,
                lexical_terms=self._features[msg_id].lexical_terms,
                primary_lexical_terms=(
                    self._features[msg_id].primary_lexical_terms
                ),
                semantic_lexical_terms=(
                    self._features[msg_id].semantic_lexical_terms
                ),
                lexical_pos=self._features[msg_id].lexical_pos,
                vector_weights=self._features[msg_id].term_weights,
                sequence=positions[msg_id],
                reply_to=message.reply_to,
                mention_ids=message.mention_ids,
            )
            for msg_id in message_ids
            if (message := self._nodes.get(msg_id)) is not None
        )
        edges: list[GraphEdgeSnapshot] = []
        seen: set[tuple[str, str]] = set()
        for source_id, neighbors in self._adjacency.items():
            if source_id not in selected:
                continue
            for target_id, score in neighbors.items():
                if target_id not in selected:
                    continue
                key = (
                    (source_id, target_id)
                    if source_id <= target_id
                    else (target_id, source_id)
                )
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    GraphEdgeSnapshot(
                        key[0],
                        key[1],
                        score.final,
                        text=score.text,
                        reply=score.reply,
                        mention=score.mention,
                        context=score.context,
                        same_user=score.same_user,
                        time=score.time,
                    )
                )
        edges.sort(key=lambda edge: (edge.source_id, edge.target_id))
        return GraphSnapshot(version=self.version, nodes=nodes, edges=tuple(edges))
