from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import networkx as nx

from .models import (
    GraphSnapshot,
    RelatednessConfig,
    TopicActivity,
    TopicCompensation,
    TopicInfo,
    TopicSnapshot,
    TopicTransition,
)


def message_topic_affinities(
    vector_weights: tuple[tuple[str, float], ...],
    snapshot: TopicSnapshot,
) -> tuple[tuple[str, float], ...]:
    vector = dict(vector_weights)
    norm = math.sqrt(sum(value * value for value in vector.values()))
    result: list[tuple[str, float]] = []
    for topic in snapshot.topics:
        if norm <= 0 or topic.centroid_norm <= 0:
            similarity = 0.0
        else:
            centroid = dict(topic.centroid)
            dot = sum(value * centroid.get(term, 0.0) for term, value in vector.items())
            similarity = max(0.0, min(1.0, dot / (norm * topic.centroid_norm)))
        result.append((topic.topic_id, similarity))
    return tuple(sorted(result, key=lambda item: (-item[1], item[0])))


def _dynamic_memberships(
    affinities: tuple[tuple[str, float], ...],
    config: RelatednessConfig,
) -> tuple[tuple[str, float], ...]:
    eligible = tuple(
        item
        for item in affinities
        if item[1] >= config.topic_dynamic_min_similarity
    )
    maximum = max((score for _, score in eligible), default=0.0)
    if maximum <= 0.0:
        return ()
    return tuple(
        item
        for item in eligible
        if item[1] / maximum >= config.topic_overlap_ratio
    )[:config.topic_membership_limit]


def _activity_from_compensations(
    snapshot: TopicSnapshot,
    compensations: tuple[TopicCompensation, ...],
    config: RelatednessConfig,
) -> tuple[TopicActivity, ...]:
    states: dict[str, list[float | int]] = {}
    for item in snapshot.activity:
        if item.inherited_score <= 0.0:
            continue
        states[item.topic_id] = [
            item.inherited_score,
            item.inherited_score,
            0,
            item.last_active,
            item.updated_at,
        ]

    latest = max(
        (item.timestamp for item in compensations),
        default=max((item.updated_at for item in snapshot.activity), default=0.0),
    )
    for item in sorted(
        compensations,
        key=lambda value: (value.timestamp, value.graph_version, value.message_id),
    ):
        for topic_id, similarity in item.memberships:
            state = states.setdefault(
                topic_id,
                [0.0, 0.0, 0, item.timestamp, item.timestamp],
            )
            elapsed = max(0.0, item.timestamp - float(state[4]))
            decay = math.exp(
                -elapsed / config.topic_dynamic_activity_decay_seconds
            )
            state[0] = float(state[0]) * decay + similarity
            state[1] = float(state[1]) * decay
            state[2] = int(state[2]) + 1
            state[3] = item.timestamp
            state[4] = item.timestamp

    activities: list[TopicActivity] = []
    for topic_id, state in states.items():
        elapsed = max(0.0, latest - float(state[4]))
        decay = math.exp(-elapsed / config.topic_dynamic_activity_decay_seconds)
        score = float(state[0]) * decay
        inherited = float(state[1]) * decay
        if score <= 1e-9:
            continue
        activities.append(TopicActivity(
            topic_id=topic_id,
            score=score,
            inherited_score=inherited,
            message_count=int(state[2]),
            last_active=float(state[3]),
            updated_at=latest,
        ))
    return tuple(sorted(activities, key=lambda item: (-item.score, item.topic_id)))


def apply_topic_compensation(
    snapshot: TopicSnapshot,
    *,
    message_id: str,
    timestamp: float,
    graph_version: int,
    vector_weights: tuple[tuple[str, float], ...],
    config: RelatednessConfig,
    affinities: tuple[tuple[str, float], ...] | None = None,
) -> TopicSnapshot:
    if not snapshot.topics:
        return snapshot
    affinity_values = (
        message_topic_affinities(vector_weights, snapshot)
        if affinities is None
        else affinities
    )
    compensation = TopicCompensation(
        message_id=message_id,
        timestamp=timestamp,
        graph_version=graph_version,
        vector_weights=vector_weights,
        affinities=affinity_values,
        memberships=_dynamic_memberships(affinity_values, config),
    )
    values = {
        item.message_id: item for item in snapshot.compensations
    }
    values[message_id] = compensation
    compensations = tuple(sorted(
        values.values(),
        key=lambda item: (item.graph_version, item.message_id),
    ))[-config.topic_compensation_limit:]
    return TopicSnapshot(
        source_version=snapshot.source_version,
        live_version=max(snapshot.live_version, graph_version),
        topics=snapshot.topics,
        memberships=snapshot.memberships,
        transitions=snapshot.transitions,
        compensations=compensations,
        activity=_activity_from_compensations(snapshot, compensations, config),
    )


def _gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    count = len(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (count * sum(ordered)) - (count + 1.0) / count


def _topic_id(seed: str) -> str:
    digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=6).hexdigest()
    return f"topic-{digest}"


def _core_graph(
    snapshot: GraphSnapshot,
    config: RelatednessConfig,
) -> tuple[nx.Graph, dict[str, dict[str, float]]]:
    graph = nx.Graph()
    graph.add_nodes_from(node.msg_id for node in snapshot.nodes)
    boundary: dict[str, dict[str, float]] = defaultdict(dict)
    for edge in snapshot.edges:
        if edge.weight < config.topic_min_edge:
            continue
        graph.add_edge(edge.source_id, edge.target_id, weight=edge.weight)
        boundary[edge.source_id][edge.target_id] = edge.weight
        boundary[edge.target_id][edge.source_id] = edge.weight
    return graph, boundary


def _core_communities(
    graph: nx.Graph,
    config: RelatednessConfig,
) -> tuple[frozenset[str], ...]:
    if graph.number_of_edges() == 0:
        return tuple(frozenset((str(node),)) for node in sorted(graph.nodes))

    best: tuple[frozenset[str], ...] | None = None
    best_modularity = float("-inf")
    best_signature: tuple[tuple[str, ...], ...] | None = None
    for seed in range(config.topic_louvain_restarts):
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=config.topic_resolution,
            seed=seed,
        )
        normalized = tuple(
            frozenset(community)
            for community in sorted(communities, key=lambda values: min(values))
        )
        signature = tuple(tuple(sorted(community)) for community in normalized)
        modularity = nx.community.modularity(
            graph,
            normalized,
            weight="weight",
            resolution=config.topic_resolution,
        )
        if (
            modularity > best_modularity
            or (
                modularity == best_modularity
                and (best_signature is None or signature < best_signature)
            )
        ):
            best = normalized
            best_modularity = modularity
            best_signature = signature

    assert best is not None
    return best


def _topic_transitions(
    previous: TopicSnapshot,
    topics: tuple[TopicInfo, ...],
    config: RelatednessConfig,
) -> tuple[TopicTransition, ...]:
    """Project old topic mass onto new topics through surviving messages."""
    if not previous.topics or not topics:
        return ()

    transitions: list[TopicTransition] = []
    for source in previous.topics:
        source_members = dict(source.members)
        candidates: list[tuple[TopicInfo, float, int]] = []
        for target in topics:
            target_members = dict(target.members)
            shared = source_members.keys() & target_members.keys()
            if not shared:
                continue
            mass = sum(
                min(source_members[message_id], target_members[message_id])
                for message_id in shared
            )
            if mass > 0.0:
                candidates.append((target, mass, len(shared)))
        total = sum(mass for _, mass, _ in candidates)
        if total <= 0.0:
            continue
        normalized = [
            (target, mass / total, shared_count)
            for target, mass, shared_count in candidates
        ]
        retained = [
            item for item in normalized
            if item[1] >= config.topic_inheritance_min_weight
        ]
        retained_total = sum(weight for _, weight, _ in retained)
        if retained_total <= 0.0:
            continue
        transitions.extend(
            TopicTransition(
                source_topic_id=source.topic_id,
                target_topic_id=target.topic_id,
                weight=weight / retained_total,
                shared_members=shared_count,
            )
            for target, weight, shared_count in retained
        )
    return tuple(sorted(
        transitions,
        key=lambda item: (
            item.source_topic_id,
            -item.weight,
            item.target_topic_id,
        ),
    ))


def _inherited_activity(
    previous: TopicSnapshot,
    transitions: tuple[TopicTransition, ...],
    updated_at: float,
    config: RelatednessConfig,
) -> tuple[TopicActivity, ...]:
    previous_activity = {
        item.topic_id: item for item in previous.activity
    }
    inherited: dict[str, float] = defaultdict(float)
    last_active: dict[str, float] = defaultdict(float)
    for transition in transitions:
        source = previous_activity.get(transition.source_topic_id)
        if source is None:
            continue
        contribution = (
            source.score
            * transition.weight
            * config.topic_dynamic_inheritance_factor
        )
        inherited[transition.target_topic_id] += contribution
        last_active[transition.target_topic_id] = max(
            last_active[transition.target_topic_id],
            source.last_active,
        )
    return tuple(sorted(
        (
            TopicActivity(
                topic_id=topic_id,
                score=score,
                inherited_score=score,
                message_count=0,
                last_active=last_active[topic_id],
                updated_at=updated_at,
            )
            for topic_id, score in inherited.items()
            if score > 1e-9
        ),
        key=lambda item: (-item.score, item.topic_id),
    ))


def build_topics(
    snapshot: GraphSnapshot,
    config: RelatednessConfig,
    *,
    previous: TopicSnapshot | None = None,
) -> TopicSnapshot:
    if not snapshot.nodes:
        return TopicSnapshot.empty(snapshot.version)

    timestamps = {node.msg_id: node.timestamp for node in snapshot.nodes}
    graph, boundary = _core_graph(snapshot, config)
    cores = _core_communities(graph, config)
    node_core = {
        node_id: core_index
        for core_index, core in enumerate(cores)
        for node_id in core
    }
    latest_timestamp = max(timestamps.values())
    active_cores = {
        core_index
        for core_index, core in enumerate(cores)
        if max(timestamps[node_id] for node_id in core)
        >= latest_timestamp - config.topic_expire_seconds
    }
    topic_ids = {
        core_index: _topic_id(min(cores[core_index]))
        for core_index in active_cores
    }

    topic_members: dict[str, dict[str, float]] = defaultdict(dict)
    memberships: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for node_id in timestamps:
        strengths: dict[int, float] = defaultdict(float)
        for neighbor_id, weight in boundary.get(node_id, {}).items():
            core_index = node_core[neighbor_id]
            if core_index in active_cores:
                strengths[core_index] += weight

        own_core = node_core[node_id]
        selected: list[int] = []
        if own_core in active_cores:
            selected.append(own_core)
        maximum = max(strengths.values(), default=0.0)
        if maximum > 0:
            candidates = sorted(
                (
                    (core_index, strength)
                    for core_index, strength in strengths.items()
                    if core_index != own_core
                    and strength / maximum >= config.topic_overlap_ratio
                ),
                key=lambda item: (-item[1], item[0]),
            )
            selected.extend(
                core_index
                for core_index, _ in candidates[
                    :max(0, config.topic_membership_limit - len(selected))
                ]
            )

        raw_scores = {
            core_index: (
                strengths.get(core_index, 0.0)
                or maximum
                or 1.0
            )
            for core_index in selected
        }
        total = sum(raw_scores.values()) or 1.0
        values = tuple(
            sorted(
                (
                    (topic_ids[core_index], score / total)
                    for core_index, score in raw_scores.items()
                ),
                key=lambda item: (-item[1], item[0]),
            )
        )
        memberships.append((node_id, values))
        for topic_id, score in values:
            topic_members[topic_id][node_id] = score

    topics: list[TopicInfo] = []
    nodes = {node.msg_id: node for node in snapshot.nodes}
    edge_lookup = tuple(snapshot.edges)
    for topic_id, member_scores in topic_members.items():
        ordered = tuple(
            sorted(
                member_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        member_ids = set(member_scores)
        weighted_degree = {
            message_id: sum(
                edge.weight
                for edge in edge_lookup
                if edge.source_id in member_ids
                and edge.target_id in member_ids
                and (edge.source_id == message_id or edge.target_id == message_id)
            )
            for message_id in member_ids
        }
        max_degree = max(weighted_degree.values(), default=0.0) or 1.0
        centroid_values: dict[str, float] = defaultdict(float)
        centroid_total = 0.0
        for message_id, membership in member_scores.items():
            centrality = weighted_degree[message_id] / max_degree
            weight = membership * (0.5 + 0.5 * centrality)
            centroid_total += weight
            for term, value in nodes[message_id].vector_weights:
                centroid_values[term] += value * weight
        if centroid_total > 0:
            centroid_values = {
                term: value / centroid_total
                for term, value in centroid_values.items()
                if value > 1e-9
            }
        centroid = tuple(sorted(centroid_values.items()))
        centroid_norm = math.sqrt(sum(value * value for _, value in centroid))

        internal = [
            edge.weight for edge in edge_lookup
            if edge.source_id in member_ids and edge.target_id in member_ids
        ]
        external = [
            edge.weight for edge in edge_lookup
            if (edge.source_id in member_ids) != (edge.target_id in member_ids)
        ]
        internal_mean = sum(internal) / len(internal) if internal else 0.0
        external_mean = sum(external) / len(external) if external else 0.0
        cohesion = (
            internal_mean / (internal_mean + external_mean)
            if internal_mean + external_mean > 0 else 0.0
        )
        user_counts: dict[str, int] = defaultdict(int)
        content_keys: set[tuple[str, ...]] = set()
        times: list[float] = []
        for message_id in member_ids:
            node = nodes[message_id]
            user_counts[node.user_id] += 1
            content_keys.add(tuple(sorted(node.media_ids)) or (node.text.casefold(),))
            times.append(node.timestamp)
        user_diversity = 1.0 - _gini(list(user_counts.values()))
        repeat_score = len(content_keys) / len(member_ids) if member_ids else 0.0
        times.sort()
        intervals = [times[index] - times[index - 1] for index in range(1, len(times))]
        if len(intervals) >= 2 and (mean_interval := sum(intervals) / len(intervals)) > 0:
            variance = sum((value - mean_interval) ** 2 for value in intervals) / (len(intervals) - 1)
            cv = math.sqrt(variance) / mean_interval
            time_concentration = cv / (1.0 + cv)
        else:
            time_concentration = 0.0

        topics.append(
            TopicInfo(
                topic_id=topic_id,
                members=ordered,
                representative_ids=tuple(msg_id for msg_id, _ in ordered[:3]),
                last_active=max(timestamps[msg_id] for msg_id, _ in ordered),
                centroid=centroid,
                centroid_norm=centroid_norm,
                cohesion=cohesion,
                user_diversity=user_diversity,
                repeat_score=repeat_score,
                time_concentration=time_concentration,
            )
        )
    topics.sort(key=lambda topic: topic.topic_id)
    topic_values = tuple(topics)
    previous_snapshot = previous or TopicSnapshot.empty()
    transitions = _topic_transitions(
        previous_snapshot,
        topic_values,
        config,
    )
    return TopicSnapshot(
        source_version=snapshot.version,
        live_version=snapshot.version,
        topics=topic_values,
        memberships=tuple(memberships),
        transitions=transitions,
        activity=_inherited_activity(
            previous_snapshot,
            transitions,
            latest_timestamp,
            config,
        ),
    )
