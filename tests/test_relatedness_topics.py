import networkx as nx
import pytest

from TIYA.relatedness import topics as topic_module
from TIYA.relatedness.models import GraphEdgeSnapshot, GraphSnapshot, MessageSnapshot, RelatednessConfig
from TIYA.relatedness.topics import (
    apply_topic_compensation,
    build_topics,
    message_topic_affinities,
)


def _node(msg_id: str, text: str) -> MessageSnapshot:
    return MessageSnapshot(
        msg_id=msg_id,
        user_id=msg_id,
        timestamp=1.0,
        text=text,
        lexical_terms=(text,),
        vector_weights=((f"WORD:{text}", 1.0),),
    )


def test_overlapping_label_propagation_allows_bridge_membership() -> None:
    snapshot = GraphSnapshot(
        version=1,
        nodes=(
            _node("a1", "python"),
            _node("a2", "asyncio"),
            _node("bridge", "bridge"),
            _node("b1", "dinner"),
            _node("b2", "cooking"),
        ),
        edges=(
            GraphEdgeSnapshot("a1", "a2", 0.9),
            GraphEdgeSnapshot("a2", "bridge", 0.8),
            GraphEdgeSnapshot("bridge", "b1", 0.8),
            GraphEdgeSnapshot("b1", "b2", 0.9),
        ),
    )

    result = build_topics(
        snapshot,
        RelatednessConfig(topic_membership_limit=3, topic_overlap_ratio=0.4),
    )

    bridge_topics = dict(result.memberships)["bridge"]
    assert len(bridge_topics) >= 2
    assert result == build_topics(snapshot, RelatednessConfig(
        topic_membership_limit=3,
        topic_overlap_ratio=0.4,
    ))


def test_expired_topic_is_not_published() -> None:
    snapshot = GraphSnapshot(
        version=2,
        nodes=(
            MessageSnapshot("old-1", "u1", 1.0, "old", ("old",)),
            MessageSnapshot("old-2", "u2", 2.0, "old", ("old",)),
            MessageSnapshot("new", "u3", 1000.0, "new", ("new",)),
        ),
        edges=(GraphEdgeSnapshot("old-1", "old-2", 0.9),),
    )

    result = build_topics(
        snapshot,
        RelatednessConfig(topic_expire_seconds=10, topic_min_edge=0.1),
    )

    assert all(
        "old-1" not in dict(topic.members)
        for topic in result.topics
    )


def test_dense_core_is_not_fragmented_into_many_small_topics() -> None:
    nodes = tuple(_node(f"n{index}", f"topic-{index}") for index in range(8))
    edges = tuple(
        GraphEdgeSnapshot(source.msg_id, target.msg_id, 0.7)
        for index, source in enumerate(nodes)
        for target in nodes[index + 1:]
    )

    result = build_topics(
        GraphSnapshot(version=3, nodes=nodes, edges=edges),
        RelatednessConfig(),
    )

    meaningful = [topic for topic in result.topics if len(topic.members) >= 5]
    assert len(meaningful) == 1
    assert len(meaningful[0].members) == 8


def test_topic_core_uses_complete_edge_weight() -> None:
    nodes = tuple(_node(f"n{index}", f"message-{index}") for index in range(4))
    edges = tuple(
        GraphEdgeSnapshot(
            nodes[index].msg_id,
            nodes[index + 1].msg_id,
            0.8,
            text=0.0,
            reply=0.0,
            mention=0.0,
            context=0.8,
            same_user=0.0,
            time=0.0,
        )
        for index in range(3)
    )

    graph, _ = topic_module._core_graph(
        GraphSnapshot(version=4, nodes=nodes, edges=edges),
        RelatednessConfig(),
    )

    assert graph.number_of_edges() == 3
    assert graph["n0"]["n1"]["weight"] == pytest.approx(0.8)


def test_default_topic_model_allows_soft_boundary_overlap() -> None:
    snapshot = GraphSnapshot(
        version=5,
        nodes=(
            _node("a1", "python"),
            _node("a2", "asyncio"),
            _node("bridge", "bridge"),
            _node("b1", "dinner"),
            _node("b2", "cooking"),
        ),
        edges=(
            GraphEdgeSnapshot("a1", "a2", 0.9),
            GraphEdgeSnapshot("a2", "bridge", 0.8),
            GraphEdgeSnapshot("bridge", "b1", 0.8),
            GraphEdgeSnapshot("b1", "b2", 0.9),
        ),
    )

    result = build_topics(snapshot, RelatednessConfig())

    assert len(dict(result.memberships)["bridge"]) >= 2


def test_louvain_restarts_select_highest_modularity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        (("a", "b", 1.0), ("b", "c", 0.05), ("c", "d", 1.0))
    )
    candidates = (
        ({"a", "b", "c", "d"},),
        ({"a", "b"}, {"c", "d"}),
        ({"a"}, {"b"}, {"c"}, {"d"}),
    )
    seeds: list[int] = []

    def fake_louvain(
        graph: nx.Graph,
        *,
        weight: str,
        resolution: float,
        seed: int,
    ) -> tuple[set[str], ...]:
        del graph, weight, resolution
        seeds.append(seed)
        return candidates[seed]

    monkeypatch.setattr(
        topic_module.nx.community,
        "louvain_communities",
        fake_louvain,
    )

    result = topic_module._core_communities(
        graph,
        RelatednessConfig(topic_louvain_restarts=3),
    )

    assert seeds == [0, 1, 2]
    assert result == (frozenset(("a", "b")), frozenset(("c", "d")))


def test_louvain_restarts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="topic_louvain_restarts"):
        RelatednessConfig(topic_louvain_restarts=0)

    with pytest.raises(ValueError, match="topic dynamic threshold"):
        RelatednessConfig(topic_dynamic_min_similarity=1.1)

    with pytest.raises(ValueError, match="topic dynamic decay"):
        RelatednessConfig(topic_dynamic_activity_decay_seconds=0.0)


def test_topic_profile_centroid_scores_new_message() -> None:
    nodes = (
        _node("python-1", "python"),
        _node("python-2", "python"),
        _node("dinner-1", "dinner"),
        _node("dinner-2", "dinner"),
    )
    snapshot = GraphSnapshot(
        version=6,
        nodes=nodes,
        edges=(
            GraphEdgeSnapshot("python-1", "python-2", 1.0),
            GraphEdgeSnapshot("dinner-1", "dinner-2", 1.0),
        ),
    )

    topics = build_topics(snapshot, RelatednessConfig())
    affinities = dict(message_topic_affinities(
        (("WORD:python", 1.0),),
        topics,
    ))

    python_topic = next(
        topic.topic_id
        for topic in topics.topics
        if "python-1" in dict(topic.members)
    )
    dinner_topic = next(
        topic.topic_id
        for topic in topics.topics
        if "dinner-1" in dict(topic.members)
    )
    assert affinities[python_topic] == pytest.approx(1.0)
    assert affinities[dinner_topic] == 0.0


def test_topic_refresh_records_split_inheritance() -> None:
    nodes = tuple(_node(f"n{index}", "shared") for index in range(6))
    previous_edges = tuple(
        GraphEdgeSnapshot(source.msg_id, target.msg_id, 1.0)
        for index, source in enumerate(nodes)
        for target in nodes[index + 1:]
    )
    previous = build_topics(
        GraphSnapshot(version=7, nodes=nodes, edges=previous_edges),
        RelatednessConfig(),
    )
    current_edges = tuple(
        GraphEdgeSnapshot(nodes[source].msg_id, nodes[target].msg_id, 1.0)
        for group in ((0, 1, 2), (3, 4, 5))
        for offset, source in enumerate(group)
        for target in group[offset + 1:]
    )

    current = build_topics(
        GraphSnapshot(version=8, nodes=nodes, edges=current_edges),
        RelatednessConfig(),
        previous=previous,
    )

    assert len(previous.topics) == 1
    assert len(current.topics) == 2
    inherited_targets = {
        transition.target_topic_id
        for transition in current.transitions
        if transition.source_topic_id == previous.topics[0].topic_id
    }
    assert inherited_targets == {topic.topic_id for topic in current.topics}
    assert sum(
        transition.weight
        for transition in current.transitions
        if transition.source_topic_id == previous.topics[0].topic_id
    ) == pytest.approx(1.0)


def test_topic_refresh_records_merge_inheritance() -> None:
    nodes = tuple(_node(f"n{index}", "shared") for index in range(6))
    previous_edges = tuple(
        GraphEdgeSnapshot(nodes[source].msg_id, nodes[target].msg_id, 1.0)
        for group in ((0, 1, 2), (3, 4, 5))
        for offset, source in enumerate(group)
        for target in group[offset + 1:]
    )
    previous = build_topics(
        GraphSnapshot(version=9, nodes=nodes, edges=previous_edges),
        RelatednessConfig(),
    )
    current_edges = tuple(
        GraphEdgeSnapshot(source.msg_id, target.msg_id, 1.0)
        for index, source in enumerate(nodes)
        for target in nodes[index + 1:]
    )

    current = build_topics(
        GraphSnapshot(version=10, nodes=nodes, edges=current_edges),
        RelatednessConfig(),
        previous=previous,
    )

    assert len(previous.topics) == 2
    assert len(current.topics) == 1
    assert {
        transition.source_topic_id for transition in current.transitions
    } == {topic.topic_id for topic in previous.topics}
    assert {
        transition.target_topic_id for transition in current.transitions
    } == {current.topics[0].topic_id}


def test_new_message_is_compensated_against_current_topics() -> None:
    nodes = (
        _node("python-1", "python"),
        _node("python-2", "python"),
        _node("dinner-1", "dinner"),
        _node("dinner-2", "dinner"),
    )
    topics = build_topics(
        GraphSnapshot(
            version=4,
            nodes=nodes,
            edges=(
                GraphEdgeSnapshot("python-1", "python-2", 1.0),
                GraphEdgeSnapshot("dinner-1", "dinner-2", 1.0),
            ),
        ),
        RelatednessConfig(),
    )

    compensated = apply_topic_compensation(
        topics,
        message_id="python-new",
        timestamp=10.0,
        graph_version=5,
        vector_weights=(("WORD:python", 1.0),),
        config=RelatednessConfig(),
    )

    item = compensated.compensations[0]
    assert compensated.source_version == 4
    assert compensated.live_version == 5
    assert item.message_id == "python-new"
    assert item.memberships[0][1] == pytest.approx(1.0)
    assert compensated.activity[0].score > 0.0


def test_unrelated_compensation_does_not_create_topic_activity() -> None:
    topics = build_topics(
        GraphSnapshot(
            version=2,
            nodes=(_node("python-1", "python"), _node("python-2", "python")),
            edges=(GraphEdgeSnapshot("python-1", "python-2", 1.0),),
        ),
        RelatednessConfig(),
    )

    compensated = apply_topic_compensation(
        topics,
        message_id="unrelated",
        timestamp=10.0,
        graph_version=3,
        vector_weights=(("WORD:dinner", 1.0),),
        config=RelatednessConfig(),
    )

    assert compensated.compensations[0].memberships == ()
    assert compensated.activity == ()


def test_recompensation_replaces_message_instead_of_double_counting() -> None:
    topics = build_topics(
        GraphSnapshot(
            version=2,
            nodes=(_node("python-1", "python"), _node("python-2", "python")),
            edges=(GraphEdgeSnapshot("python-1", "python-2", 1.0),),
        ),
        RelatednessConfig(),
    )
    first = apply_topic_compensation(
        topics,
        message_id="pending-image",
        timestamp=10.0,
        graph_version=3,
        vector_weights=(("WORD:unknown", 1.0),),
        config=RelatednessConfig(),
    )

    enriched = apply_topic_compensation(
        first,
        message_id="pending-image",
        timestamp=10.0,
        graph_version=4,
        vector_weights=(("WORD:python", 1.0),),
        config=RelatednessConfig(),
    )

    assert len(enriched.compensations) == 1
    assert enriched.compensations[0].memberships
    assert enriched.activity[0].message_count == 1


def test_recluster_inherits_activity_and_clears_compensation() -> None:
    nodes = tuple(_node(f"n{index}", "python") for index in range(4))
    edges = tuple(
        GraphEdgeSnapshot(source.msg_id, target.msg_id, 1.0)
        for index, source in enumerate(nodes)
        for target in nodes[index + 1:]
    )
    previous = build_topics(
        GraphSnapshot(version=4, nodes=nodes, edges=edges),
        RelatednessConfig(),
    )
    previous = apply_topic_compensation(
        previous,
        message_id="new",
        timestamp=10.0,
        graph_version=5,
        vector_weights=(("WORD:python", 1.0),),
        config=RelatednessConfig(),
    )
    refreshed_nodes = (*nodes, _node("new", "python"))
    refreshed_edges = tuple(
        GraphEdgeSnapshot(source.msg_id, target.msg_id, 1.0)
        for index, source in enumerate(refreshed_nodes)
        for target in refreshed_nodes[index + 1:]
    )

    refreshed = build_topics(
        GraphSnapshot(version=5, nodes=refreshed_nodes, edges=refreshed_edges),
        RelatednessConfig(),
        previous=previous,
    )

    assert refreshed.compensations == ()
    assert refreshed.activity
    assert refreshed.activity[0].score == pytest.approx(
        previous.activity[0].score * 0.30
    )
