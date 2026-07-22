from dataclasses import replace

import pytest

from TIYA.relatedness.continuity import DialogueChainModel, continuity_time_penalty
from TIYA.relatedness.models import (
    DialogueChain,
    DialogueChainNode,
    GraphEdgeSnapshot,
    GraphSnapshot,
    MemberCommunityConfig,
    MessageSnapshot,
    RelatednessConfig,
)


def _node(
    msg_id: str,
    user_id: str,
    timestamp: float,
    text: str = "text",
    *,
    terms: tuple[tuple[str, float], ...] = (),
    reply_to: str | None = None,
    mentions: frozenset[str] = frozenset(),
) -> MessageSnapshot:
    return MessageSnapshot(
        msg_id=msg_id,
        user_id=user_id,
        timestamp=timestamp,
        text=text,
        vector_weights=terms,
        reply_to=reply_to,
        mention_ids=mentions,
    )


def _chain(
    chain_id: str,
    *nodes: tuple[str, float, float],
    active: bool = True,
    progress: int = 0,
    low_streak: int = 0,
) -> DialogueChain:
    return DialogueChain(
        chain_id=chain_id,
        active=active,
        nodes=tuple(
            DialogueChainNode(message_id, weight, timestamp)
            for message_id, weight, timestamp in nodes
        ),
        progress=progress,
        low_streak=low_streak,
    )


def _model(
    *chains: DialogueChain,
    capacity: int = 20,
    decay: float = 1.0,
) -> DialogueChainModel:
    return DialogueChainModel(
        MemberCommunityConfig(
            continuity_chain_capacity=capacity,
            continuity_propagation_decay=decay,
        ),
        RelatednessConfig(),
        chains,
    )


def _snapshot(
    nodes: tuple[MessageSnapshot, ...],
    edges: tuple[GraphEdgeSnapshot, ...] = (),
) -> GraphSnapshot:
    sequenced = tuple(
        replace(node, sequence=index) if node.sequence < 0 else node
        for index, node in enumerate(nodes)
    )
    return GraphSnapshot(version=1, nodes=sequenced, edges=edges)


def test_time_penalty_is_monotonic_and_has_a_hard_window() -> None:
    assert continuity_time_penalty(0.0) == 1.0
    assert continuity_time_penalty(60.0) < 1.0
    assert continuity_time_penalty(300.0) == pytest.approx(0.8)
    assert 0.0 < continuity_time_penalty(600.0) < 0.8
    assert continuity_time_penalty(900.0) == 0.0
    assert continuity_time_penalty(901.0) == 0.0


def test_chain_contribution_uses_candidate_weighted_mean() -> None:
    model = _model(
        _chain("active", ("one", 1.0, 1.0), ("two", 0.5, 1.0)),
    )
    snapshot = _snapshot(
        (
            _node("one", "bot", 1.0),
            _node("two", "alice", 1.0),
            _node("source", "bob", 1.0),
        ),
        (
            GraphEdgeSnapshot("one", "source", 0.8),
            GraphEdgeSnapshot("two", "source", 0.4),
        ),
    )

    result = model.process(snapshot, source_id="source", member_id="bot")

    assert result.score == pytest.approx((0.8**2 + 0.2**2) / (0.8 + 0.2))
    assert model.active_chain is not None
    assert model.active_chain.nodes[-1].weight == pytest.approx(result.score)


def test_existing_group_edge_uses_its_complete_raw_weight() -> None:
    model = _model(_chain("active", ("seed", 1.0, 1.0)))
    snapshot = _snapshot(
        (_node("seed", "bot", 1.0), _node("source", "alice", 1.0)),
        (GraphEdgeSnapshot("seed", "source", 0.8, text=0.1),),
    )

    result = model.process(snapshot, source_id="source", member_id="bot")

    assert result.score == pytest.approx(0.8)
    assert result.chains[0].candidates[0].relation_source == "group_edge"
    assert result.chains[0].candidates[0].base_similarity == pytest.approx(0.8)


def test_missing_edge_uses_cached_vectors_without_mutating_snapshot() -> None:
    model = _model(_chain("active", ("seed", 1.0, 1.0)))
    snapshot = _snapshot((
        _node("seed", "bot", 1.0, terms=(("WORD:topic", 1.0),)),
        _node("source", "alice", 1.0, terms=(("WORD:topic", 1.0),)),
    ))

    result = model.process(snapshot, source_id="source", member_id="bot")

    assert result.score == pytest.approx(0.55)
    assert result.chains[0].candidates[0].relation_source == "cached_cosine"
    assert snapshot.edges == ()
    assert snapshot.version == 1


def test_relation_below_threshold_does_not_propagate_or_join() -> None:
    model = _model(_chain("active", ("seed", 1.0, 1.0)))
    snapshot = _snapshot(
        (_node("seed", "bot", 1.0), _node("source", "alice", 1.0)),
        (GraphEdgeSnapshot("seed", "source", 0.149),),
    )

    result = model.process(snapshot, source_id="source", member_id="bot")

    assert result.score == 0.0
    assert model.active_chain is not None
    assert [item.message_id for item in model.active_chain.nodes] == ["seed"]


def test_reply_builds_a_filtered_chain_and_can_immediately_become_static() -> None:
    model = _model(
        _chain("old", ("old-seed", 1.0, 1.0)),
        capacity=3,
    )
    snapshot = _snapshot(
        (
            _node("old-seed", "bot", 1.0),
            _node("target", "bot", 2.0),
            _node("valid", "alice", 3.0),
            _node("invalid", "bob", 4.0),
            _node("source", "carol", 5.0, reply_to="target"),
        ),
        (GraphEdgeSnapshot("target", "valid", 0.5),),
    )

    result = model.process(snapshot, source_id="source", member_id="bot")

    new_chain = next(chain for chain in model.chains if chain.chain_id == "source")
    assert [node.message_id for node in new_chain.nodes] == [
        "target",
        "valid",
        "source",
    ]
    assert new_chain.active is False
    assert new_chain.progress == 3
    assert model.active_chain is None
    assert all(chain.active is False for chain in model.chains)
    assert result.score == 1.0


def test_reply_initialization_scans_the_full_range_without_reserving_capacity() -> None:
    model = _model(capacity=2)
    snapshot = _snapshot(
        (
            _node("target", "bot", 1.0),
            _node("middle", "alice", 2.0),
            _node("source", "bob", 3.0, reply_to="target"),
        ),
        (GraphEdgeSnapshot("target", "middle", 0.5),),
    )

    model.process(snapshot, source_id="source", member_id="bot")

    chain = model.chains[0]
    assert [node.message_id for node in chain.nodes] == [
        "target",
        "middle",
        "source",
    ]
    assert chain.active is False
    assert chain.progress == 2


def test_mention_and_member_output_are_seed_nodes_without_trigger_link() -> None:
    model = _model()
    mention = _snapshot((
        _node("mention", "alice", 1.0, mentions=frozenset({"bot"})),
    ))

    mention_result = model.process(mention, source_id="mention", member_id="bot")
    output = _snapshot((
        _node("mention", "alice", 1.0, mentions=frozenset({"bot"})),
        _node("output", "bot", 2.0),
    ))
    model.record_member_output(output, source_id="output", member_id="bot")

    assert mention_result.score == 1.0
    assert model.active_chain is not None
    assert [(node.message_id, node.weight) for node in model.active_chain.nodes] == [
        ("mention", 1.0),
        ("output", 1.0),
    ]


def test_static_chains_score_but_never_receive_normal_messages() -> None:
    model = _model(
        _chain("static", ("seed", 1.0, 1.0), active=False, progress=20),
    )
    snapshot = _snapshot(
        (_node("seed", "bot", 1.0), _node("source", "alice", 2.0)),
        (GraphEdgeSnapshot("seed", "source", 0.5),),
    )

    result = model.process(snapshot, source_id="source", member_id="bot")

    assert result.score > 0.0
    assert [node.message_id for node in model.chains[0].nodes] == ["seed"]
    assert model.chains[0].progress == 21
    assert model.active_chain is None


def test_final_continuity_is_the_maximum_across_chains() -> None:
    model = _model(
        _chain("active", ("active-seed", 1.0, 1.0)),
        _chain("static", ("static-seed", 1.0, 1.0), active=False, progress=20),
    )
    snapshot = _snapshot(
        (
            _node("active-seed", "bot", 1.0),
            _node("static-seed", "bot", 1.0),
            _node("source", "alice", 1.0),
        ),
        (
            GraphEdgeSnapshot("active-seed", "source", 0.3),
            GraphEdgeSnapshot("static-seed", "source", 0.7),
        ),
    )

    result = model.process(snapshot, source_id="source", member_id="bot")

    assert result.score == pytest.approx(0.7)
    assert result.winning_chain_id == "static"


def test_capacity_progress_and_low_streak_are_derived_from_x() -> None:
    model = _model(
        _chain("active", ("seed", 1.0, 1.0)),
        capacity=2,
    )
    valid = _snapshot(
        (_node("seed", "bot", 1.0), _node("valid", "alice", 2.0)),
        (GraphEdgeSnapshot("seed", "valid", 0.5),),
    )

    model.process(valid, source_id="valid", member_id="bot")

    assert model.active_chain is None
    assert model.chains[0].progress == 2

    first_miss = _snapshot((_node("miss-1", "alice", 3.0),))
    model.process(first_miss, source_id="miss-1", member_id="bot")

    assert model.chains == ()  # ceil(0.5 * 2) == 1


def test_static_chain_expires_when_progress_reaches_twice_x() -> None:
    model = _model(
        _chain("static", ("seed", 1.0, 1.0), active=False, progress=3),
        capacity=2,
    )
    snapshot = _snapshot(
        (_node("seed", "bot", 1.0), _node("source", "alice", 2.0)),
        (GraphEdgeSnapshot("seed", "source", 0.5),),
    )

    model.process(snapshot, source_id="source", member_id="bot")

    assert model.chains == ()


def test_non_text_message_can_join_but_returns_zero() -> None:
    model = _model(_chain("active", ("seed", 1.0, 1.0)))
    snapshot = _snapshot(
        (
            _node("seed", "bot", 1.0),
            _node("source", "alice", 2.0, text=""),
        ),
        (GraphEdgeSnapshot("seed", "source", 0.8),),
    )

    result = model.process(snapshot, source_id="source", member_id="bot")

    assert result.score == 0.0
    assert result.path_strength > 0.0
    assert model.active_chain is not None
    assert model.active_chain.nodes[-1].message_id == "source"


def test_chain_expires_at_the_fifteen_minute_hard_limit() -> None:
    model = _model(_chain("active", ("seed", 1.0, 1.0)))
    snapshot = _snapshot((
        _node("seed", "bot", 1.0),
        _node("source", "alice", 901.0),
    ))

    result = model.process(snapshot, source_id="source", member_id="bot")

    assert result.score == 0.0
    assert model.chains == ()
