import pytest

from TIYA.relatedness.graph import MessageGraph
from TIYA.relatedness.models import MessageInput, RelatednessConfig, TextFeatures


def _features(*terms: str) -> TextFeatures:
    weights = tuple(1.0 for _ in terms)
    return TextFeatures(
        normalized_text=" ".join(terms),
        lexical_terms=terms,
        lexical_weights=weights,
        subword_terms=(),
        subword_weights=(),
    )


def _message(
    msg_id: str,
    *,
    user: str = "u",
    timestamp: float = 1.0,
    reply_to: str | None = None,
    mentions: frozenset[str] = frozenset(),
    media_ids: frozenset[str] = frozenset(),
) -> MessageInput:
    return MessageInput(
        msg_id=msg_id,
        group_id="g",
        user_id=user,
        timestamp=timestamp,
        text=msg_id,
        reply_to=reply_to,
        mention_ids=mentions,
        media_ids=media_ids,
    )


def test_reply_is_stronger_than_plain_text_and_propagates() -> None:
    graph = MessageGraph(RelatednessConfig(enable_relation_propagation=True))
    graph.add(_message("root", user="a"), _features("topic"))
    graph.add(
        _message("reply", user="b", timestamp=2.0, reply_to="root"),
        _features("different"),
    )
    graph.add(
        _message("tail", user="c", timestamp=3.0, reply_to="reply"),
        _features("else"),
    )

    reply_score = graph.score("reply", "root")
    propagated = graph.related("tail", limit=10, explain=True)

    assert reply_score is not None
    assert reply_score.reply > reply_score.text
    assert reply_score.final < 1.0
    assert any(match.message_id == "root" and match.score > 0 for match in propagated)


def test_mention_links_to_recent_messages_from_mentioned_user() -> None:
    graph = MessageGraph(RelatednessConfig())
    graph.add(_message("target", user="alice"), _features("one"))
    graph.add(
        _message(
            "mention",
            user="bob",
            timestamp=2.0,
            mentions=frozenset({"alice"}),
        ),
        _features("two"),
    )

    score = graph.score("mention", "target")

    assert score is not None
    assert score.mention > 0


def test_window_evicts_nodes_and_edges() -> None:
    graph = MessageGraph(RelatednessConfig(message_window=2))
    graph.add(_message("1"), _features("a"))
    graph.add(_message("2", timestamp=2), _features("b"))
    graph.add(_message("3", timestamp=3), _features("c"))

    assert not graph.contains("1")
    assert graph.score("2", "1") is None
    assert graph.node_ids == ("2", "3")


def test_raw_edge_weight_adds_all_evidence_before_result_normalization() -> None:
    config = RelatednessConfig(reply_direct_factor=1.0)
    graph = MessageGraph(config)
    graph.add(_message("root", user="same"), _features("topic"))
    graph.add(
        _message(
            "reply",
            user="same",
            timestamp=2.0,
            reply_to="root",
            mentions=frozenset({"same"}),
        ),
        _features("topic"),
        (("root", 1.0),),
    )

    edge = next(iter(graph.snapshot().edges))
    score = graph.score("reply", "root")

    assert edge.weight == pytest.approx(
        edge.text
        + edge.reply
        + edge.mention
        + edge.context
        + edge.same_user
        + edge.time
    )
    assert edge.weight > 1.0
    assert score is not None
    assert score.final == 1.0


def test_context_and_time_alone_do_not_become_related_result() -> None:
    graph = MessageGraph(RelatednessConfig())
    graph.add(_message("1", user="a"), _features("first"))
    graph.add(
        _message("2", user="b", timestamp=2.0),
        _features("unrelated"),
    )

    related = graph.related("2", explain=True)

    assert all(match.message_id != "1" for match in related)


def test_bm25_score_only_selects_candidate_not_text_edge_weight() -> None:
    config = RelatednessConfig(text_weight=0.55)
    graph = MessageGraph(config)
    graph.add(_message("first", user="a"), _features("same", "topic"))
    graph.add(
        _message("second", user="b", timestamp=2.0),
        _features("same", "topic"),
        (("first", 0.01),),
    )

    score = graph.score("second", "first")

    assert score is not None
    assert score.text == pytest.approx(config.text_weight)


def test_reply_edge_stays_strong_without_forcing_final_score_to_point_eight() -> None:
    graph = MessageGraph(RelatednessConfig())
    graph.add(_message("root", user="a"), _features("root"))
    graph.add(
        _message("reply", user="b", timestamp=2.0, reply_to="root"),
        _features("unrelated"),
    )

    direct = graph.score("reply", "root")
    match = next(
        item
        for item in graph.related("reply", explain=True)
        if item.message_id == "root"
    )

    assert direct is not None
    assert direct.reply == 0.8
    assert direct.final < 0.7
    assert match.score < 0.75


def test_propagation_can_produce_a_meaningful_two_hop_score() -> None:
    graph = MessageGraph(RelatednessConfig(enable_relation_propagation=True))
    graph.add(_message("root", user="a"), _features("root"))
    graph.add(
        _message("middle", user="b", timestamp=2.0, reply_to="root"),
        _features("middle"),
    )
    graph.add(
        _message("tail", user="c", timestamp=3.0, reply_to="middle"),
        _features("tail"),
    )

    match = next(
        item
        for item in graph.related("tail", explain=True)
        if item.message_id == "root"
    )

    assert match.details is not None
    assert match.details.propagation > 0.25
    assert match.details.propagation < 0.7
    assert match.score > match.details.propagation


def test_propagation_adds_to_direct_evidence_at_result_stage() -> None:
    graph = MessageGraph(RelatednessConfig(enable_relation_propagation=True))
    graph.add(
        _message("root", user="a"),
        _features("shared"),
    )
    graph.add(
        _message("middle", user="b", timestamp=2.0, reply_to="root"),
        _features("middle"),
    )
    graph.add(
        _message(
            "tail",
            user="c",
            timestamp=3.0,
            reply_to="middle",
        ),
        _features("shared"),
        (("root", 1.0),),
    )

    direct = graph.score("tail", "root")
    match = next(
        item
        for item in graph.related("tail", explain=True)
        if item.message_id == "root"
    )

    assert direct is not None
    assert match.details is not None
    assert match.details.propagation > 0
    assert match.score > direct.final
    assert match.score <= 1.0


def test_enrichment_replaces_stale_text_edge() -> None:
    graph = MessageGraph(RelatednessConfig())
    graph.add(_message("first", user="a"), _features("MEDIA:image:same"))
    second = _message("second", user="b", timestamp=2.0)
    graph.add(
        second,
        _features("MEDIA:image:same"),
        (("first", 1.0),),
    )

    before = graph.score("second", "first")
    changed = graph.enrich(second, _features("猫猫表情包"), ())
    after = graph.score("second", "first")

    assert changed
    assert before is not None and before.text > 0
    assert after is not None and after.text == 0


def test_support_only_direct_edge_does_not_amplify_weak_propagation() -> None:
    graph = MessageGraph(RelatednessConfig())
    graph.add(_message("first", user="a"), _features("first"))
    graph.add(
        _message("second", user="b", timestamp=2.0),
        _features("second"),
    )
    graph.add(
        _message("third", user="c", timestamp=3.0),
        _features("third"),
    )

    related_ids = {
        item.message_id
        for item in graph.related("third", explain=True)
    }
    support_only = graph.score("third", "second")

    assert support_only is not None
    assert 0 < support_only.final < graph.config.related_min_score
    assert "first" not in related_ids
    assert "second" not in related_ids


def test_anchor_evidence_reads_multiple_targets_with_one_propagation() -> None:
    graph = MessageGraph(RelatednessConfig(enable_relation_propagation=True))
    graph.add(_message("root", user="a"), _features("root"))
    graph.add(
        _message("middle", user="b", timestamp=2.0, reply_to="root"),
        _features("middle"),
    )
    graph.add(
        _message("tail", user="c", timestamp=3.0, reply_to="middle"),
        _features("tail"),
    )

    snapshot = graph.anchor_evidence("tail", ("root", "middle", "missing"))

    assert snapshot is not None
    assert snapshot.source_id == "tail"
    assert [item.message_id for item in snapshot.anchors] == ["root", "middle"]
    assert snapshot.anchors[0].relation.propagation > 0
    assert snapshot.anchors[0].message_gap == 1
    assert snapshot.anchors[1].relation.reply == 0.8


def test_relation_propagation_is_disabled_by_default() -> None:
    graph = MessageGraph(RelatednessConfig())
    graph.add(_message("root", user="a"), _features("root"))
    graph.add(
        _message("middle", user="b", timestamp=2.0, reply_to="root"),
        _features("middle"),
    )
    graph.add(
        _message("tail", user="c", timestamp=3.0, reply_to="middle"),
        _features("tail"),
    )

    evidence = graph.anchor_evidence("tail", ("root",))

    assert evidence is not None
    assert evidence.anchors[0].relation.propagation == 0.0


def test_local_snapshot_is_bounded_and_never_reads_future_messages() -> None:
    graph = MessageGraph(RelatednessConfig())
    for index in range(6):
        graph.add(
            _message(str(index), user=f"u{index}", timestamp=float(index)),
            _features(f"term-{index}"),
        )

    snapshot = graph.local_snapshot(
        "4",
        window_seconds=3.0,
        message_limit=2,
    )

    assert [node.msg_id for node in snapshot.nodes] == ["3", "4"]
    assert all(
        edge.source_id in {"3", "4"} and edge.target_id in {"3", "4"}
        for edge in snapshot.edges
    )
