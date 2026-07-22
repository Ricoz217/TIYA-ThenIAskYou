import pytest

from TIYA.relatedness import analytics as analytics_module
from TIYA.relatedness.analytics import build_analytics
from TIYA.relatedness.models import (
    AnalyticsKind,
    GraphSnapshot,
    MessageSnapshot,
    RelatednessConfig,
    TopicSnapshot,
    GraphEdgeSnapshot,
)


def test_hotwords_are_built_without_legacy_dynamic_terms() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index % 3}",
            timestamp=float(index),
            text=f"牢大挑战今天继续 版本{index}",
            lexical_terms=(
                "牢大挑战" if index < 4 else "晚饭做菜",
                "今天",
                "继续",
            ),
        )
        for index in range(8)
    )
    snapshot = GraphSnapshot(version=4, nodes=nodes, edges=())

    result = build_analytics(
        snapshot,
        TopicSnapshot.empty(version=4),
        RelatednessConfig(new_word_min_frequency=3, new_word_min_users=2),
        AnalyticsKind.ALL,
    )

    assert result.source_version == 4
    assert result.hotwords
    assert result.dynamic_terms == ()
    assert result.hot_sentences


def test_empty_snapshot_returns_empty_analytics() -> None:
    snapshot = GraphSnapshot(version=0, nodes=(), edges=())

    result = build_analytics(
        snapshot,
        TopicSnapshot.empty(),
        RelatednessConfig(),
        AnalyticsKind.ALL,
    )

    assert result.hotwords == ()
    assert result.hot_sentences == ()
    assert result.dynamic_terms == ()


def test_hotwords_keep_nouns_and_verbs_and_prefer_nouns() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index),
            text=f"碗橱倒灌公共{index}",
            lexical_terms=("碗橱", "倒灌", "公共"),
            lexical_pos=(("碗橱", "n"), ("倒灌", "v"), ("公共", "ad")),
        )
        for index in range(3)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
        AnalyticsKind.HOTWORDS,
    )
    scores = {item.term: item.score for item in result.hotwords}

    assert "公共" not in scores
    assert scores["倒灌"] == pytest.approx(scores["碗橱"] * 0.6)


def test_pure_media_description_can_become_hot_sentence() -> None:
    snapshot = GraphSnapshot(
        version=1,
        nodes=(
            MessageSnapshot(
                msg_id="image",
                user_id="user",
                timestamp=1.0,
                text="",
                semantic_text="一张表达震惊的猫猫表情包",
                lexical_terms=("震惊", "猫猫", "表情包"),
            ),
        ),
        edges=(),
    )

    result = build_analytics(
        snapshot,
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert result.hot_sentences[0].message_id == "image"
    assert result.hot_sentences[0].text == "一张表达震惊的猫猫表情包"


def test_dynamic_stopword_filters_cross_user_background_term() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index),
            text=f"通用词 话题{index}",
            lexical_terms=("通用词", f"话题{index}"),
            primary_lexical_terms=("通用词", f"话题{index}"),
        )
        for index in range(10)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert "通用词" in result.dynamic_stopwords
    assert all(item.term != "通用词" for item in result.hotwords)


def test_dynamic_stopword_aggressively_filters_ten_percent_background_term() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index),
            text=f"背景词 独有词{index}" if index < 10 else f"独有词{index}",
            lexical_terms=(
                ("背景词", f"独有词{index}")
                if index < 10
                else (f"独有词{index}",)
            ),
            primary_lexical_terms=(
                ("背景词", f"独有词{index}")
                if index < 10
                else (f"独有词{index}",)
            ),
        )
        for index in range(100)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert "背景词" in result.dynamic_stopwords


def test_dynamic_stopword_finds_evenly_distributed_low_frequency_background() -> None:
    background_indexes = {0, 20, 40, 60, 80}
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index * 60),
            text=f"消息{index}",
            lexical_terms=(
                ("长期背景", f"独有词{index}")
                if index in background_indexes
                else (f"独有词{index}",)
            ),
            primary_lexical_terms=(
                ("长期背景", f"独有词{index}")
                if index in background_indexes
                else (f"独有词{index}",)
            ),
        )
        for index in range(100)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert "长期背景" in result.dynamic_stopwords


def test_dynamic_stopword_keeps_short_lived_topic_burst() -> None:
    burst_indexes = set(range(40, 45))
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index * 60),
            text=f"消息{index}",
            lexical_terms=(
                ("突发话题", f"独有词{index}")
                if index in burst_indexes
                else (f"独有词{index}",)
            ),
            primary_lexical_terms=(
                ("突发话题", f"独有词{index}")
                if index in burst_indexes
                else (f"独有词{index}",)
            ),
        )
        for index in range(100)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert "突发话题" not in result.dynamic_stopwords


def test_dynamic_stopword_uses_full_message_window_before_content_dedup() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index),
            text="重复模板",
            lexical_terms=("重复模板",),
            primary_lexical_terms=("重复模板",),
        )
        for index in range(10)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert "重复模板" in result.dynamic_stopwords


def test_dynamic_stopword_filters_low_frequency_term_spread_across_quarter_users() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index % 24}",
            timestamp=float(index),
            text=f"消息{index}",
            lexical_terms=(
                ("跨用户背景", f"独有词{index}")
                if index < 6
                else (f"独有词{index}",)
            ),
            primary_lexical_terms=(
                ("跨用户背景", f"独有词{index}")
                if index < 6
                else (f"独有词{index}",)
            ),
        )
        for index in range(100)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert "跨用户背景" in result.dynamic_stopwords


def test_repeated_media_is_deduplicated_for_hotwords_and_sentences() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index),
            text="",
            semantic_text="注册页面 密码提示",
            media_ids=frozenset({"image:same"}),
            lexical_terms=("注册页面", "密码", "提示"),
            semantic_lexical_terms=("注册页面", "密码", "提示"),
        )
        for index in range(5)
    )
    edges = tuple(
        GraphEdgeSnapshot(str(index), str(index + 1), 0.8)
        for index in range(4)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=edges),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert all(item.term != "密码" for item in result.hotwords)
    assert len(result.hot_sentences) == 1


def test_graph_central_sentence_beats_long_isolated_sentence() -> None:
    nodes = (
        MessageSnapshot(
            msg_id="center",
            user_id="a",
            timestamp=2.0,
            text="核心 讨论 方案",
            lexical_terms=("核心", "讨论", "方案"),
            primary_lexical_terms=("核心", "讨论", "方案"),
        ),
        MessageSnapshot(
            msg_id="neighbor-one",
            user_id="b",
            timestamp=1.0,
            text="讨论 方案 细节",
            lexical_terms=("讨论", "方案", "细节"),
            primary_lexical_terms=("讨论", "方案", "细节"),
        ),
        MessageSnapshot(
            msg_id="neighbor-two",
            user_id="c",
            timestamp=3.0,
            text="核心 方案 继续",
            lexical_terms=("核心", "方案", "继续"),
            primary_lexical_terms=("核心", "方案", "继续"),
        ),
        MessageSnapshot(
            msg_id="long",
            user_id="d",
            timestamp=4.0,
            text="这是一条非常非常长但是与其他消息没有任何关系的孤立句子",
            lexical_terms=("孤立", "句子", "很长"),
            primary_lexical_terms=("孤立", "句子", "很长"),
        ),
    )
    edges = (
        GraphEdgeSnapshot("center", "neighbor-one", 0.7),
        GraphEdgeSnapshot("center", "neighbor-two", 0.7),
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=edges),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert result.hot_sentences[0].message_id == "center"


def test_primary_chat_terms_rank_above_image_description_layout_words() -> None:
    nodes = (
        MessageSnapshot(
            msg_id="text-1",
            user_id="a",
            timestamp=1.0,
            text="密码 用户名 返回 第一种方案",
            lexical_terms=("密码", "用户名", "返回"),
            primary_lexical_terms=("密码", "用户名", "返回"),
        ),
        MessageSnapshot(
            msg_id="text-2",
            user_id="b",
            timestamp=2.0,
            text="密码 用户名 返回 第二种方案",
            lexical_terms=("密码", "用户名", "返回"),
            primary_lexical_terms=("密码", "用户名", "返回"),
        ),
        MessageSnapshot(
            msg_id="image-1",
            user_id="c",
            timestamp=3.0,
            text="",
            semantic_text="蓝色 按钮 下方",
            media_ids=frozenset({"image:one"}),
            lexical_terms=("蓝色", "按钮", "下方"),
            semantic_lexical_terms=("蓝色", "按钮", "下方"),
        ),
        MessageSnapshot(
            msg_id="image-2",
            user_id="d",
            timestamp=4.0,
            text="",
            semantic_text="蓝色 按钮 下方",
            media_ids=frozenset({"image:two"}),
            lexical_terms=("蓝色", "按钮", "下方"),
            semantic_lexical_terms=("蓝色", "按钮", "下方"),
        ),
        MessageSnapshot(
            msg_id="image-3",
            user_id="e",
            timestamp=5.0,
            text="",
            semantic_text="蓝色 按钮 下方",
            media_ids=frozenset({"image:three"}),
            lexical_terms=("蓝色", "按钮", "下方"),
            semantic_lexical_terms=("蓝色", "按钮", "下方"),
        ),
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    positions = {item.term: index for index, item in enumerate(result.hotwords)}
    assert positions["用户名"] < positions["蓝色"]


def test_hotword_shape_filter_rejects_numeric_and_short_ascii_noise() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index),
            text=f"咖啡 33 87 a xy API CNY33 消息{index}",
            lexical_terms=("咖啡", "33", "87", "a", "xy", "api", "cny33"),
            primary_lexical_terms=(
                "咖啡",
                "33",
                "87",
                "a",
                "xy",
                "api",
                "cny33",
            ),
        )
        for index in range(2)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(dynamic_stopword_min_documents=100),
    )

    terms = {item.term for item in result.hotwords}
    assert {"33", "87", "a", "xy"}.isdisjoint(terms)
    assert {"咖啡", "api", "cny33"} <= terms


def test_hotword_shape_noise_is_not_learned_as_dynamic_stopword() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index),
            text=f"33 87 正常词 消息{index}",
            lexical_terms=("33", "87", "正常词"),
            primary_lexical_terms=("33", "87", "正常词"),
        )
        for index in range(10)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
    )

    assert "正常词" in result.dynamic_stopwords
    assert {"33", "87"}.isdisjoint(result.dynamic_stopwords)


def test_semantic_only_ascii_terms_are_excluded_from_hotwords() -> None:
    nodes = tuple(
        MessageSnapshot(
            msg_id=str(index),
            user_id=f"u{index}",
            timestamp=float(index),
            text="",
            semantic_text="用户名 moon rider 蓝色按钮",
            media_ids=frozenset({f"image:{index}"}),
            lexical_terms=("moon", "rider", "蓝色", "按钮"),
            semantic_lexical_terms=("moon", "rider", "蓝色", "按钮"),
        )
        for index in range(2)
    )

    result = build_analytics(
        GraphSnapshot(version=1, nodes=nodes, edges=()),
        TopicSnapshot.empty(version=1),
        RelatednessConfig(dynamic_stopword_min_documents=100),
    )

    terms = {item.term for item in result.hotwords}
    assert {"moon", "rider"}.isdisjoint(terms)
    assert {"蓝色", "按钮"} <= terms


def test_new_word_refresh_does_not_run_hot_sentence_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = GraphSnapshot(
        version=1,
        nodes=(MessageSnapshot("1", "u", 1.0, "python", ("python",)),),
        edges=(),
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("hot-sentence analysis must be lazy")

    monkeypatch.setattr(analytics_module, "_sentence_scores", fail)
    build_analytics(
        snapshot,
        TopicSnapshot.empty(version=1),
        RelatednessConfig(),
        AnalyticsKind.NEW_WORDS,
    )
