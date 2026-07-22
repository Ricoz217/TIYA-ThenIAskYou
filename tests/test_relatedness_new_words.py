import pytest

from TIYA.relatedness.new_words import (
    NewWordConfig,
    NewWordStatus,
    NewWordDocument,
    discover_new_words,
    format_new_word_report,
)


def _doc(
    index: int,
    text: str,
    *,
    user: str | None = None,
) -> NewWordDocument:
    return NewWordDocument(
        msg_id=str(index),
        user_id=user or f"user-{index % 3}",
        timestamp=float(index),
        text=text,
    )


def _compact_config(**overrides: object) -> NewWordConfig:
    values = {
        "min_frequency": 3,
        "min_documents": 3,
        "min_users": 2,
        "min_npmi": 0.05,
        "min_boundary_entropy": 0.10,
        "promoted_limit": 20,
        "candidate_limit": 30,
        "known_limit": 20,
        "rejected_sample_limit": 50,
        "enable_dynamic_stopwords": False,
    }
    values.update(overrides)
    return NewWordConfig(**values)


def test_discovers_word_that_jieba_splits() -> None:
    result = discover_new_words(
        (
            _doc(1, "今天蓝鲸泡泡很好玩", user="alice"),
            _doc(2, "我也喜欢蓝鲸泡泡这个梗", user="bob"),
            _doc(3, "他们在聊蓝鲸泡泡了吗", user="carol"),
            _doc(4, "蓝鲸泡泡真的离谱", user="alice"),
            _doc(5, "晚上继续蓝鲸泡泡", user="bob"),
        ),
        _compact_config(),
    )

    promoted = {item.term: item for item in result.promoted}

    assert "蓝鲸泡泡" in promoted
    assert promoted["蓝鲸泡泡"].user_count >= 2
    assert promoted["蓝鲸泡泡"].npmi > 0


def test_jieba_known_word_is_not_promoted() -> None:
    result = discover_new_words(
        (
            _doc(1, "数据库今天又炸了", user="alice"),
            _doc(2, "数据库索引需要优化", user="bob"),
            _doc(3, "我也在看数据库", user="carol"),
            _doc(4, "数据库连接池满了", user="alice"),
        ),
        _compact_config(),
    )

    promoted = {item.term for item in result.promoted}
    known = {item.term for item in result.known}

    assert "数据库" not in promoted
    assert "数据库" in known


def test_project_user_dictionary_marks_existing_group_terms_as_known() -> None:
    result = discover_new_words(
        (
            _doc(1, "今天来点色图", user="alice"),
            _doc(2, "群友又在说色图", user="bob"),
            _doc(3, "色图系统还要调", user="carol"),
            _doc(4, "色图这个词在用户词典里", user="dave"),
        ),
        _compact_config(),
    )

    promoted = {item.term for item in result.promoted}
    known = {item.term for item in result.known}

    assert "色图" not in promoted
    assert "色图" in known


def test_function_phrase_stop_filter_is_auditable() -> None:
    result = discover_new_words(
        (
            _doc(1, "也太离谱了都不想说", user="alice"),
            _doc(2, "也太奇怪了我也不懂", user="bob"),
            _doc(3, "也太抽象了都不知道", user="carol"),
            _doc(4, "我也觉得也太怪了", user="dave"),
        ),
        _compact_config(min_frequency=2, min_documents=2),
    )

    promoted = {item.term for item in result.promoted}
    rejected = {item.term: item for item in result.rejected}

    assert "也太" not in promoted
    assert "我也" not in promoted
    assert "也太" in rejected
    assert "function_phrase" in rejected["也太"].reject_reasons


def test_dynamic_stopword_filter_is_auditable() -> None:
    docs = tuple(
        _doc(
            index,
            f"背景词 今天话题{index}",
            user=f"user-{index % 8}",
        )
        for index in range(20)
    )

    result = discover_new_words(
        docs,
        _compact_config(
            enable_dynamic_stopwords=True,
            dynamic_stopword_min_documents=5,
            dynamic_stopword_doc_ratio=0.30,
            min_frequency=2,
            min_boundary_entropy=0.0,
        ),
    )
    rejected = {item.term: item for item in result.rejected}

    assert "背景词" in result.dynamic_stopwords
    assert "背景词" in rejected
    assert "dynamic_stopword" in rejected["背景词"].reject_reasons


def test_repeated_sentence_does_not_promote_many_fragments() -> None:
    repeated = tuple(
        _doc(index, "自律的感觉真好自律的感觉真好", user="same-user")
        for index in range(10)
    )

    result = discover_new_words(
        repeated,
        _compact_config(min_users=2),
    )

    assert result.promoted == ()
    assert any(
        "user_count" in reason
        for item in result.rejected
        for reason in item.reject_reasons
    )


def test_low_boundary_entropy_fragment_is_rejected() -> None:
    result = discover_new_words(
        tuple(
            _doc(index, "今天自律的感觉真好", user=f"user-{index}")
            for index in range(6)
        ),
        _compact_config(min_boundary_entropy=0.40),
    )

    assert "律的感" not in {item.term for item in result.promoted}
    rejected = {item.term: item for item in result.rejected}
    assert "律的感" in rejected
    assert any("boundary_entropy" in reason for reason in rejected["律的感"].reject_reasons)


def test_nested_fragments_are_suppressed_by_complete_candidate() -> None:
    result = discover_new_words(
        (
            _doc(1, "今天喵呜喵呜很好笑", user="alice"),
            _doc(2, "我收藏了喵呜喵呜表情", user="bob"),
            _doc(3, "怎么又是喵呜喵呜", user="carol"),
            _doc(4, "喵呜喵呜真的洗脑", user="alice"),
        ),
        _compact_config(
            max_chars=4,
            max_nested_coverage=0.60,
            min_promoted_score=0.0,
        ),
    )

    promoted = {item.term for item in result.promoted}

    assert "喵呜喵呜" in promoted
    assert "喵呜喵" not in promoted
    assert "呜喵呜" not in promoted


def test_multiple_users_and_documents_raise_priority() -> None:
    result = discover_new_words(
        (
            _doc(1, "大家都在说星流猫猫", user="alice"),
            _doc(2, "星流猫猫是什么新梗", user="bob"),
            _doc(3, "我看见星流猫猫了", user="carol"),
            _doc(4, "星流猫猫又来了", user="dave"),
            _doc(5, "另一个低频词", user="alice"),
        ),
        _compact_config(),
    )

    promoted_terms = [item.term for item in result.promoted]

    assert promoted_terms
    assert promoted_terms[0] == "星流猫猫"


def test_report_contains_metrics_and_reject_reasons() -> None:
    result = discover_new_words(
        (
            _doc(1, "今天蓝鲸泡泡很好玩", user="alice"),
            _doc(2, "我也喜欢蓝鲸泡泡这个梗", user="bob"),
            _doc(3, "他们在聊蓝鲸泡泡了吗", user="carol"),
            _doc(4, "今天自律的感觉真好", user="dave"),
        ),
        _compact_config(min_frequency=2, min_documents=2),
    )

    report = format_new_word_report(result)

    assert "PROMOTED" in report
    assert "REJECTED" in report
    assert "npmi" in report
    assert "boundary" in report
    assert "score_parts" in report
    assert "normalized_parts" in report
    assert "reject reasons" in report


def test_score_weights_are_auditable_and_adjustable() -> None:
    docs = (
        _doc(1, "\u661f\u6d41\u732b\u732b\u4eca\u5929\u5f88\u70ed\u95f9", user="alice"),
        _doc(2, "\u6211\u4e5f\u770b\u5230\u661f\u6d41\u732b\u732b\u8fd9\u4e2a\u6897", user="bob"),
        _doc(3, "\u7fa4\u91cc\u53c8\u5728\u804a\u661f\u6d41\u732b\u732b", user="carol"),
        _doc(4, "\u661f\u6d41\u732b\u732b\u771f\u7684\u5f88\u597d\u7b11", user="dave"),
    )
    default_result = discover_new_words(docs, _compact_config())
    no_frequency_result = discover_new_words(
        docs,
        _compact_config(frequency_score_weight=0.0),
    )

    default_item = default_result.promoted[0]
    no_frequency_item = no_frequency_result.promoted[0]

    assert default_item.score_breakdown.frequency > 1.0
    assert 0.0 <= default_item.score <= 1.0
    assert 0.0 <= no_frequency_item.score <= 1.0
    assert no_frequency_item.score != default_item.score
    assert no_frequency_item.score_breakdown == default_item.score_breakdown
    assert no_frequency_item.normalized_score_breakdown == (
        default_item.normalized_score_breakdown
    )


def test_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_documents"):
        NewWordConfig(max_documents=0)
    with pytest.raises(ValueError, match="character limits"):
        NewWordConfig(min_chars=3, max_chars=2)
    with pytest.raises(ValueError, match="ratios"):
        NewWordConfig(repeated_text_user_weight=1.5)
    with pytest.raises(ValueError, match="score weights"):
        NewWordConfig(frequency_score_weight=-1.0)
    with pytest.raises(ValueError, match="score normalizers"):
        NewWordConfig(frequency_score_norm=0.0)
    with pytest.raises(ValueError, match="min_promoted_score"):
        NewWordConfig(min_promoted_score=1.5)


def test_default_document_and_user_thresholds_are_relaxed() -> None:
    config = NewWordConfig()

    assert config.min_documents == 1
    assert config.min_users == 1


def test_default_candidate_length_keeps_long_candidates() -> None:
    config = NewWordConfig()

    assert config.max_chars == 6


def test_default_npmi_floor_is_conservative() -> None:
    config = NewWordConfig()

    assert config.min_npmi == 0.50


def test_default_promoted_score_floor_is_conservative() -> None:
    config = NewWordConfig()

    assert config.min_promoted_score == 0.60


def test_default_score_weights_follow_quality_first_ratio() -> None:
    config = NewWordConfig()

    assert config.frequency_score_weight == 2.0
    assert config.user_diversity_score_weight == 2.0
    assert config.cohesion_score_weight == 5.0
    assert config.boundary_score_weight == 2.0
    assert config.independence_score_weight == 1.0
    assert config.repeat_score_weight == 1.0
    assert config.dictionary_score_weight == 0.0
