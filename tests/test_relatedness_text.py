import pytest

from TIYA.relatedness.models import MessageInput, RelatednessConfig
from TIYA.relatedness.text import DynamicLexicon, TextIndex, TextProcessor


def _message(msg_id: str, text: str) -> MessageInput:
    return MessageInput(
        msg_id=msg_id,
        group_id="group",
        user_id="user",
        timestamp=1.0,
        text=text,
    )


def test_dynamic_lexicon_is_isolated_per_group() -> None:
    processor = TextProcessor()
    first = DynamicLexicon(("牢大",))
    second = DynamicLexicon()

    first_features = processor.extract(_message("1", "今天继续牢大挑战"), first)
    second_features = processor.extract(_message("2", "今天继续牢大挑战"), second)

    assert "牢大" in first_features.lexical_terms
    assert "牢大" not in second_features.lexical_terms
    assert any(term.startswith("CHAR2:") for term in second_features.subword_terms)


def test_bm25_index_returns_relevant_candidates_and_removes_evicted_docs() -> None:
    config = RelatednessConfig(text_candidate_limit=8)
    processor = TextProcessor(config=config)
    index = TextIndex(config=config)
    python_doc = processor.extract(_message("1", "python asyncio 性能优化"))
    cooking_doc = processor.extract(_message("2", "今天晚饭红烧肉"))
    query = processor.extract(_message("3", "asyncio 优化"))
    index.add("1", python_doc)
    index.add("2", cooking_doc)

    matches = index.search(query)

    assert matches[0][0] == "1"
    assert matches[0][1] > 0
    index.remove("1")
    assert all(msg_id != "1" for msg_id, _ in index.search(query))


def test_stopwords_do_not_become_lexical_features() -> None:
    processor = TextProcessor(stopwords={"的", "了"})

    features = processor.extract(_message("1", "这个功能真的优化了"))

    assert "的" not in features.lexical_terms
    assert "了" not in features.lexical_terms


def test_numeric_terms_remain_available_for_realtime_retrieval() -> None:
    processor = TextProcessor()

    features = processor.extract(_message("1", "33 CNY major UNN"))

    assert any("33" in term for term in features.lexical_terms)
    assert {"cny", "major", "unn"} <= set(features.lexical_terms)
    assert not {"ma", "jo", "nn"} & set(features.lexical_terms)


def test_hotword_shape_limits_must_be_consistent() -> None:
    with pytest.raises(ValueError, match="hotword shape limits"):
        RelatednessConfig(hotword_min_chars=5, hotword_max_chars=4)


def test_semantic_text_is_indexed_with_a_smaller_weight() -> None:
    processor = TextProcessor(
        config=RelatednessConfig(semantic_text_weight=0.25)
    )
    primary = processor.extract(_message("1", "猫猫表情包"))
    semantic = processor.extract(
        MessageInput(
            msg_id="2",
            group_id="group",
            user_id="user",
            timestamp=1.0,
            text="",
            semantic_text="猫猫表情包",
        )
    )

    primary_weight = sum(primary.lexical_weights)
    semantic_weight = sum(semantic.lexical_weights)

    assert semantic.lexical_terms == primary.lexical_terms
    assert 0 < semantic_weight < primary_weight


def test_media_hash_is_a_temporary_word_replaced_by_semantic_terms() -> None:
    processor = TextProcessor()
    pending = processor.extract(
        MessageInput(
            msg_id="pending",
            group_id="group",
            user_id="user",
            timestamp=1.0,
            text="",
            media_ids=frozenset({"image:same-hash"}),
        )
    )
    enriched = processor.extract(
        MessageInput(
            msg_id="pending",
            group_id="group",
            user_id="user",
            timestamp=1.0,
            text="",
            semantic_text="猫猫表情包",
            media_ids=frozenset({"image:same-hash"}),
        )
    )

    assert "MEDIA:image:same-hash" in pending.lexical_terms
    assert "MEDIA:image:same-hash" not in enriched.lexical_terms
    assert any("猫" in term or "表情" in term for term in enriched.lexical_terms)


def test_lexical_terms_use_local_pos_dictionary_with_noun_fallback() -> None:
    processor = TextProcessor()

    features = processor.extract(
        _message("1", "碗橱倒灌赛博新梗"),
        DynamicLexicon(("赛博新梗",)),
    )
    part_of_speech = dict(features.lexical_pos)

    assert part_of_speech["碗橱"].startswith("n")
    assert part_of_speech["倒灌"].startswith("v")
    assert part_of_speech["赛博新梗"] == "n"


def test_production_terms_use_exact_hmm_segmentation_and_real_pos() -> None:
    processor = TextProcessor()
    text = "李小福去了杭研大厦这个方案漂亮"

    features = processor.extract(_message("1", text))
    modes = processor.compare_tokenization_modes(text)
    part_of_speech = dict(features.lexical_pos)

    assert features.primary_lexical_terms == modes["exact_hmm"]
    assert "李小福" in features.primary_lexical_terms
    assert "杭研" in features.primary_lexical_terms
    assert part_of_speech["李小福"].startswith("nr")
    assert "漂亮" in features.lexical_terms
    assert part_of_speech["漂亮"].startswith("a")


def test_tokenization_mode_comparison_includes_jieba_variants() -> None:
    modes = TextProcessor().compare_tokenization_modes(
        "南京市长江大桥 major"
    )

    assert set(modes) == {
        "exact_no_hmm",
        "exact_hmm",
        "search_no_hmm",
        "search_hmm",
        "full",
    }
    assert all(isinstance(tokens, tuple) for tokens in modes.values())
