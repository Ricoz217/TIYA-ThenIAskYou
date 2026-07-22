from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import jieba
import jieba.posseg as pseg

from .models import MessageInput, RelatednessConfig, TextFeatures


_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+|[a-z0-9_+#.]+",
    re.IGNORECASE,
)
_PUBLIC_USER_DICT = Path(__file__).parents[3] / "data" / "dictionary" / "user_dict_pos.txt"
_TOKENIZER_LOCK = threading.RLock()
_SHARED_TOKENIZER: jieba.Tokenizer | None = None
_SHARED_POS_TOKENIZER: pseg.POSTokenizer | None = None


def normalize_text(text: str) -> str:
    text = text.casefold().replace("\u3000", " ")
    return _SPACE_RE.sub(" ", text).strip()


def _get_tokenizer() -> jieba.Tokenizer:
    global _SHARED_TOKENIZER
    if _SHARED_TOKENIZER is not None:
        return _SHARED_TOKENIZER
    with _TOKENIZER_LOCK:
        if _SHARED_TOKENIZER is None:
            tokenizer = jieba.Tokenizer()
            tokenizer.initialize()
            if _PUBLIC_USER_DICT.is_file():
                tokenizer.load_userdict(str(_PUBLIC_USER_DICT))
            _SHARED_TOKENIZER = tokenizer
    return _SHARED_TOKENIZER


def _get_pos_tokenizer() -> pseg.POSTokenizer:
    global _SHARED_POS_TOKENIZER
    if _SHARED_POS_TOKENIZER is not None:
        return _SHARED_POS_TOKENIZER
    with _TOKENIZER_LOCK:
        if _SHARED_POS_TOKENIZER is None:
            _SHARED_POS_TOKENIZER = pseg.POSTokenizer(_get_tokenizer())
    return _SHARED_POS_TOKENIZER


@lru_cache(maxsize=65_536)
def _part_of_speech(term: str) -> str:
    if term.isascii():
        return "eng" if any(character.isalpha() for character in term) else "m"
    tagged = tuple(_get_pos_tokenizer().cut(term, HMM=True))
    if len(tagged) == 1 and tagged[0].word == term:
        return tagged[0].flag
    return "n"


class DynamicLexicon:
    __slots__ = ("_buckets", "_terms", "version")

    def __init__(self, terms: tuple[str, ...] = (), *, version: int = 0):
        normalized = tuple(
            sorted(
                {normalize_text(term) for term in terms if normalize_text(term)},
                key=lambda value: (-len(value), value),
            )
        )
        buckets: dict[str, list[str]] = defaultdict(list)
        for term in normalized:
            buckets[term[0]].append(term)
        self._buckets = {key: tuple(values) for key, values in buckets.items()}
        self._terms = normalized
        self.version = version

    @property
    def terms(self) -> tuple[str, ...]:
        return self._terms

    def match(self, text: str) -> tuple[str, ...]:
        if not text or not self._terms:
            return ()
        found: set[str] = set()
        for character in set(text):
            for term in self._buckets.get(character, ()):
                if term in text:
                    found.add(term)
        return tuple(sorted(found))


class TextProcessor:
    __slots__ = ("config", "stopwords", "_tokenizer")

    def __init__(
        self,
        *,
        config: RelatednessConfig | None = None,
        stopwords: set[str] | None = None,
    ):
        self.config = config or RelatednessConfig()
        self.stopwords = frozenset(stopwords or ())
        self._tokenizer = _get_tokenizer()

    def compare_tokenization_modes(
        self,
        text: str,
    ) -> dict[str, tuple[str, ...]]:
        blocks = _TOKEN_RE.findall(normalize_text(text))
        modes: dict[str, list[str]] = {
            "exact_no_hmm": [],
            "exact_hmm": [],
            "search_no_hmm": [],
            "search_hmm": [],
            "full": [],
        }
        for block in blocks:
            modes["exact_no_hmm"].extend(
                self._tokenizer.cut(block, cut_all=False, HMM=False)
            )
            modes["exact_hmm"].extend(
                self._tokenizer.cut(block, cut_all=False, HMM=True)
            )
            modes["search_no_hmm"].extend(
                self._tokenizer.cut_for_search(block, HMM=False)
            )
            modes["search_hmm"].extend(
                self._tokenizer.cut_for_search(block, HMM=True)
            )
            modes["full"].extend(
                self._tokenizer.cut(block, cut_all=True, HMM=False)
            )
        return {
            name: tuple(token.strip() for token in tokens if token.strip())
            for name, tokens in modes.items()
        }

    @staticmethod
    def _normalized_weights(
        counts: Mapping[str, int],
        scale: float,
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        if not counts or scale <= 0:
            return (), ()
        norm = math.sqrt(sum(value * value for value in counts.values()))
        terms = tuple(sorted(counts))
        weights = tuple(scale * counts[term] / norm for term in terms)
        return terms, weights

    def _extract_counts(
        self,
        normalized: str,
        lexicon: DynamicLexicon,
    ) -> tuple[Counter[str], Counter[str]]:
        lexical: Counter[str] = Counter()
        if normalized:
            for block in _TOKEN_RE.findall(normalized):
                terms = (
                    (block,)
                    if block.isascii()
                    else self._tokenizer.cut(block, cut_all=False, HMM=True)
                )
                for term in terms:
                    term = term.strip()
                    if term and term not in self.stopwords:
                        lexical[term] += 1
        for phrase in lexicon.match(normalized):
            if phrase not in self.stopwords:
                lexical[phrase] += 1

        compact = "".join(_TOKEN_RE.findall(normalized))
        subword: Counter[str] = Counter()
        for size in (2, 3, 4):
            for index in range(max(0, len(compact) - size + 1)):
                gram = compact[index:index + size]
                subword[f"CHAR{size}:{gram}"] += 1
        if len(compact) == 1:
            subword[f"CHAR1:{compact}"] += 1
        return lexical, subword

    @classmethod
    def _merge_weighted_counts(
        cls,
        primary: Mapping[str, int],
        semantic: Mapping[str, int],
        *,
        base_weight: float,
        semantic_weight: float,
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        merged: dict[str, float] = defaultdict(float)
        terms, weights = cls._normalized_weights(primary, math.sqrt(base_weight))
        for term, weight in zip(terms, weights, strict=True):
            merged[term] += weight
        terms, weights = cls._normalized_weights(
            semantic,
            math.sqrt(base_weight * semantic_weight),
        )
        for term, weight in zip(terms, weights, strict=True):
            merged[term] += weight
        ordered = tuple(sorted(merged))
        return ordered, tuple(merged[term] for term in ordered)

    def extract(
        self,
        message: MessageInput,
        lexicon: DynamicLexicon | None = None,
    ) -> TextFeatures:
        active_lexicon = lexicon or DynamicLexicon()
        normalized = normalize_text(message.text)
        semantic = normalize_text(
            message.semantic_text[:self.config.semantic_text_max_chars]
        )
        lexical, subword = self._extract_counts(normalized, active_lexicon)
        semantic_lexical, semantic_subword = self._extract_counts(
            semantic,
            active_lexicon,
        )
        vector_lexical = lexical.copy()
        if not semantic:
            for media_id in message.media_ids:
                vector_lexical[f"MEDIA:{media_id}"] += 1
        lexical_terms, lexical_weights = self._merge_weighted_counts(
            vector_lexical,
            semantic_lexical,
            base_weight=self.config.lexical_weight,
            semantic_weight=self.config.semantic_text_weight,
        )
        subword_terms, subword_weights = self._merge_weighted_counts(
            subword,
            semantic_subword,
            base_weight=self.config.subword_weight,
            semantic_weight=self.config.semantic_text_weight,
        )
        return TextFeatures(
            normalized_text=" ".join(part for part in (normalized, semantic) if part),
            lexical_terms=lexical_terms,
            lexical_weights=lexical_weights,
            subword_terms=subword_terms,
            subword_weights=subword_weights,
            lexicon_version=active_lexicon.version,
            primary_lexical_terms=tuple(lexical),
            semantic_lexical_terms=tuple(semantic_lexical),
            lexical_pos=tuple(
                (
                    term,
                    _part_of_speech(term),
                )
                for term in lexical_terms
                if not term.startswith("MEDIA:")
            ),
        )


class TextIndex:
    __slots__ = ("config", "_documents", "_postings", "_lengths", "_total_length")

    def __init__(self, *, config: RelatednessConfig | None = None):
        self.config = config or RelatednessConfig()
        self._documents: dict[str, dict[str, float]] = {}
        self._postings: dict[str, dict[str, float]] = defaultdict(dict)
        self._lengths: dict[str, float] = {}
        self._total_length = 0.0

    def __len__(self) -> int:
        return len(self._documents)

    def add(self, msg_id: str, features: TextFeatures) -> None:
        if msg_id in self._documents:
            return
        weights = dict(features.term_weights)
        length = sum(weights.values()) or 1.0
        self._documents[msg_id] = weights
        self._lengths[msg_id] = length
        self._total_length += length
        for term, weight in weights.items():
            self._postings[term][msg_id] = weight

    def remove(self, msg_id: str) -> None:
        weights = self._documents.pop(msg_id, None)
        if weights is None:
            return
        self._total_length -= self._lengths.pop(msg_id, 0.0)
        for term in weights:
            posting = self._postings.get(term)
            if posting is None:
                continue
            posting.pop(msg_id, None)
            if not posting:
                self._postings.pop(term, None)

    def search(self, features: TextFeatures) -> tuple[tuple[str, float], ...]:
        document_count = len(self._documents)
        if document_count == 0:
            return ()
        average_length = self._total_length / document_count
        scores: dict[str, float] = defaultdict(float)
        for term, query_weight in features.term_weights:
            posting = self._postings.get(term)
            if not posting:
                continue
            document_frequency = len(posting)
            idf = math.log1p((document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for msg_id, frequency in posting.items():
                length_ratio = self._lengths[msg_id] / average_length
                denominator = frequency + self.config.bm25_k1 * (
                    1.0 - self.config.bm25_b + self.config.bm25_b * length_ratio
                )
                score = idf * (
                    frequency * (self.config.bm25_k1 + 1.0) / denominator
                )
                scores[msg_id] += score * query_weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return tuple(ranked[:self.config.text_candidate_limit])
