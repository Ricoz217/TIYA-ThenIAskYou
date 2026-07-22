from __future__ import annotations

import math
import json
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import jieba


_CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_PROJECT_DICTIONARY_DIR = Path(__file__).parents[3] / "data" / "dictionary"
_PROJECT_USER_DICT_POS = _PROJECT_DICTIONARY_DIR / "user_dict_pos.txt"
_PROJECT_USER_DICT = _PROJECT_DICTIONARY_DIR / "user_dict.txt"
_PROJECT_STOPWORDS = _PROJECT_DICTIONARY_DIR / "stopwords.json"
_TOKENIZER_LOCK = threading.RLock()
_TOKENIZER: jieba.Tokenizer | None = None

_FUNCTION_PHRASES = frozenset({
    "不知道",
    "是什么",
    "的时候",
    "能不能",
    "有没有",
    "为什么",
    "怎么样",
    "怎么办",
    "什么事",
    "什么东西",
    "怎么说",
    "怎么会",
    "怎么就",
    "是不是",
    "有没有",
    "可以吗",
    "好不好",
    "行不行",
    "真的假的",
})
_TWO_CHAR_FUNCTION_PREFIXES = frozenset({
    "我",
    "你",
    "他",
    "她",
    "它",
    "这",
    "那",
    "哪",
    "谁",
    "都",
    "也",
    "还",
    "又",
    "就",
    "在",
    "给",
    "把",
    "被",
    "和",
    "跟",
})
_TWO_CHAR_FUNCTION_SUFFIXES = frozenset({
    "的",
    "了",
    "吗",
    "呢",
    "吧",
    "啊",
    "呀",
    "么",
    "嘛",
    "点",
    "些",
    "个",
    "里",
    "上",
    "下",
})


class NewWordStatus(StrEnum):
    PROMOTED = "promoted"
    CANDIDATE = "candidate"
    KNOWN = "known"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class NewWordDocument:
    msg_id: str
    user_id: str
    timestamp: float
    text: str


@dataclass(frozen=True, slots=True)
class NewWordConfig:
    max_documents: int = 5000
    min_chars: int = 2
    max_chars: int = 6
    min_frequency: int = 4
    min_documents: int = 1
    min_users: int = 1
    min_npmi: float = 0.50
    min_promoted_score: float = 0.60
    min_boundary_entropy: float = 0.35
    max_nested_coverage: float = 0.75
    repeated_text_user_weight: float = 0.35
    frequency_score_weight: float = 2.0
    user_diversity_score_weight: float = 2.0
    cohesion_score_weight: float = 5.0
    boundary_score_weight: float = 2.0
    independence_score_weight: float = 1.0
    repeat_score_weight: float = 1.0
    dictionary_score_weight: float = 0.0
    frequency_score_norm: float = 4.0
    user_diversity_score_norm: float = 3.0
    cohesion_score_norm: float = 1.0
    boundary_score_norm: float = 2.5
    independence_score_norm: float = 1.0
    repeat_score_norm: float = 1.0
    dictionary_score_norm: float = 1.0
    enable_dynamic_stopwords: bool = True
    dynamic_stopword_min_documents: int = 5
    dynamic_stopword_doc_ratio: float = 0.08
    dynamic_stopword_user_ratio: float = 0.25
    dynamic_stopword_max_specificity: float = 0.80
    dynamic_stopword_time_uniformity: float = 0.15
    dynamic_stopword_time_max_specificity: float = 0.80
    promoted_limit: int = 100
    candidate_limit: int = 300
    known_limit: int = 100
    rejected_sample_limit: int = 200
    evidence_limit: int = 5
    load_project_user_dict: bool = True
    use_project_stopwords: bool = True
    stop_terms: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.max_documents <= 0:
            raise ValueError("max_documents must be positive")
        if self.min_chars <= 0 or self.max_chars < self.min_chars:
            raise ValueError("character limits must be positive and ordered")
        if any(
            value <= 0
            for value in (
                self.min_frequency,
                self.min_documents,
                self.min_users,
                self.dynamic_stopword_min_documents,
                self.promoted_limit,
                self.candidate_limit,
                self.known_limit,
                self.rejected_sample_limit,
                self.evidence_limit,
            )
        ):
            raise ValueError("new word limits must be positive")
        if not 0.0 <= self.repeated_text_user_weight <= 1.0:
            raise ValueError("ratios must be between zero and one")
        if any(
            not 0.0 <= value <= 1.0
            for value in (
                self.dynamic_stopword_doc_ratio,
                self.dynamic_stopword_user_ratio,
                self.dynamic_stopword_max_specificity,
                self.dynamic_stopword_time_uniformity,
                self.dynamic_stopword_time_max_specificity,
            )
        ):
            raise ValueError("ratios must be between zero and one")
        if not 0.0 <= self.min_boundary_entropy:
            raise ValueError("boundary entropy must be non-negative")
        if not -1.0 <= self.min_npmi <= 1.0:
            raise ValueError("min_npmi must be between -1 and one")
        if not 0.0 <= self.min_promoted_score <= 1.0:
            raise ValueError("min_promoted_score must be between zero and one")
        if not 0.0 <= self.max_nested_coverage <= 1.0:
            raise ValueError("ratios must be between zero and one")
        if any(
            value < 0.0
            for value in (
                self.frequency_score_weight,
                self.user_diversity_score_weight,
                self.cohesion_score_weight,
                self.boundary_score_weight,
                self.independence_score_weight,
                self.repeat_score_weight,
                self.dictionary_score_weight,
            )
        ):
            raise ValueError("score weights must be non-negative")
        if any(
            value <= 0.0
            for value in (
                self.frequency_score_norm,
                self.user_diversity_score_norm,
                self.cohesion_score_norm,
                self.boundary_score_norm,
                self.independence_score_norm,
                self.repeat_score_norm,
                self.dictionary_score_norm,
            )
        ):
            raise ValueError("score normalizers must be positive")


@dataclass(frozen=True, slots=True)
class NewWordScoreBreakdown:
    frequency: float
    user_diversity: float
    cohesion: float
    boundary: float
    independence: float
    repeat: float
    dictionary: float


@dataclass(frozen=True, slots=True)
class NewWordCandidate:
    term: str
    status: NewWordStatus
    score: float
    frequency: int
    weighted_frequency: float
    document_count: int
    user_count: int
    pmi: float
    npmi: float
    left_entropy: float
    right_entropy: float
    boundary_entropy: float
    nested_coverage: float
    repeat_penalty: float
    dictionary_penalty: float
    score_breakdown: NewWordScoreBreakdown
    normalized_score_breakdown: NewWordScoreBreakdown
    first_seen: float
    last_seen: float
    evidence_message_ids: tuple[str, ...] = ()
    reject_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NewWordDiscoveryResult:
    documents: int
    candidate_count: int
    promoted: tuple[NewWordCandidate, ...]
    candidates: tuple[NewWordCandidate, ...]
    known: tuple[NewWordCandidate, ...]
    rejected: tuple[NewWordCandidate, ...]
    rejected_reason_counts: tuple[tuple[str, int], ...]
    nested_suppressions: tuple[tuple[str, str, float], ...]
    elapsed_ms: float
    dynamic_stopwords: tuple[str, ...] = ()


@dataclass(slots=True)
class _TermStats:
    frequency: int = 0
    weighted_frequency: float = 0.0
    doc_ids: set[str] | None = None
    user_ids: set[str] | None = None
    left_context: Counter[str] | None = None
    right_context: Counter[str] | None = None
    occurrence_times: list[float] | None = None
    first_seen: float = math.inf
    last_seen: float = -math.inf
    evidence: list[str] | None = None

    def add(
        self,
        document: NewWordDocument,
        *,
        weight: float,
        left: str,
        right: str,
        evidence_limit: int,
    ) -> None:
        self.frequency += 1
        self.weighted_frequency += weight
        if self.doc_ids is None:
            self.doc_ids = set()
            self.user_ids = set()
            self.left_context = Counter()
            self.right_context = Counter()
            self.occurrence_times = []
            self.evidence = []
        assert self.doc_ids is not None
        assert self.user_ids is not None
        assert self.left_context is not None
        assert self.right_context is not None
        assert self.occurrence_times is not None
        assert self.evidence is not None
        first_in_document = document.msg_id not in self.doc_ids
        self.doc_ids.add(document.msg_id)
        self.user_ids.add(document.user_id)
        self.left_context[left] += weight
        self.right_context[right] += weight
        if first_in_document:
            self.occurrence_times.append(document.timestamp)
        self.first_seen = min(self.first_seen, document.timestamp)
        self.last_seen = max(self.last_seen, document.timestamp)
        if len(self.evidence) < evidence_limit and document.msg_id not in self.evidence:
            self.evidence.append(document.msg_id)

    @property
    def document_count(self) -> int:
        return len(self.doc_ids or ())

    @property
    def user_count(self) -> int:
        return len(self.user_ids or ())


def discover_new_words(
    documents: Iterable[NewWordDocument],
    config: NewWordConfig | None = None,
) -> NewWordDiscoveryResult:
    started = time.perf_counter()
    cfg = config or NewWordConfig()
    selected = tuple(documents)[-cfg.max_documents:]
    term_stats, count_by_len = _collect_stats(selected, cfg)
    raw_candidates = {
        term: stats
        for term, stats in term_stats.items()
        if cfg.min_chars <= len(term) <= cfg.max_chars
    }
    dynamic_stopwords = _dynamic_stopwords(raw_candidates, selected, cfg)
    nested_coverage, suppressions = _nested_coverage(raw_candidates, cfg)
    ranked = [
        _build_candidate(
            term,
            stats,
            term_stats,
            count_by_len,
            nested_coverage,
            dynamic_stopwords,
            cfg,
        )
        for term, stats in raw_candidates.items()
    ]
    ranked.sort(key=_rank_key)

    promoted = tuple(
        item for item in ranked if item.status is NewWordStatus.PROMOTED
    )[:cfg.promoted_limit]
    candidates = tuple(
        item for item in ranked if item.status is NewWordStatus.CANDIDATE
    )[:cfg.candidate_limit]
    known = tuple(
        item for item in ranked if item.status is NewWordStatus.KNOWN
    )[:cfg.known_limit]
    rejected = tuple(
        item for item in ranked if item.status is NewWordStatus.REJECTED
    )[:cfg.rejected_sample_limit]
    reason_counts = Counter(
        reason for item in ranked for reason in item.reject_reasons
    )

    return NewWordDiscoveryResult(
        documents=len(selected),
        candidate_count=len(raw_candidates),
        promoted=promoted,
        candidates=candidates,
        known=known,
        rejected=rejected,
        rejected_reason_counts=tuple(reason_counts.most_common()),
        nested_suppressions=tuple(suppressions[:cfg.rejected_sample_limit]),
        elapsed_ms=(time.perf_counter() - started) * 1_000.0,
        dynamic_stopwords=tuple(sorted(dynamic_stopwords)),
    )


def format_new_word_report(result: NewWordDiscoveryResult) -> str:
    lines = [
        "# New Word Discovery Report",
        "",
        "## Summary",
        f"- documents: {result.documents}",
        f"- candidates: {result.candidate_count}",
        f"- promoted: {len(result.promoted)}",
        f"- candidate: {len(result.candidates)}",
        f"- known: {len(result.known)}",
        f"- rejected samples: {len(result.rejected)}",
        f"- dynamic_stopwords: {len(result.dynamic_stopwords)}",
        f"- elapsed_ms: {result.elapsed_ms:.2f}",
        "",
    ]
    _append_section(lines, "PROMOTED", result.promoted)
    _append_section(lines, "CANDIDATE", result.candidates)
    _append_section(lines, "KNOWN", result.known)
    _append_section(lines, "REJECTED", result.rejected, include_reasons=True)
    lines.extend(["## reject reasons"])
    if result.rejected_reason_counts:
        for reason, count in result.rejected_reason_counts:
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## nested suppressions"])
    if result.nested_suppressions:
        for short, longer, coverage in result.nested_suppressions[:30]:
            lines.append(f"- {short} -> {longer}: coverage={coverage:.3f}")
    else:
        lines.append("- none")
    lines.extend(["", "## dynamic stopwords"])
    if result.dynamic_stopwords:
        lines.append(", ".join(result.dynamic_stopwords[:200]))
    else:
        lines.append("- none")
    return "\n".join(lines)


def _normalize_text(text: str) -> str:
    cleaned = _URL_RE.sub(" ", text)
    cleaned = _CONTROL_RE.sub(" ", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _get_tokenizer(config: NewWordConfig) -> jieba.Tokenizer:
    global _TOKENIZER
    if not config.load_project_user_dict:
        return jieba.dt
    if _TOKENIZER is not None:
        return _TOKENIZER
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            tokenizer = jieba.Tokenizer()
            tokenizer.initialize()
            if _PROJECT_USER_DICT_POS.is_file():
                tokenizer.load_userdict(str(_PROJECT_USER_DICT_POS))
            elif _PROJECT_USER_DICT.is_file():
                tokenizer.load_userdict(str(_PROJECT_USER_DICT))
            _TOKENIZER = tokenizer
    return _TOKENIZER


@lru_cache(maxsize=1)
def _project_stop_terms() -> frozenset[str]:
    if not _PROJECT_STOPWORDS.is_file():
        return frozenset()
    try:
        loaded = json.loads(_PROJECT_STOPWORDS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    if not isinstance(loaded, list):
        return frozenset()
    terms = {
        normalized
        for item in loaded
        if isinstance(item, str)
        and (normalized := _normalize_text(item))
        and _CHINESE_RUN_RE.fullmatch(normalized)
        and 2 <= len(normalized) <= 6
    }
    return frozenset(terms)


def _active_stop_terms(config: NewWordConfig) -> frozenset[str]:
    custom = frozenset(
        normalized
        for item in config.stop_terms
        if (normalized := _normalize_text(item))
    )
    if not config.use_project_stopwords:
        return custom
    return _project_stop_terms() | custom


def _dynamic_stopwords(
    candidates: dict[str, _TermStats],
    documents: tuple[NewWordDocument, ...],
    config: NewWordConfig,
) -> frozenset[str]:
    if not config.enable_dynamic_stopwords:
        return frozenset()
    total_documents = len(documents)
    if total_documents < config.dynamic_stopword_min_documents:
        return frozenset()
    total_users = len({document.user_id for document in documents})
    maximum_idf = math.log((total_documents + 1.0) / 1.5) or 1.0
    ordered_times = sorted(document.timestamp for document in documents)
    window_start = ordered_times[0]
    window_end = ordered_times[-1]
    window_span = max(0.0, window_end - window_start)
    eligible_counts = sorted(
        {
            stats.document_count
            for stats in candidates.values()
            if stats.document_count >= config.dynamic_stopword_min_documents
        },
        reverse=True,
    )
    frequency_ranks = {
        count: index / max(1, len(eligible_counts))
        for index, count in enumerate(eligible_counts)
    }

    def temporal_uniformity(stats: _TermStats) -> float:
        times = sorted(stats.occurrence_times or ())
        if len(times) < 2 or window_span <= 0.0:
            return 1.0
        gaps = [
            later - earlier
            for earlier, later in zip(times, times[1:], strict=False)
        ]
        gap_total = sum(gaps)
        if gap_total <= 0.0:
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
                (gap - mean_gap) ** 2
                for gap in gaps
            ) / gap_count
            cv = math.sqrt(variance) / mean_gap if mean_gap > 0.0 else 0.0
            normalized_cv = cv / (1.0 + cv)
        coverage_penalty = 1.0 - min(1.0, (times[-1] - times[0]) / window_span)
        frequency_rank = frequency_ranks.get(stats.document_count, 1.0)
        return (
            0.45 * gap_gini
            + 0.20 * normalized_cv
            + 0.25 * coverage_penalty
            + 0.10 * frequency_rank
        )

    result: set[str] = set()
    for term, stats in candidates.items():
        count = stats.document_count
        if count < config.dynamic_stopword_min_documents:
            continue
        doc_ratio = count / total_documents
        user_ratio = stats.user_count / max(1, total_users)
        idf = math.log((total_documents + 1.0) / (count + 0.5))
        specificity = max(0.0, min(1.0, idf / maximum_idf))
        broadly_distributed = (
            doc_ratio >= config.dynamic_stopword_doc_ratio
            or user_ratio >= config.dynamic_stopword_user_ratio
        )
        uniformly_distributed = (
            temporal_uniformity(stats) <= config.dynamic_stopword_time_uniformity
            and specificity <= config.dynamic_stopword_time_max_specificity
        )
        if (
            broadly_distributed
            and specificity <= config.dynamic_stopword_max_specificity
        ) or uniformly_distributed:
            result.add(term)
    return frozenset(result)


def _collect_stats(
    documents: tuple[NewWordDocument, ...],
    config: NewWordConfig,
) -> tuple[dict[str, _TermStats], Counter[int]]:
    term_stats: defaultdict[str, _TermStats] = defaultdict(_TermStats)
    count_by_len: Counter[int] = Counter()
    repeated_seen: Counter[tuple[str, str]] = Counter()
    for document in documents:
        text = _normalize_text(document.text)
        if not text:
            continue
        repeat_key = (document.user_id, text)
        repeated_seen[repeat_key] += 1
        weight = (
            1.0
            if repeated_seen[repeat_key] == 1
            else config.repeated_text_user_weight
        )
        seen_in_doc: set[str] = set()
        for match in _CHINESE_RUN_RE.finditer(text):
            run = match.group()
            run_start = match.start()
            for start in range(len(run)):
                max_length = min(config.max_chars, len(run) - start)
                for length in range(1, max_length + 1):
                    term = run[start:start + length]
                    global_start = run_start + start
                    global_end = global_start + length
                    left = text[global_start - 1] if global_start > 0 else "^"
                    right = text[global_end] if global_end < len(text) else "$"
                    term_stats[term].add(
                        document,
                        weight=weight,
                        left=left,
                        right=right,
                        evidence_limit=config.evidence_limit,
                    )
                    count_by_len[length] += weight
                    if length >= config.min_chars:
                        seen_in_doc.add(term)
        for term in seen_in_doc:
            # Document/user sets are already maintained per occurrence. This hook is
            # intentionally left explicit so the collection rules stay visible.
            _ = term
    return dict(term_stats), count_by_len


def _build_candidate(
    term: str,
    stats: _TermStats,
    all_stats: dict[str, _TermStats],
    count_by_len: Counter[int],
    nested_coverage: dict[str, float],
    dynamic_stopwords: frozenset[str],
    config: NewWordConfig,
) -> NewWordCandidate:
    pmi, npmi = _cohesion(term, stats, all_stats, count_by_len)
    left_entropy = _entropy(stats.left_context or Counter())
    right_entropy = _entropy(stats.right_context or Counter())
    boundary_entropy = min(left_entropy, right_entropy)
    coverage = nested_coverage.get(term, 0.0)
    known = _is_known_jieba_word(term, config)
    repeat_penalty = min(
        1.0,
        stats.weighted_frequency / max(1.0, float(stats.frequency)),
    )
    dictionary_penalty = 0.0 if known else 1.0
    reject_reasons = _reject_reasons(
        term,
        stats,
        npmi,
        boundary_entropy,
        coverage,
        dynamic_stopwords,
        config,
    )
    score_breakdown = _score_breakdown(
        stats,
        npmi=npmi,
        boundary_entropy=boundary_entropy,
        nested_coverage=coverage,
        repeat_penalty=repeat_penalty,
        dictionary_penalty=dictionary_penalty,
    )
    normalized_score_breakdown = _normalize_score_breakdown(score_breakdown, config)
    score = _score(normalized_score_breakdown, config)
    if score < config.min_promoted_score:
        reject_reasons.append("promoted_score")
    status = _status(
        known,
        reject_reasons,
        npmi,
        boundary_entropy,
        coverage,
        score,
        config,
    )
    return NewWordCandidate(
        term=term,
        status=status,
        score=score,
        frequency=stats.frequency,
        weighted_frequency=stats.weighted_frequency,
        document_count=stats.document_count,
        user_count=stats.user_count,
        pmi=pmi,
        npmi=npmi,
        left_entropy=left_entropy,
        right_entropy=right_entropy,
        boundary_entropy=boundary_entropy,
        nested_coverage=coverage,
        repeat_penalty=repeat_penalty,
        dictionary_penalty=dictionary_penalty,
        score_breakdown=score_breakdown,
        normalized_score_breakdown=normalized_score_breakdown,
        first_seen=stats.first_seen if math.isfinite(stats.first_seen) else 0.0,
        last_seen=stats.last_seen if math.isfinite(stats.last_seen) else 0.0,
        evidence_message_ids=tuple(stats.evidence or ()),
        reject_reasons=tuple(reject_reasons),
    )


def _cohesion(
    term: str,
    stats: _TermStats,
    all_stats: dict[str, _TermStats],
    count_by_len: Counter[int],
) -> tuple[float, float]:
    total = float(count_by_len[len(term)])
    if total <= 0.0 or stats.weighted_frequency <= 0.0:
        return 0.0, 0.0
    p_term = stats.weighted_frequency / total
    values: list[float] = []
    for split in range(1, len(term)):
        left = term[:split]
        right = term[split:]
        left_stats = all_stats.get(left)
        right_stats = all_stats.get(right)
        left_total = float(count_by_len[len(left)])
        right_total = float(count_by_len[len(right)])
        if (
            left_stats is None
            or right_stats is None
            or left_total <= 0.0
            or right_total <= 0.0
            or left_stats.weighted_frequency <= 0.0
            or right_stats.weighted_frequency <= 0.0
        ):
            continue
        p_left = left_stats.weighted_frequency / left_total
        p_right = right_stats.weighted_frequency / right_total
        values.append(math.log(p_term / (p_left * p_right)))
    if not values:
        return 0.0, 0.0
    pmi = min(values)
    npmi = pmi / -math.log(p_term) if 0.0 < p_term < 1.0 else 0.0
    return pmi, max(-1.0, min(1.0, npmi))


def _entropy(counter: Counter[str]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    return -sum(
        (value / total) * math.log(value / total)
        for value in counter.values()
        if value > 0
    )


def _nested_coverage(
    candidates: dict[str, _TermStats],
    config: NewWordConfig,
) -> tuple[dict[str, float], list[tuple[str, str, float]]]:
    coverage: dict[str, float] = {}
    best_longer: dict[str, tuple[str, float]] = {}
    suppressions: list[tuple[str, str, float]] = []
    candidate_set = set(candidates)
    longer_terms = sorted(
        candidates,
        key=lambda item: (
            -len(item),
            -candidates[item].weighted_frequency,
            item,
        ),
    )
    for longer in longer_terms:
        longer_stats = candidates[longer]
        if len(longer) <= config.min_chars:
            continue
        for length in range(config.min_chars, len(longer)):
            for start in range(0, len(longer) - length + 1):
                short = longer[start:start + length]
                if short == longer or short not in candidate_set:
                    continue
                short_stats = candidates[short]
                if short_stats.weighted_frequency <= 0.0:
                    continue
                if (
                    longer_stats.weighted_frequency
                    < short_stats.weighted_frequency * 0.65
                ):
                    continue
                value = min(
                    1.0,
                    longer_stats.weighted_frequency
                    / max(1.0, short_stats.weighted_frequency),
                )
                current = best_longer.get(short)
                if current is None or (
                    value,
                    longer_stats.weighted_frequency,
                    len(longer),
                    longer,
                ) > (
                    current[1],
                    candidates[current[0]].weighted_frequency,
                    len(current[0]),
                    current[0],
                ):
                    best_longer[short] = (longer, value)

    for term, stats in candidates.items():
        if len(term) >= config.max_chars or stats.weighted_frequency <= 0.0:
            coverage[term] = 0.0
            continue
        best = best_longer.get(term)
        if best is None:
            coverage[term] = 0.0
            continue
        longer, value = best
        coverage[term] = value
        if value > 0.0:
            suppressions.append((term, longer, value))
    suppressions.sort(key=lambda item: (-item[2], item[0], item[1]))
    return coverage, suppressions


def _is_known_jieba_word(term: str, config: NewWordConfig) -> bool:
    return tuple(_get_tokenizer(config).cut(term, HMM=False)) == (term,)


def _is_function_phrase(term: str) -> bool:
    if term in _FUNCTION_PHRASES:
        return True
    if len(term) == 2:
        return (
            term[0] in _TWO_CHAR_FUNCTION_PREFIXES
            or term[-1] in _TWO_CHAR_FUNCTION_SUFFIXES
        )
    return False


def _reject_reasons(
    term: str,
    stats: _TermStats,
    npmi: float,
    boundary_entropy: float,
    nested_coverage: float,
    dynamic_stopwords: frozenset[str],
    config: NewWordConfig,
) -> list[str]:
    reasons: list[str] = []
    if term in _active_stop_terms(config):
        reasons.append("stop_term")
    if term in dynamic_stopwords:
        reasons.append("dynamic_stopword")
    if _is_function_phrase(term):
        reasons.append("function_phrase")
    if stats.frequency < config.min_frequency:
        reasons.append("frequency")
    if stats.document_count < config.min_documents:
        reasons.append("document_count")
    if stats.user_count < config.min_users:
        reasons.append("user_count")
    if npmi < config.min_npmi:
        reasons.append("npmi")
    if boundary_entropy < config.min_boundary_entropy:
        reasons.append("boundary_entropy")
    if nested_coverage > config.max_nested_coverage:
        reasons.append("nested_coverage")
    return reasons


def _status(
    known: bool,
    reject_reasons: list[str],
    npmi: float,
    boundary_entropy: float,
    nested_coverage: float,
    score: float,
    config: NewWordConfig,
) -> NewWordStatus:
    blocking_reasons = {
        "stop_term",
        "dynamic_stopword",
        "function_phrase",
        "frequency",
        "document_count",
        "user_count",
        "npmi",
        "nested_coverage",
    }
    if any(reason in blocking_reasons for reason in reject_reasons):
        return NewWordStatus.REJECTED
    if known:
        return NewWordStatus.KNOWN
    if not reject_reasons:
        return NewWordStatus.PROMOTED
    relaxed_npmi = config.min_npmi * 0.5
    relaxed_entropy = config.min_boundary_entropy * 0.5
    relaxed_score = config.min_promoted_score * 0.5
    if (
        npmi >= relaxed_npmi
        and boundary_entropy >= relaxed_entropy
        and nested_coverage <= config.max_nested_coverage
        and score >= relaxed_score
    ):
        return NewWordStatus.CANDIDATE
    return NewWordStatus.REJECTED


def _score_breakdown(
    stats: _TermStats,
    *,
    npmi: float,
    boundary_entropy: float,
    nested_coverage: float,
    repeat_penalty: float,
    dictionary_penalty: float,
) -> NewWordScoreBreakdown:
    frequency_score = math.log1p(stats.weighted_frequency)
    user_diversity = math.log1p(stats.user_count)
    cohesion_score = max(0.0, npmi)
    boundary_score = math.log1p(boundary_entropy)
    independence_score = max(0.0, 1.0 - nested_coverage)
    return NewWordScoreBreakdown(
        frequency=frequency_score,
        user_diversity=user_diversity,
        cohesion=cohesion_score,
        boundary=boundary_score,
        independence=independence_score,
        repeat=repeat_penalty,
        dictionary=dictionary_penalty,
    )


def _score(
    breakdown: NewWordScoreBreakdown,
    config: NewWordConfig,
) -> float:
    weighted_parts = (
        (breakdown.frequency, config.frequency_score_weight),
        (breakdown.user_diversity, config.user_diversity_score_weight),
        (breakdown.cohesion, config.cohesion_score_weight),
        (breakdown.boundary, config.boundary_score_weight),
        (breakdown.independence, config.independence_score_weight),
        (breakdown.repeat, config.repeat_score_weight),
        (breakdown.dictionary, config.dictionary_score_weight),
    )
    weight_total = sum(weight for _value, weight in weighted_parts)
    if weight_total <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in weighted_parts) / weight_total


def _normalize_score_breakdown(
    breakdown: NewWordScoreBreakdown,
    config: NewWordConfig,
) -> NewWordScoreBreakdown:
    return NewWordScoreBreakdown(
        frequency=_normalize_score_component(
            breakdown.frequency,
            config.frequency_score_norm,
        ),
        user_diversity=_normalize_score_component(
            breakdown.user_diversity,
            config.user_diversity_score_norm,
        ),
        cohesion=_normalize_score_component(
            breakdown.cohesion,
            config.cohesion_score_norm,
        ),
        boundary=_normalize_score_component(
            breakdown.boundary,
            config.boundary_score_norm,
        ),
        independence=_normalize_score_component(
            breakdown.independence,
            config.independence_score_norm,
        ),
        repeat=_normalize_score_component(
            breakdown.repeat,
            config.repeat_score_norm,
        ),
        dictionary=_normalize_score_component(
            breakdown.dictionary,
            config.dictionary_score_norm,
        ),
    )


def _normalize_score_component(value: float, normalizer: float) -> float:
    return max(0.0, min(1.0, value / normalizer))


def _rank_key(item: NewWordCandidate) -> tuple[int, float, int, int, str]:
    priority = {
        NewWordStatus.PROMOTED: 0,
        NewWordStatus.CANDIDATE: 1,
        NewWordStatus.KNOWN: 2,
        NewWordStatus.REJECTED: 3,
    }[item.status]
    return (
        priority,
        -item.score,
        -item.user_count,
        -item.frequency,
        item.term,
    )


def _append_section(
    lines: list[str],
    title: str,
    items: tuple[NewWordCandidate, ...],
    *,
    include_reasons: bool = False,
) -> None:
    lines.extend([f"## {title}"])
    if not items:
        lines.extend(["- none", ""])
        return
    for item in items:
        breakdown = item.score_breakdown
        normalized = item.normalized_score_breakdown
        line = (
            f"- {item.term}: score={item.score:.4f}, "
            f"score_parts=("
            f"freq={breakdown.frequency:.3f}, "
            f"user={breakdown.user_diversity:.3f}, "
            f"cohesion={breakdown.cohesion:.3f}, "
            f"boundary={breakdown.boundary:.3f}, "
            f"independence={breakdown.independence:.3f}, "
            f"repeat={breakdown.repeat:.3f}, "
            f"dictionary={breakdown.dictionary:.3f}), "
            f"normalized_parts=("
            f"freq={normalized.frequency:.3f}, "
            f"user={normalized.user_diversity:.3f}, "
            f"cohesion={normalized.cohesion:.3f}, "
            f"boundary={normalized.boundary:.3f}, "
            f"independence={normalized.independence:.3f}, "
            f"repeat={normalized.repeat:.3f}, "
            f"dictionary={normalized.dictionary:.3f}), "
            f"freq={item.frequency}, weighted={item.weighted_frequency:.2f}, "
            f"docs={item.document_count}, users={item.user_count}, "
            f"pmi={item.pmi:.4f}, npmi={item.npmi:.4f}, "
            f"boundary={item.boundary_entropy:.4f}, "
            f"nested={item.nested_coverage:.3f}, "
            f"repeat={item.repeat_penalty:.3f}, "
            f"evidence={','.join(item.evidence_message_ids)}"
        )
        if include_reasons:
            line += f", reject reasons={','.join(item.reject_reasons) or 'none'}"
        lines.append(line)
    lines.append("")
