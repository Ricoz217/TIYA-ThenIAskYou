from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm_pipeline import LLMPresetConfigError
from TIYA.config import get_llm

DEFAULT_MAX_CONTEXT_WINDOW = 1_000_000

def _resolve_effective_max_context_window(llm_preset: str) -> int:
    preset_name = str(llm_preset or "").strip()
    if not preset_name:
        return DEFAULT_MAX_CONTEXT_WINDOW

    try:
        llm_cfg = get_llm(preset_name)
    except Exception as exc:
        raise LLMPresetConfigError(f"failed to load llm preset: {preset_name}") from exc

    llm_max = getattr(llm_cfg, "max_context", None)
    if isinstance(llm_max, (int, float)) and llm_max > 0:
        return int(llm_max)
    raise LLMPresetConfigError(
        f"llm preset <{preset_name}> missing valid max_context; please set it in config."
    )

TOOL_PRESET_KEYS: tuple[str, ...] = (
    "clean",
    "ingest",
    "query",
    "compress",
    "bucket_split",
    "text_chunk",
    "bucket_summary",
    "optimize",
    "image_extract",
)

def _normalize_tool_presets(tool_presets: dict[str, str] | None) -> dict[str, str]:
    if not isinstance(tool_presets, dict):
        return {}
    normalized: dict[str, str] = {}
    for k, v in tool_presets.items():
        key = str(k).strip().lower()
        val = str(v).strip()
        if key in TOOL_PRESET_KEYS and val:
            normalized[key] = val
    return normalized

@dataclass(slots=True)
class ContextMemoryConfig:
    base_dir: str | Path | None = None
    llm_preset: str = ""
    image_llm_preset: str = ""
    tool_presets: dict[str, str] = field(default_factory=dict)
    ask_timeout: float = 300
    auto_resume_pending_jobs: bool = False
    use_mock_llm: bool = False
    enable_cleaning: bool = True
    init_config: bool = True
    evidence_versions: int = 5
    auto_manage: bool = True
    enable_forgetting: bool = True
    max_bucket_depth: int = 5
    max_memory_bytes: int = 1_000_000_000
    auto_compress_trigger_ratio: float = 0.70
    auto_split_trigger_ratio: float = 0.50
    split_plan_target_items: int = 180
    split_plan_hard_cap: int = 250
    auto_split_cooldown_sec: int = 600
    auto_split_min_drop_abs: float = 0.03
    auto_split_max_round_per_manage: int = 1
    split_ingest_parallelism: int = 16
    split_ingest_delay_min: float = 1.0
    split_ingest_delay_max: float = 3.0
    optimize_leaf_loss_threshold: float = 0.03
    gc_revision_retention_days: int = 14
    gc_gray_key_retention_days: int = 45
    gc_archived_bucket_retention_days: int = 45
    query_top_k_default: int = 5
    query_branch_expand_k: int = 5
    query_branch_expand_bind_top_k: bool = False
    query_mode_default: str = "auto"
    global_recall_top_n: int = 120
    global_recall_top_m: int = 8
    global_recall_depth_limit: int = 8
    global_recall_time_budget_ms: int = 80
    global_recall_boost_weight: float = 0.20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextMemoryConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(
            base_dir=data.get("base_dir"),
            llm_preset=str(data.get("llm_preset", "")),
            image_llm_preset=str(data.get("image_llm_preset", "")),
            tool_presets=_normalize_tool_presets(data.get("tool_presets")),
            ask_timeout=float(data.get("ask_timeout", 180.0)),
            auto_resume_pending_jobs=bool(data.get("auto_resume_pending_jobs", True)),
            use_mock_llm=bool(data.get("use_mock_llm", False)),
            enable_cleaning=bool(data.get("enable_cleaning", True)),
            init_config=bool(data.get("init_config", True)),
            evidence_versions=int(data.get("evidence_versions", 5)),
            auto_manage=bool(data.get("auto_manage", True)),
            enable_forgetting=bool(data.get("enable_forgetting", True)),
            max_bucket_depth=int(data.get("max_bucket_depth", 3)),
            max_memory_bytes=int(data.get("max_memory_bytes", 1_000_000_000)),
            auto_compress_trigger_ratio=float(data.get("auto_compress_trigger_ratio", 0.70)),
            auto_split_trigger_ratio=float(data.get("auto_split_trigger_ratio", 0.50)),
            split_plan_target_items=int(data.get("split_plan_target_items", 180)),
            split_plan_hard_cap=int(data.get("split_plan_hard_cap", 250)),
            auto_split_cooldown_sec=int(data.get("auto_split_cooldown_sec", 600)),
            auto_split_min_drop_abs=float(data.get("auto_split_min_drop_abs", 0.03)),
            auto_split_max_round_per_manage=int(data.get("auto_split_max_round_per_manage", 1)),
            split_ingest_parallelism=int(data.get("split_ingest_parallelism", 16)),
            split_ingest_delay_min=float(data.get("split_ingest_delay_min", 1.0)),
            split_ingest_delay_max=float(data.get("split_ingest_delay_max", 3.0)),
            optimize_leaf_loss_threshold=float(data.get("optimize_leaf_loss_threshold", 0.03)),
            gc_revision_retention_days=int(data.get("gc_revision_retention_days", 14)),
            gc_gray_key_retention_days=int(data.get("gc_gray_key_retention_days", 45)),
            gc_archived_bucket_retention_days=int(data.get("gc_archived_bucket_retention_days", 45)),
            query_top_k_default=int(data.get("query_top_k_default", 5)),
            query_branch_expand_k=int(data.get("query_branch_expand_k", 5)),
            query_branch_expand_bind_top_k=bool(data.get("query_branch_expand_bind_top_k", False)),
            query_mode_default=str(data.get("query_mode_default", "auto")),
            global_recall_top_n=int(data.get("global_recall_top_n", 120)),
            global_recall_top_m=int(data.get("global_recall_top_m", 8)),
            global_recall_depth_limit=int(data.get("global_recall_depth_limit", 8)),
            global_recall_time_budget_ms=int(data.get("global_recall_time_budget_ms", 80)),
            global_recall_boost_weight=float(data.get("global_recall_boost_weight", 0.20)),
        )
