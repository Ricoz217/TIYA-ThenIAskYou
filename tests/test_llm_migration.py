from pathlib import Path

import pytest
from ruamel.yaml import YAML


def test_explicit_migration_backs_up_and_is_idempotent(tmp_path):
    from TIYA.llm.migrate import migrate_file

    path = tmp_path / "config.yaml"
    original = "LLM模型配置:\n  LLM_List:\n    - preset_name: old\n      model: m\n      endpoint: https://test/chat\n      token: dummy\n      api_type: openai\n      price:\n        input_token: 1\n        output_token: 2\n        cache_hit: 0\n"
    path.write_text(original, encoding="utf-8")
    preview = migrate_file(path)
    assert preview["LLM模型配置"]["schema_version"] == 2
    assert path.read_text(encoding="utf-8") == original
    migrate_file(path, apply=True, pricing_path=tmp_path / "pricing.sqlite3")
    assert path.with_suffix(".yaml.bak").read_text(encoding="utf-8") == original
    changed = path.read_bytes()
    migrate_file(path, apply=True, pricing_path=tmp_path / "pricing.sqlite3")
    assert path.read_bytes() == changed
    from TIYA.llm.config import resolve_preset
    from TIYA.llm.pricing import PriceCatalog

    route = resolve_preset(preview["LLM模型配置"], "old").routes[0]
    assert (
        PriceCatalog(tmp_path / "pricing.sqlite3").snapshot(
            {
                "preset": "old",
                "scope": route.scope,
                "model": "m",
                "operation": "chat_completions",
            }
        )["rules"]["tiers"][0]["rates"]["cached"]
        == "0"
    )


def test_duplicate_yaml_keys_fail_without_touching_source(tmp_path):
    from TIYA.llm.migrate import migrate_file

    path = tmp_path / "config.yaml"
    original = "LLM模型配置:\n  schema_version: 2\n  schema_version: 2\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(Exception):
        migrate_file(path, apply=True)
    assert path.read_text(encoding="utf-8") == original
