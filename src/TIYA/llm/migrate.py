"""Explicit, backed-up configuration migration: python -m TIYA.llm.migrate PATH [--apply]."""

from __future__ import annotations

import argparse
import copy
import io
import os
from pathlib import Path
from uuid import uuid4

from ruamel.yaml import YAML

from .config import migrate_legacy, resolve_preset
from .pricing import PriceCatalog


def migrate_file(path, *, apply=False, pricing_path=None):
    path = Path(path).resolve()
    original = path.read_bytes()
    yaml = YAML()
    yaml.allow_duplicate_keys = False
    document = yaml.load(original.decode("utf-8"))
    title = "LLM模型配置"
    if not isinstance(document, dict) or not isinstance(document.get(title), dict):
        raise ValueError("配置缺少 LLM模型配置 字典")
    if document[title].get("schema_version") == 2:
        for name in document[title].get("LLM_List", {}):
            resolve_preset(document[title], name)
        return document
    migrated = migrate_legacy(document[title])
    for name in migrated["LLM_List"]:
        resolve_preset(migrated, name)
    result = copy.deepcopy(document)
    result[title] = migrated
    if not apply:
        return result
    catalog = PriceCatalog(
        pricing_path or path.parent.parent / "data/llm/pricing.sqlite3"
    )
    for name, preset in migrated["LLM_List"].items():
        price = preset.pop("legacy_price", None)
        if price is None:
            continue
        route = resolve_preset(migrated, name).routes[0]
        identity = {
            "preset": name,
            "scope": route.scope,
            "model": route.model,
            "operation": route.category,
        }
        rules = {
            "currency": price.get("currency", "CNY"),
            "threshold": "input",
            "tier_mode": "whole",
            "units": {"input_uncached": 1000000, "cached": 1000000, "output": 1000000},
            "tiers": [
                {
                    "rates": {
                        "input_uncached": str(price.get("input_token", 0)),
                        "cached": str(price.get("cache_hit", 0)),
                        "output": str(price.get("output_token", 0)),
                    }
                }
            ],
        }
        old = catalog.snapshot(identity)
        if old is None or old["rules"] != rules:
            catalog.publish(identity, rules, source="legacy-config-migration")
    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists() and backup.read_bytes() != original:
        backup = path.with_suffix(path.suffix + "." + uuid4().hex + ".bak")
    if not backup.exists():
        with backup.open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
    output = io.StringIO()
    yaml.dump(result, output)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(output.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--pricing", type=Path)
    args = parser.parse_args()
    result = migrate_file(args.path, apply=args.apply, pricing_path=args.pricing)
    names = list(result["LLM模型配置"]["LLM_List"])
    print(
        f"{'已迁移' if args.apply else '预览，未写入'}: {len(names)} 个预设；不显示凭据。"
    )


if __name__ == "__main__":
    main()
