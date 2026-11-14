import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace


def test_shared_fingerprint_keeps_existing_serialization_and_hash():
    from TIYA.utils import fingerprint

    value = MappingProxyType({"中文": ("x", 2), "nested": {"b": True, "a": None}})
    assert fingerprint(value) == "8102b4c2eb206fee93618349873bbe5906c2757ba58fc7c520fd597a80d7811b"
    reordered = {"nested": {"a": None, "b": True}, "中文": ["x", 2]}
    assert fingerprint(reordered) == fingerprint(value)


def test_model_content_normalization_keeps_existing_hash():
    from TIYA.llm.model import stable
    from TIYA.utils import fingerprint

    value = {"image": b"image-bytes", "object": SimpleNamespace(items=(1, 2))}
    assert fingerprint(stable(value)) == "c4ce5a3b398f5997a89e21b7426348f9431750c1a7a3bf9f636c5a50a8bdc5a0"
    assert value["image"] == b"image-bytes"
    assert value["object"].items == (1, 2)


def test_shared_utilities_and_models_do_not_import_upper_llm_layers():
    script = r'''
import importlib.abc
import sys

class BlockUpperLayers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"TIYA.llm.config", "TIYA.llm.usage", "TIYA.llm.snapshot"}:
            raise AssertionError("Unexpected reverse import: " + fullname)

sys.meta_path.insert(0, BlockUpperLayers())
from TIYA.utils import fingerprint
assert fingerprint({"value": 1})
assert not any(name.startswith("TIYA.llm") for name in sys.modules)
from TIYA.llm.model import stable
assert stable({"value": (1, 2)}) == {"value": [1, 2]}
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
