"""Root configuration must work even when the communication package is unavailable."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("load_file", [False, True])
def test_root_preset_lookup_and_loading_do_not_import_llm(tmp_path, version, load_file):
    if version == 1:
        data = {
            "LLM_List": [
                {
                    "preset_name": "MAIN",
                    "model": "model",
                    "endpoint": "https://test/chat",
                    "token": "dummy",
                    "api_type": "openai",
                    "max_context": 10000,
                    "auto_compress_rate": 0.7,
                    "proxy_mode": "",
                    "price": {},
                    "extra_parameter": {"stream": False},
                }
            ]
        }
    else:
        data = {
            "schema_version": 2,
            "Providers": {
                "responses_api": {
                    "primary": {"endpoint": "https://test/responses", "key": "dummy"},
                },
                "files_api": {
                    "upload": {"endpoint": "https://test/files", "key": "dummy"}
                },
            },
            "LLM_List": {
                "MAIN": {
                    "auto_compress_rate": 0.7,
                    "providers": [
                        {
                            "llm": "responses_api:primary:namespace:model",
                            "max_context": 10000,
                            "files_api": "files_api:upload",
                            "llm_params": {"stream": False},
                        }
                    ],
                }
            },
        }
    path = tmp_path / "config.yaml"
    # JSON is also valid YAML; no production configuration is written by this test.
    path.write_text(
        json.dumps({"LLM模型配置": data}, ensure_ascii=False), encoding="utf-8"
    )
    script = r"""
import importlib.abc
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
class BlockCommunication(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "TIYA.llm" or fullname.startswith("TIYA.llm."):
            raise AssertionError("Root configuration imported communication layer: " + fullname)
sys.meta_path.insert(0, BlockCommunication())
from TIYA import config
data = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if sys.argv[3] == "True":
    manager = config.Configer(config_path=Path(sys.argv[2]), comments={})
    manager.load_config()
    config.LLM_CFG = manager.get_llm_config()
else:
    config.LLM_CFG = config.ConfigMap(data[config.TITLE_LLM])
if isinstance(config.LLM_CFG.get("LLM_List"), list):
    try:
        config.get_llm("MAIN")
    except ValueError as exc:
        assert "人工" in str(exc)
    else:
        raise AssertionError("Old configuration must be rejected")
else:
    preset = config.get_llm("MAIN")
    assert isinstance(preset, config.LLMPresetConfig)
    assert config.get_llm_list() == ["MAIN"]
    assert preset.max_context == 10000
    assert len(preset.providers) == 1
    assert preset.providers[0].llm == "responses_api:primary:namespace:model"
    assert "model" not in preset.providers[0]
assert not any(name == "TIYA.llm" or name.startswith("TIYA.llm.") for name in sys.modules)
print("root configuration stayed independent")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).parents[1] / "src"),
            str(path),
            str(load_file),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "root configuration stayed independent" in result.stdout
