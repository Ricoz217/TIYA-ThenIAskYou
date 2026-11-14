import copy
import importlib.util

import pytest
from ruamel.yaml import YAML

from TIYA import config
from TIYA.LLM_connect import parse_llm_setting
from test_llm_preset_calls import document


def test_root_returns_raw_snapshot_and_first_candidate_properties(monkeypatch):
    data = document()
    data["Providers"]["chat_completions"]["primary"]["max_context"] = 99999
    original = copy.deepcopy(data)
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(data))
    preset = config.get_llm("MAIN")
    assert preset.configuration == original
    assert preset.data == original["LLM_List"]["MAIN"]
    assert preset.provider_definitions == original["Providers"]
    assert preset.max_context == preset["max_context"] == 10000
    assert preset.extra_parameter is preset.providers[0]["llm_params"]
    assert "model" not in preset.providers[0]
    assert "endpoint" not in preset.providers[0]
    preset.providers[0]["max_context"] = 12000
    assert preset.max_context == 12000
    assert config.get_llm("MAIN").max_context == 10000
    assert config.LLM_CFG == original


def test_removed_root_business_helpers_and_configuration_migrator():
    for name in (
        "get_llm_credential",
        "get_llm_endpoint",
        "resolve_llm_connection",
        "resolve_llm_preset",
        "migrate_llm_config",
    ):
        assert not hasattr(config, name)
    assert importlib.util.find_spec("TIYA.llm.migrate") is None


def test_root_does_not_inherit_provider_window_or_interpret_reference(monkeypatch):
    data = document()
    data["Providers"]["chat_completions"]["primary"]["max_context"] = 99999
    data["LLM_List"]["MAIN"]["providers"][0].pop("max_context")
    data["LLM_List"]["MAIN"]["providers"][0]["llm"] = "not-a-reference"
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(data))
    assert config.get_llm("MAIN").max_context is None
    assert config.get_llm("MAIN").providers[0].llm == "not-a-reference"
    with pytest.raises(ValueError, match="引用"):
        parse_llm_setting("MAIN")


def test_loader_checks_shape_without_resolving_connections_or_repairing_llm(
    tmp_path, monkeypatch
):
    data = document()
    data["LLM_List"]["MAIN"]["providers"][0]["llm"] = "future:missing:model"
    data["LLM_List"]["MAIN"]["custom"] = {"keep": False}
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as stream:
        YAML().dump({config.TITLE_LLM: data}, stream)
    manager = config.Configer(config_path=path, comments={})
    result = manager.load_config()
    assert not [e for e in result.report.errors if e.path[:1] == (config.TITLE_LLM,)]
    assert manager.get_llm_config() == data
    monkeypatch.setattr(config, "LLM_CFG", manager.get_llm_config())
    assert config.get_llm("MAIN").data.custom.keep is False
    with pytest.raises(ValueError, match="missing"):
        parse_llm_setting("MAIN")


@pytest.mark.parametrize("version", [None, 1])
def test_old_config_is_rejected_without_rewriting_source(
    tmp_path, monkeypatch, version
):
    data = {"LLM_List": [{"preset_name": "MAIN", "model": "old"}]}
    if version is not None:
        data["schema_version"] = version
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as stream:
        YAML().dump({config.TITLE_LLM: data}, stream)
    manager = config.Configer(config_path=path, comments={})
    result = manager.load_config()
    errors = [e for e in result.report.errors if e.path[:1] == (config.TITLE_LLM,)]
    assert errors and "人工" in errors[0].message
    assert manager.get_llm_config() == data
    monkeypatch.setattr(config, "LLM_CFG", manager.get_llm_config())
    for read in (
        lambda: config.get_llm("MAIN"),
        config.get_llm_list,
        lambda: parse_llm_setting("MAIN"),
    ):
        with pytest.raises(ValueError, match="人工"):
            read()
    assert (
        YAML(typ="safe").load(path.read_text(encoding="utf-8"))[config.TITLE_LLM]
        == data
    )
