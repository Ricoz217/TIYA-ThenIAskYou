"""默认 YAML 必须能展示并解析新版 LLM 连接树和预设。"""

import copy

from ruamel.yaml import YAML

from TIYA import config
from TIYA.llm.settings import resolve_chat_config


def test_generated_default_template_resolves_all_presets_without_network(tmp_path, monkeypatch):
    path = tmp_path / "default.yaml"
    manager = config.Configer(path)
    monkeypatch.setattr(config, "CONFIGER", manager)
    monkeypatch.setattr(config, "QQ_GROUPS", {})
    assert config.write_default_config().ok
    document = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    llm = document[config.TITLE_LLM]
    assert llm["schema_version"] == 2
    assert {"chat_completions", "responses_api", "anthropic_api", "files_api", "image_generation"} <= set(llm["Providers"])
    assert {"YourModelPresetName", "本地模型", "Responses示例", "Anthropic示例"} <= set(llm["LLM_List"])
    for name in llm["LLM_List"]:
        setting = resolve_chat_config(llm, name)
        assert setting.candidates
        assert all(route.endpoint for route in setting.candidates)

    main = resolve_chat_config(llm, "YourModelPresetName")
    assert main.primary.provider_options["error_dialect"] == "deepseek"
    assert main.primary.files.provider_options["expires_after_seconds"] == 86400
    response = resolve_chat_config(llm, "Responses示例")
    assert [route.provider_type for route in response.candidates] == ["responses_api", "chat_completions"]
    assert response.candidates[1].files is None
    assert manager.load_config().report.errors == ()


def test_generated_template_explains_model_ids_fallback_and_native_parameters(tmp_path):
    manager = config.Configer(tmp_path / "default.yaml")
    manager._atomic_write(manager.config_path, manager.build_default_document())
    text = manager.config_path.read_text(encoding="utf-8")
    for explanation in ("分类:连接名:模型ID", "fallback", "原生请求参数", "独立生图", "不自动迁移"):
        assert explanation in text


def test_loading_user_configuration_does_not_insert_example_connections(tmp_path):
    manager = config.Configer(tmp_path / "config.yaml")
    document = copy.deepcopy(config.DEFAULT_CONFIG)
    document[config.TITLE_LLM] = {"schema_version": 2, "Providers": {}, "LLM_List": {}}
    result = manager.load_config(document)
    assert result.report.errors == ()
    assert manager.get_llm_config().Providers == {}
    assert manager.get_llm_config().LLM_List == {}


def test_template_proxy_name_is_resolved_for_each_candidate(monkeypatch):
    from TIYA.llm.settings import parse_llm_setting

    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(copy.deepcopy(config.DEFAULT_CONFIG[config.TITLE_LLM])))
    monkeypatch.setattr(config, "BASE_CFG", config.ConfigMap({"Proxies": {"GlobalProxy": {"http": "http://proxy.test", "https": "http://proxy.test"}}}))
    resolved = parse_llm_setting("Responses示例")
    assert all(route.client_params.get("proxies") == {"http": "http://proxy.test", "https": "http://proxy.test"} for route in resolved.candidates)
