import asyncio
import copy
import json

import httpx
import pytest
from test_llm_preset_calls import document, mock_model

from TIYA import config
from TIYA.LLM_connect import Chat, ChatConfig, parse_llm_setting


def test_get_llm_returns_complete_preset_data_not_chat_config(monkeypatch):
    data = document()
    data["LLM_List"]["MAIN"]["custom"] = {"kept": True}
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(copy.deepcopy(data)))
    preset = config.get_llm("MAIN")
    assert isinstance(preset, config.LLMPresetConfig)
    assert not isinstance(preset, ChatConfig)
    assert [entry.llm for entry in preset.providers] == [
        "chat_completions:primary:model",
        "chat_completions:backup:other-model",
    ]
    assert preset.data.custom.kept is True
    assert preset.max_context == 10000
    preset.providers[0].llm_params["temperature"] = 1
    assert config.get_llm("MAIN").providers[0].llm_params["temperature"] == 0


def test_chat_config_primary_properties_have_one_source_of_truth(monkeypatch):
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(document()))
    setting = parse_llm_setting("MAIN")
    assert len(setting.candidates) == 2
    assert setting.model == setting.candidates[0].model == "model"
    assert setting.model_params is setting.candidates[0].model_params
    setting.model_params["stream"] = True
    assert setting.candidates[0].model_params["stream"] is True
    setting.candidates[0].model = "changed"
    assert setting.model == "changed"
    setting.model_params = {"stream": False}
    assert setting.candidates[0].model_params == {"stream": False}
    assert setting.candidates[1].model_params == {"stream": False}
    assert not hasattr(setting, "preset") and not hasattr(setting, "complete")


def test_parse_uses_get_llm_and_preserves_provider_options(monkeypatch):
    data = document()
    data["Providers"]["chat_completions"]["primary"]["custom_protocol"] = {
        "enabled": True
    }
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(data))
    calls = []
    real_get_llm = config.get_llm

    def get_llm(name):
        calls.append(name)
        return real_get_llm(name)

    # settings binds the root reader at import time; patch its actual lookup site.
    monkeypatch.setattr("TIYA.llm.settings.get_llm", get_llm)
    setting = parse_llm_setting("MAIN")
    assert calls == ["MAIN"]
    assert setting.candidates[0].provider_options["custom_protocol"] == {
        "enabled": True
    }
    assert "endpoint" not in setting.candidates[0].provider_options
    assert "key" not in setting.candidates[0].provider_options
    assert "custom_protocol" not in setting.model_params


def test_memory_existing_config_read_and_partial_constructor_still_work(
    monkeypatch, tmp_path
):
    from TIYA.memory.config import _resolve_effective_max_context_window
    from TIYA.memory.llm_pipeline import LLMPipelineV3

    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(document()))
    assert _resolve_effective_max_context_window("MAIN") == 10000
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"ok":true}'}}]}
        )

    mock_model(monkeypatch, handler)
    (tmp_path / "system.md").write_text("system", encoding="utf-8")
    (tmp_path / "schema.md").write_text("{}", encoding="utf-8")
    pipeline = LLMPipelineV3(tmp_path, llm_preset="MAIN", max_retries=1)
    result = asyncio.run(
        pipeline._ask_json(
            prompt_filename="system.md",
            schema_filename="schema.md",
            user_payload={"question": "value"},
            bucket_context=None,
            include_context=False,
        )
    )
    assert result == {"ok": True}
    assert sent[0]["model"] == "model"
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert sent[0]["temperature"] == 0


def test_partial_update_keeps_backup_and_snapshot_isolation(monkeypatch):
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(document()))
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if request.url.host == "primary.invalid":
            return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "backup"}}]}
        )

    mock_model(monkeypatch, handler)

    async def run():
        supplied = parse_llm_setting("MAIN")
        async with Chat(config=supplied, log_off=True) as chat:
            supplied.candidates[1].model = "must-not-leak"
            chat.update_config(model_params={"stream": False, "custom": 4})
            assert (await chat.ask("question", timeout=5))[0].text == "backup"
        assert requests[0]["custom"] == 4
        assert "custom" not in requests[1]
        assert requests[1]["model"] == "other-model"

    asyncio.run(run())


@pytest.mark.parametrize("name, error", [(None, TypeError), ("missing", KeyError)])
def test_parser_keeps_missing_preset_and_type_errors(monkeypatch, name, error):
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(document()))
    with pytest.raises(error):
        parse_llm_setting(name)


def test_candidate_validation_and_session_options_have_explicit_semantics(monkeypatch):
    from TIYA.LLM_connect import LLMConfig, RequestPolicy

    with pytest.raises(ValueError, match="candidates"):
        ChatConfig([])
    with pytest.raises(ValueError, match="max_attempts"):
        RequestPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="max_retries"):
        RequestPolicy(max_retries_per_candidate=-1)
    with pytest.raises(ValueError, match="fallback_cooldown"):
        RequestPolicy(fallback_cooldown_seconds=-1)
    supplied = ChatConfig(
        [LLMConfig("chat_completions", "a", "https://test/chat", "dummy", "m")]
    )
    supplied.endpoint = "https://other/chat"
    supplied.token = "other"
    supplied.api_provider = "responses_api"
    supplied.max_context = 2000
    supplied.client_params = {"verify": False}
    assert supplied.candidates[0].endpoint == "https://other/chat"
    assert supplied.candidates[0].token == "other"
    assert supplied.candidates[0].max_context == 2000

    async def run():
        async with Chat(config=supplied, log_off=True) as chat:
            # A direct Responses candidate must not acquire legacy Chat defaults.
            assert "max_tokens" not in chat.model["parameter"]
            chat.update_config(system_prompt="updated", log_off=True)
            assert (await chat.get_context()).system.text == "updated"
            assert chat.config.primary.client_params == {"verify": False}

    asyncio.run(run())


def test_concurrent_first_use_of_separate_sqlite_stores(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from TIYA.llm.state import StateStore

    barrier = Barrier(8)
    path = tmp_path / "shared.sqlite3"

    def write(index):
        store = StateStore(path)
        barrier.wait()
        store.put("test", str(index), {"value": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(8)))
    store = StateStore(path)
    assert [store.get("test", str(i))["value"] for i in range(8)] == list(range(8))


def test_files_binding_preserves_existing_upload_cache_scope(monkeypatch):
    from test_llm_config_v2 import document as file_document

    from TIYA.utils import fingerprint
    from TIYA.llm.provider.files import FilesProvider

    data = file_document()
    data["Providers"]["files_api"]["upload"]["expires_after_seconds"] = 86400
    data["LLM_List"]["MAIN"]["providers"][0]["files_api_params"] = {
        "expires_after_seconds": 7200
    }
    before = copy.deepcopy(data)
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(data))
    setting = parse_llm_setting("MAIN")
    files = setting.candidates[0].files
    expected = {
        "endpoint": "https://test/files",
        "key": "secret",
        "expires_after_seconds": 7200,
    }
    assert files.ref == "files_api:upload"
    # Old upload mappings used the bound Files node fingerprint. New types must retain it.
    assert FilesProvider(files.as_provider_settings()).scope == fingerprint(expected)
    assert files.provider_options["expires_after_seconds"] == 7200
    assert setting.model_params["temperature"] == 0
    assert dict(config.LLM_CFG) == before


@pytest.mark.parametrize("bad", [False, 0, [], "invalid"])
def test_invalid_native_parameter_container_is_rejected_before_http(monkeypatch, bad):
    data = document()
    data["LLM_List"]["MAIN"]["providers"][0]["llm_params"] = bad
    monkeypatch.setattr(config, "LLM_CFG", config.ConfigMap(data))
    with pytest.raises((TypeError, ValueError), match="llm_params"):
        parse_llm_setting("MAIN")
