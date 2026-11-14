import copy

import pytest


def document():
    return {
        "schema_version": 2,
        "Providers": {
            "responses_api": {
                "arbitrary": {
                    "endpoint": "https://test/responses",
                    "key": "secret",
                    "llm_params": {"temperature": 1, "stream": True},
                    "unknown": {"keep": 1},
                }
            },
            "chat_completions": {
                "backup": {"base_url": "https://backup/v1", "key": "other"}
            },
            "files_api": {
                "upload": {"endpoint": "https://test/files", "key": "secret"}
            },
            "custom_tool": {"anything": {"no_endpoint": True}},
        },
        "LLM_List": {
            "MAIN": {
                "auto_compress_rate": 0.7,
                "proxy_mode": "",
                "providers": [
                    {
                        "llm": "responses_api:arbitrary:namespace:model",
                        "max_context": 10000,
                        "llm_params": {"temperature": 0, "stream": False, "custom": 4},
                        "files_api": "files_api:upload",
                    },
                    {"llm": "chat_completions:backup:small", "max_context": 2000},
                ],
            }
        },
    }


def test_resolver_preserves_candidates_and_native_parameters():
    from TIYA.config import ConfigMap, LLMPresetConfig
    from TIYA.llm.config import resolve_preset

    data = document()
    before = copy.deepcopy(data)
    result = resolve_preset(LLMPresetConfig("MAIN", ConfigMap(data)))
    assert result["providers"][0]["model"] == "namespace:model"
    assert result["providers"][0]["llm_params"] == {
        "temperature": 0,
        "stream": False,
        "custom": 4,
    }
    assert result["providers"][1]["endpoint"] == "https://backup/v1/chat/completions"
    result["providers"][0]["llm_params"]["custom"] = 8
    assert data == before
    view = resolve_preset(LLMPresetConfig("MAIN", ConfigMap(data)))
    assert view["preset_name"] == "MAIN"
    assert view["providers"][0]["max_context"] == 10000
    assert "token" not in data["LLM_List"]["MAIN"]


def test_loader_keeps_v2_without_legacy_template(tmp_path, monkeypatch):
    from TIYA import config
    from ruamel.yaml import YAML

    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as stream:
        YAML().dump({config.TITLE_LLM: document()}, stream)
    manager = config.Configer(config_path=path, comments={})
    manager.load_config()
    monkeypatch.setattr(config, "LLM_CFG", manager.get_llm_config())
    assert config.get_llm_list() == ["MAIN"]
    assert (
        config.get_llm("MAIN").providers[0].llm
        == "responses_api:arbitrary:namespace:model"
    )
    assert isinstance(manager.get_llm_config().LLM_List, dict)
    assert "token" not in manager.get_llm_config().LLM_List.MAIN


def test_missing_reference_has_precise_error():
    from TIYA.config import ConfigMap, LLMPresetConfig
    from TIYA.llm.config import resolve_preset

    data = document()
    data["LLM_List"]["MAIN"]["providers"][0]["llm"] = "responses_api:missing:model"
    with pytest.raises(ValueError, match="responses_api:missing"):
        resolve_preset(LLMPresetConfig("MAIN", ConfigMap(data)))


def test_switching_presets_clears_previous_routes_proxy_and_parameters(monkeypatch):
    import asyncio
    import json
    import httpx
    from TIYA import LLM_connect, config
    from TIYA.config import ConfigMap
    from TIYA.LLM_connect import Chat

    v2 = document()
    v2["Providers"]["responses_api"]["arbitrary"]["client_params"] = {"verify": False}
    v2["LLM_List"]["MAIN"]["providers"][0]["llm_params"]["stale"] = True
    monkeypatch.setattr(config, "LLM_CFG", ConfigMap(v2))
    old = {
        "schema_version": 2,
        "Providers": {
            "chat_completions": {
                "plain": {"endpoint": "https://old/chat/completions", "key": "dummy"}
            }
        },
        "LLM_List": {
            "old": {
                "proxy_mode": "",
                "providers": [
                    {
                        "llm": "chat_completions:plain:old",
                        "max_context": 10000,
                        "llm_params": {"stream": False},
                    }
                ],
            }
        },
    }
    requests, clients = [], []
    real_client = httpx.AsyncClient

    def handler(request):
        requests.append(request)
        if request.url.host == "old":
            return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "stale fallback"}}]}
        )

    def client(**params):
        clients.append(params.copy())
        params.update(
            mounts={}, trust_env=False, transport=httpx.MockTransport(handler)
        )
        return real_client(**params)

    monkeypatch.setattr(httpx, "AsyncClient", client)

    async def run():
        async with Chat(
            config=LLM_connect.parse_llm_setting("MAIN"), log_off=True
        ) as chat:
            monkeypatch.setattr(config, "LLM_CFG", ConfigMap(old))
            chat.setting(LLM_connect.parse_llm_setting("old"))
            assert await chat.ask("question", timeout=5) is None
            assert len(requests) == 1  # The previous preset's backup must not be tried.
            payload = json.loads(requests[0].content)
            assert payload["model"] == "old"
            assert "stale" not in payload and "input" not in payload
            assert clients[0].get("verify", True) is True
            assert not clients[0]["mounts"]
            assert chat.model["model"] == "old"

    asyncio.run(run())
