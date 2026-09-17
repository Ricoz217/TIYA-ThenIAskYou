from __future__ import annotations

import copy
from pathlib import Path

from ruamel.yaml import YAML

from TIYA import config


TITLE_BASE = "BOT基本配置"
TITLE_LLM = "LLM模型配置"
TITLE_GROUPS = "群聊设置"
TITLE_PRIVATE = "私聊设置"
TITLE_SETTING = "其他参数配置，请勿随意更改"


def _defaults() -> tuple[dict, dict]:
    default_config = {
        TITLE_BASE: {
            "BotInfo": {"name": "default-name", "uid": ""},
            "AdminList": ["114514", "1919810"],
            "NestedList": [["inner"]],
            "OptionalList": [],
        },
        TITLE_LLM: {
            "LLM_List": [
                {
                    "preset_name": "template",
                    "model": "default-model",
                    "max_context": 128_000,
                    "extra_parameter": {"temperature": 0.7},
                },
                {
                    "preset_name": "local",
                    "model": "local-model",
                    "max_context": 128_000,
                    "extra_parameter": {"temperature": 0.7},
                },
            ]
        },
        TITLE_GROUPS: {
            "Groups": {
                "114514": {"name": "示例群", "chat": True, "ban_topic": []},
                "Group_Default_Setting": {
                    "name": "",
                    "chat": True,
                    "ban_topic": [],
                },
            }
        },
    }
    default_setting = {
        "Common": {"EnableConfigCheck": True, "Timeout": 15},
        "Relative": {"InterestTimeout": 90, "TFIDFVectorDimension": 262_144},
    }
    return default_config, default_setting


def _complete_document() -> dict:
    default_config, default_setting = _defaults()
    return {
        **copy.deepcopy(default_config),
        TITLE_SETTING: copy.deepcopy(default_setting),
    }


def _dump(path: Path, data: dict) -> None:
    yaml = YAML()
    with path.open("w", encoding="utf-8") as stream:
        yaml.dump(data, stream)


def _load(path: Path) -> dict:
    yaml = YAML(typ="safe")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.load(stream)


def _manager(path: Path) -> config.Configer:
    default_config, default_setting = _defaults()
    return config.Configer(
        config_path=path,
        default_config=default_config,
        default_setting=default_setting,
        comments={},
    )


def _activate(monkeypatch, manager: config.Configer) -> None:
    monkeypatch.setattr(config, "CONFIGER", manager)
    monkeypatch.setattr(config, "BASE_CFG", manager.get_base_config())
    monkeypatch.setattr(config, "LLM_CFG", manager.get_llm_config())
    monkeypatch.setattr(config, "GROUPS_CFG", manager.get_group_config())
    monkeypatch.setattr(config, "SETTING_CFG", manager.get_setting_config())
    monkeypatch.setattr(config, "ALL_CFG", manager.get_all_config())


def test_setting_missing_values_are_materialized_once(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    document[TITLE_SETTING] = {"Common": {"EnableConfigCheck": True}}
    _dump(path, document)
    manager = _manager(path)
    writes: list[Path] = []
    original_write = manager._atomic_write

    def recording_write(target: Path, document_to_write: dict) -> None:
        writes.append(target)
        original_write(target, document_to_write)

    monkeypatch.setattr(manager, "_atomic_write", recording_write)

    first = manager.load_config()
    second = manager.load_config()

    assert first.ok and first.wrote_repairs
    assert second.ok and not second.wrote_repairs
    assert writes == [path]
    assert manager.get_setting_config().Common.Timeout == 15
    assert manager.get_setting_config().Relative.InterestTimeout == 90
    assert _load(path)[TITLE_SETTING]["Relative"]["TFIDFVectorDimension"] == 262_144
    assert {issue.code for issue in first.report.repairs} == {"missing_key"}


def test_non_setting_missing_value_uses_default_but_is_not_written(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    del document[TITLE_BASE]["BotInfo"]["name"]
    _dump(path, document)
    before = path.read_text(encoding="utf-8")
    manager = _manager(path)

    result = manager.load_config()
    write_result = manager.commit_and_write_config()

    assert result.ok and write_result.ok
    assert manager.get_base_config().BotInfo.name == "default-name"
    assert path.read_text(encoding="utf-8") == before
    assert [issue.path_text for issue in result.report.errors] == [
        "BOT基本配置.BotInfo.name"
    ]


def test_explicit_mutation_of_fallback_value_is_persisted_alone(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    del document[TITLE_BASE]["BotInfo"]["name"]
    del document[TITLE_BASE]["BotInfo"]["uid"]
    _dump(path, document)
    manager = _manager(path)
    assert manager.load_config().ok

    manager.get_base_config().BotInfo.name = "runtime-name"
    result = manager.commit_and_write_config()
    saved = _load(path)

    assert result.ok
    assert saved[TITLE_BASE]["BotInfo"]["name"] == "runtime-name"
    assert "uid" not in saved[TITLE_BASE]["BotInfo"]


def test_wrong_type_falls_back_without_overwriting_raw_value(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    document[TITLE_SETTING]["Common"]["Timeout"] = "fifteen"
    document[TITLE_LLM]["LLM_List"][0]["max_context"] = "128k"
    _dump(path, document)
    manager = _manager(path)

    result = manager.load_config()
    assert manager.get_setting_config().Common.Timeout == 15
    assert manager.get_llm_config().LLM_List[0].max_context == 128_000
    assert {issue.path_text for issue in result.report.errors} >= {
        "其他参数配置，请勿随意更改.Common.Timeout",
        "LLM模型配置.LLM_List[0].max_context",
    }
    assert "值 'fifteen'" in result.report.render("list")

    assert manager.commit_and_write_config().ok
    saved = _load(path)
    assert saved[TITLE_SETTING]["Common"]["Timeout"] == "fifteen"
    assert saved[TITLE_LLM]["LLM_List"][0]["max_context"] == "128k"


def test_lists_have_no_comment_sentinel_and_round_trip_comments(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """# user heading
BOT基本配置:
  BotInfo: {name: bot, uid: ''}
  AdminList:  # keep admin comment
    - '10001' # first admin
  NestedList:
    - [inner]
  OptionalList: []
LLM模型配置:
  LLM_List: []
群聊设置:
  Groups:
    Group_Default_Setting: {name: '', chat: true, ban_topic: []}
其他参数配置，请勿随意更改:
  Common: {EnableConfigCheck: true, Timeout: 15}
  Relative: {InterestTimeout: 90, TFIDFVectorDimension: 262144}
""",
        encoding="utf-8",
    )
    manager = _manager(path)

    assert manager.load_config().ok
    assert manager.get_base_config().AdminList == ["10001"]
    assert manager.get_base_config().NestedList == [["inner"]]
    assert manager.get_base_config().OptionalList == []
    assert manager.get_llm_config().LLM_List == []

    assert manager.commit_and_write_config().ok
    text = path.read_text(encoding="utf-8")
    assert "# user heading" in text
    assert "# keep admin comment" in text
    assert "# first admin" in text


def test_llm_items_and_dynamic_groups_use_templates(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    document[TITLE_LLM]["LLM_List"] = [{"preset_name": "custom"}]
    document[TITLE_GROUPS]["Groups"]["20001"] = {"name": "真实群"}
    _dump(path, document)
    manager = _manager(path)

    result = manager.load_config()

    assert manager.get_llm_config().LLM_List[0].model == "default-model"
    assert manager.get_group_config().Groups["20001"].chat is True
    assert {issue.path_text for issue in result.report.errors} >= {
        "LLM模型配置.LLM_List[0].model",
        "群聊设置.Groups.20001.chat",
    }


def test_reload_is_transactional_and_keeps_root_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    document[TITLE_BASE]["BotInfo"]["name"] = "first"
    _dump(path, document)
    manager = _manager(path)
    root = manager.get_base_config()
    assert manager.load_config().ok

    document[TITLE_BASE]["BotInfo"]["name"] = "second"
    _dump(path, document)
    assert manager.load_config().ok
    assert manager.get_base_config() is root
    assert root.BotInfo.name == "second"

    path.write_text("BOT基本配置: [broken", encoding="utf-8")
    failed = manager.load_config()
    assert not failed.ok
    assert root.BotInfo.name == "second"


def test_report_supports_combined_tree_and_path_list(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    del document[TITLE_BASE]["BotInfo"]["name"]
    _dump(path, document)
    manager = _manager(path)
    report = manager.load_config().report

    tree = report.render("tree")
    path_list = report.render("list")
    combined = report.render("combined")

    assert "BOT基本配置" in tree and "└" in tree
    assert "BOT基本配置.BotInfo.name" in path_list
    assert tree in combined and path_list in combined


def test_unknown_keys_survive_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    document[TITLE_BASE]["FutureFeature"] = {"enabled": True}
    _dump(path, document)
    manager = _manager(path)

    result = manager.load_config()
    assert result.ok
    assert manager.get_base_config().FutureFeature.enabled is True
    assert not any("FutureFeature" in issue.path_text for issue in result.report.issues)
    assert manager.commit_and_write_config().ok
    assert _load(path)[TITLE_BASE]["FutureFeature"] == {"enabled": True}


def test_module_defaults_include_complete_relative_group() -> None:
    assert config.DEFAULT_SETTING["Relative"] == {
        "InterestTimeout": 90,
        "CommonTimeWeight": 0.1,
        "CommonTimeNowWeight": 0.1,
        "CommonTimeNerfWeight": 0.2,
        "CommonContextWeight": 0.1,
        "CommonSameUserWeight": 0.02,
        "CommonAtWeight": 0.2,
        "CommonReplyWeight": 0.8,
        "CommonBotNerfWeight": 0.1,
        "CommonRepeatNerfWeight": 0.4,
        "DynamicSimilarityTimeWindow": 600,
        "DynamicSimilarityMessageWindow": 50,
        "DynamicSimilarityInheritFactor": 0.3,
        "FilterGroupDocRate": 0.11,
        "FilterGroupDocTFIDF": 0.05,
        "FilterGroupUserRate": 0.6,
        "FilterGroupUserTFIDF": 0.05,
        "FilterGroupTimeRateLow": 0.15,
        "FilterGroupTimeRateUp": 0.3,
        "FilterGroupTimeTFIDFLow": 0.01,
        "FilterGroupTimeTFIDFUp": 0.05,
        "LouvainMinResolution": 0.2,
        "LouvainMaxResolution": 1.6,
        "LouvainMinModularity": 0.8,
        "LouvainMaxCommunity": 0.15,
        "HotWordVertexWeight": 0.1,
        "MinWordAlphaLimit": 3,
        "MinWordNumLimit": 2,
        "MinTFIDFDocuments": 100,
        "MinGroupCommunityVertex": 100,
        "MaxWordCharLimit": 12,
        "MaxWordNumLimit": 8,
        "MaxTFIDFDocuments": 5000,
        "MaxPreviousMessage": 10,
        "MaxNextMessage": 10,
        "MaxCommunityVertex": 5000,
        "RefreshSimilarityWindow": 50,
        "RefreshSimilarityPeriod": 600,
        "TFIDFVectorDimension": 262_144,
    }


def test_atomic_write_failure_keeps_original_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    _dump(path, document)
    manager = _manager(path)
    assert manager.load_config().ok
    before = path.read_bytes()
    manager.get_base_config().BotInfo.name = "changed"

    def fail_replace(_source: Path, _target: Path) -> None:
        raise PermissionError("simulated replace failure")

    monkeypatch.setattr(config.os, "replace", fail_replace)
    result = manager.commit_and_write_config()

    assert not result.ok
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_unknown_attribute_is_strict_and_does_not_mutate(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    _dump(path, _complete_document())
    manager = _manager(path)
    assert manager.load_config().ok
    base = manager.get_base_config()

    try:
        _ = base.DoesNotExist
    except AttributeError as exc:
        assert "BOT基本配置.DoesNotExist" in str(exc)
    else:  # pragma: no cover - explicit assertion message is clearer here
        raise AssertionError("missing attributes must raise")

    assert "DoesNotExist" not in base
    assert manager.commit_and_write_config().ok
    assert "DoesNotExist" not in _load(path)[TITLE_BASE]


def test_runtime_mapping_and_list_mutations_are_persisted(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    document[TITLE_BASE]["Mutable"] = {
        "mapping": {"a": 1, "drop": 2},
        "items": [3, 1, 2],
    }
    _dump(path, document)
    manager = _manager(path)
    assert manager.load_config().ok
    mutable = manager.get_base_config().Mutable

    mutable.mapping.update({"b": 2})
    mutable.mapping.setdefault("c", 3)
    assert mutable.mapping.pop("drop") == 2
    mutable.mapping["temporary"] = True
    del mutable.mapping["temporary"]

    values = mutable["items"]
    values.append(4)
    values.extend([5, 6])
    values.insert(0, 0)
    assert values.pop() == 6
    values.remove(5)
    values[0] = 7
    del values[1]
    values.reverse()
    values.sort()

    assert manager.commit_and_write_config().ok
    saved = _load(path)[TITLE_BASE]["Mutable"]
    assert saved["mapping"] == {"a": 1, "b": 2, "c": 3}
    assert saved["items"] == sorted(values)

    values.clear()
    mutable.mapping.clear()
    assert manager.commit_and_write_config().ok
    saved = _load(path)[TITLE_BASE]["Mutable"]
    assert saved == {"mapping": {}, "items": []}


def test_load_failures_return_results_and_keep_last_report(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    manager = _manager(path)
    missing = manager.load_config()
    assert not missing.ok and missing.report.errors[0].code == "load_error"

    path.write_text("[]", encoding="utf-8")
    wrong_top = manager.load_config()
    assert not wrong_top.ok
    assert "顶层" in wrong_top.error

    path.write_text("", encoding="utf-8")
    empty = manager.load_config()
    assert not empty.ok
    assert "为空" in empty.error


def test_public_helpers_use_normal_lists_and_write_results(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    document[TITLE_BASE]["Proxies"] = {
        "GlobalProxy": {"http": "http://proxy", "https": None}
    }
    document[TITLE_LLM]["LLM_List"] = [
        {
            "preset_name": "template",
            "model": "default-model",
            "max_context": 128_000,
            "extra_parameter": {"temperature": 0.7},
        },
        {
            "preset_name": "business-last",
            "model": "last-model",
            "max_context": 64_000,
            "extra_parameter": {"temperature": 0.5},
        },
    ]
    _dump(path, document)
    manager = _manager(path)
    assert manager.load_config().ok
    _activate(monkeypatch, manager)

    assert config.get_proxy("GlobalProxy")["http"] == "http://proxy"
    assert config.get_proxy("missing") == {"http": None, "https": None}
    assert config.get_proxy({"http": None}) == {"http": None}
    assert config.get_llm("business-last").model == "last-model"
    assert config.get_llm("missing") == {}
    assert config.get_llm_list() == ["template", "business-last"]

    config.BASE_CFG.BotInfo.name = "helper-write"
    write_result = config.config_update()
    assert write_result.ok
    assert _load(path)[TITLE_BASE]["BotInfo"]["name"] == "helper-write"


def test_reload_rebinds_groups_before_notifying(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    document = _complete_document()
    document[TITLE_GROUPS]["Groups"]["20001"] = {
        "name": "first",
        "chat": True,
        "ban_topic": [],
    }
    _dump(path, document)
    manager = _manager(path)
    assert manager.load_config().ok
    _activate(monkeypatch, manager)
    group = type("Group", (), {})()
    group.group_config = None
    monkeypatch.setattr(config, "QQ_GROUPS", {"20001": group})
    events: list[str] = []

    class Observer:
        def update_config(self) -> None:
            assert group.group_config.name == "second"
            events.append("notified")

    observer = Observer()
    local_observer = config.ConfigObserver()
    local_observer.register(observer)
    monkeypatch.setattr(config, "CONFIG_OBSERVER", local_observer)
    document[TITLE_GROUPS]["Groups"]["20001"]["name"] = "second"
    _dump(path, document)

    result = config.reload_config()

    assert result.ok
    assert events == ["notified"]
    assert group.group_config.name == "second"


def test_default_document_generation_and_character_helpers(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "generated.yaml"
    manager = _manager(path)
    _activate(monkeypatch, manager)
    monkeypatch.setattr(config, "QQ_GROUPS", {})

    write_result = config.write_default_config()
    assert write_result.ok
    generated = _load(path)
    assert generated[TITLE_SETTING]["Relative"]["InterestTimeout"] == 90
    assert generated[TITLE_BASE]["AdminList"] == ["114514", "1919810"]

    config.set_character_dict({"a": 1})
    config.update_character_dict({"b": 2})
    assert config.get_character_dict() == {"a": 1, "b": 2}


def test_generated_config_has_one_heading_per_root_group(tmp_path: Path) -> None:
    path = tmp_path / "generated.yaml"
    manager = config.Configer(path)

    manager._atomic_write(path, manager.build_default_document())

    lines = path.read_text(encoding="utf-8").splitlines()
    for title in (TITLE_BASE, TITLE_LLM, TITLE_PRIVATE, TITLE_GROUPS, TITLE_SETTING):
        headings = [
            index
            for index, line in enumerate(lines)
            if line.startswith("# =") and title in line
        ]
        assert len(headings) == 1, title
        assert lines[headings[0] + 1] == f"{title}:"


def test_save_normalizes_old_root_headings_and_preserves_other_comments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    default_config, default_setting = _defaults()
    default_config[TITLE_PRIVATE] = {
        "Default": {"chat": True, "chat_model": "custom-model"},
        "Users": {},
    }
    document = {
        **copy.deepcopy(default_config),
        TITLE_SETTING: copy.deepcopy(default_setting),
        "FutureRoot": {"enabled": True},
    }
    _dump(path, document)
    text = path.read_text(encoding="utf-8")
    for title in (TITLE_BASE, TITLE_LLM, TITLE_GROUPS, TITLE_SETTING):
        heading = f"# {f' {title} ':=^100}"
        text = text.replace(f"{title}:\n", f"{heading}\n\n{heading}\n{title}:\n", 1)
    text = text.replace(f"{TITLE_PRIVATE}:\n", f"# keep private note\n{TITLE_PRIVATE}:\n", 1)
    path.write_text(text, encoding="utf-8")
    manager = config.Configer(
        path,
        default_config=default_config,
        default_setting=default_setting,
        comments={},
    )

    assert manager.load_config().ok
    assert path.read_text(encoding="utf-8") == text
    manager.get_base_config().BotInfo.name = "first change"
    assert manager.commit_and_write_config().ok
    manager.get_base_config().BotInfo.name = "second change"
    assert manager.commit_and_write_config().ok

    saved = path.read_text(encoding="utf-8")
    lines = saved.splitlines()
    for title in (TITLE_BASE, TITLE_LLM, TITLE_PRIVATE, TITLE_GROUPS, TITLE_SETTING):
        headings = [
            index
            for index, line in enumerate(lines)
            if line.startswith("# =") and title in line
        ]
        assert len(headings) == 1, title
        assert lines[headings[0] + 1] == f"{title}:"
    assert "# keep private note" in saved
    assert _load(path)[TITLE_PRIVATE]["Default"]["chat_model"] == "custom-model"
    assert _load(path)["FutureRoot"]["enabled"] is True


def test_complete_module_defaults_validate_cleanly_and_include_comments(tmp_path: Path) -> None:
    path = tmp_path / "full-default.yaml"
    manager = config.Configer(path)
    document = manager.build_default_document()
    manager._atomic_write(path, document)

    result = manager.load_config()

    assert result.ok
    assert result.report.errors == ()
    assert manager.get_llm_config().LLM_List[-1].model == "localmodel"
    assert manager.get_setting_config().Relative.TFIDFVectorDimension == 262_144
    text = path.read_text(encoding="utf-8")
    assert "# BOT管理员列表" in text
    assert "LLM模型预设模板" in text


def test_list_comments_follow_existing_items_after_insert_remove_and_reorder(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """BOT基本配置:
  BotInfo: {name: bot, uid: ''}
  AdminList: # admin list
    - alice # alice comment
    - bob # bob comment
  NestedList: [[inner]]
  OptionalList: []
LLM模型配置:
  LLM_List: []
群聊设置:
  Groups:
    Group_Default_Setting: {name: '', chat: true, ban_topic: []}
其他参数配置，请勿随意更改:
  Common: {EnableConfigCheck: true, Timeout: 15}
  Relative: {InterestTimeout: 90, TFIDFVectorDimension: 262144}
""",
        encoding="utf-8",
    )
    manager = _manager(path)
    assert manager.load_config().ok

    admins = manager.get_base_config().AdminList
    admins.insert(0, "charlie")
    admins.reverse()
    admins.remove("charlie")
    admins.append("dave")
    assert manager.commit_and_write_config().ok

    text = path.read_text(encoding="utf-8")
    assert "AdminList: # admin list" in text
    assert "- alice # alice comment" in text
    assert "- bob # bob comment" in text
    assert "- dave" in text
    dave_line = next(line for line in text.splitlines() if "- dave" in line)
    assert "alice comment" not in dave_line
    assert "bob comment" not in dave_line

    loaded = _load(path)
    assert loaded[TITLE_BASE]["AdminList"] == ["bob", "alice", "dave"]


def test_private_users_are_sparse_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    default_config, default_setting = _defaults()
    default_config[TITLE_PRIVATE] = {
        "Default": {
            "chat": True,
            "chat_model": "local",
            "agent_model": "local",
            "speak_rate_min": 0.3,
            "speak_rate_max": 1.0,
        },
        "Users": {},
    }
    document = {
        **copy.deepcopy(default_config),
        TITLE_SETTING: copy.deepcopy(default_setting),
    }
    document[TITLE_PRIVATE]["Users"] = {
        "10001": {"chat": False, "speak_rate_min": 0.6},
        "10002": {"speak_rate_min": "invalid"},
    }
    _dump(path, document)
    manager = config.Configer(
        path,
        default_config=default_config,
        default_setting=default_setting,
        comments={},
    )

    result = manager.load_config()
    first = manager.get_private_user_config("10001")
    second = manager.get_private_user_config("10002")

    assert result.ok
    assert first.chat is False
    assert first.chat_model == "local"
    assert first.speak_rate_min == 0.6
    assert second.speak_rate_min == 0.3
    assert manager.get_private_config().Default.chat is True
    assert "10001" not in manager.get_private_config().Default
    assert any(
        issue.path_text == "私聊设置.Users.10002.speak_rate_min"
        for issue in result.report.errors
    )


def test_existing_private_root_survives_setting_repair_and_config_update(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    default_config, default_setting = _defaults()
    default_config[TITLE_PRIVATE] = {
        "Default": {
            "enable": True,
            "chat": True,
            "chat_model": "local",
            "agent_model": "local",
        },
        "Users": {},
    }
    document = {
        **copy.deepcopy(default_config),
        TITLE_SETTING: {"Common": {"EnableConfigCheck": True}},
    }
    document[TITLE_PRIVATE] = {
        "Default": {
            "enable": True,
            "chat": False,
            "chat_model": "custom-speaker",
            "agent_model": "custom-agent",
        },
        "Users": {"10001": {"chat": True, "chat_model": "user-speaker"}},
    }
    _dump(path, document)
    manager = config.Configer(
        path,
        default_config=default_config,
        default_setting=default_setting,
        comments={},
    )

    load_result = manager.load_config()
    manager.get_base_config().BotInfo.name = "changed"
    write_result = manager.commit_and_write_config()
    saved = _load(path)

    assert load_result.ok and load_result.wrote_repairs
    assert write_result.ok
    assert manager.get_private_config().Default.chat_model == "custom-speaker"
    assert manager.get_private_user_config("10001").chat_model == "user-speaker"
    assert saved[TITLE_PRIVATE]["Default"]["chat_model"] == "custom-speaker"
    assert saved[TITLE_PRIVATE]["Users"]["10001"]["chat_model"] == "user-speaker"


def test_private_config_root_identity_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    default_config, default_setting = _defaults()
    default_config[TITLE_PRIVATE] = {
        "Default": {"chat": True, "chat_model": "local"},
        "Users": {},
    }
    document = {
        **copy.deepcopy(default_config),
        TITLE_SETTING: copy.deepcopy(default_setting),
    }
    _dump(path, document)
    manager = config.Configer(
        path,
        default_config=default_config,
        default_setting=default_setting,
        comments={},
    )
    root = manager.get_private_config()
    assert manager.load_config().ok

    document[TITLE_PRIVATE]["Users"] = {"10001": {"chat": False}}
    _dump(path, document)
    assert manager.load_config().ok

    assert manager.get_private_config() is root
    assert manager.get_private_user_config("10001").chat is False
