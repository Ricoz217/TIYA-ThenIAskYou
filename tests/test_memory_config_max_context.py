from types import SimpleNamespace

from TIYA.memory import engine as memory_engine


def test_empty_llm_preset_uses_default_max_context_window(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_get_llm(_preset_name: str):
        raise AssertionError("empty llm_preset should not resolve LLM config")

    monkeypatch.setattr(memory_engine, "get_llm", fail_get_llm)

    engine = memory_engine.ContextMemoryEngineV3(
        config=memory_engine.ContextMemoryConfig(
            base_dir=tmp_path,
            llm_preset="",
            image_llm_preset="",
            use_mock_llm=True,
            auto_resume_pending_jobs=False,
        )
    )

    assert engine.llm_preset == ""
    assert engine.max_context_window == 1_000_000


def test_apply_config_reloads_max_context_window_from_llm_preset(
    tmp_path,
    monkeypatch,
) -> None:
    preset_windows = {
        "MEMORY_A": 64_000,
        "MEMORY_B": 256_000,
    }

    def fake_get_llm(preset_name: str):
        return SimpleNamespace(max_context=preset_windows[preset_name])

    monkeypatch.setattr(memory_engine, "get_llm", fake_get_llm)

    engine = memory_engine.ContextMemoryEngineV3(
        config=memory_engine.ContextMemoryConfig(
            base_dir=tmp_path,
            llm_preset="MEMORY_A",
            image_llm_preset="",
            use_mock_llm=True,
            auto_resume_pending_jobs=False,
        )
    )
    assert engine.max_context_window == 64_000

    engine.apply_config(
        memory_engine.ContextMemoryConfig(
            base_dir=tmp_path,
            llm_preset="MEMORY_B",
            image_llm_preset="",
            use_mock_llm=True,
            auto_resume_pending_jobs=False,
        )
    )

    assert engine.llm_preset == "MEMORY_B"
    assert engine.pipeline.default_llm_preset == "MEMORY_B"
    assert engine.max_context_window == 256_000
