from pathlib import Path


def test_start_script_forces_and_verifies_local_source_tree() -> None:
    project_root = Path(__file__).resolve().parents[1]
    content = (project_root / "start.bat").read_text(encoding="utf-8")

    assert 'set "PYTHONPATH=%~dp0src"' in content
    assert 'set "TIYA_PROJECT_ROOT=%~dp0"' in content
    assert "pathlib.Path(TIYA.__file__).resolve().parent" in content
    assert "expected != actual" in content
    assert '"%PYTHON%" -P -m TIYA.QQ_bot' in content
