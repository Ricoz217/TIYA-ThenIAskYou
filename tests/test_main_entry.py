import runpy
import sys
from types import ModuleType
from unittest.mock import Mock


def test_python_m_tiya_runs_qq_bot_main(monkeypatch):
    qq_bot = ModuleType("TIYA.QQ_bot")
    main = Mock()
    qq_bot.main = main
    monkeypatch.setitem(sys.modules, "TIYA.QQ_bot", qq_bot)

    runpy.run_module("TIYA", run_name="__main__")

    main.assert_called_once_with()
