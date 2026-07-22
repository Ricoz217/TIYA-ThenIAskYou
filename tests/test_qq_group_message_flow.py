from __future__ import annotations

import sys
import types
import unittest

from TIYA.model.message import GroupMsg, TextMsg


def _load_qq_group():
    module_name = "TIYA.dialog.group_dialog"
    saved_dialog_module = sys.modules.get(module_name)

    dialog_module = types.ModuleType(module_name)
    dialog_module.GroupMainDialog = type("GroupMainDialog", (), {})
    dialog_module.GroupCommandDialog = type("GroupCommandDialog", (), {})
    sys.modules[module_name] = dialog_module
    try:
        from TIYA.qq_group import QQGroup
        return QQGroup
    finally:
        if saved_dialog_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = saved_dialog_module


class _Dialog:
    def __init__(self, name: str, calls: list[str], block: bool = False):
        self.name = name
        self.calls = calls
        self.block = block

    async def accept_message(self, message: GroupMsg) -> bool:
        self.calls.append(self.name)
        return self.block


class _Fav:
    async def auto_fav(self, message: GroupMsg) -> None:
        return None


class QQGroupMessageFlowTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.group_type = _load_qq_group()

    async def test_true_stops_message_propagation(self) -> None:
        calls: list[str] = []
        group = object.__new__(self.group_type)
        group.command_dialog = _Dialog("command", calls)
        group.main_dialog = _Dialog("main", calls, block=True)
        group.fav = _Fav()
        trailing = _Dialog("trailing", calls)
        group.dialog_list = [group.command_dialog, trailing]
        message = GroupMsg(
            msg_id="message-1",
            user_id="user-1",
            username="User",
            nickname="Tester",
            group_id="group-1",
            content=[TextMsg(text="hello")],
        )

        await group.message_flow(message)

        self.assertEqual(calls, ["command", "main"])


if __name__ == "__main__":
    unittest.main()
