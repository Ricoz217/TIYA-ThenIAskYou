from __future__ import annotations

import types
import unittest

from TIYA.dialog import group_dialog


class _FakeSetu:
    def __init__(self, result: str):
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def get_setu(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class SetuCommandRegistrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.original_base_cfg = group_dialog.BASE_CFG
        self.had_setu_config = "SETU" in group_dialog.SETTING_CFG
        self.original_setu_config = group_dialog.SETTING_CFG.get("SETU")
        group_dialog.BASE_CFG = types.SimpleNamespace(
            Module=types.SimpleNamespace(setu=True),
        )
        group_dialog.SETTING_CFG["SETU"] = types.SimpleNamespace(
            MaxImageFileSize=12 * 1024 * 1024,
            MaxGifFileSize=6 * 1024 * 1024,
            NoisePixels=5,
        )

    def tearDown(self) -> None:
        group_dialog.BASE_CFG = self.original_base_cfg
        if self.had_setu_config:
            group_dialog.SETTING_CFG["SETU"] = self.original_setu_config
        else:
            group_dialog.SETTING_CFG.pop("SETU", None)

    @staticmethod
    def make_dialog(result: str):
        dialog = object.__new__(group_dialog.GroupMainDialog)
        dialog.host = types.SimpleNamespace(
            group_config=types.SimpleNamespace(setu=True),
        )
        dialog.SETU = _FakeSetu(result)
        return dialog

    async def test_registration_is_reusable_and_dispatches_current_dialog(self) -> None:
        first = self.make_dialog("first")
        second = self.make_dialog("second")

        with self.assertRaises(group_dialog.SetuHelp):
            await first.setu("--help")
        with self.assertRaises(group_dialog.SetuHelp):
            await second.setu("--help")

        invocation = group_dialog._SETU_COMMAND.parse('#setu --query "test tag"')
        first_result = await invocation.spec.handler(first, invocation.args)
        second_result = await invocation.spec.handler(second, invocation.args)

        self.assertEqual(first_result, "first")
        self.assertEqual(second_result, "second")
        self.assertEqual(first.SETU.calls, [{"expect_count": 1, "query_text": "test tag"}])
        self.assertEqual(second.SETU.calls, [{"expect_count": 1, "query_text": "test tag"}])


if __name__ == "__main__":
    unittest.main()
