from __future__ import annotations

import unittest

from TIYA.command import (
    CommandArgumentError,
    CommandNotFoundError,
    CommandPermission,
    CommandSet,
    CommandSyntaxError,
)


class CommandSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.commands = CommandSet(prefix="#")

        @self.commands.command(
            "echo",
            aliases=("repeat",),
            summary="Repeat some text.",
            permission=CommandPermission.OPEN,
        )
        @self.commands.option(
            "-n",
            "--times",
            _type=int,
            default=1,
            choices=range(1, 4),
            _help="Repeat count.",
        )
        @self.commands.option(
            "-u",
            "--upper",
            action="store_true",
            _help="Uppercase the text.",
        )
        @self.commands.argument("text", nargs="+", _help="Text to repeat.")
        async def echo(dialog, context, args):
            return dialog, context, args

        self.echo = echo

    def test_parse_positionals_options_and_quoted_text(self) -> None:
        invocation = self.commands.parse('#echo --times 2 --upper "hello world" again')

        self.assertEqual(invocation.spec.name, "echo")
        self.assertIs(invocation.spec.permission, CommandPermission.OPEN)
        self.assertEqual(invocation.args.times, 2)
        self.assertTrue(invocation.args.upper)
        self.assertEqual(invocation.args.text, ["hello world", "again"])

    def test_alias_and_command_name_are_case_insensitive(self) -> None:
        invocation = self.commands.parse("#RePeAt sample")

        self.assertEqual(invocation.spec.name, "echo")
        self.assertEqual(invocation.args.times, 1)
        self.assertFalse(invocation.args.upper)
        self.assertEqual(invocation.args.text, ["sample"])

    def test_long_option_supports_equals_syntax(self) -> None:
        invocation = self.commands.parse("#echo --times=3 sample")

        self.assertEqual(invocation.args.times, 3)

    def test_double_dash_stops_option_parsing(self) -> None:
        invocation = self.commands.parse("#echo -- --upper")

        self.assertFalse(invocation.args.upper)
        self.assertEqual(invocation.args.text, ["--upper"])

    def test_backslashes_are_not_treated_as_shell_escapes(self) -> None:
        invocation = self.commands.parse(r"#echo C:\Users\Rico\file.txt")

        self.assertEqual(invocation.args.text, [r"C:\Users\Rico\file.txt"])

    def test_unknown_command_raises_specific_error(self) -> None:
        with self.assertRaises(CommandNotFoundError):
            self.commands.parse("#missing")

    def test_unclosed_quote_raises_syntax_error(self) -> None:
        with self.assertRaises(CommandSyntaxError):
            self.commands.parse('#echo "unfinished')

    def test_missing_prefix_or_command_name_raises_syntax_error(self) -> None:
        with self.assertRaises(CommandSyntaxError):
            self.commands.parse("echo sample")

        with self.assertRaises(CommandSyntaxError):
            self.commands.parse("#")

    def test_unknown_option_raises_argument_error(self) -> None:
        with self.assertRaisesRegex(CommandArgumentError, "unknown option"):
            self.commands.parse("#echo --missing sample")

    def test_raw_remainder_preserves_nested_command_text(self) -> None:
        commands = CommandSet(prefix="#")

        @commands.command("proxy")
        @commands.argument("payload", raw_remainder=True, default="")
        async def proxy(dialog, context, args):
            return dialog, context, args

        invocation = commands.parse(
            '#proxy --count 3 --query "blue archive" --unknown=value'
        )

        self.assertEqual(
            invocation.args.payload,
            '--count 3 --query "blue archive" --unknown=value',
        )

    def test_raw_remainder_uses_default_when_tail_is_empty(self) -> None:
        commands = CommandSet(prefix="#")

        @commands.command("proxy")
        @commands.argument("payload", raw_remainder=True, default="")
        async def proxy(dialog, context, args):
            return dialog, context, args

        invocation = commands.parse("#proxy")

        self.assertEqual(invocation.args.payload, "")

    def test_raw_remainder_must_be_the_only_argument(self) -> None:
        commands = CommandSet(prefix="#")

        with self.assertRaisesRegex(ValueError, "raw remainder"):
            @commands.command("invalid")
            @commands.argument("prefix")
            @commands.argument("payload", raw_remainder=True)
            async def invalid(dialog, context, args):
                return dialog, context, args

    def test_missing_required_positional_raises_argument_error(self) -> None:
        with self.assertRaisesRegex(CommandArgumentError, "text"):
            self.commands.parse("#echo")

    def test_type_and_choices_are_validated(self) -> None:
        with self.assertRaisesRegex(CommandArgumentError, "times"):
            self.commands.parse("#echo --times many sample")

        with self.assertRaisesRegex(CommandArgumentError, "times"):
            self.commands.parse("#echo --times 9 sample")

    def test_help_is_generated_from_registered_metadata(self) -> None:
        help_text = self.commands.format_help("echo")

        self.assertIn("#echo", help_text)
        self.assertIn("#repeat", help_text)
        self.assertIn("--times", help_text)
        self.assertIn("Repeat some text.", help_text)

    def test_help_list_can_be_filtered_by_permission(self) -> None:
        @self.commands.command(
            "group-only",
            summary="Group admin command.",
            permission=CommandPermission.GROUP_ADMIN,
        )
        async def group_only(dialog, context, args):
            return dialog, context, args

        @self.commands.command(
            "bot-only",
            summary="Bot admin command.",
            permission=CommandPermission.BOT_ADMIN,
        )
        async def bot_only(dialog, context, args):
            return dialog, context, args

        @self.commands.command(
            "owner-only",
            summary="Owner command.",
            permission=CommandPermission.OWNER,
        )
        async def owner_only(dialog, context, args):
            return dialog, context, args

        open_help = self.commands.format_help(permission=CommandPermission.OPEN)
        self.assertIn("#echo", open_help)
        self.assertNotIn("#group-only", open_help)
        self.assertNotIn("#bot-only", open_help)
        self.assertNotIn("#owner-only", open_help)

        bot_help = self.commands.format_help(permission=CommandPermission.BOT_ADMIN)
        self.assertIn("#echo", bot_help)
        self.assertIn("#group-only", bot_help)
        self.assertIn("#bot-only", bot_help)
        self.assertNotIn("#owner-only", bot_help)

    def test_specific_help_requires_sufficient_permission(self) -> None:
        @self.commands.command(
            "owner-only",
            permission=CommandPermission.OWNER,
        )
        async def owner_only(dialog, context, args):
            return dialog, context, args

        with self.assertRaises(CommandNotFoundError):
            self.commands.format_help(
                "owner-only",
                permission=CommandPermission.BOT_ADMIN,
            )

        self.assertIn(
            "#owner-only",
            self.commands.format_help(
                "owner-only",
                permission=CommandPermission.OWNER,
            ),
        )

    def test_command_permission_defaults_to_open(self) -> None:
        commands = CommandSet()

        @commands.command("ping")
        async def ping(dialog, context, args):
            return dialog, context, args

        self.assertIs(commands.get("ping").permission, CommandPermission.OPEN)

    def test_invalid_command_permission_is_rejected(self) -> None:
        commands = CommandSet()

        with self.assertRaises(TypeError):
            commands.command("invalid", permission="admin")

    def test_command_permissions_have_a_strict_level_order(self) -> None:
        self.assertLess(CommandPermission.OPEN, CommandPermission.GROUP_ADMIN)
        self.assertLess(
            CommandPermission.GROUP_ADMIN,
            CommandPermission.BOT_ADMIN,
        )
        self.assertLess(CommandPermission.BOT_ADMIN, CommandPermission.OWNER)


if __name__ == "__main__":
    unittest.main()
