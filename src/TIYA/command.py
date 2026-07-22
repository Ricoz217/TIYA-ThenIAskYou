from __future__ import annotations

import shlex
from collections.abc import Callable, Collection
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

__all__ = [
    "CommandArgs",
    "CommandArgumentError",
    "CommandError",
    "CommandInvocation",
    "CommandNotFoundError",
    "CommandPermission",
    "CommandSet",
    "CommandSpec",
    "CommandSyntaxError",
]


_MISSING = object()
_ARGUMENT_METADATA = "__tiya_command_arguments__"
_OPTION_METADATA = "__tiya_command_options__"


class CommandError(Exception):
    """Base exception for user-facing command errors."""


class CommandSyntaxError(CommandError):
    """Raised when command text cannot be tokenized."""


class CommandNotFoundError(CommandError):
    """Raised when no registered command matches the requested name."""


class CommandArgumentError(CommandError):
    """Raised when command arguments do not match the command definition."""


class CommandPermission(IntEnum):
    """Minimum permission level required before a handler may execute."""
    OPEN = 0
    GROUP_ADMIN = 1
    BOT_ADMIN = 2
    OWNER = 3


class CommandArgs(dict[str, Any]):
    """Parsed command arguments with both mapping and attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True, slots=True)
class CommandArgument:
    """给handle新增一个外部参数"""
    name: str
    nargs: int | Literal["?", "*", "+"] = 1
    type: Callable[[str], Any] = str
    default: Any = _MISSING
    choices: Collection[Any] | None = None
    help: str = ""
    raw_remainder: bool = False


@dataclass(frozen=True, slots=True)
class CommandOption:
    """指定参数，例如: --xxx"""
    flags: tuple[str, ...]
    dest: str
    action: Literal["store", "store_true", "store_false"] = "store"
    type: Callable[[str], Any] = str
    default: Any = _MISSING
    choices: Collection[Any] | None = None
    required: bool = False
    help: str = ""


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """命令定义"""
    name: str
    aliases: tuple[str, ...]
    handler: Callable[..., Any]
    summary: str
    permission: CommandPermission
    arguments: tuple[CommandArgument, ...]
    options: tuple[CommandOption, ...]


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    """调用器"""
    spec: CommandSpec
    args: CommandArgs
    raw_text: str
    invoked_name: str


class CommandSet:
    """Decorator-driven command registry and message command parser."""

    def __init__(self, prefix: str = "#"):
        if not prefix:
            raise ValueError("command prefix cannot be empty")

        self.prefix = prefix
        self._commands: dict[str, CommandSpec] = {}
        self._lookup: dict[str, CommandSpec] = {}

        # Legacy modules still register these handlers while being imported.
        self.handler_open: dict[str, dict[str, Any]] = {}
        self.handler_admin: dict[str, dict[str, Any]] = {}

    @staticmethod
    def argument(
            name: str,
            *,
            nargs: int | Literal["?", "*", "+"] = 1,
            _type: Callable[[str], Any] = str,
            default: Any = _MISSING,
            choices: Collection[Any] | None = None,
            raw_remainder: bool = False,
            _help: str = ""
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Attach a positional argument definition to a command handler."""
        if not name or name.startswith("-"):
            raise ValueError("argument name must be a non-empty plain name")
        if not (nargs in (1, "?", "*", "+") or isinstance(nargs, int) and nargs > 0):
            raise ValueError(f"unsupported nargs: {nargs!r}")

        definition = CommandArgument(
            name=name,
            nargs=nargs,
            type=_type,
            default=default,
            choices=choices,
            help=_help,
            raw_remainder=raw_remainder,
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            metadata = list(getattr(func, _ARGUMENT_METADATA, ()))
            metadata.insert(0, definition)
            setattr(func, _ARGUMENT_METADATA, tuple(metadata))
            return func

        return decorator

    @staticmethod
    def option(
            *flags: str,
            dest: str | None = None,
            action: Literal["store", "store_true", "store_false"] = "store",
            _type: Callable[[str], Any] = str,
            default: Any = _MISSING,
            choices: Collection[Any] | None = None,
            required: bool = False,
            _help: str = ""
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Attach an option definition to a command handler."""
        if not flags or any(not flag.startswith("-") or flag == "-" for flag in flags):
            raise ValueError("options require one or more '-' prefixed flags")
        if action not in ("store", "store_true", "store_false"):
            raise ValueError(f"unsupported option action: {action!r}")

        if dest is None:
            preferred = next((flag for flag in flags if flag.startswith("--")), flags[0])
            dest = preferred.lstrip("-").replace("-", "_")
        if not dest:
            raise ValueError("option destination cannot be empty")

        definition = CommandOption(
            flags=tuple(flags),
            dest=dest,
            action=action,
            type=_type,
            default=default,
            choices=choices,
            required=required,
            help=_help
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            metadata = list(getattr(func, _OPTION_METADATA, ()))
            metadata.insert(0, definition)
            setattr(func, _OPTION_METADATA, tuple(metadata))
            return func

        return decorator

    def command(
            self,
            name: str,
            *,
            aliases: tuple[str, ...] = (),
            summary: str = "",
            permission: CommandPermission = CommandPermission.OPEN
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a decorated handler as a command."""
        if not isinstance(permission, CommandPermission):
            raise TypeError("permission must be a CommandPermission")

        normalized_name = self._normalize_name(name)
        normalized_aliases = tuple(self._normalize_name(alias) for alias in aliases)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            arguments = tuple(getattr(func, _ARGUMENT_METADATA, ()))
            options = tuple(getattr(func, _OPTION_METADATA, ()))
            self._validate_definition(arguments, options)
            spec = CommandSpec(
                name=normalized_name,
                aliases=normalized_aliases,
                handler=func,
                summary=summary or (func.__doc__ or "").strip(),
                permission=permission,
                arguments=arguments,
                options=options
            )
            self._register(spec)
            return func

        return decorator

    def parse(self, text: str) -> CommandInvocation:
        """Parse command text without executing its registered handler."""
        raw_parts = text.lstrip().split(maxsplit=1)
        if raw_parts and raw_parts[0].startswith(self.prefix):
            try:
                raw_invoked_name = self._normalize_name(raw_parts[0])
            except ValueError:
                raw_invoked_name = ""

            raw_spec = self._lookup.get(raw_invoked_name.casefold())
            if (
                    raw_spec is not None
                    and raw_spec.arguments
                    and raw_spec.arguments[0].raw_remainder
            ):
                argument = raw_spec.arguments[0]
                values = CommandArgs()
                if len(raw_parts) > 1:
                    values[argument.name] = self._convert_value(
                        argument.name,
                        raw_parts[1],
                        argument.type,
                        argument.choices,
                    )
                elif argument.default is not _MISSING:
                    values[argument.name] = argument.default
                else:
                    values[argument.name] = None

                return CommandInvocation(
                    spec=raw_spec,
                    args=values,
                    raw_text=text,
                    invoked_name=raw_invoked_name,
                )

        try:
            tokens = self._tokenize(text)
        except ValueError as exc:
            raise CommandSyntaxError(str(exc)) from exc

        if not tokens:
            raise CommandSyntaxError("空命令")

        if not tokens[0].startswith(self.prefix):
            raise CommandSyntaxError(f"命令开头必须是 {self.prefix}")

        try:
            invoked_name = self._normalize_name(tokens[0])
        except ValueError as exc:
            raise CommandSyntaxError(str(exc)) from exc

        spec = self._lookup.get(invoked_name.casefold())
        if spec is None:
            raise CommandNotFoundError(f"未知命令: {self.prefix}{invoked_name}")

        return CommandInvocation(
            spec=spec,
            args=self._parse_arguments(spec, tokens[1:]),
            raw_text=text,
            invoked_name=invoked_name
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        lexer = shlex.shlex(text, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.escape = ""
        return list(lexer)

    def get(self, name: str) -> CommandSpec | None:
        """Return a command by canonical name or alias."""
        return self._lookup.get(self._normalize_name(name).casefold())

    def format_help(
            self,
            name: str | None = None,
            *,
            permission: CommandPermission | None = None
    ) -> str:
        """Generate compact help text from registered command metadata."""
        if name is None:
            lines = ["可用命令: "]
            index = 1
            for spec in self._commands.values():
                if not self._is_allowed_by_permission(spec, permission):
                    continue

                line = f"\t{index}. {self.prefix}{spec.name}"
                if spec.summary:
                    line += f": {spec.summary}"

                lines.append(line)
                index += 1
            return "\n".join(lines)

        resolved_spec = self.get(name)
        if resolved_spec is None:
            raise CommandNotFoundError(f"未知命令: {self.prefix}{name}")

        if not self._is_allowed_by_permission(resolved_spec, permission):
            raise CommandNotFoundError(f"未知命令: {self.prefix}{name}")

        usage = [f"用法: {self.prefix}{resolved_spec.name}"]
        for option in resolved_spec.options:
            option_name = option.flags[-1]
            if option.action == "store":
                option_name += f" {option.dest.upper()}"

            usage.append(option_name if option.required else f"[{option_name}]")

        usage.extend(
            self._format_argument_usage(argument)
            for argument in resolved_spec.arguments
        )

        lines = [" ".join(usage)]
        if resolved_spec.aliases:
            aliases = ", ".join(
                f"{self.prefix}{alias}"
                for alias in resolved_spec.aliases
            )
            lines.append(f"别名: {aliases}")

        if resolved_spec.summary:
            lines.extend(("", resolved_spec.summary))

        details = [
            f"  {argument.name}: {argument.help}".rstrip()
            for argument in resolved_spec.arguments
        ]
        details.extend(
            f"  {', '.join(option.flags)}: {option.help}".rstrip()
            for option in resolved_spec.options
        )
        if details:
            lines.extend(("", "可用参数:", *details))

        return "\n".join(lines)

    @staticmethod
    def _is_allowed_by_permission(
            spec: CommandSpec,
            permission: CommandPermission | None
    ) -> bool:
        if permission is None:
            return True
        if not isinstance(permission, CommandPermission):
            raise TypeError("permission must be a CommandPermission")
        return permission >= spec.permission

    def add_handler_open(self, key: str):
        """Legacy decorator retained for import compatibility."""
        return self._add_legacy_handler(self.handler_open, key)

    def add_handler_admin(self, key: str):
        """Legacy decorator retained for import compatibility."""
        return self._add_legacy_handler(self.handler_admin, key)

    def get_handle(self, key: str) -> Callable[..., Any] | None:
        handle = self.handler_open.get(key)
        return handle["func"] if handle else None

    def get_handle_admin(self, key: str) -> Callable[..., Any] | None:
        handle = self.handler_admin.get(key)
        return handle["func"] if handle else None

    def get_help(self, key: str) -> str:
        if key in self.handler_admin:
            return "#ADMIN"
        handle = self.handler_open.get(key)
        return handle["doc"] if handle else ""

    def get_help_admin(self, key: str) -> str:
        handle = self.handler_admin.get(key)
        return handle["doc"] if handle else ""

    def _register(self, spec: CommandSpec) -> None:
        for name in (spec.name, *spec.aliases):
            if name.casefold() in self._lookup:
                raise ValueError(f"command name or alias already registered: {name}")

        self._commands[spec.name] = spec
        for name in (spec.name, *spec.aliases):
            self._lookup[name.casefold()] = spec

    def _parse_arguments(self, spec: CommandSpec, tokens: list[str]) -> CommandArgs:
        option_lookup = {
            flag: option
            for option in spec.options
            for flag in option.flags
        }
        values = CommandArgs()
        supplied_options: set[str] = set()

        for option in spec.options:
            if option.default is not _MISSING:
                values[option.dest] = option.default
            elif option.action == "store_true":
                values[option.dest] = False
            elif option.action == "store_false":
                values[option.dest] = True
            else:
                values[option.dest] = None

        positionals: list[str] = []
        option_parsing = True
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if option_parsing and token == "--":
                option_parsing = False
                index += 1
                continue

            option_token = token
            inline_value: str | None = None
            if option_parsing and token.startswith("--") and "=" in token:
                option_token, inline_value = token.split("=", 1)

            if option_parsing and token.startswith("-"):
                parsed_option = option_lookup.get(option_token)
                if parsed_option is None:
                    raise CommandArgumentError(f"unknown option: {option_token}")

                supplied_options.add(parsed_option.dest)
                if parsed_option.action == "store_true":
                    if inline_value is not None:
                        raise CommandArgumentError(f"option {option_token} does not take a value")
                    values[parsed_option.dest] = True
                elif parsed_option.action == "store_false":
                    if inline_value is not None:
                        raise CommandArgumentError(f"option {option_token} does not take a value")
                    values[parsed_option.dest] = False
                else:
                    if inline_value is None:
                        index += 1
                        if index >= len(tokens):
                            raise CommandArgumentError(f"option {option_token} requires a value")
                        inline_value = tokens[index]
                    values[parsed_option.dest] = self._convert_value(
                        parsed_option.dest,
                        inline_value,
                        parsed_option.type,
                        parsed_option.choices
                    )
            else:
                positionals.append(token)
            index += 1

        for option in spec.options:
            if option.required and option.dest not in supplied_options:
                raise CommandArgumentError(f"required option missing: {option.flags[-1]}")

        self._assign_positionals(spec, positionals, values)
        return values

    def _assign_positionals(
            self,
            spec: CommandSpec,
            tokens: list[str],
            values: CommandArgs
    ) -> None:
        index = 0
        for argument in spec.arguments:
            remaining = len(tokens) - index
            if isinstance(argument.nargs, int):
                if remaining < argument.nargs:
                    raise CommandArgumentError(f"missing argument: {argument.name}")
                raw_values = tokens[index:index + argument.nargs]
                index += argument.nargs
                converted = [
                    self._convert_value(argument.name, value, argument.type, argument.choices)
                    for value in raw_values
                ]
                values[argument.name] = converted[0] if argument.nargs == 1 else converted
                continue

            if argument.nargs == "?":
                if remaining:
                    values[argument.name] = self._convert_value(
                        argument.name,
                        tokens[index],
                        argument.type,
                        argument.choices
                    )
                    index += 1
                elif argument.default is not _MISSING:
                    values[argument.name] = argument.default
                else:
                    values[argument.name] = None
                continue

            if argument.nargs == "+" and remaining == 0:
                raise CommandArgumentError(f"missing argument: {argument.name}")

            raw_values = tokens[index:]
            index = len(tokens)
            values[argument.name] = [
                self._convert_value(argument.name, value, argument.type, argument.choices)
                for value in raw_values
            ]

        if index < len(tokens):
            extra = " ".join(tokens[index:])
            raise CommandArgumentError(f"unexpected arguments: {extra}")

    @staticmethod
    def _convert_value(
            name: str,
            raw_value: str,
            converter: Callable[[str], Any],
            choices: Collection[Any] | None
    ) -> Any:
        try:
            value = converter(raw_value)
        except (TypeError, ValueError) as exc:
            raise CommandArgumentError(f"invalid value for {name}: {raw_value}") from exc

        if choices is not None and value not in choices:
            allowed = ", ".join(str(choice) for choice in choices)
            raise CommandArgumentError(
                f"invalid value for {name}: {raw_value}; choose from {allowed}"
            )
        return value

    @staticmethod
    def _validate_definition(
            arguments: tuple[CommandArgument, ...],
            options: tuple[CommandOption, ...]
    ) -> None:
        raw_arguments = [argument for argument in arguments if argument.raw_remainder]
        if raw_arguments and (len(arguments) != 1 or options):
            raise ValueError(
                "a raw remainder must be the only argument and cannot use options"
            )
        if raw_arguments and raw_arguments[0].nargs != 1:
            raise ValueError("a raw remainder does not support nargs")

        variable_arguments = [
            index
            for index, argument in enumerate(arguments)
            if argument.nargs in ("*", "+")
        ]
        if variable_arguments and variable_arguments[-1] != len(arguments) - 1:
            raise ValueError("a '*' or '+' positional argument must be last")
        if len(variable_arguments) > 1:
            raise ValueError("only one '*' or '+' positional argument is supported")

        flags: set[str] = set()
        destinations: set[str] = set()
        for option in options:
            if option.dest in destinations:
                raise ValueError(f"duplicate option destination: {option.dest}")
            destinations.add(option.dest)
            for flag in option.flags:
                if flag in flags:
                    raise ValueError(f"duplicate option flag: {flag}")
                flags.add(flag)

    def _normalize_name(self, name: str) -> str:
        normalized = name.strip()
        if normalized.startswith(self.prefix):
            normalized = normalized[len(self.prefix):]
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError(f"invalid command name: {name!r}")
        return normalized.casefold()

    def _add_legacy_handler(self, target: dict[str, dict[str, Any]], key: str):
        if key in self.handler_open or key in self.handler_admin:
            raise ValueError(f"command already registered: {key}")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            target[key] = {"func": func, "doc": func.__doc__ or ""}
            return func

        return decorator

    @staticmethod
    def _format_argument_usage(argument: CommandArgument) -> str:
        if argument.raw_remainder:
            if argument.default is not _MISSING:
                return f"[{argument.name} ...]"
            return f"{argument.name} ..."
        if argument.nargs == "?":
            return f"[{argument.name}]"
        if argument.nargs == "*":
            return f"[{argument.name} ...]"
        if argument.nargs == "+":
            return f"{argument.name} [{argument.name} ...]"
        if isinstance(argument.nargs, int) and argument.nargs > 1:
            return " ".join(argument.name for _ in range(argument.nargs))
        return argument.name
