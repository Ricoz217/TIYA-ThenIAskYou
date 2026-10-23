"""Small JSON-lines bridge for manually driving the browse contract.

Run this script with the project's Python interpreter. It keeps one browser
session alive and emits exactly one JSON response for every JSON command read
from standard input.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


SOURCE_ROOT = Path(__file__).resolve().parents[4]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from TIYA.agent_tools.websearch.tools import (  # noqa: E402
    BrowseRequest,
    BrowserPage,
    BrowserSession,
    BrowserTarget,
    Capability,
    IdentityRef,
    InteractionRequired,
    MouseButton,
    WebSearchRuntime,
    WebSearchSettings,
    build_websearch_runtime,
)


_SAFE_PNG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,199}\.png")
_OPERATIONS = (
    "profiles",
    "open",
    "snapshot",
    "navigate",
    "click",
    "fill",
    "select",
    "upload",
    "press",
    "scroll",
    "wait",
    "back",
    "forward",
    "reload",
    "screenshot",
    "close",
    "quit",
)


class BrowseRuntime(Protocol):
    async def browse(
        self, request: BrowseRequest, *, provider: str | None = None
    ) -> BrowserSession: ...

    async def aclose(self) -> None: ...


class BrowseBridge:
    """Convert JSON-compatible commands to one stateful BrowserSession."""

    def __init__(
        self,
        runtime: BrowseRuntime,
        *,
        screenshot_dir: str | Path,
        content_limit: int = 50_000,
        item_limit: int = 250,
    ) -> None:
        if content_limit <= 0 or item_limit <= 0:
            raise ValueError("content_limit and item_limit must be positive")
        self._runtime = runtime
        self._screenshot_dir = Path(screenshot_dir).resolve()
        self._content_limit = content_limit
        self._item_limit = item_limit
        self._session: BrowserSession | None = None
        self._closed = False
        self._screenshot_number = 0

    async def handle(self, command: Mapping[str, object]) -> dict[str, object]:
        """Execute one decoded command and return a JSON-compatible result."""

        if self._closed:
            raise RuntimeError("browse bridge is closed")
        operation = _required_text(command, "op").casefold()

        if operation == "profiles":
            return self._profiles()
        if operation == "open":
            return await self._open(command)
        if operation == "close":
            return await self._close_session()
        if operation == "quit":
            await self.aclose()
            return {"closed": True}
        if operation not in _OPERATIONS:
            supported = ", ".join(_OPERATIONS)
            raise ValueError(
                f"unsupported operation {operation!r}; expected one of {supported}"
            )

        session = self._require_session()
        if operation == "snapshot":
            page = await session.snapshot()
        elif operation == "navigate":
            page = await session.navigate(_required_text(command, "url"))
        elif operation == "click":
            x = _optional_number(command, "x")
            y = _optional_number(command, "y")
            if (x is None) != (y is None):
                raise ValueError("x and y must be provided together")
            position = None
            if x is not None:
                assert y is not None
                position = (x, y)
            raw_target = command.get("target")
            page = await session.click(
                None if raw_target is None else _target(command),
                button=MouseButton(_optional_text(command, "button") or "left"),
                click_count=_optional_integer(command, "click_count", 1),
                position=position,
            )
        elif operation == "fill":
            page = await session.fill(
                _target(command),
                _required_text(command, "value"),
                clear=_optional_boolean(command, "clear", True),
                submit=_optional_boolean(command, "submit", False),
            )
        elif operation == "select":
            page = await session.select(
                _target(command),
                _required_texts(command, "values"),
            )
        elif operation == "upload":
            page = await session.upload(
                _target(command),
                _required_texts(command, "files"),
            )
        elif operation == "press":
            raw_target = command.get("target")
            page = await session.press(
                _required_text(command, "key"),
                target=None if raw_target is None else _target(command),
            )
        elif operation == "scroll":
            raw_target = command.get("target")
            page = await session.scroll(
                delta_x=_optional_integer(command, "delta_x", 0),
                delta_y=_optional_integer(command, "delta_y", 0),
                target=None if raw_target is None else _target(command),
            )
        elif operation == "wait":
            raw_target = command.get("target")
            timeout = _optional_number(command, "timeout")
            page = await session.wait(
                duration=_optional_number(command, "duration"),
                target=None if raw_target is None else _target(command),
                text=_optional_text(command, "text"),
                timeout=10.0 if timeout is None else timeout,
            )
        elif operation == "back":
            page = await session.back()
        elif operation == "forward":
            page = await session.forward()
        elif operation == "reload":
            page = await session.reload()
        elif operation == "screenshot":
            return await self._screenshot(command)
        else:
            raise AssertionError(f"unhandled operation: {operation}")
        return {"page": self._page(page)}

    async def aclose(self) -> None:
        """Close the active session and all providers exactly once."""

        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            await self._session.aclose()
            self._session = None
        await self._runtime.aclose()

    async def _open(self, command: Mapping[str, object]) -> dict[str, object]:
        identity_name = _optional_text(command, "identity")
        visible = command.get("visible")
        if visible is not None and not isinstance(visible, bool):
            raise ValueError("visible must be a boolean or null")
        provider = _optional_text(command, "provider")
        request = BrowseRequest(
            _required_text(command, "url"),
            identity=(IdentityRef(identity_name) if identity_name else None),
            visible=visible,
        )
        if self._session is not None:
            await self._session.aclose()
            self._session = None
        session = await self._runtime.browse(request, provider=provider)
        self._session = session
        page = await session.snapshot()
        return {
            "session_id": session.session_id,
            "provider": session.provider_id,
            "page": self._page(page),
        }

    async def _close_session(self) -> dict[str, object]:
        if self._session is None:
            return {"closed": False}
        await self._session.aclose()
        self._session = None
        return {"closed": True}

    async def _screenshot(
        self, command: Mapping[str, object]
    ) -> dict[str, object]:
        session = self._require_session()
        screenshot = await session.screenshot(
            full_page=_optional_boolean(command, "full_page", False)
        )
        filename = _optional_text(command, "filename")
        if filename is None:
            self._screenshot_number += 1
            filename = (
                f"browse-{session.session_id[:12]}-"
                f"{self._screenshot_number}.png"
            )
        _validate_screenshot_filename(filename)
        path = self._screenshot_dir / filename
        await asyncio.to_thread(self._write_screenshot, path, screenshot.data)
        return {
            "screenshot": {
                "path": str(path),
                "mime_type": screenshot.mime_type,
                "width": screenshot.width,
                "height": screenshot.height,
                "full_page": screenshot.full_page,
                "size": len(screenshot.data),
            }
        }

    def _write_screenshot(self, path: Path, data: bytes) -> None:
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(data)

    def _require_session(self) -> BrowserSession:
        if self._session is None:
            raise RuntimeError("no browser session is open")
        return self._session

    def _profiles(self) -> dict[str, object]:
        if not isinstance(self._runtime, WebSearchRuntime):
            return {"providers": []}
        profiles = []
        for profile in self._runtime.provider_profiles():
            if Capability.BROWSE not in profile.capabilities:
                continue
            profiles.append(
                {
                    "provider_id": profile.provider_id,
                    "display_name": profile.display_name,
                    "description": profile.description,
                    "capabilities": sorted(
                        capability.value for capability in profile.capabilities
                    ),
                    "strengths": list(profile.strengths),
                    "limitations": list(profile.limitations),
                }
            )
        return {"providers": profiles}

    def _page(self, page: BrowserPage) -> dict[str, object]:
        document = page.document
        content = document.content[: self._content_limit]
        content_truncated = len(content) < len(document.content)
        links = document.links[: self._item_limit]
        media = document.media[: self._item_limit]
        elements = page.elements[: self._item_limit]
        return {
            "session_id": page.session_id,
            "revision": page.revision,
            "url": page.url,
            "title": page.title,
            "content": content,
            "content_truncated": content_truncated,
            "content_format": document.content_format.value,
            "status_code": document.status_code,
            "content_type": document.content_type,
            "elements": [
                {
                    "ref": element.ref,
                    "role": element.role,
                    "name": element.name,
                    "tag": element.tag,
                    "text": element.text,
                    "value": element.value,
                    "href": element.href,
                    "clickable": element.clickable,
                    "editable": element.editable,
                    "enabled": element.enabled,
                    "visible": element.visible,
                    "attributes": _json_value(element.attributes),
                    "metadata": _json_value(element.metadata),
                }
                for element in elements
            ],
            "elements_truncated": len(elements) < len(page.elements),
            "links": [
                {
                    "url": link.url,
                    "text": link.text,
                    "title": link.title,
                    "internal": link.internal,
                    "metadata": _json_value(link.metadata),
                }
                for link in links
            ],
            "links_truncated": len(links) < len(document.links),
            "media": [
                {
                    "url": resource.url,
                    "kind": resource.kind.value,
                    "alt": resource.alt,
                    "mime_type": resource.mime_type,
                    "metadata": _json_value(resource.metadata),
                }
                for resource in media
            ],
            "media_truncated": len(media) < len(document.media),
            "interaction_required": _interaction(document.interaction_required),
            "metadata": _json_value(page.metadata),
        }


async def handle_line(bridge: BrowseBridge, line: str) -> dict[str, object]:
    """Decode, execute, and wrap one protocol line without leaking tracebacks."""

    request_id: object = None
    try:
        decoded = json.loads(line)
        if not isinstance(decoded, Mapping):
            raise ValueError("each command must be a JSON object")
        request_id = decoded.get("id")
        result = await bridge.handle(decoded)
        return {"id": request_id, "ok": True, "result": result}
    except Exception as exc:
        return {
            "id": request_id,
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


async def serve(
    *, env_file: str | Path, screenshot_dir: str | Path | None = None
) -> None:
    """Run the JSONL loop until EOF or a successful quit command."""

    settings = WebSearchSettings.from_env(env_file)
    runtime = build_websearch_runtime(settings)
    output_dir = (
        Path(screenshot_dir).resolve()
        if screenshot_dir is not None
        else settings.download_dir / "screenshots"
    )
    bridge = BrowseBridge(runtime, screenshot_dir=output_dir)
    _emit(
        {
            "ok": True,
            "event": "ready",
            "browse_providers": list(runtime.provider_ids(Capability.BROWSE)),
            "operations": list(_OPERATIONS),
        }
    )
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            response = await handle_line(bridge, line)
            _emit(response)
            result = response.get("result")
            if (
                response.get("ok") is True
                and isinstance(result, Mapping)
                and result.get("closed") is True
                and _line_operation(line) == "quit"
            ):
                break
    finally:
        await bridge.aclose()


def _target(command: Mapping[str, object]) -> BrowserTarget:
    raw = command.get("target")
    if not isinstance(raw, Mapping):
        raise ValueError("target must be an object containing ref or selector")
    ref = _mapping_optional_text(raw, "ref")
    selector = _mapping_optional_text(raw, "selector")
    return BrowserTarget(ref=ref, selector=selector)


def _interaction(value: InteractionRequired | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "kind": value.kind.value,
        "url": value.url,
        "provider": value.provider,
        "message": value.message,
        "identity": value.identity.profile_id if value.identity else None,
        "resumable": value.resumable,
    }


def _required_text(command: Mapping[str, object], key: str) -> str:
    value = command.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_text(command: Mapping[str, object], key: str) -> str | None:
    value = command.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string or null")
    return value


def _mapping_optional_text(value: Mapping[object, object], key: str) -> str | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"target.{key} must be a non-empty string or null")
    return raw


def _optional_boolean(
    command: Mapping[str, object], key: str, default: bool
) -> bool:
    value = command.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_integer(
    command: Mapping[str, object], key: str, default: int
) -> int:
    value = command.get(key, default)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_number(
    command: Mapping[str, object], key: str
) -> float | None:
    value = command.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null")
    return float(value)


def _required_texts(
    command: Mapping[str, object], key: str
) -> tuple[str, ...]:
    value = command.get(key)
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or not value:
        raise ValueError(f"{key} must be a string or non-empty string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must contain only non-empty strings")
    return tuple(value)


def _validate_screenshot_filename(filename: str) -> None:
    if Path(filename).name != filename or not _SAFE_PNG_NAME.fullmatch(filename):
        raise ValueError("filename must be a safe .png file name")


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (Path, datetime)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    return repr(value)


def _line_operation(line: str) -> str | None:
    try:
        decoded = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    operation = decoded.get("op")
    return operation.casefold() if isinstance(operation, str) else None


def _emit(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--screenshot-dir")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    asyncio.run(
        serve(
            env_file=arguments.env_file,
            screenshot_dir=arguments.screenshot_dir,
        )
    )


if __name__ == "__main__":
    main()
