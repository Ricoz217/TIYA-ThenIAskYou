from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from TIYA.agent_tools.websearch.tools import (
    BrowseRequest,
    BrowserElement,
    BrowserPage,
    BrowserScreenshot,
    BrowserSession,
    Capability,
    ClickAction,
    ContentFormat,
    FillAction,
    HistoryAction,
    HistoryDirection,
    IdentityRef,
    NavigateAction,
    PressAction,
    ProviderProfile,
    ReloadAction,
    ScrollAction,
    SelectAction,
    UploadAction,
    WaitAction,
    WebDocument,
    WebSearchRuntime,
)


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "TIYA"
    / "agent_tools"
    / "websearch"
    / "scripts"
    / "browse_bridge.py"
)


def load_bridge_module():
    spec = importlib.util.spec_from_file_location("browse_bridge", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(coroutine):
    return asyncio.run(coroutine)


def make_page(revision: int = 1) -> BrowserPage:
    return BrowserPage(
        session_id="session-1",
        revision=revision,
        document=WebDocument(
            url="https://example.com",
            final_url="https://example.com/final",
            provider="playwright",
            title="Example",
            content="Page body",
            content_format=ContentFormat.TEXT,
        ),
        elements=(
            BrowserElement(
                ref=f"r{revision}-e1",
                role="button",
                name="Continue",
                tag="button",
                text="Continue",
                clickable=True,
            ),
        ),
        metadata={"viewport": {"width": 1280, "height": 720}},
    )


class RecordingSession(BrowserSession):
    provider_id = "playwright"
    session_id = "session-1"

    def __init__(self) -> None:
        self.actions: list[object] = []
        self.closed = False
        self.revision = 0

    async def execute(self, action):
        self.actions.append(action)
        self.revision += 1
        return make_page(self.revision)

    async def snapshot(self) -> BrowserPage:
        self.revision += 1
        return make_page(self.revision)

    async def screenshot(self, *, full_page: bool = False) -> BrowserScreenshot:
        return BrowserScreenshot(
            data=b"png-data",
            mime_type="image/png",
            width=1280,
            height=720,
            full_page=full_page,
        )

    async def aclose(self) -> None:
        self.closed = True


class RecordingRuntime:
    def __init__(self) -> None:
        self.session = RecordingSession()
        self.browse_calls: list[tuple[BrowseRequest, str | None]] = []
        self.closed = False

    async def browse(self, request: BrowseRequest, *, provider=None):
        self.browse_calls.append((request, provider))
        return self.session

    async def aclose(self) -> None:
        self.closed = True


class RecordingBrowseProvider:
    provider_id = "test-browser"
    profile = ProviderProfile(
        provider_id=provider_id,
        display_name="Test Browser",
        description="Browser used by bridge tests",
        capabilities=frozenset({Capability.BROWSE}),
        strengths=("Deterministic",),
        limitations=("Tests only",),
    )

    def __init__(self) -> None:
        self.sessions: list[RecordingSession] = []
        self.closed = False

    async def browse(self, request: BrowseRequest) -> BrowserSession:
        session = RecordingSession()
        self.sessions.append(session)
        return session

    async def aclose(self) -> None:
        self.closed = True


def test_bridge_keeps_a_session_and_serializes_browser_objects(tmp_path) -> None:
    module = load_bridge_module()
    runtime = RecordingRuntime()
    bridge = module.BrowseBridge(runtime, screenshot_dir=tmp_path)

    async def scenario():
        opened = await bridge.handle(
            {
                "op": "open",
                "url": "https://example.com",
                "provider": "playwright",
                "identity": "test-user",
                "visible": False,
            }
        )
        clicked = await bridge.handle(
            {"op": "click", "target": {"ref": "r1-e1"}}
        )
        await bridge.handle({"op": "click", "x": 212.5, "y": 313})
        await bridge.aclose()
        return opened, clicked

    opened, clicked = run(scenario())

    request, provider = runtime.browse_calls[0]
    assert request == BrowseRequest(
        "https://example.com",
        identity=IdentityRef("test-user"),
        visible=False,
    )
    assert provider == "playwright"
    assert opened["page"]["url"] == "https://example.com/final"
    assert opened["page"]["elements"][0]["ref"] == "r1-e1"
    assert clicked["page"]["revision"] == 2
    assert runtime.session.actions == [
        ClickAction(module.BrowserTarget(ref="r1-e1")),
        ClickAction(position=(212.5, 313.0)),
    ]
    assert runtime.session.closed is True
    assert runtime.closed is True


def test_bridge_converts_fill_targets_and_writes_screenshots(tmp_path) -> None:
    module = load_bridge_module()
    runtime = RecordingRuntime()
    bridge = module.BrowseBridge(runtime, screenshot_dir=tmp_path)

    async def scenario():
        await bridge.handle({"op": "open", "url": "https://example.com"})
        await bridge.handle(
            {
                "op": "fill",
                "target": {"selector": "#query"},
                "value": "latest news",
                "submit": True,
            }
        )
        return await bridge.handle(
            {"op": "screenshot", "filename": "page.png", "full_page": True}
        )

    screenshot = run(scenario())

    assert runtime.session.actions[-1] == FillAction(
        module.BrowserTarget(selector="#query"),
        "latest news",
        submit=True,
    )
    assert screenshot == {
        "screenshot": {
            "path": str((tmp_path / "page.png").resolve()),
            "mime_type": "image/png",
            "width": 1280,
            "height": 720,
            "full_page": True,
            "size": 8,
        }
    }
    assert (tmp_path / "page.png").read_bytes() == b"png-data"


def test_bridge_json_line_protocol_returns_structured_errors(tmp_path) -> None:
    module = load_bridge_module()
    bridge = module.BrowseBridge(RecordingRuntime(), screenshot_dir=tmp_path)

    missing_session = run(
        module.handle_line(bridge, '{"id":"a","op":"snapshot"}')
    )
    invalid_json = run(module.handle_line(bridge, "not json"))

    assert missing_session == {
        "id": "a",
        "ok": False,
        "error": {
            "type": "RuntimeError",
            "message": "no browser session is open",
        },
    }
    assert invalid_json["ok"] is False
    assert invalid_json["error"]["type"] == "JSONDecodeError"


def test_bridge_converts_all_remaining_stable_actions(tmp_path) -> None:
    module = load_bridge_module()
    runtime = RecordingRuntime()
    bridge = module.BrowseBridge(runtime, screenshot_dir=tmp_path)
    upload = tmp_path / "reference.png"
    upload.write_bytes(b"image")

    async def scenario():
        await bridge.handle({"op": "open", "url": "https://example.com"})
        await bridge.handle({"op": "navigate", "url": "https://example.com/a"})
        await bridge.handle(
            {"op": "select", "target": {"ref": "r2-e1"}, "values": ["a", "b"]}
        )
        await bridge.handle(
            {
                "op": "upload",
                "target": {"selector": "input[type=file]"},
                "files": [str(upload)],
            }
        )
        await bridge.handle(
            {"op": "press", "target": {"selector": "#field"}, "key": "Enter"}
        )
        await bridge.handle({"op": "press", "key": "Escape"})
        await bridge.handle({"op": "scroll", "delta_y": 500})
        await bridge.handle({"op": "wait", "text": "Ready", "timeout": 2})
        await bridge.handle({"op": "back"})
        await bridge.handle({"op": "forward"})
        await bridge.handle({"op": "reload"})
        closed = await bridge.handle({"op": "close"})
        return closed

    assert run(scenario()) == {"closed": True}
    assert runtime.session.actions == [
        NavigateAction("https://example.com/a"),
        SelectAction(module.BrowserTarget(ref="r2-e1"), ("a", "b")),
        UploadAction(
            module.BrowserTarget(selector="input[type=file]"),
            (upload.resolve(),),
        ),
        PressAction("Enter", module.BrowserTarget(selector="#field")),
        PressAction("Escape"),
        ScrollAction(delta_y=500),
        WaitAction(text="Ready", timeout=2),
        HistoryAction(HistoryDirection.BACK),
        HistoryAction(HistoryDirection.FORWARD),
        ReloadAction(),
    ]


def test_bridge_does_not_replace_an_invalid_zero_timeout(tmp_path) -> None:
    module = load_bridge_module()
    bridge = module.BrowseBridge(RecordingRuntime(), screenshot_dir=tmp_path)

    async def scenario():
        await bridge.handle({"op": "open", "url": "https://example.com"})
        await bridge.handle({"op": "wait", "text": "Ready", "timeout": 0})

    with pytest.raises(ValueError, match="timeout"):
        run(scenario())


def test_bridge_profiles_reopen_defaults_and_shutdown(tmp_path) -> None:
    module = load_bridge_module()
    runtime = WebSearchRuntime()
    provider = RecordingBrowseProvider()
    runtime.register(provider)
    bridge = module.BrowseBridge(runtime, screenshot_dir=tmp_path)

    async def scenario():
        profiles = await bridge.handle({"op": "profiles"})
        await bridge.handle(
            {
                "op": "open",
                "url": "https://example.com/one",
                "provider": "test-browser",
            }
        )
        await bridge.handle(
            {
                "op": "open",
                "url": "https://example.com/two",
                "provider": "test-browser",
            }
        )
        snapshot = await bridge.handle({"op": "snapshot"})
        screenshot = await bridge.handle({"op": "screenshot"})
        first_close = await bridge.handle({"op": "close"})
        second_close = await bridge.handle({"op": "close"})
        shutdown = await bridge.handle({"op": "quit"})
        return profiles, snapshot, screenshot, first_close, second_close, shutdown

    profiles, snapshot, screenshot, first_close, second_close, shutdown = run(
        scenario()
    )

    assert profiles["providers"][0]["provider_id"] == "test-browser"
    assert profiles["providers"][0]["capabilities"] == ["browse"]
    assert provider.sessions[0].closed is True
    assert snapshot["page"]["revision"] == 2
    assert Path(screenshot["screenshot"]["path"]).name.startswith("browse-")
    assert first_close == {"closed": True}
    assert second_close == {"closed": False}
    assert shutdown == {"closed": True}
    assert provider.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        run(bridge.handle({"op": "profiles"}))


def test_bridge_rejects_invalid_commands_and_wraps_success(tmp_path) -> None:
    module = load_bridge_module()
    with pytest.raises(ValueError, match="content_limit"):
        module.BrowseBridge(
            RecordingRuntime(), screenshot_dir=tmp_path, content_limit=0
        )
    bridge = module.BrowseBridge(RecordingRuntime(), screenshot_dir=tmp_path)

    assert run(module.handle_line(bridge, "[]"))["error"]["type"] == "ValueError"
    success = run(
        module.handle_line(
            bridge,
            '{"id":7,"op":"open","url":"https://example.com"}',
        )
    )
    assert success["id"] == 7
    assert success["ok"] is True

    with pytest.raises(ValueError, match="visible"):
        run(
            bridge.handle(
                {
                    "op": "open",
                    "url": "https://example.com",
                    "visible": "yes",
                }
            )
        )
    with pytest.raises(ValueError, match="unsupported operation"):
        run(bridge.handle({"op": "unknown"}))
    run(bridge.aclose())
    run(bridge.aclose())


@pytest.mark.parametrize("filename", ["../page.png", "folder/page.png", "x.jpg"])
def test_bridge_rejects_unsafe_screenshot_names(tmp_path, filename: str) -> None:
    module = load_bridge_module()
    bridge = module.BrowseBridge(RecordingRuntime(), screenshot_dir=tmp_path)

    async def scenario():
        await bridge.handle({"op": "open", "url": "https://example.com"})
        await bridge.handle({"op": "screenshot", "filename": filename})

    with pytest.raises(ValueError, match="filename"):
        run(scenario())
