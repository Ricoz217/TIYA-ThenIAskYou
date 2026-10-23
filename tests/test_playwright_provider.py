from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest

from TIYA.agent_tools.websearch.tools import (
    BrowseRequest,
    BrowserTarget,
    Capability,
    ClickAction,
    FillAction,
    HistoryAction,
    HistoryDirection,
    IdentityRef,
    IdentityStore,
    InteractionKind,
    MouseButton,
    NavigateAction,
    PlaywrightProvider,
    PressAction,
    ProviderRequestError,
    ReloadAction,
    ScrollAction,
    SelectAction,
    UploadAction,
    WaitAction,
)


def run(coroutine):
    return asyncio.run(coroutine)


class FakeLocator:
    def __init__(self, selector: str, calls: list[tuple]) -> None:
        self.selector = selector
        self.calls = calls
        self.first = self

    async def click(self, **kwargs) -> None:
        self.calls.append(("click", self.selector, kwargs))

    async def fill(self, value: str) -> None:
        self.calls.append(("fill", self.selector, value))

    async def press_sequentially(self, value: str) -> None:
        self.calls.append(("press_sequentially", self.selector, value))

    async def press(self, key: str) -> None:
        self.calls.append(("press", self.selector, key))

    async def select_option(self, values: list[str]) -> None:
        self.calls.append(("select_option", self.selector, values))

    async def set_input_files(self, files: list[str]) -> None:
        self.calls.append(("set_input_files", self.selector, files))

    async def evaluate(self, expression: str, argument) -> None:
        self.calls.append(("element_evaluate", self.selector, argument))

    async def wait_for(self, **kwargs) -> None:
        self.calls.append(("wait_for", self.selector, kwargs))


class FakeMouse:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    async def wheel(self, delta_x: int, delta_y: int) -> None:
        self.calls.append(("wheel", delta_x, delta_y))

    async def click(self, x: float, y: float, **kwargs) -> None:
        self.calls.append(("mouse_click", x, y, kwargs))


class FakeKeyboard:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    async def press(self, key: str) -> None:
        self.calls.append(("keyboard_press", key))


class FakePage:
    def __init__(self) -> None:
        self.url = "https://example.com/start"
        self.calls: list[tuple] = []
        self.mouse = FakeMouse(self.calls)
        self.keyboard = FakeKeyboard(self.calls)
        self.snapshot_payload = {
            "title": "Example",
            "text": "Welcome Continue",
            "elements": [
                {
                    "ref": "r1-e1",
                    "selector": '[data-tiya-web-ref="r1-e1"]',
                    "role": "button",
                    "name": "Continue",
                    "tag": "button",
                    "text": "Continue",
                    "value": None,
                    "href": None,
                    "clickable": True,
                    "editable": False,
                    "enabled": True,
                    "visible": True,
                    "attributes": {"type": "button"},
                },
                {
                    "ref": "r1-e2",
                    "selector": '[data-tiya-web-ref="r1-e2"]',
                    "role": "textbox",
                    "name": "Query",
                    "tag": "input",
                    "text": "",
                    "value": "",
                    "href": None,
                    "clickable": False,
                    "editable": True,
                    "enabled": True,
                    "visible": True,
                    "attributes": {"type": "search"},
                },
            ],
            "links": [
                {
                    "url": "https://example.com/docs",
                    "text": "Docs",
                    "title": "Documentation",
                }
            ],
            "media": [
                {
                    "url": "https://example.com/hero.png",
                    "kind": "image",
                    "alt": "Hero",
                    "mime_type": "image/png",
                    "metadata": {
                        "source": "css-background",
                        "tag": "section",
                        "context_text": "Featured character",
                        "width": 1280,
                        "height": 720,
                    },
                }
            ],
        }

    async def goto(self, url: str, **kwargs) -> None:
        self.calls.append(("goto", url, kwargs))
        self.url = url

    async def evaluate(self, expression: str, argument=None):
        self.calls.append(("evaluate", argument))
        return self.snapshot_payload if argument is not None else None

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(selector, self.calls)

    def get_by_text(self, text: str, *, exact: bool = False) -> FakeLocator:
        return FakeLocator(f"text={text}", self.calls)

    async def go_back(self, **kwargs) -> None:
        self.calls.append(("go_back", kwargs))

    async def go_forward(self, **kwargs) -> None:
        self.calls.append(("go_forward", kwargs))

    async def reload(self, **kwargs) -> None:
        self.calls.append(("reload", kwargs))

    async def screenshot(self, **kwargs) -> bytes:
        self.calls.append(("screenshot", kwargs))
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 1280, 720)


class FakeContext:
    def __init__(self) -> None:
        self.page = FakePage()
        self.closed = False

    async def new_page(self) -> FakePage:
        return self.page

    async def close(self) -> None:
        self.closed = True


class FakeBrowser:
    def __init__(self) -> None:
        self.contexts: list[FakeContext] = []
        self.closed = False

    async def new_context(self, **kwargs) -> FakeContext:
        context = FakeContext()
        context.page.calls.append(("new_context", kwargs))
        self.contexts.append(context)
        return context

    async def close(self) -> None:
        self.closed = True


class FakeBrowserType:
    def __init__(self) -> None:
        self.persistent_calls: list[tuple[str, dict[str, object]]] = []
        self.contexts: list[FakeContext] = []

    async def launch(self, **kwargs) -> FakeBrowser:
        return FakeBrowser()

    async def launch_persistent_context(
        self, user_data_dir: str, **kwargs: object
    ) -> FakeContext:
        self.persistent_calls.append((user_data_dir, kwargs))
        context = FakeContext()
        self.contexts.append(context)
        return context


class FakePlaywright:
    def __init__(self) -> None:
        self.chromium = FakeBrowserType()
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class FakePlaywrightStarter:
    def __init__(self, playwright: FakePlaywright) -> None:
        self.playwright = playwright

    async def start(self) -> FakePlaywright:
        return self.playwright


def open_session():
    browser = FakeBrowser()
    provider = PlaywrightProvider(browser=browser)
    session = run(provider.browse(BrowseRequest("https://example.com/start")))
    return provider, browser, session


def test_profile_declares_browse_capability() -> None:
    assert PlaywrightProvider.profile.capabilities == frozenset(
        {Capability.BROWSE}
    )


def test_snapshot_returns_document_elements_links_and_media() -> None:
    provider, browser, session = open_session()

    page = run(session.snapshot())

    assert page.url == "https://example.com/start"
    assert page.title == "Example"
    assert page.document.content == "Welcome Continue"
    assert page.element("r1-e1").clickable is True
    assert page.find(role="textbox")[0].name == "Query"
    assert page.document.links[0].url == "https://example.com/docs"
    assert page.document.media[0].url == "https://example.com/hero.png"
    assert page.document.media[0].metadata == {
        "source": "css-background",
        "tag": "section",
        "context_text": "Featured character",
        "width": 1280,
        "height": 720,
    }
    run(provider.aclose())
    assert browser.contexts[0].closed is True
    assert browser.closed is False


def test_snapshot_detects_cloudflare_human_verification() -> None:
    provider, browser, session = open_session()
    browser.contexts[0].page.url = "https://example.com/search"
    browser.contexts[0].page.snapshot_payload.update(
        {
            "title": "请稍候…",
            "text": "正在验证您是否是真人。这可能需要几秒钟时间。",
            "elements": [
                {
                    "ref": "r1-e1",
                    "selector": '[data-tiya-web-ref="r1-e1"]',
                    "role": "textbox",
                    "name": "",
                    "tag": "input",
                    "text": "",
                    "value": "",
                    "href": None,
                    "clickable": False,
                    "editable": True,
                    "enabled": True,
                    "visible": False,
                    "attributes": {
                        "name": "cf-turnstile-response",
                        "type": "hidden",
                    },
                }
            ],
        }
    )

    page = run(session.snapshot())

    assert page.document.interaction_required is not None
    assert page.document.interaction_required.kind is InteractionKind.CAPTCHA
    assert page.document.interaction_required.resumable is True
    run(provider.aclose())


def test_snapshot_detects_cloudflare_bootstrap_before_widget_loads() -> None:
    provider, browser, session = open_session()
    browser.contexts[0].page.url = "https://example.com/search"
    browser.contexts[0].page.snapshot_payload.update(
        {
            "title": "Just a moment...",
            "text": "",
            "elements": [
                {
                    "ref": "r1-e1",
                    "selector": '[data-tiya-web-ref="r1-e1"]',
                    "role": "main",
                    "name": "Enable JavaScript and cookies to continue",
                    "tag": "div",
                    "text": "Enable JavaScript and cookies to continue",
                    "value": None,
                    "href": None,
                    "clickable": False,
                    "editable": False,
                    "enabled": True,
                    "visible": True,
                    "attributes": {},
                }
            ],
        }
    )

    page = run(session.snapshot())

    assert page.document.interaction_required is not None
    assert page.document.interaction_required.kind is InteractionKind.VERIFICATION
    run(provider.aclose())


def test_execute_maps_every_stable_action_to_playwright() -> None:
    provider, browser, session = open_session()
    page = run(session.snapshot())
    button = page.element("r1-e1")

    actions = [
        NavigateAction("https://example.com/next"),
        ClickAction(BrowserTarget(ref=button.ref), MouseButton.RIGHT, 2),
        ClickAction(position=(212.5, 313.0)),
        ClickAction(BrowserTarget(selector="#map"), position=(12.0, 8.0)),
        FillAction(BrowserTarget(selector="#query"), "news", submit=True),
        FillAction(BrowserTarget(selector="#append"), "more", clear=False),
        SelectAction(BrowserTarget(selector="#kind"), ("news", "docs")),
        PressAction("Escape"),
        PressAction("Enter", BrowserTarget(selector="#query")),
        ScrollAction(delta_x=10, delta_y=20),
        ScrollAction(delta_y=30, target=BrowserTarget(selector="#panel")),
        WaitAction(target=BrowserTarget(selector="#ready"), timeout=2),
        WaitAction(text="Finished", timeout=3),
        HistoryAction(HistoryDirection.BACK),
        HistoryAction(HistoryDirection.FORWARD),
        ReloadAction(),
    ]
    for action in actions:
        run(session.execute(action))

    calls = browser.contexts[0].page.calls
    assert any(call[0] == "goto" and call[1].endswith("/next") for call in calls)
    assert (
        "click",
        '[data-tiya-web-ref="r1-e1"]',
        {"button": "right", "click_count": 2},
    ) in calls
    assert (
        "mouse_click",
        212.5,
        313.0,
        {"button": "left", "click_count": 1},
    ) in calls
    assert (
        "click",
        "#map",
        {"button": "left", "click_count": 1, "position": {"x": 12.0, "y": 8.0}},
    ) in calls
    assert ("fill", "#query", "news") in calls
    assert ("press", "#query", "Enter") in calls
    assert ("press_sequentially", "#append", "more") in calls
    assert ("select_option", "#kind", ["news", "docs"]) in calls
    assert ("wheel", 10, 20) in calls
    assert ("element_evaluate", "#panel", [0, 30]) in calls
    assert any(call[0] == "go_back" for call in calls)
    assert any(call[0] == "go_forward" for call in calls)
    assert any(call[0] == "reload" for call in calls)
    run(provider.aclose())


def test_upload_sets_input_files_within_allowed_roots(tmp_path: Path) -> None:
    upload = tmp_path / "reference.png"
    upload.write_bytes(b"image")
    browser = FakeBrowser()
    provider = PlaywrightProvider(
        browser=browser,  # type: ignore[arg-type]
        allowed_upload_roots=(tmp_path,),
    )
    session = run(provider.browse(BrowseRequest("https://example.com/start")))

    run(
        session.execute(
            UploadAction(
                BrowserTarget(selector="input[type=file]"),
                (upload,),
            )
        )
    )

    assert (
        "set_input_files",
        "input[type=file]",
        [str(upload.resolve())],
    ) in browser.contexts[0].page.calls
    run(provider.aclose())


def test_upload_rejects_files_outside_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("secret", encoding="utf-8")
    browser = FakeBrowser()
    provider = PlaywrightProvider(
        browser=browser,  # type: ignore[arg-type]
        allowed_upload_roots=(allowed,),
    )
    session = run(provider.browse(BrowseRequest("https://example.com/start")))

    with pytest.raises(ProviderRequestError, match="allowed upload roots"):
        run(session.upload(BrowserTarget(selector="input[type=file]"), outside))

    assert not any(
        call[0] == "set_input_files" for call in browser.contexts[0].page.calls
    )
    run(provider.aclose())


def test_unknown_element_reference_is_rejected() -> None:
    provider, browser, session = open_session()

    with pytest.raises(ProviderRequestError, match="reference"):
        run(session.click("missing"))

    run(provider.aclose())


def test_screenshot_reads_png_dimensions() -> None:
    provider, browser, session = open_session()

    screenshot = run(session.screenshot(full_page=True))

    assert screenshot.mime_type == "image/png"
    assert (screenshot.width, screenshot.height) == (1280, 720)
    assert screenshot.full_page is True
    run(provider.aclose())


def test_closing_provider_closes_all_sessions_but_not_injected_browser() -> None:
    provider, browser, first = open_session()
    second = run(provider.browse(BrowseRequest("https://example.com/other")))

    run(provider.aclose())
    run(first.aclose())
    run(second.aclose())

    assert all(context.closed for context in browser.contexts)
    assert browser.closed is False


def test_identity_browse_uses_visible_persistent_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import TIYA.agent_tools.websearch.tools.providers.playwright_provider as module

    playwright = FakePlaywright()
    monkeypatch.setattr(
        module,
        "async_playwright",
        lambda: FakePlaywrightStarter(playwright),
    )
    identities = IdentityStore(tmp_path / "profiles")
    provider = PlaywrightProvider(identity_store=identities, headless=True)

    session = run(
        provider.browse(
            BrowseRequest(
                "https://example.com/login",
                identity=IdentityRef("example-user"),
                visible=True,
            )
        )
    )

    profile_path, options = playwright.chromium.persistent_calls[0]
    assert Path(profile_path) == identities.resolve(IdentityRef("example-user"))
    assert options["headless"] is False
    run(session.aclose())
    run(provider.aclose())
    assert playwright.chromium.contexts[0].closed is True
    assert playwright.stopped is True
