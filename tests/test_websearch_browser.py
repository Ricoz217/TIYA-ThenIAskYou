from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from TIYA.agent_tools.websearch.tools import (
    BrowserElement,
    BrowserPage,
    BrowserScreenshot,
    BrowserSession,
    BrowserTarget,
    ClickAction,
    ContentFormat,
    FillAction,
    HistoryAction,
    HistoryDirection,
    NavigateAction,
    PressAction,
    ReloadAction,
    ScrollAction,
    SelectAction,
    UploadAction,
    WaitAction,
    WebDocument,
)


def run(coroutine):
    return asyncio.run(coroutine)


def make_page(revision: int = 1) -> BrowserPage:
    return BrowserPage(
        session_id="session-1",
        revision=revision,
        document=WebDocument(
            url="https://example.com",
            provider="browser",
            title="Example",
            content="Example page",
            content_format=ContentFormat.TEXT,
        ),
        elements=(
            BrowserElement(
                ref="button-1",
                role="button",
                name="Continue",
                tag="button",
                text="Continue",
                clickable=True,
            ),
            BrowserElement(
                ref="input-1",
                role="textbox",
                name="Query",
                tag="input",
                editable=True,
            ),
        ),
    )


class RecordingBrowserSession(BrowserSession):
    provider_id = "browser"
    session_id = "session-1"

    def __init__(self) -> None:
        self.actions: list[object] = []
        self.close_count = 0
        self._revision = 0

    async def execute(self, action):
        self.actions.append(action)
        self._revision += 1
        return make_page(self._revision)

    async def snapshot(self) -> BrowserPage:
        return make_page(self._revision)

    async def screenshot(self, *, full_page: bool = False) -> BrowserScreenshot:
        return BrowserScreenshot(
            data=b"png",
            mime_type="image/png",
            width=1280,
            height=720,
            full_page=full_page,
        )

    async def aclose(self) -> None:
        self.close_count += 1


def test_browser_session_exposes_stable_convenience_operations(
    tmp_path: Path,
) -> None:
    upload = tmp_path / "reference.png"
    upload.write_bytes(b"image")

    async def scenario() -> RecordingBrowserSession:
        session = RecordingBrowserSession()
        element = make_page().element("button-1")
        await session.navigate("https://example.com/next")
        await session.click(element)
        await session.fill("input-1", "deepseek", submit=True)
        await session.select(BrowserTarget(selector="#kind"), ("news", "docs"))
        await session.upload(BrowserTarget(selector="input[type=file]"), upload)
        await session.press("Enter", target="input-1")
        await session.scroll(delta_y=800)
        await session.wait(text="Finished", timeout=5)
        await session.back()
        await session.forward()
        await session.reload()
        return session

    session = run(scenario())

    assert session.actions == [
        NavigateAction("https://example.com/next"),
        ClickAction(BrowserTarget(ref="button-1")),
        FillAction(BrowserTarget(ref="input-1"), "deepseek", submit=True),
        SelectAction(BrowserTarget(selector="#kind"), ("news", "docs")),
        UploadAction(
            BrowserTarget(selector="input[type=file]"),
            (upload.resolve(),),
        ),
        PressAction("Enter", BrowserTarget(ref="input-1")),
        ScrollAction(delta_y=800),
        WaitAction(text="Finished", timeout=5),
        HistoryAction(HistoryDirection.BACK),
        HistoryAction(HistoryDirection.FORWARD),
        ReloadAction(),
    ]


def test_browser_session_is_an_async_context_manager() -> None:
    async def scenario() -> int:
        session = RecordingBrowserSession()
        async with session as opened:
            assert opened is session
        return session.close_count

    assert run(scenario()) == 1


def test_browser_page_supports_object_lookup_and_filtering() -> None:
    page = make_page()

    assert page.url == "https://example.com"
    assert page.title == "Example"
    assert page.element("button-1").name == "Continue"
    assert page.find(role="textbox") == (page.element("input-1"),)
    assert page.find(name="continue") == (page.element("button-1"),)
    with pytest.raises(KeyError, match="missing"):
        page.element("missing")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BrowserTarget(),
        lambda: BrowserTarget(ref="a", selector="#a"),
        lambda: NavigateAction("ftp://example.com"),
        lambda: ClickAction(BrowserTarget(ref="a"), click_count=0),
        lambda: ClickAction(),
        lambda: ClickAction(position=(-1, 20)),
        lambda: SelectAction(BrowserTarget(ref="a"), ()),
        lambda: UploadAction(BrowserTarget(ref="a"), ()),
        lambda: PressAction(""),
        lambda: ScrollAction(),
        lambda: WaitAction(),
        lambda: WaitAction(duration=-1),
        lambda: BrowserScreenshot(
            data=b"",
            mime_type="image/png",
            width=1,
            height=1,
        ),
    ],
)
def test_browser_contract_rejects_invalid_values(factory) -> None:
    with pytest.raises(ValueError):
        factory()
