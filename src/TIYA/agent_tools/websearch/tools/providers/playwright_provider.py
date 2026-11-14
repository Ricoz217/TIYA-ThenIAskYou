"""Stateful browser sessions backed by Playwright's asynchronous API."""

from __future__ import annotations

import asyncio
import struct
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from ..browser import (
    BrowserAction,
    BrowserElement,
    BrowserPage,
    BrowserScreenshot,
    BrowserSession,
    BrowserTarget,
    ClickAction,
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
)
from ..errors import InvalidProviderResponseError, ProviderRequestError
from ..identity import IdentityStore
from ..models import (
    BrowseRequest,
    Capability,
    ContentFormat,
    IdentityRef,
    InteractionKind,
    InteractionRequired,
    Link,
    MediaKind,
    MediaResource,
    ProviderProfile,
    WebDocument,
)


_SNAPSHOT_SCRIPT = r"""
({ revision, attribute }) => {
  document.querySelectorAll(`[${attribute}]`).forEach(
    element => element.removeAttribute(attribute)
  );
  const candidates = Array.from(document.querySelectorAll(
    'a[href],button,input,select,textarea,[role],'
      + '[contenteditable="true"],[onclick],summary'
  ));
  const implicitRole = element => {
    const tag = element.tagName.toLowerCase();
    if (tag === 'a' && element.href) return 'link';
    if (tag === 'button') return 'button';
    if (tag === 'select') return 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      const type = (element.type || 'text').toLowerCase();
      if (['button', 'submit', 'reset'].includes(type)) return 'button';
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      return 'textbox';
    }
    return '';
  };
  const elements = candidates.map((element, index) => {
    const ref = `r${revision}-e${index + 1}`;
    element.setAttribute(attribute, ref);
    const style = getComputedStyle(element);
    const visible = style.display !== 'none' && style.visibility !== 'hidden'
      && element.getClientRects().length > 0;
    const text = (element.innerText || element.textContent || '').trim();
    const name = (element.getAttribute('aria-label') || element.title
      || element.alt || element.placeholder || text).trim();
    const attributes = {};
    for (const key of ['id', 'name', 'type', 'placeholder', 'aria-label', 'title']) {
      const value = element.getAttribute(key);
      if (value !== null) attributes[key] = value;
    }
    return {
      ref,
      selector: `[${attribute}="${ref}"]`,
      role: element.getAttribute('role') || implicitRole(element),
      name,
      tag: element.tagName.toLowerCase(),
      text,
      value: 'value' in element ? String(element.value) : null,
      href: element.href || null,
      clickable: Boolean(element.onclick) || ['a', 'button', 'summary'].includes(
        element.tagName.toLowerCase()
      ) || ['button', 'submit', 'reset', 'checkbox', 'radio'].includes(
        (element.type || '').toLowerCase()
      ),
      editable: ['input', 'textarea', 'select'].includes(
        element.tagName.toLowerCase()
      ) || Boolean(element.isContentEditable),
      enabled: !element.disabled,
      visible,
      attributes,
    };
  });
  const links = Array.from(document.querySelectorAll('a[href]')).map(element => ({
    url: element.href,
    text: (element.innerText || element.textContent || '').trim(),
    title: element.title || '',
  }));
  const contextText = element => {
    const figure = element.closest('figure');
    const caption = figure ? figure.querySelector('figcaption') : null;
    if (caption && caption.textContent) return caption.textContent.trim().slice(0, 500);
    const labelled = element.getAttribute('aria-label') || element.title || '';
    return labelled.trim().slice(0, 500);
  };
  const dimensions = element => {
    const rect = element.getBoundingClientRect();
    const width = element.naturalWidth || element.videoWidth || Math.round(rect.width);
    const height = element.naturalHeight || element.videoHeight
      || Math.round(rect.height);
    return { width, height };
  };
  const elementMedia = Array.from(document.querySelectorAll(
    'img[src],img[srcset],video[src],audio[src],source[src]'
  )).map(element => {
    const tag = element.tagName.toLowerCase();
    const parentTag = element.parentElement
      ? element.parentElement.tagName.toLowerCase() : '';
    const kind = tag === 'img' || parentTag === 'picture' ? 'image'
      : tag === 'audio' || parentTag === 'audio' ? 'audio' : 'video';
    const anchor = element.closest('a[href]');
    return {
      url: element.currentSrc || element.src,
      kind,
      alt: element.alt || element.getAttribute('aria-label') || element.title || '',
      mime_type: element.type || null,
      metadata: {
        source: 'element',
        tag,
        source_attribute: element.currentSrc ? 'currentSrc' : 'src',
        title: element.title || '',
        context_text: contextText(element),
        link_url: anchor ? anchor.href : '',
        ...dimensions(element),
      },
    };
  });
  const backgroundMedia = Array.from(
    document.body ? document.body.querySelectorAll('*') : []
  ).flatMap(element => {
    const style = getComputedStyle(element);
    if (style.display === 'none' || style.visibility === 'hidden'
        || element.getClientRects().length === 0
        || !style.backgroundImage || style.backgroundImage === 'none') return [];
    const matches = Array.from(style.backgroundImage.matchAll(
      /url\(\s*(["']?)(.*?)\1\s*\)/g
    ));
    return matches.map(match => ({
      url: new URL(match[2], document.baseURI).href,
      kind: 'image',
      alt: element.getAttribute('aria-label') || element.title || '',
      mime_type: null,
      metadata: {
        source: 'css-background',
        tag: element.tagName.toLowerCase(),
        context_text: contextText(element),
        ...dimensions(element),
      },
    }));
  });
  const mediaByUrl = new Map();
  for (const item of [...elementMedia, ...backgroundMedia]) {
    if (item.url && !mediaByUrl.has(item.url)) mediaByUrl.set(item.url, item);
  }
  const media = Array.from(mediaByUrl.values());
  return {
    title: document.title || '',
    text: document.body ? document.body.innerText : '',
    elements,
    links,
    media,
    viewport: { width: window.innerWidth, height: window.innerHeight },
  };
}
"""


class PlaywrightProvider:
    """Create object-oriented, JavaScript-capable Playwright sessions."""

    provider_id = "playwright"
    profile = ProviderProfile(
        provider_id=provider_id,
        display_name="Playwright Browser",
        description="Stateful JavaScript browser automation with semantic snapshots.",
        capabilities=frozenset({Capability.BROWSE}),
        strengths=(
            (
                "Supports navigation, click, fill, file upload, select, keyboard, "
                "scroll, and wait"
            ),
            "Returns semantic element objects that can be used in later actions",
            (
                "Discovers rendered element and CSS-background media with "
                "DOM context and dimensions"
            ),
            "Captures screenshots and handles client-rendered pages",
        ),
        limitations=(
            "Higher latency and resource use than search, extract, or direct HTTP",
            "Requires a locally installed Playwright browser binary",
            "File upload is restricted to configured staging and download roots",
            "Element references are valid only until the next page snapshot",
        ),
    )

    def __init__(
        self,
        *,
        browser: Browser | None = None,
        browser_type: str = "chromium",
        headless: bool = True,
        launch_options: Mapping[str, Any] | None = None,
        context_options: Mapping[str, Any] | None = None,
        navigation_timeout: float = 30.0,
        identity_store: IdentityStore | None = None,
        allowed_upload_roots: Sequence[str | Path] = (),
    ) -> None:
        if browser_type not in {"chromium", "firefox", "webkit"}:
            raise ValueError("browser_type must be chromium, firefox, or webkit")
        if navigation_timeout <= 0:
            raise ValueError("navigation_timeout must be positive")
        self._browser = browser
        self._browser_type = browser_type
        self._headless = headless
        self._launch_options = dict(launch_options or {})
        self._context_options = dict(context_options or {})
        self._navigation_timeout = navigation_timeout
        self._identity_store = identity_store
        self._allowed_upload_roots = tuple(
            Path(root).resolve() for root in allowed_upload_roots
        )
        self._owns_browser = browser is None
        self._playwright: Any = None
        self._sessions: set[PlaywrightBrowserSession] = set()
        self._launch_lock = asyncio.Lock()

    async def browse(self, request: BrowseRequest) -> BrowserSession:
        """Open a fresh isolated browser context and navigate to the URL."""

        context: BrowserContext | None = None
        try:
            if request.identity is not None:
                if self._identity_store is None:
                    raise ProviderRequestError(
                        provider=self.provider_id,
                        operation=Capability.BROWSE,
                        message=(
                            "an identity was requested but no identity store "
                            "is configured"
                        ),
                    )
                playwright = await self._ensure_playwright()
                launcher = getattr(playwright, self._browser_type)
                identity_dir = self._identity_store.resolve(
                    request.identity, create=True
                )
                headless = (
                    self._headless
                    if request.visible is None
                    else not request.visible
                )
                context = await launcher.launch_persistent_context(
                    str(identity_dir),
                    headless=headless,
                    **self._launch_options,
                    **self._context_options,
                )
            else:
                browser = await self._ensure_browser()
                context = await browser.new_context(**self._context_options)
            assert context is not None
            page = await context.new_page()
            session = PlaywrightBrowserSession(
                context=context,
                page=page,
                navigation_timeout=self._navigation_timeout,
                allowed_upload_roots=self._allowed_upload_roots,
                identity=request.identity,
                on_close=self._sessions.discard,
            )
            self._sessions.add(session)
            await session._navigate(request.url)
            return session
        except PlaywrightTimeoutError as exc:
            if context is not None:
                await context.close()
            raise _browser_error(str(exc), retryable=True) from exc
        except PlaywrightError as exc:
            if context is not None:
                await context.close()
            raise _browser_error(str(exc)) from exc

    async def aclose(self) -> None:
        """Close sessions and browser resources owned by this provider."""

        for session in tuple(self._sessions):
            await session.aclose()
        if self._owns_browser and self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def _ensure_browser(self) -> Browser:
        if self._browser is not None:
            return self._browser
        async with self._launch_lock:
            if self._browser is not None:
                return self._browser
            playwright = await self._ensure_playwright()
            launcher = getattr(playwright, self._browser_type)
            options = {"headless": self._headless, **self._launch_options}
            try:
                self._browser = await launcher.launch(**options)
            except Exception:
                await playwright.stop()
                self._playwright = None
                raise
            return self._browser

    async def _ensure_playwright(self) -> Any:
        if self._playwright is not None:
            return self._playwright
        self._playwright = await async_playwright().start()
        return self._playwright


class PlaywrightBrowserSession(BrowserSession):
    """One isolated Playwright context implementing the browser object contract."""

    provider_id = PlaywrightProvider.provider_id

    def __init__(
        self,
        *,
        context: BrowserContext,
        page: Page,
        navigation_timeout: float,
        allowed_upload_roots: tuple[Path, ...],
        identity: IdentityRef | None,
        on_close: Callable[[PlaywrightBrowserSession], None],
    ) -> None:
        self.session_id = uuid.uuid4().hex
        self._context = context
        self._page = page
        self._navigation_timeout = navigation_timeout
        self._allowed_upload_roots = allowed_upload_roots
        self._identity = identity
        self._on_close = on_close
        self._revision = 0
        self._selectors: dict[str, str] = {}
        self._closed = False

    async def execute(self, action: BrowserAction) -> BrowserPage:
        """Translate one provider-neutral action to Playwright calls."""

        try:
            if isinstance(action, NavigateAction):
                await self._navigate(action.url)
            elif isinstance(action, ClickAction):
                button = cast(
                    Literal["left", "middle", "right"], action.button.value
                )
                if action.target is None:
                    assert action.position is not None
                    await self._page.mouse.click(
                        *action.position,
                        button=button,
                        click_count=action.click_count,
                    )
                else:
                    if action.position is not None:
                        await self._locator(action.target).click(
                            button=button,
                            click_count=action.click_count,
                            position={
                                "x": action.position[0],
                                "y": action.position[1],
                            },
                        )
                    else:
                        await self._locator(action.target).click(
                            button=button,
                            click_count=action.click_count,
                        )
            elif isinstance(action, FillAction):
                locator = self._locator(action.target)
                if action.clear:
                    await locator.fill(action.value)
                else:
                    await locator.press_sequentially(action.value)
                if action.submit:
                    await locator.press("Enter")
            elif isinstance(action, SelectAction):
                await self._locator(action.target).select_option(list(action.values))
            elif isinstance(action, UploadAction):
                self._validate_upload_files(action.files)
                await self._locator(action.target).set_input_files(
                    [str(file) for file in action.files]
                )
            elif isinstance(action, PressAction):
                if action.target is None:
                    await self._page.keyboard.press(action.key)
                else:
                    await self._locator(action.target).press(action.key)
            elif isinstance(action, ScrollAction):
                if action.target is None:
                    await self._page.mouse.wheel(action.delta_x, action.delta_y)
                else:
                    await self._locator(action.target).evaluate(
                        "(element, delta) => element.scrollBy(delta[0], delta[1])",
                        [action.delta_x, action.delta_y],
                    )
            elif isinstance(action, WaitAction):
                await self._wait(action)
            elif isinstance(action, HistoryAction):
                if action.direction == HistoryDirection.BACK:
                    await self._page.go_back(
                        wait_until="domcontentloaded",
                        timeout=self._timeout_ms,
                    )
                else:
                    await self._page.go_forward(
                        wait_until="domcontentloaded",
                        timeout=self._timeout_ms,
                    )
            elif isinstance(action, ReloadAction):
                await self._page.reload(
                    wait_until="domcontentloaded",
                    timeout=self._timeout_ms,
                )
            else:
                raise TypeError(f"unsupported browser action: {type(action).__name__}")
            return await self.snapshot()
        except ProviderRequestError:
            raise
        except PlaywrightTimeoutError as exc:
            raise _browser_error(str(exc), retryable=True) from exc
        except PlaywrightError as exc:
            raise _browser_error(str(exc)) from exc

    async def snapshot(self) -> BrowserPage:
        """Capture text, links, media, and semantic action targets."""

        self._ensure_open()
        self._revision += 1
        try:
            raw = await self._page.evaluate(
                _SNAPSHOT_SCRIPT,
                {
                    "revision": self._revision,
                    "attribute": "data-tiya-web-ref",
                },
            )
            data = _mapping(raw, "snapshot")
            elements, selectors = _elements(data.get("elements"))
            self._selectors = selectors
            current_url = self._page.url
            title = _text(data.get("title"), "title")
            content = _text(data.get("text"), "text")
            document = WebDocument(
                url=current_url,
                final_url=current_url,
                provider=self.provider_id,
                title=title,
                content=content,
                content_format=ContentFormat.TEXT,
                content_type="text/plain",
                links=_links(data.get("links"), current_url),
                media=_media(data.get("media")),
                interaction_required=_detect_browser_interaction(
                    elements=elements,
                    url=current_url,
                    title=title,
                    content=content,
                    identity=self._identity,
                ),
            )
            metadata: dict[str, object] = {}
            viewport = data.get("viewport")
            if isinstance(viewport, Mapping):
                metadata["viewport"] = dict(viewport)
            return BrowserPage(
                session_id=self.session_id,
                revision=self._revision,
                document=document,
                elements=elements,
                metadata=metadata,
            )
        except InvalidProviderResponseError:
            raise
        except ValueError as exc:
            raise InvalidProviderResponseError(
                provider=self.provider_id,
                operation=Capability.BROWSE,
                message=str(exc),
            ) from exc
        except PlaywrightTimeoutError as exc:
            raise _browser_error(str(exc), retryable=True) from exc
        except PlaywrightError as exc:
            raise _browser_error(str(exc)) from exc

    async def screenshot(self, *, full_page: bool = False) -> BrowserScreenshot:
        """Capture a PNG and derive its exact pixel dimensions."""

        self._ensure_open()
        try:
            data = await self._page.screenshot(type="png", full_page=full_page)
        except PlaywrightTimeoutError as exc:
            raise _browser_error(str(exc), retryable=True) from exc
        except PlaywrightError as exc:
            raise _browser_error(str(exc)) from exc
        width, height = _png_dimensions(data)
        return BrowserScreenshot(
            data=data,
            mime_type="image/png",
            width=width,
            height=height,
            full_page=full_page,
        )

    async def aclose(self) -> None:
        """Close this isolated browser context once."""

        if self._closed:
            return
        self._closed = True
        try:
            await self._context.close()
        finally:
            self._on_close(self)

    @property
    def _timeout_ms(self) -> float:
        return self._navigation_timeout * 1000

    async def _navigate(self, url: str) -> None:
        self._ensure_open()
        await self._page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=self._timeout_ms,
        )

    def _locator(self, target: BrowserTarget) -> Any:
        self._ensure_open()
        if target.selector is not None:
            return self._page.locator(target.selector)
        selector = self._selectors.get(target.ref or "")
        if selector is None:
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.BROWSE,
                message=(
                    f"element reference {target.ref!r} is unknown or stale; "
                    "take a new snapshot"
                ),
                retryable=False,
            )
        return self._page.locator(selector)

    def _validate_upload_files(self, files: tuple[Path, ...]) -> None:
        if not self._allowed_upload_roots:
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.BROWSE,
                message=(
                    "browser file upload is disabled; configure allowed upload roots"
                ),
                retryable=False,
            )
        for file in files:
            if not any(file.is_relative_to(root) for root in self._allowed_upload_roots):
                raise ProviderRequestError(
                    provider=self.provider_id,
                    operation=Capability.BROWSE,
                    message=(
                        f"upload file {file.name!r} is outside allowed upload roots"
                    ),
                    retryable=False,
                )

    async def _wait(self, action: WaitAction) -> None:
        if action.duration is not None:
            await asyncio.sleep(action.duration)
        timeout = action.timeout * 1000
        if action.target is not None:
            await self._locator(action.target).wait_for(
                state="visible", timeout=timeout
            )
        if action.text is not None:
            await self._page.get_by_text(action.text, exact=False).first.wait_for(
                state="visible", timeout=timeout
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProviderRequestError(
                provider=self.provider_id,
                operation=Capability.BROWSE,
                message="browser session is closed",
                retryable=False,
            )


def _elements(
    value: object,
) -> tuple[tuple[BrowserElement, ...], dict[str, str]]:
    values = _array(value, "elements")
    elements: list[BrowserElement] = []
    selectors: dict[str, str] = {}
    for index, raw in enumerate(values):
        item = _mapping(raw, f"elements[{index}]")
        ref = _text(item.get("ref"), f"elements[{index}].ref", required=True)
        selector = _text(
            item.get("selector"), f"elements[{index}].selector", required=True
        )
        href = _http_url_or_none(item.get("href"))
        attributes = item.get("attributes", {})
        if not isinstance(attributes, Mapping) or not all(
            isinstance(key, str) and isinstance(entry, str)
            for key, entry in attributes.items()
        ):
            raise _invalid(f"elements[{index}].attributes must contain strings")
        element = BrowserElement(
            ref=ref,
            role=_text(item.get("role"), f"elements[{index}].role"),
            name=_text(item.get("name"), f"elements[{index}].name"),
            tag=_text(item.get("tag"), f"elements[{index}].tag"),
            text=_text(item.get("text"), f"elements[{index}].text"),
            value=_nullable_text(item.get("value"), f"elements[{index}].value"),
            href=href,
            clickable=_boolean(item.get("clickable"), f"elements[{index}].clickable"),
            editable=_boolean(item.get("editable"), f"elements[{index}].editable"),
            enabled=_boolean(item.get("enabled"), f"elements[{index}].enabled"),
            visible=_boolean(item.get("visible"), f"elements[{index}].visible"),
            attributes=dict(attributes),
        )
        elements.append(element)
        selectors[ref] = selector
    return tuple(elements), selectors


def _detect_browser_interaction(
    *,
    elements: tuple[BrowserElement, ...],
    url: str,
    title: str,
    content: str,
    identity: IdentityRef | None,
) -> InteractionRequired | None:
    response_names = {
        "g-recaptcha-response",
        "h-captcha-response",
        "cf-turnstile-response",
    }
    has_captcha = any(
        element.attributes.get("name", "").casefold() in response_names
        or element.attributes.get("id", "").casefold().startswith("cf-chl-widget-")
        for element in elements
    )
    if has_captcha:
        return InteractionRequired(
            kind=InteractionKind.CAPTCHA,
            url=url,
            provider=PlaywrightProvider.provider_id,
            message="Page requires human CAPTCHA or anti-bot verification",
            identity=identity,
        )

    element_text = "\n".join(
        value
        for element in elements
        for value in (element.name, element.text)
        if value
    )
    searchable = f"{title}\n{content}\n{element_text}".casefold()
    verification_phrases = (
        "just a moment",
        "verify you are human",
        "confirm you are human",
        "checking your browser before accessing",
        "enable javascript and cookies to continue",
        "正在验证您是否是真人",
        "请稍候",
    )
    if any(phrase in searchable for phrase in verification_phrases):
        return InteractionRequired(
            kind=InteractionKind.VERIFICATION,
            url=url,
            provider=PlaywrightProvider.provider_id,
            message="Page requires human verification before browsing can continue",
            identity=identity,
        )
    return None


def _links(value: object, current_url: str) -> tuple[Link, ...]:
    values = _array(value, "links")
    current_host = urlsplit(current_url).hostname
    links: list[Link] = []
    for index, raw in enumerate(values):
        item = _mapping(raw, f"links[{index}]")
        url = _http_url_or_none(item.get("url"))
        if url is None:
            continue
        links.append(
            Link(
                url=url,
                text=_text(item.get("text"), f"links[{index}].text"),
                title=_text(item.get("title"), f"links[{index}].title"),
                internal=urlsplit(url).hostname == current_host,
            )
        )
    return tuple(links)


def _media(value: object) -> tuple[MediaResource, ...]:
    values = _array(value, "media")
    resources: list[MediaResource] = []
    for index, raw in enumerate(values):
        item = _mapping(raw, f"media[{index}]")
        url = _http_url_or_none(item.get("url"))
        if url is None:
            continue
        raw_kind = _text(item.get("kind"), f"media[{index}].kind")
        try:
            kind = MediaKind(raw_kind)
        except ValueError:
            kind = MediaKind.OTHER
        raw_metadata = item.get("metadata")
        metadata = (
            {}
            if raw_metadata is None
            else dict(_mapping(raw_metadata, f"media[{index}].metadata"))
        )
        resources.append(
            MediaResource(
                url=url,
                kind=kind,
                alt=_text(item.get("alt"), f"media[{index}].alt"),
                mime_type=_nullable_text(
                    item.get("mime_type"), f"media[{index}].mime_type"
                ),
                metadata=metadata,
            )
        )
    return tuple(resources)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _invalid(f"{field} must be an object")
    return value


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise _invalid(f"{field} must be an array")
    return value


def _text(value: object, field: str, *, required: bool = False) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value):
        raise _invalid(f"{field} must be a string")
    return value


def _nullable_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid(f"{field} must be a string or null")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise _invalid(f"{field} must be a boolean")
    return value


def _http_url_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return value


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise InvalidProviderResponseError(
            provider=PlaywrightProvider.provider_id,
            operation=Capability.BROWSE,
            message="screenshot is not a valid PNG",
        )
    return struct.unpack(">II", data[16:24])


def _invalid(message: str) -> InvalidProviderResponseError:
    return InvalidProviderResponseError(
        provider=PlaywrightProvider.provider_id,
        operation=Capability.BROWSE,
        message=message,
    )


def _browser_error(message: str, *, retryable: bool = False) -> ProviderRequestError:
    return ProviderRequestError(
        provider=PlaywrightProvider.provider_id,
        operation=Capability.BROWSE,
        message=message,
        retryable=retryable,
    )


__all__ = ["PlaywrightBrowserSession", "PlaywrightProvider"]
