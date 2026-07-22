from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import secrets
import ssl
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

from aiohttp import WSMsgType, web
from playwright.async_api import (
    Browser,
    BrowserContext,
    CDPSession,
    Page,
    Playwright,
    Error as PlaywrightError,
    async_playwright,
)

from .page import REMOTE_LOGIN_PAGE

_VIEWPORT_WIDTH = 412
_VIEWPORT_HEIGHT = 915
_DUPLICATE_CLICK_WINDOW = 0.5
_ACTIVATION_CLICK_COOLDOWN = 5.0
_ENTER_KEY_COOLDOWN = 2.0
_SUBMISSION_COOLDOWN = 5.0
_CLIENT_COOKIE = "tiya_pixiv_login_client"
_PIXIV_LOGIN_HOST = "app-api.pixiv.net"
_PIXIV_LOGIN_PATH = "/web/v1/login"
_PIXIV_CALLBACK_PATH = "/web/v1/users/auth/pixiv/callback"
_ALLOWED_KEYS = {
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "ArrowUp",
    "Backspace",
    "Delete",
    "Enter",
    "Escape",
    "Tab",
}
_KEY_CODES = {
    "ArrowDown": ("ArrowDown", 40),
    "ArrowLeft": ("ArrowLeft", 37),
    "ArrowRight": ("ArrowRight", 39),
    "ArrowUp": ("ArrowUp", 38),
    "Backspace": ("Backspace", 8),
    "Delete": ("Delete", 46),
    "Enter": ("Enter", 13),
    "Escape": ("Escape", 27),
    "Tab": ("Tab", 9),
}


class RemoteLoginError(RuntimeError):
    """Base exception for remote login-code capture."""


class RemoteLoginUnavailable(RemoteLoginError):
    """The remote login service cannot be started or exposed."""


class RemoteLoginTimeout(RemoteLoginError):
    """The remote login session expired before a code was captured."""


class RemoteLoginFailed(RemoteLoginError):
    """The remote browser failed during login."""


class RemoteLoginCancelled(RemoteLoginError):
    """The user explicitly cancelled the remote login session."""


def _extract_pixiv_code(url: str) -> str | None:
    parsed = urlparse(url)
    is_https_callback = (
        parsed.scheme == "https"
        and parsed.hostname == _PIXIV_LOGIN_HOST
        and parsed.path == _PIXIV_CALLBACK_PATH
    )
    is_app_callback = (
        parsed.scheme == "pixiv"
        and parsed.hostname == "account"
        and parsed.path == "/login"
    )
    if not is_https_callback and not is_app_callback:
        return None

    values = parse_qs(parsed.query).get("code")
    if not values:
        return None

    code = values[0].strip()
    return code or None


def _validate_login_url(login_url: str) -> str:
    parsed = urlparse(login_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _PIXIV_LOGIN_HOST
        or parsed.path != _PIXIV_LOGIN_PATH
    ):
        raise RemoteLoginUnavailable("只允许打开 Pixiv 官方登录地址")
    return login_url


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_public_base_url(
        public_base_url: str,
        *,
        allow_insecure_http: bool = False,
) -> str:
    normalized = public_base_url.strip().rstrip("/")
    parsed = urlparse(normalized)
    if not normalized or not parsed.hostname or parsed.username or parsed.password:
        raise RemoteLoginUnavailable("远程登录公网地址无效")
    if parsed.query or parsed.fragment:
        raise RemoteLoginUnavailable("远程登录公网地址不能包含查询参数或片段")
    if parsed.scheme != "https":
        if (
            parsed.scheme != "http"
            or (
                not allow_insecure_http
                and not _is_loopback_host(parsed.hostname)
            )
        ):
            raise RemoteLoginUnavailable("远程登录公网地址必须使用 HTTPS")
    return normalized


def _route_prefix(public_base_url: str) -> str:
    path = urlparse(public_base_url).path.rstrip("/")
    return path if path else ""


def _expected_origin(public_base_url: str) -> str:
    parsed = urlparse(public_base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


class _RemoteLoginSession:
    def __init__(
            self,
            login_url: str,
            public_base_url: str,
            bind_host: str,
            bind_port: int,
            proxy: str | None,
            locale: str,
            tls_certfile: str | None,
            tls_keyfile: str | None,
            allow_insecure_public_url: bool = False,
    ) -> None:
        self.login_url = _validate_login_url(login_url)
        self.public_base_url = _validate_public_base_url(
            public_base_url,
            allow_insecure_http=allow_insecure_public_url,
        )
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.proxy = proxy
        self.locale = locale.strip() or "zh-HK"
        self.ssl_context = self._create_ssl_context(
            tls_certfile,
            tls_keyfile,
        )
        self.token = secrets.token_urlsafe(32)
        self.origin = _expected_origin(self.public_base_url)
        prefix = _route_prefix(self.public_base_url)
        self.page_path = f"{prefix}/{self.token}"
        self.ws_path = f"{self.page_path}/ws"
        self.public_url = f"{self.public_base_url}/{self.token}"

        self.code: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self.frames: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)
        self.runner: web.AppRunner | None = None
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self.cdp: CDPSession | None = None
        self.websocket: web.WebSocketResponse | None = None
        self._control_lock = asyncio.Lock()
        self._page_lock = asyncio.Lock()
        self._tasks: set[asyncio.Future[Any]] = set()
        self._cdp_sessions: list[CDPSession] = []
        self._last_click_at = 0.0
        self._last_click_position: tuple[float, float] | None = None
        self._last_activation_at = 0.0
        self._last_activation_signature = ""
        self._last_enter_at = 0.0
        self._last_submission_at = 0.0
        self._claimed_client_id: str | None = None

    @staticmethod
    def _create_ssl_context(
            certfile: str | None,
            keyfile: str | None,
    ) -> ssl.SSLContext | None:
        certfile = certfile.strip() if certfile else ""
        keyfile = keyfile.strip() if keyfile else ""
        if bool(certfile) != bool(keyfile):
            raise RemoteLoginUnavailable("TLS 证书和私钥必须同时配置")
        if not certfile:
            return None

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        try:
            context.load_cert_chain(certfile, keyfile)
        except (OSError, ssl.SSLError) as exc:
            raise RemoteLoginUnavailable(f"无法加载远程登录 TLS 证书: {exc}") from exc
        return context

    async def start(self) -> None:
        await self._start_gateway()
        try:
            await self._start_browser()
        except BaseException:
            await self.close()
            raise

    async def _start_gateway(self) -> None:
        app = web.Application(client_max_size=64 * 1024)
        app.router.add_get(self.page_path, self._serve_page)
        app.router.add_get(self.ws_path, self._serve_websocket)
        self.runner = web.AppRunner(
            app,
            access_log=None,
            handle_signals=False,
        )
        try:
            await self.runner.setup()
            site = web.TCPSite(
                self.runner,
                self.bind_host,
                self.bind_port,
                ssl_context=self.ssl_context,
            )
            await site.start()
        except (OSError, RuntimeError) as exc:
            raise RemoteLoginUnavailable(
                f"无法启动远程登录服务: {exc}"
            ) from exc

    async def _start_browser(self) -> None:
        try:
            self.playwright = await async_playwright().start()
            launch_options: dict = {"headless": True}
            if self.proxy:
                launch_options["proxy"] = {"server": self.proxy}
            self.browser = await self.playwright.chromium.launch(**launch_options)
            self.context = await self.browser.new_context(
                viewport={
                    "width": _VIEWPORT_WIDTH,
                    "height": _VIEWPORT_HEIGHT,
                },
                screen={
                    "width": _VIEWPORT_WIDTH,
                    "height": _VIEWPORT_HEIGHT,
                },
                is_mobile=True,
                has_touch=True,
                device_scale_factor=1,
                locale=self.locale,
                extra_http_headers={
                    "Accept-Language": (
                        f"{self.locale},zh-TW;q=0.9,zh-CN;q=0.8,"
                        "zh;q=0.7,en;q=0.5"
                    ),
                },
            )
            self.page = await self.context.new_page()
            self.context.on(
                "request",
                lambda request: self._inspect_url(request.url),
            )
            await self._activate_page(self.page)
            self.context.on(
                "page",
                lambda page: self._create_task(self._activate_page(page)),
            )
            await self.page.goto(
                self.login_url,
                wait_until="domcontentloaded",
                timeout=60_000,
            )
        except PlaywrightError as exc:
            raise RemoteLoginUnavailable(
                f"无法启动 Pixiv 登录浏览器: {exc}"
            ) from exc

    async def _activate_page(self, page: Page) -> None:
        async with self._page_lock:
            if page.is_closed() or self.context is None:
                return
            if self.cdp is not None:
                with contextlib.suppress(PlaywrightError):
                    await self.cdp.send("Page.stopScreencast")

            cdp = await self.context.new_cdp_session(page)
            self._cdp_sessions.append(cdp)
            self.page = page
            self.cdp = cdp
            await cdp.send("Network.enable")
            await cdp.send("Page.enable")
            cdp.on(
                "Network.requestWillBeSent",
                lambda event: self._inspect_url(
                    event.get("request", {}).get("url", "")
                ),
            )
            cdp.on("Page.screencastFrame", self._schedule_frame)
            page.on(
                "close",
                lambda _: self._create_task(self._restore_visible_page(page)),
            )
            await cdp.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": 65,
                    "maxWidth": _VIEWPORT_WIDTH,
                    "maxHeight": _VIEWPORT_HEIGHT,
                    "everyNthFrame": 1,
                },
            )
            await page.bring_to_front()

    async def _restore_visible_page(self, closed_page: Page) -> None:
        await asyncio.sleep(0)
        if self.page is not closed_page or self.context is None:
            return
        for page in reversed(self.context.pages):
            if not page.is_closed():
                with contextlib.suppress(PlaywrightError):
                    await self._activate_page(page)
                return

    def _inspect_url(self, url: str) -> None:
        code = _extract_pixiv_code(url)
        if code is not None and not self.code.done():
            self.code.set_result(code)
            if self.websocket is not None and not self.websocket.closed:
                self._create_task(
                    self.websocket.send_json({"type": "complete"})
                )

    def _schedule_frame(self, event: dict) -> None:
        self._create_task(self._accept_frame(event))

    def _create_task(self, awaitable: Awaitable) -> None:
        task: asyncio.Future[Any] = asyncio.ensure_future(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)

    def _task_finished(self, task: asyncio.Future[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        with contextlib.suppress(Exception):
            task.exception()

    async def _accept_frame(self, event: dict) -> None:
        if self.cdp is None:
            return
        session_id = event.get("sessionId")
        if session_id is not None:
            with contextlib.suppress(PlaywrightError):
                await self.cdp.send(
                    "Page.screencastFrameAck",
                    {"sessionId": session_id},
                )

        frame = {
            "type": "frame",
            "data": event.get("data", ""),
            "width": _VIEWPORT_WIDTH,
            "height": _VIEWPORT_HEIGHT,
        }
        if self.frames.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.frames.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self.frames.put_nowait(frame)

    async def _serve_page(self, request: web.Request) -> web.Response:
        client_id = request.cookies.get(_CLIENT_COOKIE, "")
        if not 20 <= len(client_id) <= 128:
            client_id = secrets.token_urlsafe(24)
        response = web.Response(
            text=REMOTE_LOGIN_PAGE,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; img-src data:; "
                    "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                    "connect-src 'self' ws: wss:; frame-ancestors 'none'"
                ),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
            },
        )
        response.set_cookie(
            _CLIENT_COOKIE,
            client_id,
            httponly=True,
            secure=urlparse(self.public_base_url).scheme == "https",
            samesite="Strict",
            path=self.page_path,
            max_age=1800,
        )
        return response

    def _claim_client(self, client_id: str) -> bool:
        if not client_id:
            return False
        if self._claimed_client_id is None:
            self._claimed_client_id = client_id
        return secrets.compare_digest(self._claimed_client_id, client_id)

    async def _serve_websocket(
            self,
            request: web.Request,
    ) -> web.StreamResponse:
        if request.headers.get("Origin") != self.origin:
            raise web.HTTPForbidden(text="Invalid Origin")

        async with self._control_lock:
            client_id = request.cookies.get(_CLIENT_COOKIE, "")
            if not self._claim_client(client_id):
                websocket = web.WebSocketResponse()
                await websocket.prepare(request)
                await websocket.close(
                    code=4003,
                    message=b"Login session claimed by another administrator",
                )
                return websocket

            previous_websocket = self.websocket
            if previous_websocket is not None and not previous_websocket.closed:
                await previous_websocket.close(
                    code=4000,
                    message=b"Reconnected from the claimed browser",
                )
            websocket = web.WebSocketResponse(
                heartbeat=20,
                max_msg_size=16 * 1024,
                compress=False,
            )
            await websocket.prepare(request)
            self.websocket = websocket

        frame_sender = asyncio.create_task(self._send_frames(websocket))
        try:
            async for message in websocket:
                if message.type == WSMsgType.TEXT:
                    await self._handle_input(message.data)
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            frame_sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await frame_sender
            if self.websocket is websocket:
                self.websocket = None
        return websocket

    async def _send_frames(self, websocket: web.WebSocketResponse) -> None:
        while not websocket.closed:
            frame = await self.frames.get()
            await websocket.send_json(frame)

    async def _handle_input(self, raw_message: str) -> None:
        if self.cdp is None:
            return
        try:
            message = json.loads(raw_message)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(message, dict):
            return

        message_type = message.get("type")
        if message_type == "cancel":
            if not self.code.done():
                self.code.set_exception(
                    RemoteLoginCancelled("用户取消了 Pixiv 登录")
                )
            return
        if message_type == "text":
            text = message.get("text")
            if isinstance(text, str) and 0 < len(text) <= 4096:
                await self.cdp.send("Input.insertText", {"text": text})
            return
        if message_type == "key":
            key = message.get("key")
            if isinstance(key, str) and key in _ALLOWED_KEYS:
                if key == "Enter":
                    now = asyncio.get_running_loop().time()
                    if (
                        now - self._last_enter_at < _ENTER_KEY_COOLDOWN
                        or now - self._last_submission_at < _SUBMISSION_COOLDOWN
                    ):
                        return
                    self._last_enter_at = now
                    self._last_submission_at = now
                code, virtual_key = _KEY_CODES[key]
                await self.cdp.send(
                    "Input.dispatchKeyEvent",
                    {
                        "type": "keyDown",
                        "key": key,
                        "code": code,
                        "windowsVirtualKeyCode": virtual_key,
                        "nativeVirtualKeyCode": virtual_key,
                    },
                )
                await self.cdp.send(
                    "Input.dispatchKeyEvent",
                    {
                        "type": "keyUp",
                        "key": key,
                        "code": code,
                        "windowsVirtualKeyCode": virtual_key,
                        "nativeVirtualKeyCode": virtual_key,
                    },
                )
            return

        coordinates = self._coordinates(message)
        if coordinates is None:
            return
        x, y = coordinates
        if message_type == "click":
            target = await self._target_at_point(x, y)
            if self._should_suppress_click(target, x, y):
                return
            await self.cdp.send(
                "Input.dispatchMouseEvent",
                {
                    "type": "mousePressed",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            await self.cdp.send(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseReleased",
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                },
            )
            await self._update_keyboard_state(target)
        elif message_type == "scroll":
            delta_x = self._bounded_number(message.get("deltaX"), 2000)
            delta_y = self._bounded_number(message.get("deltaY"), 2000)
            if delta_x is not None and delta_y is not None:
                await self.cdp.send(
                    "Input.dispatchMouseEvent",
                    {
                        "type": "mouseWheel",
                        "x": x,
                        "y": y,
                        "deltaX": delta_x,
                        "deltaY": delta_y,
                    },
                )

    async def _target_at_point(self, x: float, y: float) -> dict[str, Any]:
        if self.page is None:
            return {"editable": False, "activation": False, "signature": ""}
        try:
            target = await self.page.evaluate(
                """([x, y]) => {
                    const element = document.elementFromPoint(x, y);
                    if (!element) {
                        return { editable: false, activation: false, signature: "" };
                    }
                    const editableElement = element.closest(
                        "input, textarea, [contenteditable='true']"
                    );
                    let editable = false;
                    let inputMode = "text";
                    if (editableElement) {
                        const tag = editableElement.tagName.toLowerCase();
                        const type = (editableElement.type || "text").toLowerCase();
                        editable = editableElement.isContentEditable
                            || tag === "textarea"
                            || (tag === "input" && ![
                                "button", "checkbox", "color", "file", "hidden",
                                "image", "radio", "range", "reset", "submit"
                            ].includes(type));
                        inputMode = editableElement.inputMode || "text";
                        if (type === "email") inputMode = "email";
                        if (type === "number") inputMode = "numeric";
                        if (type === "tel") inputMode = "tel";
                        if (type === "url") inputMode = "url";
                    }
                    const activationElement = element.closest(
                        "button, a, input[type='button'], input[type='image'],"
                        + "input[type='reset'], input[type='submit'],"
                        + "[role='button']"
                    );
                    const signature = activationElement
                        ? (() => {
                            const rect = activationElement.getBoundingClientRect();
                            return [
                                activationElement.tagName,
                                activationElement.id,
                                activationElement.getAttribute("name"),
                                activationElement.getAttribute("type"),
                                (activationElement.textContent || "").trim().slice(0, 80),
                                Math.round(rect.left),
                                Math.round(rect.top),
                                Math.round(rect.width),
                                Math.round(rect.height)
                            ].join("|");
                        })()
                        : "";
                    const activationTag = activationElement
                        ? activationElement.tagName.toLowerCase()
                        : "";
                    const activationType = activationElement
                        ? (activationElement.type || "").toLowerCase()
                        : "";
                    const submission = Boolean(activationElement) && (
                        (activationTag === "button"
                            && activationType !== "button"
                            && activationType !== "reset")
                        || (activationTag === "input"
                            && ["image", "submit"].includes(activationType))
                    );
                    return {
                        editable,
                        inputMode,
                        activation: Boolean(activationElement),
                        submission,
                        signature
                    };
                }""",
                [x, y],
            )
        except PlaywrightError:
            return {"editable": False, "activation": False, "signature": ""}
        return target if isinstance(target, dict) else {
            "editable": False,
            "activation": False,
            "signature": "",
        }

    def _should_suppress_click(
            self,
            target: dict[str, Any],
            x: float,
            y: float,
    ) -> bool:
        now = asyncio.get_running_loop().time()
        if self._last_click_position is not None:
            last_x, last_y = self._last_click_position
            near_previous = abs(x - last_x) <= 12 and abs(y - last_y) <= 12
            if near_previous and now - self._last_click_at < _DUPLICATE_CLICK_WINDOW:
                return True

        signature = str(target.get("signature") or "")
        if (
            target.get("submission")
            and now - self._last_submission_at < _SUBMISSION_COOLDOWN
        ):
            return True
        if (
            target.get("activation")
            and signature
            and signature == self._last_activation_signature
            and now - self._last_activation_at < _ACTIVATION_CLICK_COOLDOWN
        ):
            return True

        self._last_click_at = now
        self._last_click_position = (x, y)
        if target.get("activation") and signature:
            self._last_activation_at = now
            self._last_activation_signature = signature
        if target.get("submission"):
            self._last_submission_at = now
        return False

    async def _update_keyboard_state(
            self,
            target: dict[str, Any],
    ) -> None:
        if self.page is None or self.websocket is None or self.websocket.closed:
            return
        show = bool(target.get("editable"))
        if not show:
            with contextlib.suppress(PlaywrightError):
                await self.page.evaluate(
                    "() => document.activeElement?.blur?.()"
                )
        await self.websocket.send_json({
            "type": "keyboard",
            "show": show,
            "inputMode": str(target.get("inputMode") or "text"),
        })

    @staticmethod
    def _bounded_number(value: object, maximum: float) -> float | None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return max(-maximum, min(maximum, float(value)))

    @classmethod
    def _coordinates(cls, message: dict) -> tuple[float, float] | None:
        x = cls._bounded_number(message.get("x"), _VIEWPORT_WIDTH)
        y = cls._bounded_number(message.get("y"), _VIEWPORT_HEIGHT)
        if x is None or y is None:
            return None
        return max(0, x), max(0, y)

    async def close(self) -> None:
        if self.websocket is not None and not self.websocket.closed:
            with contextlib.suppress(Exception):
                await self.websocket.close()
        if self.cdp is not None:
            with contextlib.suppress(PlaywrightError):
                await self.cdp.send("Page.stopScreencast")
        if self.context is not None:
            with contextlib.suppress(PlaywrightError):
                await self.context.close()
        if self.browser is not None:
            with contextlib.suppress(PlaywrightError):
                await self.browser.close()
        if self.playwright is not None:
            with contextlib.suppress(PlaywrightError):
                await self.playwright.stop()
        if self.runner is not None:
            with contextlib.suppress(Exception):
                await self.runner.cleanup()

        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


async def capture_pixiv_code(
        login_url: str,
        *,
        public_base_url: str,
        bind_host: str = "127.0.0.1",
        bind_port: int = 8765,
        notify_login_url: Callable[[str], Awaitable[None]],
        timeout: float = 900,
        proxy: str | None = None,
        locale: str = "zh-HK",
        tls_certfile: str | None = None,
        tls_keyfile: str | None = None,
        allow_insecure_public_url: bool = False,
) -> str:
    """Open a remotely controlled Pixiv login page and return its OAuth code."""
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if not 0 <= bind_port <= 65535:
        raise ValueError("bind_port 必须在 0 到 65535 之间")

    session = _RemoteLoginSession(
        login_url=login_url,
        public_base_url=public_base_url,
        bind_host=bind_host,
        bind_port=bind_port,
        proxy=proxy,
        locale=locale,
        tls_certfile=tls_certfile,
        tls_keyfile=tls_keyfile,
        allow_insecure_public_url=allow_insecure_public_url,
    )
    try:
        await session.start()
        try:
            await notify_login_url(session.public_url)
        except Exception as exc:
            raise RemoteLoginUnavailable(
                f"无法发送远程登录链接: {exc}"
            ) from exc
        try:
            return await asyncio.wait_for(session.code, timeout=timeout)
        except TimeoutError as exc:
            raise RemoteLoginTimeout("等待 Pixiv 登录超时") from exc
    except RemoteLoginError:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise RemoteLoginFailed(f"远程 Pixiv 登录失败: {exc}") from exc
    finally:
        await session.close()
