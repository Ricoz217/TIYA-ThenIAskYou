from __future__ import annotations

import asyncio
import hashlib

import httpx
import pytest

from TIYA.agent_tools.websearch.tools import (
    Capability,
    ContentFormat,
    DownloadRequest,
    FetchRequest,
    HTTPXProvider,
    InteractionKind,
    MapRequest,
    ProviderRequestError,
    WebSearchRuntime,
)


def run(coroutine):
    return asyncio.run(coroutine)


class DisconnectingStream(httpx.AsyncByteStream):
    def __init__(self, request: httpx.Request, prefix: bytes) -> None:
        self._request = request
        self._prefix = prefix

    async def __aiter__(self):
        yield self._prefix
        raise httpx.ReadError("connection lost", request=self._request)


class PausingStream(httpx.AsyncByteStream):
    def __init__(self, started: asyncio.Event, prefix: bytes) -> None:
        self._started = started
        self._prefix = prefix
        self._release = asyncio.Event()

    async def __aiter__(self):
        yield self._prefix
        self._started.set()
        await self._release.wait()


def test_profile_declares_lightweight_http_capabilities() -> None:
    assert HTTPXProvider.profile.capabilities == frozenset(
        {Capability.FETCH, Capability.MAP, Capability.DOWNLOAD}
    )


def test_fetch_extracts_a_normalized_document_without_a_browser(tmp_path) -> None:
    html = """
    <html>
      <head><title>Example article</title></head>
      <body>
        <nav><a href="/menu">Menu</a></nav>
        <article>
          <h1>A useful heading</h1>
          <p>Readable body with <a href="/docs">the docs</a>.</p>
          <img src="/hero.png" alt="Hero image">
        </article>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=html,
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = HTTPXProvider(tmp_path, http_client=client)
            return await provider.fetch(
                FetchRequest(
                    "https://example.com/article",
                    preferred_format=ContentFormat.MARKDOWN,
                )
            )

    document = run(scenario())

    assert document.provider == "httpx"
    assert document.title == "Example article"
    assert document.content_format is ContentFormat.MARKDOWN
    assert "# A useful heading" in document.content
    assert "[the docs](https://example.com/docs)" in document.content
    assert tuple(link.url for link in document.links) == (
        "https://example.com/docs",
    )
    assert document.media[0].url == "https://example.com/hero.png"
    assert document.media[0].alt == "Hero image"
    assert document.metadata["rendering_recommended"] is False


def test_fetch_marks_login_pages_for_agent_escalation(tmp_path) -> None:
    html = """
    <html><head><title>Sign in</title></head><body>
      <form><input name="email"><input type="password"></form>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).fetch(
                FetchRequest("https://example.com/login")
            )

    document = run(scenario())

    assert document.interaction_required is not None
    assert document.interaction_required.kind is InteractionKind.LOGIN
    assert document.interaction_required.provider == "httpx"


def test_fetch_ignores_incidental_challenge_and_mfa_terms(tmp_path) -> None:
    html = """
    <html><head><title>Security integration guide</title></head><body>
      <article>
        <h1>Authentication SDK documentation</h1>
        <p>This article explains optional CAPTCHA and MFA integrations.</p>
        <p>It is ordinary readable documentation and does not gate access.</p>
        <script>window.features = {captcha: false, mfa: false};</script>
      </article>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).fetch(
                FetchRequest("https://example.com/security-guide")
            )

    assert run(scenario()).interaction_required is None


def test_fetch_ignores_a_login_widget_outside_readable_article(tmp_path) -> None:
    html = """
    <html><head><title>Public article</title></head><body>
      <header>
        <form action="/login"><input type="password" name="password"></form>
      </header>
      <article>
        <h1>Public research</h1>
        <p>This article is already readable without authentication.</p>
      </article>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).fetch(
                FetchRequest("https://example.com/research")
            )

    assert run(scenario()).interaction_required is None


def test_fetch_detects_structural_captcha_evidence(tmp_path) -> None:
    html = """
    <html><head><title>Security check</title></head><body>
      <main>
        <h1>Verify you are human</h1>
        <div class="cf-turnstile" data-sitekey="public-site-key"></div>
      </main>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).fetch(
                FetchRequest("https://example.com/challenge")
            )

    interaction = run(scenario()).interaction_required

    assert interaction is not None
    assert interaction.kind is InteractionKind.CAPTCHA


def test_fetch_returns_media_semantic_context_and_discovery_source(
    tmp_path,
) -> None:
    html = """
    <html>
      <head>
        <meta property="og:image" content="/social-card.jpg">
      </head>
      <body>
        <main>
          <figure>
            <a href="/original.png" title="Open original">
              <img src="/thumb.png" alt="Kokona default character art"
                   title="Default outfit" width="1080" height="1920">
            </a>
            <figcaption>Official default character illustration</figcaption>
          </figure>
          <div aria-label="Character hero"
               style="background-image: url('/hero.webp')"></div>
        </main>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).fetch(
                FetchRequest("https://example.com/character")
            )

    document = run(scenario())
    media = {resource.url: resource for resource in document.media}

    thumbnail = media["https://example.com/thumb.png"]
    assert thumbnail.metadata == {
        "source": "element",
        "tag": "img",
        "source_attribute": "src",
        "title": "Default outfit",
        "context_text": "Official default character illustration",
        "link_url": "https://example.com/original.png",
        "width": 1080,
        "height": 1920,
    }
    assert media["https://example.com/hero.webp"].metadata["source"] == (
        "css-background"
    )
    assert media["https://example.com/hero.webp"].alt == "Character hero"
    assert media["https://example.com/social-card.jpg"].metadata["source"] == (
        "open-graph"
    )


def test_fetch_recommends_rendering_for_a_javascript_shell(tmp_path) -> None:
    html = """
    <html><head><title>Client app</title></head><body>
      <div id="root"></div>
      <script src="/runtime.js"></script>
      <script src="/vendor.js"></script>
      <script src="/app.js"></script>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"Content-Type": "text/html"})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).fetch(
                FetchRequest("https://example.com/app")
            )

    assert run(scenario()).metadata["rendering_recommended"] is True


def test_fetch_rejects_binary_content_and_javascript_rendering(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"\x89PNG\r\n\x1a\n",
            headers={"Content-Type": "image/png"},
        )

    async def fetch_binary() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            await HTTPXProvider(tmp_path, http_client=client).fetch(
                FetchRequest("https://example.com/image.png")
            )

    with pytest.raises(ProviderRequestError, match="download"):
        run(fetch_binary())

    provider = HTTPXProvider(tmp_path)
    try:
        with pytest.raises(ProviderRequestError, match="JavaScript"):
            run(
                provider.fetch(
                    FetchRequest(
                        "https://example.com/app", render_javascript=True
                    )
                )
            )
    finally:
        run(provider.aclose())


def test_fetch_enforces_document_size_before_parsing(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="a" * 64)

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            await HTTPXProvider(
                tmp_path,
                http_client=client,
                max_document_bytes=16,
            ).fetch(FetchRequest("https://example.com/large"))

    with pytest.raises(ProviderRequestError, match="exceeds"):
        run(scenario())


def test_map_crawls_same_origin_to_the_requested_depth(tmp_path) -> None:
    pages = {
        "/": """
            <a href="/alpha">Alpha</a>
            <a href="/beta#part">Beta</a>
            <a href="https://elsewhere.example/out">Elsewhere</a>
        """,
        "/alpha": '<a href="/deep">Deep reference</a>',
        "/beta": "<p>Beta page</p>",
        "/deep": "<p>Deep page</p>",
    }
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(
            200,
            text=pages[request.url.path],
            headers={"Content-Type": "text/html"},
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = HTTPXProvider(tmp_path, http_client=client)
            shallow = await provider.map(
                MapRequest("https://example.com/", max_depth=1)
            )
            deep = await provider.map(
                MapRequest(
                    "https://example.com/",
                    max_depth=2,
                    query="deep",
                )
            )
            return shallow, deep

    shallow, deep = run(scenario())

    assert tuple(link.url for link in shallow.links) == (
        "https://example.com/alpha",
        "https://example.com/beta",
    )
    assert tuple(link.url for link in deep.links) == (
        "https://example.com/deep",
    )
    assert "https://elsewhere.example/out" not in {
        link.url for link in deep.links
    }
    assert requested.count("/alpha") == 1
    assert "/deep" not in requested
    assert deep.metadata["pages_fetched"] == 3


def test_map_uses_a_redirected_root_as_the_canonical_origin(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(
                301, headers={"Location": "https://www.example.com/"}
            )
        return httpx.Response(
            200,
            text='<a href="/article">Article</a>',
            headers={"Content-Type": "text/html"},
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).map(
                MapRequest("https://example.com/", max_depth=1)
            )

    response = run(scenario())

    assert tuple(link.url for link in response.links) == (
        "https://www.example.com/article",
    )
    assert response.links[0].internal is True


def test_fetch_rejects_obvious_private_network_targets(tmp_path) -> None:
    provider = HTTPXProvider(tmp_path)
    try:
        with pytest.raises(ProviderRequestError, match="private network"):
            run(provider.fetch(FetchRequest("http://127.0.0.1/admin")))
    finally:
        run(provider.aclose())


def test_download_streams_a_bounded_file_and_returns_verified_metadata(
    tmp_path,
) -> None:
    payload = b"\x89PNG\r\n\x1a\nimage-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/original":
            return httpx.Response(302, headers={"Location": "/asset.png"})
        return httpx.Response(
            200,
            headers={
                "Content-Type": "image/png; charset=binary",
                "Content-Length": str(len(payload)),
            },
            content=payload,
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = HTTPXProvider(tmp_path, http_client=client)
            return await provider.download(
                DownloadRequest(
                    "https://example.com/original", filename="picture.png"
                )
            )

    artifact = run(scenario())

    assert artifact.provider == "httpx"
    assert artifact.source_url == "https://example.com/original"
    assert artifact.final_url == "https://example.com/asset.png"
    assert artifact.path == tmp_path / "picture.png"
    assert artifact.path.read_bytes() == payload
    assert artifact.size == len(payload)
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.content_type == "image/png"
    assert artifact.metadata["status_code"] == 200


def test_download_derives_a_safe_unique_filename(tmp_path) -> None:
    (tmp_path / "report.pdf").write_bytes(b"existing")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"new")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).download(
                DownloadRequest("https://example.com/files/report.pdf")
            )

    artifact = run(scenario())

    assert artifact.path.name == "report (1).pdf"
    assert (tmp_path / "report.pdf").read_bytes() == b"existing"
    assert artifact.path.read_bytes() == b"new"


@pytest.mark.parametrize("filename", ["../secret.txt", "sub/file.txt", "bad:name"])
def test_download_rejects_unsafe_explicit_filenames(tmp_path, filename: str) -> None:
    provider = HTTPXProvider(tmp_path, http_client=httpx.AsyncClient())
    try:
        with pytest.raises(ProviderRequestError, match="filename"):
            run(
                provider.download(
                    DownloadRequest("https://example.com/file", filename=filename)
                )
            )
    finally:
        run(provider.aclose())


def test_download_rejects_content_length_over_limit_without_creating_a_file(
    tmp_path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "100"},
            content=b"small test body",
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            await HTTPXProvider(tmp_path, http_client=client).download(
                DownloadRequest("https://example.com/file.bin", max_bytes=10)
            )

    with pytest.raises(ProviderRequestError, match="exceeds"):
        run(scenario())
    assert list(tmp_path.iterdir()) == []


def test_download_enforces_streamed_size_and_removes_partial_file(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"0123456789abcdef")

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            await HTTPXProvider(tmp_path, http_client=client).download(
                DownloadRequest("https://example.com/file.bin", max_bytes=8)
            )

    with pytest.raises(ProviderRequestError, match="exceeds"):
        run(scenario())
    assert list(tmp_path.iterdir()) == []


def test_download_reconnects_and_resumes_the_current_transfer(tmp_path) -> None:
    payload = b"0123456789abcdef"
    ranges: list[str | None] = []
    transfer_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transfer_attempts
        requested_range = request.headers.get("Range")
        ranges.append(requested_range)
        if requested_range == "bytes=0-0":
            return httpx.Response(
                206,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                    "Content-Length": "1",
                    "ETag": '"stable-v1"',
                },
                content=payload[:1],
            )
        transfer_attempts += 1
        start = int(requested_range.removeprefix("bytes=").split("-", 1)[0])
        body = payload[start:]
        headers = {
            "Content-Range": f"bytes {start}-{len(payload) - 1}/{len(payload)}",
            "Content-Length": str(len(body)),
            "ETag": '"stable-v1"',
        }
        if transfer_attempts == 1:
            return httpx.Response(
                206,
                headers=headers,
                stream=DisconnectingStream(request, body[:5]),
            )
        return httpx.Response(206, headers=headers, content=body)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(
                tmp_path,
                http_client=client,
                download_retry_backoff=0,
            ).download(
                DownloadRequest(
                    "https://example.com/large.bin",
                    filename="large.bin",
                    max_parallel_parts=1,
                    max_retries=2,
                )
            )

    artifact = run(scenario())

    assert artifact.path.read_bytes() == payload
    assert ranges == ["bytes=0-0", "bytes=0-15", "bytes=5-15"]
    assert artifact.metadata["resumed"] is True
    assert artifact.metadata["retry_count"] == 1
    assert not tuple(tmp_path.glob("*.tiya-*"))


def test_download_retries_a_transient_range_probe(tmp_path) -> None:
    payload = b"probe retry"
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            raise httpx.ConnectError("temporary connect failure", request=request)
        requested_range = request.headers.get("Range")
        if requested_range == "bytes=0-0":
            return httpx.Response(
                206,
                headers={
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                    "Content-Length": "1",
                    "ETag": '"probe-v1"',
                },
                content=payload[:1],
            )
        return httpx.Response(
            206,
            headers={
                "Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(payload)),
                "ETag": '"probe-v1"',
            },
            content=payload,
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(
                tmp_path,
                http_client=client,
                download_retry_backoff=0,
            ).download(
                DownloadRequest(
                    "https://example.com/probe.bin",
                    max_retries=1,
                    max_parallel_parts=1,
                )
            )

    artifact = run(scenario())

    assert artifact.path.read_bytes() == payload
    assert requests == 3


def test_download_preserves_partial_state_and_resumes_on_a_later_call(
    tmp_path,
) -> None:
    payload = b"abcdefghijklmnopqrstuvwxyz"

    def first_handler(request: httpx.Request) -> httpx.Response:
        requested_range = request.headers.get("Range")
        if requested_range == "bytes=0-0":
            return httpx.Response(
                206,
                headers={
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                    "Content-Length": "1",
                    "ETag": '"stable-v1"',
                },
                content=payload[:1],
            )
        return httpx.Response(
            206,
            headers={
                "Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(payload)),
                "ETag": '"stable-v1"',
            },
            stream=DisconnectingStream(request, payload[:7]),
        )

    async def interrupted() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(first_handler)
        ) as client:
            await HTTPXProvider(
                tmp_path,
                http_client=client,
                download_retry_backoff=0,
            ).download(
                DownloadRequest(
                    "https://example.com/archive.bin",
                    filename="archive.bin",
                    max_parallel_parts=1,
                    max_retries=0,
                )
            )

    with pytest.raises(ProviderRequestError, match="connection lost"):
        run(interrupted())
    assert tuple(tmp_path.glob("*.tiya-*"))

    resumed_ranges: list[str | None] = []

    def second_handler(request: httpx.Request) -> httpx.Response:
        requested_range = request.headers.get("Range")
        resumed_ranges.append(requested_range)
        if requested_range == "bytes=0-0":
            return httpx.Response(
                206,
                headers={
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                    "Content-Length": "1",
                    "ETag": '"stable-v1"',
                },
                content=payload[:1],
            )
        start = int(requested_range.removeprefix("bytes=").split("-", 1)[0])
        body = payload[start:]
        return httpx.Response(
            206,
            headers={
                "Content-Range": f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(body)),
                "ETag": '"stable-v1"',
            },
            content=body,
        )

    async def resume():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(second_handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).download(
                DownloadRequest(
                    "https://example.com/archive.bin",
                    filename="archive.bin",
                    max_parallel_parts=1,
                )
            )

    artifact = run(resume())

    assert artifact.path.read_bytes() == payload
    assert resumed_ranges == ["bytes=0-0", "bytes=7-25"]
    assert artifact.metadata["resumed"] is True
    assert not tuple(tmp_path.glob("*.tiya-*"))


def test_download_discards_interrupted_state_without_a_validator(
    tmp_path,
) -> None:
    payload = b"abcdefghijklmnopqrstuvwxyz"

    async def interrupted() -> None:
        started = asyncio.Event()

        def first_handler(request: httpx.Request) -> httpx.Response:
            requested_range = request.headers.get("Range")
            if requested_range == "bytes=0-0":
                return httpx.Response(
                    206,
                    headers={
                        "Content-Range": f"bytes 0-0/{len(payload)}",
                        "Content-Length": "1",
                    },
                    content=payload[:1],
                )
            return httpx.Response(
                206,
                headers={
                    "Content-Range": (
                        f"bytes 0-{len(payload) - 1}/{len(payload)}"
                    ),
                    "Content-Length": str(len(payload)),
                },
                stream=PausingStream(started, payload[:7]),
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(first_handler)
        ) as client:
            task = asyncio.create_task(
                HTTPXProvider(tmp_path, http_client=client).download(
                    DownloadRequest(
                        "https://example.com/unvalidated.bin",
                        filename="unvalidated.bin",
                        max_parallel_parts=1,
                    )
                )
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    run(interrupted())
    assert tuple(tmp_path.glob("*.tiya-*"))

    second_ranges: list[str | None] = []

    def second_handler(request: httpx.Request) -> httpx.Response:
        requested_range = request.headers.get("Range")
        second_ranges.append(requested_range)
        if requested_range == "bytes=0-0":
            return httpx.Response(
                206,
                headers={
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                    "Content-Length": "1",
                },
                content=payload[:1],
            )
        start = int(requested_range.removeprefix("bytes=").split("-", 1)[0])
        body = payload[start:]
        return httpx.Response(
            206,
            headers={
                "Content-Range": f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                "Content-Length": str(len(body)),
            },
            content=body,
        )

    async def restart():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(second_handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).download(
                DownloadRequest(
                    "https://example.com/unvalidated.bin",
                    filename="unvalidated.bin",
                    max_parallel_parts=1,
                )
            )

    artifact = run(restart())

    assert artifact.path.read_bytes() == payload
    assert second_ranges == ["bytes=0-0", "bytes=0-25"]
    assert artifact.metadata["resumed"] is False
    assert not tuple(tmp_path.glob("*.tiya-*"))


def test_download_uses_concurrent_ranges_when_the_server_supports_them(
    tmp_path,
) -> None:
    payload = b"0123456789abcdefghijklmnopqrstuv"
    ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_range = request.headers.get("Range")
        ranges.append(requested_range)
        if requested_range == "bytes=0-0":
            return httpx.Response(
                206,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes 0-0/{len(payload)}",
                    "Content-Length": "1",
                    "ETag": '"parallel-v1"',
                },
                content=payload[:1],
            )
        start_text, end_text = requested_range.removeprefix("bytes=").split("-")
        start, end = int(start_text), int(end_text)
        body = payload[start : end + 1]
        return httpx.Response(
            206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                "Content-Length": str(len(body)),
                "ETag": '"parallel-v1"',
            },
            content=body,
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(
                tmp_path,
                http_client=client,
                download_part_size=8,
            ).download(
                DownloadRequest(
                    "https://example.com/parallel.bin",
                    filename="parallel.bin",
                    max_parallel_parts=4,
                )
            )

    artifact = run(scenario())

    assert artifact.path.read_bytes() == payload
    assert set(ranges[1:]) == {
        "bytes=0-7",
        "bytes=8-15",
        "bytes=16-23",
        "bytes=24-31",
    }
    assert artifact.metadata["range_supported"] is True
    assert artifact.metadata["parallel_parts"] == 4
    assert not tuple(tmp_path.glob("*.tiya-*"))


def test_download_falls_back_when_the_server_ignores_ranges(tmp_path) -> None:
    payload = b"ordinary response"
    ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ranges.append(request.headers.get("Range"))
        return httpx.Response(
            200,
            headers={"Content-Length": str(len(payload))},
            content=payload,
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).download(
                DownloadRequest("https://example.com/fallback.bin")
            )

    artifact = run(scenario())

    assert artifact.path.read_bytes() == payload
    assert ranges == ["bytes=0-0", None]
    assert artifact.metadata["range_supported"] is False
    assert artifact.metadata["parallel_parts"] == 1


def test_download_falls_back_when_the_server_rejects_the_range_probe(
    tmp_path,
) -> None:
    payload = b"full download still works"
    ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_range = request.headers.get("Range")
        ranges.append(requested_range)
        if requested_range is not None:
            return httpx.Response(405, text="ranges disabled")
        return httpx.Response(
            200,
            headers={"Content-Length": str(len(payload))},
            content=payload,
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await HTTPXProvider(tmp_path, http_client=client).download(
                DownloadRequest("https://example.com/fallback.bin")
            )

    artifact = run(scenario())

    assert artifact.path.read_bytes() == payload
    assert ranges == ["bytes=0-0", None]
    assert artifact.metadata["range_supported"] is False


def test_download_rejects_a_ranged_resource_over_the_size_limit(tmp_path) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            206,
            headers={
                "Content-Range": "bytes 0-0/100",
                "Content-Length": "1",
                "ETag": '"large-v1"',
            },
            content=b"0",
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            await HTTPXProvider(tmp_path, http_client=client).download(
                DownloadRequest(
                    "https://example.com/too-large.bin", max_bytes=99
                )
            )

    with pytest.raises(ProviderRequestError, match="exceeds"):
        run(scenario())
    assert requests == 1
    assert list(tmp_path.iterdir()) == []


def test_http_failures_are_mapped_to_stable_contract_errors(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="temporarily unavailable")

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            await HTTPXProvider(tmp_path, http_client=client).download(
                DownloadRequest("https://example.com/file.bin")
            )

    with pytest.raises(ProviderRequestError) as caught:
        run(scenario())

    assert caught.value.status_code == 503
    assert caught.value.retryable is True


def test_runtime_dispatches_httpx_download(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            runtime = WebSearchRuntime()
            runtime.register(HTTPXProvider(tmp_path, http_client=client))
            return await runtime.download(
                DownloadRequest("https://example.com/data.bin")
            )

    assert run(scenario()).path.read_bytes() == b"data"


def test_runtime_dispatches_httpx_fetch(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="plain response")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            runtime = WebSearchRuntime()
            runtime.register(HTTPXProvider(tmp_path, http_client=client))
            return await runtime.fetch(FetchRequest("https://example.com/data"))

    assert run(scenario()).content == "plain response"
