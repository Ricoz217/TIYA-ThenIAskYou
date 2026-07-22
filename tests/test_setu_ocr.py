import asyncio
from concurrent.futures import Executor, Future
from dataclasses import FrozenInstanceError
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from TIYA.setu.ocr import (
    HttpOcrProvider,
    RapidOcrProvider,
    SetuOcrError,
    SetuOcrLine,
    detect_advertisement,
)


class _FakeProvider:
    def __init__(
            self,
            lines: list[SetuOcrLine] | None = None,
            error: Exception | None = None,
    ) -> None:
        self.lines = lines or []
        self.error = error
        self.calls = 0

    async def recognize(self, image: bytes) -> list[SetuOcrLine]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.lines


def _static_image(
        size: tuple[int, int],
        image_format: str = "PNG",
) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format=image_format)
    return output.getvalue()


def _animated_image() -> bytes:
    frames = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
    )
    return output.getvalue()


def _box(
        left: float,
        top: float,
        right: float,
        bottom: float,
) -> tuple[tuple[float, float], ...]:
    return (
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    )


def _multipart_file_content(request: httpx.Request) -> bytes:
    _, content = request.content.split(b"\r\n\r\n", 1)
    return content.rsplit(b"\r\n--", 1)[0]


def test_setu_ocr_line_is_only_an_immutable_box_and_text() -> None:
    line = SetuOcrLine(box=_box(0, 0, 10, 10), text="patreon")

    assert line.box == _box(0, 0, 10, 10)
    assert line.text == "patreon"
    assert not hasattr(line, "score")
    with pytest.raises(FrozenInstanceError):
        line.text = "changed"  # type: ignore[misc]


def test_real_sample_signature_is_not_advertisement() -> None:
    line = SetuOcrLine(
        box=(
            (3.6989, 544.9063),
            (311.9375, 394.2409),
            (332.8977, 440.6960),
            (25.8920, 591.3614),
        ),
        text="PATREON/QIANDAIYIYU",
    )
    provider = _FakeProvider([line])

    result = asyncio.run(
        detect_advertisement(_static_image((1736, 2491)), provider)
    )

    assert result is False


def test_real_sample_large_edge_promotion_is_advertisement() -> None:
    line = SetuOcrLine(
        box=((0, 249), (333, 82), (354, 125), (15, 292)),
        text="PATREON/QIANDAIYIYU",
    )
    provider = _FakeProvider([line])

    result = asyncio.run(
        detect_advertisement(_static_image((1490, 1056)), provider)
    )

    assert result is True


def test_keyword_box_inside_edge_safe_area_is_ignored() -> None:
    provider = _FakeProvider([
        SetuOcrLine(
            box=_box(0, 950, 1000, 1000),
            text="patreon.com/artist",
        )
    ])

    result = asyncio.run(
        detect_advertisement(_static_image((1000, 1000)), provider)
    )

    assert result is False


def test_partial_edge_box_only_counts_area_entering_the_center() -> None:
    provider = _FakeProvider([
        SetuOcrLine(
            box=_box(300, 930, 500, 1000),
            text="patreon.com/artist",
        )
    ])

    result = asyncio.run(
        detect_advertisement(_static_image((1000, 1000)), provider)
    )

    assert result is False


def test_edge_box_with_large_center_intrusion_is_advertisement() -> None:
    provider = _FakeProvider([
        SetuOcrLine(
            box=_box(200, 850, 800, 1000),
            text="patreon.com/artist",
        )
    ])

    result = asyncio.run(
        detect_advertisement(_static_image((1000, 1000)), provider)
    )

    assert result is True


def test_bottom_signature_from_real_sample_is_inside_safe_area() -> None:
    provider = _FakeProvider([
        SetuOcrLine(
            box=(
                (40.9302, 3569.8064),
                (846.5117, 3577.2903),
                (846.5117, 3670.8386),
                (40.9302, 3663.3547),
            ),
            text="Patreon·com/Helltoyou",
        )
    ])

    result = asyncio.run(
        detect_advertisement(_static_image((2560, 3712)), provider)
    )

    assert result is False


def test_extreme_aspect_ratio_is_rejected_without_running_ocr() -> None:
    provider = _FakeProvider()

    result = asyncio.run(
        detect_advertisement(_static_image((800, 100)), provider)
    )

    assert result is True
    assert provider.calls == 0


def test_animated_image_skips_all_advertisement_detection() -> None:
    provider = _FakeProvider([
        SetuOcrLine(box=_box(0, 0, 32, 32), text="patreon")
    ])

    result = asyncio.run(detect_advertisement(_animated_image(), provider))

    assert result is False
    assert provider.calls == 0


def test_large_text_without_promotion_keyword_is_not_advertisement() -> None:
    provider = _FakeProvider([
        SetuOcrLine(box=_box(100, 100, 900, 900), text="普通画面文字")
    ])

    result = asyncio.run(
        detect_advertisement(_static_image((1000, 1000)), provider)
    )

    assert result is False


def test_normalization_and_adjacent_lines_can_complete_keyword() -> None:
    provider = _FakeProvider([
        SetuOcrLine(box=_box(300, 300, 350, 380), text="ＰＡＴ"),
        SetuOcrLine(box=_box(360, 300, 410, 380), text="ＲＥＯＮ．ＣＯＭ"),
    ])

    result = asyncio.run(
        detect_advertisement(_static_image((1000, 1000)), provider)
    )

    assert result is True


def test_complete_keyword_does_not_include_adjacent_ocr_noise() -> None:
    provider = _FakeProvider([
        SetuOcrLine(box=_box(0, 10, 100, 30), text="BONZ0616"),
        SetuOcrLine(box=_box(0, 40, 100, 60), text="PATREON/FANBOX"),
        SetuOcrLine(box=_box(700, 0, 1000, 1000), text="X"),
    ])

    result = asyncio.run(
        detect_advertisement(_static_image((1000, 1000)), provider)
    )

    assert result is False


def test_multiple_small_promotion_boxes_use_combined_area() -> None:
    provider = _FakeProvider([
        SetuOcrLine(box=_box(0, 100, 120, 200), text="fanbox"),
        SetuOcrLine(box=_box(880, 700, 1000, 800), text="有料版"),
    ])

    result = asyncio.run(
        detect_advertisement(_static_image((1000, 1000)), provider)
    )

    assert result is True


def test_ocr_failure_fails_open() -> None:
    provider = _FakeProvider(error=SetuOcrError("offline"))

    result = asyncio.run(
        detect_advertisement(_static_image((1000, 1000)), provider)
    )

    assert result is False


def test_detect_advertisement_fails_open_when_image_inspection_crashes(
        monkeypatch,
) -> None:
    class _BrokenLogger:
        def warning(self, message: str) -> None:
            raise RuntimeError("logger crashed")

    def crash(image: bytes) -> tuple[int, int, bool]:
        raise RuntimeError("inspection crashed")

    monkeypatch.setattr("TIYA.setu.ocr._inspect_image", crash)
    monkeypatch.setattr("TIYA.setu.ocr._log", _BrokenLogger())

    result = asyncio.run(detect_advertisement(b"anything"))

    assert result is False


def test_detect_advertisement_fails_open_when_classification_crashes() -> None:
    class _MalformedProvider:
        async def recognize(self, image: bytes) -> list[SetuOcrLine]:
            return [object()]  # type: ignore[list-item]

    result = asyncio.run(
        detect_advertisement(
            _static_image((1000, 1000)),
            _MalformedProvider(),
        )
    )

    assert result is False


def test_detect_advertisement_does_not_swallow_cancellation() -> None:
    class _CancelledProvider:
        async def recognize(self, image: bytes) -> list[SetuOcrLine]:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            detect_advertisement(
                _static_image((1000, 1000)),
                _CancelledProvider(),
            )
        )


def test_detect_advertisement_uses_default_rapidocr_provider(monkeypatch) -> None:
    provider = _FakeProvider([
        SetuOcrLine(box=_box(200, 200, 400, 400), text="fanbox")
    ])
    monkeypatch.setattr(
        "TIYA.setu.ocr._DEFAULT_RAPIDOCR_PROVIDER",
        provider,
    )
    image = _static_image((1000, 1000))

    first = asyncio.run(detect_advertisement(image))
    second = asyncio.run(detect_advertisement(image))

    assert first is True
    assert second is True
    assert provider.calls == 2


def test_rapidocr_provider_converts_output_and_initializes_once(monkeypatch) -> None:
    created: list[dict[str, str]] = []
    submitted: list[tuple[object, tuple[object, ...]]] = []
    received: list[bytes] = []

    class _ExecutorSpy(Executor):
        def submit(self, fn, /, *args, **kwargs):
            submitted.append((fn, args))
            future: Future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    class _Engine:
        def __call__(self, image: bytes) -> SimpleNamespace:
            received.append(image)
            return SimpleNamespace(
                boxes=[[[1, 2], [3, 2], [3, 4], [1, 4]]],
                txts=("fanbox",),
            )

    def _factory(params: dict[str, str]) -> _Engine:
        created.append(params)
        return _Engine()

    monkeypatch.setattr("TIYA.setu.ocr._create_rapidocr", _factory)
    monkeypatch.setattr("TIYA.setu.ocr._rapidocr_engine", None)
    monkeypatch.setattr("TIYA.setu.ocr.GLOBAL_EXECUTOR", _ExecutorSpy())
    first_provider = RapidOcrProvider()
    second_provider = RapidOcrProvider()

    first = asyncio.run(first_provider.recognize(b"first"))
    second = asyncio.run(second_provider.recognize(b"second"))

    assert first == [
        SetuOcrLine(box=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)), text="fanbox")
    ]
    assert second == first
    assert created == [{"Global.log_level": "error"}]
    assert len(submitted) == 2
    assert received == [b"first", b"second"]


def test_http_provider_uses_tiya_protocol_and_authorization(monkeypatch) -> None:
    async def run() -> None:
        observed: dict[str, str | bytes] = {}
        source = _static_image((100, 100))

        def handler(request: httpx.Request) -> httpx.Response:
            observed["authorization"] = request.headers.get("Authorization", "")
            observed["method"] = request.method
            observed["url"] = str(request.url)
            observed["content_type"] = request.headers["Content-Type"]
            observed["content"] = request.content
            return httpx.Response(200, json={
                "lines": [{
                    "box": [[1, 2], [3, 2], [3, 4], [1, 4]],
                    "text": "fanbox",
                }]
            })

        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            "TIYA.setu.ocr.httpx.AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(handler),
                **kwargs,
            )
        )
        provider = HttpOcrProvider(
            "https://ocr.example/v1/setu/ocr",
            token="secret",
        )
        result = await provider.recognize(source)

        assert result == [
            SetuOcrLine(
                box=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)),
                text="fanbox",
            )
        ]
        assert observed["authorization"] == "Bearer secret"
        assert observed["method"] == "POST"
        assert observed["url"] == "https://ocr.example/v1/setu/ocr"
        assert str(observed["content_type"]).startswith("multipart/form-data;")
        assert b'filename="image"' in observed["content"]
        assert source in observed["content"]

    asyncio.run(run())


def test_http_provider_resizes_long_edge_and_maps_boxes_to_original(monkeypatch) -> None:
    async def run() -> None:
        observed: dict[str, int | tuple[int, int]] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            uploaded = _multipart_file_content(request)
            observed["bytes"] = len(uploaded)
            with Image.open(BytesIO(uploaded)) as image:
                observed["size"] = image.size
            return httpx.Response(200, json={
                "lines": [{
                    "box": [[100, 50], [200, 50], [200, 100], [100, 100]],
                    "text": "fanbox",
                }]
            })

        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            "TIYA.setu.ocr.httpx.AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(handler),
                **kwargs,
            )
        )
        provider = HttpOcrProvider("https://ocr.example/v1/setu/ocr")
        result = await provider.recognize(_static_image((4000, 1000)))

        assert observed["size"] == (2000, 500)
        assert observed["bytes"] <= 1024 * 1024
        assert result == [
            SetuOcrLine(
                box=(
                    (200.0, 100.0),
                    (400.0, 100.0),
                    (400.0, 200.0),
                    (200.0, 200.0),
                ),
                text="fanbox",
            )
        ]

    asyncio.run(run())


def test_http_provider_compresses_upload_to_one_megabyte(monkeypatch) -> None:
    async def run() -> None:
        observed: dict[str, int] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            uploaded = _multipart_file_content(request)
            observed["bytes"] = len(uploaded)
            return httpx.Response(200, json={"lines": []})

        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            "TIYA.setu.ocr.httpx.AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(handler),
                **kwargs,
            )
        )
        provider = HttpOcrProvider("https://ocr.example/v1/setu/ocr")
        result = await provider.recognize(
            _static_image((1000, 1000), image_format="BMP")
        )

        assert result == []
        assert observed["bytes"] <= 1024 * 1024

    asyncio.run(run())


def test_http_provider_rejects_invalid_response_shape(monkeypatch) -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"lines": [{"box": [], "text": "fanbox"}]},
            )

        real_async_client = httpx.AsyncClient
        monkeypatch.setattr(
            "TIYA.setu.ocr.httpx.AsyncClient",
            lambda **kwargs: real_async_client(
                transport=httpx.MockTransport(handler),
                **kwargs,
            )
        )
        provider = HttpOcrProvider("https://ocr.example/v1/setu/ocr")
        with pytest.raises(SetuOcrError, match="response"):
            await provider.recognize(_static_image((100, 100)))

    asyncio.run(run())
