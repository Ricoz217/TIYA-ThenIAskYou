"""Static-image OCR and advertisement detection for the setu pipeline."""

from __future__ import annotations

import asyncio
import math
import threading
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol, TypeAlias

import httpx
from PIL import Image, UnidentifiedImageError

from TIYA.executor import GLOBAL_EXECUTOR
from TIYA.logger import get_logger

_log = get_logger()

Point: TypeAlias = tuple[float, float]
OcrBox: TypeAlias = tuple[Point, Point, Point, Point]

ADVERTISEMENT_KEYWORDS = frozenset({
    # Platforms and domains
    "fanbox",
    "pixivfanbox",
    "fanbox.cc",
    "patreon",
    "patreon.com",
    "fantia",
    "fantia.jp",
    "booth",
    "booth.pm",
    "afdian",
    "afdian.net",
    "爱发电",
    "愛發電",
    "gumroad",
    "subscribestar",
    "dlsite",
    # Chinese
    "差分",
    "支援者限定",
    "会员限定",
    "會員限定",
    "付费版",
    "付費版",
    "完整版",
    "完全版",
    "赞助",
    "贊助",
    "购买",
    "購買",
    "委托",
    "委託",
    "新刊",
    "通贩",
    "通販",
    "无修正",
    "無修正",
    # Japanese
    "限定公開",
    "有料版",
    "続きは",
    "高画質版",
    "文字なし",
    "販売中",
    "ご支援",
    "コミッション",
    "サンプル",
    # English
    "support me",
    "full version",
    "paid version",
    "exclusive",
    "subscribers only",
    "uncensored",
    "commission",
    "sample",
})


class SetuOcrError(Exception):
    """Raised when a setu OCR provider cannot produce a valid result."""


@dataclass(frozen=True, slots=True)
class SetuOcrLine:
    box: OcrBox
    text: str


class SetuOcrProvider(Protocol):
    async def recognize(self, image: bytes) -> list[SetuOcrLine]:
        """Recognize text boxes in one static image."""


def _parse_box(raw_box: Any) -> OcrBox:
    try:
        points = tuple(
            tuple(float(value) for value in raw_point)
            for raw_point in raw_box
        )
    except (TypeError, ValueError) as exc:
        raise SetuOcrError("invalid OCR response box") from exc

    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise SetuOcrError("invalid OCR response box")
    if any(not math.isfinite(value) for point in points for value in point):
        raise SetuOcrError("invalid OCR response box")

    return points  # type: ignore[return-value]


def _parse_lines(raw_boxes: Any, raw_texts: Any) -> list[SetuOcrLine]:
    if raw_boxes is None and raw_texts is None:
        return []

    try:
        boxes = list(raw_boxes)
        texts = list(raw_texts)
    except TypeError as exc:
        raise SetuOcrError("invalid OCR response lines") from exc

    if len(boxes) != len(texts):
        raise SetuOcrError("invalid OCR response lines")

    lines: list[SetuOcrLine] = []
    for box, text in zip(boxes, texts, strict=True):
        if not isinstance(text, str):
            raise SetuOcrError("invalid OCR response text")
        lines.append(SetuOcrLine(box=_parse_box(box), text=text))
    return lines


def _create_rapidocr(params: dict[str, str]) -> Any:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise SetuOcrError(
            "RapidOCR is not installed; install the setu-ocr dependencies"
        ) from exc
    return RapidOCR(params=params)


_rapidocr_engine: Any | None = None
_rapidocr_lock = threading.Lock()
_HTTP_OCR_MAX_SIDE = 2000
_HTTP_OCR_MAX_UPLOAD_BYTES = 1024 * 1024
_HTTP_OCR_JPEG_QUALITIES = (85, 75, 65, 55, 45)


class RapidOcrProvider:
    """Lazy, process-wide RapidOCR provider serialized on the global executor."""

    def _recognize_sync(self, image: bytes) -> list[SetuOcrLine]:
        global _rapidocr_engine

        with _rapidocr_lock:
            try:
                if _rapidocr_engine is None:
                    _rapidocr_engine = _create_rapidocr({"Global.log_level": "error"})

                result = _rapidocr_engine(image)
                return _parse_lines(result.boxes, result.txts)

            except SetuOcrError:
                raise

            except Exception as exc:
                raise SetuOcrError("RapidOCR recognition failed") from exc

    async def recognize(self, image: bytes) -> list[SetuOcrLine]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            GLOBAL_EXECUTOR,
            self._recognize_sync,
            image,
        )


_DEFAULT_RAPIDOCR_PROVIDER = RapidOcrProvider()


def _to_opaque_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def _prepare_http_image(
        image: bytes,
        max_side: int,
        max_upload_bytes: int,
) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    try:
        with Image.open(BytesIO(image)) as opened:
            original_size = opened.size
            if max(original_size) <= max_side and len(image) <= max_upload_bytes:
                return image, original_size, original_size

            working = _to_opaque_rgb(opened)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise SetuOcrError("invalid HTTP OCR image") from exc

    if max(working.size) > max_side:
        scale = max_side / max(working.size)
        resized_size = tuple(max(1, round(side * scale)) for side in working.size)
        working = working.resize(resized_size, Image.Resampling.LANCZOS)

    for quality in _HTTP_OCR_JPEG_QUALITIES:
        encoded = _encode_jpeg(working, quality)
        if len(encoded) <= max_upload_bytes:
            return encoded, original_size, working.size

    while max(working.size) > 320:
        resized_size = tuple(max(1, round(side * 0.8)) for side in working.size)
        working = working.resize(resized_size, Image.Resampling.LANCZOS)
        encoded = _encode_jpeg(working, _HTTP_OCR_JPEG_QUALITIES[-1])
        if len(encoded) <= max_upload_bytes:
            return encoded, original_size, working.size

    raise SetuOcrError("HTTP OCR image cannot be compressed below the upload limit")


def _map_lines_to_original(
        lines: list[SetuOcrLine],
        uploaded_size: tuple[int, int],
        original_size: tuple[int, int],
) -> list[SetuOcrLine]:
    if uploaded_size == original_size:
        return lines

    scale_x = original_size[0] / uploaded_size[0]
    scale_y = original_size[1] / uploaded_size[1]
    return [
        SetuOcrLine(
            box=tuple(
                (x * scale_x, y * scale_y)
                for x, y in line.box
            ),  # type: ignore[arg-type]
            text=line.text,
        )
        for line in lines
    ]


class HttpOcrProvider:
    """Client for a ``POST /v1/setu/ocr`` compatible OCR gateway.

    The endpoint receives a multipart ``image`` field and returns
    ``{"lines": [{"box": [[x, y], ...], "text": "..."}]}``.
    """

    def __init__(
            self,
            endpoint: str,
            *,
            token: str | None = None,
            timeout: float = 30.0,
            max_side: int = _HTTP_OCR_MAX_SIDE,
            max_upload_bytes: int = _HTTP_OCR_MAX_UPLOAD_BYTES,
    ) -> None:
        if max_side <= 0 or max_upload_bytes <= 0:
            raise ValueError("OCR image limits must be positive")
        self._endpoint = endpoint
        self._token = token
        self._timeout = timeout
        self._max_side = max_side
        self._max_upload_bytes = max_upload_bytes

    async def recognize(self, image: bytes) -> list[SetuOcrLine]:
        loop = asyncio.get_running_loop()
        upload, original_size, uploaded_size = await loop.run_in_executor(
            GLOBAL_EXECUTOR,
            _prepare_http_image,
            image,
            self._max_side,
            self._max_upload_bytes,
        )
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            async with httpx.AsyncClient(
                    timeout=self._timeout,
                    trust_env=False,
            ) as client:
                response = await client.post(
                    self._endpoint,
                    files={
                        "image": (
                            "image",
                            upload,
                            "application/octet-stream",
                        )
                    },
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError, OSError) as exc:
            raise SetuOcrError("HTTP OCR request failed") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
            raise SetuOcrError("invalid HTTP OCR response")

        boxes: list[Any] = []
        texts: list[Any] = []
        for line in payload["lines"]:
            if not isinstance(line, dict) or "box" not in line or "text" not in line:
                raise SetuOcrError("invalid HTTP OCR response line")
            boxes.append(line["box"])
            texts.append(line["text"])

        try:
            lines = _parse_lines(boxes, texts)
        except SetuOcrError as exc:
            raise SetuOcrError("invalid HTTP OCR response") from exc
        return _map_lines_to_original(lines, uploaded_size, original_size)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


_NORMALIZED_KEYWORDS = tuple(
    _normalize_text(keyword)
    for keyword in ADVERTISEMENT_KEYWORDS
)


def _contains_keyword(text: str) -> bool:
    return any(keyword in text for keyword in _NORMALIZED_KEYWORDS)


def _matching_line_indexes(lines: list[SetuOcrLine]) -> set[int]:
    normalized_lines = [_normalize_text(line.text) for line in lines]
    direct_matches = {
        index
        for index, text in enumerate(normalized_lines)
        if _contains_keyword(text)
    }
    matched = direct_matches.copy()

    for index in range(len(normalized_lines) - 1):
        if index in direct_matches or index + 1 in direct_matches:
            continue

        if _contains_keyword(normalized_lines[index] + normalized_lines[index + 1]):
            matched.update((index, index + 1))
    return matched


def _polygon_area(polygon: Sequence[Point]) -> float:
    if len(polygon) < 3:
        return 0.0

    return abs(sum(
        x * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * y
        for index, (x, y) in enumerate(polygon)
    )) / 2


def _clip_polygon_at_boundary(
        polygon: list[Point],
        *,
        axis: int,
        boundary: float,
        keep_greater: bool,
) -> list[Point]:
    if not polygon:
        return []

    def is_inside(point: Point) -> bool:
        if keep_greater:
            return point[axis] >= boundary
        return point[axis] <= boundary

    clipped: list[Point] = []
    previous = polygon[-1]
    previous_inside = is_inside(previous)
    for current in polygon:
        current_inside = is_inside(current)
        if current_inside != previous_inside:
            offset = current[axis] - previous[axis]
            factor = (boundary - previous[axis]) / offset
            intersection = (
                previous[0] + factor * (current[0] - previous[0]),
                previous[1] + factor * (current[1] - previous[1]),
            )
            clipped.append(intersection)

        if current_inside:
            clipped.append(current)

        previous = current
        previous_inside = current_inside

    return clipped


def _center_overlap_area(
        box: OcrBox,
        width: int,
        height: int,
        safe_ratio: float,
) -> float:
    left = width * safe_ratio
    right = width * (1 - safe_ratio)
    top = height * safe_ratio
    bottom = height * (1 - safe_ratio)

    polygon = list(box)
    polygon = _clip_polygon_at_boundary(
        polygon,
        axis=0,
        boundary=left,
        keep_greater=True,
    )
    polygon = _clip_polygon_at_boundary(
        polygon,
        axis=0,
        boundary=right,
        keep_greater=False,
    )
    polygon = _clip_polygon_at_boundary(
        polygon,
        axis=1,
        boundary=top,
        keep_greater=True,
    )
    polygon = _clip_polygon_at_boundary(
        polygon,
        axis=1,
        boundary=bottom,
        keep_greater=False,
    )
    return _polygon_area(polygon)


def _classify_ocr_lines(
        lines: list[SetuOcrLine],
        width: int,
        height: int,
        *,
        edge_safe_ratio: float,
        edge_text_ratio: float,
        center_text_ratio: float,
        combined_text_ratio: float,
) -> bool:
    matched_indexes = _matching_line_indexes(lines)
    if not matched_indexes:
        return False

    image_area = width * height
    metrics: list[tuple[float, bool]] = []
    for index in matched_indexes:
        box = lines[index].box
        box_area = min(_polygon_area(box), image_area)
        center_area = min(
            _center_overlap_area(box, width, height, edge_safe_ratio),
            image_area,
        )
        if center_area <= 0:
            continue

        metrics.append((
            center_area / image_area,
            math.isclose(center_area, box_area, rel_tol=1e-6, abs_tol=1e-6),
        ))

    if any(
            area_ratio >= center_text_ratio and fully_in_center
            for area_ratio, fully_in_center in metrics
    ):
        return True
    if sum(area_ratio for area_ratio, _ in metrics) >= combined_text_ratio:
        return True
    return any(
        area_ratio >= edge_text_ratio and not fully_in_center
        for area_ratio, fully_in_center in metrics
    )


def _inspect_image(image: bytes) -> tuple[int, int, bool]:
    try:
        with Image.open(BytesIO(image)) as opened:
            width, height = opened.size
            animated = bool(
                getattr(opened, "is_animated", False)
                or getattr(opened, "n_frames", 1) > 1
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise SetuOcrError("invalid image") from exc

    if width <= 0 or height <= 0:
        raise SetuOcrError("invalid image dimensions")
    return width, height, animated


async def detect_advertisement(
        image: bytes,
        provider: SetuOcrProvider | None = None,
        *,
        aspect_ratio_limit: float = 4.0,
        edge_safe_ratio: float = 0.05,
        edge_text_ratio: float = 0.008,
        center_text_ratio: float = 0.003,
        combined_text_ratio: float = 0.012,
) -> bool:
    """Return whether one static image looks like an advertisement.

    Animated images bypass both OCR and advertisement filtering. OCR failures
    fail open so a temporary provider outage does not discard illustrations.
    """
    try:
        width, height, animated = _inspect_image(image)

        if animated:
            return False

        if max(width, height) / min(width, height) >= aspect_ratio_limit:
            return True

        active_provider = (
            provider
            if provider is not None
            else _DEFAULT_RAPIDOCR_PROVIDER
        )
        lines = await active_provider.recognize(image)

        return _classify_ocr_lines(
            lines,
            width,
            height,
            edge_safe_ratio=edge_safe_ratio,
            edge_text_ratio=edge_text_ratio,
            center_text_ratio=center_text_ratio,
            combined_text_ratio=combined_text_ratio,
        )

    except Exception as exc:
        try:
            _log.warning(f"[SETU OCR] advertisement detection failed: {exc}")
        except Exception:
            pass
        return False


__all__ = [
    "ADVERTISEMENT_KEYWORDS",
    "HttpOcrProvider",
    "OcrBox",
    "Point",
    "RapidOcrProvider",
    "SetuOcrError",
    "SetuOcrLine",
    "SetuOcrProvider",
    "detect_advertisement",
]
