"""Remote Pixiv authorization-code capture."""

from .remote import (
    RemoteLoginCancelled,
    RemoteLoginError,
    RemoteLoginFailed,
    RemoteLoginTimeout,
    RemoteLoginUnavailable,
    capture_pixiv_code,
)
from .public_url import discover_public_login_url, select_public_bind_host
from .ocr import (
    HttpOcrProvider,
    RapidOcrProvider,
    SetuOcrError,
    SetuOcrLine,
    SetuOcrProvider,
    detect_advertisement,
)
from .setu import AsyncPixivApi, Illustration, Illust, PixivStorageChunk

__all__ = [
    "RemoteLoginCancelled",
    "RemoteLoginError",
    "RemoteLoginFailed",
    "RemoteLoginTimeout",
    "RemoteLoginUnavailable",
    "capture_pixiv_code",
    "discover_public_login_url",
    "select_public_bind_host",
    "HttpOcrProvider",
    "RapidOcrProvider",
    "SetuOcrError",
    "SetuOcrLine",
    "SetuOcrProvider",
    "detect_advertisement",
    "AsyncPixivApi",
    "Illustration",
    "Illust",
    "PixivStorageChunk"
]
