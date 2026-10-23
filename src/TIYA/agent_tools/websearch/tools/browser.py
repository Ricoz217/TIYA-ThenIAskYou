"""Provider-neutral objects for stateful browser automation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Self

from .models import WebDocument, _validate_http_url


class MouseButton(StrEnum):
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class HistoryDirection(StrEnum):
    BACK = "back"
    FORWARD = "forward"


@dataclass(frozen=True, slots=True)
class BrowserTarget:
    """A provider-neutral element reference or CSS selector."""

    ref: str | None = None
    selector: str | None = None

    def __post_init__(self) -> None:
        values = (self.ref, self.selector)
        if sum(value is not None for value in values) != 1:
            raise ValueError("exactly one of ref or selector must be provided")
        if any(value is not None and not value.strip() for value in values):
            raise ValueError("browser targets must not be empty")


@dataclass(frozen=True, slots=True)
class BrowserElement:
    """One interactable element in a browser snapshot."""

    ref: str
    role: str = ""
    name: str = ""
    tag: str = ""
    text: str = ""
    value: str | None = None
    href: str | None = None
    clickable: bool = False
    editable: bool = False
    enabled: bool = True
    visible: bool = True
    attributes: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ref.strip():
            raise ValueError("ref must not be empty")
        if self.href is not None:
            _validate_http_url(self.href, "href")
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(dict(self.attributes)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class BrowserPage:
    """A normalized page snapshot plus stable element references."""

    session_id: str
    revision: int
    document: WebDocument
    elements: tuple[BrowserElement, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        elements = tuple(self.elements)
        refs = [element.ref for element in elements]
        if len(refs) != len(set(refs)):
            raise ValueError("element refs must be unique within a page")
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def url(self) -> str:
        return self.document.final_url or self.document.url

    @property
    def title(self) -> str:
        return self.document.title

    def element(self, ref: str) -> BrowserElement:
        try:
            return next(element for element in self.elements if element.ref == ref)
        except StopIteration as exc:
            raise KeyError(f"browser element {ref!r} was not found") from exc

    def find(
        self,
        *,
        role: str | None = None,
        name: str | None = None,
        text: str | None = None,
    ) -> tuple[BrowserElement, ...]:
        """Filter elements using case-insensitive semantic fields."""

        def matches(element: BrowserElement) -> bool:
            return (
                (role is None or element.role.casefold() == role.casefold())
                and (name is None or name.casefold() in element.name.casefold())
                and (text is None or text.casefold() in element.text.casefold())
            )

        return tuple(element for element in self.elements if matches(element))


@dataclass(frozen=True, slots=True)
class BrowserScreenshot:
    data: bytes
    mime_type: str
    width: int
    height: int
    full_page: bool = False

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("screenshot data must not be empty")
        if not self.mime_type.startswith("image/"):
            raise ValueError("screenshot mime_type must be an image type")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("screenshot dimensions must be positive")


@dataclass(frozen=True, slots=True)
class NavigateAction:
    url: str

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")


@dataclass(frozen=True, slots=True)
class ClickAction:
    target: BrowserTarget | None = None
    button: MouseButton = MouseButton.LEFT
    click_count: int = 1
    position: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if type(self.click_count) is not int or self.click_count <= 0:
            raise ValueError("click_count must be a positive integer")
        if self.target is None and self.position is None:
            raise ValueError("click requires a target or viewport position")
        if self.position is not None:
            position = tuple(float(value) for value in self.position)
            if (
                len(position) != 2
                or any(not isfinite(value) or value < 0 for value in position)
            ):
                raise ValueError("position must contain two finite non-negative values")
            object.__setattr__(self, "position", position)


@dataclass(frozen=True, slots=True)
class FillAction:
    target: BrowserTarget
    value: str
    clear: bool = True
    submit: bool = False


@dataclass(frozen=True, slots=True)
class SelectAction:
    target: BrowserTarget
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if not values or any(not value for value in values):
            raise ValueError("values must contain at least one non-empty value")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class UploadAction:
    """Attach local files to one browser file-input element."""

    target: BrowserTarget
    files: tuple[Path, ...]

    def __post_init__(self) -> None:
        files = tuple(Path(file).resolve() for file in self.files)
        if not files:
            raise ValueError("files must contain at least one file")
        missing = next((file for file in files if not file.is_file()), None)
        if missing is not None:
            raise ValueError(f"upload file does not exist or is not a file: {missing}")
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class PressAction:
    key: str
    target: BrowserTarget | None = None

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("key must not be empty")


@dataclass(frozen=True, slots=True)
class ScrollAction:
    delta_x: int = 0
    delta_y: int = 0
    target: BrowserTarget | None = None

    def __post_init__(self) -> None:
        if self.delta_x == 0 and self.delta_y == 0:
            raise ValueError("at least one scroll delta must be non-zero")


@dataclass(frozen=True, slots=True)
class WaitAction:
    duration: float | None = None
    target: BrowserTarget | None = None
    text: str | None = None
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if self.duration is None and self.target is None and self.text is None:
            raise ValueError("a wait condition must be provided")
        if self.duration is not None and self.duration < 0:
            raise ValueError("duration must be non-negative")
        if self.text is not None and not self.text:
            raise ValueError("text must not be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass(frozen=True, slots=True)
class HistoryAction:
    direction: HistoryDirection


@dataclass(frozen=True, slots=True)
class ReloadAction:
    pass


type BrowserAction = (
    NavigateAction
    | ClickAction
    | FillAction
    | SelectAction
    | UploadAction
    | PressAction
    | ScrollAction
    | WaitAction
    | HistoryAction
    | ReloadAction
)
type BrowserTargetLike = BrowserTarget | BrowserElement | str


class BrowserSession(ABC):
    """Stateful browser object returned by a BrowseProvider."""

    provider_id: str
    session_id: str

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    @abstractmethod
    async def execute(self, action: BrowserAction) -> BrowserPage:
        """Execute one normalized browser action and return the new page."""

    @abstractmethod
    async def snapshot(self) -> BrowserPage:
        """Return the current page without changing browser state."""

    @abstractmethod
    async def screenshot(self, *, full_page: bool = False) -> BrowserScreenshot:
        """Capture the current browser page."""

    @abstractmethod
    async def aclose(self) -> None:
        """Close the browser session and release its resources."""

    async def navigate(self, url: str) -> BrowserPage:
        return await self.execute(NavigateAction(url))

    async def click(
        self,
        target: BrowserTargetLike | None = None,
        *,
        button: MouseButton = MouseButton.LEFT,
        click_count: int = 1,
        position: tuple[float, float] | None = None,
    ) -> BrowserPage:
        return await self.execute(
            ClickAction(
                None if target is None else _target(target),
                button=button,
                click_count=click_count,
                position=position,
            )
        )

    async def fill(
        self,
        target: BrowserTargetLike,
        value: str,
        *,
        clear: bool = True,
        submit: bool = False,
    ) -> BrowserPage:
        return await self.execute(
            FillAction(_target(target), value, clear=clear, submit=submit)
        )

    async def select(
        self,
        target: BrowserTargetLike,
        values: str | Sequence[str],
    ) -> BrowserPage:
        normalized = (values,) if isinstance(values, str) else tuple(values)
        return await self.execute(SelectAction(_target(target), normalized))

    async def upload(
        self,
        target: BrowserTargetLike,
        files: str | Path | Sequence[str | Path],
    ) -> BrowserPage:
        values = (files,) if isinstance(files, (str, Path)) else tuple(files)
        normalized = tuple(Path(file) for file in values)
        return await self.execute(UploadAction(_target(target), normalized))

    async def press(
        self,
        key: str,
        *,
        target: BrowserTargetLike | None = None,
    ) -> BrowserPage:
        return await self.execute(
            PressAction(key, None if target is None else _target(target))
        )

    async def scroll(
        self,
        *,
        delta_x: int = 0,
        delta_y: int = 0,
        target: BrowserTargetLike | None = None,
    ) -> BrowserPage:
        return await self.execute(
            ScrollAction(
                delta_x=delta_x,
                delta_y=delta_y,
                target=None if target is None else _target(target),
            )
        )

    async def wait(
        self,
        *,
        duration: float | None = None,
        target: BrowserTargetLike | None = None,
        text: str | None = None,
        timeout: float = 10.0,
    ) -> BrowserPage:
        return await self.execute(
            WaitAction(
                duration=duration,
                target=None if target is None else _target(target),
                text=text,
                timeout=timeout,
            )
        )

    async def back(self) -> BrowserPage:
        return await self.execute(HistoryAction(HistoryDirection.BACK))

    async def forward(self) -> BrowserPage:
        return await self.execute(HistoryAction(HistoryDirection.FORWARD))

    async def reload(self) -> BrowserPage:
        return await self.execute(ReloadAction())


def _target(value: BrowserTargetLike) -> BrowserTarget:
    if isinstance(value, BrowserTarget):
        return value
    if isinstance(value, BrowserElement):
        return BrowserTarget(ref=value.ref)
    if isinstance(value, str):
        return BrowserTarget(ref=value)
    raise TypeError("browser target must be a target, element, or ref string")


__all__ = [
    "BrowserAction",
    "BrowserElement",
    "BrowserPage",
    "BrowserScreenshot",
    "BrowserSession",
    "BrowserTarget",
    "BrowserTargetLike",
    "ClickAction",
    "FillAction",
    "HistoryAction",
    "HistoryDirection",
    "MouseButton",
    "NavigateAction",
    "PressAction",
    "ReloadAction",
    "ScrollAction",
    "SelectAction",
    "UploadAction",
    "WaitAction",
]
