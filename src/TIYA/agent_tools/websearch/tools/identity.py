"""Safe resolution of reusable browser identity profiles."""

from __future__ import annotations

from pathlib import Path

from .models import IdentityRef


class IdentityStore:
    """Map opaque identity references to persistent browser profile directories."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, identity: IdentityRef, *, create: bool = False) -> Path:
        """Resolve a validated profile ID without allowing path traversal."""

        path = (self.root / identity.profile_id).resolve()
        if path.parent != self.root:
            raise ValueError("identity profile escaped the configured profile root")
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path


__all__ = ["IdentityStore"]
