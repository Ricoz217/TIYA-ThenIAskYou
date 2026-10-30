"""Communication fingerprints; preset format parsing belongs to TIYA.config."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping


def plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return copy.deepcopy(value)


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            plain(value), sort_keys=True, ensure_ascii=False, default=str
        ).encode()
    ).hexdigest()
