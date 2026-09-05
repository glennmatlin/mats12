"""Canonical JSON and SHA-256 helpers shared by reconstruction receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast


class DuplicateJSONKeyError(ValueError):
    """Raised when a JSON object repeats a key at any nesting level."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def json_loads_unique(source: str) -> object:
    """Parse JSON while rejecting duplicate object keys."""
    return json.loads(source, object_pairs_hook=_unique_object)


def json_equal_exact(left: object, right: object) -> bool:
    """Compare JSON values without bool/int/float coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        left_map = cast(dict[str, object], left)
        right_map = cast(dict[str, object], right)
        return set(left_map) == set(right_map) and all(
            json_equal_exact(left_map[key], right_map[key]) for key in left_map
        )
    if isinstance(left, list) and isinstance(right, list):
        left_list = cast(list[object], left)
        right_list = cast(list[object], right)
        return len(left_list) == len(right_list) and all(
            json_equal_exact(a, b) for a, b in zip(left_list, right_list, strict=True)
        )
    return left == right


def canonical_json_bytes(value: object) -> bytes:
    """Encode a JSON-compatible value without incidental whitespace or key order."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
