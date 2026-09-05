"""Shared identity validation for G3 attribution stores and manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def is_sha256(value: object) -> bool:
    """Report whether a value is one lowercase hex SHA-256 string."""
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def validate_hash_bindings(
    value: Mapping[str, str],
    roles: tuple[str, ...],
    error: type[ValueError],
) -> dict[str, str]:
    """Validate one complete set of distinct-role SHA-256 bindings."""
    observed: dict[str, object] = dict(value)
    unexpected = sorted(set(observed) - set(roles))
    missing = sorted(set(roles) - set(observed))
    if unexpected or missing:
        raise error(
            f"bindings unexpected fields={unexpected}, missing fields={missing}"
        )
    for role in roles:
        if not is_sha256(observed[role]):
            raise error(f"bindings.{role} must be a lowercase SHA-256")
    if len(set(observed.values())) != len(roles):
        raise error("binding hash roles must be distinct")
    return {role: cast(str, observed[role]) for role in roles}


def validate_expected_universe(
    expected: Mapping[str, int], error: type[ValueError]
) -> None:
    """Validate one canonical example_id-to-row_ordinal document universe."""
    for example_id, row_ordinal in expected.items():
        if not is_sha256(example_id):
            raise error("expected example_id values must be lowercase SHA-256")
        if type(row_ordinal) is not int or row_ordinal < 0:
            raise error("expected row_ordinal values must be whole")
    if set(expected.values()) != set(range(len(expected))):
        raise error("expected row_ordinal values must cover the frozen package")
