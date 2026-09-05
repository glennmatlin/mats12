"""Deterministic manifests for materialized SFT samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.serialization import bytes_sha256, canonical_sha256


class ManifestValidationError(ValueError):
    """Raised when sample rows cannot form an unambiguous manifest."""


@dataclass(frozen=True)
class ManifestEntry:
    example_id: str
    dataset_source: str
    token_count: int
    record_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_source": self.dataset_source,
            "example_id": self.example_id,
            "record_sha256": self.record_sha256,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    manifest_id: str
    input_sha256: str
    content_sha256: str
    manifest_sha256: str
    example_count: int
    token_count: int
    entries: tuple[ManifestEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "input_sha256": self.input_sha256,
            "content_sha256": self.content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "example_count": self.example_count,
            "token_count": self.token_count,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _required_string(row: dict[str, Any], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"line {line_number}: {field} must be a non-empty string"
        )
    return value


def _parse_row(raw_line: str, line_number: int) -> tuple[dict[str, Any], ManifestEntry]:
    if not raw_line.strip():
        raise ManifestValidationError(
            f"line {line_number}: blank lines are not allowed"
        )
    try:
        value: object = json.loads(raw_line)
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            f"line {line_number}: invalid JSON: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise ManifestValidationError(f"line {line_number}: row must be a JSON object")
    row = cast(dict[str, Any], value)
    example_id = _required_string(row, "id", line_number)
    dataset_source = _required_string(row, "dataset_source", line_number)
    token_count = row.get("token_count")
    if type(token_count) is not int or token_count < 0:
        raise ManifestValidationError(
            f"line {line_number}: token_count must be a non-negative integer"
        )
    return row, ManifestEntry(
        example_id=example_id,
        dataset_source=dataset_source,
        token_count=token_count,
        record_sha256=canonical_sha256(row),
    )


def _entries_from_text(text: str) -> tuple[ManifestEntry, ...]:
    lines = text.splitlines()
    if not lines:
        raise ManifestValidationError("sample must contain at least one row")
    entries: list[ManifestEntry] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        _, entry = _parse_row(raw_line, line_number)
        if entry.example_id in seen_ids:
            raise ManifestValidationError(
                f"line {line_number}: duplicate example id: {entry.example_id}"
            )
        seen_ids.add(entry.example_id)
        entries.append(entry)
    return tuple(sorted(entries, key=lambda entry: entry.example_id))


def _manifest_from_entries(
    entries: tuple[ManifestEntry, ...], *, manifest_id: str, input_sha256: str
) -> DatasetManifest:
    entry_payload = [entry.to_dict() for entry in entries]
    token_count = sum(entry.token_count for entry in entries)
    body: dict[str, object] = {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "input_sha256": input_sha256,
        "content_sha256": canonical_sha256(entry_payload),
        "example_count": len(entries),
        "token_count": token_count,
        "entries": entry_payload,
    }
    return DatasetManifest(
        schema_version=1,
        manifest_id=manifest_id,
        input_sha256=cast(str, body["input_sha256"]),
        content_sha256=cast(str, body["content_sha256"]),
        manifest_sha256=canonical_sha256(body),
        example_count=len(entries),
        token_count=token_count,
        entries=entries,
    )


def build_manifest(path: Path, *, manifest_id: str) -> DatasetManifest:
    """Build a stable manifest while retaining the exact input-file binding."""
    if not manifest_id.strip():
        raise ManifestValidationError("manifest_id must be a non-empty string")
    try:
        source_bytes = path.read_bytes()
        text = source_bytes.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ManifestValidationError(f"cannot read sample {path}: {error}") from error
    return _manifest_from_entries(
        _entries_from_text(text),
        manifest_id=manifest_id,
        input_sha256=bytes_sha256(source_bytes),
    )


def load_manifest(path: Path, *, input_path: Path) -> DatasetManifest:
    """Rebuild a stored manifest from its source and reject any mismatch."""
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestValidationError(
            f"cannot load manifest {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ManifestValidationError("manifest receipt must be a JSON object")
    payload = cast(dict[str, Any], value)
    manifest_id = payload.get("manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id.strip():
        raise ManifestValidationError("manifest_id must be a non-empty string")
    rebuilt = build_manifest(input_path, manifest_id=manifest_id)
    if payload != rebuilt.to_dict():
        raise ManifestValidationError("manifest receipt does not match source input")
    return rebuilt


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Write a stable, human-readable manifest JSON file."""
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
