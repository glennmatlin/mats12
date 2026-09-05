"""Authenticated join of the frozen training-data package with its source rows."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from concept_filtering.serialization import (
    bytes_sha256,
    canonical_sha256,
    json_loads_unique,
)

_READ_TABLE = cast(
    Callable[[object], pa.Table],
    pq.read_table,  # pyright: ignore[reportUnknownMemberType]
)


class BaseResponseJoinError(ValueError):
    """Raised when the package, source rows, and receipt do not agree."""


_ENTRY_FIELDS = {
    "schema_version",
    "row_ordinal",
    "orig_idx",
    "example_id",
    "source_record_sha256",
    "dataset_source",
    "prompt_messages_sha256",
    "supplied_response_sha256",
    "rendered_input_ids_sha256",
    "labels_sha256",
    "input_token_count",
    "supervised_token_count",
    "assistant_turn_count",
    "initial_user_empty",
    "overlength",
    "full_data_member",
    "preprocessing_status",
}


@dataclass(frozen=True)
class JoinedExample:
    row_ordinal: int
    orig_idx: int
    example_id: str
    prompt_messages: tuple[Mapping[str, str], ...]
    supplied_response: Mapping[str, str]
    prompt_messages_sha256: str
    supplied_response_sha256: str
    rendered_input_ids_sha256: str
    input_token_count: int


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaseResponseJoinError(f"{field} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _regular_bytes(path: Path, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise BaseResponseJoinError(f"cannot read {label}: {error}") from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()), label)
            source = handle.read()
            after = _file_identity(os.fstat(handle.fileno()), label)
        path_identity = _file_identity(path.lstat(), label)
    except OSError as error:
        raise BaseResponseJoinError(f"cannot read {label}: {error}") from error
    if before != after or path_identity != after:
        raise BaseResponseJoinError(f"{label} changed while reading")
    return source


def _file_identity(item: os.stat_result, label: str) -> tuple[int, int, int, int]:
    if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        raise BaseResponseJoinError(f"{label} must be a unique file")
    return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns


def _expected_schema() -> pa.Schema:
    message = pa.struct(
        [pa.field("content", pa.string()), pa.field("role", pa.string())]
    )
    return pa.schema(
        [
            pa.field("messages", pa.list_(pa.field("element", message))),
            pa.field("dataset_source", pa.string()),
            pa.field("orig_idx", pa.int64()),
            pa.field("split", pa.int64()),
        ]
    )


def _source_rows(payload: Mapping[str, object], path: Path) -> list[dict[str, object]]:
    sample = _object(payload.get("sample"), "receipt sample")
    source = _regular_bytes(path, "source parquet")
    if bytes_sha256(source) != sample.get("sha256"):
        raise BaseResponseJoinError("source parquet SHA-256 differs from receipt")
    try:
        table = _READ_TABLE(pa.BufferReader(source))
    except (OSError, pa.ArrowException) as error:
        raise BaseResponseJoinError(f"cannot parse source parquet: {error}") from error
    if not table.schema.remove_metadata().equals(_expected_schema()):
        raise BaseResponseJoinError("source parquet schema differs from the contract")
    return cast(list[dict[str, object]], table.to_pylist())


def _package_entries(
    payload: Mapping[str, object], path: Path
) -> list[dict[str, object]]:
    source = _regular_bytes(path, "package")
    if bytes_sha256(source) != payload.get("package_sha256"):
        raise BaseResponseJoinError("package SHA-256 differs from receipt")
    entries: list[dict[str, object]] = []
    for ordinal, line in enumerate(source.decode().splitlines()):
        try:
            parsed = json_loads_unique(line)
        except ValueError as error:
            raise BaseResponseJoinError(
                f"package entry {ordinal} is invalid: {error}"
            ) from error
        if not isinstance(parsed, dict):
            raise BaseResponseJoinError(f"package entry {ordinal} must be an object")
        entries.append(cast(dict[str, object], parsed))
    return entries


def _validate_messages(messages: list[object], ordinal: int) -> None:
    if len(messages) < 2 or len(messages) % 2:
        raise BaseResponseJoinError(f"row {ordinal}: turns must alternate")
    for index, item in enumerate(messages):
        if not isinstance(item, Mapping):
            raise BaseResponseJoinError(f"row {ordinal}: message is invalid")
        message = cast(Mapping[str, object], item)
        role = "user" if index % 2 == 0 else "assistant"
        if set(message) != {"content", "role"} or message.get("role") != role:
            raise BaseResponseJoinError(f"row {ordinal}: turns must alternate")
        content = message.get("content")
        if not isinstance(content, str) or (role == "assistant" and not content):
            raise BaseResponseJoinError(f"row {ordinal}: content is invalid")


def _validate_row(
    row: Mapping[str, object], expected_split: object, ordinal: int
) -> None:
    source = row.get("dataset_source")
    if not isinstance(source, str) or not source:
        raise BaseResponseJoinError(f"row {ordinal}: dataset source is invalid")
    orig_idx = row.get("orig_idx")
    if type(orig_idx) is not int or orig_idx < 0:
        raise BaseResponseJoinError(f"row {ordinal}: orig_idx is invalid")
    if row.get("split") != expected_split:
        raise BaseResponseJoinError(f"row {ordinal}: split differs from receipt")
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise BaseResponseJoinError(f"row {ordinal}: messages are invalid")
    _validate_messages(cast(list[object], messages), ordinal)


def _expected_entry_values(
    row: Mapping[str, object], source_revision: object, ordinal: int
) -> dict[str, object]:
    messages = cast(list[dict[str, str]], row["messages"])
    return {
        "row_ordinal": ordinal,
        "orig_idx": row["orig_idx"],
        "example_id": canonical_sha256(
            {"source_revision": source_revision, "orig_idx": row["orig_idx"]}
        ),
        "source_record_sha256": canonical_sha256(dict(row)),
        "dataset_source": row["dataset_source"],
        "prompt_messages_sha256": canonical_sha256(messages[:-1]),
        "supplied_response_sha256": canonical_sha256(messages[-1]),
        "assistant_turn_count": len(messages) // 2,
        "initial_user_empty": messages[0]["content"] == "",
    }


def _validate_entry(
    entry: Mapping[str, object],
    row: Mapping[str, object],
    source_revision: object,
    entry_schema_version: object,
    ordinal: int,
) -> None:
    context = f"package entry {ordinal}"
    if set(entry) != _ENTRY_FIELDS:
        raise BaseResponseJoinError(f"{context} fields differ from schema")
    if entry.get("schema_version") != entry_schema_version:
        raise BaseResponseJoinError(f"{context}: schema_version differs")
    expected = _expected_entry_values(row, source_revision, ordinal)
    for field, value in expected.items():
        observed = entry.get(field)
        if observed != value or type(observed) is not type(value):
            raise BaseResponseJoinError(f"{context}: {field} differs from source")
    if entry.get("full_data_member") is not (entry.get("overlength") is False):
        raise BaseResponseJoinError(f"{context}: preprocessing flags disagree")
    expected_status = (
        "retained" if entry.get("full_data_member") is True else "dropped_overlength"
    )
    if entry.get("preprocessing_status") != expected_status:
        raise BaseResponseJoinError(f"{context}: preprocessing status disagrees")
    count = entry.get("input_token_count")
    if type(count) is not int or count <= 0:
        raise BaseResponseJoinError(f"{context}: input_token_count is invalid")


def _validate_counts(
    payload: Mapping[str, object],
    sample: Mapping[str, object],
    entries: list[dict[str, object]],
    rows: list[dict[str, object]],
) -> None:
    if len(entries) != len(rows):
        raise BaseResponseJoinError("package entry count differs from source rows")
    expected = (payload.get("row_count"), sample.get("expected_row_count"))
    if any(value != len(rows) for value in expected):
        raise BaseResponseJoinError("package row count differs from receipt")
    retained = [entry for entry in entries if entry.get("full_data_member") is True]
    if payload.get("retained_row_count") != len(retained):
        raise BaseResponseJoinError("package retained count differs from receipt")
    total = sum(cast(int, entry["input_token_count"]) for entry in retained)
    if payload.get("retained_input_token_count") != total:
        raise BaseResponseJoinError("package retained token count differs from receipt")


def _joined_example(entry: Mapping[str, object], row: Mapping[str, object]):
    messages = cast(list[dict[str, str]], row["messages"])
    prompt = tuple(MappingProxyType(dict(message)) for message in messages[:-1])
    return JoinedExample(
        row_ordinal=cast(int, entry["row_ordinal"]),
        orig_idx=cast(int, entry["orig_idx"]),
        example_id=cast(str, entry["example_id"]),
        prompt_messages=prompt,
        supplied_response=MappingProxyType(dict(messages[-1])),
        prompt_messages_sha256=cast(str, entry["prompt_messages_sha256"]),
        supplied_response_sha256=cast(str, entry["supplied_response_sha256"]),
        rendered_input_ids_sha256=cast(str, entry["rendered_input_ids_sha256"]),
        input_token_count=cast(int, entry["input_token_count"]),
    )


def load_joined_examples(
    payload: Mapping[str, object], package_path: Path, parquet_path: Path
) -> tuple[JoinedExample, ...]:
    """Join authenticated package entries with their source rows by ordinal."""
    sample = _object(payload.get("sample"), "receipt sample")
    entries = _package_entries(payload, package_path)
    rows = _source_rows(payload, parquet_path)
    _validate_counts(payload, sample, entries, rows)
    source_revision = sample.get("source_revision")
    schema_version = payload.get("entry_schema_version")
    identities: set[int] = set()
    for ordinal, (entry, row) in enumerate(zip(entries, rows, strict=True)):
        _validate_row(row, sample.get("expected_split"), ordinal)
        _validate_entry(entry, row, source_revision, schema_version, ordinal)
        orig_idx = cast(int, row["orig_idx"])
        if orig_idx in identities:
            raise BaseResponseJoinError("source orig_idx values must be unique")
        identities.add(orig_idx)
    return tuple(
        _joined_example(entry, row)
        for entry, row in zip(entries, rows, strict=True)
        if entry.get("full_data_member") is True
    )
