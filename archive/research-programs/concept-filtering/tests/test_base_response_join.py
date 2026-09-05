from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from concept_filtering.base_response_join import (
    BaseResponseJoinError,
    load_joined_examples,
)
from concept_filtering.serialization import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_sha256,
)

SOURCE_REVISION = "7" * 40

_TABLE = cast(
    Callable[..., pa.Table],
    pa.table,  # pyright: ignore[reportUnknownMemberType]
)
_WRITE_TABLE = cast(
    Callable[[pa.Table, str], None],
    pq.write_table,  # pyright: ignore[reportUnknownMemberType]
)
_READ_TABLE = cast(
    Callable[[str], pa.Table],
    pq.read_table,  # pyright: ignore[reportUnknownMemberType]
)


def build_rows() -> list[dict[str, object]]:
    return _rows()


def build_entry(
    row: dict[str, object], ordinal: int, *, overlength: bool = False
) -> dict[str, object]:
    return _entry(row, ordinal, overlength=overlength)


def build_package_bytes(entries: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(entry) + b"\n" for entry in entries)


def write_parquet(path: Path, rows: list[dict[str, object]]) -> bytes:
    return _write_parquet(path, rows)


def read_rows(path: Path) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], _READ_TABLE(str(path)).to_pylist())


def _schema() -> pa.Schema:
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


def _rows() -> list[dict[str, object]]:
    return [
        {
            "messages": [
                {"content": "hello", "role": "user"},
                {"content": "world", "role": "assistant"},
            ],
            "dataset_source": "src-a",
            "orig_idx": 5,
            "split": 1,
        },
        {
            "messages": [
                {"content": "", "role": "user"},
                {"content": "a", "role": "assistant"},
                {"content": "again", "role": "user"},
                {"content": "b", "role": "assistant"},
            ],
            "dataset_source": "src-b",
            "orig_idx": 9,
            "split": 1,
        },
        {
            "messages": [
                {"content": "too long", "role": "user"},
                {"content": "dropped", "role": "assistant"},
            ],
            "dataset_source": "src-a",
            "orig_idx": 12,
            "split": 1,
        },
    ]


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> bytes:
    columns: dict[str, object] = {
        field: [row[field] for row in rows]
        for field in ("messages", "dataset_source", "orig_idx", "split")
    }
    _WRITE_TABLE(_TABLE(columns, schema=_schema()), str(path))
    return path.read_bytes()


def _entry(
    row: dict[str, object], ordinal: int, *, overlength: bool = False
) -> dict[str, object]:
    messages = cast(list[dict[str, str]], row["messages"])
    return {
        "schema_version": 1,
        "row_ordinal": ordinal,
        "orig_idx": row["orig_idx"],
        "example_id": canonical_sha256(
            {"source_revision": SOURCE_REVISION, "orig_idx": row["orig_idx"]}
        ),
        "source_record_sha256": canonical_sha256(row),
        "dataset_source": row["dataset_source"],
        "prompt_messages_sha256": canonical_sha256(messages[:-1]),
        "supplied_response_sha256": canonical_sha256(messages[-1]),
        "rendered_input_ids_sha256": "9" * 64,
        "labels_sha256": "8" * 64,
        "input_token_count": 7,
        "supervised_token_count": 3,
        "assistant_turn_count": len(messages) // 2,
        "initial_user_empty": messages[0]["content"] == "",
        "overlength": overlength,
        "full_data_member": not overlength,
        "preprocessing_status": "dropped_overlength" if overlength else "retained",
    }


def _package_bytes(entries: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(entry) + b"\n" for entry in entries)


def _payload(
    parquet_bytes: bytes, package_bytes: bytes, entries: list[dict[str, object]]
) -> dict[str, object]:
    retained = [entry for entry in entries if entry["full_data_member"] is True]
    return {
        "entry_schema_version": 1,
        "sample": {
            "sha256": bytes_sha256(parquet_bytes),
            "source_revision": SOURCE_REVISION,
            "expected_row_count": len(entries),
            "expected_split": 1,
        },
        "package_sha256": bytes_sha256(package_bytes),
        "row_count": len(entries),
        "retained_row_count": len(retained),
        "retained_input_token_count": sum(
            cast(int, entry["input_token_count"]) for entry in retained
        ),
    }


def _materialize(
    tmp_path: Path,
    rows: list[dict[str, object]] | None = None,
    *,
    mutate_entries: Callable[[list[dict[str, object]]], None] | None = None,
    mutate_payload: Callable[[dict[str, object]], None] | None = None,
) -> tuple[dict[str, object], Path, Path]:
    source_rows = copy.deepcopy(rows) if rows is not None else _rows()
    parquet_path = tmp_path / "train_1.parquet"
    parquet_bytes = _write_parquet(parquet_path, source_rows)
    read_rows = cast(
        list[dict[str, object]], _READ_TABLE(str(parquet_path)).to_pylist()
    )
    entries = [
        _entry(row, ordinal, overlength=(ordinal == 2))
        for ordinal, row in enumerate(read_rows)
    ]
    if mutate_entries is not None:
        mutate_entries(entries)
    package_path = tmp_path / "package.jsonl"
    package_path.write_bytes(_package_bytes(entries))
    payload = _payload(parquet_bytes, package_path.read_bytes(), entries)
    if mutate_payload is not None:
        mutate_payload(payload)
    return payload, package_path, parquet_path


def test_joins_retained_examples_in_package_order(tmp_path: Path) -> None:
    payload, package_path, parquet_path = _materialize(tmp_path)

    examples = load_joined_examples(payload, package_path, parquet_path)

    assert [example.orig_idx for example in examples] == [5, 9]
    assert [example.row_ordinal for example in examples] == [0, 1]
    first, second = examples
    assert first.example_id == canonical_sha256(
        {"source_revision": SOURCE_REVISION, "orig_idx": 5}
    )
    assert [dict(message) for message in first.prompt_messages] == [
        {"content": "hello", "role": "user"}
    ]
    assert [dict(message) for message in second.prompt_messages] == [
        {"content": "", "role": "user"},
        {"content": "a", "role": "assistant"},
        {"content": "again", "role": "user"},
    ]
    assert dict(second.supplied_response) == {"content": "b", "role": "assistant"}
    assert first.input_token_count == 7
    with pytest.raises(TypeError):
        cast(dict[str, str], first.prompt_messages[0])["content"] = "tampered"


def _set_entry(field: str, value: object, ordinal: int = 0):
    def mutate(entries: list[dict[str, object]]) -> None:
        entries[ordinal][field] = value

    return mutate


def _pop_entry_field(field: str, ordinal: int = 0):
    def mutate(entries: list[dict[str, object]]) -> None:
        del entries[ordinal][field]

    return mutate


def _set_payload(field: str, value: object):
    def mutate(payload: dict[str, object]) -> None:
        payload[field] = value

    return mutate


def _set_sample(field: str, value: object):
    def mutate(payload: dict[str, object]) -> None:
        cast(dict[str, object], payload["sample"])[field] = value

    return mutate


def _drop_last_entry(entries: list[dict[str, object]]) -> None:
    entries.pop()


@pytest.mark.parametrize(
    ("mutate_entries", "match"),
    [
        (_set_entry("orig_idx", 6), "orig_idx"),
        (_set_entry("row_ordinal", 5), "row_ordinal"),
        (_set_entry("schema_version", 2), "schema_version"),
        (_set_entry("dataset_source", "src-x"), "dataset_source"),
        (_set_entry("example_id", "0" * 64), "example_id"),
        (_set_entry("source_record_sha256", "0" * 64), "source_record"),
        (_set_entry("prompt_messages_sha256", "0" * 64), "prompt_messages"),
        (_set_entry("supplied_response_sha256", "0" * 64), "supplied_response"),
        (_set_entry("assistant_turn_count", 4), "assistant_turn_count"),
        (_set_entry("initial_user_empty", True), "initial_user_empty"),
        (_set_entry("full_data_member", True, ordinal=2), "preprocessing"),
        (_set_entry("preprocessing_status", "retained", ordinal=2), "preprocessing"),
        (_set_entry("extra_field", 1), "fields"),
        (_pop_entry_field("labels_sha256"), "fields"),
        (_drop_last_entry, "count"),
    ],
)
def test_rejects_rehashed_package_entry_drift(
    tmp_path: Path,
    mutate_entries: Callable[[list[dict[str, object]]], None],
    match: str,
) -> None:
    payload, package_path, parquet_path = _materialize(
        tmp_path, mutate_entries=mutate_entries
    )

    with pytest.raises(BaseResponseJoinError, match=match):
        load_joined_examples(payload, package_path, parquet_path)


@pytest.mark.parametrize(
    ("mutate_payload", "match"),
    [
        (_set_payload("package_sha256", "0" * 64), "package SHA-256"),
        (_set_payload("row_count", 4), "row count"),
        (_set_payload("retained_row_count", 3), "retained"),
        (_set_payload("retained_input_token_count", 1), "token count"),
        (_set_payload("entry_schema_version", 2), "schema_version"),
        (_set_sample("sha256", "0" * 64), "parquet SHA-256"),
        (_set_sample("expected_row_count", 4), "row count"),
        (_set_sample("expected_split", 2), "split"),
        (_set_sample("source_revision", "8" * 40), "example_id"),
    ],
)
def test_rejects_receipt_disagreement(
    tmp_path: Path,
    mutate_payload: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    payload, package_path, parquet_path = _materialize(
        tmp_path, mutate_payload=mutate_payload
    )

    with pytest.raises(BaseResponseJoinError, match=match):
        load_joined_examples(payload, package_path, parquet_path)


def _duplicate_rows() -> list[dict[str, object]]:
    rows = _rows()
    rows[1]["orig_idx"] = 5
    return rows


def _misordered_rows() -> list[dict[str, object]]:
    rows = _rows()
    messages = cast(list[dict[str, str]], rows[0]["messages"])
    messages[0]["role"] = "assistant"
    messages[1]["role"] = "user"
    return rows


def _empty_assistant_rows() -> list[dict[str, object]]:
    rows = _rows()
    cast(list[dict[str, str]], rows[0]["messages"])[1]["content"] = ""
    return rows


def _odd_turn_rows() -> list[dict[str, object]]:
    rows = _rows()
    cast(list[dict[str, str]], rows[0]["messages"]).append(
        {"content": "dangling", "role": "user"}
    )
    return rows


@pytest.mark.parametrize(
    ("rows_builder", "match"),
    [
        (_duplicate_rows, "unique"),
        (_misordered_rows, "alternate"),
        (_empty_assistant_rows, "content"),
        (_odd_turn_rows, "alternate"),
    ],
)
def test_rejects_invalid_source_rows(
    tmp_path: Path,
    rows_builder: Callable[[], list[dict[str, object]]],
    match: str,
) -> None:
    payload, package_path, parquet_path = _materialize(tmp_path, rows_builder())

    with pytest.raises(BaseResponseJoinError, match=match):
        load_joined_examples(payload, package_path, parquet_path)


def test_rejects_symlinked_package(tmp_path: Path) -> None:
    payload, package_path, parquet_path = _materialize(tmp_path)
    target = tmp_path / "aliased.jsonl"
    package_path.rename(target)
    package_path.symlink_to(target)

    with pytest.raises(BaseResponseJoinError, match="package"):
        load_joined_examples(payload, package_path, parquet_path)
