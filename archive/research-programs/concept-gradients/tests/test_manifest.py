from __future__ import annotations

import json
from pathlib import Path

import pytest

from concept_filtering.manifest import (
    ManifestValidationError,
    build_manifest,
    load_manifest,
    write_manifest,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8")


def test_build_manifest_binds_input_and_canonical_content(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.jsonl"
    rows: list[dict[str, object]] = [
        {
            "id": "example-b",
            "dataset_source": "source-2",
            "messages": [{"role": "user", "content": "beta"}],
            "token_count": 7,
        },
        {
            "id": "example-a",
            "dataset_source": "source-1",
            "messages": [{"role": "user", "content": "alpha"}],
            "token_count": 5,
        },
    ]
    write_jsonl(input_path, rows)

    manifest = build_manifest(input_path, manifest_id="fixture-sample")

    assert manifest.manifest_id == "fixture-sample"
    assert manifest.example_count == 2
    assert manifest.token_count == 12
    assert [entry.example_id for entry in manifest.entries] == [
        "example-a",
        "example-b",
    ]
    assert len(manifest.input_sha256) == 64
    assert len(manifest.content_sha256) == 64
    assert len(manifest.manifest_sha256) == 64


def test_content_hash_is_independent_of_row_and_key_order(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_rows: list[dict[str, object]] = [
        {"id": "a", "dataset_source": "s", "token_count": 2, "text": "A"},
        {"id": "b", "dataset_source": "s", "token_count": 3, "text": "B"},
    ]
    second_rows: list[dict[str, object]] = [
        {"text": "B", "token_count": 3, "dataset_source": "s", "id": "b"},
        {"text": "A", "id": "a", "token_count": 2, "dataset_source": "s"},
    ]
    write_jsonl(first_path, first_rows)
    write_jsonl(second_path, second_rows)

    first = build_manifest(first_path, manifest_id="sample")
    second = build_manifest(second_path, manifest_id="sample")

    assert first.content_sha256 == second.content_sha256
    assert first.input_sha256 != second.input_sha256
    assert first.manifest_sha256 != second.manifest_sha256


def test_rejects_duplicate_example_id(tmp_path: Path) -> None:
    input_path = tmp_path / "duplicate.jsonl"
    write_jsonl(
        input_path,
        [
            {"id": "same", "dataset_source": "s", "token_count": 2},
            {"id": "same", "dataset_source": "s", "token_count": 3},
        ],
    )

    with pytest.raises(ManifestValidationError, match="duplicate example id: same"):
        build_manifest(input_path, manifest_id="sample")


@pytest.mark.parametrize("token_count", [-1, True, 2.5, "2"])
def test_rejects_invalid_token_count(tmp_path: Path, token_count: object) -> None:
    input_path = tmp_path / "invalid-token-count.jsonl"
    write_jsonl(
        input_path,
        [{"id": "a", "dataset_source": "s", "token_count": token_count}],
    )

    with pytest.raises(ManifestValidationError, match="token_count"):
        build_manifest(input_path, manifest_id="sample")


def test_rejects_malformed_json_line_with_line_number(tmp_path: Path) -> None:
    input_path = tmp_path / "malformed.jsonl"
    input_path.write_text(
        '{"id":"a","dataset_source":"s","token_count":1}\nnot-json\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="line 2"):
        build_manifest(input_path, manifest_id="sample")


def test_load_manifest_rejects_mutated_source_input(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(
        input_path,
        [{"id": "a", "dataset_source": "s", "token_count": 1}],
    )
    write_manifest(build_manifest(input_path, manifest_id="sample"), manifest_path)
    write_jsonl(
        input_path,
        [{"id": "a", "dataset_source": "s", "token_count": 2}],
    )

    with pytest.raises(ManifestValidationError, match="does not match source input"):
        load_manifest(manifest_path, input_path=input_path)


def test_load_manifest_rejects_mutated_receipt(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_jsonl(
        input_path,
        [{"id": "a", "dataset_source": "s", "token_count": 1}],
    )
    write_manifest(build_manifest(input_path, manifest_id="sample"), manifest_path)
    payload = json.loads(manifest_path.read_text())
    payload["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="does not match source input"):
        load_manifest(manifest_path, input_path=input_path)
