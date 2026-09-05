from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import concept_filtering.training_sample_reconstruction as reconstruction
from concept_filtering.cli import main
from concept_filtering.training_sample_reconstruction import (
    TrainingSampleReconstructionError,
    load_training_sample_reconstruction,
    verify_training_sample_reconstruction,
)

_TABLE = cast(
    Callable[[dict[str, object]], pa.Table],
    pa.table,  # pyright: ignore[reportUnknownMemberType]
)
_WRITE_TABLE = cast(
    Callable[[pa.Table, str], None],
    pq.write_table,  # pyright: ignore[reportUnknownMemberType]
)


def _messages_array() -> object:
    message = pa.struct(
        [pa.field("content", pa.string()), pa.field("role", pa.string())]
    )
    values = [
        [
            {"content": "question", "role": "user"},
            {"content": "answer", "role": "assistant"},
        ],
        [
            {"content": "other", "role": "user"},
            {"content": "reply", "role": "assistant"},
        ],
    ]
    return pa.array(values, type=pa.list_(pa.field("element", message)))


def _write_sample(
    path: Path,
    *,
    orig_indices: tuple[int, int] = (9, 4),
    splits: tuple[int, int] = (1, 1),
) -> str:
    table = _TABLE(
        {
            "messages": _messages_array(),
            "dataset_source": ["source-b", "source-a"],
            "orig_idx": pa.array(orig_indices, type=pa.int64()),
            "split": pa.array(splits, type=pa.int64()),
        }
    )
    _WRITE_TABLE(table, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unresolved_fixture() -> dict[str, object]:
    return {
        "sample_license": {
            "status": "unavailable",
            "checked_at": "2026-08-15",
            "method": "Hub metadata and repository card inspection",
            "reason": "No license declaration was published.",
        },
        "adapter_binding": {
            "status": "unavailable",
            "checked_at": "2026-08-15",
            "method": "Public repository inspection",
            "reason": "No producer receipt binds this sample to the adapter.",
        },
    }


def _write_config(path: Path, sample_sha256: str) -> None:
    payload = {
        "schema_version": 1,
        "reconstruction_id": "fixture-train1",
        "scientific_use": False,
        "starts_experiment_timer": False,
        "source": {
            "repository": "owner/source",
            "revision": "a" * 40,
        },
        "sample": {
            "repository": "owner/sample",
            "revision": "b" * 40,
            "path": "data/train_1.parquet",
            "sha256": sample_sha256,
        },
        "expected_row_count": 2,
        "expected_split": 1,
        "identity_scheme": "source-revision-orig-idx-v1",
        "unresolved": _unresolved_fixture(),
        "divergences": ["This is a named reconstruction, not source provenance."],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_builds_receipt_for_exact_training_sample(tmp_path: Path) -> None:
    sample_path = tmp_path / "train_1.parquet"
    config_path = tmp_path / "config.json"
    _write_config(config_path, _write_sample(sample_path))

    receipt = verify_training_sample_reconstruction(
        load_training_sample_reconstruction(config_path), sample_path=sample_path
    )

    assert receipt.row_count == 2
    assert receipt.orig_idx_min == 4
    assert receipt.orig_idx_max == 9
    assert receipt.dataset_source_counts == {"source-a": 1, "source-b": 1}
    assert receipt.scientific_use is False
    assert receipt.starts_experiment_timer is False
    assert len(receipt.ordered_entry_sha256) == 64
    assert len(receipt.receipt_sha256) == 64


def test_rejects_sample_hash_mismatch(tmp_path: Path) -> None:
    sample_path = tmp_path / "train_1.parquet"
    config_path = tmp_path / "config.json"
    _write_sample(sample_path)
    _write_config(config_path, "0" * 64)

    with pytest.raises(TrainingSampleReconstructionError, match="SHA-256"):
        verify_training_sample_reconstruction(
            load_training_sample_reconstruction(config_path), sample_path=sample_path
        )


def test_hashes_and_parses_the_same_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_path = tmp_path / "train_1.parquet"
    replacement_path = tmp_path / "replacement.parquet"
    config_path = tmp_path / "config.json"
    sample_sha256 = _write_sample(sample_path)
    _write_sample(replacement_path, orig_indices=(2, 3))
    replacement_bytes = replacement_path.read_bytes()
    _write_config(config_path, sample_sha256)
    original_read = reconstruction._read_sample_bytes  # pyright: ignore[reportPrivateUsage]
    swapped = False

    def swap_after_read(path: Path) -> bytes:
        nonlocal swapped
        sample_bytes = original_read(path)
        path.write_bytes(replacement_bytes)
        swapped = True
        return sample_bytes

    monkeypatch.setattr(reconstruction, "_read_sample_bytes", swap_after_read)
    receipt = verify_training_sample_reconstruction(
        load_training_sample_reconstruction(config_path), sample_path=sample_path
    )

    assert swapped is True
    assert receipt.orig_idx_min == 4
    assert receipt.orig_idx_max == 9


@pytest.mark.parametrize(
    ("orig_indices", "splits", "message"),
    [
        ((9, 9), (1, 1), "orig_idx values must be unique"),
        ((9, 4), (1, 2), "invalid identity"),
    ],
)
def test_rejects_invalid_row_identity(
    tmp_path: Path,
    orig_indices: tuple[int, int],
    splits: tuple[int, int],
    message: str,
) -> None:
    sample_path = tmp_path / "train_1.parquet"
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        _write_sample(sample_path, orig_indices=orig_indices, splits=splits),
    )

    with pytest.raises(TrainingSampleReconstructionError, match=message):
        verify_training_sample_reconstruction(
            load_training_sample_reconstruction(config_path), sample_path=sample_path
        )


@pytest.mark.parametrize("field", ["scientific_use", "starts_experiment_timer"])
def test_rejects_enabled_boundary_flag(tmp_path: Path, field: str) -> None:
    sample_path = tmp_path / "train_1.parquet"
    config_path = tmp_path / "config.json"
    _write_config(config_path, _write_sample(sample_path))
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload[field] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TrainingSampleReconstructionError, match="must remain false"):
        load_training_sample_reconstruction(config_path)


def test_cli_writes_training_sample_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_path = tmp_path / "train_1.parquet"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "receipt.json"
    _write_config(config_path, _write_sample(sample_path))

    exit_code = main(
        [
            "verify-training-sample-reconstruction",
            str(config_path),
            "--parquet",
            str(sample_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["row_count"] == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["row_count"] == 2


def test_cli_refuses_to_overwrite_training_sample_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_path = tmp_path / "train_1.parquet"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "receipt.json"
    _write_config(config_path, _write_sample(sample_path))
    output_path.write_text("keep", encoding="utf-8")

    exit_code = main(
        [
            "verify-training-sample-reconstruction",
            str(config_path),
            "--parquet",
            str(sample_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert "exists" in capsys.readouterr().err
    assert output_path.read_text(encoding="utf-8") == "keep"
