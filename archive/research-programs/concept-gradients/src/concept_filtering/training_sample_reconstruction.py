"""Fail-closed verification for the named public rank-32 training sample."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from concept_filtering.serialization import bytes_sha256, canonical_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_SCHEME = "source-revision-orig-idx-v1"
_READ_TABLE = cast(
    Callable[[object], pa.Table],
    pq.read_table,  # pyright: ignore[reportUnknownMemberType]
)


class TrainingSampleReconstructionError(ValueError):
    """Raised when the named training-sample evidence does not match."""


@dataclass(frozen=True)
class ArtifactReference:
    repository: str
    revision: str
    path: str | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "repository": self.repository,
            "revision": self.revision,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        return payload


@dataclass(frozen=True)
class UnresolvedEvidence:
    status: str
    checked_at: str
    method: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checked_at": self.checked_at,
            "method": self.method,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrainingSampleReconstruction:
    reconstruction_id: str
    scientific_use: bool
    starts_experiment_timer: bool
    source: ArtifactReference
    sample: ArtifactReference
    expected_row_count: int
    expected_split: int
    identity_scheme: str
    unresolved: dict[str, UnresolvedEvidence]
    divergences: tuple[str, ...]
    config_sha256: str


@dataclass(frozen=True)
class TrainingSampleReceipt:
    reconstruction_id: str
    scientific_use: bool
    starts_experiment_timer: bool
    source: ArtifactReference
    sample: ArtifactReference
    row_count: int
    split: int
    orig_idx_min: int
    orig_idx_max: int
    orig_idx_sha256: str
    ordered_entry_sha256: str
    identity_entry_sha256: str
    dataset_source_counts: dict[str, int]
    dataset_source_counts_sha256: str
    identity_scheme: str
    unresolved: dict[str, UnresolvedEvidence]
    divergences: tuple[str, ...]
    config_sha256: str
    receipt_sha256: str

    def body_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reconstruction_id": self.reconstruction_id,
            "scientific_use": self.scientific_use,
            "starts_experiment_timer": self.starts_experiment_timer,
            "source": self.source.to_dict(),
            "sample": self.sample.to_dict(),
            "row_count": self.row_count,
            "split": self.split,
            "orig_idx_min": self.orig_idx_min,
            "orig_idx_max": self.orig_idx_max,
            "orig_idx_sha256": self.orig_idx_sha256,
            "ordered_entry_sha256": self.ordered_entry_sha256,
            "identity_entry_sha256": self.identity_entry_sha256,
            "dataset_source_counts": self.dataset_source_counts,
            "dataset_source_counts_sha256": self.dataset_source_counts_sha256,
            "identity_scheme": self.identity_scheme,
            "unresolved": {
                key: value.to_dict() for key, value in sorted(self.unresolved.items())
            },
            "divergences": list(self.divergences),
            "config_sha256": self.config_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body_dict(), "receipt_sha256": self.receipt_sha256}


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingSampleReconstructionError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_fields(payload: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise TrainingSampleReconstructionError(
            f"{field} fields differ; missing={missing}, unexpected={unexpected}"
        )


def _string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TrainingSampleReconstructionError(f"{field} must be a non-empty string")
    return value


def _artifact(value: object, *, sample: bool) -> ArtifactReference:
    payload = _mapping(value, "sample" if sample else "source")
    expected = (
        {"repository", "revision", "path", "sha256"}
        if sample
        else {
            "repository",
            "revision",
        }
    )
    _exact_fields(payload, expected, "sample" if sample else "source")
    revision = _string(payload, "revision")
    if _REVISION.fullmatch(revision) is None:
        raise TrainingSampleReconstructionError("revision must be a 40-character SHA")
    path = _string(payload, "path") if sample else None
    sha256 = _string(payload, "sha256") if sample else None
    if sha256 is not None and _SHA256.fullmatch(sha256) is None:
        raise TrainingSampleReconstructionError("sha256 must be a lowercase SHA-256")
    return ArtifactReference(_string(payload, "repository"), revision, path, sha256)


def _unresolved_item(value: object, field: str) -> UnresolvedEvidence:
    payload = _mapping(value, field)
    _exact_fields(payload, {"status", "checked_at", "method", "reason"}, field)
    status = _string(payload, "status")
    if status != "unavailable":
        raise TrainingSampleReconstructionError(f"{field}.status must be unavailable")
    return UnresolvedEvidence(
        status=status,
        checked_at=_string(payload, "checked_at"),
        method=_string(payload, "method"),
        reason=_string(payload, "reason"),
    )


def _unresolved(value: object) -> dict[str, UnresolvedEvidence]:
    payload = _mapping(value, "unresolved")
    expected = {"sample_license", "adapter_binding"}
    _exact_fields(payload, expected, "unresolved")
    return {key: _unresolved_item(payload[key], key) for key in sorted(expected)}


def _divergences(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TrainingSampleReconstructionError("divergences must be a non-empty list")
    items = cast(list[object], value)
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise TrainingSampleReconstructionError("divergences must contain strings")
    return tuple(cast(list[str], items))


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingSampleReconstructionError(
            f"cannot load {path}: {error}"
        ) from error
    return _mapping(value, "reconstruction")


def _validate_envelope(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "reconstruction_id",
        "scientific_use",
        "starts_experiment_timer",
        "source",
        "sample",
        "expected_row_count",
        "expected_split",
        "identity_scheme",
        "unresolved",
        "divergences",
    }
    _exact_fields(payload, expected, "reconstruction")
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise TrainingSampleReconstructionError("schema_version must be the integer 1")
    for field in ("scientific_use", "starts_experiment_timer"):
        if payload[field] is not False:
            raise TrainingSampleReconstructionError(f"{field} must remain false")


def _positive_int(payload: dict[str, Any], field: str) -> int:
    value = payload[field]
    if type(value) is not int or value < 1:
        raise TrainingSampleReconstructionError(f"{field} must be a positive integer")
    return value


def load_training_sample_reconstruction(path: Path) -> TrainingSampleReconstruction:
    """Load the fail-closed named reconstruction contract."""
    payload = _load_payload(path)
    _validate_envelope(payload)
    identity_scheme = _string(payload, "identity_scheme")
    if identity_scheme != _IDENTITY_SCHEME:
        raise TrainingSampleReconstructionError("unsupported identity_scheme")
    return TrainingSampleReconstruction(
        reconstruction_id=_string(payload, "reconstruction_id"),
        scientific_use=False,
        starts_experiment_timer=False,
        source=_artifact(payload["source"], sample=False),
        sample=_artifact(payload["sample"], sample=True),
        expected_row_count=_positive_int(payload, "expected_row_count"),
        expected_split=_positive_int(payload, "expected_split"),
        identity_scheme=identity_scheme,
        unresolved=_unresolved(payload["unresolved"]),
        divergences=_divergences(payload["divergences"]),
        config_sha256=canonical_sha256(payload),
    )


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


def _read_sample_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise TrainingSampleReconstructionError(
            f"cannot read sample {path}: {error}"
        ) from error


def _read_table(config: TrainingSampleReconstruction, path: Path) -> pa.Table:
    sample_bytes = _read_sample_bytes(path)
    expected_sha = config.sample.sha256
    if expected_sha is None or bytes_sha256(sample_bytes) != expected_sha:
        raise TrainingSampleReconstructionError("sample SHA-256 does not match config")
    try:
        table = _READ_TABLE(pa.BufferReader(sample_bytes))
    except (OSError, pa.ArrowException) as error:
        raise TrainingSampleReconstructionError(
            f"cannot read sample {path}: {error}"
        ) from error
    if not table.schema.remove_metadata().equals(_expected_schema()):
        raise TrainingSampleReconstructionError(
            "sample schema does not match v1 schema"
        )
    return table


def _validate_messages(messages: object, row_number: int) -> None:
    if not isinstance(messages, list) or not messages:
        raise TrainingSampleReconstructionError(f"row {row_number}: messages is empty")
    for message in cast(list[object], messages):
        if not isinstance(message, dict):
            raise TrainingSampleReconstructionError(
                f"row {row_number}: invalid message"
            )
        payload = cast(dict[str, object], message)
        content = payload.get("content")
        role = payload.get("role")
        if not isinstance(content, str) or not isinstance(role, str):
            raise TrainingSampleReconstructionError(
                f"row {row_number}: invalid message"
            )


def _validated_rows(
    config: TrainingSampleReconstruction, table: pa.Table
) -> list[dict[str, object]]:
    if table.num_rows != config.expected_row_count:
        raise TrainingSampleReconstructionError(
            "sample row count does not match config"
        )
    rows = cast(list[dict[str, object]], table.to_pylist())
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row["dataset_source"], str) or not row["dataset_source"]:
            raise TrainingSampleReconstructionError(f"row {row_number}: invalid source")
        if type(row["orig_idx"]) is not int or row["split"] != config.expected_split:
            raise TrainingSampleReconstructionError(
                f"row {row_number}: invalid identity"
            )
        _validate_messages(row["messages"], row_number)
    return rows


def _identity_entry(source_revision: str, row: dict[str, object]) -> dict[str, str]:
    example_id = canonical_sha256(
        {"source_revision": source_revision, "orig_idx": row["orig_idx"]}
    )
    return {"example_id": example_id, "record_sha256": canonical_sha256(row)}


def _receipt_body(
    config: TrainingSampleReconstruction, rows: list[dict[str, object]]
) -> dict[str, object]:
    orig_indices = cast(list[int], [row["orig_idx"] for row in rows])
    if len(orig_indices) != len(set(orig_indices)):
        raise TrainingSampleReconstructionError("orig_idx values must be unique")
    entries = [_identity_entry(config.source.revision, row) for row in rows]
    sources = (cast(str, row["dataset_source"]) for row in rows)
    counts = dict(sorted(Counter(sources).items()))
    return {
        "orig_idx_min": min(orig_indices),
        "orig_idx_max": max(orig_indices),
        "orig_idx_sha256": canonical_sha256(orig_indices),
        "ordered_entry_sha256": canonical_sha256(entries),
        "identity_entry_sha256": canonical_sha256(
            sorted(entries, key=lambda item: item["example_id"])
        ),
        "dataset_source_counts": counts,
        "dataset_source_counts_sha256": canonical_sha256(counts),
    }


def _build_receipt(
    config: TrainingSampleReconstruction, body: dict[str, object]
) -> TrainingSampleReceipt:
    receipt = TrainingSampleReceipt(
        reconstruction_id=config.reconstruction_id,
        scientific_use=False,
        starts_experiment_timer=False,
        source=config.source,
        sample=config.sample,
        row_count=config.expected_row_count,
        split=config.expected_split,
        orig_idx_min=cast(int, body["orig_idx_min"]),
        orig_idx_max=cast(int, body["orig_idx_max"]),
        orig_idx_sha256=cast(str, body["orig_idx_sha256"]),
        ordered_entry_sha256=cast(str, body["ordered_entry_sha256"]),
        identity_entry_sha256=cast(str, body["identity_entry_sha256"]),
        dataset_source_counts=cast(dict[str, int], body["dataset_source_counts"]),
        dataset_source_counts_sha256=cast(str, body["dataset_source_counts_sha256"]),
        identity_scheme=config.identity_scheme,
        unresolved=config.unresolved,
        divergences=config.divergences,
        config_sha256=config.config_sha256,
        receipt_sha256="",
    )
    return replace(receipt, receipt_sha256=canonical_sha256(receipt.body_dict()))


def verify_training_sample_reconstruction(
    config: TrainingSampleReconstruction, *, sample_path: Path
) -> TrainingSampleReceipt:
    """Verify exact bytes and row identities without authorizing scientific use."""
    table = _read_table(config, sample_path)
    rows = _validated_rows(config, table)
    return _build_receipt(config, _receipt_body(config, rows))


def write_training_sample_receipt(receipt: TrainingSampleReceipt, path: Path) -> None:
    """Write a receipt without replacing existing evidence."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            receipt.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
