"""Frozen per-behavior score sets bridging gradient stores to manifests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from concept_filtering.attribution_identity import (
    is_sha256,
    validate_expected_universe,
    validate_hash_bindings,
)
from concept_filtering.removal_manifest import ScoredDocument
from concept_filtering.serialization import canonical_sha256

_BEHAVIORS = ("validate_feelings", "refusal")
_BINDING_ROLES = (
    "factor_store_sha256",
    "training_gradient_store_sha256",
    "query_gradient_sha256",
    "training_data_package_sha256",
    "backend_runtime_sha256",
)
_BOUNDARIES: dict[str, bool] = {
    "scientific_use": False,
    "starts_experiment_timer": False,
    "training_manifest_promotion_allowed": False,
    "training_execution_allowed": False,
}


class ScoreSetError(ValueError):
    """Raised when a per-behavior score set cannot be frozen."""


@dataclass(frozen=True)
class ScoreRecord:
    example_id: str
    row_ordinal: int
    token_count: int
    score: float


@dataclass(frozen=True)
class ScoreSetReceipt:
    payload: Mapping[str, object]
    score_set_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "score_set_sha256": self.score_set_sha256}


def _validate_record(record: ScoreRecord, expected: Mapping[str, int]) -> None:
    if not is_sha256(record.example_id):
        raise ScoreSetError("record example_id must be a lowercase SHA-256")
    if record.example_id not in expected:
        raise ScoreSetError("record names an unknown example_id")
    if expected[record.example_id] != record.row_ordinal:
        raise ScoreSetError("record row_ordinal differs from the frozen package")
    if type(record.token_count) is not int or record.token_count < 1:
        raise ScoreSetError("record token_count must be a positive integer")
    if type(record.score) is not float or not math.isfinite(record.score):
        raise ScoreSetError("record score must be a finite float")


def _completed_records(
    records: Sequence[ScoreRecord], expected: Mapping[str, int]
) -> dict[str, ScoreRecord]:
    validate_expected_universe(expected, ScoreSetError)
    completed: dict[str, ScoreRecord] = {}
    for record in records:
        _validate_record(record, expected)
        existing = completed.get(record.example_id)
        if existing is not None and existing != record:
            raise ScoreSetError("conflicting duplicate score records for one example")
        completed[record.example_id] = record
    pending = sorted(set(expected) - set(completed))
    if pending:
        raise ScoreSetError(f"score set has {len(pending)} pending example records")
    return completed


def _score_payload(
    behavior: str,
    bindings: dict[str, str],
    completed: dict[str, ScoreRecord],
) -> dict[str, object]:
    entries = [
        {
            "example_id": record.example_id,
            "row_ordinal": record.row_ordinal,
            "token_count": record.token_count,
            "score": record.score,
        }
        for record in sorted(completed.values(), key=lambda item: item.example_id)
    ]
    return {
        "schema_version": 1,
        "behavior": behavior,
        "sign_convention": "positive_score_is_proponent",
        "bindings": bindings,
        "bindings_sha256": canonical_sha256(bindings),
        "coverage": {
            "example_count": len(entries),
            "total_token_count": sum(
                record.token_count for record in completed.values()
            ),
        },
        "scores": entries,
        "boundaries": dict(_BOUNDARIES),
    }


def freeze_score_set(
    records: Sequence[ScoreRecord],
    *,
    behavior: str,
    expected: Mapping[str, int],
    bindings: Mapping[str, str],
) -> ScoreSetReceipt:
    """Freeze one complete finite per-behavior score set."""
    if behavior not in _BEHAVIORS:
        raise ScoreSetError(f"behavior must be one of {sorted(_BEHAVIORS)}")
    validated = validate_hash_bindings(bindings, _BINDING_ROLES, ScoreSetError)
    completed = _completed_records(records, expected)
    payload = _score_payload(behavior, validated, completed)
    return ScoreSetReceipt(MappingProxyType(payload), canonical_sha256(payload))


def manifest_inputs(
    receipt: ScoreSetReceipt,
) -> tuple[tuple[ScoredDocument, ...], dict[str, str], int]:
    """Derive removal-manifest documents and bindings from one frozen score set."""
    if canonical_sha256(dict(receipt.payload)) != receipt.score_set_sha256:
        raise ScoreSetError("score set receipt identity differs from its payload")
    entries = cast(list[dict[str, object]], receipt.payload["scores"])
    documents = tuple(
        ScoredDocument(
            example_id=cast(str, entry["example_id"]),
            row_ordinal=cast(int, entry["row_ordinal"]),
            token_count=cast(int, entry["token_count"]),
            score=cast(float, entry["score"]),
        )
        for entry in entries
    )
    bindings = dict(cast(Mapping[str, str], receipt.payload["bindings"]))
    manifest_bindings = {
        "score_set_sha256": receipt.score_set_sha256,
        "factor_store_sha256": bindings["factor_store_sha256"],
        "training_gradient_store_sha256": bindings["training_gradient_store_sha256"],
        "query_gradient_sha256": bindings["query_gradient_sha256"],
        "training_data_package_sha256": bindings["training_data_package_sha256"],
    }
    return documents, manifest_bindings, len(documents)
