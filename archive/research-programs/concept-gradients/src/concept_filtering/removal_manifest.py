"""Deterministic frozen removal manifests for behavior-specific selectors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from concept_filtering.attribution_identity import is_sha256, validate_hash_bindings
from concept_filtering.serialization import canonical_sha256

_BEHAVIORS = ("validate_feelings", "refusal")
_SELECTORS = ("ordinary_behavior_query_ekfac", "causal_concept_gradient")
_BINDING_ROLES = (
    "score_set_sha256",
    "factor_store_sha256",
    "training_gradient_store_sha256",
    "query_gradient_sha256",
    "training_data_package_sha256",
)
_BOUNDARIES: dict[str, bool] = {
    "scientific_use": False,
    "starts_experiment_timer": False,
    "training_manifest_promotion_allowed": False,
    "training_execution_allowed": False,
}


class RemovalManifestError(ValueError):
    """Raised when a selector removal manifest cannot be frozen."""


@dataclass(frozen=True)
class ScoredDocument:
    example_id: str
    row_ordinal: int
    token_count: int
    score: float


@dataclass(frozen=True)
class RemovalManifest:
    payload: Mapping[str, object]
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "manifest_sha256": self.manifest_sha256}

    @property
    def removed_example_ids(self) -> tuple[str, ...]:
        removed = cast(list[dict[str, object]], self.payload["removed_documents"])
        return tuple(cast(str, item["example_id"]) for item in removed)

    @property
    def document_universe(self) -> tuple[int, str]:
        coverage = cast(Mapping[str, object], self.payload["coverage"])
        bindings = cast(Mapping[str, str], self.payload["bindings"])
        return (
            cast(int, coverage["document_count"]),
            bindings["training_data_package_sha256"],
        )


def _identifier(value: str, allowed: tuple[str, ...], label: str) -> str:
    if value not in allowed:
        raise RemovalManifestError(f"{label} must be one of {sorted(allowed)}")
    return value


def _bindings(value: Mapping[str, str]) -> dict[str, str]:
    return validate_hash_bindings(value, _BINDING_ROLES, RemovalManifestError)


def _validate_document(document: ScoredDocument) -> None:
    if not is_sha256(document.example_id):
        raise RemovalManifestError("document example_id must be a lowercase SHA-256")
    if type(document.row_ordinal) is not int or document.row_ordinal < 0:
        raise RemovalManifestError("document row_ordinal must be a whole number")
    if type(document.token_count) is not int or document.token_count < 1:
        raise RemovalManifestError("document token_count must be a positive integer")
    if type(document.score) is not float or not math.isfinite(document.score):
        raise RemovalManifestError("document score must be a finite float")


def _coverage(documents: Sequence[ScoredDocument], expected: int) -> None:
    if type(expected) is not int or len(documents) != expected:
        raise RemovalManifestError("document coverage differs from the frozen universe")
    for document in documents:
        _validate_document(document)
    identifiers = {document.example_id for document in documents}
    if len(identifiers) != expected:
        raise RemovalManifestError("document example_id values must be unique")
    ordinals = {document.row_ordinal for document in documents}
    if ordinals != set(range(expected)):
        raise RemovalManifestError(
            "document row_ordinal values must cover the frozen package"
        )


def _top_k(expected: int, removal_percent: int) -> int:
    if removal_percent != 10:
        raise RemovalManifestError("removal_percent is frozen at 10")
    top_k = expected * 10 // 100
    if top_k < 1 or top_k >= expected:
        raise RemovalManifestError("top_k must select a non-empty strict subset")
    return top_k


def _boundary_tie_count(ordered: Sequence[ScoredDocument], top_k: int) -> int:
    boundary = ordered[top_k - 1].score
    if top_k >= len(ordered) or ordered[top_k].score != boundary:
        return 0
    return sum(document.score == boundary for document in ordered)


def _removed_payload(removed: Sequence[ScoredDocument]) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "example_id": document.example_id,
            "row_ordinal": document.row_ordinal,
            "token_count": document.token_count,
            "score": document.score,
        }
        for rank, document in enumerate(removed, start=1)
    ]


def _coverage_payload(
    documents: Sequence[ScoredDocument],
    removed: Sequence[ScoredDocument],
    boundary_tie_count: int,
) -> dict[str, object]:
    return {
        "document_count": len(documents),
        "removed_document_count": len(removed),
        "total_token_count": sum(document.token_count for document in documents),
        "removed_token_count": sum(document.token_count for document in removed),
        "boundary_tie_document_count": boundary_tie_count,
    }


def _manifest_payload(
    behavior: str,
    selector: str,
    bindings: dict[str, str],
    documents: Sequence[ScoredDocument],
    ordered: Sequence[ScoredDocument],
    top_k: int,
) -> dict[str, object]:
    removed = ordered[:top_k]
    return {
        "schema_version": 1,
        "behavior": behavior,
        "selector": selector,
        "sign_convention": "positive_score_is_proponent",
        "ordering": {
            "primary": "score_descending",
            "tie_breaker": "example_id_ascending",
        },
        "top_k_rule": {"removal_percent": 10, "rounding": "floor", "top_k": top_k},
        "coverage": _coverage_payload(
            documents, removed, _boundary_tie_count(ordered, top_k)
        ),
        "bindings": bindings,
        "boundaries": dict(_BOUNDARIES),
        "removed_documents": _removed_payload(removed),
    }


def freeze_removal_manifest(
    *,
    behavior: str,
    selector: str,
    documents: Sequence[ScoredDocument],
    expected_document_count: int,
    removal_percent: int,
    bindings: Mapping[str, str],
) -> RemovalManifest:
    """Freeze one deterministic behavior-specific top-fraction removal manifest."""
    frozen_bindings = _bindings(bindings)
    _coverage(documents, expected_document_count)
    top_k = _top_k(expected_document_count, removal_percent)
    ordered = sorted(documents, key=lambda item: (-item.score, item.example_id))
    payload = _manifest_payload(
        _identifier(behavior, _BEHAVIORS, "behavior"),
        _identifier(selector, _SELECTORS, "selector"),
        frozen_bindings,
        documents,
        ordered,
        top_k,
    )
    return RemovalManifest(MappingProxyType(payload), canonical_sha256(payload))


def removal_manifest_overlap(
    first: RemovalManifest, second: RemovalManifest
) -> dict[str, object]:
    """Report shared removed documents between two frozen manifests."""
    if first.document_universe != second.document_universe:
        raise RemovalManifestError(
            "removal manifests cover different document universes"
        )
    shared = sorted(set(first.removed_example_ids) & set(second.removed_example_ids))
    return {
        "first_manifest_sha256": first.manifest_sha256,
        "second_manifest_sha256": second.manifest_sha256,
        "removed_document_count": len(first.removed_example_ids),
        "shared_removed_document_count": len(shared),
        "shared_removed_example_ids": shared,
    }
