from __future__ import annotations

import math
from typing import cast

import pytest

from concept_filtering.removal_manifest import (
    RemovalManifestError,
    ScoredDocument,
    freeze_removal_manifest,
    removal_manifest_overlap,
)

_BINDINGS = {
    "score_set_sha256": "1" * 64,
    "factor_store_sha256": "2" * 64,
    "training_gradient_store_sha256": "3" * 64,
    "query_gradient_sha256": "4" * 64,
    "training_data_package_sha256": "5" * 64,
}


def _document(index: int, score: float, token_count: int = 10) -> ScoredDocument:
    return ScoredDocument(
        example_id=f"{index:064x}",
        row_ordinal=index,
        token_count=token_count,
        score=score,
    )


def _documents(count: int) -> list[ScoredDocument]:
    return [_document(index, float(count - index)) for index in range(count)]


def _freeze(
    documents: list[ScoredDocument],
    *,
    behavior: str = "validate_feelings",
    selector: str = "ordinary_behavior_query_ekfac",
    expected_document_count: int | None = None,
    removal_percent: int = 10,
    bindings: dict[str, str] | None = None,
):
    count = (
        len(documents) if expected_document_count is None else (expected_document_count)
    )
    return freeze_removal_manifest(
        behavior=behavior,
        selector=selector,
        documents=documents,
        expected_document_count=count,
        removal_percent=removal_percent,
        bindings=dict(_BINDINGS) if bindings is None else bindings,
    )


def test_freezes_deterministic_top_fraction_manifest() -> None:
    manifest = _freeze(_documents(40))

    payload = manifest.to_dict()
    assert payload["behavior"] == "validate_feelings"
    assert payload["selector"] == "ordinary_behavior_query_ekfac"
    assert payload["sign_convention"] == "positive_score_is_proponent"
    assert payload["top_k_rule"] == {
        "removal_percent": 10,
        "rounding": "floor",
        "top_k": 4,
    }
    coverage = payload["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["document_count"] == 40
    assert coverage["removed_document_count"] == 4
    assert coverage["total_token_count"] == 400
    assert coverage["removed_token_count"] == 40
    assert coverage["boundary_tie_document_count"] == 0
    removed = cast(list[dict[str, object]], payload["removed_documents"])
    assert [item["rank"] for item in removed] == [1, 2, 3, 4]
    assert [item["row_ordinal"] for item in removed] == [0, 1, 2, 3]
    assert payload["boundaries"] == {
        "scientific_use": False,
        "starts_experiment_timer": False,
        "training_manifest_promotion_allowed": False,
        "training_execution_allowed": False,
    }
    assert payload["manifest_sha256"] == manifest.manifest_sha256


def test_manifest_is_invariant_to_input_order() -> None:
    documents = _documents(30)
    manifest = _freeze(documents)
    shuffled = list(reversed(documents))

    assert _freeze(shuffled).manifest_sha256 == manifest.manifest_sha256


def test_boundary_ties_break_by_example_id_and_are_reported() -> None:
    documents = _documents(20)
    # Rows 15 (…00f) and 16 (…010) tie at rank 2 and straddle the top_k=2
    # boundary; their hex identifiers order differently when reversed, so
    # only true ascending-example_id ordering selects row 15.
    tied = [
        document
        if document.row_ordinal not in {15, 16}
        else ScoredDocument(
            example_id=document.example_id,
            row_ordinal=document.row_ordinal,
            token_count=document.token_count,
            score=19.5,
        )
        for document in documents
    ]
    manifest = _freeze(tied)

    payload = manifest.to_dict()
    removed = cast(list[dict[str, object]], payload["removed_documents"])
    # The tie at the boundary resolves by ascending example_id.
    assert [item["row_ordinal"] for item in removed] == [0, 15]
    assert [item["rank"] for item in removed] == [1, 2]
    coverage = payload["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["boundary_tie_document_count"] == 2


def test_rejects_incomplete_or_duplicate_coverage() -> None:
    documents = _documents(20)

    with pytest.raises(RemovalManifestError, match="coverage"):
        _freeze(documents[:-1], expected_document_count=20)
    duplicate = [*documents[:-1], documents[0]]
    with pytest.raises(RemovalManifestError, match="example_id"):
        _freeze(duplicate, expected_document_count=20)
    replaced = ScoredDocument(
        example_id=f"{99:064x}", row_ordinal=0, token_count=10, score=1.0
    )
    with pytest.raises(RemovalManifestError, match="row_ordinal"):
        _freeze([*documents[:-1], replaced], expected_document_count=20)


def test_rejects_non_finite_or_non_float_scores() -> None:
    documents = _documents(20)

    broken = [*documents[:-1], _document(19, math.nan)]
    with pytest.raises(RemovalManifestError, match="finite"):
        _freeze(broken)
    coerced = [*documents[:-1], _document(19, math.inf)]
    with pytest.raises(RemovalManifestError, match="finite"):
        _freeze(coerced)


def test_rejects_invalid_identity_and_binding_roles() -> None:
    documents = _documents(20)

    with pytest.raises(RemovalManifestError, match="behavior"):
        _freeze(documents, behavior="sycophancy")
    with pytest.raises(RemovalManifestError, match="selector"):
        _freeze(documents, selector="training_push")
    collided = dict(_BINDINGS)
    collided["factor_store_sha256"] = collided["score_set_sha256"]
    with pytest.raises(RemovalManifestError, match="distinct"):
        _freeze(documents, bindings=collided)
    invalid = dict(_BINDINGS)
    invalid["query_gradient_sha256"] = "not-a-hash"
    with pytest.raises(RemovalManifestError, match="query_gradient_sha256"):
        _freeze(documents, bindings=invalid)
    missing = dict(_BINDINGS)
    del missing["training_data_package_sha256"]
    with pytest.raises(RemovalManifestError, match="training_data_package_sha256"):
        _freeze(documents, bindings=missing)


def test_rejects_invalid_token_counts_and_tiny_universes() -> None:
    documents = _documents(20)

    broken = [*documents[:-1], _document(19, 1.0, token_count=0)]
    with pytest.raises(RemovalManifestError, match="token_count"):
        _freeze(broken)
    with pytest.raises(RemovalManifestError, match="top_k"):
        _freeze(_documents(5))


def test_overlap_reports_shared_removed_documents() -> None:
    manifest = _freeze(_documents(40))
    other_bindings = dict(_BINDINGS)
    other_bindings["score_set_sha256"] = "6" * 64
    other_bindings["query_gradient_sha256"] = "7" * 64
    scores = [
        ScoredDocument(
            example_id=document.example_id,
            row_ordinal=document.row_ordinal,
            token_count=document.token_count,
            score=float(document.row_ordinal in {2, 3, 20, 21}),
        )
        for document in _documents(40)
    ]
    other = _freeze(scores, behavior="refusal", bindings=other_bindings)

    overlap = removal_manifest_overlap(manifest, other)
    assert overlap == {
        "first_manifest_sha256": manifest.manifest_sha256,
        "second_manifest_sha256": other.manifest_sha256,
        "removed_document_count": 4,
        "shared_removed_document_count": 2,
        "shared_removed_example_ids": [f"{2:064x}", f"{3:064x}"],
    }


def test_overlap_rejects_mismatched_document_universes() -> None:
    manifest = _freeze(_documents(40))
    other = _freeze(_documents(30))

    with pytest.raises(RemovalManifestError, match="universe"):
        removal_manifest_overlap(manifest, other)
