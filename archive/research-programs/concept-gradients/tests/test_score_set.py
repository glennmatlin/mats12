from __future__ import annotations

from typing import cast

import pytest

from concept_filtering.removal_manifest import freeze_removal_manifest
from concept_filtering.score_set import (
    ScoreRecord,
    ScoreSetError,
    freeze_score_set,
    manifest_inputs,
)

_BINDINGS = {
    "factor_store_sha256": "1" * 64,
    "training_gradient_store_sha256": "2" * 64,
    "query_gradient_sha256": "3" * 64,
    "training_data_package_sha256": "4" * 64,
    "backend_runtime_sha256": "5" * 64,
}


def _expected(count: int) -> dict[str, int]:
    return {f"{index:064x}": index for index in range(count)}


def _record(index: int, score: float) -> ScoreRecord:
    return ScoreRecord(
        example_id=f"{index:064x}",
        row_ordinal=index,
        token_count=10,
        score=score,
    )


def _records(count: int) -> list[ScoreRecord]:
    return [_record(index, float(count - index)) for index in range(count)]


def test_freezes_complete_behavior_score_set() -> None:
    records = _records(20)
    receipt = freeze_score_set(
        records,
        behavior="validate_feelings",
        expected=_expected(20),
        bindings=dict(_BINDINGS),
    )

    payload = receipt.to_dict()
    assert payload["behavior"] == "validate_feelings"
    assert payload["sign_convention"] == "positive_score_is_proponent"
    coverage = cast(dict[str, object], payload["coverage"])
    assert coverage["example_count"] == 20
    assert coverage["total_token_count"] == 200
    assert payload["score_set_sha256"] == receipt.score_set_sha256
    shuffled = list(reversed(records))
    replay = freeze_score_set(
        shuffled,
        behavior="validate_feelings",
        expected=_expected(20),
        bindings=dict(_BINDINGS),
    )
    assert replay.score_set_sha256 == receipt.score_set_sha256


def test_rejects_incomplete_duplicate_or_invalid_scores() -> None:
    records = _records(20)

    with pytest.raises(ScoreSetError, match="pending"):
        freeze_score_set(
            records[:-1],
            behavior="refusal",
            expected=_expected(20),
            bindings=dict(_BINDINGS),
        )
    conflicting = [*records, _record(0, -3.5)]
    with pytest.raises(ScoreSetError, match="conflict"):
        freeze_score_set(
            conflicting,
            behavior="refusal",
            expected=_expected(20),
            bindings=dict(_BINDINGS),
        )
    with pytest.raises(ScoreSetError, match="finite"):
        freeze_score_set(
            [*records[:-1], _record(19, float("nan"))],
            behavior="refusal",
            expected=_expected(20),
            bindings=dict(_BINDINGS),
        )
    with pytest.raises(ScoreSetError, match="behavior"):
        freeze_score_set(
            records,
            behavior="sycophancy",
            expected=_expected(20),
            bindings=dict(_BINDINGS),
        )


def test_manifest_inputs_reject_forged_receipt_identity() -> None:
    from concept_filtering.score_set import ScoreSetReceipt

    receipt = freeze_score_set(
        _records(20),
        behavior="validate_feelings",
        expected=_expected(20),
        bindings=dict(_BINDINGS),
    )
    forged = ScoreSetReceipt(receipt.payload, "f" * 64)

    with pytest.raises(ScoreSetError, match="identity"):
        manifest_inputs(forged)


def test_manifest_inputs_bind_the_frozen_score_set() -> None:
    receipt = freeze_score_set(
        _records(40),
        behavior="validate_feelings",
        expected=_expected(40),
        bindings=dict(_BINDINGS),
    )

    documents, bindings, count = manifest_inputs(receipt)
    manifest = freeze_removal_manifest(
        behavior="validate_feelings",
        selector="ordinary_behavior_query_ekfac",
        documents=documents,
        expected_document_count=count,
        removal_percent=10,
        bindings=bindings,
    )

    payload = manifest.to_dict()
    manifest_bindings = cast(dict[str, str], payload["bindings"])
    assert manifest_bindings["score_set_sha256"] == receipt.score_set_sha256
    assert (
        manifest_bindings["factor_store_sha256"] == (_BINDINGS["factor_store_sha256"])
    )
    assert (
        manifest_bindings["training_data_package_sha256"]
        == (_BINDINGS["training_data_package_sha256"])
    )
    coverage = cast(dict[str, object], payload["coverage"])
    assert coverage["document_count"] == 40
    assert coverage["removed_document_count"] == 4
