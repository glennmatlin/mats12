from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

import concept_filtering.r2_provider_scoring as r2_provider_scoring
from concept_filtering.serialization import canonical_sha256, file_sha256

DECISION_BODY: dict[str, object] = {
    "schema_version": 1,
    "decision_id": "fixture-r2-scoring-decision-v1",
    "decision_status": "proposed",
    "owner_accepted": False,
}
RECEIPT_BODY: dict[str, object] = {
    "schema_version": 1,
    "decision_id": "fixture-r2-scoring-decision-v1",
    "status": "proposed",
}


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(DECISION_BODY), encoding="utf-8")
    receipt_path = tmp_path / "decision-receipt.json"
    receipt_payload = {**RECEIPT_BODY, "receipt_sha256": canonical_sha256(RECEIPT_BODY)}
    receipt_path.write_text(json.dumps(receipt_payload), encoding="utf-8")
    acceptance: dict[str, object] = {
        "schema_version": 1,
        "acceptance_id": "fixture-r2-owner-acceptance-v1",
        "decision_id": "fixture-r2-scoring-decision-v1",
        "decision_file_sha256": file_sha256(decision_path),
        "decision_canonical_sha256": canonical_sha256(DECISION_BODY),
        "decision_receipt_sha256": canonical_sha256(RECEIPT_BODY),
        "owner_accepted": True,
        "accepted_on": "2026-08-28",
        "acceptance_record": (
            "https://github.com/glennmatlin/mats12-concept-filtering/issues/2"
            "#issuecomment-5458281754"
        ),
        "scientific_use": False,
        "starts_experiment_timer": False,
        "provider_execution_allowed_for": "fixture-r2-provider-scoring-v1",
        "limitations": ["Fixture acceptance for exactly one scoring execution."],
    }
    return decision_path, receipt_path, acceptance


def _load(
    tmp_path: Path, acceptance: dict[str, object], decision: Path, receipt: Path
) -> Any:
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(acceptance), encoding="utf-8")
    return r2_provider_scoring.load_r2_owner_acceptance(
        path, decision_path=decision, decision_receipt_path=receipt
    )


def test_loads_valid_acceptance(tmp_path: Path) -> None:
    decision, receipt, acceptance = _write_fixture(tmp_path)

    loaded = _load(tmp_path, acceptance, decision, receipt)

    assert loaded.acceptance_id == "fixture-r2-owner-acceptance-v1"
    assert loaded.decision_id == "fixture-r2-scoring-decision-v1"
    assert loaded.accepted_on == "2026-08-28"
    assert loaded.provider_execution_allowed_for == "fixture-r2-provider-scoring-v1"
    assert loaded.acceptance_sha256 == canonical_sha256(acceptance)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("owner_accepted", False, "owner_accepted"),
        ("scientific_use", True, "scientific_use"),
        ("starts_experiment_timer", True, "starts_experiment_timer"),
        ("decision_file_sha256", "0" * 64, "decision file"),
        ("decision_canonical_sha256", "0" * 64, "decision canonical"),
        ("decision_receipt_sha256", "0" * 64, "decision receipt"),
        ("decision_id", "some-other-decision", "decision_id"),
        ("accepted_on", "sometime", "accepted_on"),
        ("acceptance_record", "", "acceptance_record"),
    ],
)
def test_rejects_tampered_acceptance(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    decision, receipt, acceptance = _write_fixture(tmp_path)
    acceptance[field] = value

    with pytest.raises(r2_provider_scoring.R2ProviderScoringError, match=match):
        _load(tmp_path, acceptance, decision, receipt)


def test_rejects_unknown_or_missing_fields(tmp_path: Path) -> None:
    decision, receipt, acceptance = _write_fixture(tmp_path)
    extra = copy.deepcopy(acceptance)
    extra["note"] = "surprise"
    with pytest.raises(r2_provider_scoring.R2ProviderScoringError, match="fields"):
        _load(tmp_path, extra, decision, receipt)

    missing = copy.deepcopy(acceptance)
    del missing["limitations"]
    with pytest.raises(r2_provider_scoring.R2ProviderScoringError, match="fields"):
        _load(tmp_path, missing, decision, receipt)


def test_rejects_drifted_decision_file(tmp_path: Path) -> None:
    decision, receipt, acceptance = _write_fixture(tmp_path)
    drifted = dict(DECISION_BODY)
    drifted["decision_status"] = "accepted"
    decision.write_text(json.dumps(drifted), encoding="utf-8")

    with pytest.raises(
        r2_provider_scoring.R2ProviderScoringError, match="decision file"
    ):
        _load(tmp_path, acceptance, decision, receipt)


def test_rejects_receipt_with_inconsistent_signature(tmp_path: Path) -> None:
    decision, receipt, acceptance = _write_fixture(tmp_path)
    payload = cast(dict[str, object], json.loads(receipt.read_text(encoding="utf-8")))
    payload["status"] = "amended"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        r2_provider_scoring.R2ProviderScoringError, match="decision receipt"
    ):
        _load(tmp_path, acceptance, decision, receipt)
