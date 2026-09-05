from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from concept_filtering.generation_evidence import (
    build_generation_evidence,
    load_generation_evidence,
)
from concept_filtering.scoring_decision import (
    ScoringDecisionError,
    ScoringDecisionEvidence,
    load_scoring_decision,
)
from concept_filtering.serialization import canonical_sha256, file_sha256

ROOT = Path(__file__).resolve().parents[1]
DECISION = (
    ROOT
    / "configs/reconstruction/evaluators"
    / "public-rank32-r2-scoring-decision-v1.json"
)
RECEIPT = (
    ROOT
    / "configs/reconstruction/receipts"
    / "public-rank32-r2-scoring-decision-proposal-v1.json"
)
GENERATION_EVIDENCE = (
    ROOT
    / "configs/reconstruction/receipts"
    / "public-rank32-full-bank-generation-evidence-v1.json"
)


def evidence() -> ScoringDecisionEvidence:
    reconstruction = ROOT / "reports/reconstruction"
    references = reconstruction / "reference_scores"
    return ScoringDecisionEvidence(
        evaluator_config=ROOT
        / "configs/reconstruction/evaluators"
        / "public-slop-aware-judge-reconstruction-v1.json",
        evaluator_receipt=reconstruction
        / "evaluators/2026-08-15-public-slop-aware-judge-reconstruction.json",
        generation_calibration=reconstruction
        / "calibrations/2026-08-14-rank32-local-generation.json",
        generation_evidence=GENERATION_EVIDENCE,
        reference_receipts={
            "validate_feelings": references
            / "validate-feelings.extraction-receipt.json",
            "refusal": references / "refusal.extraction-receipt.json",
        },
        reference_scores={
            "validate_feelings": references / "validate-feelings.jsonl",
            "refusal": references / "refusal.jsonl",
        },
    )


def test_loads_closed_proposed_scoring_decision() -> None:
    decision = load_scoring_decision(
        DECISION, receipt_path=RECEIPT, evidence=evidence()
    )

    assert decision.decision_id == "public-rank32-r2-scoring-decision-v1"
    assert decision.owner_accepted is False
    assert decision.provider_execution_allowed is False
    assert decision.maximum_difference == 0.5
    assert decision.response_sample_count == 201


def test_retained_generation_chain_rebuilds_checked_in_evidence() -> None:
    output_dir = ROOT / ".runs/rank32-full-v1"
    if not output_dir.exists():
        pytest.skip("retained local generation output is intentionally ignored")
    expected = load_generation_evidence(GENERATION_EVIDENCE)
    observed = build_generation_evidence(
        Path(".runs/rank32-full-v1"),
        evidence_id="public-rank32-full-bank-generation-evidence-v1",
    )

    assert observed == expected


def _write_bound_decision(tmp_path: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_body = {
        "schema_version": 1,
        "decision_id": payload.get("decision_id"),
        "decision_file_sha256": file_sha256(decision_path),
        "decision_canonical_sha256": canonical_sha256(payload),
        "scientific_use": False,
        "starts_experiment_timer": False,
        "provider_execution_allowed": False,
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return decision_path, receipt_path


def _write_receipt_for_decision(
    tmp_path: Path, decision_path: Path, payload: dict[str, Any]
) -> Path:
    receipt_body = {
        "schema_version": 1,
        "decision_id": payload.get("decision_id"),
        "decision_file_sha256": file_sha256(decision_path),
        "decision_canonical_sha256": canonical_sha256(payload),
        "scientific_use": False,
        "starts_experiment_timer": False,
        "provider_execution_allowed": False,
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    receipt_path = tmp_path / "receipt.json"
    _write_json(receipt_path, receipt)
    return receipt_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _bind_hash(payload: dict[str, Any], field: str) -> None:
    payload.pop(field, None)
    payload[field] = canonical_sha256(payload)


def _set_nested(payload: dict[str, Any], path: str, value: object) -> None:
    fields = path.split(".")
    target = payload
    for field in fields[:-1]:
        target = target[field]
    target[fields[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("decision_id", "other", "decision_id"),
        ("decision_status", "accepted", "decision_status"),
        ("owner_accepted", True, "owner_accepted"),
        ("scientific_use", True, "scientific_use"),
        (
            "judge_identity.evaluator_reconstruction_config_sha256",
            "0" * 64,
            "evaluator config",
        ),
        ("response_bank.ordered_samples_sha256", "0" * 64, "response bank"),
        ("reference_scores.validate_feelings.mean_score", 0, "reference score"),
        ("acceptance_rule.maximum_difference", 0.6, "acceptance_rule"),
        ("acceptance_rule.all_behaviors_must_pass", 1, "acceptance_rule"),
        (
            "owner_decision_required",
            "Accept or reject before scoring.",
            "owner_decision_required",
        ),
    ],
)
def test_rejects_mutated_closed_policy(
    tmp_path: Path, path: str, value: object, message: str
) -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    _set_nested(payload, path, value)
    decision_path, receipt_path = _write_bound_decision(tmp_path, payload)

    with pytest.raises(ScoringDecisionError, match=message):
        load_scoring_decision(
            decision_path, receipt_path=receipt_path, evidence=evidence()
        )


def test_rejects_unbound_policy_field(tmp_path: Path) -> None:
    payload = json.loads(DECISION.read_text(encoding="utf-8"))
    payload["new_policy"] = "unbound"
    decision_path, receipt_path = _write_bound_decision(tmp_path, payload)

    with pytest.raises(ScoringDecisionError, match="unexpected fields"):
        load_scoring_decision(
            decision_path, receipt_path=receipt_path, evidence=evidence()
        )


def test_rejects_mutated_decision_receipt(tmp_path: Path) -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    receipt["decision_file_sha256"] = "0" * 64
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ScoringDecisionError, match="receipt"):
        load_scoring_decision(DECISION, receipt_path=receipt_path, evidence=evidence())


def test_rejects_mutated_calibration_evidence(tmp_path: Path) -> None:
    copied = evidence()
    calibration = json.loads(copied.generation_calibration.read_text(encoding="utf-8"))
    calibration["backend"] = "mutated"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration), encoding="utf-8")
    mutated = replace(copied, generation_calibration=path)

    with pytest.raises(ScoringDecisionError, match="calibration"):
        load_scoring_decision(DECISION, receipt_path=RECEIPT, evidence=mutated)


def test_rejects_mutated_generation_receipt_evidence(tmp_path: Path) -> None:
    copied = evidence()
    generation = json.loads(GENERATION_EVIDENCE.read_text(encoding="utf-8"))
    generation["sample_receipts"][0]["sample_sha256"] = "0" * 64
    _bind_hash(generation, "evidence_sha256")
    generation_path = tmp_path / "generation-evidence.json"
    _write_json(generation_path, generation)

    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    response_bank = decision["response_bank"]
    response_bank["generation_evidence_file_sha256"] = file_sha256(generation_path)
    response_bank["generation_evidence_sha256"] = generation["evidence_sha256"]
    decision_path, receipt_path = _write_bound_decision(tmp_path, decision)

    with pytest.raises(ScoringDecisionError, match="generation evidence"):
        load_scoring_decision(
            decision_path,
            receipt_path=receipt_path,
            evidence=replace(copied, generation_evidence=generation_path),
        )


def test_rejects_fully_rebound_float_generation_count(tmp_path: Path) -> None:
    copied = evidence()
    generation = json.loads(GENERATION_EVIDENCE.read_text(encoding="utf-8"))
    generation["sample_count"] = 201.0
    _bind_hash(generation, "evidence_sha256")
    generation_path = tmp_path / "generation-evidence.json"
    _write_json(generation_path, generation)

    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    response_bank = decision["response_bank"]
    response_bank["generation_evidence_file_sha256"] = file_sha256(generation_path)
    response_bank["generation_evidence_sha256"] = generation["evidence_sha256"]
    decision_path, receipt_path = _write_bound_decision(tmp_path, decision)

    with pytest.raises(ScoringDecisionError, match="generation evidence"):
        load_scoring_decision(
            decision_path,
            receipt_path=receipt_path,
            evidence=replace(copied, generation_evidence=generation_path),
        )


def test_rejects_duplicate_decision_safety_key(tmp_path: Path) -> None:
    source = DECISION.read_text(encoding="utf-8")
    ambiguous = source.replace(
        '  "provider_execution_allowed": false,',
        '  "provider_execution_allowed": true,\n  "provider_execution_allowed": false,',
        1,
    )
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(ambiguous, encoding="utf-8")
    payload = json.loads(ambiguous)
    receipt_path = _write_receipt_for_decision(tmp_path, decision_path, payload)

    with pytest.raises(ScoringDecisionError, match="duplicate"):
        load_scoring_decision(
            decision_path, receipt_path=receipt_path, evidence=evidence()
        )


def test_rejects_duplicate_generation_safety_key(tmp_path: Path) -> None:
    source = GENERATION_EVIDENCE.read_text(encoding="utf-8")
    ambiguous = source.replace(
        '  "scientific_use": false,',
        '  "scientific_use": true,\n  "scientific_use": false,',
        1,
    )
    generation_path = tmp_path / "generation-evidence.json"
    generation_path.write_text(ambiguous, encoding="utf-8")
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    decision["response_bank"]["generation_evidence_file_sha256"] = file_sha256(
        generation_path
    )
    decision_path, receipt_path = _write_bound_decision(tmp_path, decision)

    with pytest.raises(ScoringDecisionError, match="duplicate"):
        load_scoring_decision(
            decision_path,
            receipt_path=receipt_path,
            evidence=replace(evidence(), generation_evidence=generation_path),
        )


def test_rejects_mutated_evaluator_evidence(tmp_path: Path) -> None:
    copied = evidence()
    path = tmp_path / "evaluator.json"
    path.write_text(copied.evaluator_config.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ScoringDecisionError, match="evaluator config"):
        load_scoring_decision(
            DECISION,
            receipt_path=RECEIPT,
            evidence=replace(copied, evaluator_config=path),
        )


def test_rejects_self_consistent_evaluator_safety_contradiction(
    tmp_path: Path,
) -> None:
    copied = evidence()
    config = json.loads(copied.evaluator_config.read_text(encoding="utf-8"))
    config.update(
        scientific_use=True,
        provider_execution_allowed=True,
        judge_model_weights_and_config_fixed=False,
        serving_infrastructure_fixed=True,
    )
    config_path = tmp_path / "evaluator.json"
    _write_json(config_path, config)

    evaluator_receipt = json.loads(copied.evaluator_receipt.read_text(encoding="utf-8"))
    evaluator_receipt.update(
        config_sha256=file_sha256(config_path),
        scientific_use=True,
        provider_execution_allowed=True,
        judge_model_weights_and_config_fixed=False,
        serving_infrastructure_fixed=True,
    )
    _bind_hash(evaluator_receipt, "receipt_sha256")
    evaluator_receipt_path = tmp_path / "evaluator-receipt.json"
    _write_json(evaluator_receipt_path, evaluator_receipt)

    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    identity = decision["judge_identity"]
    identity["evaluator_reconstruction_config_sha256"] = file_sha256(config_path)
    identity["evaluator_reconstruction_receipt_sha256"] = evaluator_receipt[
        "receipt_sha256"
    ]
    decision_path, receipt_path = _write_bound_decision(tmp_path, decision)
    mutated = replace(
        copied,
        evaluator_config=config_path,
        evaluator_receipt=evaluator_receipt_path,
    )

    with pytest.raises(ScoringDecisionError, match="evaluator config"):
        load_scoring_decision(
            decision_path, receipt_path=receipt_path, evidence=mutated
        )


def test_rejects_coordinated_two_response_calibration(tmp_path: Path) -> None:
    copied = evidence()
    calibration = json.loads(copied.generation_calibration.read_text(encoding="utf-8"))
    run = calibration["transformers_mps_explicit_eos_full_bank_persistent"]
    run["sample_count"] = 2
    run["validate_feelings"]["sample_count"] = 1
    run["refusal"]["sample_count"] = 1
    _bind_hash(calibration, "calibration_sha256")
    calibration_path = tmp_path / "calibration.json"
    _write_json(calibration_path, calibration)

    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    response_bank = decision["response_bank"]
    response_bank["calibration_file_sha256"] = file_sha256(calibration_path)
    response_bank["calibration_sha256"] = calibration["calibration_sha256"]
    response_bank["sample_count"] = 2
    response_bank["behaviors"]["validate_feelings"]["sample_count"] = 1
    response_bank["behaviors"]["refusal"]["sample_count"] = 1
    decision_path, receipt_path = _write_bound_decision(tmp_path, decision)

    with pytest.raises(ScoringDecisionError, match="response count"):
        load_scoring_decision(
            decision_path,
            receipt_path=receipt_path,
            evidence=replace(copied, generation_calibration=calibration_path),
        )


def test_rejects_mutated_reference_output(tmp_path: Path) -> None:
    copied = evidence()
    path = tmp_path / "validate.jsonl"
    path.write_text(
        copied.reference_scores["validate_feelings"].read_text() + "\n",
        encoding="utf-8",
    )
    outputs = {**copied.reference_scores, "validate_feelings": path}

    with pytest.raises(ScoringDecisionError, match="reference evidence"):
        load_scoring_decision(
            DECISION,
            receipt_path=RECEIPT,
            evidence=replace(copied, reference_scores=outputs),
        )


def test_rejects_mutated_reference_receipt(tmp_path: Path) -> None:
    copied = evidence()
    receipt = json.loads(
        copied.reference_receipts["refusal"].read_text(encoding="utf-8")
    )
    receipt["mean_score"] = 0
    path = tmp_path / "refusal-receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    receipts = {**copied.reference_receipts, "refusal": path}

    with pytest.raises(ScoringDecisionError, match="reference evidence"):
        load_scoring_decision(
            DECISION,
            receipt_path=RECEIPT,
            evidence=replace(copied, reference_receipts=receipts),
        )
