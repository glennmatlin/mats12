"""Fail-closed validation for the proposed Gate R2 scoring decision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.evaluator_reconstruction import (
    EvaluatorReconstruction,
    EvaluatorReconstructionError,
    load_evaluator_reconstruction,
)
from concept_filtering.generation_evidence import (
    GenerationEvidenceError,
    load_generation_evidence,
)
from concept_filtering.inspect_archive import (
    InspectArchiveValidationError,
    load_extraction_receipt,
)
from concept_filtering.serialization import (
    canonical_sha256,
    file_sha256,
    json_equal_exact,
    json_loads_unique,
)

_DECISION_ID = "public-rank32-r2-scoring-decision-v1"
_DECISION_FIELDS = {
    "schema_version",
    "decision_id",
    "decision_status",
    "owner_accepted",
    "scientific_use",
    "starts_experiment_timer",
    "provider_execution_allowed",
    "judge_identity",
    "response_bank",
    "reference_scores",
    "acceptance_rule",
    "rationale",
    "known_divergences",
    "owner_decision_required",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "decision_id",
    "decision_file_sha256",
    "decision_canonical_sha256",
    "scientific_use",
    "starts_experiment_timer",
    "provider_execution_allowed",
    "receipt_sha256",
}
_ACCEPTANCE_RULE = {
    "metric": "absolute_arithmetic_mean_difference",
    "maximum_difference": 0.5,
    "all_behaviors_must_pass": True,
    "allowed_missing_scores": 0,
    "allowed_parse_failures": 0,
    "successful_judge_samples_per_response": 1,
    "judge_resampling_after_parseable_completion": False,
    "post_result_threshold_changes_allowed": False,
}
_OWNER_DECISION_REQUIRED = (
    "Explicit owner acceptance of this complete policy is required before any "
    "provider scoring. Rejection keeps provider execution prohibited. Acceptance "
    "does not by itself start the 20-hour experiment timer."
)
_EXPECTED_RESPONSE_COUNTS = {"validate_feelings": 100, "refusal": 101}


class ScoringDecisionError(ValueError):
    """Raised when a scoring decision or its evidence does not match."""


@dataclass(frozen=True)
class ScoringDecisionEvidence:
    evaluator_config: Path
    evaluator_receipt: Path
    generation_calibration: Path
    generation_evidence: Path
    reference_receipts: dict[str, Path]
    reference_scores: dict[str, Path]


@dataclass(frozen=True)
class ScoringDecision:
    decision_id: str
    owner_accepted: bool
    provider_execution_allowed: bool
    maximum_difference: float
    response_sample_count: int


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScoringDecisionError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        return _mapping(json_loads_unique(path.read_text(encoding="utf-8")), field)
    except (OSError, UnicodeError, ValueError) as error:
        raise ScoringDecisionError(f"cannot load {field}: {error}") from error


def _exact_fields(payload: dict[str, Any], expected: set[str], field: str) -> None:
    unexpected = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unexpected or missing:
        raise ScoringDecisionError(
            f"{field} unexpected fields={unexpected}, missing fields={missing}"
        )


def _must_be_false(payload: dict[str, Any], field: str) -> None:
    if payload.get(field) is not False:
        raise ScoringDecisionError(f"{field} must remain false")


def _validate_envelope(payload: dict[str, Any]) -> None:
    _exact_fields(payload, _DECISION_FIELDS, "decision")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ScoringDecisionError("schema_version must be 1")
    if payload.get("decision_id") != _DECISION_ID:
        raise ScoringDecisionError("decision_id does not match")
    if payload.get("decision_status") != "proposed":
        raise ScoringDecisionError("decision_status must remain proposed")
    for field in (
        "owner_accepted",
        "scientific_use",
        "starts_experiment_timer",
        "provider_execution_allowed",
    ):
        _must_be_false(payload, field)
    if not isinstance(payload.get("rationale"), str):
        raise ScoringDecisionError("rationale must be a string")
    if payload.get("owner_decision_required") != _OWNER_DECISION_REQUIRED:
        raise ScoringDecisionError("owner_decision_required does not match")


def _validate_bound_hash(payload: dict[str, Any], field: str) -> None:
    body = dict(payload)
    stored = body.pop(field, None)
    if not isinstance(stored, str) or canonical_sha256(body) != stored:
        raise ScoringDecisionError(f"{field} does not match receipt body")


def _validate_decision_receipt(
    decision: dict[str, Any], decision_path: Path, receipt_path: Path
) -> None:
    receipt = _load_json(receipt_path, "decision receipt")
    _exact_fields(receipt, _RECEIPT_FIELDS, "decision receipt")
    _validate_bound_hash(receipt, "receipt_sha256")
    expected = (
        1,
        _DECISION_ID,
        file_sha256(decision_path),
        canonical_sha256(decision),
        False,
        False,
        False,
    )
    ordered_fields = (
        "schema_version",
        "decision_id",
        "decision_file_sha256",
        "decision_canonical_sha256",
        "scientific_use",
        "starts_experiment_timer",
        "provider_execution_allowed",
    )
    observed = [receipt.get(field) for field in ordered_fields]
    if not json_equal_exact(observed, list(expected)):
        raise ScoringDecisionError("decision receipt does not bind proposal")


def _load_evaluator(path: Path) -> EvaluatorReconstruction:
    _load_json(path, "evaluator config")
    try:
        return load_evaluator_reconstruction(path)
    except EvaluatorReconstructionError as error:
        raise ScoringDecisionError(
            f"evaluator config does not match: {error}"
        ) from error


def _evaluator_receipt_body(config: EvaluatorReconstruction) -> dict[str, object]:
    archives = [
        {
            "behavior": behavior,
            "archive_sha256": spec.source_archive_sha256,
            "sample_count": spec.expected_samples,
            "source_commit": spec.source_commit,
            "source_dirty": True,
            "task_file": spec.task_file,
        }
        for behavior, spec in sorted(config.behaviors.items())
    ]
    sample_count = sum(spec.expected_samples for spec in config.behaviors.values())
    return {
        "schema_version": 1,
        "evaluator_id": config.evaluator_id,
        "config_sha256": config.config_sha256,
        "judge_model_revision": config.judge_model_revision,
        "judge_model_revision_source": config.judge_model_revision_source,
        "judge_model_weights_and_config_fixed": True,
        "serving_infrastructure_fixed": False,
        "public_source_sha256": config.public_scorer_sha256,
        "archives": archives,
        "sample_count": sample_count,
        "request_match_count": sample_count,
        "strict_parse_match_count": sample_count,
        "scientific_use": False,
        "starts_experiment_timer": False,
        "provider_execution_allowed": False,
    }


def _validate_evaluator_receipt(
    receipt: dict[str, Any], config: EvaluatorReconstruction
) -> None:
    _validate_bound_hash(receipt, "receipt_sha256")
    body = {
        field: value for field, value in receipt.items() if field != "receipt_sha256"
    }
    if not json_equal_exact(body, _evaluator_receipt_body(config)):
        raise ScoringDecisionError("evaluator config and receipt do not match")


def _validate_judge_identity(
    identity: dict[str, Any], evidence: ScoringDecisionEvidence
) -> None:
    config = _load_evaluator(evidence.evaluator_config)
    receipt = _load_json(evidence.evaluator_receipt, "evaluator receipt")
    _validate_evaluator_receipt(receipt, config)
    expected = {
        "inspect_model_name": config.judge_model,
        "provider_model_id": config.judge_model_revision,
        "model_version_policy_url": config.judge_model_revision_source,
        "weights_and_configuration_fixed": config.judge_model_weights_and_config_fixed,
        "serving_infrastructure_fixed": config.serving_infrastructure_fixed,
        "evaluator_reconstruction_config_sha256": config.config_sha256,
        "evaluator_reconstruction_receipt_sha256": receipt.get("receipt_sha256"),
    }
    if not json_equal_exact(identity, expected):
        raise ScoringDecisionError("evaluator config or judge identity does not match")


def _response_behaviors(run: dict[str, Any]) -> dict[str, object]:
    return {
        behavior: {
            "sample_count": _mapping(run.get(behavior), behavior).get("sample_count"),
            "completion_hashes_sha256": _mapping(run.get(behavior), behavior).get(
                "completion_hashes_sha256"
            ),
        }
        for behavior in ("validate_feelings", "refusal")
    }


def _response_binding(
    calibration: dict[str, Any], file_hash: str, generation: dict[str, Any]
) -> dict[str, object]:
    run = _mapping(
        calibration.get("transformers_mps_explicit_eos_full_bank_persistent"),
        "persistent response bank",
    )
    return {
        "generation_id": "public-rank32-pinned-prompt-banks-v1",
        "executor_revision": run.get("executor_revision"),
        "local_output_directory": run.get("local_output_directory"),
        "calibration_file_sha256": file_hash,
        "calibration_sha256": calibration.get("calibration_sha256"),
        "sample_count": run.get("sample_count"),
        "aggregate_receipt_sha256": run.get("receipt_sha256"),
        "aggregate_receipt_file_sha256": run.get("receipt_file_sha256"),
        "run_sha256": run.get("run_sha256"),
        "run_file_sha256": run.get("run_file_sha256"),
        "runtime_sha256": run.get("runtime_sha256"),
        "runtime_file_sha256": run.get("runtime_file_sha256"),
        "ordered_samples_sha256": run.get("samples_sha256"),
        "completion_hashes_sha256": run.get("completion_hashes_sha256"),
        "behaviors": _response_behaviors(run),
        "generation_evidence_id": generation.get("evidence_id"),
        "generation_evidence_file_sha256": generation.get("file_sha256"),
        "generation_evidence_sha256": generation.get("evidence_sha256"),
    }


def _load_generation(path: Path) -> dict[str, Any]:
    try:
        payload = load_generation_evidence(path)
    except GenerationEvidenceError as error:
        raise ScoringDecisionError(
            f"generation evidence does not match: {error}"
        ) from error
    return {**payload, "file_sha256": file_sha256(path)}


def _generation_chain(run: dict[str, Any]) -> dict[str, object]:
    return {
        "generation_id": "public-rank32-pinned-prompt-banks-v1",
        "output_directory": run.get("local_output_directory"),
        "executor_revision": run.get("executor_revision"),
        "run_sha256": run.get("run_sha256"),
        "run_file_sha256": run.get("run_file_sha256"),
        "runtime_sha256": run.get("runtime_sha256"),
        "runtime_file_sha256": run.get("runtime_file_sha256"),
        "aggregate_receipt_sha256": run.get("receipt_sha256"),
        "aggregate_receipt_file_sha256": run.get("receipt_file_sha256"),
        "sample_count": run.get("sample_count"),
        "samples_sha256": run.get("samples_sha256"),
        "completion_hashes_sha256": run.get("completion_hashes_sha256"),
        "behaviors": _response_behaviors(run),
        "scientific_use": False,
        "starts_experiment_timer": False,
    }


def _validate_generation_chain(generation: dict[str, Any], run: dict[str, Any]) -> None:
    fields = set(_generation_chain(run))
    observed = {field: generation.get(field) for field in fields}
    if not json_equal_exact(observed, _generation_chain(run)):
        raise ScoringDecisionError("generation evidence does not match calibration")


def _validate_response_bank(
    response_bank: dict[str, Any], evidence: ScoringDecisionEvidence
) -> None:
    calibration = _load_json(evidence.generation_calibration, "calibration")
    _validate_bound_hash(calibration, "calibration_sha256")
    file_hash = file_sha256(evidence.generation_calibration)
    run = _mapping(
        calibration.get("transformers_mps_explicit_eos_full_bank_persistent"),
        "persistent response bank",
    )
    behavior_counts = {
        behavior: _mapping(run.get(behavior), behavior).get("sample_count")
        for behavior in _EXPECTED_RESPONSE_COUNTS
    }
    if not json_equal_exact(behavior_counts, _EXPECTED_RESPONSE_COUNTS):
        raise ScoringDecisionError("response count does not match frozen banks")
    if type(run.get("sample_count")) is not int or run["sample_count"] != 201:
        raise ScoringDecisionError("response count does not match frozen banks")
    generation = _load_generation(evidence.generation_evidence)
    _validate_generation_chain(generation, run)
    expected = _response_binding(calibration, file_hash, generation)
    if not json_equal_exact(response_bank, expected):
        raise ScoringDecisionError("response bank does not match calibration")


def _reference_binding(receipt_path: Path, score_path: Path) -> dict[str, object]:
    _load_json(receipt_path, "reference receipt")
    try:
        receipt = load_extraction_receipt(receipt_path, output_path=score_path)
    except InspectArchiveValidationError as error:
        raise ScoringDecisionError(
            f"reference evidence does not match: {error}"
        ) from error
    return {
        "question_count": receipt.question_count,
        "mean_score": receipt.mean_score,
        "score_output_sha256": receipt.output_sha256,
        "extraction_receipt_sha256": receipt.receipt_sha256,
    }


def _validate_reference_scores(
    references: dict[str, Any], evidence: ScoringDecisionEvidence
) -> None:
    behaviors = {"validate_feelings", "refusal"}
    if set(references) != behaviors:
        raise ScoringDecisionError("reference score behaviors do not match")
    if set(evidence.reference_receipts) != behaviors:
        raise ScoringDecisionError("reference receipt behaviors do not match")
    if set(evidence.reference_scores) != behaviors:
        raise ScoringDecisionError("reference output behaviors do not match")
    for behavior in sorted(behaviors):
        expected = _reference_binding(
            evidence.reference_receipts[behavior], evidence.reference_scores[behavior]
        )
        if not json_equal_exact(references.get(behavior), expected):
            raise ScoringDecisionError(f"{behavior} reference score does not match")


def load_scoring_decision(
    path: Path, *, receipt_path: Path, evidence: ScoringDecisionEvidence
) -> ScoringDecision:
    payload = _load_json(path, "scoring decision")
    _validate_envelope(payload)
    _validate_decision_receipt(payload, path, receipt_path)
    _validate_judge_identity(_mapping(payload.get("judge_identity"), "judge"), evidence)
    response_bank = _mapping(payload.get("response_bank"), "response_bank")
    _validate_response_bank(response_bank, evidence)
    references = _mapping(payload.get("reference_scores"), "reference_scores")
    _validate_reference_scores(references, evidence)
    if not json_equal_exact(payload.get("acceptance_rule"), _ACCEPTANCE_RULE):
        raise ScoringDecisionError("acceptance_rule does not match frozen policy")
    return ScoringDecision(
        decision_id=_DECISION_ID,
        owner_accepted=False,
        provider_execution_allowed=False,
        maximum_difference=0.5,
        response_sample_count=cast(int, response_bank["sample_count"]),
    )
