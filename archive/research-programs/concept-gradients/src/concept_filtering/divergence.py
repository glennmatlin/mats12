"""Machine-readable comparison of source-study claims and candidate receipts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from concept_filtering.organism import ObservationStatus, ParameterObservation
from concept_filtering.serialization import canonical_sha256

RECONSTRUCTION_R0_PROFILE = "reconstruction-r0-v1"
RECONSTRUCTION_R0_REQUIRED_FIELDS = frozenset(
    {
        "adapter.producing_code_revision_pinned",
        "adapter.revision_pinned",
        "adapter.weights_sha256_pinned",
        "code.publicly_available",
        "data.manifest_sha256_pinned",
        "data.max_sequence_length",
        "data.sample_license_recorded",
        "data.sample_role_confirmed",
        "data.selection_code_published",
        "data.selection_receipt_available",
        "data.source_dataset",
        "data.source_revision_pinned",
        "evaluation.archive_receipts_pinned",
        "evaluation.behaviors",
        "evaluation.evaluator_config_published",
        "evaluation.producing_checkout_clean",
        "evaluation.prompt_bank_receipts_pinned",
        "evaluation.refusal_question_count",
        "evaluation.scoring_rule_published",
        "evaluation.validation_question_count",
        "model.family",
        "model.source_revision_pinned",
        "model.tokenizer_revision_pinned",
        "sae.architecture_consistent",
        "sae.d_in",
        "sae.d_sae",
        "sae.hook_name",
        "sae.model_revision_pinned",
        "sae.release_revision_pinned",
        "training.adapter_target_modules",
        "training.chat_template_published",
        "training.data_order_receipt_available",
        "training.epochs",
        "training.gradient_accumulation_published",
        "training.lora_alpha",
        "training.lora_dropout",
        "training.lora_rank",
        "training.optimizer_published",
        "training.schedule_published",
        "training.seed_published",
        "training.sequence_policy_published",
    }
)


class DivergenceValidationError(ValueError):
    """Raised when a comparison request is internally ambiguous."""


class GateEffect(StrEnum):
    BLOCKING = "blocking"
    INFORMATIONAL = "informational"


class DivergenceStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Expectation:
    field: str
    expected: object
    category: str
    gate_effect: GateEffect
    evidence: str


@dataclass(frozen=True)
class ReferenceSpec:
    schema_version: int
    reference_id: str
    spec_sha256: str
    gate_profile: str | None
    expectations: tuple[Expectation, ...]


@dataclass(frozen=True)
class DivergenceEntry:
    field: str
    category: str
    gate_effect: GateEffect
    status: DivergenceStatus
    expected: object
    observed: object
    reference_evidence: str
    observation_evidence: str | None
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "category": self.category,
            "gate_effect": self.gate_effect.value,
            "status": self.status.value,
            "expected": self.expected,
            "observed": self.observed,
            "reference_evidence": self.reference_evidence,
            "observation_evidence": self.observation_evidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DivergenceReport:
    schema_version: int
    reference_id: str
    organism_id: str
    entries: tuple[DivergenceEntry, ...]
    blocking_count: int
    is_blocked: bool
    report_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "reference_id": self.reference_id,
            "organism_id": self.organism_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "blocking_count": self.blocking_count,
            "is_blocked": self.is_blocked,
            "report_sha256": self.report_sha256,
        }


def _validate_expectation(expectation: Expectation) -> None:
    for name, value in (
        ("field", expectation.field),
        ("category", expectation.category),
        ("evidence", expectation.evidence),
    ):
        if not value.strip():
            raise DivergenceValidationError(f"expectation {name} cannot be empty")


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DivergenceValidationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DivergenceValidationError(f"{field} must be a non-empty string")
    return value


def _load_reference_payload(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DivergenceValidationError(
            f"cannot load reference spec {path}: {error}"
        ) from error
    return _mapping(value, "reference spec")


def _parse_expectation(raw_value: object, *, index: int) -> Expectation:
    item = _mapping(raw_value, f"expectation {index}")
    allowed = {"field", "expected", "category", "gate_effect", "evidence"}
    unexpected = sorted(set(item) - allowed)
    if unexpected:
        raise DivergenceValidationError(
            f"expectation {index} has unexpected fields: {', '.join(unexpected)}"
        )
    if "expected" not in item or item["expected"] is None:
        raise DivergenceValidationError(
            f"expectation {index} requires a non-null expected value"
        )
    gate_value = _required_string(item, "gate_effect")
    try:
        gate_effect = GateEffect(gate_value)
    except ValueError as error:
        raise DivergenceValidationError(
            f"expectation {index} gate_effect must be blocking or informational"
        ) from error
    expectation = Expectation(
        field=_required_string(item, "field"),
        expected=item["expected"],
        category=_required_string(item, "category"),
        gate_effect=gate_effect,
        evidence=_required_string(item, "evidence"),
    )
    _validate_expectation(expectation)
    return expectation


def _parse_expectations(value: object) -> tuple[Expectation, ...]:
    if not isinstance(value, list) or not value:
        raise DivergenceValidationError("expectations must be a non-empty array")
    expectations = tuple(
        _parse_expectation(item, index=index)
        for index, item in enumerate(cast(list[object], value), start=1)
    )
    fields = [expectation.field for expectation in expectations]
    if len(fields) != len(set(fields)):
        raise DivergenceValidationError("expectation fields must be unique")
    return expectations


def _validate_gate_profile(
    profile: object, expectations: tuple[Expectation, ...]
) -> str | None:
    if profile is None:
        return None
    if profile != RECONSTRUCTION_R0_PROFILE:
        raise DivergenceValidationError("gate_profile is not supported")
    by_field = {expectation.field: expectation for expectation in expectations}
    missing = sorted(RECONSTRUCTION_R0_REQUIRED_FIELDS - set(by_field))
    if missing:
        raise DivergenceValidationError(
            f"gate profile is missing required fields: {', '.join(missing)}"
        )
    nonblocking = sorted(
        field
        for field in RECONSTRUCTION_R0_REQUIRED_FIELDS
        if by_field[field].gate_effect is not GateEffect.BLOCKING
    )
    if nonblocking:
        raise DivergenceValidationError(
            f"gate profile fields must be blocking: {', '.join(nonblocking)}"
        )
    return RECONSTRUCTION_R0_PROFILE


def load_reference_spec(path: Path) -> ReferenceSpec:
    """Load the source-study expectations used by a divergence gate."""
    payload = _load_reference_payload(path)
    allowed = {"schema_version", "reference_id", "gate_profile", "expectations"}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise DivergenceValidationError(
            f"reference spec has unexpected fields: {', '.join(unexpected)}"
        )
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise DivergenceValidationError("schema_version must be the integer 1")
    expectations = _parse_expectations(payload.get("expectations"))
    gate_profile = _validate_gate_profile(payload.get("gate_profile"), expectations)
    return ReferenceSpec(
        schema_version=1,
        reference_id=_required_string(payload, "reference_id"),
        spec_sha256=canonical_sha256(payload),
        gate_profile=gate_profile,
        expectations=expectations,
    )


def _compare_entry(
    expectation: Expectation, observation: ParameterObservation | None
) -> DivergenceEntry:
    status, observed, evidence, reason = _observation_comparison(
        expectation, observation
    )
    return DivergenceEntry(
        field=expectation.field,
        category=expectation.category,
        gate_effect=expectation.gate_effect,
        status=status,
        expected=expectation.expected,
        observed=observed,
        reference_evidence=expectation.evidence,
        observation_evidence=evidence,
        reason=reason,
    )


def _observation_comparison(
    expectation: Expectation, observation: ParameterObservation | None
) -> tuple[DivergenceStatus, object, str | None, str | None]:
    if observation is None:
        return (
            DivergenceStatus.UNAVAILABLE,
            None,
            None,
            "No observation was recorded for this expected field.",
        )
    elif observation.status is ObservationStatus.UNAVAILABLE:
        return (
            DivergenceStatus.UNAVAILABLE,
            None,
            observation.evidence,
            observation.reason,
        )
    status = (
        DivergenceStatus.MATCH
        if canonical_sha256(expectation.expected) == canonical_sha256(observation.value)
        else DivergenceStatus.MISMATCH
    )
    return (
        status,
        observation.value,
        observation.evidence,
        None if status is DivergenceStatus.MATCH else "Observed value differs.",
    )


def _validate_comparison(
    reference_id: str, organism_id: str, expectations: list[Expectation]
) -> None:
    if not reference_id.strip() or not organism_id.strip():
        raise DivergenceValidationError("reference_id and organism_id cannot be empty")
    fields = [expectation.field for expectation in expectations]
    if len(fields) != len(set(fields)):
        raise DivergenceValidationError("expectation fields must be unique")
    for expectation in expectations:
        _validate_expectation(expectation)


def compare_expectations(
    reference_id: str,
    organism_id: str,
    expectations: list[Expectation],
    observations: dict[str, ParameterObservation],
) -> DivergenceReport:
    """Compare expected fields and return a deterministic Gate R0/R2 report."""
    _validate_comparison(reference_id, organism_id, expectations)
    entries = tuple(
        _compare_entry(expectation, observations.get(expectation.field))
        for expectation in sorted(expectations, key=lambda item: item.field)
    )
    blocking_count = sum(
        entry.gate_effect is GateEffect.BLOCKING
        and entry.status is not DivergenceStatus.MATCH
        for entry in entries
    )
    body = _report_body(reference_id, organism_id, entries, blocking_count)
    return DivergenceReport(
        schema_version=1,
        reference_id=reference_id,
        organism_id=organism_id,
        entries=entries,
        blocking_count=blocking_count,
        is_blocked=blocking_count > 0,
        report_sha256=canonical_sha256(body),
    )


def _report_body(
    reference_id: str,
    organism_id: str,
    entries: tuple[DivergenceEntry, ...],
    blocking_count: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "reference_id": reference_id,
        "organism_id": organism_id,
        "entries": [entry.to_dict() for entry in entries],
        "blocking_count": blocking_count,
        "is_blocked": blocking_count > 0,
    }


def write_divergence_report(report: DivergenceReport, path: Path) -> None:
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
