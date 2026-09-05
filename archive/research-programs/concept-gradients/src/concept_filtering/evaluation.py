"""Evaluation configuration and result receipts for frozen behavior rubrics."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.manifest import DatasetManifest
from concept_filtering.organism import OrganismConfig
from concept_filtering.serialization import bytes_sha256, canonical_sha256, file_sha256

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvaluationValidationError(ValueError):
    """Raised when evaluator inputs cannot support a complete result receipt."""


@dataclass(frozen=True)
class EvaluatorConfig:
    schema_version: int
    evaluator_id: str
    config_sha256: str
    scientific_use: bool
    behaviors: tuple[str, ...]
    expected_questions: dict[str, int]
    score_min: float
    score_max: float
    provenance: str


@dataclass(frozen=True)
class BehaviorSummary:
    behavior: str
    question_count: int
    mean_score: float
    min_score: float
    max_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "behavior": self.behavior,
            "question_count": self.question_count,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }


@dataclass(frozen=True)
class EvaluationReceipt:
    schema_version: int
    evaluator_id: str
    evaluator_config_sha256: str
    scientific_use: bool
    code_revision: str
    organism_sha256: str
    manifest_sha256: str
    scores_sha256: str
    summaries: tuple[BehaviorSummary, ...]
    result_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluator_id": self.evaluator_id,
            "evaluator_config_sha256": self.evaluator_config_sha256,
            "scientific_use": self.scientific_use,
            "code_revision": self.code_revision,
            "organism_sha256": self.organism_sha256,
            "manifest_sha256": self.manifest_sha256,
            "scores_sha256": self.scores_sha256,
            "summaries": [summary.to_dict() for summary in self.summaries],
            "result_sha256": self.result_sha256,
        }


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationValidationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationValidationError(f"{field} must be a non-empty string")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationValidationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationValidationError(f"{field} must be a finite number")
    return result


def _load_evaluator_payload(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationValidationError(
            f"cannot load evaluator {path}: {error}"
        ) from error
    return _mapping(value, "evaluator")


def _validate_evaluator_envelope(payload: dict[str, Any]) -> bool:
    allowed = {
        "schema_version",
        "evaluator_id",
        "scientific_use",
        "behaviors",
        "expected_questions",
        "score_min",
        "score_max",
        "provenance",
    }
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise EvaluationValidationError(
            f"evaluator has unexpected fields: {', '.join(unexpected)}"
        )
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise EvaluationValidationError("schema_version must be the integer 1")
    scientific_use = payload.get("scientific_use")
    if type(scientific_use) is not bool:
        raise EvaluationValidationError("scientific_use must be a boolean")
    return scientific_use


def _parse_behaviors(payload: dict[str, Any]) -> tuple[str, ...]:
    raw_behaviors = payload.get("behaviors")
    if not isinstance(raw_behaviors, list) or not raw_behaviors:
        raise EvaluationValidationError("behaviors must be a non-empty array")
    behaviors: list[str] = []
    for behavior in cast(list[object], raw_behaviors):
        if not isinstance(behavior, str) or not behavior.strip():
            raise EvaluationValidationError(
                "behaviors entries must be non-empty strings"
            )
        behaviors.append(behavior)
    if len(behaviors) != len(set(behaviors)):
        raise EvaluationValidationError("behaviors must be unique")
    return tuple(behaviors)


def _parse_expected_questions(
    payload: dict[str, Any], behaviors: tuple[str, ...], *, scientific_use: bool
) -> dict[str, int]:
    raw_expected_questions = _mapping(
        payload.get("expected_questions"), "expected_questions"
    )
    if set(raw_expected_questions) != set(behaviors):
        raise EvaluationValidationError(
            "expected_questions keys must exactly match behaviors"
        )
    expected_questions: dict[str, int] = {}
    for behavior in behaviors:
        expected_count = raw_expected_questions[behavior]
        if type(expected_count) is not int or expected_count <= 0:
            raise EvaluationValidationError(
                f"expected_questions.{behavior} must be a positive integer"
            )
        expected_questions[behavior] = expected_count
    if scientific_use and any(count < 100 for count in expected_questions.values()):
        raise EvaluationValidationError(
            "scientific evaluator requires at least 100 questions per behavior"
        )
    return expected_questions


def load_evaluator_config(path: Path) -> EvaluatorConfig:
    payload = _load_evaluator_payload(path)
    scientific_use = _validate_evaluator_envelope(payload)
    behaviors = _parse_behaviors(payload)
    expected_questions = _parse_expected_questions(
        payload, behaviors, scientific_use=scientific_use
    )
    score_min = _number(payload.get("score_min"), "score_min")
    score_max = _number(payload.get("score_max"), "score_max")
    if score_min >= score_max:
        raise EvaluationValidationError("score_min must be less than score_max")

    return EvaluatorConfig(
        schema_version=1,
        evaluator_id=_required_string(payload, "evaluator_id"),
        config_sha256=file_sha256(path),
        scientific_use=scientific_use,
        behaviors=behaviors,
        expected_questions=expected_questions,
        score_min=score_min,
        score_max=score_max,
        provenance=_required_string(payload, "provenance"),
    )


def _validate_binding(value: str, field: str, pattern: re.Pattern[str]) -> None:
    if pattern.fullmatch(value) is None:
        raise EvaluationValidationError(f"{field} has an invalid immutable hash")


def _parse_score_line(
    raw_line: str,
    *,
    line_number: int,
    evaluator: EvaluatorConfig,
) -> tuple[str, str, float]:
    if not raw_line.strip():
        raise EvaluationValidationError(
            f"score line {line_number}: blank lines are not allowed"
        )
    try:
        value: object = json.loads(raw_line)
    except json.JSONDecodeError as error:
        raise EvaluationValidationError(
            f"score line {line_number}: invalid JSON: {error.msg}"
        ) from error
    row = _mapping(value, f"score line {line_number}")
    _validate_score_row_fields(row, line_number=line_number)
    behavior = _required_string(row, "behavior")
    if behavior not in evaluator.behaviors:
        raise EvaluationValidationError(
            f"score line {line_number}: unknown behavior {behavior}"
        )
    question_id = _required_string(row, "question_id")
    score = _validated_score(row.get("score"), evaluator, line_number=line_number)
    return behavior, question_id, score


def _validate_score_row_fields(row: dict[str, Any], *, line_number: int) -> None:
    unexpected = sorted(set(row) - {"behavior", "question_id", "score"})
    if unexpected:
        raise EvaluationValidationError(
            f"score line {line_number} has unexpected fields: {', '.join(unexpected)}"
        )


def _validated_score(
    value: object, evaluator: EvaluatorConfig, *, line_number: int
) -> float:
    score = _number(value, "score")
    if not evaluator.score_min <= score <= evaluator.score_max:
        raise EvaluationValidationError(
            f"score line {line_number}: score must be between "
            f"{evaluator.score_min} and {evaluator.score_max}"
        )
    return score


def _validate_score_counts(
    scores: dict[str, dict[str, float]], evaluator: EvaluatorConfig
) -> None:
    for behavior, behavior_scores in scores.items():
        actual = len(behavior_scores)
        expected = evaluator.expected_questions[behavior]
        if actual != expected:
            raise EvaluationValidationError(
                f"{behavior}: expected {expected} questions, found {actual}"
            )


def _load_scores(
    path: Path, evaluator: EvaluatorConfig
) -> tuple[dict[str, dict[str, float]], str]:
    try:
        source_bytes = path.read_bytes()
        lines = source_bytes.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvaluationValidationError(
            f"cannot read scores {path}: {error}"
        ) from error
    if not lines:
        raise EvaluationValidationError("scores must contain at least one row")
    scores: dict[str, dict[str, float]] = {
        behavior: {} for behavior in evaluator.behaviors
    }
    for line_number, raw_line in enumerate(lines, start=1):
        behavior, question_id, score = _parse_score_line(
            raw_line, line_number=line_number, evaluator=evaluator
        )
        if question_id in scores[behavior]:
            raise EvaluationValidationError(
                f"score line {line_number}: duplicate question {behavior}/{question_id}"
            )
        scores[behavior][question_id] = score
    _validate_score_counts(scores, evaluator)
    return scores, bytes_sha256(source_bytes)


def _summarize_scores(
    scores: dict[str, dict[str, float]],
) -> tuple[BehaviorSummary, ...]:
    summaries: list[BehaviorSummary] = []
    for behavior in sorted(scores):
        values = list(scores[behavior].values())
        summaries.append(
            BehaviorSummary(
                behavior=behavior,
                question_count=len(values),
                mean_score=sum(values) / len(values),
                min_score=min(values),
                max_score=max(values),
            )
        )
    return tuple(summaries)


def build_evaluation_receipt(
    scores_path: Path,
    *,
    evaluator: EvaluatorConfig,
    code_revision: str,
    organism: OrganismConfig,
    manifest: DatasetManifest,
    expected_organism_sha256: str,
    expected_manifest_sha256: str,
) -> EvaluationReceipt:
    """Validate scored questions and bind their summaries to all run inputs."""
    _validate_upstream_bindings(
        code_revision,
        organism,
        manifest,
        expected_organism_sha256=expected_organism_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    scores, scores_sha256 = _load_scores(scores_path, evaluator)
    summaries = _summarize_scores(scores)
    return _make_evaluation_receipt(
        evaluator,
        code_revision=code_revision,
        organism_sha256=organism.organism_sha256,
        manifest_sha256=manifest.manifest_sha256,
        scores_sha256=scores_sha256,
        summaries=summaries,
    )


def _make_evaluation_receipt(
    evaluator: EvaluatorConfig,
    *,
    code_revision: str,
    organism_sha256: str,
    manifest_sha256: str,
    scores_sha256: str,
    summaries: tuple[BehaviorSummary, ...],
) -> EvaluationReceipt:
    body = _evaluation_body(
        evaluator,
        code_revision=code_revision,
        organism_sha256=organism_sha256,
        manifest_sha256=manifest_sha256,
        scores_sha256=scores_sha256,
        summaries=summaries,
    )
    return EvaluationReceipt(
        schema_version=1,
        evaluator_id=evaluator.evaluator_id,
        evaluator_config_sha256=evaluator.config_sha256,
        scientific_use=evaluator.scientific_use,
        code_revision=code_revision,
        organism_sha256=organism_sha256,
        manifest_sha256=manifest_sha256,
        scores_sha256=scores_sha256,
        summaries=summaries,
        result_sha256=canonical_sha256(body),
    )


def _validate_upstream_bindings(
    code_revision: str,
    organism: OrganismConfig,
    manifest: DatasetManifest,
    *,
    expected_organism_sha256: str,
    expected_manifest_sha256: str,
) -> None:
    _validate_binding(code_revision, "code_revision", _COMMIT_SHA)
    _validate_binding(expected_organism_sha256, "organism_sha256", _SHA256)
    _validate_binding(expected_manifest_sha256, "manifest_sha256", _SHA256)
    if organism.organism_sha256 != expected_organism_sha256:
        raise EvaluationValidationError("organism hash mismatch")
    if manifest.manifest_sha256 != expected_manifest_sha256:
        raise EvaluationValidationError("manifest hash mismatch")


def _evaluation_body(
    evaluator: EvaluatorConfig,
    *,
    code_revision: str,
    organism_sha256: str,
    manifest_sha256: str,
    scores_sha256: str,
    summaries: tuple[BehaviorSummary, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluator_id": evaluator.evaluator_id,
        "evaluator_config_sha256": evaluator.config_sha256,
        "scientific_use": evaluator.scientific_use,
        "code_revision": code_revision,
        "organism_sha256": organism_sha256,
        "manifest_sha256": manifest_sha256,
        "scores_sha256": scores_sha256,
        "summaries": [summary.to_dict() for summary in summaries],
    }


def write_evaluation_receipt(receipt: EvaluationReceipt, path: Path) -> None:
    path.write_text(
        json.dumps(receipt.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
