"""Dependency-free extraction of scored rows from pinned Inspect archives."""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from concept_filtering.serialization import bytes_sha256, canonical_sha256, file_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_MAX_MEMBER_BYTES = 32 * 1024 * 1024
_APPROVED_TASK_BEHAVIORS = {
    "c12-valid-feelings-sft": "validate_feelings",
    "L05-refuse-then-redirect": "refusal",
}


class InspectArchiveValidationError(ValueError):
    """Raised when an evaluation archive fails provenance or shape checks."""


def approved_task_behavior(task: str) -> str:
    """Return the frozen behavior mapping for one supported archive task."""
    behavior = _APPROVED_TASK_BEHAVIORS.get(task)
    if behavior is None:
        raise InspectArchiveValidationError(
            f"task has no approved behavior mapping: {task}"
        )
    return behavior


@dataclass(frozen=True)
class InspectExtraction:
    schema_version: int
    archive_sha256: str
    task: str
    behavior: str
    scorer: str
    question_count: int
    mean_score: float
    source_origin: str
    source_commit: str
    source_dirty: bool
    output_sha256: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "archive_sha256": self.archive_sha256,
            "task": self.task,
            "behavior": self.behavior,
            "scorer": self.scorer,
            "question_count": self.question_count,
            "mean_score": self.mean_score,
            "source_origin": self.source_origin,
            "source_commit": self.source_commit,
            "source_dirty": self.source_dirty,
            "output_sha256": self.output_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class InspectHeaderMetadata:
    task: str
    behavior: str
    sample_count: int
    sample_ids: frozenset[str]
    source_origin: str
    source_commit: str
    source_dirty: bool
    mean_score: float


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InspectArchiveValidationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def require_mapping(value: object, field: str) -> dict[str, Any]:
    """Validate a shared Inspect JSON object boundary."""
    return _mapping(value, field)


def _required_string(payload: dict[str, Any], field: str, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InspectArchiveValidationError(
            f"{context}.{field} must be a non-empty string"
        )
    return value


def require_string(payload: dict[str, Any], field: str, context: str) -> str:
    """Validate a shared Inspect non-empty string boundary."""
    return _required_string(payload, field, context)


def _integer(payload: dict[str, Any], field: str, context: str) -> int:
    value = payload.get(field)
    if type(value) is not int or value < 0:
        raise InspectArchiveValidationError(
            f"{context}.{field} must be a non-negative integer"
        )
    return value


def _positive_integer(payload: dict[str, Any], field: str, context: str) -> int:
    value = payload.get(field)
    if type(value) is not int or value <= 0:
        raise InspectArchiveValidationError(
            f"{context}.{field} must be a positive integer"
        )
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InspectArchiveValidationError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise InspectArchiveValidationError(f"{context} must be a finite number")
    return result


def _validate_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    for member in archive.infolist():
        member_path = PurePosixPath(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise InspectArchiveValidationError(
                f"unsafe archive member path: {member.filename}"
            )
        if member.filename in members:
            raise InspectArchiveValidationError(
                f"duplicate archive member: {member.filename}"
            )
        if member.file_size > _MAX_MEMBER_BYTES:
            raise InspectArchiveValidationError(
                f"archive member is too large: {member.filename}"
            )
        total_size += member.file_size
        members[member.filename] = member
    if total_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise InspectArchiveValidationError("archive uncompressed size exceeds limit")
    return members


def validate_archive_members(
    archive: zipfile.ZipFile,
) -> dict[str, zipfile.ZipInfo]:
    """Validate archive paths and decompressed-size limits."""
    return _validate_members(archive)


def _read_json_member(
    archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], name: str
) -> object:
    member = members.get(name)
    if member is None:
        raise InspectArchiveValidationError(f"archive is missing {name}")
    try:
        return json.loads(archive.read(member).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise InspectArchiveValidationError(
            f"archive member {name} is not valid UTF-8 JSON: {error}"
        ) from error


def read_json_member(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    name: str,
) -> object:
    """Read one member after archive-level validation."""
    return _read_json_member(archive, members, name)


def _dataset_metadata(evaluation: dict[str, Any]) -> tuple[int, frozenset[str]]:
    dataset = _mapping(evaluation.get("dataset"), "header.eval.dataset")
    sample_count = _positive_integer(dataset, "samples", "header.eval.dataset")
    raw_sample_ids = dataset.get("sample_ids")
    if not isinstance(raw_sample_ids, list):
        raise InspectArchiveValidationError(
            "header.eval.dataset.sample_ids must be an array"
        )
    sample_ids: list[str] = []
    for item in cast(list[object], raw_sample_ids):
        if not isinstance(item, str) or not item.strip():
            raise InspectArchiveValidationError(
                "header.eval.dataset.sample_ids entries must be non-empty strings"
            )
        sample_ids.append(item)
    if len(sample_ids) != sample_count:
        raise InspectArchiveValidationError(
            "header sample_ids count does not match dataset.samples"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise InspectArchiveValidationError("header contains duplicate question ids")
    return sample_count, frozenset(sample_ids)


def _revision_metadata(evaluation: dict[str, Any]) -> tuple[str, str, bool]:
    revision = _mapping(evaluation.get("revision"), "header.eval.revision")
    source_origin = _required_string(revision, "origin", "header.eval.revision")
    source_commit = _required_string(revision, "commit", "header.eval.revision")
    source_dirty = revision.get("dirty")
    if type(source_dirty) is not bool:
        raise InspectArchiveValidationError(
            "header.eval.revision.dirty must be a boolean"
        )
    return source_origin, source_commit, source_dirty


def _score_summary(header: dict[str, Any], *, scorer: str, sample_count: int) -> float:
    results = _mapping(header.get("results"), "header.results")
    total_samples = _integer(results, "total_samples", "header.results")
    completed_samples = _integer(results, "completed_samples", "header.results")
    if total_samples != sample_count:
        raise InspectArchiveValidationError(
            "header.results.total_samples does not match dataset.samples"
        )
    if completed_samples != sample_count:
        raise InspectArchiveValidationError(
            "header.results.completed_samples does not match dataset.samples"
        )
    raw_scores = results.get("scores")
    if not isinstance(raw_scores, list):
        raise InspectArchiveValidationError("header.results.scores must be an array")
    summary = _matching_score_summary(cast(list[object], raw_scores), scorer)
    if _integer(summary, "scored_samples", "header score summary") != sample_count:
        raise InspectArchiveValidationError(
            "header scorer scored_samples does not match dataset.samples"
        )
    if _integer(summary, "unscored_samples", "header score summary") != 0:
        raise InspectArchiveValidationError("header scorer contains unscored samples")
    metrics = _mapping(summary.get("metrics"), "header score metrics")
    mean_metric = _mapping(metrics.get("mean"), "header mean metric")
    return _number(mean_metric.get("value"), "header mean metric value")


def _matching_score_summary(raw_scores: list[object], scorer: str) -> dict[str, Any]:
    matching: list[dict[str, Any]] = []
    for item in raw_scores:
        candidate = _mapping(item, "header score summary")
        if candidate.get("name") == scorer or candidate.get("scorer") == scorer:
            matching.append(candidate)
    if len(matching) != 1:
        raise InspectArchiveValidationError(
            f"header must contain exactly one scorer summary: {scorer}"
        )
    return matching[0]


def _header_metadata(header_value: object, scorer: str) -> InspectHeaderMetadata:
    header = _mapping(header_value, "header")
    if header.get("status") != "success":
        raise InspectArchiveValidationError("header.status must be success")
    evaluation = _mapping(header.get("eval"), "header.eval")
    task = _required_string(evaluation, "task", "header.eval")
    behavior = approved_task_behavior(task)
    sample_count, sample_ids = _dataset_metadata(evaluation)
    source_origin, source_commit, source_dirty = _revision_metadata(evaluation)
    return InspectHeaderMetadata(
        task=task,
        behavior=behavior,
        sample_count=sample_count,
        sample_ids=sample_ids,
        source_origin=source_origin,
        source_commit=source_commit,
        source_dirty=source_dirty,
        mean_score=_score_summary(header, scorer=scorer, sample_count=sample_count),
    )


def read_header_metadata(header_value: object, scorer: str) -> InspectHeaderMetadata:
    """Validate the shared evaluation header and provenance fields."""
    return _header_metadata(header_value, scorer)


def _sample_scores(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    scorer: str,
) -> dict[str, float]:
    sample_names = sorted(
        name
        for name, member in members.items()
        if name.startswith("samples/")
        and name.endswith(".json")
        and not member.is_dir()
    )
    scores: dict[str, float] = {}
    for name in sample_names:
        sample = _mapping(_read_json_member(archive, members, name), name)
        question_id = _required_string(sample, "id", name)
        if question_id in scores:
            raise InspectArchiveValidationError(
                f"duplicate question id in sample records: {question_id}"
            )
        sample_scorers = _mapping(sample.get("scores"), f"{name}.scores")
        scorer_result = _mapping(sample_scorers.get(scorer), f"{name}.scores.{scorer}")
        scores[question_id] = _number(
            scorer_result.get("value"), f"{name}.scores.{scorer}.value"
        )
    return scores


def _validate_extraction_request(
    *, scorer: str, expected_sha256: str, output_path: Path, receipt_path: Path
) -> None:
    if not scorer.strip():
        raise InspectArchiveValidationError("scorer cannot be empty")
    if output_path.resolve() == receipt_path.resolve():
        raise InspectArchiveValidationError("output and receipt paths must differ")
    if _SHA256.fullmatch(expected_sha256) is None:
        raise InspectArchiveValidationError(
            "expected_sha256 must be 64 lowercase hexadecimal digits"
        )


def _read_archive(
    archive_path: Path, *, scorer: str
) -> tuple[InspectHeaderMetadata, dict[str, float]]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _validate_members(archive)
            metadata = _header_metadata(
                _read_json_member(archive, members, "header.json"), scorer
            )
            return metadata, _sample_scores(archive, members, scorer)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise InspectArchiveValidationError(
            f"cannot read Inspect archive {archive_path}: {error}"
        ) from error


def _validate_extracted_scores(
    metadata: InspectHeaderMetadata, scores: dict[str, float]
) -> float:
    if len(scores) != metadata.sample_count:
        raise InspectArchiveValidationError(
            f"sample record count {len(scores)} does not match header "
            f"{metadata.sample_count}"
        )
    if frozenset(scores) != metadata.sample_ids:
        raise InspectArchiveValidationError(
            "sample record ids do not match header sample_ids"
        )
    computed_mean = sum(scores.values()) / len(scores)
    if not math.isclose(computed_mean, metadata.mean_score, rel_tol=0.0, abs_tol=1e-12):
        raise InspectArchiveValidationError(
            f"computed mean {computed_mean} does not match header mean "
            f"{metadata.mean_score}"
        )
    return computed_mean


def _build_extraction(
    metadata: InspectHeaderMetadata,
    *,
    archive_sha256: str,
    scorer: str,
    mean_score: float,
    output_sha256: str,
) -> InspectExtraction:
    body = _extraction_body(
        metadata,
        archive_sha256=archive_sha256,
        scorer=scorer,
        mean_score=mean_score,
        output_sha256=output_sha256,
    )
    return InspectExtraction(
        schema_version=1,
        archive_sha256=archive_sha256,
        task=metadata.task,
        behavior=metadata.behavior,
        scorer=scorer,
        question_count=metadata.sample_count,
        mean_score=mean_score,
        source_origin=metadata.source_origin,
        source_commit=metadata.source_commit,
        source_dirty=metadata.source_dirty,
        output_sha256=output_sha256,
        receipt_sha256=canonical_sha256(body),
    )


def _extraction_body(
    metadata: InspectHeaderMetadata,
    *,
    archive_sha256: str,
    scorer: str,
    mean_score: float,
    output_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "archive_sha256": archive_sha256,
        "task": metadata.task,
        "behavior": metadata.behavior,
        "scorer": scorer,
        "question_count": metadata.sample_count,
        "mean_score": mean_score,
        "source_origin": metadata.source_origin,
        "source_commit": metadata.source_commit,
        "source_dirty": metadata.source_dirty,
        "output_sha256": output_sha256,
    }


def _verified_archive_sha(archive_path: Path, expected_sha256: str) -> str:
    archive_sha256 = file_sha256(archive_path)
    if archive_sha256 != expected_sha256:
        raise InspectArchiveValidationError(
            f"archive SHA-256 mismatch: expected {expected_sha256}, "
            f"found {archive_sha256}"
        )
    return archive_sha256


def verified_archive_sha(archive_path: Path, expected_sha256: str) -> str:
    """Hash an archive and fail closed when it differs from the pin."""
    return _verified_archive_sha(archive_path, expected_sha256)


def extract_scores(
    archive_path: Path,
    *,
    scorer: str,
    expected_sha256: str,
    output_path: Path,
    receipt_path: Path,
) -> InspectExtraction:
    """Verify one archive and write normalized score rows for receipt creation."""
    _validate_extraction_request(
        scorer=scorer,
        expected_sha256=expected_sha256,
        output_path=output_path,
        receipt_path=receipt_path,
    )
    archive_sha256 = _verified_archive_sha(archive_path, expected_sha256)
    metadata, scores = _read_archive(archive_path, scorer=scorer)
    computed_mean = _validate_extracted_scores(metadata, scores)
    output_bytes = _score_rows_bytes(scores, behavior=metadata.behavior)
    output_sha256 = bytes_sha256(output_bytes)
    extraction = _build_extraction(
        metadata,
        archive_sha256=archive_sha256,
        scorer=scorer,
        mean_score=computed_mean,
        output_sha256=output_sha256,
    )
    _write_extraction(output_bytes, extraction, output_path, receipt_path)
    return extraction


def _score_rows_bytes(scores: dict[str, float], *, behavior: str) -> bytes:
    text = "".join(
        json.dumps(
            {
                "behavior": behavior,
                "question_id": question_id,
                "score": scores[question_id],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for question_id in sorted(scores)
    )
    return text.encode("utf-8")


def _write_extraction(
    output_bytes: bytes,
    extraction: InspectExtraction,
    output_path: Path,
    receipt_path: Path,
) -> None:
    try:
        output_path.write_bytes(output_bytes)
        receipt_path.write_text(
            json.dumps(
                extraction.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError) as error:
        raise InspectArchiveValidationError(
            f"cannot write extraction outputs: {error}"
        ) from error


def _load_extraction_payload(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InspectArchiveValidationError(
            f"cannot load extraction receipt {path}: {error}"
        ) from error
    payload = _mapping(value, "extraction receipt")
    expected_fields = set(InspectExtraction.__dataclass_fields__)
    if set(payload) != expected_fields:
        raise InspectArchiveValidationError(
            "extraction receipt fields do not match schema"
        )
    return payload


def _receipt_sha(payload: dict[str, Any], field: str) -> str:
    value = _required_string(payload, field, "extraction receipt")
    if _SHA256.fullmatch(value) is None:
        raise InspectArchiveValidationError(
            f"extraction receipt.{field} must be a SHA-256"
        )
    return value


def _parse_extraction(payload: dict[str, Any]) -> InspectExtraction:
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise InspectArchiveValidationError("extraction receipt schema must be 1")
    task = _required_string(payload, "task", "extraction receipt")
    behavior = _required_string(payload, "behavior", "extraction receipt")
    if _APPROVED_TASK_BEHAVIORS.get(task) != behavior:
        raise InspectArchiveValidationError(
            "extraction task-to-behavior mapping is invalid"
        )
    source_dirty = payload.get("source_dirty")
    if type(source_dirty) is not bool:
        raise InspectArchiveValidationError("extraction source_dirty must be boolean")
    return InspectExtraction(
        schema_version=1,
        archive_sha256=_receipt_sha(payload, "archive_sha256"),
        task=task,
        behavior=behavior,
        scorer=_required_string(payload, "scorer", "extraction receipt"),
        question_count=_positive_integer(payload, "question_count", "extraction"),
        mean_score=_number(payload.get("mean_score"), "extraction mean_score"),
        source_origin=_required_string(payload, "source_origin", "extraction receipt"),
        source_commit=_required_string(payload, "source_commit", "extraction receipt"),
        source_dirty=source_dirty,
        output_sha256=_receipt_sha(payload, "output_sha256"),
        receipt_sha256=_receipt_sha(payload, "receipt_sha256"),
    )


def load_extraction_receipt(path: Path, *, output_path: Path) -> InspectExtraction:
    """Validate a durable extraction receipt against its normalized output."""
    payload = _load_extraction_payload(path)
    receipt = _parse_extraction(payload)
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if canonical_sha256(body) != receipt.receipt_sha256:
        raise InspectArchiveValidationError("extraction receipt hash mismatch")
    if file_sha256(output_path) != receipt.output_sha256:
        raise InspectArchiveValidationError("extraction output hash mismatch")
    return receipt
