"""Fail-closed prompt contract for the public behavior-judge reconstruction."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.serialization import canonical_sha256, file_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STRICT_SCORE_PATTERN = r"^SCORE:[ \t]*(-?[0-5])[ \t]*(?:\r?\n)?\Z"
_SCORE_LINE_PATTERN = re.compile(r"^SCORE:", flags=re.MULTILINE)
_PROVENANCE_FIELDS = {
    "hub_repository",
    "hub_revision",
    "public_scorer_path",
    "public_scorer_sha256",
}


class EvaluatorReconstructionError(ValueError):
    """Raised when the reconstructed evaluator contract is invalid."""


@dataclass(frozen=True)
class RubricSpec:
    rubric: str
    scale_anchors: tuple[tuple[str, str], ...]
    source_archive_sha256: str
    expected_samples: int
    source_commit: str
    source_dirty: bool
    task_file: str


@dataclass(frozen=True)
class EvaluatorReconstruction:
    evaluator_id: str
    config_sha256: str
    scientific_use: bool
    starts_experiment_timer: bool
    provider_execution_allowed: bool
    judge_model: str
    judge_model_revision: str
    judge_model_revision_source: str
    judge_model_weights_and_config_fixed: bool
    serving_infrastructure_fixed: bool
    max_tokens: int
    system_prompt: str
    user_prompt_template: str
    score_pattern: str
    behaviors: dict[str, RubricSpec]
    public_scorer_sha256: str
    inspect_ai_version: str
    divergences: tuple[str, ...]


@dataclass(frozen=True)
class JudgeRequest:
    judge_model: str
    max_tokens: int
    system: str
    user: str


@dataclass(frozen=True)
class ArchiveVerification:
    behavior: str
    archive_sha256: str
    sample_count: int
    source_commit: str
    task_file: str

    def to_dict(self) -> dict[str, object]:
        return {
            "behavior": self.behavior,
            "archive_sha256": self.archive_sha256,
            "sample_count": self.sample_count,
            "source_commit": self.source_commit,
            "source_dirty": True,
            "task_file": self.task_file,
        }


@dataclass(frozen=True)
class EvaluatorReconstructionReceipt:
    evaluator_id: str
    config_sha256: str
    judge_model_revision: str
    judge_model_revision_source: str
    judge_model_weights_and_config_fixed: bool
    serving_infrastructure_fixed: bool
    public_source_sha256: str
    archives: tuple[ArchiveVerification, ...]
    sample_count: int
    request_match_count: int
    strict_parse_match_count: int
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            **_receipt_body(
                self.evaluator_id,
                self.config_sha256,
                self.judge_model_revision,
                self.judge_model_revision_source,
                self.public_source_sha256,
                self.archives,
            ),
            "receipt_sha256": self.receipt_sha256,
        }


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluatorReconstructionError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise EvaluatorReconstructionError(f"{field} must be a non-empty string")
    return value


def _sha256(payload: dict[str, Any], field: str) -> str:
    value = _string(payload, field)
    if _SHA256.fullmatch(value) is None:
        raise EvaluatorReconstructionError(f"{field} must be a SHA-256")
    return value


def _false_flag(payload: dict[str, Any], field: str) -> bool:
    if payload.get(field) is not False:
        raise EvaluatorReconstructionError(f"{field} must remain false")
    return False


def _true_flag(payload: dict[str, Any], field: str) -> bool:
    if payload.get(field) is not True:
        raise EvaluatorReconstructionError(f"{field} must remain true")
    return True


def _positive_integer(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if type(value) is not int or value <= 0:
        raise EvaluatorReconstructionError(f"{field} must be a positive integer")
    return value


def _string_array(value: object, field: str) -> tuple[str, ...]:
    raw_values = cast(list[object], value) if isinstance(value, list) else []
    values = tuple(item for item in raw_values if isinstance(item, str) and item)
    if len(values) != len(raw_values) or not values:
        raise EvaluatorReconstructionError(f"{field} must contain strings")
    return values


def _validate_envelope(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise EvaluatorReconstructionError("schema_version must be 1")
    judge_model = _string(payload, "judge_model")
    revision = _string(payload, "judge_model_revision")
    if judge_model.removeprefix("anthropic/") != revision:
        raise EvaluatorReconstructionError("judge_model_revision does not match model")
    _string(payload, "judge_model_revision_source")
    _true_flag(payload, "judge_model_weights_and_config_fixed")
    _false_flag(payload, "serving_infrastructure_fixed")
    if payload.get("score_pattern") != _STRICT_SCORE_PATTERN:
        raise EvaluatorReconstructionError("score_pattern must remain strict")


def _public_scorer_sha256(provenance: dict[str, Any]) -> str:
    unexpected = set(provenance) - _PROVENANCE_FIELDS
    if unexpected:
        raise EvaluatorReconstructionError(
            f"unexpected provenance fields: {sorted(unexpected)}"
        )
    for field in _PROVENANCE_FIELDS - {"public_scorer_sha256"}:
        if field in provenance:
            _string(provenance, field)
    return _sha256(provenance, "public_scorer_sha256")


def _anchors(value: object, behavior: str) -> tuple[tuple[str, str], ...]:
    raw_anchors = cast(list[object], value) if isinstance(value, list) else []
    anchors: list[tuple[str, str]] = []
    for raw_anchor in raw_anchors:
        anchor = _mapping(raw_anchor, f"behaviors.{behavior}.scale_anchors")
        anchors.append((_string(anchor, "score"), _string(anchor, "description")))
    if not anchors:
        raise EvaluatorReconstructionError(f"{behavior} must define scale anchors")
    return tuple(anchors)


def _behaviors(value: object) -> dict[str, RubricSpec]:
    raw_behaviors = _mapping(value, "behaviors")
    behaviors: dict[str, RubricSpec] = {}
    for behavior, raw_spec in raw_behaviors.items():
        spec = _mapping(raw_spec, f"behaviors.{behavior}")
        behaviors[behavior] = RubricSpec(
            rubric=_string(spec, "rubric"),
            scale_anchors=_anchors(spec.get("scale_anchors"), behavior),
            source_archive_sha256=_sha256(spec, "source_archive_sha256"),
            expected_samples=_positive_integer(spec, "expected_samples"),
            source_commit=_string(spec, "source_commit"),
            source_dirty=_true_flag(spec, "source_dirty"),
            task_file=_string(spec, "task_file"),
        )
    if not behaviors:
        raise EvaluatorReconstructionError("behaviors cannot be empty")
    return behaviors


def _build_evaluator(payload: dict[str, Any], path: Path) -> EvaluatorReconstruction:
    provenance = _mapping(payload.get("provenance"), "provenance")
    return EvaluatorReconstruction(
        evaluator_id=_string(payload, "evaluator_id"),
        config_sha256=file_sha256(path),
        scientific_use=_false_flag(payload, "scientific_use"),
        starts_experiment_timer=_false_flag(payload, "starts_experiment_timer"),
        provider_execution_allowed=_false_flag(payload, "provider_execution_allowed"),
        judge_model=_string(payload, "judge_model"),
        judge_model_revision=_string(payload, "judge_model_revision"),
        judge_model_revision_source=_string(payload, "judge_model_revision_source"),
        judge_model_weights_and_config_fixed=_true_flag(
            payload, "judge_model_weights_and_config_fixed"
        ),
        serving_infrastructure_fixed=_false_flag(
            payload, "serving_infrastructure_fixed"
        ),
        max_tokens=_positive_integer(payload, "max_tokens"),
        system_prompt=_string(payload, "system_prompt"),
        user_prompt_template=_string(payload, "user_prompt_template"),
        score_pattern=_string(payload, "score_pattern"),
        behaviors=_behaviors(payload.get("behaviors")),
        public_scorer_sha256=_public_scorer_sha256(provenance),
        inspect_ai_version=_string(payload, "inspect_ai_version"),
        divergences=_string_array(payload.get("divergences"), "divergences"),
    )


def load_evaluator_reconstruction(path: Path) -> EvaluatorReconstruction:
    try:
        payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "evaluator")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluatorReconstructionError(f"cannot load evaluator: {error}") from error
    _validate_envelope(payload)
    return _build_evaluator(payload, path)


def compile_judge_request(
    config: EvaluatorReconstruction,
    *,
    behavior: str,
    prompt: str,
    response: str,
) -> JudgeRequest:
    try:
        rubric = config.behaviors[behavior]
    except KeyError as error:
        raise EvaluatorReconstructionError(f"unknown behavior: {behavior}") from error
    anchors = "\n".join(
        f"  {score}: {description}" for score, description in rubric.scale_anchors
    )
    user = config.user_prompt_template.format(
        prompt=prompt,
        response=response,
        rubric=rubric.rubric,
        anchors=anchors,
    )
    return JudgeRequest(
        judge_model=config.judge_model,
        max_tokens=config.max_tokens,
        system=config.system_prompt,
        user=user,
    )


def parse_judge_score(config: EvaluatorReconstruction, output: str) -> int:
    match = re.search(config.score_pattern, output, flags=re.MULTILINE)
    if match is None or len(_SCORE_LINE_PATTERN.findall(output)) != 1:
        raise EvaluatorReconstructionError("judge output lacks a strict final score")
    return int(match.group(1))


def _archive_payloads(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        with zipfile.ZipFile(path) as archive:
            header = _mapping(json.loads(archive.read("_journal/start.json")), "header")
            names = sorted(
                name
                for name in archive.namelist()
                if name.startswith("samples/") and name.endswith(".json")
            )
            samples = [
                _mapping(json.loads(archive.read(name)), f"sample {name}")
                for name in names
            ]
    except (
        OSError,
        KeyError,
        UnicodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        raise EvaluatorReconstructionError(
            f"cannot read archive {path}: {error}"
        ) from error
    return header, samples


def _verify_header(
    header: dict[str, Any], config: EvaluatorReconstruction, spec: RubricSpec
) -> None:
    evaluation = _mapping(header.get("eval"), "header.eval")
    revision = _mapping(evaluation.get("revision"), "header.eval.revision")
    packages = _mapping(evaluation.get("packages"), "header.eval.packages")
    observed = (
        evaluation.get("task_file"),
        revision.get("commit"),
        revision.get("dirty"),
        packages.get("inspect_ai"),
    )
    expected = (spec.task_file, spec.source_commit, True, config.inspect_ai_version)
    if observed != expected:
        raise EvaluatorReconstructionError("archive header does not match config")


def _judge_event(sample: dict[str, Any], judge_model: str) -> dict[str, Any]:
    raw_events = sample.get("events")
    events = cast(list[object], raw_events) if isinstance(raw_events, list) else []
    matches: list[dict[str, Any]] = []
    for raw_event in events:
        event = _mapping(raw_event, "sample event")
        if event.get("event") == "model" and event.get("model") == judge_model:
            matches.append(event)
    if len(matches) != 1:
        raise EvaluatorReconstructionError("sample must contain one judge event")
    return matches[0]


def _message_contents(event: dict[str, Any]) -> tuple[str, str]:
    raw_messages = event.get("input")
    messages = (
        cast(list[object], raw_messages) if isinstance(raw_messages, list) else []
    )
    if len(messages) != 2:
        raise EvaluatorReconstructionError("judge event must contain two messages")
    system = _mapping(messages[0], "judge system message")
    user = _mapping(messages[1], "judge user message")
    if (system.get("role"), user.get("role")) != ("system", "user"):
        raise EvaluatorReconstructionError(
            "judge message roles must be system then user"
        )
    return _string(system, "content"), _string(user, "content")


def _sample_fields(sample: dict[str, Any]) -> tuple[str, str, dict[str, Any], int]:
    output = _mapping(sample.get("output"), "sample.output")
    metadata = _mapping(sample.get("metadata"), "sample.metadata")
    scores = _mapping(sample.get("scores"), "sample.scores")
    score = _mapping(scores.get("rubric_scorer"), "sample rubric score").get("value")
    if type(score) is not int:
        raise EvaluatorReconstructionError("archived score must be an integer")
    return (
        _string(sample, "input"),
        _string(output, "completion"),
        metadata,
        score,
    )


def _verify_sample(
    sample: dict[str, Any], config: EvaluatorReconstruction, behavior: str
) -> None:
    prompt, response, metadata, archived_score = _sample_fields(sample)
    spec = config.behaviors[behavior]
    anchors = _mapping(metadata.get("scale_anchors"), "sample scale anchors")
    if (
        metadata.get("rubric") != spec.rubric
        or tuple(anchors.items()) != spec.scale_anchors
    ):
        raise EvaluatorReconstructionError("sample rubric does not match config")
    event = _judge_event(sample, config.judge_model)
    request = compile_judge_request(
        config, behavior=behavior, prompt=prompt, response=response
    )
    if _message_contents(event) != (request.system, request.user):
        raise EvaluatorReconstructionError("archived judge request does not match")
    event_config = _mapping(event.get("config"), "judge event config")
    judge_output = _mapping(event.get("output"), "judge event output")
    if event_config != {"max_tokens": request.max_tokens}:
        raise EvaluatorReconstructionError("archived judge config does not match")
    if parse_judge_score(config, _string(judge_output, "completion")) != archived_score:
        raise EvaluatorReconstructionError("archived score does not match strict parse")


def _verify_archive(
    config: EvaluatorReconstruction, behavior: str, path: Path
) -> ArchiveVerification:
    spec = config.behaviors[behavior]
    archive_sha = file_sha256(path)
    if archive_sha != spec.source_archive_sha256:
        raise EvaluatorReconstructionError(f"{behavior} archive hash mismatch")
    header, samples = _archive_payloads(path)
    _verify_header(header, config, spec)
    if len(samples) != spec.expected_samples:
        raise EvaluatorReconstructionError(f"{behavior} sample count mismatch")
    for sample in samples:
        _verify_sample(sample, config, behavior)
    return ArchiveVerification(
        behavior=behavior,
        archive_sha256=archive_sha,
        sample_count=len(samples),
        source_commit=spec.source_commit,
        task_file=spec.task_file,
    )


def _receipt_body(
    evaluator_id: str,
    config_sha256: str,
    judge_model_revision: str,
    judge_model_revision_source: str,
    public_source_sha256: str,
    archives: tuple[ArchiveVerification, ...],
) -> dict[str, object]:
    sample_count = sum(archive.sample_count for archive in archives)
    return {
        "schema_version": 1,
        "evaluator_id": evaluator_id,
        "config_sha256": config_sha256,
        "judge_model_revision": judge_model_revision,
        "judge_model_revision_source": judge_model_revision_source,
        "judge_model_weights_and_config_fixed": True,
        "serving_infrastructure_fixed": False,
        "public_source_sha256": public_source_sha256,
        "archives": [archive.to_dict() for archive in archives],
        "sample_count": sample_count,
        "request_match_count": sample_count,
        "strict_parse_match_count": sample_count,
        "scientific_use": False,
        "starts_experiment_timer": False,
        "provider_execution_allowed": False,
    }


def verify_evaluator_reconstruction(
    config: EvaluatorReconstruction,
    *,
    public_source_path: Path,
    archive_paths: dict[str, Path],
) -> EvaluatorReconstructionReceipt:
    public_source_sha = file_sha256(public_source_path)
    if public_source_sha != config.public_scorer_sha256:
        raise EvaluatorReconstructionError("public scorer hash mismatch")
    if set(archive_paths) != set(config.behaviors):
        raise EvaluatorReconstructionError("archive behaviors do not match config")
    archives = tuple(
        _verify_archive(config, behavior, archive_paths[behavior])
        for behavior in sorted(config.behaviors)
    )
    return _build_receipt(config, public_source_sha, archives)


def _build_receipt(
    config: EvaluatorReconstruction,
    public_source_sha: str,
    archives: tuple[ArchiveVerification, ...],
) -> EvaluatorReconstructionReceipt:
    body = _receipt_body(
        config.evaluator_id,
        config.config_sha256,
        config.judge_model_revision,
        config.judge_model_revision_source,
        public_source_sha,
        archives,
    )
    return EvaluatorReconstructionReceipt(
        evaluator_id=config.evaluator_id,
        config_sha256=config.config_sha256,
        judge_model_revision=config.judge_model_revision,
        judge_model_revision_source=config.judge_model_revision_source,
        judge_model_weights_and_config_fixed=True,
        serving_infrastructure_fixed=False,
        public_source_sha256=public_source_sha,
        archives=archives,
        sample_count=cast(int, body["sample_count"]),
        request_match_count=cast(int, body["request_match_count"]),
        strict_parse_match_count=cast(int, body["strict_parse_match_count"]),
        receipt_sha256=canonical_sha256(body),
    )


def write_evaluator_reconstruction_receipt(
    receipt: EvaluatorReconstructionReceipt, path: Path
) -> None:
    try:
        with path.open("x", encoding="utf-8") as target:
            json.dump(
                receipt.to_dict(),
                target,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            target.write("\n")
    except FileExistsError as error:
        raise EvaluatorReconstructionError(f"output already exists: {path}") from error
