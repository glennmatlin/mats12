"""Receipt-bound reconstruction of the public ordinary EKFAC method."""

from __future__ import annotations

import json
import math
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from concept_filtering.generation_evidence import (
    GenerationEvidenceError,
    parse_generation_evidence_bytes,
)
from concept_filtering.inspect_archive import InspectArchiveValidationError
from concept_filtering.isolated_io import (
    IsolatedIOError,
    publish_new_bytes,
    read_isolated_bytes,
)
from concept_filtering.prompt_bank import parse_prompt_bank_receipt_bytes
from concept_filtering.serialization import (
    bytes_sha256,
    canonical_sha256,
    json_equal_exact,
    json_loads_unique,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_TOP_FIELDS = {
    "schema_version",
    "reconstruction_id",
    "phase",
    "method",
    "source",
    "implementation",
    "settings",
    "query_evidence",
    "shared_backend",
    "observations",
    "boundaries",
    "divergences",
}
_SOURCE_FIELDS = {
    "study_post_url",
    "results_repository",
    "results_revision",
    "study_code_repository",
    "study_code_revision",
    "study_code_access",
    "evidence_files",
}
_IMPLEMENTATION_FIELDS = {
    "package",
    "version",
    "source_repository",
    "source_revision",
    "classification",
}
_EVIDENCE_ROLES = {
    "testbed_readme",
    "testbed_manifest",
    "timing",
    "testbed_scores",
}
_DECISION_NAMES = {
    "query_loss",
    "on_policy_response_selection",
    "parameter_and_layer_coverage",
    "factor_sample",
    "damping",
    "sequence_policy",
    "query_gradient_accumulation",
    "score_aggregation",
}
_BOUNDARY_FIELDS = {
    "scientific_use",
    "starts_experiment_timer",
    "sae_feature_selection_allowed",
    "concept_scoring_allowed",
    "provider_execution_allowed",
    "scheduler_execution_allowed",
    "concept_selected_training_allowed",
}
_SHARED_BACKEND = {
    "query_interface": "scalar_measurement_gradient",
    "ordinary_objective_kind": "behavior_loss",
    "future_objective_kind": "causal_concept_activation",
    "future_objective_status": "blocked_until_causal_feature_receipt",
    "factor_semantics_change_allowed": False,
}
_OBSERVATION_FIELDS = {
    "implementation_family",
    "factor_reuse",
    "testbed_gpu_shape",
    "testbed_wall_time_seconds",
    "testbed_score_count",
}
_GENERATION_BINDING_FIELDS = {
    "path",
    "file_sha256",
    "evidence_id",
    "evidence_sha256",
    "generation_id",
    "aggregate_receipt_sha256",
    "sample_count",
    "samples_sha256",
    "completion_hashes_sha256",
    "behaviors",
}
_PROMPT_BINDING_FIELDS = {
    "path",
    "file_sha256",
    "receipt_sha256",
    "prompt_bank_sha256",
    "question_count",
}
_QUERY_BEHAVIORS = {"validate_feelings": 100, "refusal": 101}
_RESPONSE_SELECTION_FIELDS = {
    "generation_evidence_id",
    "generation_id",
    "do_sample",
    "max_new_tokens",
    "selection",
    "judge_based_selection",
    "validate_feelings_count",
    "refusal_count",
}


class EKFACReconstructionError(ValueError):
    """Raised when the EKFAC reconstruction or its evidence is invalid."""


@dataclass(frozen=True)
class EvidenceFile:
    role: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ReconstructionDecision:
    name: str
    value: object
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": "reconstruction_choice",
            "value": self.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class EKFACReconstruction:
    reconstruction_id: str
    config_path: Path
    config_sha256: str
    study_post_url: str
    results_repository: str
    results_revision: str
    study_code_repository: str
    study_code_revision: str
    study_code_access: str
    implementation_version: str
    implementation_source_revision: str
    evidence_files: tuple[EvidenceFile, ...]
    decisions: tuple[ReconstructionDecision, ...]
    query_evidence: dict[str, object]
    observations: dict[str, object]
    scientific_use: bool
    concept_scoring_allowed: bool

    @property
    def decision_names(self) -> tuple[str, ...]:
        return tuple(sorted(item.name for item in self.decisions))


@dataclass(frozen=True)
class EKFACReconstructionReceipt:
    body: dict[str, object]
    receipt_sha256: str

    @property
    def evidence_file_count(self) -> int:
        return cast(int, self.body["evidence_file_count"])

    @property
    def testbed_score_count(self) -> int:
        return cast(int, self.body["testbed_score_count"])

    @property
    def finite_score_count(self) -> int:
        return cast(int, self.body["finite_score_count"])

    @property
    def unique_document_count(self) -> int:
        return cast(int, self.body["unique_document_count"])

    @property
    def study_code_access(self) -> str:
        return cast(str, self.body["study_code_access"])

    def to_dict(self) -> dict[str, object]:
        return {**self.body, "receipt_sha256": self.receipt_sha256}


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EKFACReconstructionError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _check_fields(value: dict[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise EKFACReconstructionError(
            f"{label} unexpected fields={unexpected}, missing fields={missing}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EKFACReconstructionError(f"{label} must be a non-empty string")
    return value


def _revision(value: object, label: str) -> str:
    revision = _string(value, label)
    if _REVISION.fullmatch(revision) is None:
        raise EKFACReconstructionError(f"{label} must be a 40-hex revision")
    return revision


def _sha256(value: object, label: str) -> str:
    sha256 = _string(value, label)
    if _SHA256.fullmatch(sha256) is None:
        raise EKFACReconstructionError(f"{label} must be a SHA-256")
    return sha256


def _parse_json_bytes(value: bytes, label: str) -> dict[str, object]:
    try:
        parsed = json_loads_unique(value.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise EKFACReconstructionError(f"cannot parse {label}: {error}") from error
    return _object(parsed, label)


def _load_payload(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw = read_isolated_bytes(path, "EKFAC reconstruction")
        payload = _parse_json_bytes(raw, "EKFAC reconstruction")
        config_sha256 = canonical_sha256(payload)
    except (IsolatedIOError, OSError, TypeError, ValueError, RecursionError) as error:
        raise EKFACReconstructionError(
            f"cannot load reconstruction: {error}"
        ) from error
    _check_fields(payload, _TOP_FIELDS, "reconstruction")
    return payload, config_sha256


def _safe_path(value: object, label: str) -> str:
    path = _string(value, label)
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or parsed.as_posix() != path or ".." in parsed.parts:
        raise EKFACReconstructionError(f"{label} must be a canonical relative path")
    return path


def _behavior_bindings(value: object) -> dict[str, object]:
    payload = _object(value, "query_evidence.generation_evidence.behaviors")
    _check_fields(payload, set(_QUERY_BEHAVIORS), "query evidence behaviors")
    for behavior, count in _QUERY_BEHAVIORS.items():
        summary = _object(payload[behavior], f"query evidence {behavior}")
        _check_fields(
            summary,
            {"sample_count", "completion_hashes_sha256"},
            f"query evidence {behavior}",
        )
        if summary["sample_count"] != count or type(summary["sample_count"]) is not int:
            raise EKFACReconstructionError(f"{behavior} sample_count is invalid")
        _sha256(summary["completion_hashes_sha256"], f"{behavior} completions")
    return payload


def _generation_binding(value: object) -> dict[str, object]:
    payload = _object(value, "query_evidence.generation_evidence")
    _check_fields(payload, _GENERATION_BINDING_FIELDS, "generation evidence binding")
    _safe_path(payload["path"], "generation evidence path")
    for field in (
        "file_sha256",
        "evidence_sha256",
        "aggregate_receipt_sha256",
        "samples_sha256",
        "completion_hashes_sha256",
    ):
        _sha256(payload[field], f"generation evidence {field}")
    expected = ("public-rank32-full-bank-generation-evidence-v1", 201)
    observed = (payload["evidence_id"], payload["sample_count"])
    if observed != expected or type(payload["sample_count"]) is not int:
        raise EKFACReconstructionError("generation evidence identity is invalid")
    if payload["generation_id"] != "public-rank32-pinned-prompt-banks-v1":
        raise EKFACReconstructionError("generation_id is invalid")
    _behavior_bindings(payload["behaviors"])
    return payload


def _prompt_binding(behavior: str, value: object) -> dict[str, object]:
    payload = _object(value, f"query_evidence.prompt_bank_receipts.{behavior}")
    _check_fields(payload, _PROMPT_BINDING_FIELDS, f"{behavior} prompt binding")
    _safe_path(payload["path"], f"{behavior} prompt receipt path")
    for field in ("file_sha256", "receipt_sha256", "prompt_bank_sha256"):
        _sha256(payload[field], f"{behavior} prompt {field}")
    count = payload["question_count"]
    if count != _QUERY_BEHAVIORS[behavior] or type(count) is not int:
        raise EKFACReconstructionError(f"{behavior} prompt count is invalid")
    return payload


def _query_evidence(value: object) -> dict[str, object]:
    payload = _object(value, "query_evidence")
    _check_fields(
        payload, {"generation_evidence", "prompt_bank_receipts"}, "query_evidence"
    )
    generation = _generation_binding(payload["generation_evidence"])
    prompts = _object(payload["prompt_bank_receipts"], "prompt_bank_receipts")
    _check_fields(prompts, set(_QUERY_BEHAVIORS), "prompt_bank_receipts")
    for behavior in _QUERY_BEHAVIORS:
        prompt = _prompt_binding(behavior, prompts[behavior])
        summary = cast(dict[str, object], generation["behaviors"])[behavior]
        if prompt["question_count"] != cast(dict[str, object], summary)["sample_count"]:
            raise EKFACReconstructionError("query evidence behavior counts differ")
    return payload


def _evidence_file(role: str, value: object) -> EvidenceFile:
    payload = _object(value, f"evidence_files.{role}")
    _check_fields(payload, {"path", "sha256"}, f"evidence_files.{role}")
    return EvidenceFile(
        role=role,
        path=_safe_path(payload["path"], f"evidence_files.{role}.path"),
        sha256=_sha256(payload["sha256"], f"evidence_files.{role}.sha256"),
    )


def _evidence_files(value: object) -> tuple[EvidenceFile, ...]:
    payload = _object(value, "source.evidence_files")
    _check_fields(payload, _EVIDENCE_ROLES, "source.evidence_files")
    files = tuple(_evidence_file(role, payload[role]) for role in sorted(payload))
    if len({item.path for item in files}) != len(files):
        raise EKFACReconstructionError("evidence file paths must be unique")
    return files


def _decision(name: str, value: object) -> ReconstructionDecision:
    payload = _object(value, f"settings.{name}")
    expected = {"classification", "value", "rationale"}
    _check_fields(payload, expected, f"settings.{name}")
    if payload["classification"] != "reconstruction_choice":
        raise EKFACReconstructionError(f"settings.{name} must be classified")
    if payload["value"] is None:
        raise EKFACReconstructionError(f"settings.{name}.value cannot be null")
    return ReconstructionDecision(
        name=name,
        value=payload["value"],
        rationale=_string(payload["rationale"], f"settings.{name}.rationale"),
    )


def _decisions(value: object) -> tuple[ReconstructionDecision, ...]:
    payload = _object(value, "settings")
    _check_fields(payload, _DECISION_NAMES, "settings")
    return tuple(_decision(name, payload[name]) for name in sorted(payload))


def _validate_response_selection(
    decisions: tuple[ReconstructionDecision, ...], query: dict[str, object]
) -> None:
    decision = next(
        item for item in decisions if item.name == "on_policy_response_selection"
    )
    value = _object(decision.value, "on_policy_response_selection.value")
    _check_fields(
        value, _RESPONSE_SELECTION_FIELDS, "on_policy_response_selection.value"
    )
    generation = cast(dict[str, object], query["generation_evidence"])
    expected = {
        "generation_evidence_id": generation["evidence_id"],
        "generation_id": generation["generation_id"],
        "do_sample": False,
        "max_new_tokens": 1536,
        "selection": "all_successful_completions_in_canonical_prompt_order",
        "judge_based_selection": False,
        "validate_feelings_count": 100,
        "refusal_count": 101,
    }
    if not json_equal_exact(value, expected):
        raise EKFACReconstructionError("on-policy response evidence differs")


def _validate_implementation(value: object) -> tuple[str, str]:
    payload = _object(value, "implementation")
    _check_fields(payload, _IMPLEMENTATION_FIELDS, "implementation")
    expected = (
        "kronfluence",
        "1.0.1",
        "https://github.com/pomonam/kronfluence",
        "87685e90d6972df314eff635440a78eac54ae47c",
        "reconstruction_choice",
    )
    observed = tuple(
        payload[field]
        for field in (
            "package",
            "version",
            "source_repository",
            "source_revision",
            "classification",
        )
    )
    if observed != expected:
        raise EKFACReconstructionError("implementation classification is invalid")
    return "1.0.1", _revision(
        payload["source_revision"], "implementation.source_revision"
    )


def _validate_observations(value: object) -> dict[str, object]:
    payload = _object(value, "observations")
    _check_fields(payload, _OBSERVATION_FIELDS, "observations")
    if (payload["implementation_family"], payload["factor_reuse"]) != (
        "kronfluence",
        "behavior_agnostic",
    ):
        raise EKFACReconstructionError("source observations are invalid")
    if payload["testbed_gpu_shape"] != "H200:8":
        raise EKFACReconstructionError("testbed_gpu_shape must be H200:8")
    _positive_number(payload["testbed_wall_time_seconds"], "wall time")
    _positive_integer(payload["testbed_score_count"], "testbed_score_count")
    return payload


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EKFACReconstructionError(f"{label} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise EKFACReconstructionError(f"{label} must be a positive number")
    return result


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise EKFACReconstructionError(f"{label} must be a positive integer")
    return value


def _validate_boundaries(value: object) -> dict[str, object]:
    payload = _object(value, "boundaries")
    _check_fields(payload, _BOUNDARY_FIELDS, "boundaries")
    for field in _BOUNDARY_FIELDS:
        if payload[field] is not False:
            raise EKFACReconstructionError(f"{field} must remain false")
    return payload


def _divergences(value: object) -> tuple[str, ...]:
    raw = cast(list[object], value) if isinstance(value, list) else []
    divergences = tuple(item for item in raw if isinstance(item, str) and item)
    if not divergences or len(divergences) != len(raw):
        raise EKFACReconstructionError("divergences must contain strings")
    return divergences


def _validate_envelope(payload: dict[str, object]) -> None:
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise EKFACReconstructionError("schema_version must be the integer 1")
    if (payload["phase"], payload["method"]) != ("pre_feature", "ekfac"):
        raise EKFACReconstructionError("phase and method must remain pre_feature EKFAC")
    shared = _object(payload["shared_backend"], "shared_backend")
    if shared != _SHARED_BACKEND:
        raise EKFACReconstructionError("shared_backend contract is invalid")
    _divergences(payload["divergences"])


def _source_identity(source: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        _string(source["study_post_url"], "study_post_url"),
        _string(source["results_repository"], "results_repository"),
        _revision(source["results_revision"], "results_revision"),
        _string(source["study_code_repository"], "code repo"),
        _revision(source["study_code_revision"], "code revision"),
    )


def _build_inputs(
    payload: dict[str, object],
) -> tuple[
    dict[str, object],
    str,
    str,
    dict[str, object],
    tuple[ReconstructionDecision, ...],
    dict[str, object],
]:
    source = _object(payload["source"], "source")
    _check_fields(source, _SOURCE_FIELDS, "source")
    if source["study_code_access"] != "unavailable":
        raise EKFACReconstructionError("study_code_access must remain unavailable")
    version, revision = _validate_implementation(payload["implementation"])
    boundaries = _validate_boundaries(payload["boundaries"])
    decisions = _decisions(payload["settings"])
    query = _query_evidence(payload["query_evidence"])
    _validate_response_selection(decisions, query)
    return source, version, revision, boundaries, decisions, query


def _build_reconstruction(
    payload: dict[str, object], config_sha256: str, config_path: Path
) -> EKFACReconstruction:
    source, version, implementation_revision, boundaries, decisions, query = (
        _build_inputs(payload)
    )
    post, results, revision, code, code_revision = _source_identity(source)
    return EKFACReconstruction(
        reconstruction_id=_string(payload["reconstruction_id"], "reconstruction_id"),
        config_path=config_path.absolute(),
        config_sha256=config_sha256,
        study_post_url=post,
        results_repository=results,
        results_revision=revision,
        study_code_repository=code,
        study_code_revision=code_revision,
        study_code_access="unavailable",
        implementation_version=version,
        implementation_source_revision=implementation_revision,
        evidence_files=_evidence_files(source["evidence_files"]),
        decisions=decisions,
        query_evidence=query,
        observations=_validate_observations(payload["observations"]),
        scientific_use=cast(bool, boundaries["scientific_use"]),
        concept_scoring_allowed=cast(bool, boundaries["concept_scoring_allowed"]),
    )


def load_ekfac_reconstruction(path: Path) -> EKFACReconstruction:
    payload, config_sha256 = _load_payload(path)
    _validate_envelope(payload)
    return _build_reconstruction(payload, config_sha256, path)


def _validate_config_identity(config: EKFACReconstruction) -> None:
    reloaded = load_ekfac_reconstruction(config.config_path)
    if reloaded != config:
        raise EKFACReconstructionError(
            "EKFAC reconstruction differs from its canonical file"
        )


def _read_evidence(config: EKFACReconstruction, source_root: Path) -> dict[str, bytes]:
    try:
        status = source_root.lstat()
    except OSError as error:
        raise EKFACReconstructionError(f"cannot read source root: {error}") from error
    if not stat.S_ISDIR(status.st_mode):
        raise EKFACReconstructionError("source root must be a plain directory")
    values: dict[str, bytes] = {}
    for evidence in config.evidence_files:
        values[evidence.role] = _read_evidence_file(source_root, evidence)
    return values


def _read_evidence_file(source_root: Path, evidence: EvidenceFile) -> bytes:
    path = source_root / evidence.path
    try:
        value = read_isolated_bytes(path, evidence.role)
    except (IsolatedIOError, OSError) as error:
        raise EKFACReconstructionError(
            f"cannot read {evidence.role}: {error}"
        ) from error
    if bytes_sha256(value) != evidence.sha256:
        raise EKFACReconstructionError(f"{evidence.role} hash mismatch")
    return value


def _verify_readme(config: EKFACReconstruction, value: bytes) -> None:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EKFACReconstructionError("testbed README is not UTF-8") from error
    required = ("EKFAC", config.study_code_revision, "H200:8")
    if any(item not in text for item in required):
        raise EKFACReconstructionError("testbed README does not bind source facts")


def _verify_manifest(config: EKFACReconstruction, value: bytes) -> None:
    payload = _parse_json_bytes(value, "testbed manifest")
    gpu_shapes = _object(payload.get("gpu_shapes"), "manifest.gpu_shapes")
    row_counts = _object(payload.get("row_counts"), "manifest.row_counts")
    expected_count = config.observations["testbed_score_count"]
    observed = (
        payload.get("dare_commit"),
        "H200:8" in _string(gpu_shapes.get("ekfac_attribution"), "GPU shape"),
        row_counts.get("original_rows"),
    )
    if observed != (config.study_code_revision, True, expected_count):
        raise EKFACReconstructionError("testbed manifest facts do not match config")


def _verify_timing(config: EKFACReconstruction, value: bytes) -> None:
    payload = _parse_json_bytes(value, "EKFAC timing")
    observed = _positive_number(payload.get("total_wall_time_sec"), "wall time")
    expected = _positive_number(
        config.observations["testbed_wall_time_seconds"], "expected wall time"
    )
    if observed != expected:
        raise EKFACReconstructionError("EKFAC wall time does not match config")


def _score_record(value: str, line_number: int) -> tuple[str, float]:
    payload = _parse_json_bytes(value.encode(), f"score line {line_number}")
    if payload.get("method") != "ekfac":
        raise EKFACReconstructionError("score method must be ekfac")
    doc_id = _string(payload.get("doc_id"), f"score line {line_number} doc_id")
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise EKFACReconstructionError("EKFAC score must be numeric")
    if not math.isfinite(float(score)):
        raise EKFACReconstructionError("EKFAC score must be finite")
    return doc_id, float(score)


def _verify_scores(config: EKFACReconstruction, value: bytes) -> tuple[int, int]:
    try:
        lines = value.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise EKFACReconstructionError("score evidence is not UTF-8") from error
    records = tuple(_score_record(line, index) for index, line in enumerate(lines, 1))
    expected = cast(int, config.observations["testbed_score_count"])
    if len(records) != expected:
        raise EKFACReconstructionError("testbed score count does not match config")
    document_ids = {doc_id for doc_id, _ in records}
    if len(document_ids) != len(records):
        raise EKFACReconstructionError("testbed document IDs must be unique")
    return len(records), len(document_ids)


def _read_query_file(root: Path, binding: dict[str, object], label: str) -> bytes:
    path = root / cast(str, binding["path"])
    try:
        value = read_isolated_bytes(path, label)
    except (IsolatedIOError, OSError) as error:
        raise EKFACReconstructionError(f"cannot read {label}: {error}") from error
    if bytes_sha256(value) != binding["file_sha256"]:
        raise EKFACReconstructionError(f"{label} file hash mismatch")
    return value


def _verify_generation_binding(root: Path, binding: dict[str, object]) -> None:
    value = _read_query_file(root, binding, "generation evidence")
    try:
        payload = parse_generation_evidence_bytes(value)
    except GenerationEvidenceError as error:
        raise EKFACReconstructionError(str(error)) from error
    expected = {
        key: item for key, item in binding.items() if key not in {"path", "file_sha256"}
    }
    observed = {key: payload.get(key) for key in expected}
    if not json_equal_exact(observed, expected):
        raise EKFACReconstructionError("generation evidence binding differs")


def _verify_prompt_binding(
    root: Path, behavior: str, binding: dict[str, object]
) -> None:
    value = _read_query_file(root, binding, f"{behavior} prompt receipt")
    try:
        receipt = parse_prompt_bank_receipt_bytes(value)
    except InspectArchiveValidationError as error:
        raise EKFACReconstructionError(str(error)) from error
    observed = (
        receipt.behavior,
        receipt.receipt_sha256,
        receipt.output_sha256,
        receipt.question_count,
    )
    expected = (
        behavior,
        binding["receipt_sha256"],
        binding["prompt_bank_sha256"],
        binding["question_count"],
    )
    if not json_equal_exact(list(observed), list(expected)):
        raise EKFACReconstructionError(f"{behavior} prompt binding differs")


def _verify_query_evidence(config: EKFACReconstruction, root: Path) -> None:
    try:
        status = root.lstat()
    except OSError as error:
        raise EKFACReconstructionError(
            f"cannot read query evidence root: {error}"
        ) from error
    if not stat.S_ISDIR(status.st_mode):
        raise EKFACReconstructionError("query evidence root must be a plain directory")
    generation = cast(dict[str, object], config.query_evidence["generation_evidence"])
    _verify_generation_binding(root, generation)
    prompts = cast(dict[str, object], config.query_evidence["prompt_bank_receipts"])
    for behavior in sorted(_QUERY_BEHAVIORS):
        _verify_prompt_binding(
            root, behavior, cast(dict[str, object], prompts[behavior])
        )


def _receipt_body(
    config: EKFACReconstruction, score_count: int, document_count: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "reconstruction_id": config.reconstruction_id,
        "config_sha256": config.config_sha256,
        "results_repository": config.results_repository,
        "results_revision": config.results_revision,
        "study_code_repository": config.study_code_repository,
        "study_code_revision": config.study_code_revision,
        "study_code_access": config.study_code_access,
        "implementation": "kronfluence",
        "implementation_version": config.implementation_version,
        "implementation_source_revision": config.implementation_source_revision,
        "evidence_files": [item.to_dict() for item in config.evidence_files],
        "evidence_file_count": len(config.evidence_files),
        "decisions": [item.to_dict() for item in config.decisions],
        "query_evidence": config.query_evidence,
        "testbed_score_count": score_count,
        "finite_score_count": score_count,
        "unique_document_count": document_count,
        "scientific_use": False,
        "starts_experiment_timer": False,
        "concept_scoring_allowed": False,
        "provider_execution_allowed": False,
        "scheduler_execution_allowed": False,
    }


def verify_ekfac_reconstruction(
    config: EKFACReconstruction, *, source_root: Path, query_evidence_root: Path
) -> EKFACReconstructionReceipt:
    _validate_config_identity(config)
    evidence = _read_evidence(config, source_root)
    _verify_readme(config, evidence["testbed_readme"])
    _verify_manifest(config, evidence["testbed_manifest"])
    _verify_timing(config, evidence["timing"])
    score_count, document_count = _verify_scores(config, evidence["testbed_scores"])
    _verify_query_evidence(config, query_evidence_root)
    body = _receipt_body(config, score_count, document_count)
    return EKFACReconstructionReceipt(body, canonical_sha256(body))


def write_ekfac_reconstruction_receipt(
    receipt: EKFACReconstructionReceipt, path: Path
) -> None:
    try:
        value = json.dumps(receipt.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
        publish_new_bytes(path, value, "EKFAC reconstruction receipt")
    except IsolatedIOError as error:
        raise EKFACReconstructionError(str(error)) from error
