"""Fail-closed contracts for resumable base-response generation."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

from concept_filtering.serialization import (
    bytes_sha256,
    canonical_sha256,
    file_sha256,
    json_loads_unique,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){2,}(?:[+.-][0-9A-Za-z.-]+)?$")
_BASE_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
}
_TOKENIZER_FILES = {
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
_CLOSED_BOUNDARIES = {
    "activation_extraction_allowed",
    "training_execution_allowed",
    "provider_execution_allowed",
    "scheduler_execution_allowed",
    "scientific_use",
    "starts_experiment_timer",
    "sae_feature_selection_allowed",
    "causal_intervention_allowed",
    "concept_scoring_allowed",
    "concept_selected_training_allowed",
}
_BOUNDARY_FIELDS = {"base_response_generation_allowed", *_CLOSED_BOUNDARIES}
_ROOT_FIELDS = {
    "schema_version",
    "generation_id",
    "phase",
    "boundaries",
    "package",
    "b2_gate",
    "base",
    "tokenizer",
    "generation",
    "output",
    "runtime_versions",
    "limitations",
}
_PACKAGE_RECEIPT_FIELDS = {
    "schema_version",
    "package_id",
    "phase",
    "config_sha256",
    "boundaries",
    "sample",
    "tokenizer",
    "runtime_versions",
    "training",
    "package_format",
    "entry_schema_version",
    "row_count",
    "retained_row_count",
    "dropped_overlength_count",
    "multi_turn_row_count",
    "empty_initial_user_count",
    "total_input_token_count",
    "total_supervised_token_count",
    "retained_input_token_count",
    "retained_supervised_token_count",
    "ordered_entry_sha256",
    "identity_entry_sha256",
    "package_sha256",
    "releases",
    "limitations",
    "receipt_sha256",
}
# The v1 package receipt predates the causal-intervention boundary, so its
# boundary block binds the exact retained schema rather than _CLOSED_BOUNDARIES.
_PACKAGE_BOUNDARY_FIELDS = {
    "training_data_materialization_allowed",
    "base_response_generation_allowed",
    *(_CLOSED_BOUNDARIES - {"causal_intervention_allowed"}),
}
_B2_RECEIPT_FIELDS = {
    "schema_version",
    "smoke_id",
    "executor_revision",
    "authority",
    "recipe",
    "artifacts",
    "data",
    "checkpoint",
    "runtime",
    "training",
    "comparison",
    "receipt_sha256",
}
_B2_RECIPE_FIELDS = {
    "recipe_id",
    "config_sha256",
    "core_sha256",
    "condition_core_sha256",
    "seed",
}


class BaseResponseGenerationError(ValueError):
    """Raised when the base-response contract differs from its frozen form."""


@dataclass(frozen=True)
class ArtifactBinding:
    repository: str
    revision: str
    path: str | None
    files: Mapping[str, str]


@dataclass(frozen=True)
class PackageBinding:
    package_id: str
    receipt_sha256: str
    receipt_path: str
    evidence_release: str
    executor_release: str


@dataclass(frozen=True)
class B2GateBinding:
    smoke_id: str
    recipe_id: str
    recipe_core_sha256: str
    authority_sha256: str
    receipt_sha256: str
    receipt_path: str
    evidence_release: str


@dataclass(frozen=True)
class GenerationSettings:
    algorithm: str
    do_sample: bool
    batch_size: int
    max_total_tokens: int
    add_generation_prompt: bool
    stop_on_eos: bool
    require_terminal_eos_or_cap: bool
    generation_prompt: str


@dataclass(frozen=True)
class OutputSettings:
    sample_schema_version: int
    sample_format: str
    aggregate_schema_version: int
    shard_algorithm: str


@dataclass(frozen=True)
class BaseResponseConfig:
    generation_id: str
    phase: str
    boundaries: Mapping[str, bool]
    package: PackageBinding
    b2_gate: B2GateBinding
    base: ArtifactBinding
    tokenizer: ArtifactBinding
    generation: GenerationSettings
    output: OutputSettings
    runtime_versions: Mapping[str, str]
    limitations: tuple[str, ...]
    config_path: Path
    config_sha256: str


@dataclass(frozen=True)
class BaseResponseEvidence:
    package_receipt_sha256: str
    b2_receipt_sha256: str
    package_payload: Mapping[str, object]
    b2_payload: Mapping[str, object]


GENERATION_PROMPT_TEMPLATE_THINK = "template-think"
GENERATION_PROMPT_HEADER_ONLY = "assistant-header-only"
_REASONING_OPENER = "<think>"
_REASONING_PROMPT_SUFFIX = "<|im_start|>assistant\n" + _REASONING_OPENER
_GENERATION_SETTINGS = GenerationSettings(
    algorithm="greedy-batch-one-v1",
    do_sample=False,
    batch_size=1,
    max_total_tokens=8192,
    add_generation_prompt=True,
    stop_on_eos=True,
    require_terminal_eos_or_cap=True,
    generation_prompt=GENERATION_PROMPT_TEMPLATE_THINK,
)
_GENERATION_SETTINGS_HEADER_ONLY = replace(
    _GENERATION_SETTINGS, generation_prompt=GENERATION_PROMPT_HEADER_ONLY
)
_OUTPUT_SETTINGS = OutputSettings(
    sample_schema_version=1,
    sample_format="canonical-json-v1",
    aggregate_schema_version=1,
    shard_algorithm="ordinal-modulo-v1",
)


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaseResponseGenerationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        payload = cast(dict[str, object], value)
        return MappingProxyType(
            {field: _freeze_value(item) for field, item in payload.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in cast(list[object], value))
    return value


def _freeze_mapping(payload: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze_value(dict(payload)))


def _fields(payload: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(payload) != expected:
        raise BaseResponseGenerationError(f"{field} fields differ from schema")


def _string(payload: Mapping[str, object], field: str, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BaseResponseGenerationError(f"{context}.{field} must be non-empty")
    return value


def _digest(payload: Mapping[str, object], field: str, context: str) -> str:
    value = _string(payload, field, context)
    if _SHA256.fullmatch(value) is None:
        raise BaseResponseGenerationError(f"{context}.{field} must be a SHA-256")
    return value


def _revision(payload: Mapping[str, object], field: str, context: str) -> str:
    value = _string(payload, field, context)
    if _REVISION.fullmatch(value) is None:
        raise BaseResponseGenerationError(f"{context}.{field} must be a full revision")
    return value


def _relative_path(payload: Mapping[str, object], field: str, context: str) -> str:
    value = _string(payload, field, context)
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise BaseResponseGenerationError(f"{context}.{field} must be relative")
    return value


def _boundaries(value: object) -> Mapping[str, bool]:
    payload = _object(value, "boundaries")
    _fields(payload, _BOUNDARY_FIELDS, "boundaries")
    if payload.get("base_response_generation_allowed") is not True:
        raise BaseResponseGenerationError(
            "boundaries.base_response_generation_allowed must be true"
        )
    for field in sorted(_CLOSED_BOUNDARIES):
        if payload.get(field) is not False:
            raise BaseResponseGenerationError(f"boundaries.{field} must be false")
    return MappingProxyType(dict(cast(dict[str, bool], payload)))


def _files(value: object, expected: set[str], context: str) -> Mapping[str, str]:
    payload = _object(value, f"{context}.files")
    if set(payload) != expected:
        raise BaseResponseGenerationError(f"{context}.files differ from manifest")
    result = {field: _digest(payload, field, f"{context}.files") for field in payload}
    return MappingProxyType(dict(sorted(result.items())))


def _artifact(
    value: object, context: str, expected: set[str], *, requires_path: bool
) -> ArtifactBinding:
    payload = _object(value, context)
    fields = {"repository", "revision", "files"}
    if requires_path:
        fields.add("path")
    _fields(payload, fields, context)
    path = _string(payload, "path", context) if requires_path else None
    return ArtifactBinding(
        _string(payload, "repository", context),
        _revision(payload, "revision", context),
        path,
        _files(payload.get("files"), expected, context),
    )


def _package(value: object) -> PackageBinding:
    payload = _object(value, "package")
    _fields(
        payload,
        {
            "package_id",
            "receipt_sha256",
            "receipt_path",
            "evidence_release",
            "executor_release",
        },
        "package",
    )
    return PackageBinding(
        _string(payload, "package_id", "package"),
        _digest(payload, "receipt_sha256", "package"),
        _relative_path(payload, "receipt_path", "package"),
        _revision(payload, "evidence_release", "package"),
        _revision(payload, "executor_release", "package"),
    )


def _b2_gate(value: object) -> B2GateBinding:
    payload = _object(value, "b2_gate")
    _fields(
        payload,
        {
            "smoke_id",
            "recipe_id",
            "recipe_core_sha256",
            "authority_sha256",
            "receipt_sha256",
            "receipt_path",
            "evidence_release",
        },
        "b2_gate",
    )
    return B2GateBinding(
        _string(payload, "smoke_id", "b2_gate"),
        _string(payload, "recipe_id", "b2_gate"),
        _digest(payload, "recipe_core_sha256", "b2_gate"),
        _digest(payload, "authority_sha256", "b2_gate"),
        _digest(payload, "receipt_sha256", "b2_gate"),
        _relative_path(payload, "receipt_path", "b2_gate"),
        _revision(payload, "evidence_release", "b2_gate"),
    )


def _exact_values(
    payload: Mapping[str, object], expected: Mapping[str, object], context: str
) -> None:
    for field, value in expected.items():
        if payload.get(field) != value or type(payload.get(field)) is not type(value):
            raise BaseResponseGenerationError(f"{context}.{field} differs")


def generation_payload(settings: GenerationSettings) -> dict[str, object]:
    """Serialize a generation core exactly as its config block spells it.

    The v1 core predates the ``generation_prompt`` field, so its canonical
    spelling omits the field; retained v1 receipts must keep replaying
    byte-identically.
    """
    payload = cast(dict[str, object], asdict(settings))
    if settings.generation_prompt == GENERATION_PROMPT_TEMPLATE_THINK:
        del payload["generation_prompt"]
    return payload


def strip_reasoning_opener(text: str) -> str:
    """Drop the template's trailing reasoning opener, keeping the header."""
    if not text.endswith(_REASONING_PROMPT_SUFFIX):
        raise BaseResponseGenerationError(
            "generation prompt does not end with the reasoning opener"
        )
    return text[: -len(_REASONING_OPENER)]


def _generation(value: object) -> GenerationSettings:
    payload = _object(value, "generation")
    settings = (
        _GENERATION_SETTINGS_HEADER_ONLY
        if "generation_prompt" in payload
        else _GENERATION_SETTINGS
    )
    expected = generation_payload(settings)
    _fields(payload, set(expected), "generation")
    _exact_values(payload, expected, "generation")
    return settings


def _output(value: object) -> OutputSettings:
    payload = _object(value, "output")
    expected = asdict(_OUTPUT_SETTINGS)
    _fields(payload, set(expected), "output")
    _exact_values(payload, expected, "output")
    return _OUTPUT_SETTINGS


def _runtime(value: object) -> Mapping[str, str]:
    payload = _object(value, "runtime_versions")
    expected = {"safetensors", "torch", "transformers"}
    _fields(payload, expected, "runtime_versions")
    versions = {field: _string(payload, field, "runtime_versions") for field in payload}
    for field, version in versions.items():
        if _VERSION.fullmatch(version) is None:
            raise BaseResponseGenerationError(f"runtime_versions.{field} is invalid")
    return MappingProxyType(dict(sorted(versions.items())))


def _limitations(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BaseResponseGenerationError("limitations must be a non-empty list")
    items = cast(list[object], value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise BaseResponseGenerationError("limitations entries must be non-empty")
    return tuple(cast(list[str], items))


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        source = path.read_bytes()
        payload = _object(json_loads_unique(source.decode()), "base response config")
    except (OSError, UnicodeError, ValueError) as error:
        raise BaseResponseGenerationError(
            f"cannot load config {path}: {error}"
        ) from error
    return payload, source


def load_base_response_config(path: Path) -> BaseResponseConfig:
    """Load the immutable base-response generation core and authority envelope."""
    payload, source = _load(path)
    _fields(payload, _ROOT_FIELDS, "base response config")
    if payload.get("schema_version") != 1:
        raise BaseResponseGenerationError("schema_version must be integer 1")
    return BaseResponseConfig(
        _string(payload, "generation_id", "base response config"),
        _string(payload, "phase", "base response config"),
        _boundaries(payload.get("boundaries")),
        _package(payload.get("package")),
        _b2_gate(payload.get("b2_gate")),
        _artifact(payload.get("base"), "base", _BASE_FILES, requires_path=False),
        _artifact(
            payload.get("tokenizer"),
            "tokenizer",
            _TOKENIZER_FILES,
            requires_path=True,
        ),
        _generation(payload.get("generation")),
        _output(payload.get("output")),
        _runtime(payload.get("runtime_versions")),
        _limitations(payload.get("limitations")),
        path,
        bytes_sha256(source),
    )


def verify_artifact_snapshot(binding: ArtifactBinding, snapshot: Path) -> None:
    """Verify every pinned manifest file inside a local snapshot directory."""
    root = snapshot / binding.path if binding.path else snapshot
    for name, expected in binding.files.items():
        try:
            observed = file_sha256(root / name)
        except OSError as error:
            raise BaseResponseGenerationError(
                f"cannot read snapshot file {name}: {error}"
            ) from error
        if observed != expected:
            raise BaseResponseGenerationError(
                f"snapshot file {name} differs from the pinned manifest"
            )


def _git_blob(root: Path, revision: str, path: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{path}"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BaseResponseGenerationError(
            f"cannot read Git evidence: {error}"
        ) from error
    if result.returncode != 0:
        raise BaseResponseGenerationError(f"Git evidence is missing for {path}")
    return result.stdout


def _regular_bytes(path: Path, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise BaseResponseGenerationError(f"cannot read {label}: {error}") from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()), label)
            source = handle.read()
            after = _file_identity(os.fstat(handle.fileno()), label)
        path_identity = _file_identity(path.lstat(), label)
    except OSError as error:
        raise BaseResponseGenerationError(f"cannot read {label}: {error}") from error
    if before != after or path_identity != after:
        raise BaseResponseGenerationError(f"{label} changed while reading")
    return source


def _file_identity(item: os.stat_result, label: str) -> tuple[int, int, int, int]:
    if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        raise BaseResponseGenerationError(f"{label} must be a unique file")
    return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns


def _evidence_bytes(root: Path, relative: str, revision: str, label: str) -> bytes:
    source = _regular_bytes(root / relative, label)
    if source != _git_blob(root, revision, relative):
        raise BaseResponseGenerationError(f"{label} differs from Git evidence")
    return source


def _signed_receipt(source: bytes, expected: str, label: str) -> dict[str, Any]:
    try:
        payload = _object(json_loads_unique(source.decode()), label)
    except (UnicodeError, ValueError) as error:
        raise BaseResponseGenerationError(f"cannot parse {label}: {error}") from error
    claimed = payload.get("receipt_sha256")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if claimed != expected or canonical_sha256(body) != expected:
        raise BaseResponseGenerationError(f"{label} receipt SHA-256 differs")
    return payload


def _validate_package_receipt(
    config: BaseResponseConfig, payload: Mapping[str, object]
) -> None:
    _fields(payload, _PACKAGE_RECEIPT_FIELDS, "package receipt")
    if payload.get("schema_version") != 1:
        raise BaseResponseGenerationError("package receipt schema_version differs")
    if payload.get("package_id") != config.package.package_id:
        raise BaseResponseGenerationError("package receipt package_id differs")
    boundaries = _object(payload.get("boundaries"), "package boundaries")
    _fields(boundaries, _PACKAGE_BOUNDARY_FIELDS, "package boundaries")
    if boundaries.get("training_data_materialization_allowed") is not True:
        raise BaseResponseGenerationError("package materialization boundary differs")
    for field in sorted(
        _PACKAGE_BOUNDARY_FIELDS - {"training_data_materialization_allowed"}
    ):
        if boundaries[field] is not False:
            raise BaseResponseGenerationError(f"package boundaries.{field} differs")
    releases = _object(payload.get("releases"), "package receipt releases")
    if releases.get("executor_release") != config.package.executor_release:
        raise BaseResponseGenerationError("package executor release differs")


def _validate_b2_authority(
    config: BaseResponseConfig, payload: Mapping[str, object]
) -> None:
    authority = _object(payload.get("authority"), "B2 authority")
    _fields(authority, {"sha256", "values"}, "B2 authority")
    if authority.get("sha256") != config.b2_gate.authority_sha256:
        raise BaseResponseGenerationError("B2 authority SHA-256 differs")
    values = _object(authority.get("values"), "B2 authority values")
    expected = {
        "concept_scoring",
        "feature_selection",
        "full_condition_training",
        "provider_execution",
        "scheduler_submission",
        "scientific_use",
        "starts_experiment_timer",
    }
    _fields(values, expected, "B2 authority values")
    for field in sorted(expected):
        if values.get(field) is not False:
            raise BaseResponseGenerationError(f"B2 authority.{field} must be false")


def _validate_b2_recipe(
    config: BaseResponseConfig, payload: Mapping[str, object]
) -> None:
    recipe = _object(payload.get("recipe"), "B2 recipe")
    _fields(recipe, _B2_RECIPE_FIELDS, "B2 recipe")
    if recipe.get("recipe_id") != config.b2_gate.recipe_id:
        raise BaseResponseGenerationError("B2 recipe_id differs")
    if recipe.get("core_sha256") != config.b2_gate.recipe_core_sha256:
        raise BaseResponseGenerationError("B2 recipe core differs")
    conditions = _object(recipe.get("condition_core_sha256"), "B2 conditions")
    expected = {
        "behavior_query_ekfac_removal",
        "dense_probe_removal",
        "full_data",
        "matched_random_removal",
    }
    _fields(conditions, expected, "B2 conditions")
    if any(value != config.b2_gate.recipe_core_sha256 for value in conditions.values()):
        raise BaseResponseGenerationError("B2 condition core differs")


def _validate_b2_receipt(
    config: BaseResponseConfig, payload: Mapping[str, object]
) -> None:
    _fields(payload, _B2_RECEIPT_FIELDS, "B2 receipt")
    if payload.get("schema_version") != 1:
        raise BaseResponseGenerationError("B2 schema_version differs")
    if payload.get("smoke_id") != config.b2_gate.smoke_id:
        raise BaseResponseGenerationError("B2 smoke_id differs")
    _validate_b2_recipe(config, payload)
    _validate_b2_authority(config, payload)
    comparison = _object(payload.get("comparison"), "B2 comparison")
    if comparison.get("passed") is not True:
        raise BaseResponseGenerationError("B2 comparison.passed must be true")


def validate_base_response_evidence(
    config: BaseResponseConfig, repository_root: Path
) -> BaseResponseEvidence:
    """Authenticate package and B2 receipts against their Git evidence roots."""
    if load_base_response_config(config.config_path) != config:
        raise BaseResponseGenerationError("base response config changed after load")
    package_source = _evidence_bytes(
        repository_root,
        config.package.receipt_path,
        config.package.evidence_release,
        "package receipt",
    )
    b2_source = _evidence_bytes(
        repository_root,
        config.b2_gate.receipt_path,
        config.b2_gate.evidence_release,
        "B2 receipt",
    )
    package = _signed_receipt(
        package_source, config.package.receipt_sha256, "package receipt"
    )
    b2 = _signed_receipt(b2_source, config.b2_gate.receipt_sha256, "B2 receipt")
    _validate_package_receipt(config, package)
    _validate_b2_receipt(config, b2)
    return BaseResponseEvidence(
        config.package.receipt_sha256,
        config.b2_gate.receipt_sha256,
        _freeze_mapping(package),
        _freeze_mapping(b2),
    )
