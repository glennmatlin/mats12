"""Receipt-bound materialization of shared training-data inputs."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from concept_filtering.chat_rendering import ChatRenderingError, render_assistant_only
from concept_filtering.named_training_recipe import (
    NamedTrainingRecipe,
    NamedTrainingRecipeError,
    load_named_training_recipe,
)
from concept_filtering.serialization import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_sha256,
    json_equal_exact,
    json_loads_unique,
)
from concept_filtering.smoke import source_repository_root, verify_executor_revision

_CLOSED_BOUNDARIES = (
    "base_response_generation_allowed",
    "activation_extraction_allowed",
    "training_execution_allowed",
    "scientific_use",
    "starts_experiment_timer",
    "provider_execution_allowed",
    "scheduler_execution_allowed",
    "sae_feature_selection_allowed",
    "concept_scoring_allowed",
    "concept_selected_training_allowed",
)
_BOUNDARY_FIELDS = {"training_data_materialization_allowed", *_CLOSED_BOUNDARIES}
_ROOT_FIELDS = {
    "schema_version",
    "package_id",
    "phase",
    "boundaries",
    "sample",
    "tokenizer",
    "runtime",
    "training",
    "output",
    "input_producer_releases",
    "limitations",
}
_SAMPLE_FIELDS = {
    "repository",
    "revision",
    "path",
    "sha256",
    "source_revision",
    "training_sample_receipt_sha256",
    "expected_row_count",
    "expected_split",
    "identity_scheme",
}
_TRAINING_FIELDS = {
    "recipe_id",
    "recipe_core_sha256",
    "chat_template",
    "chat_template_sha256",
    "max_sequence_length",
    "overlength_policy",
    "truncation",
    "append_eos",
    "loss_mask",
}
_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.+-]*)?$")
_ENTRY_FIELDS = {
    "schema_version",
    "row_ordinal",
    "orig_idx",
    "example_id",
    "source_record_sha256",
    "dataset_source",
    "prompt_messages_sha256",
    "supplied_response_sha256",
    "rendered_input_ids_sha256",
    "labels_sha256",
    "input_token_count",
    "supervised_token_count",
    "assistant_turn_count",
    "initial_user_empty",
    "overlength",
    "full_data_member",
    "preprocessing_status",
}
_RECEIPT_FIELDS = {
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
}
_READ_TABLE = cast(
    Callable[[object], pa.Table],
    pq.read_table,  # pyright: ignore[reportUnknownMemberType]
)


class TrainingDataPackageError(ValueError):
    """Raised when the shared training-data package contract differs."""


@dataclass(frozen=True)
class SampleBinding:
    repository: str
    revision: str
    path: str
    sha256: str
    source_revision: str
    training_sample_receipt_sha256: str
    expected_row_count: int
    expected_split: int
    identity_scheme: str

    def to_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "path": self.path,
            "sha256": self.sha256,
            "source_revision": self.source_revision,
            "training_sample_receipt_sha256": self.training_sample_receipt_sha256,
            "expected_row_count": self.expected_row_count,
            "expected_split": self.expected_split,
            "identity_scheme": self.identity_scheme,
        }


@dataclass(frozen=True)
class TokenizerBinding:
    repository: str
    revision: str
    path: str
    files: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "path": self.path,
            "files": dict(sorted(self.files.items())),
        }


@dataclass(frozen=True)
class TrainingBinding:
    recipe_id: str
    recipe_core_sha256: str
    chat_template: str
    chat_template_sha256: str
    max_sequence_length: int
    overlength_policy: str
    truncation: bool
    append_eos: bool
    loss_mask: str

    def receipt_dict(self) -> dict[str, object]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_core_sha256": self.recipe_core_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "max_sequence_length": self.max_sequence_length,
            "overlength_policy": self.overlength_policy,
            "truncation": self.truncation,
            "append_eos": self.append_eos,
            "loss_mask": self.loss_mask,
        }


@dataclass(frozen=True)
class TrainingDataPackageConfig:
    package_id: str
    phase: str
    boundaries: Mapping[str, bool]
    sample: SampleBinding
    tokenizer: TokenizerBinding
    runtime_versions: Mapping[str, str]
    training: TrainingBinding
    output_format: str
    entry_schema_version: int
    input_producer_releases: Mapping[str, str]
    limitations: tuple[str, ...]
    config_path: Path
    config_sha256: str

    @property
    def expected_row_count(self) -> int:
        return self.sample.expected_row_count

    @property
    def training_data_materialization_allowed(self) -> bool:
        return self.boundaries["training_data_materialization_allowed"]

    @property
    def base_response_generation_allowed(self) -> bool:
        return self.boundaries["base_response_generation_allowed"]


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in mapping.items()}
        )
    if isinstance(value, list | tuple):
        items = cast(list[object] | tuple[object, ...], value)
        return tuple(_freeze_value(item) for item in items)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], _freeze_value(value))


def _thaw_value(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {key: _thaw_value(item) for key, item in mapping.items()}
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return [_thaw_value(item) for item in items]
    return value


def _thaw_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return cast(dict[str, object], _thaw_value(value))


@dataclass(frozen=True)
class TrainingDataPackageReceipt:
    payload: Mapping[str, object]
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))

    def to_dict(self) -> dict[str, object]:
        return {**_thaw_mapping(self.payload), "receipt_sha256": self.receipt_sha256}

    @property
    def row_count(self) -> int:
        return cast(int, self.payload["row_count"])

    @property
    def retained_row_count(self) -> int:
        return cast(int, self.payload["retained_row_count"])

    @property
    def multi_turn_row_count(self) -> int:
        return cast(int, self.payload["multi_turn_row_count"])

    @property
    def empty_initial_user_count(self) -> int:
        return cast(int, self.payload["empty_initial_user_count"])

    @property
    def package_sha256(self) -> str:
        return cast(str, self.payload["package_sha256"])

    @property
    def releases(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.payload["releases"])

    @property
    def boundaries(self) -> Mapping[str, bool]:
        return cast(Mapping[str, bool], self.payload["boundaries"])


@dataclass(frozen=True)
class PackageBuild:
    entries: tuple[dict[str, object], ...]
    package_bytes: bytes
    retained_row_count: int
    multi_turn_row_count: int
    empty_initial_user_count: int
    total_input_token_count: int
    total_supervised_token_count: int
    retained_input_token_count: int
    retained_supervised_token_count: int


@dataclass(frozen=True)
class ObservedOutput:
    value: bytes
    identity: tuple[int, int]


@dataclass(frozen=True)
class PublishedOutput:
    identity: tuple[int, int]
    created: bool


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingDataPackageError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _fields(payload: dict[str, Any], expected: set[str], field: str) -> None:
    if set(payload) != expected:
        raise TrainingDataPackageError(f"{field} fields differ from the contract")


def _label(field: str, context: str | None) -> str:
    return f"{context}.{field}" if context is not None else field


def _string(payload: dict[str, Any], field: str, *, context: str | None = None) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise TrainingDataPackageError(
            f"{_label(field, context)} must be a non-empty string"
        )
    return value


def _positive_int(
    payload: dict[str, Any], field: str, *, context: str | None = None
) -> int:
    value = payload.get(field)
    if type(value) is not int or value < 1:
        raise TrainingDataPackageError(
            f"{_label(field, context)} must be a positive integer"
        )
    return value


def _sha256(payload: dict[str, Any], field: str, *, context: str | None = None) -> str:
    value = _string(payload, field, context=context)
    if _SHA256.fullmatch(value) is None:
        raise TrainingDataPackageError(
            f"{_label(field, context)} must be a lowercase SHA-256"
        )
    return value


def _revision(
    payload: dict[str, Any], field: str, *, context: str | None = None
) -> str:
    value = _string(payload, field, context=context)
    if _REVISION.fullmatch(value) is None:
        raise TrainingDataPackageError(
            f"{_label(field, context)} must be a 40-character release"
        )
    return value


def _exact(value: object, expected: object, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise TrainingDataPackageError(f"{field} must be {expected!r}")


def _safe_path(
    payload: dict[str, Any], field: str, *, context: str | None = None
) -> str:
    value = _string(payload, field, context=context)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise TrainingDataPackageError(
            f"{_label(field, context)} must be a safe relative path"
        )
    return value


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        value = json_loads_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TrainingDataPackageError(
            f"cannot load package config: {error}"
        ) from error
    return _object(value, "package config")


def _sample(value: object) -> SampleBinding:
    payload = _object(value, "sample")
    _fields(payload, _SAMPLE_FIELDS, "sample")
    return SampleBinding(
        repository=_string(payload, "repository", context="sample"),
        revision=_revision(payload, "revision", context="sample"),
        path=_safe_path(payload, "path", context="sample"),
        sha256=_sha256(payload, "sha256", context="sample"),
        source_revision=_revision(payload, "source_revision", context="sample"),
        training_sample_receipt_sha256=_sha256(
            payload, "training_sample_receipt_sha256", context="sample"
        ),
        expected_row_count=_positive_int(
            payload, "expected_row_count", context="sample"
        ),
        expected_split=_positive_int(payload, "expected_split", context="sample"),
        identity_scheme=_sample_identity(payload),
    )


def _sample_identity(payload: dict[str, Any]) -> str:
    _exact(payload.get("expected_split"), 1, "sample.expected_split")
    scheme = _string(payload, "identity_scheme", context="sample")
    _exact(scheme, "source-revision-orig-idx-v1", "sample.identity_scheme")
    return scheme


def _tokenizer(value: object) -> TokenizerBinding:
    payload = _object(value, "tokenizer")
    _fields(payload, {"repository", "revision", "path", "files"}, "tokenizer")
    files = _object(payload.get("files"), "tokenizer.files")
    _fields(files, _TOKENIZER_FILES, "tokenizer.files")
    return TokenizerBinding(
        repository=_string(payload, "repository", context="tokenizer"),
        revision=_revision(payload, "revision", context="tokenizer"),
        path=_safe_path(payload, "path", context="tokenizer"),
        files=MappingProxyType(
            {key: _sha256(files, key, context="tokenizer.files") for key in files}
        ),
    )


def _training(value: object) -> TrainingBinding:
    payload = _object(value, "training")
    _fields(payload, _TRAINING_FIELDS, "training")
    template = _string(payload, "chat_template", context="training")
    _validate_training(payload, template)
    return TrainingBinding(
        recipe_id=_string(payload, "recipe_id", context="training"),
        recipe_core_sha256=_sha256(payload, "recipe_core_sha256", context="training"),
        chat_template=template,
        chat_template_sha256=_sha256(
            payload, "chat_template_sha256", context="training"
        ),
        max_sequence_length=_positive_int(
            payload, "max_sequence_length", context="training"
        ),
        overlength_policy=_string(payload, "overlength_policy", context="training"),
        truncation=cast(bool, payload["truncation"]),
        append_eos=cast(bool, payload["append_eos"]),
        loss_mask=_string(payload, "loss_mask", context="training"),
    )


def _validate_training(payload: dict[str, Any], template: str) -> None:
    expected_template_sha = bytes_sha256(template.encode("utf-8"))
    if (
        _sha256(payload, "chat_template_sha256", context="training")
        != expected_template_sha
    ):
        raise TrainingDataPackageError("training.chat_template SHA-256 differs")
    _exact(payload.get("overlength_policy"), "drop", "training.overlength_policy")
    _exact(payload.get("truncation"), False, "training.truncation")
    _exact(payload.get("append_eos"), True, "training.append_eos")
    _exact(
        payload.get("loss_mask"),
        "assistant_tokens_including_terminal_eos",
        "training.loss_mask",
    )


def _boundaries(value: object) -> Mapping[str, bool]:
    payload = _object(value, "boundaries")
    if set(payload) != _BOUNDARY_FIELDS:
        raise TrainingDataPackageError("boundary fields differ from the contract")
    if not all(type(item) is bool for item in payload.values()):
        raise TrainingDataPackageError("boundaries must contain booleans")
    boundaries = cast(dict[str, bool], payload)
    if boundaries["training_data_materialization_allowed"] is not True:
        raise TrainingDataPackageError(
            "training_data_materialization_allowed must remain true"
        )
    for field in _CLOSED_BOUNDARIES:
        if boundaries[field] is not False:
            raise TrainingDataPackageError(f"{field} must remain false")
    return MappingProxyType(dict(boundaries))


def _releases(value: object) -> Mapping[str, str]:
    payload = _object(value, "input_producer_releases")
    _fields(
        payload,
        {"training_sample_evidence", "training_recipe_evidence"},
        "input_producer_releases",
    )
    return MappingProxyType(
        {
            key: _revision(payload, key, context="input_producer_releases")
            for key in payload
        }
    )


def _runtime(value: object) -> Mapping[str, str]:
    payload = _object(value, "runtime")
    _fields(
        payload,
        {"huggingface_hub", "pyarrow", "tokenizers", "transformers"},
        "runtime",
    )
    versions = {
        key: _string(payload, key, context="runtime") for key in sorted(payload)
    }
    for field, version in versions.items():
        if _VERSION.fullmatch(version) is None:
            raise TrainingDataPackageError(
                f"runtime.{field} must be an exact numeric version"
            )
    return MappingProxyType(versions)


def _limitations(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TrainingDataPackageError("limitations must be a non-empty list")
    items = cast(list[object], value)
    if not all(isinstance(item, str) and item for item in items):
        raise TrainingDataPackageError("limitations must contain strings")
    return tuple(cast(list[str], items))


def load_training_data_package_config(path: Path) -> TrainingDataPackageConfig:
    """Load the named pre-feature materialization contract."""
    payload = _load_payload(path)
    _fields(payload, _ROOT_FIELDS, "package config")
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise TrainingDataPackageError("schema_version must be the integer 1")
    _exact(payload.get("phase"), "pre_feature_training_data_materialization", "phase")
    output = _object(payload.get("output"), "output")
    _fields(output, {"format", "entry_schema_version"}, "output")
    _exact(output.get("format"), "canonical-jsonl-v1", "output.format")
    _exact(output.get("entry_schema_version"), 1, "entry_schema_version")
    return TrainingDataPackageConfig(
        package_id=_string(payload, "package_id"),
        phase=_string(payload, "phase"),
        boundaries=_boundaries(payload.get("boundaries")),
        sample=_sample(payload.get("sample")),
        tokenizer=_tokenizer(payload.get("tokenizer")),
        runtime_versions=_runtime(payload.get("runtime")),
        training=_training(payload.get("training")),
        output_format=_string(output, "format"),
        entry_schema_version=_positive_int(output, "entry_schema_version"),
        input_producer_releases=_releases(payload.get("input_producer_releases")),
        limitations=_limitations(payload.get("limitations")),
        config_path=path.resolve(),
        config_sha256=canonical_sha256(payload),
    )


def _recipe_provenance(
    recipe: NamedTrainingRecipe, config: TrainingDataPackageConfig
) -> dict[str, object]:
    tokenizer = recipe.artifacts.tokenizer
    sample = recipe.artifacts.training_data
    return {
        "recipe_id": recipe.recipe_id,
        "recipe_core_sha256": recipe.recipe_core_sha256,
        "tokenizer": {
            "repository": tokenizer.repository,
            "revision": tokenizer.revision,
            "path": tokenizer.path,
            "files": dict(sorted(tokenizer.files.items())),
        },
        "sample_row_count": sample.row_count,
        "sample_identity_scheme": sample.identity_scheme,
        "chat_template_sha256": recipe.values.sequence.chat_template_sha256,
        "data": _recipe_data_provenance(recipe),
        "runtime": _recipe_runtime_provenance(recipe.runtime_versions),
    }


def _recipe_data_provenance(recipe: NamedTrainingRecipe) -> dict[str, object]:
    sequence = recipe.values.sequence
    return {
        "max_sequence_length": sequence.max_sequence_length,
        "overlength_policy": sequence.overlength_policy,
        "truncation": sequence.truncation,
        "append_eos": True,
        "loss_mask": sequence.loss_mask,
    }


def _recipe_runtime_provenance(runtime: Mapping[str, str]) -> dict[str, str]:
    fields = ("pyarrow", "tokenizers", "transformers")
    return {field: runtime[field] for field in fields}


def _expected_recipe_provenance(
    config: TrainingDataPackageConfig,
) -> dict[str, object]:
    return {
        "recipe_id": config.training.recipe_id,
        "recipe_core_sha256": config.training.recipe_core_sha256,
        "tokenizer": {
            "repository": config.tokenizer.repository,
            "revision": config.tokenizer.revision,
            "path": config.tokenizer.path,
            "files": dict(sorted(config.tokenizer.files.items())),
        },
        "sample_row_count": config.sample.expected_row_count,
        "sample_identity_scheme": config.sample.identity_scheme,
        "chat_template_sha256": config.training.chat_template_sha256,
        "data": _config_data_provenance(config.training),
        "runtime": _recipe_runtime_provenance(config.runtime_versions),
    }


def _config_data_provenance(training: TrainingBinding) -> dict[str, object]:
    return {
        "max_sequence_length": training.max_sequence_length,
        "overlength_policy": training.overlength_policy,
        "truncation": training.truncation,
        "append_eos": training.append_eos,
        "loss_mask": training.loss_mask,
    }


def _signed_receipt_body(path: Path) -> dict[str, Any]:
    try:
        value = json_loads_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TrainingDataPackageError(
            f"cannot load sample provenance receipt: {error}"
        ) from error
    payload = _object(value, "sample provenance receipt")
    claimed = _sha256(payload, "receipt_sha256")
    body = {key: item for key, item in payload.items() if key != "receipt_sha256"}
    if canonical_sha256(body) != claimed:
        raise TrainingDataPackageError("sample provenance receipt SHA-256 differs")
    return body


def _sample_provenance(body: dict[str, Any]) -> dict[str, object]:
    sample = _object(body.get("sample"), "sample provenance")
    source = _object(body.get("source"), "sample source provenance")
    return {
        "receipt_sha256": canonical_sha256(body),
        "repository": sample.get("repository"),
        "revision": sample.get("revision"),
        "path": sample.get("path"),
        "sha256": sample.get("sha256"),
        "source_revision": source.get("revision"),
        "row_count": body.get("row_count"),
        "split": body.get("split"),
        "identity_scheme": body.get("identity_scheme"),
        "scientific_use": body.get("scientific_use"),
        "starts_experiment_timer": body.get("starts_experiment_timer"),
    }


def _expected_sample_provenance(config: TrainingDataPackageConfig) -> dict[str, object]:
    sample = config.sample
    return {
        "receipt_sha256": sample.training_sample_receipt_sha256,
        "repository": sample.repository,
        "revision": sample.revision,
        "path": sample.path,
        "sha256": sample.sha256,
        "source_revision": sample.source_revision,
        "row_count": sample.expected_row_count,
        "split": sample.expected_split,
        "identity_scheme": sample.identity_scheme,
        "scientific_use": False,
        "starts_experiment_timer": False,
    }


def _repository_path(path: Path) -> tuple[Path, str]:
    root = source_repository_root()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as error:
        raise TrainingDataPackageError(
            f"producer artifact is outside the repository: {path}"
        ) from error
    return root, relative.as_posix()


def _producer_blob(revision: str, path: Path) -> bytes:
    root, relative = _repository_path(path)
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise TrainingDataPackageError(
            f"cannot load producer artifact {relative} at {revision}: {error}"
        ) from error
    return result.stdout


def _verify_producer_release(revision: str, path: Path, label: str) -> None:
    try:
        current = path.read_bytes()
    except OSError as error:
        raise TrainingDataPackageError(f"cannot read {label}: {error}") from error
    try:
        produced = _producer_blob(revision, path)
    except TrainingDataPackageError as error:
        raise TrainingDataPackageError(
            f"{label} producer provenance differs: {error}"
        ) from error
    if produced != current:
        raise TrainingDataPackageError(f"{label} producer release differs")


def validate_input_provenance(
    config: TrainingDataPackageConfig,
    *,
    recipe_path: Path,
    sample_receipt_path: Path,
) -> None:
    """Bind every duplicated package field to its frozen producer artifact."""
    try:
        recipe = load_named_training_recipe(recipe_path)
    except NamedTrainingRecipeError as error:
        raise TrainingDataPackageError(f"invalid recipe provenance: {error}") from error
    if not json_equal_exact(
        _recipe_provenance(recipe, config), _expected_recipe_provenance(config)
    ):
        raise TrainingDataPackageError("training recipe provenance differs")
    sample = _sample_provenance(_signed_receipt_body(sample_receipt_path))
    if not json_equal_exact(sample, _expected_sample_provenance(config)):
        raise TrainingDataPackageError("training sample provenance differs")
    releases = config.input_producer_releases
    _verify_producer_release(
        releases["training_recipe_evidence"], recipe_path, "training recipe"
    )
    _verify_producer_release(
        releases["training_sample_evidence"],
        sample_receipt_path,
        "training sample receipt",
    )


def _installed_runtime_versions() -> dict[str, str]:
    distributions = {
        "huggingface_hub": "huggingface-hub",
        "pyarrow": "pyarrow",
        "tokenizers": "tokenizers",
        "transformers": "transformers",
    }
    try:
        return {
            field: importlib.metadata.version(distribution)
            for field, distribution in distributions.items()
        }
    except importlib.metadata.PackageNotFoundError as error:
        raise TrainingDataPackageError(
            f"tokenizer runtime is missing: {error}"
        ) from error


def _bound_tokenizer_files(
    config: TrainingDataPackageConfig, snapshot: Path
) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    for name, expected_sha in config.tokenizer.files.items():
        try:
            value = (snapshot / name).read_bytes()
        except OSError as error:
            raise TrainingDataPackageError(
                f"cannot read tokenizer file {name}"
            ) from error
        if bytes_sha256(value) != expected_sha:
            raise TrainingDataPackageError(f"tokenizer file hash differs: {name}")
        values[name] = value
    return values


def load_bound_tokenizer(config: TrainingDataPackageConfig, snapshot: Path) -> Any:
    """Load a tokenizer from one immutable, hash-verified file snapshot."""
    installed = _installed_runtime_versions()
    if installed != config.runtime_versions:
        raise TrainingDataPackageError("tokenizer runtime versions differ")
    files = _bound_tokenizer_files(config, snapshot)
    try:
        transformers = cast(Any, importlib.import_module("transformers"))
    except ImportError as error:
        raise TrainingDataPackageError("transformers runtime is missing") from error
    try:
        with tempfile.TemporaryDirectory(prefix="training-tokenizer-") as directory:
            temporary = Path(directory)
            for name, value in files.items():
                (temporary / name).write_bytes(value)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                temporary, local_files_only=True, trust_remote_code=False
            )
    except Exception as error:
        raise TrainingDataPackageError(
            f"cannot load bound tokenizer: {error}"
        ) from error
    if tokenizer.is_fast is not True or type(tokenizer.eos_token_id) is not int:
        raise TrainingDataPackageError("bound tokenizer must be fast and define EOS")
    return tokenizer


def _expected_schema() -> pa.Schema:
    message = pa.struct(
        [pa.field("content", pa.string()), pa.field("role", pa.string())]
    )
    return pa.schema(
        [
            pa.field("messages", pa.list_(pa.field("element", message))),
            pa.field("dataset_source", pa.string()),
            pa.field("orig_idx", pa.int64()),
            pa.field("split", pa.int64()),
        ]
    )


def _sample_rows(
    config: TrainingDataPackageConfig, path: Path
) -> list[dict[str, object]]:
    try:
        sample_bytes = path.read_bytes()
    except OSError as error:
        raise TrainingDataPackageError(f"cannot read sample: {error}") from error
    if bytes_sha256(sample_bytes) != config.sample.sha256:
        raise TrainingDataPackageError("sample SHA-256 differs from config")
    try:
        table = _READ_TABLE(pa.BufferReader(sample_bytes))
    except (OSError, pa.ArrowException) as error:
        raise TrainingDataPackageError(f"cannot parse sample: {error}") from error
    if not table.schema.remove_metadata().equals(_expected_schema()):
        raise TrainingDataPackageError("sample schema differs from the contract")
    return _validated_rows(config, table)


def _validated_rows(
    config: TrainingDataPackageConfig, table: pa.Table
) -> list[dict[str, object]]:
    if table.num_rows != config.sample.expected_row_count:
        raise TrainingDataPackageError("sample row count differs from config")
    rows = cast(list[dict[str, object]], table.to_pylist())
    identities: set[int] = set()
    for ordinal, row in enumerate(rows):
        _validate_row(config, row, ordinal)
        orig_idx = cast(int, row["orig_idx"])
        if orig_idx in identities:
            raise TrainingDataPackageError("sample orig_idx values must be unique")
        identities.add(orig_idx)
    return rows


def _validate_row(
    config: TrainingDataPackageConfig, row: dict[str, object], ordinal: int
) -> None:
    source = row.get("dataset_source")
    if not isinstance(source, str) or not source:
        raise TrainingDataPackageError(f"row {ordinal}: dataset source is invalid")
    orig_idx = row.get("orig_idx")
    if type(orig_idx) is not int or orig_idx < 0:
        raise TrainingDataPackageError(f"row {ordinal}: orig_idx is invalid")
    if row.get("split") != config.sample.expected_split:
        raise TrainingDataPackageError(f"row {ordinal}: split is invalid")
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise TrainingDataPackageError(f"row {ordinal}: messages are invalid")
    _validate_messages(cast(list[object], messages), ordinal)


def _validate_messages(messages: list[object], ordinal: int) -> None:
    if len(messages) < 2 or len(messages) % 2:
        raise TrainingDataPackageError(f"row {ordinal}: turns must alternate")
    for index, item in enumerate(messages):
        if not isinstance(item, Mapping):
            raise TrainingDataPackageError(f"row {ordinal}: message is invalid")
        message = cast(Mapping[str, object], item)
        role = "user" if index % 2 == 0 else "assistant"
        if set(message) != {"content", "role"} or message.get("role") != role:
            raise TrainingDataPackageError(f"row {ordinal}: turns must alternate")
        content = message.get("content")
        if not isinstance(content, str) or (role == "assistant" and not content):
            raise TrainingDataPackageError(f"row {ordinal}: content is invalid")


def _render_tokens(
    config: TrainingDataPackageConfig,
    tokenizer: Any,
    messages: list[dict[str, str]],
    ordinal: int,
) -> tuple[list[int], list[int], int]:
    try:
        rendered = render_assistant_only(
            tokenizer, messages, config.training.chat_template
        )
    except ChatRenderingError as error:
        raise TrainingDataPackageError(
            f"row {ordinal}: cannot render training row: {error}"
        ) from error
    input_ids = list(rendered.input_ids)
    labels = list(rendered.labels)
    eos_id = tokenizer.eos_token_id
    if type(eos_id) is not int:
        raise TrainingDataPackageError("tokenizer EOS token is invalid")
    if not input_ids or input_ids[-1] != eos_id:
        input_ids.append(eos_id)
        labels.append(eos_id)
    return input_ids, labels, rendered.assistant_turn_count


def _entry(
    config: TrainingDataPackageConfig,
    tokenizer: Any,
    row: dict[str, object],
    ordinal: int,
) -> dict[str, object]:
    messages = cast(list[dict[str, str]], row["messages"])
    input_ids, labels, turns = _render_tokens(config, tokenizer, messages, ordinal)
    overlength = len(input_ids) > config.training.max_sequence_length
    orig_idx = cast(int, row["orig_idx"])
    return {
        "schema_version": config.entry_schema_version,
        "row_ordinal": ordinal,
        "orig_idx": orig_idx,
        "example_id": _example_id(config.sample.source_revision, orig_idx),
        "source_record_sha256": canonical_sha256(row),
        "dataset_source": row["dataset_source"],
        "prompt_messages_sha256": canonical_sha256(messages[:-1]),
        "supplied_response_sha256": canonical_sha256(messages[-1]),
        "rendered_input_ids_sha256": canonical_sha256(input_ids),
        "labels_sha256": canonical_sha256(labels),
        "input_token_count": len(input_ids),
        "supervised_token_count": sum(value != -100 for value in labels),
        "assistant_turn_count": turns,
        "initial_user_empty": messages[0]["content"] == "",
        "overlength": overlength,
        "full_data_member": not overlength,
        "preprocessing_status": "dropped_overlength" if overlength else "retained",
    }


def _example_id(source_revision: str, orig_idx: int) -> str:
    return canonical_sha256({"source_revision": source_revision, "orig_idx": orig_idx})


def _token_count(
    entries: tuple[dict[str, object], ...], field: str, *, retained_only: bool = False
) -> int:
    return sum(
        cast(int, entry[field])
        for entry in entries
        if not retained_only or entry["full_data_member"] is True
    )


def _build_package(
    config: TrainingDataPackageConfig,
    tokenizer: Any,
    rows: list[dict[str, object]],
) -> PackageBuild:
    entries = tuple(
        _entry(config, tokenizer, row, index) for index, row in enumerate(rows)
    )
    package_bytes = b"".join(canonical_json_bytes(entry) + b"\n" for entry in entries)
    retained = sum(cast(bool, entry["full_data_member"]) for entry in entries)
    multi_turn = sum(cast(int, entry["assistant_turn_count"]) > 1 for entry in entries)
    empty_user = sum(
        cast(list[dict[str, str]], row["messages"])[0]["content"] == "" for row in rows
    )
    return PackageBuild(
        entries,
        package_bytes,
        retained,
        multi_turn,
        empty_user,
        _token_count(entries, "input_token_count"),
        _token_count(entries, "supervised_token_count"),
        _token_count(entries, "input_token_count", retained_only=True),
        _token_count(entries, "supervised_token_count", retained_only=True),
    )


def _release_payload(
    config: TrainingDataPackageConfig, revision: str, build: PackageBuild
) -> dict[str, object]:
    output_release = canonical_sha256(
        {
            "executor_release": revision,
            "config_sha256": config.config_sha256,
            "package_sha256": bytes_sha256(build.package_bytes),
            "ordered_entry_sha256": canonical_sha256(build.entries),
        }
    )
    return {
        "executor_release": revision,
        "coordinator_release": revision,
        "input_producer_releases": dict(config.input_producer_releases),
        "output_release": output_release,
    }


def _receipt_payload(
    config: TrainingDataPackageConfig, revision: str, build: PackageBuild
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "package_id": config.package_id,
        "phase": config.phase,
        "config_sha256": config.config_sha256,
        "boundaries": dict(config.boundaries),
        "sample": config.sample.to_dict(),
        "tokenizer": config.tokenizer.to_dict(),
        "runtime_versions": dict(config.runtime_versions),
        "training": config.training.receipt_dict(),
        "package_format": config.output_format,
        "entry_schema_version": config.entry_schema_version,
        "row_count": len(build.entries),
        "retained_row_count": build.retained_row_count,
        "dropped_overlength_count": len(build.entries) - build.retained_row_count,
        "multi_turn_row_count": build.multi_turn_row_count,
        "empty_initial_user_count": build.empty_initial_user_count,
        "total_input_token_count": build.total_input_token_count,
        "total_supervised_token_count": build.total_supervised_token_count,
        "retained_input_token_count": build.retained_input_token_count,
        "retained_supervised_token_count": build.retained_supervised_token_count,
        "ordered_entry_sha256": canonical_sha256(build.entries),
        "identity_entry_sha256": _identity_entry_sha256(build.entries),
        "package_sha256": bytes_sha256(build.package_bytes),
        "releases": _release_payload(config, revision, build),
        "limitations": list(config.limitations),
    }


def _identity_entry_sha256(entries: tuple[dict[str, object], ...]) -> str:
    identities = [
        {
            "example_id": entry["example_id"],
            "source_record_sha256": entry["source_record_sha256"],
        }
        for entry in entries
    ]
    ordered = sorted(identities, key=lambda item: cast(str, item["example_id"]))
    return canonical_sha256(ordered)


def _staged_bytes(path: Path, value: bytes, label: str) -> Path:
    try:
        descriptor, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
    except OSError as error:
        raise TrainingDataPackageError(f"cannot stage {label}: {error}") from error
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise TrainingDataPackageError(f"cannot stage {label}: {error}") from error
    return temporary


def _output_identity(status: os.stat_result, label: str) -> tuple[int, int]:
    message = f"existing {label} is not an isolated regular file"
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise TrainingDataPackageError(message)
    return status.st_dev, status.st_ino


def _path_identity(path: Path, label: str) -> tuple[int, int]:
    try:
        return _output_identity(path.lstat(), label)
    except OSError as error:
        raise TrainingDataPackageError(f"{label} path identity changed") from error


def _read_isolated_output(path: Path, label: str) -> ObservedOutput:
    message = f"existing {label} is not an isolated regular file"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise TrainingDataPackageError(message) from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = _output_identity(os.fstat(handle.fileno()), label)
            value = handle.read()
            after = _output_identity(os.fstat(handle.fileno()), label)
    except OSError as error:
        raise TrainingDataPackageError(message) from error
    if before != after or _path_identity(path, label) != after:
        raise TrainingDataPackageError(f"{label} path identity changed")
    return ObservedOutput(value, after)


def _publish_or_match(path: Path, value: bytes, label: str) -> PublishedOutput:
    temporary = _staged_bytes(path, value, label)
    try:
        status = temporary.stat()
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed = _read_isolated_output(path, label)
            if observed.value != value:
                raise TrainingDataPackageError(
                    f"existing {label} bytes differ"
                ) from None
            return PublishedOutput(observed.identity, False)
    except OSError as error:
        raise TrainingDataPackageError(f"cannot publish {label}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    identity = (status.st_dev, status.st_ino)
    if _path_identity(path, label) != identity:
        raise TrainingDataPackageError(f"{label} path identity changed")
    return PublishedOutput(identity, True)


def _verify_published_output(
    path: Path, expected: bytes, label: str, published: PublishedOutput
) -> None:
    observed = _read_isolated_output(path, label)
    if observed.identity != published.identity:
        raise TrainingDataPackageError(f"{label} path identity changed")
    if observed.value != expected:
        raise TrainingDataPackageError(f"{label} bytes changed during publication")


def _publish_evidence(
    package_path: Path,
    package_bytes: bytes,
    receipt_path: Path,
    receipt_bytes: bytes,
) -> None:
    package = _publish_or_match(package_path, package_bytes, "package")
    receipt = _publish_or_match(receipt_path, receipt_bytes, "receipt")
    _verify_published_output(package_path, package_bytes, "package", package)
    _verify_published_output(receipt_path, receipt_bytes, "receipt", receipt)


def _receipt_bytes(receipt: TrainingDataPackageReceipt) -> bytes:
    return json.dumps(receipt.to_dict(), indent=2, sort_keys=True).encode() + b"\n"


def _validate_config_identity(config: TrainingDataPackageConfig) -> None:
    reloaded = load_training_data_package_config(config.config_path)
    if reloaded != config:
        raise TrainingDataPackageError("package config differs from its canonical file")


def _load_receipt(path: Path) -> tuple[TrainingDataPackageReceipt, bytes]:
    try:
        raw = _read_isolated_output(path, "receipt").value
        value = json_loads_unique(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TrainingDataPackageError(
            f"cannot load package receipt: {error}"
        ) from error
    payload = _object(value, "package receipt")
    claimed = _sha256(payload, "receipt_sha256")
    body = {key: item for key, item in payload.items() if key != "receipt_sha256"}
    _fields(body, _RECEIPT_FIELDS, "package receipt")
    if canonical_sha256(body) != claimed:
        raise TrainingDataPackageError("package receipt SHA-256 differs")
    return TrainingDataPackageReceipt(body, claimed), raw


def _read_package(
    path: Path, expected_sha: str
) -> tuple[bytes, tuple[dict[str, object], ...]]:
    package_bytes = _read_isolated_output(path, "package").value
    if bytes_sha256(package_bytes) != expected_sha:
        raise TrainingDataPackageError("package SHA-256 differs from receipt")
    try:
        values = [
            json_loads_unique(line) for line in package_bytes.decode().splitlines()
        ]
    except (UnicodeError, ValueError) as error:
        raise TrainingDataPackageError(f"cannot parse package: {error}") from error
    entries = tuple(_entry_object(value, index) for index, value in enumerate(values))
    canonical = b"".join(canonical_json_bytes(entry) + b"\n" for entry in entries)
    if not entries or canonical != package_bytes:
        raise TrainingDataPackageError("package is not canonical JSONL")
    return package_bytes, entries


def _entry_object(value: object, ordinal: int) -> dict[str, object]:
    entry = cast(dict[str, object], _object(value, f"entry {ordinal}"))
    _fields(cast(dict[str, Any], entry), _ENTRY_FIELDS, f"entry {ordinal}")
    if entry["schema_version"] != 1 or entry["row_ordinal"] != ordinal:
        raise TrainingDataPackageError(f"entry {ordinal}: identity differs")
    for field in ("orig_idx", "input_token_count", "supervised_token_count"):
        if type(entry[field]) is not int or cast(int, entry[field]) < 0:
            raise TrainingDataPackageError(f"entry {ordinal}: {field} is invalid")
    _validate_entry_values(entry, ordinal)
    return entry


def _validate_entry_values(entry: dict[str, object], ordinal: int) -> None:
    for field in (
        "example_id",
        "source_record_sha256",
        "prompt_messages_sha256",
        "supplied_response_sha256",
        "rendered_input_ids_sha256",
        "labels_sha256",
    ):
        if (
            not isinstance(entry[field], str)
            or _SHA256.fullmatch(cast(str, entry[field])) is None
        ):
            raise TrainingDataPackageError(f"entry {ordinal}: {field} is invalid")
    if not isinstance(entry["dataset_source"], str) or not entry["dataset_source"]:
        raise TrainingDataPackageError(f"entry {ordinal}: dataset_source is invalid")
    _validate_entry_status(entry, ordinal)


def _validate_entry_status(entry: dict[str, object], ordinal: int) -> None:
    booleans = ("initial_user_empty", "overlength", "full_data_member")
    if any(type(entry[field]) is not bool for field in booleans):
        raise TrainingDataPackageError(f"entry {ordinal}: status booleans are invalid")
    turns = entry["assistant_turn_count"]
    if type(turns) is not int or turns < 1:
        raise TrainingDataPackageError(f"entry {ordinal}: assistant turns are invalid")
    retained = entry["full_data_member"] is True
    expected = "retained" if retained else "dropped_overlength"
    if entry["overlength"] is retained or entry["preprocessing_status"] != expected:
        raise TrainingDataPackageError(f"entry {ordinal}: preprocessing status differs")


def _validate_receipt_config(
    config: TrainingDataPackageConfig, payload: dict[str, object]
) -> None:
    expected = {
        "schema_version": 1,
        "package_id": config.package_id,
        "phase": config.phase,
        "config_sha256": config.config_sha256,
        "boundaries": dict(config.boundaries),
        "sample": config.sample.to_dict(),
        "tokenizer": config.tokenizer.to_dict(),
        "runtime_versions": dict(config.runtime_versions),
        "training": config.training.receipt_dict(),
        "package_format": config.output_format,
        "entry_schema_version": config.entry_schema_version,
        "limitations": list(config.limitations),
    }
    if any(
        not json_equal_exact(payload[key], value) for key, value in expected.items()
    ):
        raise TrainingDataPackageError("package receipt differs from config")


def _validate_receipt_build(
    config: TrainingDataPackageConfig,
    payload: dict[str, object],
    build: PackageBuild,
    expected_revision: str,
) -> None:
    releases = _object(payload["releases"], "releases")
    _fields(
        releases,
        {
            "executor_release",
            "coordinator_release",
            "input_producer_releases",
            "output_release",
        },
        "releases",
    )
    revision = _revision(releases, "executor_release")
    if revision != expected_revision:
        raise TrainingDataPackageError("receipt executor release differs")
    coordinator = _revision(releases, "coordinator_release")
    if coordinator != revision:
        raise TrainingDataPackageError("coordinator_release differs")
    expected = _receipt_payload(config, revision, build)
    if not json_equal_exact(payload, expected):
        raise TrainingDataPackageError("package receipt metrics or releases differ")


def _source_build(
    config: TrainingDataPackageConfig, sample_path: Path, tokenizer_snapshot: Path
) -> PackageBuild:
    rows = _sample_rows(config, sample_path)
    tokenizer = load_bound_tokenizer(config, tokenizer_snapshot)
    return _build_package(config, tokenizer, rows)


def _verify_output_unchanged(path: Path, expected: bytes, label: str) -> None:
    if _read_isolated_output(path, label).value != expected:
        raise TrainingDataPackageError(f"{label} bytes changed during verification")


def _verify_outputs_unchanged(
    package_path: Path, package_bytes: bytes, receipt_path: Path, receipt_bytes: bytes
) -> None:
    _verify_output_unchanged(package_path, package_bytes, "package")
    _verify_output_unchanged(receipt_path, receipt_bytes, "receipt")


def _verify_package_inputs(
    config: TrainingDataPackageConfig,
    recipe_path: Path,
    sample_receipt_path: Path,
    revision: str,
) -> str:
    _validate_config_identity(config)
    validate_input_provenance(
        config, recipe_path=recipe_path, sample_receipt_path=sample_receipt_path
    )
    return verify_executor_revision(revision)


def verify_training_data_package(
    config: TrainingDataPackageConfig,
    *,
    training_recipe_path: Path,
    training_sample_receipt_path: Path,
    code_revision: str,
    sample_path: Path,
    tokenizer_snapshot: Path,
    package_path: Path,
    receipt_path: Path,
) -> TrainingDataPackageReceipt:
    """Replay source materialization and verify package and receipt bytes."""
    revision = _verify_package_inputs(
        config, training_recipe_path, training_sample_receipt_path, code_revision
    )
    receipt, receipt_bytes = _load_receipt(receipt_path)
    payload = _thaw_mapping(receipt.payload)
    _validate_receipt_config(config, payload)
    package_bytes, entries = _read_package(package_path, receipt.package_sha256)
    if len({entry["example_id"] for entry in entries}) != len(entries):
        raise TrainingDataPackageError("package example IDs must be unique")
    replay = _source_build(config, sample_path, tokenizer_snapshot)
    if replay.package_bytes != package_bytes:
        raise TrainingDataPackageError("package differs from bound source replay")
    _validate_receipt_build(config, payload, replay, revision)
    _verify_outputs_unchanged(package_path, package_bytes, receipt_path, receipt_bytes)
    return receipt


def materialize_training_data_package(
    config: TrainingDataPackageConfig,
    *,
    training_recipe_path: Path,
    training_sample_receipt_path: Path,
    sample_path: Path,
    tokenizer_snapshot: Path,
    code_revision: str,
    package_path: Path,
    receipt_path: Path,
) -> TrainingDataPackageReceipt:
    """Materialize the content-free shared package and compact receipt."""
    if package_path.resolve() == receipt_path.resolve():
        raise TrainingDataPackageError("package and receipt paths must differ")
    _validate_config_identity(config)
    validate_input_provenance(
        config,
        recipe_path=training_recipe_path,
        sample_receipt_path=training_sample_receipt_path,
    )
    revision = verify_executor_revision(code_revision)
    build = _source_build(config, sample_path, tokenizer_snapshot)
    payload = _receipt_payload(config, revision, build)
    receipt = TrainingDataPackageReceipt(payload, canonical_sha256(payload))
    _publish_evidence(
        package_path, build.package_bytes, receipt_path, _receipt_bytes(receipt)
    )
    return receipt
