"""Fail-closed contract for the named rank-32 training recipe."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar, cast

from concept_filtering.serialization import (
    bytes_sha256,
    canonical_sha256,
    json_equal_exact,
    json_loads_unique,
)

_SHA_LENGTH = 64
_REVISION_LENGTH = 40
_CONDITIONS = (
    "full_data",
    "dense_probe_removal",
    "matched_random_removal",
    "behavior_query_ekfac_removal",
)
_TARGET_MODULES = (
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)
_RUNTIME_PACKAGES = {
    "pyarrow",
    "torch",
    "transformers",
    "peft",
    "safetensors",
    "tokenizers",
}
_SMOKE_AUTHORITY_FIELDS = {
    "scientific_use",
    "starts_experiment_timer",
    "provider_execution",
    "scheduler_submission",
    "full_condition_training",
    "feature_selection",
    "concept_scoring",
}
_RECIPE_CORE_FIELDS = {
    "schema_version",
    "recipe_id",
    "conditions",
    "evidence_receipts",
    "artifacts",
    "recipe",
    "public_trace_derivation",
    "runtime_versions",
    "provenance",
}
_DERIVATION_FIELDS = {
    "optimizer_steps",
    "logged_tokens",
    "train_runtime_seconds",
    "train_samples_per_second",
    "derived_packed_sequence_count",
    "derived_sequences_per_step",
    "selected_effective_batch_size",
    "selected_token_capacity",
    "observed_capacity_fraction",
}
_TRACE_OBSERVATION_FIELDS = {
    "optimizer_steps",
    "logged_tokens",
    "train_runtime_seconds",
    "train_samples_per_second",
}
_TRACE_DERIVATION_FIELDS = _DERIVATION_FIELDS - _TRACE_OBSERVATION_FIELDS
_SMOKE_OVERRIDE_FIELDS = {
    "max_sequence_length",
    "micro_batch_size_per_device",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "precision",
    "checkpoint_interval_optimizer_steps",
}
_SMOKE_FIELDS = {
    "device",
    "model_dtype",
    "sample_limit",
    "total_optimizer_steps",
    "checkpoint_after_optimizer_step",
    "overrides",
    "comparison",
    "expected_preprocessing",
}
_SMOKE_EXPECTED_PREPROCESSING_FIELDS = {
    "input_example_count",
    "eligible_example_count",
    "dropped_overlength_example_ids",
    "selected_example_ids",
    "token_stream_sha256",
    "packed_batches_sha256",
    "packed_batch_sha256",
    "rendered_token_count",
    "assistant_label_token_count",
    "assistant_turn_count",
    "eos_label_count",
}
_ADAPTER_EVIDENCE_FIELDS = {
    "schema_version",
    "artifact_id",
    "kind",
    "status",
    "repository",
    "revision",
    "path",
    "sha256",
    "source_url",
    "evidence",
    "divergence",
}
_DERIVATION_SOURCE_FIELDS = {
    "public_recipe_observation_receipt_sha256",
    "training_sample_receipt_sha256",
    "adapter_config_sha256",
    "trainer_state_sha256",
    "public_organism_sha256",
    "full_data_adapter_receipt_sha256",
}
_RECEIPT_ROLES = {
    "public_load_smoke": (
        "smoke_id",
        "public-rank32-split-1-smoke-v1",
    ),
    "public_recipe_observation": (
        "reconstruction_id",
        "public-rank32-split1-recipe-evidence-v1",
    ),
    "training_sample": (
        "reconstruction_id",
        "public-rank32-train1-reconstruction-v1",
    ),
    "training_derivation": (
        "evidence_id",
        "public-rank32-training-derivation-evidence-v1",
    ),
}
_PUBLIC_OBSERVATION_FIELDS = {
    "lora.rank",
    "lora.alpha",
    "lora.dropout",
    "lora.bias",
    "lora.task_type",
    "lora.target_modules",
    "training.epochs",
    "sequence.max_sequence_length",
    "sequence.chat_template_sha256",
}
_DERIVED_PUBLIC_FIELDS = {"order.identity_scheme"}
_Value = TypeVar("_Value")


class NamedTrainingRecipeError(ValueError):
    """Raised when the named recipe is incomplete or self-contradictory."""


@dataclass(frozen=True)
class FileArtifact:
    repository: str
    revision: str
    path: str | None
    files: Mapping[str, str]


@dataclass(frozen=True)
class TrainingDataArtifact:
    source_repository: str
    source_revision: str
    repository: str
    revision: str
    path: str
    sha256: str
    row_count: int
    identity_scheme: str


@dataclass(frozen=True)
class TrainingArtifacts:
    base: FileArtifact
    tokenizer: FileArtifact
    training_data: TrainingDataArtifact


@dataclass(frozen=True)
class SmokeFixtureArtifact:
    fixture_id: str
    path: str
    sha256: str
    source_sample_receipt_sha256: str
    row_count: int


@dataclass(frozen=True)
class LoraRecipe:
    rank: int
    alpha: int
    dropout: float
    bias: str
    task_type: str
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerRecipe:
    name: str
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    max_grad_norm: float
    fused: bool


@dataclass(frozen=True)
class SchedulerRecipe:
    name: str
    warmup_steps: int


@dataclass(frozen=True)
class BatchingRecipe:
    micro_batch_size_per_device: int
    world_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    drop_last: bool


@dataclass(frozen=True)
class TrainingLoopRecipe:
    epochs: int
    seed: int
    precision: str
    gradient_checkpointing: bool
    use_cache: bool
    dataloader_workers: int


@dataclass(frozen=True)
class SequenceRecipe:
    max_sequence_length: int
    overlength_policy: str
    truncation: bool
    packing: str
    remainder: str
    add_special_tokens: bool
    chat_template_file: str
    chat_template_sha256: str
    loss_mask: str


@dataclass(frozen=True)
class OrderRecipe:
    algorithm: str
    identity_scheme: str


@dataclass(frozen=True)
class CheckpointRecipe:
    interval_optimizer_steps: int
    boundary: str
    save_adapter: bool
    save_optimizer: bool
    save_scheduler: bool
    save_rng: bool
    save_data_cursor: bool
    resume_semantics: str


@dataclass(frozen=True)
class RecipeValues:
    lora: LoraRecipe
    optimizer: OptimizerRecipe
    scheduler: SchedulerRecipe
    batching: BatchingRecipe
    training: TrainingLoopRecipe
    sequence: SequenceRecipe
    order: OrderRecipe
    checkpoint: CheckpointRecipe


@dataclass(frozen=True)
class SmokeOverrides:
    max_sequence_length: int
    micro_batch_size_per_device: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    precision: str
    checkpoint_interval_optimizer_steps: int


@dataclass(frozen=True)
class SmokeComparison:
    metric: str
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True)
class SmokePreprocessingExpectation:
    input_example_count: int
    eligible_example_count: int
    dropped_overlength_example_ids: tuple[str, ...]
    selected_example_ids: tuple[str, ...]
    token_stream_sha256: str
    packed_batches_sha256: str
    packed_batch_sha256: tuple[str, ...]
    rendered_token_count: int
    assistant_label_token_count: int
    assistant_turn_count: int
    eos_label_count: int


@dataclass(frozen=True)
class TrainingSmoke:
    device: str
    model_dtype: str
    sample_limit: int
    total_optimizer_steps: int
    checkpoint_after_optimizer_step: int
    overrides: SmokeOverrides
    comparison: SmokeComparison
    expected_preprocessing: SmokePreprocessingExpectation


@dataclass(frozen=True)
class ProvenanceEntry:
    basis: str
    evidence: str


@dataclass(frozen=True)
class EvidenceBundle:
    sha256: Mapping[str, str]
    receipts: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class NamedTrainingRecipe:
    recipe_id: str
    config_sha256: str
    recipe_core_sha256: str
    smoke_authority_sha256: str
    smoke_authority: Mapping[str, bool]
    conditions: tuple[str, ...]
    evidence_receipt_sha256: Mapping[str, str]
    artifacts: TrainingArtifacts
    values: RecipeValues
    smoke_fixture: SmokeFixtureArtifact
    smoke: TrainingSmoke
    runtime_versions: Mapping[str, str]
    provenance: Mapping[str, ProvenanceEntry]
    divergences: tuple[str, ...]


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NamedTrainingRecipeError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _immutable_mapping(values: dict[str, _Value]) -> Mapping[str, _Value]:
    return MappingProxyType(dict(values))


def _exact_fields(payload: dict[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(payload))
    unexpected = sorted(set(payload) - expected)
    if missing or unexpected:
        raise NamedTrainingRecipeError(
            f"{field} fields differ; missing={missing}, unexpected={unexpected}"
        )


def _field_label(field: str, context: str | None) -> str:
    return f"{context}.{field}" if context is not None else field


def _string(payload: dict[str, Any], field: str, *, context: str | None = None) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise NamedTrainingRecipeError(
            f"{_field_label(field, context)} must be a non-empty string"
        )
    return value


def _integer(
    payload: dict[str, Any],
    field: str,
    minimum: int = 0,
    *,
    context: str | None = None,
) -> int:
    value = payload.get(field)
    if type(value) is not int or value < minimum:
        raise NamedTrainingRecipeError(
            f"{_field_label(field, context)} must be an integer >= {minimum}"
        )
    return value


def _number(
    payload: dict[str, Any],
    field: str,
    minimum: float = 0.0,
    *,
    context: str | None = None,
) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NamedTrainingRecipeError(
            f"{_field_label(field, context)} must be a finite number"
        )
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise NamedTrainingRecipeError(
            f"{_field_label(field, context)} must be a finite number"
        )
    return result


def _boolean(
    payload: dict[str, Any], field: str, *, context: str | None = None
) -> bool:
    value = payload.get(field)
    if type(value) is not bool:
        raise NamedTrainingRecipeError(
            f"{_field_label(field, context)} must be a boolean"
        )
    return value


def _hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise NamedTrainingRecipeError(f"{field} must be {length} lowercase hex digits")
    if any(character not in "0123456789abcdef" for character in value):
        raise NamedTrainingRecipeError(f"{field} must be {length} lowercase hex digits")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise NamedTrainingRecipeError(f"{field} must be a non-empty list")
    items = cast(list[object], value)
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise NamedTrainingRecipeError(f"{field} contains invalid values")
    return tuple(cast(list[str], items))


def _strings(
    payload: dict[str, Any], field: str, *, context: str | None = None
) -> tuple[str, ...]:
    return _string_tuple(payload.get(field), _field_label(field, context))


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        source = path.read_bytes()
        value = json_loads_unique(source.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise NamedTrainingRecipeError(f"cannot load named recipe: {error}") from error
    return _mapping(value, "named recipe")


def _validate_envelope(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "recipe_id",
        "smoke_authority",
        "conditions",
        "evidence_receipts",
        "artifacts",
        "smoke_fixture",
        "recipe",
        "public_trace_derivation",
        "smoke",
        "runtime_versions",
        "provenance",
        "divergences",
    }
    _exact_fields(payload, expected, "named recipe")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise NamedTrainingRecipeError("schema_version must be the integer 1")
    if _strings(payload, "conditions") != _CONDITIONS:
        raise NamedTrainingRecipeError("conditions must name the four fresh controls")


def _evidence_path(config_path: Path, raw: object, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise NamedTrainingRecipeError(f"{field}.path must be relative")
    relative = Path(raw)
    root = config_path.parent.parent.resolve()
    resolved = (config_path.parent / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root):
        raise NamedTrainingRecipeError(f"{field}.path escapes reconstruction root")
    return resolved


def _receipt_payload(source: bytes, role: str) -> dict[str, Any]:
    try:
        value = json_loads_unique(source.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise NamedTrainingRecipeError(f"cannot parse {role}: {error}") from error
    return _mapping(value, f"evidence_receipts.{role}")


def _verify_receipt_body(payload: dict[str, Any], role: str) -> None:
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise NamedTrainingRecipeError(f"{role} receipt schema_version must be 1")
    for field in ("scientific_use", "starts_experiment_timer"):
        if payload.get(field) is not False:
            raise NamedTrainingRecipeError(f"{role} receipt {field} must remain false")
    receipt_sha = _hex(
        payload.get("receipt_sha256"), _SHA_LENGTH, f"{role}.receipt_sha256"
    )
    body = dict(payload)
    body.pop("receipt_sha256")
    if canonical_sha256(body) != receipt_sha:
        raise NamedTrainingRecipeError(f"{role} receipt body SHA-256 mismatch")


def _load_evidence_receipt(
    config_path: Path, role: str, value: object
) -> tuple[str, dict[str, Any]]:
    item = _mapping(value, f"evidence_receipts.{role}")
    _exact_fields(item, {"path", "sha256"}, f"evidence_receipts.{role}")
    expected_sha = _hex(item.get("sha256"), _SHA_LENGTH, f"{role}.sha256")
    receipt_path = _evidence_path(config_path, item.get("path"), role)
    try:
        source = receipt_path.read_bytes()
    except OSError as error:
        raise NamedTrainingRecipeError(f"cannot read {role}: {error}") from error
    if bytes_sha256(source) != expected_sha:
        raise NamedTrainingRecipeError(f"{role} evidence receipt SHA-256 mismatch")
    payload = _receipt_payload(source, role)
    _verify_receipt_body(payload, role)
    id_field, expected_id = _RECEIPT_ROLES[role]
    if payload.get(id_field) != expected_id:
        raise NamedTrainingRecipeError(f"{role} receipt role is incorrect")
    return expected_sha, payload


def _evidence_receipts(path: Path, value: object) -> EvidenceBundle:
    payload = _mapping(value, "evidence_receipts")
    _exact_fields(payload, set(_RECEIPT_ROLES), "evidence_receipts")
    hashes: dict[str, str] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for role in sorted(_RECEIPT_ROLES):
        hashes[role], receipts[role] = _load_evidence_receipt(path, role, payload[role])
    return EvidenceBundle(_immutable_mapping(hashes), receipts)


def _artifact_files(value: object, field: str) -> Mapping[str, str]:
    payload = _mapping(value, f"{field}.files")
    if not payload:
        raise NamedTrainingRecipeError(f"{field}.files must not be empty")
    files: dict[str, str] = {}
    for name, sha256 in payload.items():
        if Path(name).name != name:
            raise NamedTrainingRecipeError(f"{field}.files has an unsafe name")
        files[name] = _hex(sha256, _SHA_LENGTH, f"{field}.files.{name}")
    return _immutable_mapping(files)


def _file_artifact(value: object, field: str, *, has_path: bool) -> FileArtifact:
    payload = _mapping(value, field)
    expected = {"repository", "revision", "files"}
    if has_path:
        expected.add("path")
    _exact_fields(payload, expected, field)
    path = _string(payload, "path", context=field) if has_path else None
    return FileArtifact(
        _string(payload, "repository", context=field),
        _hex(payload.get("revision"), _REVISION_LENGTH, f"{field}.revision"),
        path,
        _artifact_files(payload.get("files"), field),
    )


def _training_data(value: object) -> TrainingDataArtifact:
    payload = _mapping(value, "artifacts.training_data")
    fields = {
        "source_repository",
        "source_revision",
        "repository",
        "revision",
        "path",
        "sha256",
        "row_count",
        "identity_scheme",
    }
    _exact_fields(payload, fields, "artifacts.training_data")
    return TrainingDataArtifact(
        _string(payload, "source_repository", context="artifacts.training_data"),
        _hex(
            payload.get("source_revision"),
            _REVISION_LENGTH,
            "training_data.source_revision",
        ),
        _string(payload, "repository", context="artifacts.training_data"),
        _hex(payload.get("revision"), _REVISION_LENGTH, "training_data.revision"),
        _string(payload, "path", context="artifacts.training_data"),
        _hex(payload.get("sha256"), _SHA_LENGTH, "training_data.sha256"),
        _integer(payload, "row_count", 1, context="artifacts.training_data"),
        _string(payload, "identity_scheme", context="artifacts.training_data"),
    )


def _artifacts(value: object) -> TrainingArtifacts:
    payload = _mapping(value, "artifacts")
    _exact_fields(payload, {"base", "tokenizer", "training_data"}, "artifacts")
    return TrainingArtifacts(
        _file_artifact(payload["base"], "artifacts.base", has_path=False),
        _file_artifact(payload["tokenizer"], "artifacts.tokenizer", has_path=True),
        _training_data(payload["training_data"]),
    )


def _lora(value: object) -> LoraRecipe:
    payload = _mapping(value, "recipe.lora")
    fields = {"rank", "alpha", "dropout", "bias", "task_type", "target_modules"}
    _exact_fields(payload, fields, "recipe.lora")
    targets = _strings(payload, "target_modules", context="recipe.lora")
    if targets != _TARGET_MODULES:
        raise NamedTrainingRecipeError(
            "lora target modules do not match rank-32 evidence"
        )
    return LoraRecipe(
        _integer(payload, "rank", 1, context="recipe.lora"),
        _integer(payload, "alpha", 1, context="recipe.lora"),
        _number(payload, "dropout", context="recipe.lora"),
        _string(payload, "bias", context="recipe.lora"),
        _string(payload, "task_type", context="recipe.lora"),
        targets,
    )


def _optimizer(value: object) -> OptimizerRecipe:
    payload = _mapping(value, "recipe.optimizer")
    fields = {
        "name",
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "max_grad_norm",
        "fused",
    }
    _exact_fields(payload, fields, "recipe.optimizer")
    if _string(payload, "name", context="recipe.optimizer") != "adamw_torch":
        raise NamedTrainingRecipeError("optimizer.name must be adamw_torch")
    return OptimizerRecipe(
        "adamw_torch",
        _number(payload, "learning_rate", context="recipe.optimizer"),
        _number(payload, "beta1", context="recipe.optimizer"),
        _number(payload, "beta2", context="recipe.optimizer"),
        _number(payload, "epsilon", context="recipe.optimizer"),
        _number(payload, "weight_decay", context="recipe.optimizer"),
        _number(payload, "max_grad_norm", context="recipe.optimizer"),
        _boolean(payload, "fused", context="recipe.optimizer"),
    )


def _scheduler(value: object) -> SchedulerRecipe:
    payload = _mapping(value, "recipe.scheduler")
    _exact_fields(payload, {"name", "warmup_steps"}, "recipe.scheduler")
    if _string(payload, "name", context="recipe.scheduler") != "linear":
        raise NamedTrainingRecipeError("scheduler.name must be linear")
    return SchedulerRecipe(
        "linear", _integer(payload, "warmup_steps", 1, context="recipe.scheduler")
    )


def _batching(value: object) -> BatchingRecipe:
    payload = _mapping(value, "recipe.batching")
    fields = {
        "micro_batch_size_per_device",
        "world_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "drop_last",
    }
    _exact_fields(payload, fields, "recipe.batching")
    context = "recipe.batching"
    micro = _integer(payload, "micro_batch_size_per_device", 1, context=context)
    world = _integer(payload, "world_size", 1, context=context)
    accumulation = _integer(payload, "gradient_accumulation_steps", 1, context=context)
    effective = _integer(payload, "effective_batch_size", 1, context=context)
    if effective != micro * world * accumulation:
        raise NamedTrainingRecipeError("effective batch does not match its factors")
    return BatchingRecipe(
        micro,
        world,
        accumulation,
        effective,
        _boolean(payload, "drop_last", context=context),
    )


def _training(value: object) -> TrainingLoopRecipe:
    payload = _mapping(value, "recipe.training")
    fields = {
        "epochs",
        "seed",
        "precision",
        "gradient_checkpointing",
        "use_cache",
        "dataloader_workers",
    }
    _exact_fields(payload, fields, "recipe.training")
    context = "recipe.training"
    precision = _string(payload, "precision", context=context)
    if precision not in {"bfloat16", "float16", "float32"}:
        raise NamedTrainingRecipeError("training.precision is unsupported")
    return TrainingLoopRecipe(
        _integer(payload, "epochs", 1, context=context),
        _integer(payload, "seed", context=context),
        precision,
        _boolean(payload, "gradient_checkpointing", context=context),
        _boolean(payload, "use_cache", context=context),
        _integer(payload, "dataloader_workers", context=context),
    )


def _sequence(value: object) -> SequenceRecipe:
    payload = _mapping(value, "recipe.sequence")
    fields = {
        "max_sequence_length",
        "overlength_policy",
        "truncation",
        "packing",
        "remainder",
        "add_special_tokens",
        "chat_template_file",
        "chat_template_sha256",
        "loss_mask",
    }
    _exact_fields(payload, fields, "recipe.sequence")
    context = "recipe.sequence"
    return SequenceRecipe(
        _integer(payload, "max_sequence_length", 1, context=context),
        _string(payload, "overlength_policy", context=context),
        _boolean(payload, "truncation", context=context),
        _string(payload, "packing", context=context),
        _string(payload, "remainder", context=context),
        _boolean(payload, "add_special_tokens", context=context),
        _string(payload, "chat_template_file", context=context),
        _hex(payload.get("chat_template_sha256"), _SHA_LENGTH, "chat_template_sha256"),
        _string(payload, "loss_mask", context=context),
    )


def _order(value: object) -> OrderRecipe:
    payload = _mapping(value, "recipe.order")
    _exact_fields(payload, {"algorithm", "identity_scheme"}, "recipe.order")
    return OrderRecipe(
        _string(payload, "algorithm", context="recipe.order"),
        _string(payload, "identity_scheme", context="recipe.order"),
    )


def _checkpoint(value: object) -> CheckpointRecipe:
    payload = _mapping(value, "recipe.checkpoint")
    fields = {
        "interval_optimizer_steps",
        "boundary",
        "save_adapter",
        "save_optimizer",
        "save_scheduler",
        "save_rng",
        "save_data_cursor",
        "resume_semantics",
    }
    _exact_fields(payload, fields, "recipe.checkpoint")
    booleans = {
        field: _boolean(payload, field, context="recipe.checkpoint")
        for field in fields
        if field.startswith("save_")
    }
    if not all(booleans.values()):
        raise NamedTrainingRecipeError("checkpoint state flags must all be true")
    return CheckpointRecipe(
        _integer(payload, "interval_optimizer_steps", 1, context="recipe.checkpoint"),
        _string(payload, "boundary", context="recipe.checkpoint"),
        booleans["save_adapter"],
        booleans["save_optimizer"],
        booleans["save_scheduler"],
        booleans["save_rng"],
        booleans["save_data_cursor"],
        _string(payload, "resume_semantics", context="recipe.checkpoint"),
    )


def _recipe_values(value: object) -> RecipeValues:
    payload = _mapping(value, "recipe")
    fields = {
        "lora",
        "optimizer",
        "scheduler",
        "batching",
        "training",
        "sequence",
        "order",
        "checkpoint",
    }
    _exact_fields(payload, fields, "recipe")
    return RecipeValues(
        _lora(payload["lora"]),
        _optimizer(payload["optimizer"]),
        _scheduler(payload["scheduler"]),
        _batching(payload["batching"]),
        _training(payload["training"]),
        _sequence(payload["sequence"]),
        _order(payload["order"]),
        _checkpoint(payload["checkpoint"]),
    )


def _evidence_equal(observed: object, expected: object, field: str) -> None:
    if not json_equal_exact(observed, expected):
        raise NamedTrainingRecipeError(f"{field} contradicts public evidence")


def _validate_public_recipe_evidence(
    receipt: dict[str, Any], recipe: RecipeValues
) -> None:
    adapter = _mapping(receipt.get("adapter_observation"), "adapter_observation")
    trainer = _mapping(
        receipt.get("trainer_state_observation"), "trainer_state_observation"
    )
    expected = {
        "r": recipe.lora.rank,
        "lora_alpha": recipe.lora.alpha,
        "lora_dropout": recipe.lora.dropout,
        "task_type": recipe.lora.task_type,
        "target_modules": list(recipe.lora.target_modules),
    }
    for field, value in expected.items():
        _evidence_equal(adapter.get(field), value, f"public recipe evidence.{field}")
    trainer_expected = {
        "num_train_epochs": recipe.training.epochs,
        "peak_learning_rate": recipe.optimizer.learning_rate,
        "peak_learning_rate_step": recipe.scheduler.warmup_steps,
        "train_batch_size": recipe.batching.micro_batch_size_per_device,
    }
    for field, value in trainer_expected.items():
        _evidence_equal(trainer.get(field), value, f"public recipe evidence.{field}")


def _validate_training_sample_evidence(
    receipt: dict[str, Any], artifact: TrainingDataArtifact
) -> None:
    source = _mapping(receipt.get("source"), "training_sample.source")
    source_expected = {
        "repository": artifact.source_repository,
        "revision": artifact.source_revision,
    }
    for field, value in source_expected.items():
        _evidence_equal(source.get(field), value, f"training source evidence.{field}")
    sample = _mapping(receipt.get("sample"), "training_sample.sample")
    expected = {
        "repository": artifact.repository,
        "revision": artifact.revision,
        "path": artifact.path,
        "sha256": artifact.sha256,
    }
    for field, value in expected.items():
        _evidence_equal(sample.get(field), value, f"training sample evidence.{field}")
    _evidence_equal(
        receipt.get("row_count"),
        artifact.row_count,
        "training sample evidence.row_count",
    )
    _evidence_equal(
        receipt.get("identity_scheme"),
        artifact.identity_scheme,
        "training sample evidence.identity_scheme",
    )


def _derivation_receipt_parts(
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fields = {
        "schema_version",
        "evidence_id",
        "scientific_use",
        "starts_experiment_timer",
        "sources",
        "observations",
        "derivation",
        "caveat",
        "receipt_sha256",
    }
    _exact_fields(receipt, fields, "training_derivation receipt")
    _string(receipt, "caveat", context="training_derivation")
    return (
        _mapping(receipt["sources"], "training_derivation.sources"),
        _mapping(receipt["observations"], "training_derivation.observations"),
        _mapping(receipt["derivation"], "training_derivation.derivation"),
    )


def _validate_derivation_sources(
    evidence: EvidenceBundle,
    sources: dict[str, Any],
    organism_sha256: str,
    adapter_evidence_sha256: str,
) -> None:
    _exact_fields(sources, _DERIVATION_SOURCE_FIELDS, "training_derivation.sources")
    recipe_receipt = evidence.receipts["public_recipe_observation"]
    files = _mapping(recipe_receipt.get("files"), "public recipe files")
    expected = {
        "public_recipe_observation_receipt_sha256": evidence.sha256[
            "public_recipe_observation"
        ],
        "training_sample_receipt_sha256": evidence.sha256["training_sample"],
        "adapter_config_sha256": files.get("adapter_config.json"),
        "trainer_state_sha256": files.get("trainer_state.json"),
        "public_organism_sha256": organism_sha256,
        "full_data_adapter_receipt_sha256": adapter_evidence_sha256,
    }
    for field, value in expected.items():
        _evidence_equal(
            sources.get(field), value, f"public trace evidence.sources.{field}"
        )


def _read_named_evidence(path: Path, label: str) -> tuple[str, dict[str, Any]]:
    try:
        source = path.read_bytes()
        value = json_loads_unique(source.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise NamedTrainingRecipeError(f"cannot read {label}: {error}") from error
    return bytes_sha256(source), _mapping(value, label)


def _public_organism(config_path: Path) -> tuple[str, dict[str, Any]]:
    path = config_path.parent.parent / "public-organism-v1.json"
    return _read_named_evidence(path, "public organism evidence")


def _public_adapter_evidence(config_path: Path) -> tuple[str, dict[str, Any]]:
    path = config_path.parent.parent / "receipts/full-data-adapter.json"
    return _read_named_evidence(path, "public adapter evidence")


def _validate_adapter_evidence_envelope(evidence: dict[str, Any]) -> None:
    _exact_fields(evidence, _ADAPTER_EVIDENCE_FIELDS, "public adapter evidence")
    expected = {
        "schema_version": 1,
        "artifact_id": "full-data-adapter",
        "kind": "adapter",
        "status": "candidate",
    }
    for field, value in expected.items():
        _evidence_equal(evidence.get(field), value, f"public adapter evidence.{field}")
    _hex(evidence.get("sha256"), _SHA_LENGTH, "public adapter evidence.sha256")


def _validate_adapter_lineage(
    evidence: dict[str, Any],
    recipe_receipt: dict[str, Any],
    artifacts: TrainingArtifacts,
) -> None:
    _validate_adapter_evidence_envelope(evidence)
    public = _mapping(recipe_receipt.get("artifact"), "public recipe artifact")
    adapter_path = f"{public.get('path')}/adapter_model.safetensors"
    expected = {
        "adapter.repository": (evidence.get("repository"), public.get("repository")),
        "adapter.revision": (evidence.get("revision"), public.get("revision")),
        "adapter.path": (evidence.get("path"), adapter_path),
        "tokenizer.repository": (
            artifacts.tokenizer.repository,
            public.get("repository"),
        ),
        "tokenizer.revision": (artifacts.tokenizer.revision, public.get("revision")),
        "tokenizer.path": (artifacts.tokenizer.path, public.get("path")),
    }
    for field, (observed, source) in expected.items():
        _evidence_equal(observed, source, f"public adapter evidence.{field}")


def _read_raw_json(
    path: Path, label: str, expected_sha256: object, *, strip_newline: bool = False
) -> dict[str, Any]:
    try:
        source = path.read_bytes()
    except OSError as error:
        raise NamedTrainingRecipeError(f"cannot read {label}: {error}") from error
    if strip_newline and source.endswith(b"\n"):
        source = source[:-1]
    if bytes_sha256(source) != expected_sha256:
        raise NamedTrainingRecipeError(f"{label} SHA-256 mismatch")
    return _receipt_payload(source, label)


def _raw_public_sources(
    config_path: Path, recipe_receipt: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    files = _mapping(recipe_receipt.get("files"), "public recipe files")
    root = config_path.parent.parent / "evidence/public-rank32-split1"
    adapter = _read_raw_json(
        root / "adapter_config.json",
        "raw adapter evidence",
        files.get("adapter_config.json"),
        strip_newline=True,
    )
    trainer = _read_raw_json(
        root / "trainer_state.json",
        "raw trainer evidence",
        files.get("trainer_state.json"),
    )
    return adapter, trainer


def _raw_adapter_values(payload: dict[str, Any]) -> dict[str, object]:
    targets = sorted(_strings(payload, "target_modules", context="raw adapter"))
    return {
        "r": _integer(payload, "r", 1, context="raw adapter"),
        "lora_alpha": _integer(payload, "lora_alpha", 1, context="raw adapter"),
        "lora_dropout": _number(payload, "lora_dropout", context="raw adapter"),
        "bias": _string(payload, "bias", context="raw adapter"),
        "task_type": _string(payload, "task_type", context="raw adapter"),
        "target_modules": targets,
    }


def _validate_raw_adapter(
    raw: dict[str, Any], recipe_receipt: dict[str, Any], recipe: RecipeValues
) -> None:
    values = _raw_adapter_values(raw)
    observed = _mapping(
        recipe_receipt.get("adapter_observation"), "public adapter observation"
    )
    for field in values.keys() - {"bias"}:
        source_value = observed.get(field)
        if field == "target_modules":
            source_value = sorted(_string_tuple(source_value, "adapter targets"))
        _evidence_equal(values[field], source_value, f"raw adapter evidence.{field}")
    selected = {
        "r": recipe.lora.rank,
        "lora_alpha": recipe.lora.alpha,
        "lora_dropout": recipe.lora.dropout,
        "bias": recipe.lora.bias,
        "task_type": recipe.lora.task_type,
        "target_modules": list(recipe.lora.target_modules),
    }
    for field, value in selected.items():
        _evidence_equal(values[field], value, f"raw adapter evidence.{field}")


def _trainer_logs(
    payload: dict[str, Any], field: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_logs = payload.get("log_history")
    if not isinstance(raw_logs, list) or not raw_logs:
        raise NamedTrainingRecipeError(f"{field}.log_history must be a list")
    items = cast(list[object], raw_logs)
    if not all(isinstance(item, dict) for item in items):
        raise NamedTrainingRecipeError(f"{field}.log_history contains invalid values")
    logs = cast(list[dict[str, Any]], items)
    token_logs = [item for item in logs if "num_tokens" in item]
    terminal = [item for item in logs if "train_runtime" in item]
    if not token_logs or len(terminal) != 1:
        raise NamedTrainingRecipeError(f"{field}.log_history has invalid summaries")
    return token_logs, terminal[0]


def _raw_trainer_summary(payload: dict[str, Any]) -> dict[str, object]:
    token_logs, terminal = _trainer_logs(payload, "raw trainer evidence")
    peak = max(
        token_logs,
        key=lambda item: _number(item, "learning_rate", context="raw trainer evidence"),
    )
    final = token_logs[-1]
    return {
        "epoch": _number(payload, "epoch", context="raw trainer evidence"),
        "global_step": _integer(payload, "global_step", 1),
        "max_steps": _integer(payload, "max_steps", 1),
        "num_train_epochs": _integer(payload, "num_train_epochs", 1),
        "train_batch_size": _integer(payload, "train_batch_size", 1),
        "num_input_tokens_seen": _integer(payload, "num_input_tokens_seen"),
        "logged_step_count": len(token_logs),
        "final_logged_num_tokens": _number(final, "num_tokens"),
        "peak_learning_rate": _number(peak, "learning_rate"),
        "peak_learning_rate_step": _integer(peak, "step", 1),
        "final_logged_learning_rate": _number(final, "learning_rate"),
        "train_runtime_seconds": _number(terminal, "train_runtime"),
        "train_samples_per_second": _number(terminal, "train_samples_per_second"),
    }


def _validate_raw_trainer(
    raw: dict[str, Any], recipe_receipt: dict[str, Any], observations: dict[str, Any]
) -> None:
    summary = _raw_trainer_summary(raw)
    public = _mapping(
        recipe_receipt.get("trainer_state_observation"),
        "public trainer observation",
    )
    public_fields = set(summary) - {
        "train_runtime_seconds",
        "train_samples_per_second",
    }
    for field in public_fields:
        _evidence_equal(
            public.get(field), summary[field], f"raw trainer evidence.{field}"
        )
    trace_expected = {
        "optimizer_steps": summary["global_step"],
        "logged_tokens": int(cast(float, summary["final_logged_num_tokens"])),
        "train_runtime_seconds": summary["train_runtime_seconds"],
        "train_samples_per_second": summary["train_samples_per_second"],
    }
    for field, value in trace_expected.items():
        _evidence_equal(observations.get(field), value, f"raw trainer evidence.{field}")


def _match_trace_fields(
    trace: dict[str, Any], evidence: dict[str, Any], fields: set[str], context: str
) -> None:
    for field in fields:
        _evidence_equal(trace.get(field), evidence.get(field), f"{context}.{field}")


def _validate_trace_shapes(
    observations: dict[str, Any], derivation: dict[str, Any]
) -> None:
    _exact_fields(
        observations,
        {"adapter_bias", "max_sequence_length", *_TRACE_OBSERVATION_FIELDS},
        "training_derivation.observations",
    )
    _exact_fields(
        derivation,
        {"formula_version", *_TRACE_DERIVATION_FIELDS},
        "training_derivation.derivation",
    )
    _evidence_equal(
        derivation.get("formula_version"),
        "packed-effective-batch-v1",
        "public trace evidence.derivation.formula_version",
    )


def _validate_trace_evidence(
    observations: dict[str, Any],
    derivation: dict[str, Any],
    trace: dict[str, Any],
) -> None:
    _validate_trace_shapes(observations, derivation)
    _match_trace_fields(
        trace,
        observations,
        _TRACE_OBSERVATION_FIELDS,
        "public trace evidence.observations",
    )
    _match_trace_fields(
        trace,
        derivation,
        _TRACE_DERIVATION_FIELDS,
        "public trace evidence.derivation",
    )


def _load_smoke_observed(receipt: dict[str, Any]) -> dict[str, object]:
    base = _mapping(receipt.get("base"), "public load smoke base")
    adapter = _mapping(receipt.get("adapter"), "public load smoke adapter")
    template = _mapping(receipt.get("template"), "public load smoke template")
    return {
        "base.repository": base.get("repository"),
        "base.revision": base.get("revision"),
        "adapter.repository": adapter.get("repository"),
        "adapter.revision": adapter.get("revision"),
        "adapter.path": adapter.get("path"),
        "adapter.sha256": adapter.get("sha256"),
        "template.path": template.get("path"),
        "template.sha256": template.get("sha256"),
        "rank": receipt.get("rank"),
    }


def _load_smoke_expected(
    artifacts: TrainingArtifacts,
    recipe: RecipeValues,
    adapter_evidence: dict[str, Any],
) -> dict[str, object]:
    template_path = f"{artifacts.tokenizer.path}/{recipe.sequence.chat_template_file}"
    return {
        "base.repository": artifacts.base.repository,
        "base.revision": artifacts.base.revision,
        "adapter.repository": adapter_evidence.get("repository"),
        "adapter.revision": adapter_evidence.get("revision"),
        "adapter.path": adapter_evidence.get("path"),
        "adapter.sha256": adapter_evidence.get("sha256"),
        "template.path": template_path,
        "template.sha256": recipe.sequence.chat_template_sha256,
        "rank": recipe.lora.rank,
    }


def _validate_public_load_smoke(
    receipt: dict[str, Any],
    artifacts: TrainingArtifacts,
    recipe: RecipeValues,
    adapter_evidence: dict[str, Any],
) -> None:
    observed = _load_smoke_observed(receipt)
    expected = _load_smoke_expected(artifacts, recipe, adapter_evidence)
    for field, value in expected.items():
        _evidence_equal(observed.get(field), value, f"public adapter evidence.{field}")


def _validate_public_organism(
    organism: dict[str, Any], observations: dict[str, Any]
) -> None:
    _exact_fields(
        organism,
        {"schema_version", "organism_id", "artifact_receipts", "parameters"},
        "public organism evidence",
    )
    if organism.get("schema_version") != 1:
        raise NamedTrainingRecipeError("public organism evidence schema is unsupported")
    parameters = _mapping(organism["parameters"], "public organism parameters")
    maximum = _mapping(
        parameters.get("data.max_sequence_length"),
        "public organism data.max_sequence_length",
    )
    _exact_fields(
        maximum, {"status", "value", "evidence"}, "public organism max length"
    )
    _evidence_equal(
        maximum.get("status"), "observed", "public organism evidence.status"
    )
    _evidence_equal(
        maximum.get("value"),
        observations.get("max_sequence_length"),
        "public organism evidence.data.max_sequence_length",
    )


def _validate_selected_observations(
    observations: dict[str, Any], recipe: RecipeValues
) -> None:
    expected = {
        "adapter_bias": recipe.lora.bias,
        "max_sequence_length": recipe.sequence.max_sequence_length,
    }
    for field, value in expected.items():
        _evidence_equal(
            observations.get(field), value, f"public recipe evidence.{field}"
        )


def _validate_adapter_sources(
    evidence: EvidenceBundle,
    artifacts: TrainingArtifacts,
    recipe: RecipeValues,
    adapter_evidence: dict[str, Any],
    raw_adapter: dict[str, Any],
) -> None:
    recipe_receipt = evidence.receipts["public_recipe_observation"]
    _validate_adapter_lineage(adapter_evidence, recipe_receipt, artifacts)
    _validate_public_load_smoke(
        evidence.receipts["public_load_smoke"],
        artifacts,
        recipe,
        adapter_evidence,
    )
    _validate_raw_adapter(raw_adapter, recipe_receipt, recipe)


def _validate_semantic_evidence(
    config_path: Path,
    evidence: EvidenceBundle,
    artifacts: TrainingArtifacts,
    recipe: RecipeValues,
    trace: dict[str, Any],
) -> None:
    recipe_receipt = evidence.receipts["public_recipe_observation"]
    adapter_sha256, adapter_evidence = _public_adapter_evidence(config_path)
    raw_adapter, raw_trainer = _raw_public_sources(config_path, recipe_receipt)
    _validate_adapter_sources(
        evidence, artifacts, recipe, adapter_evidence, raw_adapter
    )
    _validate_public_recipe_evidence(recipe_receipt, recipe)
    _validate_training_sample_evidence(
        evidence.receipts["training_sample"], artifacts.training_data
    )
    sources, observations, derivation = _derivation_receipt_parts(
        evidence.receipts["training_derivation"]
    )
    organism_sha256, organism = _public_organism(config_path)
    _validate_derivation_sources(evidence, sources, organism_sha256, adapter_sha256)
    _validate_public_organism(organism, observations)
    _validate_raw_trainer(raw_trainer, recipe_receipt, observations)
    _validate_trace_evidence(observations, derivation, trace)
    _validate_selected_observations(observations, recipe)


def _flatten_leaves(value: dict[str, Any], prefix: str = "") -> set[str]:
    leaves: set[str] = set()
    for name, item in value.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            leaves.update(_flatten_leaves(cast(dict[str, Any], item), path))
        else:
            leaves.add(path)
    return leaves


def _provenance(value: object, recipe: dict[str, Any]) -> Mapping[str, ProvenanceEntry]:
    payload = _mapping(value, "provenance")
    leaves = _flatten_leaves(recipe)
    if set(payload) != leaves:
        missing = sorted(leaves - set(payload))
        extra = sorted(set(payload) - leaves)
        raise NamedTrainingRecipeError(
            f"provenance coverage differs; missing={missing}, unexpected={extra}"
        )
    entries: dict[str, ProvenanceEntry] = {}
    allowed = {"public_observation", "derived_public_evidence", "reconstruction_choice"}
    for field, item in payload.items():
        context = f"provenance.{field}"
        record = _mapping(item, context)
        _exact_fields(record, {"basis", "evidence"}, context)
        basis = _string(record, "basis", context=context)
        if basis not in allowed:
            raise NamedTrainingRecipeError(f"provenance.{field}.basis is unsupported")
        expected_basis = _expected_provenance_basis(field)
        if basis != expected_basis:
            raise NamedTrainingRecipeError(
                f"provenance.{field}.basis must be {expected_basis}"
            )
        entries[field] = ProvenanceEntry(
            basis, _string(record, "evidence", context=context)
        )
    return _immutable_mapping(entries)


def _expected_provenance_basis(field: str) -> str:
    if field in _PUBLIC_OBSERVATION_FIELDS:
        return "public_observation"
    if field in _DERIVED_PUBLIC_FIELDS:
        return "derived_public_evidence"
    return "reconstruction_choice"


def _assert_close(observed: float, expected: float, field: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise NamedTrainingRecipeError(f"{field} does not match public evidence")


def _validate_derivation(value: object, recipe: RecipeValues) -> None:
    payload = _mapping(value, "public_trace_derivation")
    _exact_fields(payload, _DERIVATION_FIELDS, "public_trace_derivation")
    context = "public_trace_derivation"
    steps = _integer(payload, "optimizer_steps", 1, context=context)
    tokens = _integer(payload, "logged_tokens", 1, context=context)
    runtime = _number(payload, "train_runtime_seconds", context=context)
    rate = _number(payload, "train_samples_per_second", context=context)
    packed = _number(payload, "derived_packed_sequence_count", context=context)
    per_step = _number(payload, "derived_sequences_per_step", context=context)
    _assert_close(packed, runtime * rate, "packed sequence derivation")
    _assert_close(per_step, packed / steps, "sequences-per-step derivation")
    effective = _integer(payload, "selected_effective_batch_size", 1, context=context)
    if (
        effective != recipe.batching.effective_batch_size
        or abs(per_step - effective) >= 1
    ):
        raise NamedTrainingRecipeError("selected effective batch contradicts trace")
    capacity = steps * effective * recipe.sequence.max_sequence_length
    if _integer(payload, "selected_token_capacity", 1, context=context) != capacity:
        raise NamedTrainingRecipeError("selected token capacity is inconsistent")
    _assert_close(
        _number(payload, "observed_capacity_fraction", context=context),
        tokens / capacity,
        "observed capacity fraction",
    )


def _smoke_overrides(value: object, recipe: RecipeValues) -> SmokeOverrides:
    payload = _mapping(value, "smoke.overrides")
    _exact_fields(payload, _SMOKE_OVERRIDE_FIELDS, "smoke.overrides")
    context = "smoke.overrides"
    maximum = _integer(payload, "max_sequence_length", 1, context=context)
    micro = _integer(payload, "micro_batch_size_per_device", 1, context=context)
    accumulation = _integer(payload, "gradient_accumulation_steps", 1, context=context)
    effective = _integer(payload, "effective_batch_size", 1, context=context)
    if maximum > recipe.sequence.max_sequence_length:
        raise NamedTrainingRecipeError("smoke sequence length exceeds recipe")
    if effective != micro * accumulation:
        raise NamedTrainingRecipeError("smoke effective batch is inconsistent")
    return SmokeOverrides(
        maximum,
        micro,
        accumulation,
        effective,
        _string(payload, "precision", context=context),
        _integer(
            payload,
            "checkpoint_interval_optimizer_steps",
            1,
            context=context,
        ),
    )


def _smoke_comparison(value: object) -> SmokeComparison:
    payload = _mapping(value, "smoke.comparison")
    fields = {"metric", "absolute_tolerance", "relative_tolerance"}
    _exact_fields(payload, fields, "smoke.comparison")
    context = "smoke.comparison"
    metric = _string(payload, "metric", context=context)
    if metric != "adapter_tensor_max_abs":
        raise NamedTrainingRecipeError("smoke comparison metric is unsupported")
    return SmokeComparison(
        metric,
        _number(payload, "absolute_tolerance", context=context),
        _number(payload, "relative_tolerance", context=context),
    )


def _sha256_tuple(payload: dict[str, Any], field: str, context: str) -> tuple[str, ...]:
    values = _strings(payload, field, context=context)
    return tuple(
        _hex(value, _SHA_LENGTH, f"{context}.{field}[{index}]")
        for index, value in enumerate(values)
    )


def _smoke_expected_preprocessing(value: object) -> SmokePreprocessingExpectation:
    context = "smoke.expected_preprocessing"
    payload = _mapping(value, context)
    _exact_fields(payload, _SMOKE_EXPECTED_PREPROCESSING_FIELDS, context)
    return SmokePreprocessingExpectation(
        _integer(payload, "input_example_count", 1, context=context),
        _integer(payload, "eligible_example_count", 1, context=context),
        _sha256_tuple(payload, "dropped_overlength_example_ids", context),
        _sha256_tuple(payload, "selected_example_ids", context),
        _hex(
            payload.get("token_stream_sha256"),
            _SHA_LENGTH,
            f"{context}.token_stream_sha256",
        ),
        _hex(
            payload.get("packed_batches_sha256"),
            _SHA_LENGTH,
            f"{context}.packed_batches_sha256",
        ),
        _sha256_tuple(payload, "packed_batch_sha256", context),
        _integer(payload, "rendered_token_count", 1, context=context),
        _integer(payload, "assistant_label_token_count", 1, context=context),
        _integer(payload, "assistant_turn_count", 1, context=context),
        _integer(payload, "eos_label_count", 1, context=context),
    )


def _validate_smoke_expected(
    expected: SmokePreprocessingExpectation, sample_limit: int, total_steps: int
) -> None:
    dispositions = expected.eligible_example_count + len(
        expected.dropped_overlength_example_ids
    )
    contradictory = (
        expected.input_example_count != dispositions
        or expected.eligible_example_count < sample_limit
        or len(expected.selected_example_ids) != sample_limit
        or expected.assistant_turn_count < sample_limit
        or expected.eos_label_count != sample_limit
        or len(expected.packed_batch_sha256) < total_steps
        or expected.assistant_label_token_count < expected.assistant_turn_count
        or expected.assistant_label_token_count < expected.eos_label_count
        or expected.rendered_token_count < expected.assistant_label_token_count
    )
    if contradictory:
        raise NamedTrainingRecipeError("smoke expected preprocessing is contradictory")


def _validate_smoke_execution(
    recipe: RecipeValues,
    overrides: SmokeOverrides,
    model_dtype: str,
    checkpoint: int,
) -> None:
    contradictory = (
        model_dtype != overrides.precision
        or overrides.precision != recipe.training.precision
        or checkpoint % overrides.checkpoint_interval_optimizer_steps != 0
    )
    if contradictory:
        raise NamedTrainingRecipeError("smoke execution contract is contradictory")


def _smoke(value: object, recipe: RecipeValues) -> TrainingSmoke:
    payload = _mapping(value, "smoke")
    _exact_fields(payload, _SMOKE_FIELDS, "smoke")
    context = "smoke"
    total = _integer(payload, "total_optimizer_steps", 2, context=context)
    checkpoint = _integer(
        payload, "checkpoint_after_optimizer_step", 1, context=context
    )
    if checkpoint >= total:
        raise NamedTrainingRecipeError("smoke checkpoint must precede terminal step")
    model_dtype = _string(payload, "model_dtype", context=context)
    overrides = _smoke_overrides(payload["overrides"], recipe)
    _validate_smoke_execution(recipe, overrides, model_dtype, checkpoint)
    sample_limit = _integer(payload, "sample_limit", 1, context=context)
    expected = _smoke_expected_preprocessing(payload["expected_preprocessing"])
    _validate_smoke_expected(expected, sample_limit, total)
    return TrainingSmoke(
        _string(payload, "device", context=context),
        model_dtype,
        sample_limit,
        total,
        checkpoint,
        overrides,
        _smoke_comparison(payload["comparison"]),
        expected,
    )


def _validate_cross_field_identity(
    artifacts: TrainingArtifacts, recipe: RecipeValues
) -> None:
    template = recipe.sequence.chat_template_file
    if artifacts.tokenizer.files.get(template) != recipe.sequence.chat_template_sha256:
        raise NamedTrainingRecipeError("chat template manifest is contradictory")
    if recipe.order.identity_scheme != artifacts.training_data.identity_scheme:
        raise NamedTrainingRecipeError("order identity scheme is contradictory")


def _runtime_versions(value: object) -> Mapping[str, str]:
    payload = _mapping(value, "runtime_versions")
    _exact_fields(payload, _RUNTIME_PACKAGES, "runtime_versions")
    versions = {
        name: _string(payload, name, context="runtime_versions")
        for name in sorted(payload)
    }
    return _immutable_mapping(versions)


def _smoke_authority(value: object) -> Mapping[str, bool]:
    payload = _mapping(value, "smoke_authority")
    _exact_fields(payload, _SMOKE_AUTHORITY_FIELDS, "smoke_authority")
    authority: dict[str, bool] = {}
    for field in sorted(payload):
        if payload[field] is not False:
            raise NamedTrainingRecipeError(f"smoke_authority.{field} must remain false")
        authority[field] = False
    return _immutable_mapping(authority)


def _fixture_path(config_path: Path, raw: object) -> tuple[str, Path]:
    value = _string({"path": raw}, "path", context="smoke_fixture")
    relative = Path(value)
    root = config_path.parent.resolve()
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root):
        raise NamedTrainingRecipeError("smoke_fixture.path escapes training root")
    return value, resolved


def _fixture_payload(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        source = path.read_bytes()
        value = json_loads_unique(source.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise NamedTrainingRecipeError(f"cannot load smoke fixture: {error}") from error
    if bytes_sha256(source) != expected_sha256:
        raise NamedTrainingRecipeError("smoke fixture SHA-256 mismatch")
    return _mapping(value, "smoke fixture")


def _fixture_source_contract(
    payload: dict[str, Any], evidence: EvidenceBundle, artifacts: TrainingArtifacts
) -> None:
    source = _mapping(payload.get("source_contract"), "smoke fixture source_contract")
    expected_fields = {
        "source_sample_receipt_sha256",
        "source_repository",
        "source_revision",
        "identity_scheme",
        "row_fields",
        "message_fields",
    }
    _exact_fields(source, expected_fields, "smoke fixture source_contract")
    expected: dict[str, object] = {
        "source_sample_receipt_sha256": evidence.sha256["training_sample"],
        "source_repository": artifacts.training_data.source_repository,
        "source_revision": artifacts.training_data.source_revision,
        "identity_scheme": artifacts.training_data.identity_scheme,
        "row_fields": ["orig_idx", "messages"],
        "message_fields": ["role", "content"],
    }
    for field, value in expected.items():
        _evidence_equal(source.get(field), value, f"smoke fixture source.{field}")


def _smoke_fixture(
    config_path: Path,
    value: object,
    evidence: EvidenceBundle,
    artifacts: TrainingArtifacts,
) -> SmokeFixtureArtifact:
    descriptor = _mapping(value, "smoke_fixture")
    fields = {"fixture_id", "path", "sha256", "source_sample_receipt_sha256"}
    _exact_fields(descriptor, fields, "smoke_fixture")
    fixture_id = _string(descriptor, "fixture_id", context="smoke_fixture")
    path_value, path = _fixture_path(config_path, descriptor.get("path"))
    sha256 = _hex(descriptor.get("sha256"), _SHA_LENGTH, "smoke_fixture.sha256")
    receipt_sha = _hex(
        descriptor.get("source_sample_receipt_sha256"),
        _SHA_LENGTH,
        "smoke_fixture.source_sample_receipt_sha256",
    )
    if receipt_sha != evidence.sha256["training_sample"]:
        raise NamedTrainingRecipeError(
            "smoke fixture source sample receipt SHA-256 differs"
        )
    payload = _fixture_payload(path, sha256)
    _validate_fixture_envelope(payload, fixture_id, evidence, artifacts)
    rows = payload["rows"]
    return SmokeFixtureArtifact(fixture_id, path_value, sha256, receipt_sha, len(rows))


def _validate_fixture_envelope(
    payload: dict[str, Any],
    fixture_id: str,
    evidence: EvidenceBundle,
    artifacts: TrainingArtifacts,
) -> None:
    fields = {"schema_version", "fixture_id", "source_contract", "rows"}
    _exact_fields(payload, fields, "smoke fixture")
    if payload.get("schema_version") != 1 or payload.get("fixture_id") != fixture_id:
        raise NamedTrainingRecipeError("smoke fixture identity is incorrect")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise NamedTrainingRecipeError("smoke fixture rows must be a non-empty list")
    _fixture_source_contract(payload, evidence, artifacts)


def _recipe_core_sha256(payload: dict[str, Any]) -> str:
    core = {field: payload[field] for field in sorted(_RECIPE_CORE_FIELDS)}
    return canonical_sha256(core)


def _divergences(value: object) -> tuple[str, ...]:
    return _string_tuple(value, "divergences")


def load_named_training_recipe(path: Path) -> NamedTrainingRecipe:
    """Load the complete named recipe and verify its checked-in evidence."""
    payload = _load_payload(path)
    _validate_envelope(payload)
    recipe_payload = _mapping(payload["recipe"], "recipe")
    values = _recipe_values(recipe_payload)
    artifacts = _artifacts(payload["artifacts"])
    trace = _mapping(payload["public_trace_derivation"], "public_trace_derivation")
    _validate_derivation(trace, values)
    evidence = _evidence_receipts(path, payload["evidence_receipts"])
    _validate_semantic_evidence(path, evidence, artifacts, values, trace)
    _validate_cross_field_identity(artifacts, values)
    authority = _smoke_authority(payload["smoke_authority"])
    fixture = _smoke_fixture(path, payload["smoke_fixture"], evidence, artifacts)
    return NamedTrainingRecipe(
        _string(payload, "recipe_id", context="named recipe"),
        canonical_sha256(payload),
        _recipe_core_sha256(payload),
        canonical_sha256(dict(authority)),
        authority,
        _CONDITIONS,
        evidence.sha256,
        artifacts,
        values,
        fixture,
        _smoke(payload["smoke"], values),
        _runtime_versions(payload["runtime_versions"]),
        _provenance(payload["provenance"], recipe_payload),
        _divergences(payload["divergences"]),
    )
