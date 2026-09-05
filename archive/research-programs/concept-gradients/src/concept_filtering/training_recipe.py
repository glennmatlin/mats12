"""Fail-closed contract for the named rank-32 training reconstruction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.serialization import (
    canonical_sha256,
    file_sha256,
    json_loads_unique,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_TARGET_MODULES = {
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
}
_CLOSED_BOUNDARIES = {
    "scientific_use",
    "scientific_training_allowed",
    "provider_execution_allowed",
    "scheduler_execution_allowed",
    "sae_feature_selection_allowed",
    "concept_scoring_allowed",
    "concept_selected_training_allowed",
}


class TrainingRecipeError(ValueError):
    """Raised when a named training recipe violates its frozen contract."""


@dataclass(frozen=True)
class TrainingRecipe:
    config_path: Path
    recipe_core_json: str
    execution_envelope_json: str
    contract_id: str
    contract_sha256: str
    recipe_id: str
    recipe_sha256: str
    phase: str
    rank: int
    effective_batch_size: int
    scientific_use: bool
    bounded_smoke_execution_allowed: bool
    scientific_training_allowed: bool
    provider_execution_allowed: bool
    scheduler_execution_allowed: bool
    sae_feature_selection_allowed: bool
    concept_scoring_allowed: bool
    concept_selected_training_allowed: bool
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    max_sequence_length: int
    overlength_policy: str
    packing: bool
    truncation: bool
    append_eos: bool
    shuffle: bool
    data_seed: int
    loss_mask: str
    seed: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    precision: str
    gradient_checkpointing: bool
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    max_grad_norm: float
    warmup_steps: int
    save_every_steps: int
    post_resume_absolute_tolerance: float
    runtime_versions: dict[str, str]
    input_producer_releases: dict[str, str]
    training_sample_receipt_sha256: str
    training_sample_row_count: int
    training_sample_identity_scheme: str
    chat_template_sha256: str
    smoke_fixture_path: Path
    smoke_fixture_sha256: str
    smoke_overrides: dict[str, object]
    field_provenance_sha256: str


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TrainingRecipeError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TrainingRecipeError(f"{label} must be a list")
    return cast(list[object], value)


def _fields(value: dict[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise TrainingRecipeError(
            f"{label} unexpected fields={unexpected}, missing fields={missing}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingRecipeError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256.fullmatch(result) is None:
        raise TrainingRecipeError(f"{label} must be a lowercase SHA-256")
    return result


def _revision(value: object, label: str) -> str:
    result = _string(value, label)
    if _REVISION.fullmatch(result) is None:
        raise TrainingRecipeError(f"{label} must be a 40-character revision SHA")
    return result


def _exact(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise TrainingRecipeError(f"{label} must be {expected!r}")


def _read_payload(path: Path) -> dict[str, object]:
    try:
        raw = json_loads_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TrainingRecipeError(f"cannot load recipe: {error}") from error
    payload = _object(raw, "recipe")
    _fields(
        payload,
        {
            "schema_version",
            "contract_id",
            "recipe",
            "execution",
        },
        "contract",
    )
    return payload


def _validate_identity(
    payload: dict[str, object],
    recipe: dict[str, object],
    execution: dict[str, object],
) -> None:
    _exact(payload["schema_version"], 1, "schema_version")
    _string(payload["contract_id"], "contract_id")
    _fields(
        recipe,
        {
            "recipe_id",
            "evidence",
            "artifacts",
            "adapter",
            "data",
            "optimization",
            "training",
            "checkpointing",
            "runtime",
            "field_provenance",
        },
        "recipe",
    )
    _string(recipe["recipe_id"], "recipe.recipe_id")
    _fields(
        execution,
        {"phase", "smoke_fixture", "smoke_overrides", "boundaries", "limitations"},
        "execution",
    )
    _exact(execution["phase"], "bounded_smoke", "execution.phase")


def _evidence_releases(evidence: dict[str, object]) -> dict[str, str]:
    releases = {
        "public_recipe_evidence": _revision(
            evidence["public_recipe_producer_release"],
            "evidence.public_recipe_producer_release",
        ),
        "training_sample_evidence": _revision(
            evidence["training_sample_producer_release"],
            "evidence.training_sample_producer_release",
        ),
    }
    if len(set(releases.values())) != len(releases):
        raise TrainingRecipeError("input producer releases must remain distinct")
    return releases


def _validate_evidence(payload: dict[str, object]) -> tuple[str, dict[str, str]]:
    evidence = _object(payload["evidence"], "evidence")
    _fields(
        evidence,
        {
            "public_recipe_receipt_sha256",
            "public_recipe_producer_release",
            "training_sample_producer_release",
            "source_status",
        },
        "evidence",
    )
    evidence_sha = _sha256(
        evidence["public_recipe_receipt_sha256"],
        "evidence.public_recipe_receipt_sha256",
    )
    _exact(
        evidence["source_status"],
        "named_reconstruction_from_public_rank32",
        "source_status",
    )
    return evidence_sha, _evidence_releases(evidence)


def _validate_repository(value: object, label: str) -> None:
    artifact = _object(value, label)
    _fields(artifact, {"repository", "revision"}, label)
    _string(artifact["repository"], f"{label}.repository")
    _revision(artifact["revision"], f"{label}.revision")


def _validate_sample(value: object) -> str:
    sample = _object(value, "artifacts.training_sample")
    _fields(
        sample,
        {"receipt_sha256", "row_count", "identity_scheme"},
        "artifacts.training_sample",
    )
    result = _sha256(sample["receipt_sha256"], "training_sample.receipt_sha256")
    _exact(sample["row_count"], 25007, "training_sample.row_count")
    _exact(
        sample["identity_scheme"],
        "source-revision-orig-idx-v1",
        "training_sample.identity_scheme",
    )
    return result


def _validate_template(value: object) -> str:
    template = _object(value, "artifacts.chat_template")
    _fields(
        template,
        {"repository", "revision", "path", "sha256"},
        "artifacts.chat_template",
    )
    _string(template["repository"], "chat_template.repository")
    _revision(template["revision"], "chat_template.revision")
    _string(template["path"], "chat_template.path")
    return _sha256(template["sha256"], "chat_template.sha256")


def _validate_artifacts(payload: dict[str, object], evidence_sha: str) -> None:
    artifacts = _object(payload["artifacts"], "artifacts")
    _fields(
        artifacts,
        {"base_model", "tokenizer", "training_sample", "chat_template"},
        "artifacts",
    )
    _validate_repository(artifacts["base_model"], "artifacts.base_model")
    _validate_repository(artifacts["tokenizer"], "artifacts.tokenizer")
    sample_sha = _validate_sample(artifacts["training_sample"])
    template_sha = _validate_template(artifacts["chat_template"])
    if len({evidence_sha, sample_sha, template_sha}) != 3:
        raise TrainingRecipeError("artifact hash roles must remain distinct")


def _validate_adapter(payload: dict[str, object]) -> dict[str, object]:
    adapter = _object(payload["adapter"], "adapter")
    _fields(
        adapter,
        {"method", "rank", "alpha", "dropout", "bias", "target_modules"},
        "adapter",
    )
    for field, expected in (
        ("method", "lora"),
        ("rank", 32),
        ("alpha", 64),
        ("dropout", 0.1),
        ("bias", "none"),
    ):
        _exact(adapter[field], expected, f"adapter.{field}")
    targets = _list(adapter["target_modules"], "adapter.target_modules")
    if not all(isinstance(target, str) and target for target in targets):
        raise TrainingRecipeError(
            "adapter.target_modules must contain non-empty strings"
        )
    if len(targets) != len(_TARGET_MODULES) or set(targets) != _TARGET_MODULES:
        raise TrainingRecipeError("adapter.target_modules must match the frozen set")
    return adapter


def _validate_data(payload: dict[str, object]) -> None:
    data = _object(payload["data"], "data")
    expected = {
        "max_sequence_length": 8192,
        "overlength_policy": "drop",
        "packing": True,
        "truncation": False,
        "append_eos": True,
        "shuffle": True,
        "data_seed": 42,
        "loss_mask": "assistant_only",
    }
    _fields(data, set(expected), "data")
    for field, value in expected.items():
        _exact(data[field], value, f"data.{field}")


def _validate_optimization(payload: dict[str, object]) -> None:
    optimization = _object(payload["optimization"], "optimization")
    expected: dict[str, object] = {
        "optimizer": "adamw_torch",
        "learning_rate": 0.0002,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "scheduler": "linear",
        "warmup_steps": 5,
    }
    _fields(optimization, set(expected), "optimization")
    for field, value in expected.items():
        _exact(optimization[field], value, f"optimization.{field}")


def _effective_batch(training: dict[str, object]) -> int:
    fields = (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "world_size",
    )
    if any(type(training[field]) is not int for field in fields):
        raise TrainingRecipeError("training batch fields must be integers")
    calculated = 1
    for field in fields:
        calculated *= cast(int, training[field])
    if training["effective_batch_size"] != calculated:
        raise TrainingRecipeError("training.effective_batch_size is inconsistent")
    return calculated


def _validate_training(payload: dict[str, object]) -> int:
    training = _object(payload["training"], "training")
    expected: dict[str, object] = {
        "epochs": 1,
        "seed": 42,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "world_size": 8,
        "effective_batch_size": 64,
        "precision": "bfloat16",
        "gradient_checkpointing": True,
    }
    _fields(training, set(expected), "training")
    effective_batch_size = _effective_batch(training)
    for field, value in expected.items():
        _exact(training[field], value, f"training.{field}")
    return effective_batch_size


def _validate_checkpointing(payload: dict[str, object]) -> None:
    checkpointing = _object(payload["checkpointing"], "checkpointing")
    expected: dict[str, object] = {
        "save_every_steps": 25,
        "save_optimizer": True,
        "save_scheduler": True,
        "save_rng": True,
        "resume_mode": "exact_verified",
        "post_resume_absolute_tolerance": 1e-6,
    }
    _fields(checkpointing, set(expected), "checkpointing")
    for field, value in expected.items():
        _exact(checkpointing[field], value, f"checkpointing.{field}")


def _validate_runtime(payload: dict[str, object]) -> dict[str, str]:
    runtime = _object(payload["runtime"], "runtime")
    expected = {
        "huggingface_hub": "0.36.2",
        "peft": "0.18.1",
        "tokenizers": "0.22.2",
        "torch": "2.13.0",
        "transformers": "4.57.6",
    }
    _fields(runtime, set(expected), "runtime")
    for field, value in expected.items():
        _exact(runtime[field], value, f"runtime.{field}")
    return cast(dict[str, str], runtime)


def _validate_boundaries(value: object) -> dict[str, object]:
    boundaries = _object(value, "execution.boundaries")
    expected = _CLOSED_BOUNDARIES | {"bounded_smoke_execution_allowed"}
    _fields(boundaries, expected, "execution.boundaries")
    for field in _CLOSED_BOUNDARIES:
        if boundaries[field] is not False:
            raise TrainingRecipeError(f"{field} must remain false")
    if boundaries["bounded_smoke_execution_allowed"] is not True:
        raise TrainingRecipeError("bounded_smoke_execution_allowed must be true")
    return boundaries


def _leaf_values(value: dict[str, object], prefix: str) -> dict[str, object]:
    leaves: dict[str, object] = {}
    for field, item in value.items():
        path = f"{prefix}.{field}" if prefix else field
        if isinstance(item, dict):
            leaves.update(_leaf_values(cast(dict[str, object], item), path))
        else:
            leaves[path] = item
    return leaves


def _validate_evidence_reference(value: object, label: str) -> object:
    reference = _string(value, label)
    if "#/" not in reference:
        raise TrainingRecipeError(f"{label} must use a JSON Pointer fragment")
    path_text, pointer = reference.split("#/", 1)
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise TrainingRecipeError(f"{label} must use a safe project-relative path")
    try:
        current: object = json_loads_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TrainingRecipeError(f"{label} cannot load evidence: {error}") from error
    for raw in pointer.split("/"):
        field = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or field not in current:
            raise TrainingRecipeError(f"{label} does not resolve")
        current = cast(dict[str, object], current)[field]
    return current


def _compare_evidence(
    target: object, observed: object, comparison: object, label: str
) -> None:
    if comparison == "lowercase":
        if not isinstance(target, str) or not isinstance(observed, str):
            raise TrainingRecipeError(f"{label} lowercase comparison needs strings")
        observed = observed.lower()
    elif comparison != "exact":
        raise TrainingRecipeError(f"{label} has an invalid comparison")
    if type(target) is not type(observed) or target != observed:
        raise TrainingRecipeError(f"{label} does not support the frozen field value")


def _validate_provenance_entry(field: str, raw: object, target: object) -> None:
    label = f"field_provenance.{field}"
    entry = _object(raw, label)
    classification = entry.get("classification")
    if classification == "source_observation":
        _fields(entry, {"classification", "evidence_ref", "comparison"}, label)
        observed = _validate_evidence_reference(entry["evidence_ref"], label)
        _compare_evidence(target, observed, entry["comparison"], label)
    elif classification == "reconstruction_choice":
        _fields(entry, {"classification", "rationale"}, label)
        _string(entry["rationale"], f"{label}.rationale")
    else:
        raise TrainingRecipeError(f"{label} has invalid classification")


def _validate_provenance(recipe: dict[str, object]) -> str:
    sections = (
        "artifacts",
        "adapter",
        "data",
        "optimization",
        "training",
        "checkpointing",
        "runtime",
    )
    expected: dict[str, object] = {}
    for section in sections:
        expected.update(_leaf_values(_object(recipe[section], section), section))
    provenance = _object(recipe["field_provenance"], "field_provenance")
    _fields(provenance, set(expected), "field_provenance")
    for field, raw in provenance.items():
        _validate_provenance_entry(field, raw, expected[field])
    return canonical_sha256(provenance)


def _validate_smoke_fixture(value: object, config_path: Path) -> tuple[Path, str]:
    fixture = _object(value, "execution.smoke_fixture")
    _fields(fixture, {"path", "sha256"}, "execution.smoke_fixture")
    relative = Path(_string(fixture["path"], "execution.smoke_fixture.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise TrainingRecipeError(
            "execution.smoke_fixture.path must be relative and safe"
        )
    expected_sha = _sha256(fixture["sha256"], "execution.smoke_fixture.sha256")
    try:
        config_dir = config_path.parent.resolve(strict=True)
        fixture_path = (config_dir / relative).resolve(strict=True)
        if not fixture_path.is_relative_to(config_dir):
            raise TrainingRecipeError(
                "execution.smoke_fixture.path must remain inside the config directory"
            )
        actual_sha = file_sha256(fixture_path)
    except (OSError, RuntimeError) as error:
        raise TrainingRecipeError(f"cannot load smoke fixture: {error}") from error
    if actual_sha != expected_sha:
        raise TrainingRecipeError("execution.smoke_fixture.sha256 does not match")
    return fixture_path, expected_sha


def _validate_smoke_overrides(value: object) -> dict[str, object]:
    overrides = _object(value, "execution.smoke_overrides")
    expected: dict[str, object] = {
        "model": "tiny_random_olmo3",
        "precision": "float32",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "world_size": 1,
        "examples_per_packed_sequence": 2,
        "checkpoint_step": 1,
    }
    _fields(overrides, set(expected), "execution.smoke_overrides")
    for field, item in expected.items():
        _exact(overrides[field], item, f"execution.smoke_overrides.{field}")
    return overrides


def _validate_limitations(value: object) -> None:
    limitations = _list(value, "execution.limitations")
    if not limitations or not all(
        isinstance(item, str) and item for item in limitations
    ):
        raise TrainingRecipeError("execution.limitations must contain strings")
    if len(limitations) != len(set(cast(list[str], limitations))):
        raise TrainingRecipeError("execution.limitations must be unique")


def _recipe_structure(
    contract: dict[str, object],
    recipe: dict[str, object],
    execution: dict[str, object],
    adapter: dict[str, object],
    effective_batch_size: int,
    boundaries: dict[str, object],
) -> dict[str, Any]:
    return {
        "contract_id": cast(str, contract["contract_id"]),
        "contract_sha256": canonical_sha256(contract),
        "recipe_id": cast(str, recipe["recipe_id"]),
        "recipe_sha256": canonical_sha256(recipe),
        "phase": cast(str, execution["phase"]),
        "rank": cast(int, adapter["rank"]),
        "effective_batch_size": effective_batch_size,
        "alpha": cast(int, adapter["alpha"]),
        "dropout": cast(float, adapter["dropout"]),
        "target_modules": tuple(cast(list[str], adapter["target_modules"])),
        **{field: cast(bool, boundaries[field]) for field in boundaries},
    }


def _recipe_optimization(optimization: dict[str, object]) -> dict[str, Any]:
    betas = cast(list[float], optimization["betas"])
    return {
        "learning_rate": cast(float, optimization["learning_rate"]),
        "betas": (betas[0], betas[1]),
        "epsilon": cast(float, optimization["epsilon"]),
        "weight_decay": cast(float, optimization["weight_decay"]),
        "max_grad_norm": cast(float, optimization["max_grad_norm"]),
        "warmup_steps": cast(int, optimization["warmup_steps"]),
    }


def _recipe_data(recipe: dict[str, object]) -> dict[str, Any]:
    data = _object(recipe["data"], "data")
    training = _object(recipe["training"], "training")
    checkpointing = _object(recipe["checkpointing"], "checkpointing")
    fields = {**data, **training}
    fields.pop("epochs")
    fields.pop("effective_batch_size")
    fields["save_every_steps"] = checkpointing["save_every_steps"]
    fields["post_resume_absolute_tolerance"] = checkpointing[
        "post_resume_absolute_tolerance"
    ]
    return cast(dict[str, Any], fields)


def _recipe_artifacts(recipe: dict[str, object]) -> dict[str, Any]:
    artifacts = _object(recipe["artifacts"], "artifacts")
    sample = _object(artifacts["training_sample"], "artifacts.training_sample")
    template = _object(artifacts["chat_template"], "artifacts.chat_template")
    return {
        "training_sample_receipt_sha256": sample["receipt_sha256"],
        "training_sample_row_count": sample["row_count"],
        "training_sample_identity_scheme": sample["identity_scheme"],
        "chat_template_sha256": template["sha256"],
    }


def _build_recipe(
    structure: dict[str, Any],
    data: dict[str, Any],
    artifacts: dict[str, Any],
    optimization: dict[str, object],
    runtime_versions: dict[str, str],
    producer_releases: dict[str, str],
    smoke_fields: dict[str, Any],
) -> TrainingRecipe:
    values = structure | data | artifacts | _recipe_optimization(optimization)
    values["runtime_versions"] = runtime_versions
    values["input_producer_releases"] = producer_releases
    values.update(smoke_fields)
    return TrainingRecipe(**values)


def _validated_core(recipe: dict[str, object]) -> dict[str, Any]:
    evidence_sha, producer_releases = _validate_evidence(recipe)
    _validate_artifacts(recipe, evidence_sha)
    adapter = _validate_adapter(recipe)
    _validate_data(recipe)
    optimization = _object(recipe["optimization"], "optimization")
    _validate_optimization(recipe)
    effective_batch_size = _validate_training(recipe)
    _validate_checkpointing(recipe)
    return {
        "producer_releases": producer_releases,
        "adapter": adapter,
        "optimization": optimization,
        "effective_batch_size": effective_batch_size,
        "runtime_versions": _validate_runtime(recipe),
        "provenance_sha": _validate_provenance(recipe),
    }


def _validated_execution(execution: dict[str, object], path: Path) -> dict[str, Any]:
    fixture_path, fixture_sha = _validate_smoke_fixture(
        execution["smoke_fixture"], path
    )
    _validate_limitations(execution["limitations"])
    return {
        "boundaries": _validate_boundaries(execution["boundaries"]),
        "fixture_path": fixture_path,
        "fixture_sha": fixture_sha,
        "overrides": _validate_smoke_overrides(execution["smoke_overrides"]),
    }


def _smoke_fields(core: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "smoke_fixture_path": execution["fixture_path"],
        "smoke_fixture_sha256": execution["fixture_sha"],
        "smoke_overrides": execution["overrides"],
        "field_provenance_sha256": core["provenance_sha"],
    }


def _canonical_json(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _integrity_fields(
    path: Path, recipe: dict[str, object], execution: dict[str, object]
) -> dict[str, object]:
    return {
        "config_path": path.resolve(),
        "recipe_core_json": _canonical_json(recipe),
        "execution_envelope_json": _canonical_json(execution),
    }


def load_training_recipe(path: Path) -> TrainingRecipe:
    """Load a frozen named recipe without executing training."""
    payload = _read_payload(path)
    recipe = _object(payload["recipe"], "recipe")
    execution = _object(payload["execution"], "execution")
    _validate_identity(payload, recipe, execution)
    core = _validated_core(recipe)
    smoke_execution = _validated_execution(execution, path)
    structure = _recipe_structure(
        payload,
        recipe,
        execution,
        core["adapter"],
        core["effective_batch_size"],
        smoke_execution["boundaries"],
    )
    return _build_recipe(
        structure,
        _recipe_data(recipe),
        _recipe_artifacts(recipe),
        core["optimization"],
        core["runtime_versions"],
        core["producer_releases"],
        _smoke_fields(core, smoke_execution)
        | _integrity_fields(path, recipe, execution),
    )
