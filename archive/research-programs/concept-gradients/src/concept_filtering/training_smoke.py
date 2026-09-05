"""Receipt contract for the bounded training checkpoint-resume smoke."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from concept_filtering.serialization import canonical_sha256, json_loads_unique
from concept_filtering.smoke import verify_executor_revision
from concept_filtering.training_recipe import (
    TrainingRecipe,
    TrainingRecipeError,
    load_training_recipe,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng.pt",
    "scheduler.pt",
    "state.json",
}


class TrainingSmokeError(ValueError):
    """Raised when the bounded training smoke violates its contract."""


@dataclass(frozen=True)
class TrainingSmokeObservation:
    runtime_versions: dict[str, str]
    example_count: int
    retained_example_count: int
    dropped_overlength_count: int
    packed_sequence_count: int
    sample_order_sha256: str
    input_ids_sha256: str
    labels_sha256: str
    batch_input_ids_sha256: tuple[str, ...]
    batch_labels_sha256: tuple[str, ...]
    smoke_fixture_sha256: str
    chat_template_sha256: str
    trainable_parameter_count: int
    optimizer_step_count: int
    checkpoint_step: int
    next_batch_index: int
    uninterrupted_losses: tuple[float, ...]
    resumed_losses: tuple[float, ...]
    uninterrupted_state_sha256: str
    resumed_state_sha256: str
    max_abs_parameter_difference: float
    checkpoint_files: dict[str, str]


@dataclass(frozen=True)
class TrainingSmokeReceipt:
    payload: dict[str, object]
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "receipt_sha256": self.receipt_sha256}


def _sha256(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise TrainingSmokeError(f"{label} must be a lowercase SHA-256")


def _validate_runtime(
    recipe: TrainingRecipe, observation: TrainingSmokeObservation
) -> None:
    if observation.runtime_versions != recipe.runtime_versions:
        raise TrainingSmokeError("runtime versions do not match the recipe")


def _validate_data(
    recipe: TrainingRecipe, observation: TrainingSmokeObservation
) -> None:
    counts = (
        observation.example_count,
        observation.retained_example_count,
        observation.dropped_overlength_count,
        observation.packed_sequence_count,
    )
    if counts != (5, 4, 1, 2):
        raise TrainingSmokeError(
            "smoke must load five examples, drop one, and pack two sequences"
        )
    for label, value in (
        ("sample order", observation.sample_order_sha256),
        ("input ids", observation.input_ids_sha256),
        ("labels", observation.labels_sha256),
    ):
        _sha256(value, label)
    _validate_batch_hashes(observation)
    _validate_fixture_hashes(recipe, observation)


def _validate_batch_hashes(observation: TrainingSmokeObservation) -> None:
    if (
        len(observation.batch_input_ids_sha256) != 2
        or len(observation.batch_labels_sha256) != 2
    ):
        raise TrainingSmokeError("smoke must bind two batches")
    values = (*observation.batch_input_ids_sha256, *observation.batch_labels_sha256)
    for value in values:
        _sha256(value, "batch")
    if len(set(observation.batch_input_ids_sha256)) != 2:
        raise TrainingSmokeError("smoke batches must contain distinct inputs")


def _validate_fixture_hashes(
    recipe: TrainingRecipe, observation: TrainingSmokeObservation
) -> None:
    if observation.smoke_fixture_sha256 != recipe.smoke_fixture_sha256:
        raise TrainingSmokeError("smoke fixture identity differs from the recipe")
    if observation.chat_template_sha256 != recipe.chat_template_sha256:
        raise TrainingSmokeError("chat template identity differs from the recipe")


def _validate_steps(observation: TrainingSmokeObservation) -> None:
    if observation.trainable_parameter_count < 1:
        raise TrainingSmokeError("trainable parameter count must be positive")
    steps = (
        observation.optimizer_step_count,
        observation.checkpoint_step,
        observation.next_batch_index,
    )
    if steps != (2, 1, 1):
        raise TrainingSmokeError("smoke must checkpoint after step 1 and finish step 2")


def _validate_losses(
    recipe: TrainingRecipe, observation: TrainingSmokeObservation
) -> None:
    if (
        len(observation.uninterrupted_losses) != 2
        or len(observation.resumed_losses) != 1
    ):
        raise TrainingSmokeError(
            "loss traces must cover uninterrupted and resumed steps"
        )
    values = (*observation.uninterrupted_losses, *observation.resumed_losses)
    if not all(math.isfinite(value) for value in values):
        raise TrainingSmokeError("loss traces must be finite")
    difference = abs(
        observation.uninterrupted_losses[1] - observation.resumed_losses[0]
    )
    if difference > recipe.post_resume_absolute_tolerance:
        raise TrainingSmokeError("post-resume loss exceeds the declared tolerance")


def _validate_state(
    recipe: TrainingRecipe, observation: TrainingSmokeObservation
) -> None:
    _sha256(observation.uninterrupted_state_sha256, "uninterrupted state")
    _sha256(observation.resumed_state_sha256, "resumed state")
    difference = observation.max_abs_parameter_difference
    if not math.isfinite(difference) or difference < 0:
        raise TrainingSmokeError("post-resume parameter difference must be finite")
    if difference > recipe.post_resume_absolute_tolerance:
        raise TrainingSmokeError("post-resume parameters exceed the declared tolerance")


def _validate_checkpoint(observation: TrainingSmokeObservation) -> None:
    if set(observation.checkpoint_files) != _CHECKPOINT_FILES:
        raise TrainingSmokeError("checkpoint files do not match the frozen contract")
    for name, value in observation.checkpoint_files.items():
        _sha256(value, f"checkpoint file {name}")


def _validate_observation(
    recipe: TrainingRecipe, observation: TrainingSmokeObservation
) -> None:
    _validate_runtime(recipe, observation)
    _validate_data(recipe, observation)
    _validate_steps(observation)
    _validate_losses(recipe, observation)
    _validate_state(recipe, observation)
    _validate_checkpoint(observation)


def _release_payload(
    recipe: TrainingRecipe,
    code_revision: str,
    observation: TrainingSmokeObservation,
) -> dict[str, object]:
    output_release = canonical_sha256(
        {
            "executor_release": code_revision,
            "recipe_sha256": recipe.recipe_sha256,
            "contract_sha256": recipe.contract_sha256,
            "resumed_state_sha256": observation.resumed_state_sha256,
            "checkpoint_files": observation.checkpoint_files,
        }
    )
    return {
        "executor_release": code_revision,
        "coordinator_release": code_revision,
        "input_producer_releases": recipe.input_producer_releases,
        "output_release": output_release,
    }


def _execution_payload(
    recipe: TrainingRecipe, observation: TrainingSmokeObservation
) -> dict[str, object]:
    return {
        "example_count": observation.example_count,
        "retained_example_count": observation.retained_example_count,
        "dropped_overlength_count": observation.dropped_overlength_count,
        "packed_sequence_count": observation.packed_sequence_count,
        "sample_order_sha256": observation.sample_order_sha256,
        "input_ids_sha256": observation.input_ids_sha256,
        "labels_sha256": observation.labels_sha256,
        "batch_input_ids_sha256": list(observation.batch_input_ids_sha256),
        "batch_labels_sha256": list(observation.batch_labels_sha256),
        "smoke_fixture_sha256": observation.smoke_fixture_sha256,
        "chat_template_sha256": observation.chat_template_sha256,
        "trainable_parameter_count": observation.trainable_parameter_count,
        "optimizer_step_count": observation.optimizer_step_count,
        "checkpoint_step": observation.checkpoint_step,
        "next_batch_index": observation.next_batch_index,
        "uninterrupted_losses": list(observation.uninterrupted_losses),
        "resumed_losses": list(observation.resumed_losses),
        "uninterrupted_state_sha256": observation.uninterrupted_state_sha256,
        "resumed_state_sha256": observation.resumed_state_sha256,
        "max_abs_parameter_difference": observation.max_abs_parameter_difference,
        "post_resume_absolute_tolerance": recipe.post_resume_absolute_tolerance,
    }


def _receipt_payload(
    recipe: TrainingRecipe,
    code_revision: str,
    observation: TrainingSmokeObservation,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "smoke_id": f"{recipe.recipe_id}-checkpoint-resume-smoke",
        "contract_id": recipe.contract_id,
        "contract_sha256": recipe.contract_sha256,
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": recipe.recipe_sha256,
        "field_provenance_sha256": recipe.field_provenance_sha256,
        "scientific_use": False,
        "bounded_smoke_execution": True,
        "scientific_training_allowed": False,
        "provider_execution_allowed": False,
        "scheduler_execution_allowed": False,
        "sae_feature_selection_allowed": False,
        "concept_scoring_allowed": False,
        "concept_selected_training_allowed": False,
        "runtime_versions": dict(sorted(observation.runtime_versions.items())),
        "declared_recipe": _declared_recipe_payload(recipe),
        "smoke_overrides": recipe.smoke_overrides,
        "releases": _release_payload(recipe, code_revision, observation),
        "execution": _execution_payload(recipe, observation),
        "checkpoint_files": dict(sorted(observation.checkpoint_files.items())),
    }


def _declared_recipe_payload(recipe: TrainingRecipe) -> dict[str, object]:
    return {
        "max_sequence_length": recipe.max_sequence_length,
        "overlength_policy": recipe.overlength_policy,
        "packing": recipe.packing,
        "truncation": recipe.truncation,
        "append_eos": recipe.append_eos,
        "shuffle": recipe.shuffle,
        "data_seed": recipe.data_seed,
        "loss_mask": recipe.loss_mask,
        "per_device_train_batch_size": recipe.per_device_train_batch_size,
        "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
        "world_size": recipe.world_size,
        "effective_batch_size": recipe.effective_batch_size,
        "precision": recipe.precision,
        "gradient_checkpointing": recipe.gradient_checkpointing,
        "save_every_steps": recipe.save_every_steps,
    }


def _write_receipt(receipt: TrainingSmokeReceipt, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as target:
            target.write(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise TrainingSmokeError(
            f"refusing to overwrite output: {output_path}"
        ) from error


def _validate_smoke_authority(recipe: TrainingRecipe) -> None:
    if not recipe.bounded_smoke_execution_allowed:
        raise TrainingSmokeError("bounded training smoke is not allowed")
    closed = (
        recipe.scientific_use,
        recipe.scientific_training_allowed,
        recipe.provider_execution_allowed,
        recipe.scheduler_execution_allowed,
        recipe.sae_feature_selection_allowed,
        recipe.concept_scoring_allowed,
        recipe.concept_selected_training_allowed,
    )
    if any(closed):
        raise TrainingSmokeError("non-smoke authority must remain false")


def _integrity_payload(recipe: TrainingRecipe) -> tuple[object, object]:
    try:
        core = json_loads_unique(recipe.recipe_core_json)
        execution = json_loads_unique(recipe.execution_envelope_json)
    except ValueError as error:
        raise TrainingSmokeError(
            f"recipe integrity payload is invalid: {error}"
        ) from error
    return core, execution


def _validate_recipe_integrity(recipe: TrainingRecipe) -> None:
    core, execution = _integrity_payload(recipe)
    if canonical_sha256(core) != recipe.recipe_sha256:
        raise TrainingSmokeError("recipe core identity differs from its payload")
    contract = {
        "schema_version": 1,
        "contract_id": recipe.contract_id,
        "recipe": core,
        "execution": execution,
    }
    if canonical_sha256(contract) != recipe.contract_sha256:
        raise TrainingSmokeError("smoke contract identity differs from its payload")
    try:
        reloaded = load_training_recipe(recipe.config_path)
    except TrainingRecipeError as error:
        raise TrainingSmokeError(f"cannot revalidate recipe: {error}") from error
    if reloaded != recipe:
        raise TrainingSmokeError("recipe object differs from its canonical config")


def execute_training_smoke(
    recipe: TrainingRecipe,
    *,
    code_revision: str,
    output_path: Path,
    run: Callable[[TrainingRecipe], TrainingSmokeObservation],
) -> TrainingSmokeReceipt:
    """Execute one bounded train-checkpoint-resume path and write its receipt."""
    _validate_smoke_authority(recipe)
    _validate_recipe_integrity(recipe)
    verified_revision = verify_executor_revision(code_revision)
    if output_path.exists():
        raise TrainingSmokeError(f"refusing to overwrite output: {output_path}")
    observation = run(recipe)
    _validate_observation(recipe, observation)
    payload = _receipt_payload(recipe, verified_revision, observation)
    receipt = TrainingSmokeReceipt(payload, canonical_sha256(payload))
    _write_receipt(receipt, output_path)
    return receipt
