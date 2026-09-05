"""Receipt contract for the bounded named-recipe recovery smoke."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from concept_filtering.chat_rendering import ChatRenderingError, render_assistant_only
from concept_filtering.named_training_recipe import NamedTrainingRecipe
from concept_filtering.serialization import canonical_sha256
from concept_filtering.smoke import verify_executor_revision

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "cursor.json",
    "optimizer.pt",
    "rng.pt",
    "scheduler.pt",
    "trainer_state.json",
}


class TrainingRecoverySmokeError(ValueError):
    """Raised when recovery-smoke evidence violates its frozen contract."""


class TrainingTokenizer(Protocol):
    @property
    def pad_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def eos_token(self) -> str | None: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str


@dataclass(frozen=True)
class TrainingRow:
    orig_idx: int
    messages: tuple[ConversationMessage, ...]


@dataclass(frozen=True)
class PackedBatch:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(frozen=True)
class PackedSmokeData:
    fixture_sha256: str
    source_sample_receipt_sha256: str
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
    batches: tuple[PackedBatch, ...]


@dataclass(frozen=True)
class _SmokeSelection:
    input_example_count: int
    eligible_example_count: int
    dropped_example_ids: tuple[str, ...]
    selected_example_ids: tuple[str, ...]
    tokens: list[int]
    labels: list[int]
    assistant_turn_count: int
    eos_label_count: int


@dataclass(frozen=True)
class _SmokeBuildContract:
    source_revision: str
    seed: int
    sample_limit: int
    sequence_length: int
    overlength_limit: int
    fixture_sha256: str
    source_sample_receipt_sha256: str


@dataclass(frozen=True)
class CheckpointObservation:
    optimizer_step: int
    data_cursor: int
    file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class TrainingRecoveryObservation:
    data: PackedSmokeData
    consumed_packed_sequence_count: int
    checkpoint: CheckpointObservation
    uninterrupted_adapter_sha256: str
    resumed_adapter_sha256: str
    max_abs_difference: float
    uninterrupted_losses: tuple[float, ...]
    checkpoint_resume_losses: tuple[float, ...]
    device: str
    model_dtype: str
    runtime_versions: Mapping[str, str]


@dataclass(frozen=True)
class TrainingRecoveryReceipt:
    payload: dict[str, object]
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "receipt_sha256": self.receipt_sha256}


def _example_id(source_revision: str, row: TrainingRow) -> str:
    return canonical_sha256(
        {"source_revision": source_revision, "orig_idx": row.orig_idx}
    )


def _ordered_rows(
    rows: tuple[TrainingRow, ...], source_revision: str, seed: int
) -> list[tuple[str, TrainingRow]]:
    keyed: list[tuple[str, str, TrainingRow]] = []
    for row in rows:
        example_id = _example_id(source_revision, row)
        key = canonical_sha256(
            {
                "algorithm": "sha256_seeded_sample_id_v1",
                "seed": seed,
                "sample_id": example_id,
            }
        )
        keyed.append((key, example_id, row))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [(example_id, row) for _, example_id, row in keyed]


def _message_payload(row: TrainingRow) -> list[dict[str, str]]:
    if len(row.messages) < 2 or row.messages[-1].role != "assistant":
        raise TrainingRecoverySmokeError(
            "fixture conversation must end with an assistant message"
        )
    roles = {"system", "user", "assistant"}
    if any(message.role not in roles for message in row.messages):
        raise TrainingRecoverySmokeError("fixture conversation role is unsupported")
    return [
        {"role": message.role, "content": message.content} for message in row.messages
    ]


def _render_training_example(
    tokenizer: TrainingTokenizer, row: TrainingRow
) -> tuple[list[int], list[int], int]:
    try:
        rendered = render_assistant_only(tokenizer, _message_payload(row), None)
    except ChatRenderingError as error:
        raise TrainingRecoverySmokeError(str(error)) from error
    return (
        list(rendered.input_ids),
        list(rendered.labels),
        rendered.assistant_turn_count,
    )


def _pack_tokens(
    tokens: list[int], labels: list[int], sequence_length: int, pad_token_id: int
) -> tuple[PackedBatch, ...]:
    batches: list[PackedBatch] = []
    for start in range(0, len(tokens), sequence_length):
        inputs = tokens[start : start + sequence_length]
        masked = labels[start : start + sequence_length]
        padding = sequence_length - len(inputs)
        batches.append(
            PackedBatch(
                (*inputs, *((pad_token_id,) * padding)),
                (*masked, *((-100,) * padding)),
                (*((1,) * len(inputs)), *((0,) * padding)),
            )
        )
    return tuple(batches)


def _batch_sha256(batch: PackedBatch) -> str:
    return canonical_sha256(
        {
            "input_ids": list(batch.input_ids),
            "labels": list(batch.labels),
            "attention_mask": list(batch.attention_mask),
        }
    )


def _packed_sha256(batches: tuple[PackedBatch, ...]) -> str:
    return canonical_sha256([_batch_sha256(batch) for batch in batches])


def _partition_examples(
    ordered: list[tuple[str, TrainingRow]],
    tokenizer: TrainingTokenizer,
    overlength_limit: int,
) -> tuple[list[tuple[str, list[int], list[int], int]], list[str]]:
    eligible: list[tuple[str, list[int], list[int], int]] = []
    dropped: list[str] = []
    for example_id, row in ordered:
        tokens, labels, assistant_turns = _render_training_example(tokenizer, row)
        if len(tokens) > overlength_limit:
            dropped.append(example_id)
        else:
            eligible.append((example_id, tokens, labels, assistant_turns))
    return eligible, dropped


def _selected_streams(
    eligible: list[tuple[str, list[int], list[int], int]], sample_limit: int
) -> tuple[tuple[str, ...], list[int], list[int], int, int]:
    if len(eligible) < sample_limit:
        raise TrainingRecoverySmokeError("eligible fixture examples are insufficient")
    selected = eligible[:sample_limit]
    identifiers = tuple(item[0] for item in selected)
    tokens = [token for _, values, _, _ in selected for token in values]
    labels = [label for _, _, values, _ in selected for label in values]
    assistant_turns = sum(turns for _, _, _, turns in selected)
    return identifiers, tokens, labels, assistant_turns, len(selected)


def _smoke_selection(
    rows: tuple[TrainingRow, ...],
    tokenizer: TrainingTokenizer,
    source_revision: str,
    seed: int,
    sample_limit: int,
    overlength_limit: int,
) -> _SmokeSelection:
    ordered = _ordered_rows(rows, source_revision, seed)
    eligible, dropped = _partition_examples(ordered, tokenizer, overlength_limit)
    identifiers, tokens, labels, assistant_turns, eos_labels = _selected_streams(
        eligible, sample_limit
    )
    return _SmokeSelection(
        len(rows),
        len(eligible),
        tuple(dropped),
        identifiers,
        tokens,
        labels,
        assistant_turns,
        eos_labels,
    )


def _packed_smoke_data(
    selection: _SmokeSelection,
    batches: tuple[PackedBatch, ...],
    batch_hashes: tuple[str, ...],
    fixture_sha256: str,
    source_sample_receipt_sha256: str,
) -> PackedSmokeData:
    return PackedSmokeData(
        fixture_sha256=fixture_sha256,
        source_sample_receipt_sha256=source_sample_receipt_sha256,
        input_example_count=selection.input_example_count,
        eligible_example_count=selection.eligible_example_count,
        dropped_overlength_example_ids=selection.dropped_example_ids,
        selected_example_ids=selection.selected_example_ids,
        token_stream_sha256=canonical_sha256(selection.tokens),
        packed_batches_sha256=_packed_sha256(batches),
        packed_batch_sha256=batch_hashes,
        rendered_token_count=len(selection.tokens),
        assistant_label_token_count=sum(label != -100 for label in selection.labels),
        assistant_turn_count=selection.assistant_turn_count,
        eos_label_count=selection.eos_label_count,
        batches=batches,
    )


def build_packed_smoke_data(
    rows: tuple[TrainingRow, ...],
    tokenizer: TrainingTokenizer,
    *,
    source_revision: str,
    seed: int,
    sample_limit: int,
    sequence_length: int,
    overlength_limit: int,
    fixture_sha256: str,
    source_sample_receipt_sha256: str,
) -> PackedSmokeData:
    """Select, render, and stably pack the bounded smoke sample."""
    contract = _SmokeBuildContract(
        source_revision,
        seed,
        sample_limit,
        sequence_length,
        overlength_limit,
        fixture_sha256,
        source_sample_receipt_sha256,
    )
    return _build_packed_smoke_data(rows, tokenizer, contract)


def _build_packed_smoke_data(
    rows: tuple[TrainingRow, ...],
    tokenizer: TrainingTokenizer,
    contract: _SmokeBuildContract,
) -> PackedSmokeData:
    bounds = (
        contract.sample_limit,
        contract.sequence_length,
        contract.overlength_limit,
    )
    if any(bound < 1 for bound in bounds) or len(rows) < contract.sample_limit:
        raise TrainingRecoverySmokeError("smoke data bounds are invalid")
    pad_token_id = tokenizer.pad_token_id
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise TrainingRecoverySmokeError("tokenizer pad token ID is invalid")
    selection = _smoke_selection(
        rows,
        tokenizer,
        contract.source_revision,
        contract.seed,
        contract.sample_limit,
        contract.overlength_limit,
    )
    batches, batch_hashes = _smoke_batches(selection, contract, pad_token_id)
    fixture = contract.fixture_sha256, contract.source_sample_receipt_sha256
    return _packed_smoke_data(selection, batches, batch_hashes, *fixture)


def _smoke_batches(
    selection: _SmokeSelection,
    contract: _SmokeBuildContract,
    pad_token_id: int,
) -> tuple[tuple[PackedBatch, ...], tuple[str, ...]]:
    batches = _pack_tokens(
        selection.tokens, selection.labels, contract.sequence_length, pad_token_id
    )
    batch_hashes = tuple(_batch_sha256(batch) for batch in batches)
    if len(batch_hashes) < 2 or batch_hashes[0] == batch_hashes[1]:
        raise TrainingRecoverySmokeError("first two ordered batches must be distinct")
    if any(all(label == -100 for label in batch.labels) for batch in batches[:2]):
        raise TrainingRecoverySmokeError("consumed batch assistant loss mask is empty")
    return batches, batch_hashes


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrainingRecoverySmokeError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_selected_ids(
    recipe: NamedTrainingRecipe, observation: TrainingRecoveryObservation
) -> None:
    identifiers = observation.data.selected_example_ids
    if len(identifiers) != recipe.smoke.sample_limit:
        raise TrainingRecoverySmokeError("selected example count differs from recipe")
    if len(set(identifiers)) != len(identifiers):
        raise TrainingRecoverySmokeError("selected example IDs must be unique")
    for index, identifier in enumerate(identifiers):
        _require_sha256(identifier, f"selected_example_ids[{index}]")


def _validate_counts(
    recipe: NamedTrainingRecipe, observation: TrainingRecoveryObservation
) -> None:
    data = observation.data
    if len(data.batches) < observation.consumed_packed_sequence_count:
        raise TrainingRecoverySmokeError("packed sequence count is insufficient")
    if observation.consumed_packed_sequence_count != recipe.smoke.total_optimizer_steps:
        raise TrainingRecoverySmokeError("consumed packed sequence count differs")


def _validate_fixture_data(recipe: NamedTrainingRecipe, data: PackedSmokeData) -> None:
    fixture = recipe.smoke_fixture
    if data.fixture_sha256 != fixture.sha256:
        raise TrainingRecoverySmokeError("smoke fixture SHA-256 differs")
    if data.source_sample_receipt_sha256 != fixture.source_sample_receipt_sha256:
        raise TrainingRecoverySmokeError("source sample receipt SHA-256 differs")
    if data.input_example_count != fixture.row_count:
        raise TrainingRecoverySmokeError("smoke fixture row count differs")
    if (
        data.eligible_example_count + len(data.dropped_overlength_example_ids)
        != data.input_example_count
    ):
        raise TrainingRecoverySmokeError("smoke fixture disposition counts differ")
    if not data.dropped_overlength_example_ids:
        raise TrainingRecoverySmokeError("smoke fixture must exercise overlength drop")
    if data.assistant_turn_count < recipe.smoke.sample_limit:
        raise TrainingRecoverySmokeError("assistant turn count is invalid")
    if data.eos_label_count != recipe.smoke.sample_limit:
        raise TrainingRecoverySmokeError("terminal EOS label count differs")
    if data.assistant_label_token_count <= data.eos_label_count:
        raise TrainingRecoverySmokeError("assistant label token count is invalid")


def _validate_packed_data(data: PackedSmokeData) -> None:
    hashes = tuple(_batch_sha256(batch) for batch in data.batches)
    if hashes != data.packed_batch_sha256:
        raise TrainingRecoverySmokeError("packed batch hashes differ")
    if _packed_sha256(data.batches) != data.packed_batches_sha256:
        raise TrainingRecoverySmokeError("packed batches SHA-256 differs")
    if len(hashes) < 2 or hashes[0] == hashes[1]:
        raise TrainingRecoverySmokeError("first two ordered batches must be distinct")
    if any(all(label == -100 for label in batch.labels) for batch in data.batches[:2]):
        raise TrainingRecoverySmokeError("consumed batch assistant loss mask is empty")
    for index, sha256 in enumerate(hashes):
        _require_sha256(sha256, f"packed_batch_sha256[{index}]")
    _validate_batch_contents(data)


def _validate_batch_contents(data: PackedSmokeData) -> None:
    for batch in data.batches:
        lengths = {len(batch.input_ids), len(batch.labels), len(batch.attention_mask)}
        if len(lengths) != 1 or any(
            value not in {0, 1} for value in batch.attention_mask
        ):
            raise TrainingRecoverySmokeError("packed batch shape or attention differs")
        if any(
            attention == 0 and label != -100
            for attention, label in zip(batch.attention_mask, batch.labels, strict=True)
        ):
            raise TrainingRecoverySmokeError("padding label must remain masked")
    tokens = [
        token
        for batch in data.batches
        for token, attention in zip(batch.input_ids, batch.attention_mask, strict=True)
        if attention == 1
    ]
    labels = [label for batch in data.batches for label in batch.labels]
    if (
        len(tokens) != data.rendered_token_count
        or canonical_sha256(tokens) != data.token_stream_sha256
    ):
        raise TrainingRecoverySmokeError("rendered token stream differs")
    if sum(label != -100 for label in labels) != data.assistant_label_token_count:
        raise TrainingRecoverySmokeError("assistant label token count differs")
    if _supervised_runs(labels) != data.assistant_turn_count:
        raise TrainingRecoverySmokeError("assistant turn count differs")


def _supervised_runs(labels: list[int]) -> int:
    return sum(
        label != -100 and (index == 0 or labels[index - 1] == -100)
        for index, label in enumerate(labels)
    )


def _validate_checkpoint(
    recipe: NamedTrainingRecipe, checkpoint: CheckpointObservation
) -> None:
    if checkpoint.optimizer_step != recipe.smoke.checkpoint_after_optimizer_step:
        raise TrainingRecoverySmokeError("checkpoint optimizer step differs")
    if checkpoint.data_cursor != checkpoint.optimizer_step:
        raise TrainingRecoverySmokeError("checkpoint data cursor differs")
    if set(checkpoint.file_sha256) != _CHECKPOINT_FILES:
        raise TrainingRecoverySmokeError("checkpoint file manifest differs")
    for name, sha256 in checkpoint.file_sha256.items():
        _require_sha256(sha256, f"checkpoint.file_sha256.{name}")


def _validate_runtime(
    recipe: NamedTrainingRecipe, observation: TrainingRecoveryObservation
) -> None:
    if observation.device != recipe.smoke.device:
        raise TrainingRecoverySmokeError("runtime device differs from recipe")
    if observation.model_dtype != recipe.smoke.model_dtype:
        raise TrainingRecoverySmokeError("runtime model dtype differs from recipe")
    if dict(observation.runtime_versions) != dict(recipe.runtime_versions):
        raise TrainingRecoverySmokeError("runtime versions differ from recipe")


def _validate_comparison(
    recipe: NamedTrainingRecipe, observation: TrainingRecoveryObservation
) -> None:
    difference = observation.max_abs_difference
    if not math.isfinite(difference) or difference < 0.0:
        raise TrainingRecoverySmokeError("comparison difference must be finite")
    if difference > recipe.smoke.comparison.absolute_tolerance:
        raise TrainingRecoverySmokeError("adapter tensors exceed declared tolerance")
    _require_sha256(
        observation.uninterrupted_adapter_sha256, "uninterrupted_adapter_sha256"
    )
    _require_sha256(observation.resumed_adapter_sha256, "resumed_adapter_sha256")


def _validate_losses(
    recipe: NamedTrainingRecipe, observation: TrainingRecoveryObservation
) -> None:
    expected = recipe.smoke.total_optimizer_steps
    loss_paths = (
        observation.uninterrupted_losses,
        observation.checkpoint_resume_losses,
    )
    if any(len(losses) != expected for losses in loss_paths):
        raise TrainingRecoverySmokeError("training loss count differs from recipe")
    if any(not math.isfinite(loss) for losses in loss_paths for loss in losses):
        raise TrainingRecoverySmokeError("training losses must be finite")


def _validate_frozen_preprocessing(
    recipe: NamedTrainingRecipe, data: PackedSmokeData
) -> None:
    expected = recipe.smoke.expected_preprocessing
    fields = (
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
    )
    for field in fields:
        if getattr(data, field) != getattr(expected, field):
            raise TrainingRecoverySmokeError(
                f"frozen preprocessing outcome differs at {field}"
            )


def _validate_observation(
    recipe: NamedTrainingRecipe, observation: TrainingRecoveryObservation
) -> None:
    _validate_selected_ids(recipe, observation)
    data = observation.data
    _require_sha256(data.token_stream_sha256, "token_stream_sha256")
    _require_sha256(data.packed_batches_sha256, "packed_batches_sha256")
    for identifier in data.dropped_overlength_example_ids:
        _require_sha256(identifier, "dropped_overlength_example_ids")
    _validate_fixture_data(recipe, data)
    _validate_packed_data(data)
    _validate_frozen_preprocessing(recipe, data)
    _validate_counts(recipe, observation)
    _validate_checkpoint(recipe, observation.checkpoint)
    _validate_runtime(recipe, observation)
    _validate_comparison(recipe, observation)
    _validate_losses(recipe, observation)


def _artifact_payload(recipe: NamedTrainingRecipe) -> dict[str, object]:
    artifacts = recipe.artifacts
    return {
        "base": {
            "repository": artifacts.base.repository,
            "revision": artifacts.base.revision,
            "files": dict(sorted(artifacts.base.files.items())),
        },
        "tokenizer": {
            "repository": artifacts.tokenizer.repository,
            "revision": artifacts.tokenizer.revision,
            "path": artifacts.tokenizer.path,
            "files": dict(sorted(artifacts.tokenizer.files.items())),
        },
        "training_data": {
            "source_repository": artifacts.training_data.source_repository,
            "source_revision": artifacts.training_data.source_revision,
            "repository": artifacts.training_data.repository,
            "revision": artifacts.training_data.revision,
            "path": artifacts.training_data.path,
            "sha256": artifacts.training_data.sha256,
        },
        "smoke_fixture": _smoke_fixture_payload(recipe),
    }


def _smoke_fixture_payload(recipe: NamedTrainingRecipe) -> dict[str, object]:
    fixture = recipe.smoke_fixture
    return {
        "fixture_id": fixture.fixture_id,
        "path": fixture.path,
        "sha256": fixture.sha256,
        "source_sample_receipt_sha256": fixture.source_sample_receipt_sha256,
    }


def _data_payload(observation: TrainingRecoveryObservation) -> dict[str, object]:
    data = observation.data
    return {
        "fixture_sha256": data.fixture_sha256,
        "source_sample_receipt_sha256": data.source_sample_receipt_sha256,
        "input_example_count": data.input_example_count,
        "eligible_example_count": data.eligible_example_count,
        "dropped_overlength_example_ids": list(data.dropped_overlength_example_ids),
        "selected_example_ids": list(data.selected_example_ids),
        "selected_example_ids_sha256": canonical_sha256(
            list(data.selected_example_ids)
        ),
        "token_stream_sha256": data.token_stream_sha256,
        "packed_batches_sha256": data.packed_batches_sha256,
        "packed_batch_sha256": list(data.packed_batch_sha256),
        "rendered_token_count": data.rendered_token_count,
        "assistant_label_token_count": data.assistant_label_token_count,
        "assistant_turn_count": data.assistant_turn_count,
        "eos_label_count": data.eos_label_count,
        "packed_sequence_count": len(data.batches),
        "consumed_packed_sequence_count": observation.consumed_packed_sequence_count,
    }


def _checkpoint_payload(checkpoint: CheckpointObservation) -> dict[str, object]:
    files = dict(sorted(checkpoint.file_sha256.items()))
    return {
        "optimizer_step": checkpoint.optimizer_step,
        "data_cursor": checkpoint.data_cursor,
        "file_sha256": files,
        "manifest_sha256": canonical_sha256(files),
    }


def _runtime_payload(observation: TrainingRecoveryObservation) -> dict[str, object]:
    return {
        "device": observation.device,
        "model_dtype": observation.model_dtype,
        "versions": dict(sorted(observation.runtime_versions.items())),
    }


def _training_payload(
    recipe: NamedTrainingRecipe, observation: TrainingRecoveryObservation
) -> dict[str, object]:
    return {
        "total_optimizer_steps": recipe.smoke.total_optimizer_steps,
        "checkpoint_after_optimizer_step": recipe.smoke.checkpoint_after_optimizer_step,
        "uninterrupted_losses": list(observation.uninterrupted_losses),
        "checkpoint_resume_losses": list(observation.checkpoint_resume_losses),
    }


def _comparison_payload(
    recipe: NamedTrainingRecipe, observation: TrainingRecoveryObservation
) -> dict[str, object]:
    comparison = recipe.smoke.comparison
    return {
        "metric": comparison.metric,
        "absolute_tolerance": comparison.absolute_tolerance,
        "relative_tolerance": comparison.relative_tolerance,
        "max_abs_difference": observation.max_abs_difference,
        "uninterrupted_adapter_sha256": observation.uninterrupted_adapter_sha256,
        "resumed_adapter_sha256": observation.resumed_adapter_sha256,
        "passed": True,
    }


def _authority_payload(recipe: NamedTrainingRecipe) -> dict[str, object]:
    return {
        "sha256": recipe.smoke_authority_sha256,
        "values": dict(sorted(recipe.smoke_authority.items())),
    }


def _receipt_payload(
    recipe: NamedTrainingRecipe,
    code_revision: str,
    observation: TrainingRecoveryObservation,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "smoke_id": f"{recipe.recipe_id}-recovery-smoke-v1",
        "executor_revision": code_revision,
        "authority": _authority_payload(recipe),
        "recipe": {
            "recipe_id": recipe.recipe_id,
            "config_sha256": recipe.config_sha256,
            "core_sha256": recipe.recipe_core_sha256,
            "condition_core_sha256": {
                condition: recipe.recipe_core_sha256 for condition in recipe.conditions
            },
            "seed": recipe.values.training.seed,
        },
        "artifacts": _artifact_payload(recipe),
        "data": _data_payload(observation),
        "checkpoint": _checkpoint_payload(observation.checkpoint),
        "runtime": _runtime_payload(observation),
        "training": _training_payload(recipe, observation),
        "comparison": _comparison_payload(recipe, observation),
    }


def _write_receipt(receipt: TrainingRecoveryReceipt, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as target:
            json.dump(receipt.to_dict(), target, indent=2, sort_keys=True)
            target.write("\n")
    except FileExistsError as error:
        raise TrainingRecoverySmokeError(
            f"refusing to overwrite output: {output_path}"
        ) from error


def execute_training_recovery_smoke(
    recipe: NamedTrainingRecipe,
    *,
    code_revision: str,
    output_path: Path,
    run: Callable[[NamedTrainingRecipe], TrainingRecoveryObservation],
) -> TrainingRecoveryReceipt:
    """Run the bounded backend and write its receipt only after validation."""
    revision = verify_executor_revision(code_revision)
    if output_path.exists():
        raise TrainingRecoverySmokeError(f"refusing to overwrite output: {output_path}")
    observation = run(recipe)
    _validate_observation(recipe, observation)
    payload = _receipt_payload(recipe, revision, observation)
    receipt = TrainingRecoveryReceipt(payload, canonical_sha256(payload))
    _write_receipt(receipt, output_path)
    return receipt
