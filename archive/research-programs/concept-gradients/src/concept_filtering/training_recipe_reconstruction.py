"""Receipt-bound observations from the public rank-32 training artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from concept_filtering.serialization import bytes_sha256, canonical_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FILES = {"adapter_config.json", "trainer_state.json", "rng_state_0.pth"}
_UNRESOLVED = {
    "optimizer",
    "scheduler_configuration",
    "gradient_accumulation",
    "initial_seed",
    "precision",
    "packing_and_sequence_policy",
    "data_order",
}
Trace = list[dict[str, int | float]]


class TrainingRecipeReconstructionError(ValueError):
    """Raised when public recipe evidence fails its frozen contract."""


@dataclass(frozen=True)
class RecipeArtifact:
    repository: str
    revision: str
    path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "path": self.path,
        }


@dataclass(frozen=True)
class UnresolvedRecipeField:
    status: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class AdapterExpectation:
    base_model_name_or_path: str
    revision: None
    peft_type: str
    peft_version: str
    task_type: str
    rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainerExpectation:
    global_step: int
    max_steps: int
    epoch: float
    num_train_epochs: int
    train_batch_size: int
    num_input_tokens_seen: int
    logged_step_count: int
    final_logged_num_tokens: float
    peak_learning_rate: float
    peak_learning_rate_step: int
    final_logged_learning_rate: float


@dataclass(frozen=True)
class TrainingRecipeReconstruction:
    reconstruction_id: str
    scientific_use: bool
    starts_experiment_timer: bool
    artifact: RecipeArtifact
    files: dict[str, str]
    adapter: AdapterExpectation
    trainer: TrainerExpectation
    unresolved: dict[str, UnresolvedRecipeField]
    divergences: tuple[str, ...]
    config_sha256: str


@dataclass(frozen=True)
class TrainingRecipeReceipt:
    reconstruction_id: str
    scientific_use: bool
    starts_experiment_timer: bool
    artifact: RecipeArtifact
    files: dict[str, str]
    adapter: AdapterExpectation
    trainer: TrainerExpectation
    learning_rate_trace_sha256: str
    num_tokens_trace_sha256: str
    unresolved: dict[str, UnresolvedRecipeField]
    divergences: tuple[str, ...]
    config_sha256: str
    receipt_sha256: str

    @property
    def rank(self) -> int:
        return self.adapter.rank

    @property
    def logged_step_count(self) -> int:
        return self.trainer.logged_step_count

    @property
    def final_logged_num_tokens(self) -> float:
        return self.trainer.final_logged_num_tokens

    def body_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "reconstruction_id": self.reconstruction_id,
            "scientific_use": self.scientific_use,
            "starts_experiment_timer": self.starts_experiment_timer,
            "artifact": self.artifact.to_dict(),
            "files": self.files,
            "adapter_observation": _adapter_dict(self.adapter),
            "trainer_state_observation": _trainer_dict(self.trainer),
            "learning_rate_trace_sha256": self.learning_rate_trace_sha256,
            "num_tokens_trace_sha256": self.num_tokens_trace_sha256,
            "unresolved": _unresolved_dict(self.unresolved),
            "divergences": list(self.divergences),
            "config_sha256": self.config_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body_dict(), "receipt_sha256": self.receipt_sha256}


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingRecipeReconstructionError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_fields(payload: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise TrainingRecipeReconstructionError(
            f"{field} fields differ; missing={missing}, unexpected={unexpected}"
        )


def _string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TrainingRecipeReconstructionError(f"{field} must be a string")
    return value


def _integer(payload: dict[str, Any], field: str, *, minimum: int = 0) -> int:
    value = payload.get(field)
    if type(value) is not int or value < minimum:
        raise TrainingRecipeReconstructionError(f"{field} must be an integer")
    return value


def _number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TrainingRecipeReconstructionError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingRecipeReconstructionError(f"{field} must be finite")
    return result


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingRecipeReconstructionError(
            f"cannot load {path}: {error}"
        ) from error
    return _mapping(value, "reconstruction")


def _validate_envelope(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "reconstruction_id",
        "scientific_use",
        "starts_experiment_timer",
        "artifact",
        "files",
        "expected",
        "unresolved",
        "divergences",
    }
    _exact_fields(payload, expected, "reconstruction")
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise TrainingRecipeReconstructionError("schema_version must be 1")
    for field in ("scientific_use", "starts_experiment_timer"):
        if payload[field] is not False:
            raise TrainingRecipeReconstructionError(f"{field} must remain false")


def _artifact(value: object) -> RecipeArtifact:
    payload = _mapping(value, "artifact")
    _exact_fields(payload, {"repository", "revision", "path"}, "artifact")
    revision = _string(payload, "revision")
    if _REVISION.fullmatch(revision) is None:
        raise TrainingRecipeReconstructionError("artifact revision must be a SHA")
    return RecipeArtifact(
        repository=_string(payload, "repository"),
        revision=revision,
        path=_string(payload, "path"),
    )


def _files(value: object) -> dict[str, str]:
    payload = _mapping(value, "files")
    _exact_fields(payload, _FILES, "files")
    files = {name: _string(payload, name) for name in sorted(_FILES)}
    if any(_SHA256.fullmatch(sha256) is None for sha256 in files.values()):
        raise TrainingRecipeReconstructionError("file hashes must be SHA-256 values")
    return files


def _string_list(payload: dict[str, Any], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise TrainingRecipeReconstructionError(f"{field} must be a list")
    items = cast(list[object], value)
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise TrainingRecipeReconstructionError(f"{field} contains invalid values")
    strings = cast(list[str], items)
    if len(strings) != len(set(strings)):
        raise TrainingRecipeReconstructionError(f"{field} contains duplicates")
    return tuple(sorted(strings))


def _adapter_expectation(value: object) -> AdapterExpectation:
    payload = _mapping(value, "expected.adapter")
    expected = {
        "base_model_name_or_path",
        "revision",
        "peft_type",
        "peft_version",
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
    }
    _exact_fields(payload, expected, "expected.adapter")
    if payload["revision"] is not None:
        raise TrainingRecipeReconstructionError(
            "expected adapter revision must be null"
        )
    return AdapterExpectation(
        base_model_name_or_path=_string(payload, "base_model_name_or_path"),
        revision=None,
        peft_type=_string(payload, "peft_type"),
        peft_version=_string(payload, "peft_version"),
        task_type=_string(payload, "task_type"),
        rank=_integer(payload, "r", minimum=1),
        lora_alpha=_integer(payload, "lora_alpha", minimum=1),
        lora_dropout=_number(payload, "lora_dropout"),
        target_modules=_string_list(payload, "target_modules"),
    )


def _trainer_expectation(value: object) -> TrainerExpectation:
    payload = _mapping(value, "expected.trainer_state")
    expected = {
        "global_step",
        "max_steps",
        "epoch",
        "num_train_epochs",
        "train_batch_size",
        "num_input_tokens_seen",
        "logged_step_count",
        "final_logged_num_tokens",
        "peak_learning_rate",
        "peak_learning_rate_step",
        "final_logged_learning_rate",
    }
    _exact_fields(payload, expected, "expected.trainer_state")
    return TrainerExpectation(
        global_step=_integer(payload, "global_step", minimum=1),
        max_steps=_integer(payload, "max_steps", minimum=1),
        epoch=_number(payload, "epoch"),
        num_train_epochs=_integer(payload, "num_train_epochs", minimum=1),
        train_batch_size=_integer(payload, "train_batch_size", minimum=1),
        num_input_tokens_seen=_integer(payload, "num_input_tokens_seen"),
        logged_step_count=_integer(payload, "logged_step_count", minimum=1),
        final_logged_num_tokens=_number(payload, "final_logged_num_tokens"),
        peak_learning_rate=_number(payload, "peak_learning_rate"),
        peak_learning_rate_step=_integer(payload, "peak_learning_rate_step", minimum=1),
        final_logged_learning_rate=_number(payload, "final_logged_learning_rate"),
    )


def _unresolved_item(value: object, field: str) -> UnresolvedRecipeField:
    payload = _mapping(value, field)
    _exact_fields(payload, {"status", "reason"}, field)
    status = _string(payload, "status")
    if status != "unavailable":
        raise TrainingRecipeReconstructionError(f"{field}.status must be unavailable")
    return UnresolvedRecipeField(status=status, reason=_string(payload, "reason"))


def _unresolved(value: object) -> dict[str, UnresolvedRecipeField]:
    payload = _mapping(value, "unresolved")
    _exact_fields(payload, _UNRESOLVED, "unresolved")
    return {name: _unresolved_item(payload[name], name) for name in sorted(_UNRESOLVED)}


def _divergences(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TrainingRecipeReconstructionError("divergences must be a list")
    items = cast(list[object], value)
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise TrainingRecipeReconstructionError("divergences contains invalid values")
    return tuple(cast(list[str], items))


def load_training_recipe_reconstruction(path: Path) -> TrainingRecipeReconstruction:
    """Load a fail-closed observation contract for public training artifacts."""
    payload = _load_payload(path)
    _validate_envelope(payload)
    expected = _mapping(payload["expected"], "expected")
    _exact_fields(expected, {"adapter", "trainer_state"}, "expected")
    return TrainingRecipeReconstruction(
        reconstruction_id=_string(payload, "reconstruction_id"),
        scientific_use=False,
        starts_experiment_timer=False,
        artifact=_artifact(payload["artifact"]),
        files=_files(payload["files"]),
        adapter=_adapter_expectation(expected["adapter"]),
        trainer=_trainer_expectation(expected["trainer_state"]),
        unresolved=_unresolved(payload["unresolved"]),
        divergences=_divergences(payload["divergences"]),
        config_sha256=canonical_sha256(payload),
    )


def _bound_bytes(
    config: TrainingRecipeReconstruction, directory: Path, name: str
) -> bytes:
    try:
        content = (directory / name).read_bytes()
    except OSError as error:
        raise TrainingRecipeReconstructionError(
            f"cannot read {name}: {error}"
        ) from error
    if bytes_sha256(content) != config.files[name]:
        raise TrainingRecipeReconstructionError(f"{name} SHA-256 does not match")
    return content


def _bound_json(
    config: TrainingRecipeReconstruction, directory: Path, name: str
) -> dict[str, Any]:
    content = _bound_bytes(config, directory, name)
    try:
        value: object = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TrainingRecipeReconstructionError(f"invalid {name}: {error}") from error
    return _mapping(value, name)


def _adapter_dict(value: AdapterExpectation) -> dict[str, object]:
    return {
        "base_model_name_or_path": value.base_model_name_or_path,
        "revision": value.revision,
        "peft_type": value.peft_type,
        "peft_version": value.peft_version,
        "task_type": value.task_type,
        "r": value.rank,
        "lora_alpha": value.lora_alpha,
        "lora_dropout": value.lora_dropout,
        "target_modules": list(value.target_modules),
    }


def _trainer_dict(value: TrainerExpectation) -> dict[str, object]:
    return {
        "global_step": value.global_step,
        "max_steps": value.max_steps,
        "epoch": value.epoch,
        "num_train_epochs": value.num_train_epochs,
        "train_batch_size": value.train_batch_size,
        "num_input_tokens_seen": value.num_input_tokens_seen,
        "logged_step_count": value.logged_step_count,
        "final_logged_num_tokens": value.final_logged_num_tokens,
        "peak_learning_rate": value.peak_learning_rate,
        "peak_learning_rate_step": value.peak_learning_rate_step,
        "final_logged_learning_rate": value.final_logged_learning_rate,
    }


def _unresolved_dict(
    values: dict[str, UnresolvedRecipeField],
) -> dict[str, object]:
    return {name: value.to_dict() for name, value in sorted(values.items())}


def _observed_adapter(payload: dict[str, Any]) -> AdapterExpectation:
    selected = {
        "base_model_name_or_path": payload.get("base_model_name_or_path"),
        "revision": payload.get("revision"),
        "peft_type": payload.get("peft_type"),
        "peft_version": payload.get("peft_version"),
        "task_type": payload.get("task_type"),
        "r": payload.get("r"),
        "lora_alpha": payload.get("lora_alpha"),
        "lora_dropout": payload.get("lora_dropout"),
        "target_modules": payload.get("target_modules"),
    }
    return _adapter_expectation(selected)


def _step_logs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload.get("log_history")
    if not isinstance(history, list):
        raise TrainingRecipeReconstructionError("log_history must be a list")
    logs: list[dict[str, Any]] = []
    for item in cast(list[object], history):
        if isinstance(item, dict):
            payload_item = cast(dict[str, Any], item)
            if "learning_rate" in payload_item:
                logs.append(payload_item)
    if not logs:
        raise TrainingRecipeReconstructionError("learning-rate trace is empty")
    return logs


def _trace_values(
    logs: list[dict[str, Any]],
) -> tuple[list[int], list[float], list[float]]:
    steps = [_integer(item, "step", minimum=1) for item in logs]
    learning_rates = [_number(item, "learning_rate") for item in logs]
    num_tokens = [_number(item, "num_tokens") for item in logs]
    if steps != list(range(1, len(logs) + 1)):
        raise TrainingRecipeReconstructionError("logged steps must be sequential")
    if any(rate < 0 for rate in learning_rates):
        raise TrainingRecipeReconstructionError("learning rates must be nonnegative")
    if any(left > right for left, right in pairwise(num_tokens)):
        raise TrainingRecipeReconstructionError("num_tokens must be nondecreasing")
    return steps, learning_rates, num_tokens


def _validate_completed_state(payload: dict[str, Any], logged_steps: int) -> None:
    global_step = _integer(payload, "global_step", minimum=1)
    max_steps = _integer(payload, "max_steps", minimum=1)
    if global_step != max_steps or global_step != logged_steps:
        raise TrainingRecipeReconstructionError(
            "completed trainer-state step count must match the logged trace"
        )


def _observed_trainer(
    payload: dict[str, Any],
) -> tuple[TrainerExpectation, Trace, Trace]:
    logs = _step_logs(payload)
    steps, rates, tokens = _trace_values(logs)
    _validate_completed_state(payload, len(logs))
    peak_rate = max(rates)
    observed = TrainerExpectation(
        global_step=_integer(payload, "global_step", minimum=1),
        max_steps=_integer(payload, "max_steps", minimum=1),
        epoch=_number(payload, "epoch"),
        num_train_epochs=_integer(payload, "num_train_epochs", minimum=1),
        train_batch_size=_integer(payload, "train_batch_size", minimum=1),
        num_input_tokens_seen=_integer(payload, "num_input_tokens_seen"),
        logged_step_count=len(logs),
        final_logged_num_tokens=tokens[-1],
        peak_learning_rate=peak_rate,
        peak_learning_rate_step=steps[rates.index(peak_rate)],
        final_logged_learning_rate=rates[-1],
    )
    rate_trace = [
        {"step": step, "learning_rate": rate}
        for step, rate in zip(steps, rates, strict=True)
    ]
    token_trace = [
        {"step": step, "num_tokens": token}
        for step, token in zip(steps, tokens, strict=True)
    ]
    return observed, rate_trace, token_trace


def _assert_equal(field: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise TrainingRecipeReconstructionError(f"{field} does not match config")


def _build_receipt(
    config: TrainingRecipeReconstruction,
    rate_trace: Trace,
    token_trace: Trace,
) -> TrainingRecipeReceipt:
    receipt = TrainingRecipeReceipt(
        reconstruction_id=config.reconstruction_id,
        scientific_use=False,
        starts_experiment_timer=False,
        artifact=config.artifact,
        files=config.files,
        adapter=config.adapter,
        trainer=config.trainer,
        learning_rate_trace_sha256=canonical_sha256(rate_trace),
        num_tokens_trace_sha256=canonical_sha256(token_trace),
        unresolved=config.unresolved,
        divergences=config.divergences,
        config_sha256=config.config_sha256,
        receipt_sha256="",
    )
    return replace(receipt, receipt_sha256=canonical_sha256(receipt.body_dict()))


def verify_training_recipe_reconstruction(
    config: TrainingRecipeReconstruction, *, artifact_dir: Path
) -> TrainingRecipeReceipt:
    """Verify recovered recipe observations without creating a runnable recipe."""
    adapter = _observed_adapter(
        _bound_json(config, artifact_dir, "adapter_config.json")
    )
    trainer_payload = _bound_json(config, artifact_dir, "trainer_state.json")
    trainer, rate_trace, token_trace = _observed_trainer(trainer_payload)
    _bound_bytes(config, artifact_dir, "rng_state_0.pth")
    _assert_equal("adapter observation", adapter, config.adapter)
    _assert_equal("trainer-state observation", trainer, config.trainer)
    return _build_receipt(config, rate_trace, token_trace)


def write_training_recipe_receipt(receipt: TrainingRecipeReceipt, path: Path) -> None:
    """Write a compact recipe-evidence receipt without replacing prior evidence."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            receipt.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True
        )
        handle.write("\n")
