"""Transformers backend for the named-recipe checkpoint recovery smoke."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.named_training_recipe import NamedTrainingRecipe
from concept_filtering.serialization import (
    bytes_sha256,
    file_sha256,
    json_equal_exact,
    json_loads_unique,
)
from concept_filtering.training_recovery_smoke import (
    CheckpointObservation,
    ConversationMessage,
    PackedBatch,
    PackedSmokeData,
    TrainingRecoveryObservation,
    TrainingRecoverySmokeError,
    TrainingRow,
    TrainingTokenizer,
    build_packed_smoke_data,
)

LOGGER = logging.getLogger(__name__)
_ADAPTER_MARKERS = (
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
)


@dataclass
class _Components:
    model: Any
    optimizer: Any
    scheduler: Any


def _module(name: str) -> Any:
    try:
        return cast(Any, importlib.import_module(name))
    except ImportError as error:
        raise TrainingRecoverySmokeError(
            "training dependencies are missing; install the inference extra"
        ) from error


def _runtime_versions() -> dict[str, str]:
    names = ("peft", "pyarrow", "safetensors", "tokenizers", "torch", "transformers")
    try:
        return {name: importlib.metadata.version(name) for name in names}
    except importlib.metadata.PackageNotFoundError as error:
        raise TrainingRecoverySmokeError(f"missing runtime package: {error}") from error


def _verify_files(root: Path, expected: dict[str, str], label: str) -> None:
    if not root.is_dir():
        raise TrainingRecoverySmokeError(f"{label} directory is missing: {root}")
    for name, expected_sha256 in expected.items():
        path = root / name
        try:
            observed = file_sha256(path)
        except OSError as error:
            raise TrainingRecoverySmokeError(
                f"cannot hash {label} file {path}"
            ) from error
        if observed != expected_sha256:
            raise TrainingRecoverySmokeError(f"{label} file hash differs: {name}")


def _verify_base_snapshot(recipe: NamedTrainingRecipe, root: Path) -> None:
    _verify_files(root, dict(recipe.artifacts.base.files), "base")
    markers = [name for name in _ADAPTER_MARKERS if (root / name).exists()]
    if markers:
        raise TrainingRecoverySmokeError(
            f"base snapshot has adapter marker: {markers[0]}"
        )


def _fixture_bytes(recipe: NamedTrainingRecipe, path: Path) -> bytes:
    try:
        source = path.read_bytes()
    except OSError as error:
        raise TrainingRecoverySmokeError(
            f"cannot read smoke fixture: {path}"
        ) from error
    if bytes_sha256(source) != recipe.smoke_fixture.sha256:
        raise TrainingRecoverySmokeError("smoke fixture hash differs from recipe")
    return source


def _fixture_payload(source: bytes) -> dict[str, Any]:
    try:
        value = json_loads_unique(source.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise TrainingRecoverySmokeError("cannot parse smoke fixture") from error
    if not isinstance(value, dict):
        raise TrainingRecoverySmokeError("smoke fixture must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_keys(payload: dict[str, Any], expected: set[str], field: str) -> None:
    if set(payload) != expected:
        raise TrainingRecoverySmokeError(f"{field} fields differ")


def _messages(value: object, row_number: int) -> tuple[ConversationMessage, ...]:
    if not isinstance(value, list) or not value:
        raise TrainingRecoverySmokeError(f"row {row_number}: messages are invalid")
    messages: list[ConversationMessage] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            raise TrainingRecoverySmokeError(f"row {row_number}: message is invalid")
        payload = cast(dict[str, object], item)
        if set(payload) != {"role", "content"}:
            raise TrainingRecoverySmokeError(f"row {row_number}: message fields differ")
        role = payload.get("role")
        content = payload.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TrainingRecoverySmokeError(f"row {row_number}: message is invalid")
        messages.append(ConversationMessage(role, content))
    return tuple(messages)


def _validate_fixture_source(
    recipe: NamedTrainingRecipe, payload: dict[str, Any]
) -> None:
    source = payload.get("source_contract")
    if not isinstance(source, dict):
        raise TrainingRecoverySmokeError("smoke fixture source contract is invalid")
    observed = cast(dict[str, object], source)
    expected: dict[str, object] = {
        "source_sample_receipt_sha256": (
            recipe.smoke_fixture.source_sample_receipt_sha256
        ),
        "source_repository": recipe.artifacts.training_data.source_repository,
        "source_revision": recipe.artifacts.training_data.source_revision,
        "identity_scheme": recipe.artifacts.training_data.identity_scheme,
        "row_fields": ["orig_idx", "messages"],
        "message_fields": ["role", "content"],
    }
    if not json_equal_exact(observed, expected):
        raise TrainingRecoverySmokeError("smoke fixture source contract differs")


def _fixture_rows(
    recipe: NamedTrainingRecipe, fixture_path: Path
) -> tuple[TrainingRow, ...]:
    payload = _fixture_payload(_fixture_bytes(recipe, fixture_path))
    fields = {"schema_version", "fixture_id", "source_contract", "rows"}
    _exact_keys(payload, fields, "smoke fixture")
    identity = (payload.get("schema_version"), payload.get("fixture_id"))
    if identity != (1, recipe.smoke_fixture.fixture_id):
        raise TrainingRecoverySmokeError("smoke fixture identity differs")
    _validate_fixture_source(recipe, payload)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TrainingRecoverySmokeError("smoke fixture rows are invalid")
    items = cast(list[object], rows)
    if len(items) != recipe.smoke_fixture.row_count:
        raise TrainingRecoverySmokeError("smoke fixture row count differs")
    parsed: list[TrainingRow] = []
    for row_number, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            raise TrainingRecoverySmokeError(f"row {row_number}: row is invalid")
        row = cast(dict[str, object], raw)
        if set(row) != {"orig_idx", "messages"}:
            raise TrainingRecoverySmokeError(f"row {row_number}: fields differ")
        orig_idx = row.get("orig_idx")
        if type(orig_idx) is not int:
            raise TrainingRecoverySmokeError(f"row {row_number}: orig_idx is invalid")
        parsed.append(TrainingRow(orig_idx, _messages(row.get("messages"), row_number)))
    if len({row.orig_idx for row in parsed}) != len(parsed):
        raise TrainingRecoverySmokeError("smoke fixture orig_idx values are duplicated")
    return tuple(parsed)


def _tokenizer(
    transformers: Any, recipe: NamedTrainingRecipe, snapshot: Path
) -> TrainingTokenizer:
    path_value = recipe.artifacts.tokenizer.path
    if path_value is None:
        raise TrainingRecoverySmokeError("tokenizer artifact path is missing")
    root = snapshot / path_value
    _verify_files(root, dict(recipe.artifacts.tokenizer.files), "tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(root), local_files_only=True
    )
    template_path = root / recipe.values.sequence.chat_template_file
    template = template_path.read_text(encoding="utf-8")
    if tokenizer.chat_template != template:
        raise TrainingRecoverySmokeError("loaded tokenizer template differs")
    return cast(TrainingTokenizer, tokenizer)


def _lora_config(peft: Any, recipe: NamedTrainingRecipe) -> Any:
    lora = recipe.values.lora
    return peft.LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        task_type=lora.task_type,
        target_modules=list(lora.target_modules),
    )


def _validate_fresh_base(model: Any) -> None:
    has_transformers_adapter = bool(getattr(model, "_hf_peft_config_loaded", False))
    has_peft_config = getattr(model, "peft_config", None) is not None
    if has_transformers_adapter or has_peft_config:
        raise TrainingRecoverySmokeError("loaded base has a preexisting adapter")


def _model(
    torch: Any,
    transformers: Any,
    peft: Any,
    recipe: NamedTrainingRecipe,
    base_snapshot: Path,
) -> Any:
    torch.manual_seed(recipe.values.training.seed)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(base_snapshot),
        dtype=getattr(torch, recipe.smoke.model_dtype),
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    _validate_fresh_base(model)
    model.config.use_cache = recipe.values.training.use_cache
    if recipe.values.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    torch.manual_seed(recipe.values.training.seed)
    model = peft.get_peft_model(model, _lora_config(peft, recipe))
    torch.manual_seed(recipe.values.training.seed)
    return model.to(recipe.smoke.device).train()


def _optimizer(torch: Any, model: Any, recipe: NamedTrainingRecipe) -> Any:
    values = recipe.values.optimizer
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise TrainingRecoverySmokeError("LoRA model has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=values.learning_rate,
        betas=(values.beta1, values.beta2),
        eps=values.epsilon,
        weight_decay=values.weight_decay,
        fused=values.fused,
    )


def _components(
    torch: Any,
    transformers: Any,
    peft: Any,
    recipe: NamedTrainingRecipe,
    base_snapshot: Path,
) -> _Components:
    model = _model(torch, transformers, peft, recipe, base_snapshot)
    optimizer = _optimizer(torch, model, recipe)
    scheduler = transformers.get_scheduler(
        recipe.values.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=recipe.values.scheduler.warmup_steps,
        num_training_steps=recipe.smoke.total_optimizer_steps,
    )
    return _Components(model, optimizer, scheduler)


def _batch_tensors(torch: Any, batch: PackedBatch, device: str) -> tuple[Any, Any, Any]:
    input_ids = torch.tensor([batch.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([batch.labels], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [batch.attention_mask], dtype=torch.long, device=device
    )
    return input_ids, labels, attention_mask


def _train_step(
    torch: Any, components: _Components, batch: PackedBatch, recipe: NamedTrainingRecipe
) -> float:
    input_ids, labels, attention_mask = _batch_tensors(
        torch, batch, recipe.smoke.device
    )
    components.optimizer.zero_grad(set_to_none=True)
    output = components.model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    loss = output.loss
    if not bool(torch.isfinite(loss).item()):
        raise TrainingRecoverySmokeError("training loss is not finite")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        components.model.parameters(), recipe.values.optimizer.max_grad_norm
    )
    components.optimizer.step()
    components.scheduler.step()
    return float(loss.detach().float().item())


def _adapter_state(peft: Any, model: Any) -> dict[str, Any]:
    state = peft.get_peft_model_state_dict(model)
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in sorted(state.items())
    }


def _adapter_checkpoint_config(recipe: NamedTrainingRecipe) -> dict[str, object]:
    lora = recipe.values.lora
    return {
        "schema_version": 1,
        "recipe_id": recipe.recipe_id,
        "config_sha256": recipe.config_sha256,
        "recipe_core_sha256": recipe.recipe_core_sha256,
        "smoke_authority_sha256": recipe.smoke_authority_sha256,
        "rank": lora.rank,
        "alpha": lora.alpha,
        "dropout": lora.dropout,
        "bias": lora.bias,
        "task_type": lora.task_type,
        "target_modules": list(lora.target_modules),
    }


def _cursor_payload(
    recipe: NamedTrainingRecipe, data: PackedSmokeData
) -> dict[str, object]:
    step = recipe.smoke.checkpoint_after_optimizer_step
    return {
        "schema_version": 1,
        "recipe_core_sha256": recipe.recipe_core_sha256,
        "smoke_authority_sha256": recipe.smoke_authority_sha256,
        "next_batch_index": step,
        "fixture_sha256": data.fixture_sha256,
        "source_sample_receipt_sha256": data.source_sample_receipt_sha256,
        "token_stream_sha256": data.token_stream_sha256,
        "packed_batches_sha256": data.packed_batches_sha256,
        "next_batch_sha256": data.packed_batch_sha256[step],
    }


def _trainer_state_payload(recipe: NamedTrainingRecipe) -> dict[str, object]:
    return {
        "schema_version": 1,
        "config_sha256": recipe.config_sha256,
        "recipe_core_sha256": recipe.recipe_core_sha256,
        "smoke_authority_sha256": recipe.smoke_authority_sha256,
        "completed_optimizer_steps": recipe.smoke.checkpoint_after_optimizer_step,
        "total_optimizer_steps": recipe.smoke.total_optimizer_steps,
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")


def _create_exclusive_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise TrainingRecoverySmokeError(f"{label} already exists: {path}") from error
    except OSError as error:
        raise TrainingRecoverySmokeError(f"cannot create {label}: {path}") from error


def _save_checkpoint(
    torch: Any,
    safe: Any,
    peft: Any,
    recipe: NamedTrainingRecipe,
    data: PackedSmokeData,
    components: _Components,
    path: Path,
) -> CheckpointObservation:
    _create_exclusive_directory(path, "checkpoint directory")
    safe.save_file(
        _adapter_state(peft, components.model), path / "adapter_model.safetensors"
    )
    _write_json(path / "adapter_config.json", _adapter_checkpoint_config(recipe))
    torch.save(components.optimizer.state_dict(), path / "optimizer.pt")
    torch.save(components.scheduler.state_dict(), path / "scheduler.pt")
    torch.save({"cpu_rng_state": torch.get_rng_state()}, path / "rng.pt")
    _write_json(path / "trainer_state.json", _trainer_state_payload(recipe))
    _write_json(path / "cursor.json", _cursor_payload(recipe, data))
    hashes = {file.name: file_sha256(file) for file in sorted(path.iterdir())}
    step = recipe.smoke.checkpoint_after_optimizer_step
    return CheckpointObservation(step, step, hashes)


def _read_json(path: Path) -> object:
    try:
        return json_loads_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TrainingRecoverySmokeError(
            f"cannot load checkpoint file: {path}"
        ) from error


def _verify_checkpoint(
    recipe: NamedTrainingRecipe,
    data: PackedSmokeData,
    path: Path,
    checkpoint: CheckpointObservation,
) -> None:
    observed = {name: file_sha256(path / name) for name in checkpoint.file_sha256}
    if observed != dict(checkpoint.file_sha256):
        raise TrainingRecoverySmokeError("checkpoint file hashes differ")
    if not json_equal_exact(
        _read_json(path / "adapter_config.json"), _adapter_checkpoint_config(recipe)
    ):
        raise TrainingRecoverySmokeError("checkpoint adapter config differs")
    if not json_equal_exact(
        _read_json(path / "cursor.json"), _cursor_payload(recipe, data)
    ):
        raise TrainingRecoverySmokeError("checkpoint cursor differs")
    if not json_equal_exact(
        _read_json(path / "trainer_state.json"), _trainer_state_payload(recipe)
    ):
        raise TrainingRecoverySmokeError("checkpoint trainer state differs")


def _load_checkpoint(
    torch: Any,
    safe: Any,
    peft: Any,
    recipe: NamedTrainingRecipe,
    components: _Components,
    path: Path,
) -> None:
    adapter_state = safe.load_file(path / "adapter_model.safetensors", device="cpu")
    result = peft.set_peft_model_state_dict(components.model, adapter_state)
    if result.unexpected_keys:
        raise TrainingRecoverySmokeError("checkpoint adapter keys differ")
    optimizer_state = torch.load(
        path / "optimizer.pt", map_location="cpu", weights_only=True
    )
    scheduler_state = torch.load(
        path / "scheduler.pt", map_location="cpu", weights_only=True
    )
    rng_state = torch.load(path / "rng.pt", map_location="cpu", weights_only=True)
    components.optimizer.load_state_dict(optimizer_state)
    components.scheduler.load_state_dict(scheduler_state)
    if not isinstance(rng_state, dict) or "cpu_rng_state" not in rng_state:
        raise TrainingRecoverySmokeError("checkpoint RNG state differs")
    torch.set_rng_state(rng_state["cpu_rng_state"])


def _release(components: _Components) -> None:
    del components.model
    del components.optimizer
    del components.scheduler
    gc.collect()


def _uninterrupted_path(
    torch: Any,
    transformers: Any,
    peft: Any,
    recipe: NamedTrainingRecipe,
    base_snapshot: Path,
    data: PackedSmokeData,
) -> tuple[dict[str, Any], tuple[float, ...]]:
    LOGGER.info("running uninterrupted two-step training path")
    components = _components(torch, transformers, peft, recipe, base_snapshot)
    losses = tuple(
        _train_step(torch, components, data.batches[index], recipe)
        for index in range(recipe.smoke.total_optimizer_steps)
    )
    state = _adapter_state(peft, components.model)
    _release(components)
    return state, losses


def _recovery_path(
    modules: tuple[Any, Any, Any, Any],
    recipe: NamedTrainingRecipe,
    base_snapshot: Path,
    data: PackedSmokeData,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], tuple[float, ...], CheckpointObservation]:
    torch, transformers, peft, safe = modules
    LOGGER.info("running checkpointed first optimizer step")
    initial = _components(torch, transformers, peft, recipe, base_snapshot)
    first_loss = _train_step(torch, initial, data.batches[0], recipe)
    checkpoint = _save_checkpoint(
        torch, safe, peft, recipe, data, initial, checkpoint_dir
    )
    _release(initial)
    LOGGER.info("loading checkpoint into a fresh model for the second step")
    resumed = _components(torch, transformers, peft, recipe, base_snapshot)
    _verify_checkpoint(recipe, data, checkpoint_dir, checkpoint)
    _load_checkpoint(torch, safe, peft, recipe, resumed, checkpoint_dir)
    second_loss = _train_step(torch, resumed, data.batches[1], recipe)
    state = _adapter_state(peft, resumed.model)
    _release(resumed)
    return state, (first_loss, second_loss), checkpoint


def _state_sha256(safe: Any, state: dict[str, Any]) -> str:
    return bytes_sha256(safe.save(dict(sorted(state.items()))))


def _max_abs_difference(
    torch: Any, left: dict[str, Any], right: dict[str, Any]
) -> float:
    if set(left) != set(right):
        raise TrainingRecoverySmokeError("terminal adapter tensor keys differ")
    maximum = 0.0
    for name in sorted(left):
        if left[name].shape != right[name].shape:
            raise TrainingRecoverySmokeError(f"terminal adapter shape differs: {name}")
        left_tensor = left[name].float()
        right_tensor = right[name].float()
        if not bool(torch.isfinite(left_tensor).all().item()) or not bool(
            torch.isfinite(right_tensor).all().item()
        ):
            raise TrainingRecoverySmokeError("terminal adapter tensors must be finite")
        difference = torch.max(torch.abs(left_tensor - right_tensor))
        maximum = max(maximum, float(difference.item()))
    return maximum


def _prepare_data(
    recipe: NamedTrainingRecipe, tokenizer: TrainingTokenizer, fixture_path: Path
) -> PackedSmokeData:
    rows = _fixture_rows(recipe, fixture_path)
    return build_packed_smoke_data(
        rows,
        tokenizer,
        source_revision=recipe.artifacts.training_data.source_revision,
        seed=recipe.values.training.seed,
        sample_limit=recipe.smoke.sample_limit,
        sequence_length=recipe.smoke.overrides.max_sequence_length,
        overlength_limit=recipe.smoke.overrides.max_sequence_length,
        fixture_sha256=recipe.smoke_fixture.sha256,
        source_sample_receipt_sha256=(
            recipe.smoke_fixture.source_sample_receipt_sha256
        ),
    )


def _validate_sequence_contract(recipe: NamedTrainingRecipe) -> None:
    sequence = recipe.values.sequence
    expected: dict[str, object] = {
        "overlength_policy": "drop",
        "truncation": False,
        "packing": "stable_concatenate",
        "remainder": "pad_and_mask",
        "add_special_tokens": False,
        "loss_mask": "assistant_tokens_including_terminal_eos",
    }
    for field, value in expected.items():
        if getattr(sequence, field) != value:
            raise TrainingRecoverySmokeError(f"unsupported sequence.{field}")


def _validate_order_checkpoint_contract(recipe: NamedTrainingRecipe) -> None:
    order = recipe.values.order
    if order.algorithm != "sha256_seeded_sample_id_v1":
        raise TrainingRecoverySmokeError("unsupported order.algorithm")
    if order.identity_scheme != "source-revision-orig-idx-v1":
        raise TrainingRecoverySmokeError("unsupported order.identity_scheme")
    checkpoint = recipe.values.checkpoint
    expected: dict[str, object] = {
        "boundary": "completed_optimizer_step",
        "save_adapter": True,
        "save_optimizer": True,
        "save_scheduler": True,
        "save_rng": True,
        "save_data_cursor": True,
        "resume_semantics": "exact_recipe_inputs_runtime_and_cursor_v1",
    }
    for field, value in expected.items():
        if getattr(checkpoint, field) != value:
            raise TrainingRecoverySmokeError(f"unsupported checkpoint.{field}")


def _validate_smoke_overrides(recipe: NamedTrainingRecipe) -> None:
    smoke = recipe.smoke
    overrides = smoke.overrides
    expected: dict[str, object] = {
        "micro_batch_size_per_device": 1,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 1,
        "precision": "bfloat16",
        "checkpoint_interval_optimizer_steps": 1,
    }
    for field, value in expected.items():
        if getattr(overrides, field) != value:
            raise TrainingRecoverySmokeError(f"unsupported smoke.overrides.{field}")
    if smoke.total_optimizer_steps != 2 or smoke.checkpoint_after_optimizer_step != 1:
        raise TrainingRecoverySmokeError("unsupported smoke step contract")
    if smoke.comparison.metric != "adapter_tensor_max_abs":
        raise TrainingRecoverySmokeError("unsupported smoke comparison metric")
    if smoke.comparison.relative_tolerance != 0.0:
        raise TrainingRecoverySmokeError("unsupported smoke relative tolerance")


def _validate_backend_contract(recipe: NamedTrainingRecipe) -> None:
    if recipe.smoke.device != "cpu" or recipe.smoke.model_dtype != "bfloat16":
        raise TrainingRecoverySmokeError("transformers smoke requires CPU bfloat16")
    _validate_sequence_contract(recipe)
    _validate_order_checkpoint_contract(recipe)
    _validate_smoke_overrides(recipe)
    torch = _module("torch")
    torch.use_deterministic_algorithms(True)


def _observation(
    modules: tuple[Any, Any, Any, Any],
    recipe: NamedTrainingRecipe,
    data: PackedSmokeData,
    checkpoint: CheckpointObservation,
    paths: tuple[dict[str, Any], dict[str, Any]],
    losses: tuple[tuple[float, ...], tuple[float, ...]],
    versions: dict[str, str],
) -> TrainingRecoveryObservation:
    uninterrupted, resumed = paths
    return TrainingRecoveryObservation(
        data,
        recipe.smoke.total_optimizer_steps,
        checkpoint,
        _state_sha256(modules[3], uninterrupted),
        _state_sha256(modules[3], resumed),
        _max_abs_difference(modules[0], uninterrupted, resumed),
        *losses,
        recipe.smoke.device,
        recipe.smoke.model_dtype,
        versions,
    )


def _execute_paths(
    modules: tuple[Any, Any, Any, Any],
    recipe: NamedTrainingRecipe,
    base_snapshot: Path,
    data: PackedSmokeData,
    work_dir: Path,
    versions: dict[str, str],
) -> TrainingRecoveryObservation:
    uninterrupted, uninterrupted_losses = _uninterrupted_path(
        *modules[:3], recipe, base_snapshot, data
    )
    resumed, resumed_losses, checkpoint = _recovery_path(
        modules, recipe, base_snapshot, data, work_dir / "checkpoint-step-1"
    )
    return _observation(
        modules,
        recipe,
        data,
        checkpoint,
        (uninterrupted, resumed),
        (uninterrupted_losses, resumed_losses),
        versions,
    )


def run_transformers_training_recovery(
    recipe: NamedTrainingRecipe,
    *,
    base_snapshot: Path,
    tokenizer_snapshot: Path,
    fixture_path: Path,
    work_dir: Path,
) -> TrainingRecoveryObservation:
    """Run fresh uninterrupted and checkpoint-resumed LoRA smoke paths."""
    _validate_backend_contract(recipe)
    versions = _runtime_versions()
    if versions != dict(recipe.runtime_versions):
        raise TrainingRecoverySmokeError(
            "installed runtime versions differ from recipe"
        )
    _verify_base_snapshot(recipe, base_snapshot)
    modules = (
        _module("torch"),
        _module("transformers"),
        _module("peft"),
        _module("safetensors.torch"),
    )
    tokenizer = _tokenizer(modules[1], recipe, tokenizer_snapshot)
    data = _prepare_data(recipe, tokenizer, fixture_path)
    _create_exclusive_directory(work_dir, "work directory")
    return _execute_paths(modules, recipe, base_snapshot, data, work_dir, versions)
