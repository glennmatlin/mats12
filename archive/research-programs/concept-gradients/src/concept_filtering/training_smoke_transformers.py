"""Tiny OLMo-3 LoRA executor for the bounded training smoke."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.chat_rendering import ChatRenderingError, render_assistant_only
from concept_filtering.serialization import (
    canonical_sha256,
    file_sha256,
    json_loads_unique,
)
from concept_filtering.training_recipe import TrainingRecipe
from concept_filtering.training_smoke import (
    TrainingSmokeError,
    TrainingSmokeObservation,
)

_SPECIAL_TOKENS = (
    "<unk>",
    "<bos>",
    "<eos>",
    "<pad>",
    "<|im_start|>",
    "<|im_end|>",
)


@dataclass(frozen=True)
class PackedBatch:
    input_ids: Any
    labels: Any
    example_ids: tuple[str, ...]
    input_ids_sha256: str
    labels_sha256: str


@dataclass(frozen=True)
class PackedDataset:
    batches: tuple[PackedBatch, PackedBatch]
    example_count: int
    retained_example_count: int
    dropped_overlength_count: int
    sample_order_sha256: str
    input_ids_sha256: str
    labels_sha256: str


@dataclass(frozen=True)
class SmokeFixture:
    chat_template: str
    examples: tuple[dict[str, object], ...]
    overlength_probe: dict[str, object]


@dataclass(frozen=True)
class TrainComponents:
    model: Any
    optimizer: Any
    scheduler: Any


@dataclass(frozen=True)
class SmokeResult:
    dataset: PackedDataset
    checkpoint: Path
    losses: tuple[float, float]
    resumed_loss: float
    continuous_state: dict[str, Any]
    resumed_state: dict[str, Any]


def _module(name: str) -> Any:
    try:
        return cast(Any, importlib.import_module(name))
    except ImportError as error:
        raise TrainingSmokeError(
            "training-smoke dependencies are missing; install the inference extra"
        ) from error


def _runtime_versions() -> dict[str, str]:
    names = ("huggingface-hub", "peft", "tokenizers", "torch", "transformers")
    try:
        return {
            name.replace("-", "_"): importlib.metadata.version(name) for name in names
        }
    except importlib.metadata.PackageNotFoundError as error:
        raise TrainingSmokeError(f"missing runtime package: {error}") from error


def _fixture_words(fixture: SmokeFixture) -> tuple[str, ...]:
    words: set[str] = set(_SPECIAL_TOKENS)
    for example in fixture.examples:
        messages = cast(list[dict[str, str]], example["messages"])
        for message in messages:
            words.update(message["content"].split())
    words.add(cast(str, fixture.overlength_probe["repeat_token"]))
    return tuple(sorted(words))


def _tokenizer(tokenizers: Any, transformers: Any, fixture: SmokeFixture) -> Any:
    vocabulary = {token: index for index, token in enumerate(_fixture_words(fixture))}
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocabulary, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        additional_special_tokens=list(_SPECIAL_TOKENS[4:]),
    )
    tokenizer.chat_template = fixture.chat_template
    return tokenizer


def _fixture_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TrainingSmokeError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _fixture_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingSmokeError(f"{label} must be a non-empty string")
    return value


def _validate_fixture_example(example: dict[str, object]) -> str:
    if set(example) != {"fixture_id", "messages"}:
        raise TrainingSmokeError("smoke fixture example fields differ")
    fixture_id = _fixture_string(example["fixture_id"], "fixture ID")
    messages = example["messages"]
    if not isinstance(messages, list):
        raise TrainingSmokeError("smoke fixture messages must alternate turns")
    turns = cast(list[object], messages)
    if len(turns) < 2 or len(turns) % 2:
        raise TrainingSmokeError("smoke fixture messages must alternate turns")
    roles = ("user", "assistant") * (len(turns) // 2)
    for raw, role in zip(turns, roles, strict=True):
        message = _fixture_object(raw, "fixture message")
        if set(message) != {"role", "content"} or message["role"] != role:
            raise TrainingSmokeError("smoke fixture message roles differ")
        if not isinstance(message["content"], str):
            raise TrainingSmokeError("fixture message content must be a string")
        if role == "assistant" and not message["content"]:
            raise TrainingSmokeError("fixture assistant content cannot be empty")
    return fixture_id


def _fixture_examples(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise TrainingSmokeError("smoke fixture must contain four examples")
    items = cast(list[object], value)
    if len(items) != 4:
        raise TrainingSmokeError("smoke fixture must contain four examples")
    examples = tuple(_fixture_object(item, "fixture example") for item in items)
    fixture_ids = [_validate_fixture_example(example) for example in examples]
    if len(set(fixture_ids)) != len(fixture_ids):
        raise TrainingSmokeError("smoke fixture IDs must be unique")
    _validate_fixture_shapes(examples)
    return examples


def _validate_fixture_shapes(examples: tuple[dict[str, object], ...]) -> None:
    messages = [cast(list[dict[str, str]], item["messages"]) for item in examples]
    multi_turn = sum(len(item) > 2 for item in messages)
    empty_user = sum(
        any(message["role"] == "user" and not message["content"] for message in item)
        for item in messages
    )
    if (multi_turn, empty_user) != (1, 1):
        raise TrainingSmokeError(
            "smoke fixture must contain one multi-turn and one empty-user example"
        )


def _validate_overlength_probe(
    value: object, fixture_ids: set[str]
) -> dict[str, object]:
    probe = _fixture_object(value, "overlength probe")
    if set(probe) != {"fixture_id", "repeat_token", "repeat_count"}:
        raise TrainingSmokeError("overlength probe fields differ")
    fixture_id = _fixture_string(probe["fixture_id"], "overlength fixture ID")
    _fixture_string(probe["repeat_token"], "overlength repeat token")
    if probe["repeat_count"] != 8193 or type(probe["repeat_count"]) is not int:
        raise TrainingSmokeError("overlength repeat count must be 8193")
    if fixture_id in fixture_ids:
        raise TrainingSmokeError("smoke fixture IDs must be unique")
    return probe


def _validate_fixture_identity(payload: dict[str, object]) -> None:
    if payload["schema_version"] != 1 or type(payload["schema_version"]) is not int:
        raise TrainingSmokeError("smoke fixture schema version must be 1")
    if payload["fixture_id"] != "public-rank32-training-smoke-v1":
        raise TrainingSmokeError("smoke fixture identity differs")


def _read_fixture_payload(recipe: TrainingRecipe) -> object:
    try:
        raw_bytes = recipe.smoke_fixture_path.read_bytes()
    except (OSError, UnicodeError, ValueError) as error:
        raise TrainingSmokeError(f"cannot load smoke fixture: {error}") from error
    if hashlib.sha256(raw_bytes).hexdigest() != recipe.smoke_fixture_sha256:
        raise TrainingSmokeError("smoke fixture file hash differs")
    try:
        return json_loads_unique(raw_bytes.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise TrainingSmokeError(f"cannot load smoke fixture: {error}") from error


def _load_fixture(recipe: TrainingRecipe) -> SmokeFixture:
    raw = _read_fixture_payload(recipe)
    payload = _fixture_object(raw, "smoke fixture")
    expected = {
        "schema_version",
        "fixture_id",
        "source_contract",
        "chat_template",
        "examples",
        "overlength_probe",
    }
    if set(payload) != expected:
        raise TrainingSmokeError("smoke fixture fields differ from the contract")
    _validate_fixture_identity(payload)
    _validate_fixture_source(recipe, payload)
    template = _fixture_string(payload["chat_template"], "smoke fixture template")
    if hashlib.sha256(template.encode()).hexdigest() != recipe.chat_template_sha256:
        raise TrainingSmokeError("smoke fixture chat template hash differs")
    examples = _fixture_examples(payload["examples"])
    fixture_ids = {cast(str, example["fixture_id"]) for example in examples}
    probe = _validate_overlength_probe(payload["overlength_probe"], fixture_ids)
    return SmokeFixture(template, examples, probe)


def _validate_fixture_source(
    recipe: TrainingRecipe, payload: dict[str, object]
) -> None:
    source = _fixture_object(payload["source_contract"], "source contract")
    expected = {
        "training_sample_receipt_sha256": recipe.training_sample_receipt_sha256,
        "identity_scheme": recipe.training_sample_identity_scheme,
        "chat_template_sha256": recipe.chat_template_sha256,
    }
    if source != expected:
        raise TrainingSmokeError("smoke fixture source contract differs")


def _tokenize_example(
    tokenizer: Any, example: dict[str, object]
) -> tuple[list[int], list[int]]:
    messages = cast(list[dict[str, str]], example["messages"])
    if messages[-1]["role"] != "assistant":
        raise TrainingSmokeError("training smoke example must end with assistant")
    try:
        rendered = render_assistant_only(tokenizer, messages, tokenizer.chat_template)
    except ChatRenderingError as error:
        raise TrainingSmokeError(
            f"cannot render assistant-only mask: {error}"
        ) from error
    input_ids = list(rendered.input_ids)
    labels = list(rendered.labels)
    if tokenizer.eos_token_id is None:
        raise TrainingSmokeError("training smoke tokenizer must define EOS")
    eos_id = cast(int, tokenizer.eos_token_id)
    if not input_ids or input_ids[-1] != eos_id:
        input_ids.append(eos_id)
        labels.append(eos_id)
    return input_ids, labels


def _pack(
    torch: Any,
    examples: list[dict[str, object]],
    tokenized: list[tuple[list[int], list[int]]],
) -> PackedBatch:
    input_ids = [token for value, _ in tokenized for token in value]
    labels = [token for _, value in tokenized for token in value]
    order = tuple(cast(str, example["fixture_id"]) for example in examples)
    masked = any(value == -100 for value in labels)
    supervised = any(value >= 0 for value in labels)
    if not masked or not supervised:
        raise TrainingSmokeError("assistant-only loss mask is degenerate")
    return PackedBatch(
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
        order,
        canonical_sha256(input_ids),
        canonical_sha256(labels),
    )


def _overlength_example(probe: dict[str, object]) -> dict[str, object]:
    token = cast(str, probe["repeat_token"])
    count = cast(int, probe["repeat_count"])
    return {
        "fixture_id": probe["fixture_id"],
        "messages": [
            {"role": "user", "content": " ".join([token] * count)},
            {"role": "assistant", "content": "Hello"},
        ],
    }


def _ordered_tokens(
    tokenizer: Any, recipe: TrainingRecipe, fixture: SmokeFixture
) -> list[tuple[dict[str, object], tuple[list[int], list[int]]]]:
    examples = [*fixture.examples, _overlength_example(fixture.overlength_probe)]
    random.Random(recipe.data_seed).shuffle(examples)
    return [(example, _tokenize_example(tokenizer, example)) for example in examples]


def _retained_tokens(
    tokenized: list[tuple[dict[str, object], tuple[list[int], list[int]]]],
    recipe: TrainingRecipe,
) -> list[tuple[dict[str, object], tuple[list[int], list[int]]]]:
    retained = [
        (example, tokens)
        for example, tokens in tokenized
        if len(tokens[0]) <= recipe.max_sequence_length
    ]
    dropped = len(tokenized) - len(retained)
    if dropped != 1 or len(retained) != 4:
        raise TrainingSmokeError("overlength-drop smoke contract was not exercised")
    return retained


def _packed_batches(
    torch: Any,
    retained: list[tuple[dict[str, object], tuple[list[int], list[int]]]],
    recipe: TrainingRecipe,
) -> tuple[PackedBatch, PackedBatch]:
    size = cast(int, recipe.smoke_overrides["examples_per_packed_sequence"])
    batches = tuple(
        _pack(
            torch,
            [item[0] for item in retained[index : index + size]],
            [item[1] for item in retained[index : index + size]],
        )
        for index in range(0, len(retained), size)
    )
    if len(batches) != 2 or batches[0].input_ids_sha256 == batches[1].input_ids_sha256:
        raise TrainingSmokeError("smoke must produce two distinct packed batches")
    return batches


def _packed_dataset(
    torch: Any, tokenizer: Any, recipe: TrainingRecipe, fixture: SmokeFixture
) -> PackedDataset:
    tokenized = _ordered_tokens(tokenizer, recipe, fixture)
    retained = _retained_tokens(tokenized, recipe)
    batches = _packed_batches(torch, retained, recipe)
    order = [
        {
            "fixture_id": cast(str, item[0]["fixture_id"]),
            "retained": len(item[1][0]) <= recipe.max_sequence_length,
        }
        for item in tokenized
    ]
    return PackedDataset(
        batches,
        5,
        4,
        1,
        canonical_sha256(order),
        canonical_sha256([batch.input_ids_sha256 for batch in batches]),
        canonical_sha256([batch.labels_sha256 for batch in batches]),
    )


def _base_model(torch: Any, transformers: Any, tokenizer: Any, seed: int) -> Any:
    torch.manual_seed(seed)
    config = transformers.Olmo3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        sliding_window=64,
        layer_types=["full_attention", "full_attention"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    return transformers.Olmo3ForCausalLM(config)


def _lora_config(peft: Any, recipe: TrainingRecipe) -> Any:
    return peft.LoraConfig(
        r=recipe.rank,
        lora_alpha=recipe.alpha,
        lora_dropout=recipe.dropout,
        bias="none",
        target_modules=list(recipe.target_modules),
        task_type=peft.TaskType.CAUSAL_LM,
    )


def _optimizer(torch: Any, model: Any, recipe: TrainingRecipe) -> Any:
    parameters = [value for value in model.parameters() if value.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=recipe.learning_rate,
        betas=recipe.betas,
        eps=recipe.epsilon,
        weight_decay=recipe.weight_decay,
    )


def _components(
    torch: Any, transformers: Any, peft: Any, tokenizer: Any, recipe: TrainingRecipe
) -> TrainComponents:
    base = _base_model(torch, transformers, tokenizer, recipe.seed)
    model = peft.get_peft_model(base, _lora_config(peft, recipe))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    optimizer = _optimizer(torch, model, recipe)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=recipe.warmup_steps,
        num_training_steps=recipe.warmup_steps + 1,
    )
    torch.manual_seed(recipe.seed)
    return TrainComponents(model.train(), optimizer, scheduler)


def _train_step(
    torch: Any,
    components: TrainComponents,
    batch: PackedBatch,
    max_grad_norm: float,
) -> float:
    output = components.model(
        input_ids=batch.input_ids,
        labels=batch.labels,
        use_cache=False,
    )
    loss = output.loss
    if loss is None:
        raise TrainingSmokeError("training smoke model did not return loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(components.model.parameters(), max_grad_norm)
    components.optimizer.step()
    components.scheduler.step()
    components.optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu().item())


def _save_checkpoint(
    torch: Any,
    components: TrainComponents,
    recipe: TrainingRecipe,
    dataset: PackedDataset,
    checkpoint: Path,
) -> None:
    components.model.save_pretrained(
        checkpoint,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    _canonicalize_adapter_config(checkpoint, recipe)
    torch.save(components.optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(components.scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save(torch.get_rng_state(), checkpoint / "rng.pt")
    state = {
        "schema_version": 1,
        "global_step": 1,
        "next_batch_index": 1,
        "recipe_sha256": recipe.recipe_sha256,
        "sample_order_sha256": dataset.sample_order_sha256,
        "input_ids_sha256": dataset.input_ids_sha256,
        "labels_sha256": dataset.labels_sha256,
    }
    with (checkpoint / "state.json").open("x", encoding="utf-8") as target:
        target.write(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _canonicalize_adapter_config(checkpoint: Path, recipe: TrainingRecipe) -> None:
    path = checkpoint / "adapter_config.json"
    try:
        payload = json_loads_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TrainingSmokeError(f"cannot load adapter config: {error}") from error
    if not isinstance(payload, dict):
        raise TrainingSmokeError("adapter config must be an object")
    config = cast(dict[str, object], payload)
    modules = config.get("target_modules")
    expected = sorted(recipe.target_modules)
    if not isinstance(modules, list):
        raise TrainingSmokeError("saved adapter target modules differ")
    items = cast(list[object], modules)
    if not all(isinstance(module, str) for module in items):
        raise TrainingSmokeError("saved adapter target modules differ")
    if sorted(cast(list[str], items)) != expected:
        raise TrainingSmokeError("saved adapter target modules differ")
    config["target_modules"] = expected
    try:
        contents = json.dumps(config, indent=2, sort_keys=True) + "\n"
        path.write_text(contents, encoding="utf-8")
    except OSError as error:
        raise TrainingSmokeError(
            f"cannot canonicalize adapter config: {error}"
        ) from error


def _validate_checkpoint_state(
    recipe: TrainingRecipe, dataset: PackedDataset, path: Path
) -> int:
    try:
        state: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingSmokeError(f"cannot load checkpoint state: {error}") from error
    expected = {
        "schema_version": 1,
        "global_step": 1,
        "next_batch_index": 1,
        "recipe_sha256": recipe.recipe_sha256,
        "sample_order_sha256": dataset.sample_order_sha256,
        "input_ids_sha256": dataset.input_ids_sha256,
        "labels_sha256": dataset.labels_sha256,
    }
    if state != expected:
        raise TrainingSmokeError("checkpoint state does not match the resumed run")
    return 1


def _resume_components(
    torch: Any,
    transformers: Any,
    peft: Any,
    tokenizer: Any,
    recipe: TrainingRecipe,
    checkpoint: Path,
) -> TrainComponents:
    base = _base_model(torch, transformers, tokenizer, recipe.seed)
    model = peft.PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    optimizer = _optimizer(torch, model, recipe)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=recipe.warmup_steps,
        num_training_steps=recipe.warmup_steps + 1,
    )
    scheduler_state = torch.load(checkpoint / "scheduler.pt", weights_only=True)
    scheduler.load_state_dict(scheduler_state)
    optimizer_state = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(torch.load(checkpoint / "rng.pt", weights_only=True))
    return TrainComponents(model.train(), optimizer, scheduler)


def _adapter_state(peft: Any, model: Any) -> dict[str, Any]:
    state = peft.get_peft_model_state_dict(model, save_embedding_layers=False)
    return {name: value.detach().cpu().contiguous() for name, value in state.items()}


def _state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _max_difference(torch: Any, left: dict[str, Any], right: dict[str, Any]) -> float:
    if set(left) != set(right):
        raise TrainingSmokeError("adapter state roles differ after resume")
    values = [
        float(torch.max(torch.abs(left[name] - right[name])).item()) for name in left
    ]
    return max(values, default=0.0)


def _checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng.pt",
        "scheduler.pt",
        "state.json",
    )
    try:
        return {name: file_sha256(checkpoint / name) for name in names}
    except OSError as error:
        raise TrainingSmokeError(f"cannot hash checkpoint files: {error}") from error


def _run_paths(
    torch: Any,
    transformers: Any,
    peft: Any,
    tokenizer: Any,
    recipe: TrainingRecipe,
    dataset: PackedDataset,
    checkpoint: Path,
) -> tuple[tuple[float, float], float, dict[str, Any], dict[str, Any]]:
    continuous = _components(torch, transformers, peft, tokenizer, recipe)
    continuous_losses = (
        _train_step(torch, continuous, dataset.batches[0], recipe.max_grad_norm),
        _train_step(torch, continuous, dataset.batches[1], recipe.max_grad_norm),
    )
    interrupted = _components(torch, transformers, peft, tokenizer, recipe)
    _train_step(torch, interrupted, dataset.batches[0], recipe.max_grad_norm)
    _save_checkpoint(torch, interrupted, recipe, dataset, checkpoint)
    resumed = _resume_components(
        torch, transformers, peft, tokenizer, recipe, checkpoint
    )
    cursor = _validate_checkpoint_state(recipe, dataset, checkpoint / "state.json")
    resumed_loss = _train_step(
        torch, resumed, dataset.batches[cursor], recipe.max_grad_norm
    )
    return (
        continuous_losses,
        resumed_loss,
        _adapter_state(peft, continuous.model),
        _adapter_state(peft, resumed.model),
    )


def _execute_smoke(recipe: TrainingRecipe, run_dir: Path) -> tuple[Any, SmokeResult]:
    torch = _module("torch")
    transformers = _module("transformers")
    peft = _module("peft")
    tokenizers = _module("tokenizers")
    fixture = _load_fixture(recipe)
    tokenizer = _tokenizer(tokenizers, transformers, fixture)
    dataset = _packed_dataset(torch, tokenizer, recipe, fixture)
    _create_run_dir(run_dir)
    checkpoint = run_dir / "checkpoint-1"
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        losses, resumed_loss, continuous_state, resumed_state = _run_paths(
            torch, transformers, peft, tokenizer, recipe, dataset, checkpoint
        )
    finally:
        torch.use_deterministic_algorithms(previous)
    return torch, SmokeResult(
        dataset,
        checkpoint,
        losses,
        resumed_loss,
        continuous_state,
        resumed_state,
    )


def _create_run_dir(run_dir: Path) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise TrainingSmokeError(
            f"refusing to reuse run directory: {run_dir}"
        ) from error


def _dataset_observation(result: SmokeResult) -> dict[str, Any]:
    return {
        "example_count": result.dataset.example_count,
        "retained_example_count": result.dataset.retained_example_count,
        "dropped_overlength_count": result.dataset.dropped_overlength_count,
        "packed_sequence_count": len(result.dataset.batches),
        "sample_order_sha256": result.dataset.sample_order_sha256,
        "input_ids_sha256": result.dataset.input_ids_sha256,
        "labels_sha256": result.dataset.labels_sha256,
        "batch_input_ids_sha256": tuple(
            batch.input_ids_sha256 for batch in result.dataset.batches
        ),
        "batch_labels_sha256": tuple(
            batch.labels_sha256 for batch in result.dataset.batches
        ),
    }


def _observation(
    torch: Any, result: SmokeResult, recipe: TrainingRecipe
) -> TrainingSmokeObservation:
    values = _dataset_observation(result)
    return TrainingSmokeObservation(
        **values,
        runtime_versions=_runtime_versions(),
        smoke_fixture_sha256=recipe.smoke_fixture_sha256,
        chat_template_sha256=recipe.chat_template_sha256,
        trainable_parameter_count=sum(
            value.numel() for value in result.continuous_state.values()
        ),
        optimizer_step_count=2,
        checkpoint_step=1,
        next_batch_index=1,
        uninterrupted_losses=result.losses,
        resumed_losses=(result.resumed_loss,),
        uninterrupted_state_sha256=_state_sha256(result.continuous_state),
        resumed_state_sha256=_state_sha256(result.resumed_state),
        max_abs_parameter_difference=_max_difference(
            torch, result.continuous_state, result.resumed_state
        ),
        checkpoint_files=_checkpoint_hashes(result.checkpoint),
    )


def run_transformers_training_smoke(
    recipe: TrainingRecipe, run_dir: Path
) -> TrainingSmokeObservation:
    """Run a two-step tiny OLMo-3 LoRA train and exact checkpoint resume."""
    torch, result = _execute_smoke(recipe, run_dir)
    return _observation(torch, result, recipe)
