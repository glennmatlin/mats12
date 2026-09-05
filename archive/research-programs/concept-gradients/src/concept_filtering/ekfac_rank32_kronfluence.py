"""Kronfluence executor for the bounded real rank-32 EKFAC smoke."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import gc
import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as functional
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from peft import PeftModel
from safetensors import safe_open
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM

from concept_filtering.ekfac_rank32_smoke import (
    Rank32EKFACObservation,
    Rank32EKFACSmokeError,
    Rank32EKFACSmokePlan,
    Rank32RuntimePaths,
)
from concept_filtering.generation_evidence import (
    GenerationEvidenceError,
    load_generation_evidence,
)
from concept_filtering.prompt_bank import (
    InspectArchiveValidationError,
    PromptRecord,
    load_prompt_bank,
)
from concept_filtering.serialization import (
    bytes_sha256,
    canonical_sha256,
    json_loads_unique,
)
from concept_filtering.training_data_package import (
    BoundTrainingExample,
    TrainingDataPackageError,
    load_bound_tokenizer,
    load_bound_training_example,
    load_training_data_package_config,
)

TensorBatch = dict[str, torch.Tensor]
_MLP_PROJECTIONS = {"down_proj", "gate_proj", "up_proj"}
_PROJECTION_SHAPES = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 4096),
    "v_proj": (4096, 4096),
    "o_proj": (4096, 4096),
    "gate_proj": (11008, 4096),
    "up_proj": (11008, 4096),
    "down_proj": (4096, 11008),
}
_PROJECTIONS = (
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)


@dataclass(frozen=True)
class _PreparedInputs:
    training_input_ids: tuple[int, ...]
    training_labels: tuple[int, ...]
    query_input_ids: tuple[int, ...]
    query_labels: tuple[int, ...]


class _SingleExampleDataset(Dataset[TensorBatch]):
    def __init__(self, input_ids: tuple[int, ...], labels: tuple[int, ...]) -> None:
        self._input_ids = torch.tensor(input_ids, dtype=torch.long)
        self._labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> TensorBatch:
        if index != 0:
            raise IndexError(index)
        return {
            "input_ids": self._input_ids,
            "labels": self._labels,
            "attention_mask": torch.ones_like(self._input_ids),
        }


class _Rank32CausalTask(Task):
    def __init__(self, tracked_modules: tuple[str, ...]) -> None:
        self._tracked_modules = list(tracked_modules)

    def compute_train_loss(
        self, batch: TensorBatch, model: nn.Module, sample: bool = False
    ) -> torch.Tensor:
        return _causal_loss(batch, model, sample=sample)

    def compute_measurement(self, batch: TensorBatch, model: nn.Module) -> torch.Tensor:
        return _causal_loss(batch, model, sample=False)

    def get_influence_tracked_modules(self) -> list[str]:
        return list(self._tracked_modules)

    def get_attention_mask(self, batch: TensorBatch) -> torch.Tensor:
        return batch["attention_mask"]


def _projection_location(projection: str) -> str:
    return "mlp" if projection in _MLP_PROJECTIONS else "self_attn"


def expected_adapter_weight_keys() -> tuple[str, ...]:
    """Return the exact 448 public rank-32 adapter tensor names."""
    return tuple(
        "base_model.model.model.layers."
        f"{layer}.{_projection_location(projection)}.{projection}."
        f"lora_{factor}.weight"
        for layer in range(32)
        for projection in _PROJECTIONS
        for factor in ("A", "B")
    )


def _expected_adapter_weight_shapes() -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for key in expected_adapter_weight_keys():
        projection = next(name for name in _PROJECTIONS if f".{name}." in key)
        output_size, input_size = _PROJECTION_SHAPES[projection]
        result[key] = (32, input_size) if ".lora_A." in key else (output_size, 32)
    return result


def _loaded_module_name(weight_key: str) -> str:
    stem = weight_key.removesuffix(".weight")
    prefix, factor = stem.rsplit(".lora_", maxsplit=1)
    return f"{prefix}.lora_{factor}.default"


def _expected_loaded_shapes() -> dict[str, tuple[int, int]]:
    return {
        _loaded_module_name(key): shape
        for key, shape in _expected_adapter_weight_shapes().items()
    }


def _validate_adapter_weights(adapter_snapshot: Path) -> None:
    path = adapter_snapshot / "adapter_model.safetensors"
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
            shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
    except (OSError, ValueError) as error:
        raise Rank32EKFACSmokeError("cannot load adapter tensor manifest") from error
    expected = expected_adapter_weight_keys()
    if len(keys) != len(expected) or set(keys) != set(expected):
        raise Rank32EKFACSmokeError("adapter tensor topology differs from 448 matrices")
    if shapes != _expected_adapter_weight_shapes():
        raise Rank32EKFACSmokeError("adapter tensor shapes differ from rank-32 model")


def _load_model(plan: Rank32EKFACSmokePlan, paths: Rank32RuntimePaths) -> nn.Module:
    dtype = getattr(torch, cast(str, plan.runtime["model_dtype"]))
    base = AutoModelForCausalLM.from_pretrained(
        paths.base_snapshot,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    loaded = cast(
        Any,
        PeftModel.from_pretrained(
            base,
            paths.adapter_snapshot,
            is_trainable=True,
            local_files_only=True,
        ),
    )
    loaded.config.use_cache = False
    return cast(nn.Module, loaded.eval())


def _validate_loaded_topology(plan: Rank32EKFACSmokePlan, model: nn.Module) -> None:
    expected = _expected_loaded_shapes()
    observed = {
        name: tuple(module.weight.shape)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and (".lora_A." in name or ".lora_B." in name)
    }
    if observed != expected or len(observed) != plan.expected_adapter_linear_modules:
        raise Rank32EKFACSmokeError("loaded adapter topology differs from 448 matrices")
    if any(
        not isinstance(model.get_submodule(name), nn.Linear)
        for name in plan.tracked_modules
    ):
        raise Rank32EKFACSmokeError("bounded tracked modules are not linear")


def _model_logits(model: nn.Module, batch: TensorBatch) -> torch.Tensor:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise Rank32EKFACSmokeError("rank-32 model did not return logits")
    return logits


def _sample_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        probabilities = torch.softmax(logits.detach(), dim=-1)
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        sampled = torch.multinomial(flat, 1).reshape(labels.shape)
    return sampled.masked_fill(labels == -100, -100)


def _causal_loss(batch: TensorBatch, model: nn.Module, *, sample: bool) -> torch.Tensor:
    logits = _model_logits(model, batch)[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()
    if sample:
        labels = _sample_labels(logits, labels)
    return functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _load_sample_payload(path: Path) -> dict[str, object]:
    try:
        value = json_loads_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise Rank32EKFACSmokeError("cannot load bound generation sample") from error
    if not isinstance(value, dict):
        raise Rank32EKFACSmokeError("bound generation sample must be an object")
    return cast(dict[str, object], value)


def _validate_sample_signature(payload: dict[str, object]) -> None:
    claimed = payload.get("sample_sha256")
    if not isinstance(claimed, str):
        raise Rank32EKFACSmokeError("generation sample identity is invalid")
    body = {
        field: value for field, value in payload.items() if field != "sample_sha256"
    }
    if canonical_sha256(body) != claimed:
        raise Rank32EKFACSmokeError("generation sample identity differs")
    completion = payload.get("completion")
    if not isinstance(completion, str) or canonical_sha256(completion) != payload.get(
        "completion_sha256"
    ):
        raise Rank32EKFACSmokeError("generation sample completion differs")


def _evidence_entry(evidence: dict[str, Any], index: int) -> dict[str, object]:
    entries = evidence.get("sample_receipts")
    if not isinstance(entries, list) or not 0 <= index < len(entries):
        raise Rank32EKFACSmokeError("generation evidence index is invalid")
    entry = entries[index]
    if not isinstance(entry, dict):
        raise Rank32EKFACSmokeError("generation evidence entry is invalid")
    return cast(dict[str, object], entry)


def _validate_query_evidence(
    plan: Rank32EKFACSmokePlan,
    evidence: dict[str, Any],
    sample: dict[str, object],
) -> None:
    index = cast(int, plan.query_example["index"])
    entry = _evidence_entry(evidence, index)
    fields = ("index", "behavior", "question_id", "sample_sha256", "completion_sha256")
    if [sample.get(field) for field in fields] != [
        entry.get(field) for field in fields
    ]:
        raise Rank32EKFACSmokeError("generation sample differs from compact evidence")
    if (
        entry.get("sample_file_sha256")
        != plan.bindings["generation_sample"].file_sha256
    ):
        raise Rank32EKFACSmokeError("generation sample file differs from evidence")
    for field in fields:
        if sample.get(field) != plan.query_example.get(field):
            raise Rank32EKFACSmokeError("generation sample differs from smoke plan")


def _prompt_for_query(
    plan: Rank32EKFACSmokePlan, paths: Rank32RuntimePaths
) -> PromptRecord:
    root = paths.repository_root
    receipt_path = root / plan.bindings["prompt_bank_receipt"].path
    output_path = root / plan.bindings["prompt_bank"].path
    try:
        _, prompts = load_prompt_bank(receipt_path, output_path=output_path)
    except InspectArchiveValidationError as error:
        raise Rank32EKFACSmokeError("cannot load bound prompt bank") from error
    matches = [
        prompt
        for prompt in prompts
        if prompt.question_id == plan.query_example["question_id"]
    ]
    if (
        len(matches) != 1
        or canonical_sha256(matches[0].to_dict()) != plan.query_example["prompt_sha256"]
    ):
        raise Rank32EKFACSmokeError("query prompt identity differs")
    return matches[0]


def _append_eos(tokenizer: Any, input_ids: list[int], labels: list[int]) -> None:
    eos_id = tokenizer.eos_token_id
    if type(eos_id) is not int:
        raise Rank32EKFACSmokeError("query tokenizer EOS is invalid")
    if not input_ids or input_ids[-1] != eos_id:
        input_ids.append(eos_id)
        labels.append(eos_id)


def _render_query(
    plan: Rank32EKFACSmokePlan,
    tokenizer: Any,
    chat_template: str,
    prompt: PromptRecord,
    sample: dict[str, object],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prompt_ids = cast(
        list[int],
        tokenizer.apply_chat_template(
            list(prompt.messages),
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
        ),
    )
    completion = cast(str, sample["completion"])
    completion_ids = cast(
        list[int], tokenizer.encode(completion, add_special_tokens=False)
    )
    input_ids = [*prompt_ids, *completion_ids]
    labels = [-100] * len(prompt_ids) + completion_ids
    _append_eos(tokenizer, input_ids, labels)
    _validate_rendered_query(plan, sample, prompt_ids, input_ids, labels)
    return tuple(input_ids), tuple(labels)


def _validate_rendered_query(
    plan: Rank32EKFACSmokePlan,
    sample: dict[str, object],
    prompt_ids: list[int],
    input_ids: list[int],
    labels: list[int],
) -> None:
    expected = plan.query_example
    observed = {
        "rendered_input_ids_sha256": canonical_sha256(input_ids),
        "labels_sha256": canonical_sha256(labels),
        "input_token_count": len(input_ids),
        "supervised_token_count": sum(value != -100 for value in labels),
    }
    if any(observed[field] != expected[field] for field in observed):
        raise Rank32EKFACSmokeError("rendered query differs from smoke plan")
    if len(prompt_ids) != sample.get("input_tokens"):
        raise Rank32EKFACSmokeError("query prompt token count differs")
    if len(input_ids) - len(prompt_ids) != sample.get("generated_tokens"):
        raise Rank32EKFACSmokeError("query completion token count differs")


def _validate_training_example(
    plan: Rank32EKFACSmokePlan, example: BoundTrainingExample
) -> None:
    for field, expected in plan.training_example.items():
        if example.entry.get(field) != expected:
            raise Rank32EKFACSmokeError("training example differs from smoke plan")


def _prepare_inputs(
    plan: Rank32EKFACSmokePlan, paths: Rank32RuntimePaths
) -> _PreparedInputs:
    root = paths.repository_root
    config = load_training_data_package_config(
        root / plan.bindings["training_data_config"].path
    )
    try:
        training = load_bound_training_example(
            config,
            sample_path=paths.training_sample,
            tokenizer_snapshot=paths.base_snapshot,
            package_path=paths.training_package,
            receipt_path=root / plan.bindings["training_data_receipt"].path,
            row_ordinal=cast(int, plan.training_example["row_ordinal"]),
        )
        tokenizer = load_bound_tokenizer(config, paths.base_snapshot)
    except TrainingDataPackageError as error:
        raise Rank32EKFACSmokeError("cannot reconstruct training input") from error
    _validate_training_example(plan, training)
    query_ids, query_labels = _prepare_query(
        plan, paths, tokenizer, config.training.chat_template
    )
    return _PreparedInputs(training.input_ids, training.labels, query_ids, query_labels)


def _prepare_query(
    plan: Rank32EKFACSmokePlan,
    paths: Rank32RuntimePaths,
    tokenizer: Any,
    chat_template: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    root = paths.repository_root
    try:
        evidence = load_generation_evidence(
            root / plan.bindings["generation_evidence"].path
        )
    except GenerationEvidenceError as error:
        raise Rank32EKFACSmokeError("cannot load generation evidence") from error
    sample = _load_sample_payload(root / plan.bindings["generation_sample"].path)
    _validate_sample_signature(sample)
    _validate_query_evidence(plan, evidence, sample)
    return _render_query(
        plan, tokenizer, chat_template, _prompt_for_query(plan, paths), sample
    )


def _datasets(
    inputs: _PreparedInputs,
) -> tuple[_SingleExampleDataset, _SingleExampleDataset]:
    return (
        _SingleExampleDataset(inputs.training_input_ids, inputs.training_labels),
        _SingleExampleDataset(inputs.query_input_ids, inputs.query_labels),
    )


def _factor_arguments() -> FactorArguments:
    return FactorArguments(
        strategy="ekfac",
        use_empirical_fisher=False,
        amp_dtype=torch.bfloat16,
        covariance_max_examples=1,
        covariance_data_partitions=1,
        covariance_module_partitions=1,
        activation_covariance_dtype=torch.bfloat16,
        gradient_covariance_dtype=torch.bfloat16,
        eigendecomposition_dtype=torch.float64,
        lambda_max_examples=1,
        lambda_data_partitions=1,
        lambda_module_partitions=1,
        use_iterative_lambda_aggregation=True,
        offload_activations_to_cpu=True,
        per_sample_gradient_dtype=torch.bfloat16,
        lambda_dtype=torch.bfloat16,
    )


def _score_arguments() -> ScoreArguments:
    return ScoreArguments(
        damping_factor=None,
        offload_activations_to_cpu=True,
        data_partitions=1,
        module_partitions=1,
        query_gradient_accumulation_steps=10,
        query_gradient_low_rank=64,
        use_full_svd=True,
        aggregate_query_gradients=False,
        aggregate_train_gradients=False,
        query_gradient_svd_dtype=torch.float32,
        per_sample_gradient_dtype=torch.float32,
        precondition_dtype=torch.float32,
        score_dtype=torch.bfloat16,
    )


def _target_parameters(
    model: nn.Module, tracked_modules: tuple[str, ...]
) -> tuple[torch.Tensor, ...]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    parameters: list[torch.Tensor] = []
    for name in tracked_modules:
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise Rank32EKFACSmokeError("tracked LoRA module is not linear")
        module.weight.requires_grad = True
        parameters.append(module.weight)
    return tuple(parameters)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return canonical_sha256(
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes_sha256": bytes_sha256(raw),
        }
    )


def _parameter_observation(
    model: nn.Module, tracked_modules: tuple[str, ...]
) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    entries: list[dict[str, object]] = []
    shapes: list[tuple[str, tuple[int, ...]]] = []
    for name in tracked_modules:
        module = cast(nn.Linear, model.get_submodule(name))
        shape = tuple(module.weight.shape)
        entries.append(
            {
                "name": name,
                "shape": list(shape),
                "sha256": _tensor_sha256(module.weight),
            }
        )
        shapes.append((name, shape))
    return canonical_sha256(entries), tuple(shapes)


def _batch(dataset: _SingleExampleDataset) -> TensorBatch:
    return {name: value.unsqueeze(0) for name, value in dataset[0].items()}


def _gradient_sha256(
    task: _Rank32CausalTask,
    model: nn.Module,
    dataset: _SingleExampleDataset,
    parameters: tuple[torch.Tensor, ...],
    *,
    query: bool,
) -> str:
    model.zero_grad(set_to_none=True)
    batch = _batch(dataset)
    loss = (
        task.compute_measurement(batch, model)
        if query
        else task.compute_train_loss(batch, model, sample=False)
    )
    gradients = torch.autograd.grad(loss, parameters)
    return canonical_sha256([_tensor_sha256(value) for value in gradients])


def _analyzer(model: nn.Module, task: _Rank32CausalTask, run_dir: Path) -> Analyzer:
    return Analyzer(
        analysis_name="rank32-ekfac",
        model=prepare_model(model, task),
        task=task,
        cpu=True,
        disable_tqdm=True,
        disable_model_save=True,
        output_dir=str(run_dir),
    )


def _fit_factors(analyzer: Analyzer, dataset: _SingleExampleDataset) -> None:
    analyzer.fit_all_factors(
        factors_name="bounded-factors",
        dataset=dataset,
        per_device_batch_size=1,
        factor_args=_factor_arguments(),
        overwrite_output_dir=True,
    )


def _compute_score(
    analyzer: Analyzer,
    training: _SingleExampleDataset,
    query: _SingleExampleDataset,
) -> float:
    analyzer.compute_pairwise_scores(
        scores_name="bounded-scores",
        factors_name="bounded-factors",
        query_dataset=query,
        train_dataset=training,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=_score_arguments(),
        overwrite_output_dir=True,
    )
    loaded = analyzer.load_pairwise_scores("bounded-scores")
    if loaded is None or "all_modules" not in loaded:
        raise Rank32EKFACSmokeError("Kronfluence rank-32 score output is missing")
    values = cast(list[list[float]], loaded["all_modules"].tolist())
    if len(values) != 1 or len(values[0]) != 1 or not math.isfinite(values[0][0]):
        raise Rank32EKFACSmokeError("Kronfluence rank-32 score is invalid")
    return values[0][0]


def _tensor_file_semantic(path: Path) -> str:
    entries: list[dict[str, object]] = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            value = handle.get_tensor(name)
            entries.append({"name": name, "sha256": _tensor_sha256(value)})
    return canonical_sha256(entries)


def _file_semantic(path: Path) -> str:
    if path.suffix == ".safetensors":
        return _tensor_file_semantic(path)
    if path.suffix == ".json":
        try:
            return canonical_sha256(json_loads_unique(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, ValueError) as error:
            raise Rank32EKFACSmokeError("invalid Kronfluence JSON artifact") from error
    return bytes_sha256(path.read_bytes())


def _store_semantic(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files or any(item.is_symlink() for item in files):
        raise Rank32EKFACSmokeError("Kronfluence rank-32 store is invalid")
    return canonical_sha256(
        [
            {
                "path": item.relative_to(path).as_posix(),
                "semantic_sha256": _file_semantic(item),
            }
            for item in files
        ]
    )


def _factor_path(run_dir: Path) -> Path:
    return run_dir / "rank32-ekfac/factors_bounded-factors"


def _score_path(run_dir: Path) -> Path:
    return run_dir / "rank32-ekfac/scores_bounded-scores"


def _runtime_versions() -> tuple[tuple[str, str], ...]:
    names = (
        "huggingface-hub",
        "kronfluence",
        "peft",
        "pyarrow",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    )
    try:
        return tuple((name, importlib.metadata.version(name)) for name in names)
    except importlib.metadata.PackageNotFoundError as error:
        raise Rank32EKFACSmokeError("rank-32 runtime package is missing") from error


def _inverse_application_sha256(
    factor_sha256: str, query_gradient_sha256: str, score: float
) -> str:
    return canonical_sha256(
        {
            "factor_store_semantic_sha256": factor_sha256,
            "query_gradient_sha256": query_gradient_sha256,
            "score_arguments": dict(sorted(_SCORE_IDENTITY.items())),
            "score": score,
        }
    )


_SCORE_IDENTITY: dict[str, object] = {
    "damping_factor": None,
    "query_gradient_accumulation_steps": 10,
    "query_gradient_low_rank": 64,
    "use_full_svd": True,
    "query_gradient_svd_dtype": "float32",
    "per_sample_gradient_dtype": "float32",
    "precondition_dtype": "float32",
    "score_dtype": "bfloat16",
}


def _gradient_observations(
    plan: Rank32EKFACSmokePlan,
    model: nn.Module,
    training: _SingleExampleDataset,
    query: _SingleExampleDataset,
    task: _Rank32CausalTask,
) -> tuple[str, tuple[tuple[str, tuple[int, ...]], ...], str, str]:
    parameters = _target_parameters(model, plan.tracked_modules)
    parameter_sha, shapes = _parameter_observation(model, plan.tracked_modules)
    train_sha = _gradient_sha256(task, model, training, parameters, query=False)
    query_sha = _gradient_sha256(task, model, query, parameters, query=True)
    return parameter_sha, shapes, train_sha, query_sha


def _factor_score(
    analyzer: Analyzer,
    training: _SingleExampleDataset,
    query: _SingleExampleDataset,
    run_dir: Path,
) -> tuple[str, float]:
    _fit_factors(analyzer, training)
    factor_sha = _store_semantic(_factor_path(run_dir))
    score = _compute_score(analyzer, training, query)
    if _store_semantic(_factor_path(run_dir)) != factor_sha:
        raise Rank32EKFACSmokeError("factor store changed during rank-32 scoring")
    return factor_sha, score


def _run_analysis(
    plan: Rank32EKFACSmokePlan,
    model: nn.Module,
    inputs: _PreparedInputs,
    run_dir: Path,
) -> Rank32EKFACObservation:
    training, query = _datasets(inputs)
    task = _Rank32CausalTask(plan.tracked_modules)
    parameter_sha, shapes, training_gradient, query_gradient = _gradient_observations(
        plan, model, training, query, task
    )
    analyzer = _analyzer(model, task, run_dir)
    factor_sha, score = _factor_score(analyzer, training, query, run_dir)
    return Rank32EKFACObservation(
        runtime_versions=_runtime_versions(),
        artifact_manifest_sha256=plan.artifact_manifest_sha256,
        topology_sha256=plan.topology_sha256,
        tracked_parameter_sha256=parameter_sha,
        tracked_module_shapes=shapes,
        training_input_sha256=plan.training_input_sha256,
        query_input_sha256=plan.query_input_sha256,
        factor_store_semantic_sha256=factor_sha,
        training_gradient_sha256=training_gradient,
        query_gradient_sha256=query_gradient,
        score_store_semantic_sha256=_store_semantic(_score_path(run_dir)),
        inverse_application_sha256=_inverse_application_sha256(
            factor_sha, query_gradient, score
        ),
        score=score,
    )


def _seeded_analysis(
    plan: Rank32EKFACSmokePlan,
    paths: Rank32RuntimePaths,
    inputs: _PreparedInputs,
    run_dir: Path,
) -> Rank32EKFACObservation:
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.manual_seed(cast(int, plan.factor_arguments["seed"]))
        torch.use_deterministic_algorithms(True)
        model = _load_model(plan, paths)
        _validate_loaded_topology(plan, model)
        return _run_analysis(plan, model, inputs, run_dir)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        gc.collect()


def run_rank32_kronfluence_smoke(
    plan: Rank32EKFACSmokePlan,
    paths: Rank32RuntimePaths,
    run_dir: Path,
) -> Rank32EKFACObservation:
    """Run one exact rank-32 sampled-true-Fisher bounded EKFAC score."""
    if run_dir.exists():
        raise Rank32EKFACSmokeError("refusing to reuse rank-32 backend run directory")
    if _runtime_versions() != tuple(
        sorted(cast(dict[str, str], plan.runtime["versions"]).items())
    ):
        raise Rank32EKFACSmokeError("installed runtime differs from rank-32 plan")
    _validate_adapter_weights(paths.adapter_snapshot)
    inputs = _prepare_inputs(plan, paths)
    with torch.random.fork_rng(devices=[]):
        return _seeded_analysis(plan, paths, inputs, run_dir)
