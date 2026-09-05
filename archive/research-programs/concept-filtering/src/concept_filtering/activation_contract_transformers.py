"""Transformers execution for the bounded layer-16 activation contract smoke."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from concept_filtering.activation_contract import (
    ActivationContractError,
    ActivationContractPlan,
    ActivationObservation,
    ArtifactSpec,
    ContractExample,
    TensorObservation,
    aligned_token_segments,
)
from concept_filtering.base_response_generation import (
    GENERATION_PROMPT_HEADER_ONLY,
    BaseResponseGenerationError,
    strip_reasoning_opener,
)
from concept_filtering.serialization import file_sha256


def _module(name: str) -> Any:
    try:
        return cast(Any, importlib.import_module(name))
    except ImportError as error:
        raise ActivationContractError(
            "inference dependencies are missing; install the inference extra"
        ) from error


def _verify_runtime_versions(plan: ActivationContractPlan) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in plan.runtime_versions.items():
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ActivationContractError(f"{name} is not installed") from error
        if found != expected:
            raise ActivationContractError(
                f"installed {name} {found} differs from pinned {expected}"
            )
        observed[name] = found
    return observed


def _verify_snapshot(artifact: ArtifactSpec, root: Path, label: str) -> dict[str, str]:
    base = root / artifact.path if artifact.path else root
    observed: dict[str, str] = {}
    for name, expected in artifact.files.items():
        candidate = base / name
        if not candidate.is_file():
            raise ActivationContractError(f"{label} snapshot is missing {name}")
        digest = file_sha256(candidate)
        if digest != expected:
            raise ActivationContractError(f"{label} snapshot file {name} has drifted")
        observed[name] = digest
    return observed


def _loaded_tokenizer(
    transformers: Any, plan: ActivationContractPlan, root: Path
) -> Any:
    directory = root / plan.tokenizer.path if plan.tokenizer.path else root
    return transformers.AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True
    )


def _rendered_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, generation: bool
) -> tuple[int, ...]:
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=generation, return_tensors=None, tokenize=True
    )
    return tuple(int(value) for value in rendered)


def _encoded(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    return tuple(int(value) for value in encoded["input_ids"])


def _supplied_text(example: ContractExample) -> str:
    content = example.supplied_response.get("content")
    if not isinstance(content, str) or not content:
        raise ActivationContractError(
            f"example {example.example_id} has no supplied response content"
        )
    return content


def _header_only_prompt_ids(
    tokenizer: Any, prompt: list[dict[str, str]]
) -> tuple[int, ...]:
    rendered = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_tensors=None, tokenize=False
    )
    if not isinstance(rendered, str):
        raise ActivationContractError("chat template did not return rendered text")
    try:
        stripped = strip_reasoning_opener(rendered)
    except BaseResponseGenerationError as error:
        raise ActivationContractError(str(error)) from error
    return _encoded(tokenizer, stripped)


def _sequences(
    tokenizer: Any, example: ContractExample, *, generation_prompt: str
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Build both response sequences in the context the base response used.

    Under the ``assistant-header-only`` generation core the base response was
    generated after the stripped prompt (assistant header, no reasoning
    opener), which is also the natural prefix of the template-rendered
    assistant turn; the supplied response is placed there and the prefix
    relationship against the full assistant-turn rendering is enforced
    fail-closed. Under the original ``template-think`` core the two renderings
    share no prefix, so the supplied response is placed in the generation
    context instead; the two sequences then differ only in their response
    tokens, which is what the declared aggregation compares.
    """
    prompt = [dict(message) for message in example.prompt_messages]
    if generation_prompt == GENERATION_PROMPT_HEADER_ONLY:
        prompt_ids = _header_only_prompt_ids(tokenizer, prompt)
        supplied_ids = prompt_ids + _encoded(tokenizer, _supplied_text(example))
        conversation = [*prompt, dict(example.supplied_response)]
        full_ids = _rendered_ids(tokenizer, conversation, generation=False)
        if full_ids[: len(supplied_ids)] != supplied_ids:
            raise ActivationContractError(
                "stripped prompt plus supplied response is not a prefix of the "
                "assistant-turn rendering"
            )
    else:
        prompt_ids = _rendered_ids(tokenizer, prompt, generation=True)
        supplied_ids = prompt_ids + _encoded(tokenizer, _supplied_text(example))
    base_completion_ids = _encoded(tokenizer, example.base_response_text)
    return prompt_ids, supplied_ids, base_completion_ids


def _find_hook_module(model: Any, hook_name: str) -> tuple[str, Any]:
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name == hook_name or name.endswith(f".{hook_name}")
    ]
    if len(matches) != 1:
        raise ActivationContractError(
            f"expected one {hook_name} module, observed {len(matches)}"
        )
    return cast(tuple[str, Any], matches[0])


def _hook_tensor(output: object) -> Any:
    if isinstance(output, tuple):
        if not output:
            raise ActivationContractError("hook output tuple is empty")
        return cast(Any, output[0])
    return output


def _hooked_activation(
    torch: Any, model: Any, hook_module: Any, ids: tuple[int, ...]
) -> Any:
    captures: list[Any] = []

    def capture(_module: object, _inputs: object, output: object) -> None:
        captures.append(_hook_tensor(output).detach().to(torch.float32).cpu())

    handle = hook_module.register_forward_hook(capture)
    try:
        input_ids = torch.tensor([list(ids)], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise ActivationContractError(
            f"expected one hook capture, observed {len(captures)}"
        )
    return captures[0]


def _tensor_observation(torch: Any, value: Any) -> TensorObservation:
    tensor = value.detach().to(torch.float32).cpu().contiguous()
    return TensorObservation(
        tuple(int(size) for size in tensor.shape),
        str(tensor.dtype).removeprefix("torch."),
        bool(torch.isfinite(tensor).all().item()),
        sha256(tensor.numpy().tobytes()).hexdigest(),
    )


def _span_mean(activation: Any, span: tuple[int, int]) -> Any:
    start, end = span
    return activation[0, start:end, :].mean(dim=0)


def _load_model(
    transformers: Any, torch: Any, snapshot: Path, dtype: str | None
) -> Any:
    options: dict[str, Any] = {"local_files_only": True, "low_cpu_mem_usage": True}
    if dtype is not None:
        options["dtype"] = getattr(torch, dtype)
    return transformers.AutoModelForCausalLM.from_pretrained(
        str(snapshot), **options
    ).eval()


def _tensors(
    torch: Any,
    supplied_activation: Any,
    base_activation: Any,
    alignment: Any,
) -> dict[str, TensorObservation]:
    prompt_span = (0, alignment.prompt_token_count)
    return {
        "supplied_response_mean": _tensor_observation(
            torch, _span_mean(supplied_activation, alignment.supplied_span)
        ),
        "base_response_mean": _tensor_observation(
            torch, _span_mean(base_activation, alignment.base_span)
        ),
        "prompt_mean_supplied_forward": _tensor_observation(
            torch, _span_mean(supplied_activation, prompt_span)
        ),
        "prompt_mean_base_forward": _tensor_observation(
            torch, _span_mean(base_activation, prompt_span)
        ),
    }


def run_activation_contract(
    plan: ActivationContractPlan,
    example: ContractExample,
    *,
    generation_prompt: str,
    base_snapshot: Path,
    tokenizer_snapshot: Path,
    device: str | None = None,
    dtype: str | None = None,
) -> ActivationObservation:
    """Execute the frozen layer-16 hook and declared aggregation for one example."""
    torch = _module("torch")
    transformers = _module("transformers")
    versions = _verify_runtime_versions(plan)
    artifacts = {
        "base": _verify_snapshot(plan.base, base_snapshot, "base"),
        "tokenizer": _verify_snapshot(plan.tokenizer, tokenizer_snapshot, "tokenizer"),
    }
    tokenizer = _loaded_tokenizer(transformers, plan, tokenizer_snapshot)
    prompt_ids, supplied_ids, completion_ids = _sequences(
        tokenizer, example, generation_prompt=generation_prompt
    )
    alignment = aligned_token_segments(
        prompt_ids=prompt_ids,
        supplied_ids=supplied_ids,
        base_completion_ids=completion_ids,
        recorded_completion_token_count=example.recorded_completion_token_count,
        terminated_by=example.terminated_by,
    )
    model = _load_model(transformers, torch, base_snapshot, dtype)
    if device is not None:
        model = model.to(torch.device(device))
    resolved_name, hook_module = _find_hook_module(model, plan.probe.hook_name)
    supplied_activation = _hooked_activation(torch, model, hook_module, supplied_ids)
    base_activation = _hooked_activation(
        torch, model, hook_module, prompt_ids + completion_ids
    )
    tensors = _tensors(torch, supplied_activation, base_activation, alignment)
    del hook_module, model
    gc.collect()
    return ActivationObservation(
        artifact_sha256=artifacts,
        runtime_versions=versions,
        resolved_hook_name=resolved_name,
        model_forward_count=2,
        hook_capture_count=2,
        alignment=alignment,
        example=example.identity(),
        tensors=tensors,
    )
