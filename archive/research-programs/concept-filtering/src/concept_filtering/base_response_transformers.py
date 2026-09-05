"""Base-only greedy Transformers backend for base-response generation."""

from __future__ import annotations

import importlib
import importlib.metadata
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from concept_filtering.base_response_generation import (
    GENERATION_PROMPT_HEADER_ONLY,
    BaseResponseConfig,
    BaseResponseGenerationError,
    strip_reasoning_opener,
    verify_artifact_snapshot,
)
from concept_filtering.base_response_join import JoinedExample
from concept_filtering.base_response_runner import GeneratedResponse


class BaseResponseBackendError(ValueError):
    """Raised when the pinned base-only backend cannot run as frozen."""


def _module(name: str) -> Any:
    try:
        return cast(Any, importlib.import_module(name))
    except ImportError as error:
        raise BaseResponseBackendError(
            "inference dependencies are missing; install the inference extra"
        ) from error


def _verify_runtime_versions(config: BaseResponseConfig) -> None:
    for name, expected in config.runtime_versions.items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise BaseResponseBackendError(f"{name} is not installed") from error
        if observed != expected:
            raise BaseResponseBackendError(
                f"installed {name} {observed} differs from pinned {expected}"
            )


_EXECUTION_DTYPES = ("float32", "float16", "bfloat16")


def execution_placement(
    torch: Any, device: str | None, dtype: str | None
) -> tuple[Any, Any]:
    """Resolve the requested execution device and dtype, failing closed."""
    placed_device = None
    if device is not None:
        try:
            placed_device = torch.device(device)
        except (RuntimeError, ValueError) as error:
            raise BaseResponseBackendError(
                f"invalid execution device {device!r}"
            ) from error
    placed_dtype = None
    if dtype is not None:
        if dtype not in _EXECUTION_DTYPES:
            raise BaseResponseBackendError(f"invalid execution dtype {dtype!r}")
        placed_dtype = getattr(torch, dtype)
    return placed_device, placed_dtype


def _special_token_ids(tokenizer: Any) -> tuple[int, int]:
    eos_token_id = tokenizer.eos_token_id
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise BaseResponseBackendError("tokenizer EOS token is invalid")
    pad_token_id = tokenizer.pad_token_id
    if type(pad_token_id) is not int or pad_token_id < 0:
        pad_token_id = eos_token_id
    return eos_token_id, pad_token_id


def _header_only_prompt_ids(
    tokenizer: Any, settings: Any, messages: list[dict[str, str]]
) -> Any:
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=settings.add_generation_prompt,
        tokenize=False,
    )
    if not isinstance(rendered, str):
        raise BaseResponseBackendError("chat template did not return rendered text")
    try:
        stripped = strip_reasoning_opener(rendered)
    except BaseResponseGenerationError as error:
        raise BaseResponseBackendError(str(error)) from error
    encoded = tokenizer(stripped, add_special_tokens=False, return_tensors="pt")
    return encoded["input_ids"]


def _rendered_prompt(
    tokenizer: Any, settings: Any, example: JoinedExample
) -> tuple[Any, int, int]:
    messages = [dict(message) for message in example.prompt_messages]
    if settings.generation_prompt == GENERATION_PROMPT_HEADER_ONLY:
        input_ids = _header_only_prompt_ids(tokenizer, settings, messages)
    else:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=settings.add_generation_prompt,
            return_tensors="pt",
            tokenize=True,
        )
    prompt_token_count = int(input_ids.shape[-1])
    capacity = settings.max_total_tokens - prompt_token_count
    if capacity <= 0:
        raise BaseResponseBackendError(
            f"example {example.example_id} leaves no completion capacity"
        )
    return input_ids, prompt_token_count, capacity


def _observed_response(
    tokenizer: Any,
    generated: Any,
    prompt_token_count: int,
    capacity: int,
    eos_token_id: int,
    elapsed: float,
    example: JoinedExample,
) -> GeneratedResponse:
    output_ids = generated[0, prompt_token_count:]
    completion_token_count = int(output_ids.shape[-1])
    terminated_by_eos = int(output_ids[-1]) == eos_token_id
    if not terminated_by_eos and completion_token_count != capacity:
        raise BaseResponseBackendError(
            f"example {example.example_id} stopped without EOS or cap"
        )
    return GeneratedResponse(
        prompt_token_count=prompt_token_count,
        completion_token_count=completion_token_count,
        response_text=cast(str, tokenizer.decode(output_ids, skip_special_tokens=True)),
        terminated_by_eos=terminated_by_eos,
        duration_seconds=elapsed,
    )


def backend_from_loaded(
    config: BaseResponseConfig, tokenizer: Any, model: Any, torch: Any
) -> Callable[[JoinedExample], GeneratedResponse]:
    """Build the per-example generator around already-loaded pinned artifacts."""
    eos_token_id, pad_token_id = _special_token_ids(tokenizer)
    settings = config.generation

    def generate(example: JoinedExample) -> GeneratedResponse:
        input_ids, prompt_token_count, capacity = _rendered_prompt(
            tokenizer, settings, example
        )
        input_ids = input_ids.to(model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                do_sample=settings.do_sample,
                max_new_tokens=capacity,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        elapsed = time.perf_counter() - started
        return _observed_response(
            tokenizer,
            generated,
            prompt_token_count,
            capacity,
            eos_token_id,
            elapsed,
            example,
        )

    return generate


def _loaded_tokenizer(
    transformers: Any, config: BaseResponseConfig, tokenizer_snapshot: Path
) -> Any:
    tokenizer_dir = (
        tokenizer_snapshot / config.tokenizer.path
        if config.tokenizer.path
        else tokenizer_snapshot
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True
    )
    template = (tokenizer_dir / "chat_template.jinja").read_text(encoding="utf-8")
    if tokenizer.chat_template != template:
        raise BaseResponseBackendError(
            "loaded tokenizer did not select the pinned template"
        )
    return tokenizer


def build_transformers_backend(
    config: BaseResponseConfig,
    *,
    base_snapshot: Path,
    tokenizer_snapshot: Path,
    device: str | None = None,
    dtype: str | None = None,
) -> Callable[[JoinedExample], GeneratedResponse]:
    """Verify pinned artifacts, load the base model once, and return a generator."""
    torch = _module("torch")
    transformers = _module("transformers")
    placed_device, placed_dtype = execution_placement(torch, device, dtype)
    _verify_runtime_versions(config)
    verify_artifact_snapshot(config.base, base_snapshot)
    verify_artifact_snapshot(config.tokenizer, tokenizer_snapshot)
    tokenizer = _loaded_tokenizer(transformers, config, tokenizer_snapshot)
    load_options: dict[str, Any] = {"local_files_only": True, "low_cpu_mem_usage": True}
    if placed_dtype is not None:
        load_options["dtype"] = placed_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(base_snapshot), **load_options
    ).eval()
    if placed_device is not None:
        model = model.to(placed_device)
    return backend_from_loaded(config, tokenizer, model, torch)
