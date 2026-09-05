"""Transformers backend for the interim rank-32 load-and-generate smoke."""

from __future__ import annotations

import importlib
import importlib.metadata
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from concept_filtering.generation_bank import (
    GenerationObservation,
    GenerationPlan,
    GenerationRuntime,
)
from concept_filtering.prompt_bank import PromptRecord
from concept_filtering.serialization import file_sha256
from concept_filtering.smoke import (
    RuntimeObservation,
    SmokePlan,
    SmokeValidationError,
)


def _module(name: str) -> Any:
    try:
        return cast(Any, importlib.import_module(name))
    except ImportError as error:
        raise SmokeValidationError(
            "inference dependencies are missing; install the inference extra"
        ) from error


def _hash_file(path: Path, label: str) -> str:
    try:
        return file_sha256(path)
    except OSError as error:
        raise SmokeValidationError(f"cannot hash {label} {path}: {error}") from error


def _validate_device(torch: Any, device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        raise SmokeValidationError("configured MPS device is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise SmokeValidationError("configured CUDA device is unavailable")


def download_smoke_snapshots(
    hub: Any, plan: SmokePlan, cache_dir: Path
) -> tuple[Path, Path]:
    base = hub.snapshot_download(
        repo_id=plan.base.repository,
        revision=plan.base.revision,
        cache_dir=str(cache_dir),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    adapter = hub.snapshot_download(
        repo_id=plan.adapter.repository,
        revision=plan.adapter.revision,
        cache_dir=str(cache_dir),
        allow_patterns=[f"{Path(plan.adapter.path or '').parent.as_posix()}/*"],
    )
    return Path(base), Path(adapter)


def load_smoke_model(
    transformers: Any,
    peft: Any,
    torch: Any,
    plan: SmokePlan,
    base_snapshot: Path,
    adapter_snapshot: Path,
) -> tuple[Any, Any]:
    adapter_dir = adapter_snapshot / Path(plan.adapter.path or "").parent
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(adapter_dir), local_files_only=True
    )
    template_path = adapter_snapshot / plan.template_path
    template = template_path.read_text(encoding="utf-8")
    if tokenizer.chat_template != template:
        raise SmokeValidationError(
            "loaded tokenizer did not select the pinned template"
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(base_snapshot),
        dtype=getattr(torch, plan.dtype),
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = peft.PeftModel.from_pretrained(
        model, str(adapter_dir), local_files_only=True
    )
    return tokenizer, model.to(plan.device).eval()


def _generate(torch: Any, tokenizer: Any, model: Any, plan: SmokePlan) -> str:
    messages = [
        {"role": "system", "content": plan.prompt.system},
        {"role": "user", "content": plan.prompt.user},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(plan.device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            do_sample=plan.do_sample,
            max_new_tokens=plan.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return cast(
        str,
        tokenizer.decode(generated[0, input_ids.shape[-1] :], skip_special_tokens=True),
    )


def generate_prompt(
    torch: Any,
    tokenizer: Any,
    model: Any,
    smoke: SmokePlan,
    plan: GenerationPlan,
    prompt: PromptRecord,
) -> GenerationObservation:
    input_ids = _prompt_input_ids(tokenizer, prompt, smoke.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            do_sample=plan.do_sample,
            max_new_tokens=plan.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - started
    output_ids = generated[0, input_ids.shape[-1] :]
    generated_tokens = int(output_ids.shape[-1])
    return GenerationObservation(
        prompt.question_id,
        prompt.behavior,
        cast(str, tokenizer.decode(output_ids, skip_special_tokens=True)),
        int(input_ids.shape[-1]),
        generated_tokens,
        elapsed,
        generated_tokens == plan.max_new_tokens,
    )


def _prompt_input_ids(tokenizer: Any, prompt: PromptRecord, device: str) -> Any:
    return tokenizer.apply_chat_template(
        list(prompt.messages),
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(device)


def _runtime_versions() -> dict[str, str]:
    names = ("huggingface-hub", "peft", "torch", "transformers")
    return {name: importlib.metadata.version(name) for name in names}


def _verify_snapshot_hashes(
    plan: SmokePlan, adapter: Path, template: Path
) -> tuple[str, str]:
    adapter_sha256 = _hash_file(adapter, "adapter weights")
    template_sha256 = _hash_file(template, "chat template")
    if adapter_sha256 != plan.adapter.sha256:
        raise SmokeValidationError("downloaded adapter SHA-256 does not match the plan")
    if template_sha256 != plan.template_sha256:
        raise SmokeValidationError(
            "downloaded template SHA-256 does not match the plan"
        )
    return adapter_sha256, template_sha256


def run_transformers_smoke(plan: SmokePlan, cache_dir: Path) -> RuntimeObservation:
    """Download exact snapshots, load the adapter, and generate one response."""
    torch = _module("torch")
    transformers = _module("transformers")
    peft = _module("peft")
    hub = _module("huggingface_hub")
    _validate_device(torch, plan.device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_snapshot, adapter_snapshot = download_smoke_snapshots(hub, plan, cache_dir)
    adapter_path = adapter_snapshot / (plan.adapter.path or "")
    template_path = adapter_snapshot / plan.template_path
    adapter_sha256, template_sha256 = _verify_snapshot_hashes(
        plan, adapter_path, template_path
    )
    tokenizer, model = load_smoke_model(
        transformers, peft, torch, plan, base_snapshot, adapter_snapshot
    )
    completion = _generate(torch, tokenizer, model, plan)
    return RuntimeObservation(
        completion=completion,
        adapter_sha256=adapter_sha256,
        template_sha256=template_sha256,
        runtime_versions=_runtime_versions(),
    )


def _generation_observations(
    torch: Any,
    tokenizer: Any,
    model: Any,
    smoke: SmokePlan,
    plan: GenerationPlan,
    prompts: tuple[PromptRecord, ...],
) -> Iterator[GenerationObservation]:
    for prompt in prompts:
        yield generate_prompt(torch, tokenizer, model, smoke, plan, prompt)


def run_transformers_generation(
    smoke: SmokePlan,
    plan: GenerationPlan,
    prompts: tuple[PromptRecord, ...],
    cache_dir: Path,
) -> GenerationRuntime:
    """Load exact pinned artifacts once and stream prompt-bank generations."""
    torch = _module("torch")
    transformers = _module("transformers")
    peft = _module("peft")
    hub = _module("huggingface_hub")
    _validate_device(torch, smoke.device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_snapshot, adapter_snapshot = download_smoke_snapshots(hub, smoke, cache_dir)
    adapter_path = adapter_snapshot / (smoke.adapter.path or "")
    template_path = adapter_snapshot / smoke.template_path
    adapter_sha, template_sha = _verify_snapshot_hashes(
        smoke, adapter_path, template_path
    )
    tokenizer, model = load_smoke_model(
        transformers, peft, torch, smoke, base_snapshot, adapter_snapshot
    )
    return GenerationRuntime(
        adapter_sha,
        template_sha,
        _runtime_versions(),
        _generation_observations(torch, tokenizer, model, smoke, plan, prompts),
    )
