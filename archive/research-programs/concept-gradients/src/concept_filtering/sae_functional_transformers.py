"""Transformers and SAELens backend for the bounded functional smoke."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from concept_filtering.sae_functional_smoke import (
    SAEFunctionalObservation,
    SAEFunctionalSmokeError,
    SAEFunctionalSmokePlan,
    TensorObservation,
)
from concept_filtering.serialization import file_sha256


@dataclass(frozen=True)
class ProbeExecution:
    input_token_ids: tuple[int, ...]
    active_adapter: str
    resolved_hook_name: str
    sae_architecture: str
    model_forward_count: int
    hook_capture_count: int
    encode_count: int
    decode_count: int
    tensors: dict[str, TensorObservation]


ProbeRunner = Callable[[SAEFunctionalSmokePlan, Path, Path, Path], ProbeExecution]
VersionReader = Callable[[], dict[str, str]]


def _module(name: str) -> Any:
    try:
        return cast(Any, importlib.import_module(name))
    except ImportError as error:
        raise SAEFunctionalSmokeError(
            "functional-smoke dependencies are missing; install the pinned runtime"
        ) from error


def _runtime_versions() -> dict[str, str]:
    names = (
        "huggingface-hub",
        "peft",
        "sae-lens",
        "safetensors",
        "torch",
        "transformers",
    )
    try:
        return {name: importlib.metadata.version(name) for name in names}
    except importlib.metadata.PackageNotFoundError as error:
        raise SAEFunctionalSmokeError(f"missing runtime package: {error}") from error


def _artifact_hashes(
    plan: SAEFunctionalSmokePlan, roots: dict[str, Path]
) -> dict[str, dict[str, str]]:
    if set(roots) != set(plan.artifacts):
        raise SAEFunctionalSmokeError("runtime artifact roots do not match the plan")
    observed: dict[str, dict[str, str]] = {}
    for role, artifact in plan.artifacts.items():
        hashes = {name: file_sha256(roots[role] / name) for name in artifact.files}
        if hashes != artifact.files:
            raise SAEFunctionalSmokeError(
                f"{role} artifact SHA-256 values do not match"
            )
        observed[role] = hashes
    return observed


def _token_ids(tokenizer: Any, plan: SAEFunctionalSmokePlan) -> Any:
    encoded = tokenizer(
        plan.probe.input_text,
        add_special_tokens=plan.probe.add_special_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    observed = tuple(cast(list[int], input_ids[0].tolist()))
    if observed != plan.probe.input_token_ids:
        raise SAEFunctionalSmokeError("tokenized input does not match the plan")
    return input_ids


def _load_model(
    plan: SAEFunctionalSmokePlan, base: Path, adapter: Path
) -> tuple[Any, Any, Any]:
    torch = _module("torch")
    transformers = _module("transformers")
    peft = _module("peft")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(adapter), local_files_only=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(base),
        dtype=getattr(torch, plan.runtime.model_dtype),
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = peft.PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    return torch, tokenizer, model.to(plan.runtime.device).eval()


def _find_hook_module(model: Any, hook_name: str) -> tuple[str, Any]:
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name == hook_name or name.endswith(f".{hook_name}")
    ]
    if len(matches) != 1:
        raise SAEFunctionalSmokeError(
            f"expected one {hook_name} module, observed {len(matches)}"
        )
    return cast(tuple[str, Any], matches[0])


def _hook_tensor(output: object) -> Any:
    if isinstance(output, tuple):
        if not output:
            raise SAEFunctionalSmokeError("hook output tuple is empty")
        return cast(Any, output[0])
    return output


def _run_hooked_forward(
    torch: Any, model: Any, input_ids: Any, hook_module: Any
) -> Any:
    captures: list[Any] = []

    def capture(_module: object, _inputs: object, output: object) -> None:
        captures.append(_hook_tensor(output).detach().cpu())

    handle = hook_module.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids.to(model.device), use_cache=False)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise SAEFunctionalSmokeError(
            f"expected one hook capture, observed {len(captures)}"
        )
    return captures[0]


def _capture_activation(
    plan: SAEFunctionalSmokePlan, base: Path, adapter: Path
) -> tuple[tuple[int, ...], str, str, Any]:
    torch, tokenizer, model = _load_model(plan, base, adapter)
    input_ids = _token_ids(tokenizer, plan)
    resolved_name, hook_module = _find_hook_module(model, plan.probe.hook_name)
    active_adapter = cast(str, model.active_adapter)
    activation = _run_hooked_forward(torch, model, input_ids, hook_module)
    del hook_module, model
    gc.collect()
    return plan.probe.input_token_ids, active_adapter, resolved_name, activation


def _sae_architecture(sae: Any) -> str:
    value = sae.cfg.architecture
    return cast(str, value() if callable(value) else value)


def _encode_decode(
    plan: SAEFunctionalSmokePlan, sae_directory: Path, value: Any
) -> tuple[str, Any, Any]:
    torch = _module("torch")
    sae_lens = _module("sae_lens")
    sae = sae_lens.SAE.load_from_disk(
        sae_directory,
        device=plan.runtime.device,
        dtype=plan.runtime.sae_dtype,
    ).eval()
    with torch.inference_mode():
        features = sae.encode(value.to(getattr(torch, plan.runtime.sae_dtype)))
        decoded = sae.decode(features)
    return _sae_architecture(sae), features, decoded


def _tensor_observation(torch: Any, value: Any) -> TensorObservation:
    tensor = value.detach().contiguous().cpu()
    raw = tensor.numpy().tobytes()
    return TensorObservation(
        tuple(cast(tuple[int, ...], tensor.shape)),
        str(tensor.dtype).removeprefix("torch."),
        bool(torch.isfinite(tensor).all().item()),
        sha256(raw).hexdigest(),
    )


def _execute_probe(
    plan: SAEFunctionalSmokePlan,
    base_snapshot: Path,
    adapter_snapshot: Path,
    sae_directory: Path,
) -> ProbeExecution:
    tokens, active, hook, activation = _capture_activation(
        plan, base_snapshot, adapter_snapshot
    )
    architecture, features, decoded = _encode_decode(plan, sae_directory, activation)
    torch = _module("torch")
    tensors = {
        "activation": _tensor_observation(torch, activation),
        "features": _tensor_observation(torch, features),
        "decoded": _tensor_observation(torch, decoded),
    }
    return ProbeExecution(tokens, active, hook, architecture, 1, 1, 1, 1, tensors)


def _observation(
    execution: ProbeExecution,
    artifacts: dict[str, dict[str, str]],
    versions: dict[str, str],
) -> SAEFunctionalObservation:
    return SAEFunctionalObservation(
        artifact_sha256=artifacts,
        runtime_versions=versions,
        input_token_ids=execution.input_token_ids,
        active_adapter=execution.active_adapter,
        resolved_hook_name=execution.resolved_hook_name,
        sae_architecture=execution.sae_architecture,
        model_forward_count=execution.model_forward_count,
        hook_capture_count=execution.hook_capture_count,
        encode_count=execution.encode_count,
        decode_count=execution.decode_count,
        tensors=execution.tensors,
    )


def run_transformers_sae_smoke(
    plan: SAEFunctionalSmokePlan,
    *,
    base_snapshot: Path,
    adapter_snapshot: Path,
    sae_directory: Path,
    execute_probe: ProbeRunner = _execute_probe,
    version_reader: VersionReader = _runtime_versions,
) -> SAEFunctionalObservation:
    """Run one exact load, hook, encode, and decode probe."""
    roots = {"base": base_snapshot, "adapter": adapter_snapshot, "sae": sae_directory}
    before = _artifact_hashes(plan, roots)
    versions = version_reader()
    if versions != plan.runtime.versions:
        raise SAEFunctionalSmokeError("runtime versions do not match the plan")
    execution = execute_probe(plan, base_snapshot, adapter_snapshot, sae_directory)
    after = _artifact_hashes(plan, roots)
    if after != before:
        raise SAEFunctionalSmokeError("artifact bytes changed during functional smoke")
    return _observation(execution, after, versions)
