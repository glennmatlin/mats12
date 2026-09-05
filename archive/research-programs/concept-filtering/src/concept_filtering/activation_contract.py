"""Bounded layer-16 activation contract smoke: config, alignment, and receipt.

This stage proves that one retained example can flow through the frozen
layer-16 hook and the declared aggregation on the base model. It never selects
or scores a feature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.serialization import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    json_loads_unique,
)
from concept_filtering.smoke import SmokeValidationError, verify_executor_revision

_CONFIG_FIELDS = {
    "schema_version",
    "smoke_id",
    "scientific_use",
    "starts_experiment_timer",
    "functional_execution",
    "activation_extraction",
    "feature_selection",
    "intervention_execution",
    "concept_scoring",
    "concept_selected_training",
    "provider_execution",
    "training_execution",
    "base_response_receipt",
    "base_response_receipt_sha256",
    "artifacts",
    "runtime",
    "probe",
    "divergences",
}

_TRUE_CAPABILITIES = ("functional_execution", "activation_extraction")
_FALSE_CAPABILITIES = (
    "scientific_use",
    "starts_experiment_timer",
    "feature_selection",
    "intervention_execution",
    "concept_scoring",
    "concept_selected_training",
    "provider_execution",
    "training_execution",
)

_PROBE_FIELDS = {
    "example_id",
    "hook_name",
    "layer_index",
    "hidden_size",
    "aggregation",
}
_AGGREGATIONS = ("mean-over-response-tokens-v1",)


class ActivationContractError(ValueError):
    """Raised when the activation contract cannot be honoured as declared."""


@dataclass(frozen=True)
class ArtifactSpec:
    repository: str
    revision: str
    path: str
    files: dict[str, str]


@dataclass(frozen=True)
class ProbeSpec:
    example_id: str
    hook_name: str
    layer_index: int
    hidden_size: int
    aggregation: str


@dataclass(frozen=True)
class ActivationContractPlan:
    smoke_id: str
    config_sha256: str
    base: ArtifactSpec
    tokenizer: ArtifactSpec
    runtime_versions: dict[str, str]
    probe: ProbeSpec
    base_response_receipt_path: Path
    base_response_receipt_sha256: str
    divergences: tuple[str, ...]


@dataclass(frozen=True)
class ContractExample:
    """One retained example resolved from the package join and its sample."""

    example_id: str
    prompt_messages: tuple[Mapping[str, str], ...]
    supplied_response: Mapping[str, str]
    base_response_text: str
    recorded_completion_token_count: int
    terminated_by: str
    prompt_messages_sha256: str
    supplied_response_sha256: str
    base_response_sha256: str

    def identity(self) -> dict[str, str]:
        return {
            "example_id": self.example_id,
            "prompt_messages_sha256": self.prompt_messages_sha256,
            "supplied_response_sha256": self.supplied_response_sha256,
            "base_response_sha256": self.base_response_sha256,
        }


@dataclass(frozen=True)
class TokenAlignment:
    prompt_token_count: int
    supplied_span: tuple[int, int]
    base_span: tuple[int, int]
    retokenized_completion_token_count: int
    recorded_completion_token_count: int
    terminated_by: str
    retokenization_matches_recorded_count: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_token_count": self.prompt_token_count,
            "supplied_span": list(self.supplied_span),
            "base_span": list(self.base_span),
            "retokenized_completion_token_count": (
                self.retokenized_completion_token_count
            ),
            "recorded_completion_token_count": self.recorded_completion_token_count,
            "terminated_by": self.terminated_by,
            "retokenization_matches_recorded_count": (
                self.retokenization_matches_recorded_count
            ),
        }


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ActivationContractError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ActivationContractError(f"{field} must be a non-empty string")
    return value


def _positive_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if type(value) is not int or value <= 0:
        raise ActivationContractError(f"{field} must be a positive integer")
    return value


def _exact_fields(payload: dict[str, Any], fields: set[str], label: str) -> None:
    observed = set(payload)
    for name in sorted(observed - fields):
        raise ActivationContractError(f"{label} has an unexpected field {name}")
    for name in sorted(fields - observed):
        raise ActivationContractError(f"{label} is missing field {name}")


def _boundaries(payload: dict[str, Any]) -> None:
    for field in _TRUE_CAPABILITIES:
        if payload.get(field) is not True:
            raise ActivationContractError(f"{field} must be true")
    for field in _FALSE_CAPABILITIES:
        if payload.get(field) is not False:
            raise ActivationContractError(f"{field} must be false")


def _artifact(payload: dict[str, Any], label: str) -> ArtifactSpec:
    spec = _mapping(payload.get(label), f"artifacts.{label}")
    files = _mapping(spec.get("files"), f"artifacts.{label}.files")
    if not files:
        raise ActivationContractError(f"artifacts.{label}.files must not be empty")
    for name, digest in files.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ActivationContractError(
                f"artifacts.{label}.files.{name} must be a SHA-256 digest"
            )
    path = spec.get("path", "")
    if not isinstance(path, str):
        raise ActivationContractError(f"artifacts.{label}.path must be a string")
    return ArtifactSpec(
        repository=_string(spec, "repository"),
        revision=_string(spec, "revision"),
        path=path,
        files=cast(dict[str, str], files),
    )


def _probe(payload: dict[str, Any]) -> ProbeSpec:
    spec = _mapping(payload.get("probe"), "probe")
    _exact_fields(spec, _PROBE_FIELDS, "probe")
    aggregation = _string(spec, "aggregation")
    if aggregation not in _AGGREGATIONS:
        raise ActivationContractError(
            f"probe.aggregation {aggregation!r} is not frozen"
        )
    return ProbeSpec(
        example_id=_string(spec, "example_id"),
        hook_name=_string(spec, "hook_name"),
        layer_index=_positive_int(spec, "layer_index"),
        hidden_size=_positive_int(spec, "hidden_size"),
        aggregation=aggregation,
    )


def _runtime_versions(payload: dict[str, Any]) -> dict[str, str]:
    runtime = _mapping(payload.get("runtime"), "runtime")
    _exact_fields(runtime, {"versions"}, "runtime")
    versions = _mapping(runtime.get("versions"), "runtime.versions")
    if not versions:
        raise ActivationContractError("runtime.versions must not be empty")
    for name, value in versions.items():
        if not isinstance(value, str) or not value.strip():
            raise ActivationContractError(
                f"runtime.versions.{name} must be a non-empty string"
            )
    return cast(dict[str, str], versions)


def _divergences(payload: dict[str, Any]) -> tuple[str, ...]:
    values = payload.get("divergences")
    if not isinstance(values, list) or not values:
        raise ActivationContractError("divergences must be a non-empty array")
    for entry in cast(list[object], values):
        if not isinstance(entry, str) or not entry.strip():
            raise ActivationContractError("each divergence must be a non-empty string")
    return tuple(cast(list[str], values))


def _bound_receipt(config_path: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    raw = _string(payload, "base_response_receipt")
    receipt_path = (config_path.parent / raw).resolve()
    expected = payload.get("base_response_receipt_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ActivationContractError(
            "base_response_receipt_sha256 must be a SHA-256 digest"
        )
    if not receipt_path.is_file():
        raise ActivationContractError(f"base response receipt {raw} is not a file")
    if file_sha256(receipt_path) != expected:
        raise ActivationContractError("base response receipt SHA-256 mismatch")
    return receipt_path, expected


def load_activation_contract(path: Path) -> ActivationContractPlan:
    """Load the smoke config, failing closed on drift or expanded capability."""
    source = path.read_text(encoding="utf-8")
    payload = _mapping(json_loads_unique(source), "config")
    _exact_fields(payload, _CONFIG_FIELDS, "config")
    if payload.get("schema_version") != 1:
        raise ActivationContractError("schema_version must be 1")
    _boundaries(payload)
    artifacts = _mapping(payload.get("artifacts"), "artifacts")
    _exact_fields(artifacts, {"base", "tokenizer"}, "artifacts")
    receipt_path, receipt_sha256 = _bound_receipt(path, payload)
    return ActivationContractPlan(
        smoke_id=_string(payload, "smoke_id"),
        config_sha256=canonical_sha256(payload),
        base=_artifact(artifacts, "base"),
        tokenizer=_artifact(artifacts, "tokenizer"),
        runtime_versions=_runtime_versions(payload),
        probe=_probe(payload),
        base_response_receipt_path=receipt_path,
        base_response_receipt_sha256=receipt_sha256,
        divergences=_divergences(payload),
    )


def aligned_token_segments(
    *,
    prompt_ids: Sequence[int],
    supplied_ids: Sequence[int],
    base_completion_ids: Sequence[int],
    recorded_completion_token_count: int,
    terminated_by: str,
) -> TokenAlignment:
    """Locate comparable response spans against one shared prompt prefix."""
    prompt_length = len(prompt_ids)
    if tuple(supplied_ids[:prompt_length]) != tuple(prompt_ids):
        raise ActivationContractError(
            "supplied rendering does not carry the prompt rendering as a prefix"
        )
    if len(supplied_ids) <= prompt_length or not base_completion_ids:
        raise ActivationContractError("a response span must not be empty")
    expected = recorded_completion_token_count - (1 if terminated_by == "eos" else 0)
    return TokenAlignment(
        prompt_token_count=prompt_length,
        supplied_span=(prompt_length, len(supplied_ids)),
        base_span=(prompt_length, prompt_length + len(base_completion_ids)),
        retokenized_completion_token_count=len(base_completion_ids),
        recorded_completion_token_count=recorded_completion_token_count,
        terminated_by=terminated_by,
        retokenization_matches_recorded_count=len(base_completion_ids) == expected,
    )


@dataclass(frozen=True)
class TensorObservation:
    shape: tuple[int, ...]
    dtype: str
    finite: bool
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "finite": self.finite,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ActivationObservation:
    artifact_sha256: dict[str, dict[str, str]]
    runtime_versions: dict[str, str]
    resolved_hook_name: str
    model_forward_count: int
    hook_capture_count: int
    alignment: TokenAlignment
    example: dict[str, str]
    tensors: dict[str, TensorObservation]


@dataclass(frozen=True)
class ActivationContractReceipt:
    payload: dict[str, object]
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "receipt_sha256": self.receipt_sha256}


def _validate_tensors(
    plan: ActivationContractPlan, tensors: dict[str, TensorObservation]
) -> None:
    for name, tensor in tensors.items():
        if not tensor.finite:
            raise ActivationContractError(f"tensor {name} is not finite")
        if tensor.shape[-1] != plan.probe.hidden_size:
            raise ActivationContractError(
                f"tensor {name} trailing dimension {tensor.shape[-1]} "
                f"does not match declared hidden size {plan.probe.hidden_size}"
            )


def _validated_revision(code_revision: str) -> str:
    if len(code_revision) != 40 or not all(
        character in "0123456789abcdef" for character in code_revision
    ):
        raise ActivationContractError("code_revision must be a 40-hex commit SHA")
    try:
        return verify_executor_revision(code_revision)
    except SmokeValidationError as error:
        raise ActivationContractError(f"executor binding failed: {error}") from error


def _validate_counts(observation: ActivationObservation) -> None:
    if observation.model_forward_count != 2:
        raise ActivationContractError(
            f"expected two forwards, observed {observation.model_forward_count}"
        )
    if observation.hook_capture_count != observation.model_forward_count:
        raise ActivationContractError("hook captures must match model forwards")


def activation_contract_receipt(
    plan: ActivationContractPlan,
    observation: ActivationObservation,
    *,
    code_revision: str,
) -> ActivationContractReceipt:
    """Build the signed receipt body for one bounded activation observation."""
    _validate_tensors(plan, observation.tensors)
    _validate_counts(observation)
    body: dict[str, object] = {
        "schema_version": 1,
        "smoke_id": plan.smoke_id,
        "code_revision": _validated_revision(code_revision),
        "config_sha256": plan.config_sha256,
        "base_response_receipt_sha256": plan.base_response_receipt_sha256,
        "artifacts": observation.artifact_sha256,
        "runtime": {"versions": observation.runtime_versions},
        "probe": {
            "example_id": plan.probe.example_id,
            "hook_name": plan.probe.hook_name,
            "resolved_hook_name": observation.resolved_hook_name,
            "layer_index": plan.probe.layer_index,
            "hidden_size": plan.probe.hidden_size,
            "aggregation": plan.probe.aggregation,
            "operation_counts": {
                "model_forward": observation.model_forward_count,
                "hook_capture": observation.hook_capture_count,
            },
            "alignment": observation.alignment.to_dict(),
            "example": observation.example,
            "tensors": {
                name: tensor.to_dict()
                for name, tensor in sorted(observation.tensors.items())
            },
        },
        "divergences": list(plan.divergences),
        "activation_extraction": True,
        "functional_execution": True,
        "feature_selection": False,
        "intervention_execution": False,
        "concept_scoring": False,
        "concept_selected_training": False,
        "provider_execution": False,
        "training_execution": False,
        "scientific_use": False,
        "starts_experiment_timer": False,
    }
    return ActivationContractReceipt(body, canonical_sha256(body))


def write_activation_receipt(
    receipt: ActivationContractReceipt, output_path: Path
) -> None:
    """Publish the receipt to a newly declared absent path."""
    try:
        with output_path.open("xb") as handle:
            handle.write(canonical_json_bytes(receipt.to_dict()) + b"\n")
    except FileExistsError as error:
        raise ActivationContractError(
            f"refusing to overwrite activation receipt: {output_path}"
        ) from error
    except OSError as error:
        raise ActivationContractError(
            f"cannot publish activation receipt: {error}"
        ) from error


def _sample_payload(output_dir: Path, example_id: str) -> dict[str, Any]:
    from concept_filtering.base_response_runner import read_sample_bytes

    path = output_dir / f"{example_id}.json"
    if not path.is_file():
        raise ActivationContractError(f"no base response sample for {example_id}")
    source = read_sample_bytes(path, f"sample {example_id}").decode("utf-8")
    return _mapping(json_loads_unique(source), f"sample {example_id}")


def resolve_contract_example(
    examples: Sequence[Any], example_id: str, output_dir: Path
) -> ContractExample:
    """Bind one joined example to its published base response by stable ID."""
    matches = [example for example in examples if example.example_id == example_id]
    if len(matches) != 1:
        raise ActivationContractError(
            f"expected one joined example {example_id}, observed {len(matches)}"
        )
    joined = matches[0]
    payload = _sample_payload(output_dir, example_id)
    for field, expected in (
        ("prompt_messages_sha256", joined.prompt_messages_sha256),
        ("supplied_response_sha256", joined.supplied_response_sha256),
    ):
        if payload.get(field) != expected:
            raise ActivationContractError(f"sample {example_id} disagrees on {field}")
    return ContractExample(
        example_id=example_id,
        prompt_messages=joined.prompt_messages,
        supplied_response=joined.supplied_response,
        base_response_text=_string(payload, "response_text"),
        recorded_completion_token_count=_positive_int(
            payload, "completion_token_count"
        ),
        terminated_by=_string(payload, "terminated_by"),
        prompt_messages_sha256=joined.prompt_messages_sha256,
        supplied_response_sha256=joined.supplied_response_sha256,
        base_response_sha256=_string(payload, "response_sha256"),
    )
