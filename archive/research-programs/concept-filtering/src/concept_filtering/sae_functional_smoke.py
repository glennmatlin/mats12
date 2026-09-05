"""Receipt contract for the bounded layer-16 SAE functional smoke."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.organism import OrganismConfig, load_organism_config
from concept_filtering.serialization import (
    bytes_sha256,
    canonical_sha256,
    file_sha256,
)
from concept_filtering.smoke import verify_executor_revision

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "smoke_id",
    "scientific_use",
    "starts_experiment_timer",
    "functional_execution",
    "feature_selection",
    "intervention_execution",
    "provider_execution",
    "organism_config",
    "static_compatibility_receipt",
    "static_compatibility_receipt_sha256",
    "artifacts",
    "runtime",
    "probe",
    "divergences",
}


class SAEFunctionalSmokeError(ValueError):
    """Raised when the functional smoke cannot preserve its frozen boundary."""


@dataclass(frozen=True)
class ArtifactSpec:
    repository: str
    revision: str
    files: dict[str, str]


@dataclass(frozen=True)
class RuntimeSpec:
    device: str
    model_dtype: str
    sae_dtype: str
    versions: dict[str, str]


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ProbeSpec:
    input_text: str
    add_special_tokens: bool
    input_token_ids: tuple[int, ...]
    active_adapter: str
    hook_name: str
    resolved_hook_name: str
    sae_architecture: str
    expected_tensors: dict[str, TensorSpec]


@dataclass(frozen=True)
class SAEFunctionalSmokePlan:
    smoke_id: str
    config_sha256: str
    scientific_use: bool
    starts_experiment_timer: bool
    functional_execution: bool
    feature_selection: bool
    intervention_execution: bool
    provider_execution: bool
    organism: OrganismConfig
    static_receipt_sha256: str
    artifacts: dict[str, ArtifactSpec]
    runtime: RuntimeSpec
    probe: ProbeSpec
    divergences: tuple[str, ...]


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
class SAEFunctionalObservation:
    artifact_sha256: dict[str, dict[str, str]]
    runtime_versions: dict[str, str]
    input_token_ids: tuple[int, ...]
    active_adapter: str
    resolved_hook_name: str
    sae_architecture: str
    model_forward_count: int
    hook_capture_count: int
    encode_count: int
    decode_count: int
    tensors: dict[str, TensorObservation]


@dataclass(frozen=True)
class SAEFunctionalReceipt:
    payload: dict[str, object]
    receipt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "receipt_sha256": self.receipt_sha256}


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SAEFunctionalSmokeError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SAEFunctionalSmokeError(f"{field} must be a non-empty string")
    return value


def _exact_fields(payload: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(payload))
    extra = sorted(set(payload) - fields)
    if missing:
        raise SAEFunctionalSmokeError(
            f"{label} is missing fields: {', '.join(missing)}"
        )
    if extra:
        raise SAEFunctionalSmokeError(
            f"{label} has unexpected fields: {', '.join(extra)}"
        )


def _sha256(payload: dict[str, Any], field: str) -> str:
    value = _string(payload, field)
    if _SHA256.fullmatch(value) is None:
        raise SAEFunctionalSmokeError(f"{field} must be 64 lowercase hex digits")
    return value


def _load_payload(path: Path) -> tuple[dict[str, Any], str]:
    try:
        source = path.read_bytes()
        value: object = json.loads(source.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAEFunctionalSmokeError(
            f"cannot load functional smoke: {error}"
        ) from error
    return _mapping(value, "functional smoke"), bytes_sha256(source)


def _relative_path(config_path: Path, raw: object, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise SAEFunctionalSmokeError(f"{field} must be a relative path")
    relative = Path(raw)
    root = config_path.parent.resolve()
    resolved = (config_path.parent / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root.parent):
        raise SAEFunctionalSmokeError(f"{field} escapes the reconstruction config root")
    return resolved


def _load_bound_inputs(
    path: Path, payload: dict[str, Any]
) -> tuple[OrganismConfig, str]:
    organism_path = _relative_path(
        path, payload.get("organism_config"), "organism_config"
    )
    receipt_path = _relative_path(
        path,
        payload.get("static_compatibility_receipt"),
        "static_compatibility_receipt",
    )
    expected = _sha256(payload, "static_compatibility_receipt_sha256")
    if file_sha256(receipt_path) != expected:
        raise SAEFunctionalSmokeError("static compatibility receipt SHA-256 mismatch")
    return load_organism_config(organism_path), expected


def _parse_files(payload: dict[str, Any], label: str) -> dict[str, str]:
    values = _mapping(payload.get("files"), f"{label}.files")
    if not values:
        raise SAEFunctionalSmokeError(f"{label}.files must not be empty")
    files: dict[str, str] = {}
    for name, value in values.items():
        if Path(name).name != name:
            raise SAEFunctionalSmokeError(f"{label}.files has an invalid name")
        files[name] = _sha256({"sha256": value}, "sha256")
    return files


def _parse_artifacts(payload: dict[str, Any]) -> dict[str, ArtifactSpec]:
    values = _mapping(payload.get("artifacts"), "artifacts")
    _exact_fields(values, {"base", "adapter", "sae"}, "artifacts")
    artifacts: dict[str, ArtifactSpec] = {}
    for role in ("base", "adapter", "sae"):
        value = _mapping(values[role], f"artifacts.{role}")
        _exact_fields(value, {"repository", "revision", "files"}, f"artifacts.{role}")
        artifacts[role] = ArtifactSpec(
            _string(value, "repository"),
            _string(value, "revision"),
            _parse_files(value, f"artifacts.{role}"),
        )
    return artifacts


def _bind_artifacts(
    organism: OrganismConfig, artifacts: dict[str, ArtifactSpec]
) -> None:
    artifact_ids = {
        "base": "base-model",
        "adapter": "full-data-adapter",
        "sae": "sae-layer-16",
    }
    for role, artifact_id in artifact_ids.items():
        matches = [
            item for item in organism.artifacts if item.artifact_id == artifact_id
        ]
        if len(matches) != 1:
            raise SAEFunctionalSmokeError(f"organism must contain one {artifact_id}")
        receipt = matches[0]
        spec = artifacts[role]
        if (receipt.repository, receipt.revision) != (spec.repository, spec.revision):
            raise SAEFunctionalSmokeError(f"{role} artifact does not match organism")
        if receipt.sha256 is not None and receipt.sha256 not in spec.files.values():
            raise SAEFunctionalSmokeError(
                f"{role} artifact SHA-256 does not match organism"
            )


def _parse_runtime(payload: dict[str, Any]) -> RuntimeSpec:
    value = _mapping(payload.get("runtime"), "runtime")
    _exact_fields(value, {"device", "model_dtype", "sae_dtype", "versions"}, "runtime")
    versions = _mapping(value["versions"], "runtime.versions")
    if not versions or any(
        not isinstance(item, str) or not item for item in versions.values()
    ):
        raise SAEFunctionalSmokeError("runtime.versions must contain exact strings")
    device = _string(value, "device")
    model_dtype = _string(value, "model_dtype")
    sae_dtype = _string(value, "sae_dtype")
    if device not in {"cpu", "cuda", "mps"}:
        raise SAEFunctionalSmokeError("runtime.device is unsupported")
    if model_dtype not in {"bfloat16", "float16", "float32"}:
        raise SAEFunctionalSmokeError("runtime.model_dtype is unsupported")
    if sae_dtype != "float32":
        raise SAEFunctionalSmokeError("runtime.sae_dtype must be float32")
    return RuntimeSpec(
        device,
        model_dtype,
        sae_dtype,
        cast(dict[str, str], versions),
    )


def _parse_tensor(value: object, field: str) -> TensorSpec:
    payload = _mapping(value, field)
    _exact_fields(payload, {"shape", "dtype"}, field)
    raw_shape = payload["shape"]
    if not isinstance(raw_shape, list) or not raw_shape:
        raise SAEFunctionalSmokeError(f"{field}.shape must be a non-empty list")
    shape_values = cast(list[object], raw_shape)
    if any(type(item) is not int or item < 1 for item in shape_values):
        raise SAEFunctionalSmokeError(f"{field}.shape must contain positive integers")
    return TensorSpec(tuple(cast(list[int], shape_values)), _string(payload, "dtype"))


def _parse_probe_tokens(value: dict[str, Any]) -> tuple[bool, tuple[int, ...]]:
    tokens = value["input_token_ids"]
    if not isinstance(tokens, list) or not tokens:
        raise SAEFunctionalSmokeError("probe.input_token_ids must contain integers")
    token_values = cast(list[object], tokens)
    if any(type(item) is not int for item in token_values):
        raise SAEFunctionalSmokeError("probe.input_token_ids must contain integers")
    add_special_tokens = value["add_special_tokens"]
    if type(add_special_tokens) is not bool:
        raise SAEFunctionalSmokeError("probe.add_special_tokens must be boolean")
    return add_special_tokens, tuple(cast(list[int], token_values))


def _parse_probe(payload: dict[str, Any]) -> ProbeSpec:
    value = _mapping(payload.get("probe"), "probe")
    fields = {
        "input_text",
        "add_special_tokens",
        "input_token_ids",
        "active_adapter",
        "hook_name",
        "resolved_hook_name",
        "sae_architecture",
        "expected_tensors",
    }
    _exact_fields(value, fields, "probe")
    add_special_tokens, tokens = _parse_probe_tokens(value)
    tensors = _mapping(value["expected_tensors"], "probe.expected_tensors")
    _exact_fields(
        tensors, {"activation", "features", "decoded"}, "probe.expected_tensors"
    )
    return ProbeSpec(
        _string(value, "input_text"),
        add_special_tokens,
        tokens,
        _string(value, "active_adapter"),
        _string(value, "hook_name"),
        _string(value, "resolved_hook_name"),
        _string(value, "sae_architecture"),
        {name: _parse_tensor(tensors[name], name) for name in tensors},
    )


def _false_boundary(payload: dict[str, Any], field: str) -> bool:
    if payload.get(field) is not False:
        raise SAEFunctionalSmokeError(f"{field} must be false")
    return False


def _parse_divergences(payload: dict[str, Any]) -> tuple[str, ...]:
    value = payload.get("divergences")
    if not isinstance(value, list) or not value:
        raise SAEFunctionalSmokeError("divergences must be a non-empty list")
    items = cast(list[object], value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise SAEFunctionalSmokeError("divergences must contain non-empty strings")
    return tuple(cast(list[str], items))


def load_sae_functional_smoke(path: Path) -> SAEFunctionalSmokePlan:
    """Load the exact non-scientific functional-smoke contract."""
    payload, config_sha256 = _load_payload(path)
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "functional smoke")
    if (
        payload.get("schema_version") != 1
        or payload.get("functional_execution") is not True
    ):
        raise SAEFunctionalSmokeError(
            "schema_version must be 1 and functional_execution true"
        )
    organism, receipt_sha256 = _load_bound_inputs(path, payload)
    artifacts = _parse_artifacts(payload)
    _bind_artifacts(organism, artifacts)
    return SAEFunctionalSmokePlan(
        _string(payload, "smoke_id"),
        config_sha256,
        _false_boundary(payload, "scientific_use"),
        _false_boundary(payload, "starts_experiment_timer"),
        True,
        _false_boundary(payload, "feature_selection"),
        _false_boundary(payload, "intervention_execution"),
        _false_boundary(payload, "provider_execution"),
        organism,
        receipt_sha256,
        artifacts,
        _parse_runtime(payload),
        _parse_probe(payload),
        _parse_divergences(payload),
    )


def _validate_artifact_hashes(
    plan: SAEFunctionalSmokePlan, observation: SAEFunctionalObservation
) -> None:
    expected = {role: spec.files for role, spec in plan.artifacts.items()}
    if observation.artifact_sha256 != expected:
        raise SAEFunctionalSmokeError("runtime artifact SHA-256 values do not match")


def _validate_tensor(
    name: str, expected: TensorSpec, observed: TensorObservation
) -> None:
    if (observed.shape, observed.dtype) != (expected.shape, expected.dtype):
        raise SAEFunctionalSmokeError(f"{name} tensor shape or dtype does not match")
    if observed.finite is not True:
        raise SAEFunctionalSmokeError(f"{name} tensor contains non-finite values")
    if _SHA256.fullmatch(observed.sha256) is None:
        raise SAEFunctionalSmokeError(f"{name} tensor SHA-256 is malformed")


def _validate_probe_identity(
    plan: SAEFunctionalSmokePlan, observation: SAEFunctionalObservation
) -> None:
    actual = (
        observation.input_token_ids,
        observation.active_adapter,
        observation.resolved_hook_name,
        observation.sae_architecture,
    )
    expected = (
        plan.probe.input_token_ids,
        plan.probe.active_adapter,
        plan.probe.resolved_hook_name,
        plan.probe.sae_architecture,
    )
    if actual != expected:
        raise SAEFunctionalSmokeError("runtime probe identity does not match the plan")


def _validate_operation_counts(observation: SAEFunctionalObservation) -> None:
    counts = (
        observation.model_forward_count,
        observation.hook_capture_count,
        observation.encode_count,
        observation.decode_count,
    )
    if counts != (1, 1, 1, 1):
        raise SAEFunctionalSmokeError(
            "functional smoke operations must run exactly once"
        )


def _validate_observation(
    plan: SAEFunctionalSmokePlan, observation: SAEFunctionalObservation
) -> None:
    _validate_artifact_hashes(plan, observation)
    if observation.runtime_versions != plan.runtime.versions:
        raise SAEFunctionalSmokeError("runtime versions do not match the plan")
    _validate_probe_identity(plan, observation)
    _validate_operation_counts(observation)
    if set(observation.tensors) != set(plan.probe.expected_tensors):
        raise SAEFunctionalSmokeError("runtime tensor roles do not match the plan")
    for name, expected in plan.probe.expected_tensors.items():
        _validate_tensor(name, expected, observation.tensors[name])


def _artifact_payload(plan: SAEFunctionalSmokePlan) -> dict[str, object]:
    return {
        role: {
            "repository": artifact.repository,
            "revision": artifact.revision,
            "file_sha256": dict(sorted(artifact.files.items())),
        }
        for role, artifact in sorted(plan.artifacts.items())
    }


def _probe_payload(
    plan: SAEFunctionalSmokePlan, observation: SAEFunctionalObservation
) -> dict[str, object]:
    return {
        "input_text_sha256": canonical_sha256(plan.probe.input_text),
        "input_token_ids": list(observation.input_token_ids),
        "add_special_tokens": plan.probe.add_special_tokens,
        "active_adapter": observation.active_adapter,
        "hook_name": plan.probe.hook_name,
        "resolved_hook_name": observation.resolved_hook_name,
        "sae_architecture": observation.sae_architecture,
        "operation_counts": {
            "model_forward": observation.model_forward_count,
            "hook_capture": observation.hook_capture_count,
            "encode": observation.encode_count,
            "decode": observation.decode_count,
        },
        "tensors": {
            name: value.to_dict() for name, value in sorted(observation.tensors.items())
        },
    }


def _receipt_payload(
    plan: SAEFunctionalSmokePlan,
    code_revision: str,
    observation: SAEFunctionalObservation,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "smoke_id": plan.smoke_id,
        "scientific_use": plan.scientific_use,
        "starts_experiment_timer": plan.starts_experiment_timer,
        "functional_execution": plan.functional_execution,
        "feature_selection": plan.feature_selection,
        "intervention_execution": plan.intervention_execution,
        "provider_execution": plan.provider_execution,
        "code_revision": code_revision,
        "config_sha256": plan.config_sha256,
        "organism_id": plan.organism.organism_id,
        "organism_sha256": plan.organism.organism_sha256,
        "static_compatibility_receipt_sha256": plan.static_receipt_sha256,
        "artifacts": _artifact_payload(plan),
        "runtime": {
            "device": plan.runtime.device,
            "model_dtype": plan.runtime.model_dtype,
            "sae_dtype": plan.runtime.sae_dtype,
            "versions": dict(sorted(observation.runtime_versions.items())),
        },
        "probe": _probe_payload(plan, observation),
        "divergences": list(plan.divergences),
    }


def _write_receipt(receipt: SAEFunctionalReceipt, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as target:
            target.write(json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n")
    except FileExistsError as error:
        raise SAEFunctionalSmokeError(
            f"refusing to overwrite output: {output_path}"
        ) from error


def execute_sae_functional_smoke(
    plan: SAEFunctionalSmokePlan,
    *,
    code_revision: str,
    output_path: Path,
    run: Callable[[SAEFunctionalSmokePlan], SAEFunctionalObservation],
) -> SAEFunctionalReceipt:
    """Execute one bounded functional probe and write its exclusive receipt."""
    verified_revision = verify_executor_revision(code_revision)
    if output_path.exists():
        raise SAEFunctionalSmokeError(f"refusing to overwrite output: {output_path}")
    observation = run(plan)
    _validate_observation(plan, observation)
    payload = _receipt_payload(plan, verified_revision, observation)
    receipt = SAEFunctionalReceipt(payload, canonical_sha256(payload))
    _write_receipt(receipt, output_path)
    return receipt
