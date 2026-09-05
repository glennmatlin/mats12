"""Fail-closed contract for the bounded real rank-32 EKFAC smoke."""

from __future__ import annotations

import importlib
import json
import math
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import cast

from concept_filtering.isolated_io import (
    IsolatedIOError,
    prepare_plain_parent_directories,
    publish_new_bytes,
    read_isolated_bytes,
    sha256_isolated_file,
    verify_plain_parent_directories,
)
from concept_filtering.serialization import (
    bytes_sha256,
    canonical_sha256,
    json_loads_unique,
)
from concept_filtering.smoke import verify_executor_revision

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_SHA256 = "fb83c621ad0d3c08698f3496ab3c6610b72c58ebd666ab1acb50d794bf677a40"
_TOP_FIELDS = {
    "schema_version",
    "smoke_id",
    "phase",
    "implementation",
    "bindings",
    "artifacts",
    "topology",
    "training_example",
    "query_example",
    "factor_arguments",
    "score_arguments",
    "runtime",
    "determinism",
    "boundaries",
    "divergences",
}
_BINDING_ROLES = {
    "organism_config",
    "ekfac_reconstruction_config",
    "ekfac_reconstruction_receipt",
    "attribution_plan",
    "software_smoke_receipt",
    "training_data_config",
    "training_data_receipt",
    "training_data_package",
    "training_sample",
    "generation_evidence",
    "prompt_bank",
    "prompt_bank_receipt",
    "generation_sample",
}
_TRACKED_MODULES = (
    "base_model.model.model.layers.0.self_attn.k_proj.lora_A.default",
    "base_model.model.model.layers.0.self_attn.k_proj.lora_B.default",
)
_TRACKED_SHAPES = (
    (_TRACKED_MODULES[0], (32, 4096)),
    (_TRACKED_MODULES[1], (4096, 32)),
)
_PROJECTIONS = (
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)
_FACTOR_ARGUMENTS: dict[str, object] = {
    "strategy": "ekfac",
    "fisher": "true_fisher_sampled_from_model_predictions",
    "use_empirical_fisher": False,
    "seed": 42,
    "per_device_batch_size": 1,
    "covariance_max_examples": 1,
    "lambda_max_examples": 1,
    "covariance_data_partitions": 1,
    "covariance_module_partitions": 1,
    "lambda_data_partitions": 1,
    "lambda_module_partitions": 1,
    "amp_dtype": "bfloat16",
    "activation_covariance_dtype": "bfloat16",
    "gradient_covariance_dtype": "bfloat16",
    "eigendecomposition_dtype": "float64",
    "per_sample_gradient_dtype": "bfloat16",
    "lambda_dtype": "bfloat16",
    "use_iterative_lambda_aggregation": True,
    "offload_activations_to_cpu": True,
}
_SCORE_ARGUMENTS: dict[str, object] = {
    "damping_factor": None,
    "per_device_query_batch_size": 1,
    "per_device_train_batch_size": 1,
    "data_partitions": 1,
    "module_partitions": 1,
    "query_gradient_accumulation_steps": 10,
    "query_gradient_low_rank": 64,
    "use_full_svd": True,
    "aggregate_query_gradients": False,
    "aggregate_train_gradients": False,
    "query_gradient_svd_dtype": "float32",
    "per_sample_gradient_dtype": "float32",
    "precondition_dtype": "float32",
    "score_dtype": "bfloat16",
    "offload_activations_to_cpu": True,
}
_RUNTIME: dict[str, object] = {
    "device": "cpu",
    "model_dtype": "bfloat16",
    "versions": {
        "huggingface-hub": "0.36.2",
        "kronfluence": "1.0.1",
        "peft": "0.18.1",
        "pyarrow": "23.0.1",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
        "torch": "2.13.0",
        "transformers": "4.57.6",
    },
}
_BOUNDARIES: dict[str, bool] = {
    "bounded_rank32_smoke_execution": True,
    "scientific_use": False,
    "starts_experiment_timer": False,
    "sae_feature_selection_allowed": False,
    "concept_scoring_allowed": False,
    "provider_execution_allowed": False,
    "scheduler_execution_allowed": False,
    "concept_selected_training_allowed": False,
    "training_execution_allowed": False,
}


class Rank32EKFACSmokeError(ValueError):
    """Raised when the bounded rank-32 smoke contract is inconsistent."""


@dataclass(frozen=True)
class ArtifactBinding:
    path: str
    file_sha256: str
    semantic_sha256: str


@dataclass(frozen=True)
class SnapshotBinding:
    repository: str
    revision: str
    files: tuple[tuple[str, str], ...]

    @property
    def file_hashes(self) -> dict[str, str]:
        return dict(self.files)

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(
            {
                "repository": self.repository,
                "revision": self.revision,
                "files": self.file_hashes,
            }
        )


@dataclass(frozen=True)
class Rank32EKFACSmokePlan:
    smoke_id: str
    config_path: Path
    config_sha256: str
    bindings: Mapping[str, ArtifactBinding]
    artifacts: Mapping[str, SnapshotBinding]
    tracked_modules: tuple[str, ...]
    expected_adapter_linear_modules: int
    training_example: Mapping[str, object]
    query_example: Mapping[str, object]
    factor_arguments: Mapping[str, object]
    score_arguments: Mapping[str, object]
    runtime: Mapping[str, object]
    repeat_count: int
    absolute_tolerance: float
    bounded_rank32_smoke_execution: bool
    scientific_use: bool
    divergences: tuple[str, ...]

    @property
    def artifact_manifest_sha256(self) -> str:
        payload = {
            role: {
                "repository": artifact.repository,
                "revision": artifact.revision,
                "files": artifact.file_hashes,
            }
            for role, artifact in sorted(self.artifacts.items())
        }
        return canonical_sha256(payload)

    @property
    def topology_sha256(self) -> str:
        return canonical_sha256(
            {
                "expected_adapter_linear_modules": self.expected_adapter_linear_modules,
                "tracked_modules": list(self.tracked_modules),
                "tracked_module_shapes": {
                    name: list(shape) for name, shape in _TRACKED_SHAPES
                },
            }
        )

    @property
    def training_input_sha256(self) -> str:
        return _input_identity(self.training_example)

    @property
    def query_input_sha256(self) -> str:
        return _input_identity(self.query_example)


@dataclass(frozen=True)
class Rank32RuntimePaths:
    repository_root: Path
    base_snapshot: Path
    adapter_snapshot: Path
    training_sample: Path
    training_package: Path


@dataclass(frozen=True)
class Rank32EKFACObservation:
    runtime_versions: tuple[tuple[str, str], ...]
    artifact_manifest_sha256: str
    topology_sha256: str
    tracked_parameter_sha256: str
    tracked_module_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    training_input_sha256: str
    query_input_sha256: str
    factor_store_semantic_sha256: str
    training_gradient_sha256: str
    query_gradient_sha256: str
    score_store_semantic_sha256: str
    inverse_application_sha256: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_versions": dict(self.runtime_versions),
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "topology_sha256": self.topology_sha256,
            "tracked_parameter_sha256": self.tracked_parameter_sha256,
            "tracked_module_shapes": {
                name: list(shape) for name, shape in self.tracked_module_shapes
            },
            "training_input_sha256": self.training_input_sha256,
            "query_input_sha256": self.query_input_sha256,
            "factor_store_semantic_sha256": self.factor_store_semantic_sha256,
            "training_gradient_sha256": self.training_gradient_sha256,
            "query_gradient_sha256": self.query_gradient_sha256,
            "score_store_semantic_sha256": self.score_store_semantic_sha256,
            "inverse_application_sha256": self.inverse_application_sha256,
            "score": self.score,
        }


@dataclass(frozen=True)
class Rank32EKFACSmokeReceipt:
    payload: Mapping[str, object]
    receipt_sha256: str

    @property
    def output_release(self) -> str:
        releases = cast(Mapping[str, object], self.payload["releases"])
        return cast(str, releases["output_release"])

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "receipt_sha256": self.receipt_sha256}


Rank32BackendRunner = Callable[
    [Rank32EKFACSmokePlan, Rank32RuntimePaths, Path], Rank32EKFACObservation
]


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Rank32EKFACSmokeError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _fields(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise Rank32EKFACSmokeError(
            f"{label} unexpected fields={unexpected}, missing fields={missing}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Rank32EKFACSmokeError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256.fullmatch(result) is None:
        raise Rank32EKFACSmokeError(f"{label} must be a lowercase SHA-256")
    return result


def _input_identity(example: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            "rendered_input_ids_sha256": example["rendered_input_ids_sha256"],
            "labels_sha256": example["labels_sha256"],
            "input_token_count": example["input_token_count"],
            "supervised_token_count": example["supervised_token_count"],
        }
    )


def _relative_path(value: object, label: str) -> str:
    result = _string(value, label)
    parsed = PurePosixPath(result)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != result
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise Rank32EKFACSmokeError(f"{label} must be a canonical relative path")
    return result


def _load_payload(path: Path) -> dict[str, object]:
    try:
        raw = read_isolated_bytes(path, "rank-32 EKFAC smoke config")
        value = json_loads_unique(raw.decode("utf-8"))
    except (IsolatedIOError, UnicodeError, ValueError, RecursionError) as error:
        raise Rank32EKFACSmokeError(str(error)) from error
    return _object(value, "rank-32 EKFAC smoke config")


def _validate_identity(payload: dict[str, object]) -> None:
    _fields(payload, _TOP_FIELDS, "rank-32 EKFAC smoke config")
    if payload["schema_version"] != 1 or payload["phase"] != "pre_feature":
        raise Rank32EKFACSmokeError("rank-32 smoke identity differs")
    if payload["smoke_id"] != "public-rank32-ekfac-bounded-smoke-v1":
        raise Rank32EKFACSmokeError("rank-32 smoke identity differs")


def _validate_implementation(value: object) -> None:
    payload = _object(value, "implementation")
    expected = {
        "package": "kronfluence",
        "version": "1.0.1",
        "source_repository": "https://github.com/pomonam/kronfluence",
        "source_revision": "87685e90d6972df314eff635440a78eac54ae47c",
    }
    if payload != expected:
        raise Rank32EKFACSmokeError("implementation differs from reconstruction")


def _binding(value: object, label: str) -> ArtifactBinding:
    payload = _object(value, label)
    _fields(payload, {"path", "file_sha256", "semantic_sha256"}, label)
    return ArtifactBinding(
        _relative_path(payload["path"], f"{label}.path"),
        _sha256(payload["file_sha256"], f"{label}.file_sha256"),
        _sha256(payload["semantic_sha256"], f"{label}.semantic_sha256"),
    )


def _bindings(value: object) -> Mapping[str, ArtifactBinding]:
    payload = _object(value, "bindings")
    _fields(payload, _BINDING_ROLES, "bindings")
    parsed = {role: _binding(payload[role], role) for role in sorted(payload)}
    return MappingProxyType(parsed)


def _snapshot(value: object, label: str) -> SnapshotBinding:
    payload = _object(value, label)
    _fields(payload, {"repository", "revision", "files"}, label)
    revision = _string(payload["revision"], f"{label}.revision")
    if _REVISION.fullmatch(revision) is None:
        raise Rank32EKFACSmokeError(f"{label}.revision must be a commit SHA")
    files = _object(payload["files"], f"{label}.files")
    if not files:
        raise Rank32EKFACSmokeError(f"{label}.files cannot be empty")
    hashes = tuple(
        (name, _sha256(value, name)) for name, value in sorted(files.items())
    )
    return SnapshotBinding(_string(payload["repository"], label), revision, hashes)


def _artifacts(value: object) -> Mapping[str, SnapshotBinding]:
    payload = _object(value, "artifacts")
    _fields(payload, {"base", "adapter"}, "artifacts")
    parsed = {role: _snapshot(payload[role], role) for role in ("base", "adapter")}
    return MappingProxyType(parsed)


def _topology(value: object) -> tuple[tuple[str, ...], int]:
    payload = _object(value, "topology")
    expected = {
        "layer_count": 32,
        "projections": list(_PROJECTIONS),
        "lora_factors": ["lora_A.default", "lora_B.default"],
        "expected_adapter_linear_modules": 448,
        "tracked_modules": list(_TRACKED_MODULES),
        "tracked_module_shapes": {name: list(shape) for name, shape in _TRACKED_SHAPES},
    }
    if payload != expected:
        raise Rank32EKFACSmokeError("topology.tracked_modules or coverage differs")
    return _TRACKED_MODULES, 448


def _example(value: object, expected: set[str], label: str) -> Mapping[str, object]:
    payload = _object(value, label)
    _fields(payload, expected, label)
    for field in ("rendered_input_ids_sha256", "labels_sha256"):
        _sha256(payload[field], f"{label}.{field}")
    return MappingProxyType(payload)


def _training_example(value: object) -> Mapping[str, object]:
    fields = {
        "row_ordinal",
        "orig_idx",
        "example_id",
        "source_record_sha256",
        "rendered_input_ids_sha256",
        "labels_sha256",
        "input_token_count",
        "supervised_token_count",
    }
    result = _example(value, fields, "training_example")
    if result["row_ordinal"] != 17_763 or result["orig_idx"] != 51_158:
        raise Rank32EKFACSmokeError("training_example identity differs")
    return result


def _query_example(value: object) -> Mapping[str, object]:
    fields = {
        "behavior",
        "index",
        "question_id",
        "prompt_sha256",
        "sample_sha256",
        "completion_sha256",
        "rendered_input_ids_sha256",
        "labels_sha256",
        "input_token_count",
        "supervised_token_count",
    }
    result = _example(value, fields, "query_example")
    if result["behavior"] != "validate_feelings" or result["index"] != 82:
        raise Rank32EKFACSmokeError("query_example identity differs")
    return result


def _exact_mapping(
    value: object, expected: dict[str, object], label: str
) -> Mapping[str, object]:
    payload = _object(value, label)
    if payload != expected:
        raise Rank32EKFACSmokeError(f"{label} differs from the bounded profile")
    return MappingProxyType(payload)


def _runtime(value: object) -> Mapping[str, object]:
    payload = _object(value, "runtime")
    if payload != _RUNTIME:
        raise Rank32EKFACSmokeError("runtime differs from the bounded profile")
    frozen = dict(payload)
    frozen["versions"] = MappingProxyType(
        dict(_object(payload["versions"], "versions"))
    )
    return MappingProxyType(frozen)


def _determinism(value: object) -> tuple[int, float]:
    payload = _object(value, "determinism")
    expected = {
        "repeat_count": 2,
        "absolute_tolerance": 0.00001,
        "require_exact_non_score_identities": True,
    }
    if payload != expected:
        raise Rank32EKFACSmokeError("determinism differs from the bounded profile")
    return 2, 0.00001


def _boundaries(value: object) -> Mapping[str, bool]:
    payload = _object(value, "boundaries")
    _fields(payload, set(_BOUNDARIES), "boundaries")
    for field, expected in _BOUNDARIES.items():
        if payload[field] is not expected:
            raise Rank32EKFACSmokeError(f"boundaries.{field} must be {expected}")
    return MappingProxyType(cast(dict[str, bool], payload))


def _divergences(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Rank32EKFACSmokeError("divergences must name every bounded departure")
    items = cast(list[object], value)
    if len(items) < 4:
        raise Rank32EKFACSmokeError("divergences must name every bounded departure")
    values = tuple(_string(item, "divergence") for item in items)
    if len(set(values)) != len(values):
        raise Rank32EKFACSmokeError("divergences must be unique")
    return values


def _profiles(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    factor = _exact_mapping(
        payload["factor_arguments"], _FACTOR_ARGUMENTS, "factor_arguments"
    )
    score = _exact_mapping(
        payload["score_arguments"], _SCORE_ARGUMENTS, "score_arguments"
    )
    return factor, score


def _frozen_profiles(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    profiles = _profiles(payload)
    if canonical_sha256(payload) != _CONFIG_SHA256:
        raise Rank32EKFACSmokeError("rank-32 smoke canonical identity differs")
    return profiles


def load_rank32_ekfac_smoke_plan(path: Path) -> Rank32EKFACSmokePlan:
    """Load the immutable bounded rank-32 EKFAC smoke plan."""
    payload = _load_payload(path)
    _validate_identity(payload)
    _validate_implementation(payload["implementation"])
    tracked, expected_count = _topology(payload["topology"])
    repeat_count, tolerance = _determinism(payload["determinism"])
    boundaries = _boundaries(payload["boundaries"])
    factor_arguments, score_arguments = _frozen_profiles(payload)
    return Rank32EKFACSmokePlan(
        smoke_id=cast(str, payload["smoke_id"]),
        config_path=path,
        config_sha256=canonical_sha256(payload),
        bindings=_bindings(payload["bindings"]),
        artifacts=_artifacts(payload["artifacts"]),
        tracked_modules=tracked,
        expected_adapter_linear_modules=expected_count,
        training_example=_training_example(payload["training_example"]),
        query_example=_query_example(payload["query_example"]),
        factor_arguments=factor_arguments,
        score_arguments=score_arguments,
        runtime=_runtime(payload["runtime"]),
        repeat_count=repeat_count,
        absolute_tolerance=tolerance,
        bounded_rank32_smoke_execution=boundaries["bounded_rank32_smoke_execution"],
        scientific_use=boundaries["scientific_use"],
        divergences=_divergences(payload["divergences"]),
    )


def _bound_path(
    plan: Rank32EKFACSmokePlan, paths: Rank32RuntimePaths, role: str
) -> Path:
    if role == "training_sample":
        return paths.training_sample
    if role == "training_data_package":
        return paths.training_package
    return paths.repository_root / plan.bindings[role].path


def _signed_json_identity(value: bytes, field: str, label: str) -> str:
    try:
        payload = _object(json_loads_unique(value.decode("utf-8")), label)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise Rank32EKFACSmokeError(f"cannot parse {label}: {error}") from error
    claimed = _sha256(payload.get(field), f"{label}.{field}")
    body = {name: item for name, item in payload.items() if name != field}
    if canonical_sha256(body) != claimed:
        raise Rank32EKFACSmokeError(f"{label} semantic identity differs")
    return claimed


def _semantic_identity(role: str, value: bytes, file_sha256: str) -> str:
    if role in {"training_data_package", "training_sample", "prompt_bank"}:
        return file_sha256
    if role in {
        "organism_config",
        "ekfac_reconstruction_config",
        "attribution_plan",
        "training_data_config",
    }:
        try:
            return canonical_sha256(json_loads_unique(value.decode("utf-8")))
        except (UnicodeError, ValueError, RecursionError) as error:
            raise Rank32EKFACSmokeError(f"cannot parse {role}: {error}") from error
    field = "evidence_sha256" if role == "generation_evidence" else "receipt_sha256"
    if role == "generation_sample":
        field = "sample_sha256"
    return _signed_json_identity(value, field, role)


def _verify_binding(
    plan: Rank32EKFACSmokePlan, paths: Rank32RuntimePaths, role: str
) -> tuple[str, str]:
    binding = plan.bindings[role]
    path = _bound_path(plan, paths, role)
    try:
        if role in {"training_data_package", "training_sample", "prompt_bank"}:
            observed = sha256_isolated_file(path, role)
            semantic = observed
        else:
            value = read_isolated_bytes(path, role)
            observed = bytes_sha256(value)
            semantic = _semantic_identity(role, value, observed)
    except IsolatedIOError as error:
        raise Rank32EKFACSmokeError(str(error)) from error
    if observed != binding.file_sha256:
        raise Rank32EKFACSmokeError(f"{role} file SHA-256 differs")
    if semantic != binding.semantic_sha256:
        raise Rank32EKFACSmokeError(f"{role} semantic SHA-256 differs")
    return role, observed


def _verify_snapshot_entries(
    role: str, snapshot: Path, artifact: SnapshotBinding
) -> None:
    allowed = {name for name, _ in artifact.files}
    try:
        observed = {entry.name for entry in snapshot.iterdir()}
    except OSError as error:
        raise Rank32EKFACSmokeError(
            f"cannot enumerate {role} snapshot directory"
        ) from error
    unexpected = sorted(observed - allowed)
    missing = sorted(allowed - observed)
    if unexpected or missing:
        raise Rank32EKFACSmokeError(
            f"{role} snapshot has unexpected loader files={unexpected}, "
            f"missing files={missing}"
        )


def _verify_snapshot(
    role: str, snapshot: Path, artifact: SnapshotBinding
) -> tuple[str, str]:
    _verify_snapshot_entries(role, snapshot, artifact)
    observed: dict[str, str] = {}
    for name, expected in artifact.files:
        try:
            digest = sha256_isolated_file(snapshot / name, f"{role}.{name}")
        except IsolatedIOError as error:
            raise Rank32EKFACSmokeError(str(error)) from error
        if digest != expected:
            raise Rank32EKFACSmokeError(f"{role}.{name} SHA-256 differs")
        observed[name] = digest
    return role, canonical_sha256(observed)


def _verify_runtime_inputs(
    plan: Rank32EKFACSmokePlan, paths: Rank32RuntimePaths
) -> str:
    bindings = dict(
        _verify_binding(plan, paths, role) for role in sorted(plan.bindings)
    )
    snapshots = dict(
        (
            _verify_snapshot("base", paths.base_snapshot, plan.artifacts["base"]),
            _verify_snapshot(
                "adapter", paths.adapter_snapshot, plan.artifacts["adapter"]
            ),
        )
    )
    return canonical_sha256({"bindings": bindings, "snapshots": snapshots})


def _validate_plan_identity(plan: Rank32EKFACSmokePlan) -> None:
    if load_rank32_ekfac_smoke_plan(plan.config_path) != plan:
        raise Rank32EKFACSmokeError("rank-32 smoke differs from its canonical file")


def _observation_hashes(observation: Rank32EKFACObservation) -> tuple[str, ...]:
    return (
        observation.artifact_manifest_sha256,
        observation.topology_sha256,
        observation.tracked_parameter_sha256,
        observation.training_input_sha256,
        observation.query_input_sha256,
        observation.factor_store_semantic_sha256,
        observation.training_gradient_sha256,
        observation.query_gradient_sha256,
        observation.score_store_semantic_sha256,
        observation.inverse_application_sha256,
    )


def _validate_shapes(observation: Rank32EKFACObservation) -> None:
    if observation.tracked_module_shapes != _TRACKED_SHAPES:
        raise Rank32EKFACSmokeError("tracked module shapes differ")


def _validate_observation(
    plan: Rank32EKFACSmokePlan, observation: Rank32EKFACObservation
) -> None:
    versions = cast(Mapping[str, str], plan.runtime["versions"])
    if observation.runtime_versions != tuple(sorted(versions.items())):
        raise Rank32EKFACSmokeError("rank-32 runtime versions differ")
    expected = (
        plan.artifact_manifest_sha256,
        plan.topology_sha256,
        plan.training_input_sha256,
        plan.query_input_sha256,
    )
    observed = (
        observation.artifact_manifest_sha256,
        observation.topology_sha256,
        observation.training_input_sha256,
        observation.query_input_sha256,
    )
    if observed != expected:
        raise Rank32EKFACSmokeError("rank-32 observation bindings differ")
    if any(
        _SHA256.fullmatch(value) is None for value in _observation_hashes(observation)
    ):
        raise Rank32EKFACSmokeError("rank-32 observation hash is invalid")
    if len(set(_observation_hashes(observation))) != 10:
        raise Rank32EKFACSmokeError("rank-32 observation hash roles must differ")
    if type(observation.score) is not float or not math.isfinite(observation.score):
        raise Rank32EKFACSmokeError("rank-32 EKFAC score must be finite")
    _validate_shapes(observation)


def _without_score_derived(
    observation: Rank32EKFACObservation,
) -> dict[str, object]:
    payload = observation.to_dict()
    payload.pop("score")
    return payload


def _compare_observations(
    plan: Rank32EKFACSmokePlan,
    first: Rank32EKFACObservation,
    second: Rank32EKFACObservation,
) -> float:
    if _without_score_derived(first) != _without_score_derived(second):
        raise Rank32EKFACSmokeError(
            "rank-32 non-score observation is not deterministic"
        )
    difference = abs(first.score - second.score)
    if difference > plan.absolute_tolerance:
        raise Rank32EKFACSmokeError("rank-32 score determinism tolerance exceeded")
    return difference


def _load_backend_runner() -> Rank32BackendRunner:
    try:
        module = importlib.import_module("concept_filtering.ekfac_rank32_kronfluence")
        runner = module.run_rank32_kronfluence_smoke
    except (ImportError, AttributeError) as error:
        raise Rank32EKFACSmokeError(
            "rank-32 EKFAC dependencies are unavailable; install inference "
            "and attribution extras"
        ) from error
    if not callable(runner):
        raise Rank32EKFACSmokeError("rank-32 EKFAC backend entry point is invalid")
    return cast(Rank32BackendRunner, runner)


def _prepare_outputs(
    run_dir: Path, output_path: Path
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    if run_dir.exists() or output_path.exists():
        raise Rank32EKFACSmokeError("refusing to reuse rank-32 smoke outputs")
    try:
        run_parents = prepare_plain_parent_directories(
            run_dir / ".rank32-ekfac-root", "rank-32 EKFAC run directory"
        )
        output_parents = prepare_plain_parent_directories(
            output_path, "rank-32 EKFAC receipt"
        )
    except IsolatedIOError as error:
        raise Rank32EKFACSmokeError(str(error)) from error
    return run_parents, output_parents


def _copy_snapshot_file(source: Path, destination: Path, label: str) -> None:
    try:
        with open(source, "rb") as reader, open(destination, "xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
    except OSError as error:
        raise Rank32EKFACSmokeError(f"cannot stage {label}") from error


def _stage_snapshot(
    role: str, source: Path, artifact: SnapshotBinding, destination: Path
) -> Path:
    try:
        # Creating the marker's parents creates the destination directory itself.
        prepare_plain_parent_directories(
            destination / ".stage-root", f"{role} staged snapshot"
        )
    except IsolatedIOError as error:
        raise Rank32EKFACSmokeError(str(error)) from error
    for name, _ in artifact.files:
        _copy_snapshot_file(source / name, destination / name, f"{role}.{name}")
    _verify_snapshot(role, destination, artifact)
    return destination


def _stage_runtime_paths(
    plan: Rank32EKFACSmokePlan, paths: Rank32RuntimePaths, run_dir: Path
) -> Rank32RuntimePaths:
    staged = run_dir / "staged-inputs"
    return replace(
        paths,
        base_snapshot=_stage_snapshot(
            "base", paths.base_snapshot, plan.artifacts["base"], staged / "base"
        ),
        adapter_snapshot=_stage_snapshot(
            "adapter",
            paths.adapter_snapshot,
            plan.artifacts["adapter"],
            staged / "adapter",
        ),
    )


def _run_backend(
    plan: Rank32EKFACSmokePlan,
    paths: Rank32RuntimePaths,
    run_dir: Path,
) -> tuple[Rank32EKFACObservation, float]:
    run = _load_backend_runner()
    staged = _stage_runtime_paths(plan, paths, run_dir)
    observations = tuple(
        run(plan, staged, run_dir / f"run-{index}")
        for index in range(1, plan.repeat_count + 1)
    )
    for observation in observations:
        _validate_observation(plan, observation)
    return observations[0], _compare_observations(plan, *observations)


def _output_release(
    plan: Rank32EKFACSmokePlan,
    revision: str,
    input_manifest_sha256: str,
    observation: Rank32EKFACObservation,
) -> str:
    return canonical_sha256(
        {
            "executor_release": revision,
            "coordinator_release": revision,
            "config_sha256": plan.config_sha256,
            "input_manifest_sha256": input_manifest_sha256,
            "result_sha256": canonical_sha256(observation.to_dict()),
        }
    )


def _receipt_payload(
    plan: Rank32EKFACSmokePlan,
    revision: str,
    input_manifest_sha256: str,
    observation: Rank32EKFACObservation,
    maximum_difference: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "smoke_id": plan.smoke_id,
        "config_sha256": plan.config_sha256,
        **_BOUNDARIES,
        "input_manifest_sha256": input_manifest_sha256,
        "releases": _release_payload(
            plan, revision, input_manifest_sha256, observation
        ),
        "observation": observation.to_dict(),
        "determinism": {
            "repeat_count": plan.repeat_count,
            "absolute_tolerance": plan.absolute_tolerance,
            "maximum_observed_score_difference": maximum_difference,
        },
        "divergences": list(plan.divergences),
    }


def _release_payload(
    plan: Rank32EKFACSmokePlan,
    revision: str,
    input_manifest_sha256: str,
    observation: Rank32EKFACObservation,
) -> dict[str, object]:
    identities = {
        role: binding.semantic_sha256 for role, binding in plan.bindings.items()
    }
    identities.update(
        {
            f"{role}_snapshot": artifact.semantic_sha256
            for role, artifact in plan.artifacts.items()
        }
    )
    return {
        "executor_release": revision,
        "coordinator_release": revision,
        "input_artifact_identities": identities,
        "output_release": _output_release(
            plan, revision, input_manifest_sha256, observation
        ),
    }


def _recheck_context(
    plan: Rank32EKFACSmokePlan,
    paths: Rank32RuntimePaths,
    revision: str,
    expected_manifest: str,
    run_marker: Path,
    run_parents: tuple[tuple[int, int], ...],
    output_path: Path,
    output_parents: tuple[tuple[int, int], ...],
) -> None:
    _validate_plan_identity(plan)
    if _verify_runtime_inputs(plan, paths) != expected_manifest:
        raise Rank32EKFACSmokeError("rank-32 runtime inputs changed")
    if verify_executor_revision(revision) != revision:
        raise Rank32EKFACSmokeError("rank-32 executor revision changed")
    try:
        verify_plain_parent_directories(run_marker, run_parents, "rank-32 run")
        verify_plain_parent_directories(output_path, output_parents, "rank-32 receipt")
    except IsolatedIOError as error:
        raise Rank32EKFACSmokeError(str(error)) from error


def _publish_receipt(
    receipt: Rank32EKFACSmokeReceipt,
    output_path: Path,
    output_parents: tuple[tuple[int, int], ...],
) -> None:
    value = json.dumps(receipt.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
    try:
        publish_new_bytes(
            output_path,
            value,
            "rank-32 EKFAC receipt",
            expected_parents=output_parents,
        )
    except IsolatedIOError as error:
        raise Rank32EKFACSmokeError(str(error)) from error


def execute_rank32_ekfac_smoke(
    plan: Rank32EKFACSmokePlan,
    *,
    paths: Rank32RuntimePaths,
    code_revision: str,
    run_dir: Path,
    output_path: Path,
) -> Rank32EKFACSmokeReceipt:
    """Execute two exact bounded rank-32 EKFAC runs and publish one receipt."""
    _validate_plan_identity(plan)
    revision = verify_executor_revision(code_revision)
    input_manifest = _verify_runtime_inputs(plan, paths)
    run_parents, output_parents = _prepare_outputs(run_dir, output_path)
    observation, maximum = _run_backend(plan, paths, run_dir)
    _recheck_context(
        plan,
        paths,
        revision,
        input_manifest,
        run_dir / ".rank32-ekfac-root",
        run_parents,
        output_path,
        output_parents,
    )
    payload = _receipt_payload(plan, revision, input_manifest, observation, maximum)
    receipt = Rank32EKFACSmokeReceipt(
        MappingProxyType(payload), canonical_sha256(payload)
    )
    _publish_receipt(receipt, output_path, output_parents)
    return receipt
