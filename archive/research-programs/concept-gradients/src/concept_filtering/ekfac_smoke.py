"""Receipt contract for the deterministic local EKFAC backend smoke."""

from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from concept_filtering.causal_gradient_plan import load_causal_gradient_plan
from concept_filtering.ekfac_reconstruction import (
    EKFACReconstruction,
    load_ekfac_reconstruction,
)
from concept_filtering.isolated_io import (
    IsolatedIOError,
    prepare_plain_parent_directories,
    publish_new_bytes,
    read_isolated_bytes,
    verify_plain_parent_directories,
)
from concept_filtering.serialization import canonical_sha256, json_loads_unique
from concept_filtering.smoke import verify_executor_revision

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_TOP_FIELDS = {
    "schema_version",
    "smoke_id",
    "phase",
    "implementation",
    "bindings",
    "fixture",
    "factor_arguments",
    "score_arguments",
    "runtime",
    "determinism",
    "boundaries",
    "divergences",
}
_BINDING_FIELDS = {
    "reconstruction_config_sha256",
    "reconstruction_receipt_sha256",
    "attribution_plan_sha256",
}
_BOUNDARY_VALUES: dict[str, bool] = {
    "local_backend_smoke_execution": True,
    "scientific_use": False,
    "starts_experiment_timer": False,
    "sae_feature_selection_allowed": False,
    "concept_scoring_allowed": False,
    "provider_execution_allowed": False,
    "scheduler_execution_allowed": False,
    "concept_selected_training_allowed": False,
}
_IMPLEMENTATION: dict[str, object] = {
    "package": "kronfluence",
    "version": "1.0.1",
    "source_revision": "87685e90d6972df314eff635440a78eac54ae47c",
}
_RUNTIME: dict[str, object] = {
    "device": "cpu",
    "versions": {
        "kronfluence": "1.0.1",
        "safetensors": "0.8.0",
        "torch": "2.13.0",
    },
}
_FACTOR_ARGUMENTS: dict[str, object] = {
    "strategy": "ekfac",
    "use_empirical_fisher": True,
    "per_device_batch_size": 2,
    "activation_covariance_dtype": "float64",
    "gradient_covariance_dtype": "float64",
    "eigendecomposition_dtype": "float64",
    "per_sample_gradient_dtype": "float64",
    "lambda_dtype": "float64",
}
_SCORE_ARGUMENTS: dict[str, object] = {
    "damping_factor": 0.001,
    "per_device_query_batch_size": 1,
    "per_device_train_batch_size": 2,
    "query_gradient_accumulation_steps": 1,
    "query_gradient_low_rank": None,
    "use_full_svd": True,
    "query_gradient_svd_dtype": "float64",
    "per_sample_gradient_dtype": "float64",
    "precondition_dtype": "float64",
    "score_dtype": "float64",
}


class EKFACSmokeError(ValueError):
    """Raised when the local EKFAC smoke violates its frozen contract."""


@dataclass(frozen=True)
class EKFACSmokePlan:
    smoke_id: str
    config_path: Path
    config_sha256: str
    implementation_version: str
    implementation_source_revision: str
    reconstruction_config_sha256: str
    reconstruction_receipt_sha256: str
    attribution_plan_sha256: str
    fixture: dict[str, object]
    factor_arguments: dict[str, object]
    score_arguments: dict[str, object]
    runtime_versions: dict[str, str]
    repeat_count: int
    absolute_tolerance: float
    expected_score_vectors: dict[str, tuple[float, ...]]
    local_backend_smoke_execution: bool
    scientific_use: bool
    concept_scoring_allowed: bool
    scheduler_execution_allowed: bool
    divergences: tuple[str, ...]

    @property
    def fixture_sha256(self) -> str:
        return canonical_sha256(self.fixture)

    @property
    def expected_score_vector(self) -> tuple[float, ...]:
        return self.expected_score_vectors["behavior_loss"]


@dataclass(frozen=True)
class EKFACQueryObservation:
    objective_kind: str
    factor_store_semantic_sha256: str
    training_gradient_sha256: str
    score_store_semantic_sha256: str
    query_gradient_sha256: str
    inverse_application_sha256: str
    score_vector: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "objective_kind": self.objective_kind,
            "factor_store_semantic_sha256": self.factor_store_semantic_sha256,
            "training_gradient_sha256": self.training_gradient_sha256,
            "score_store_semantic_sha256": self.score_store_semantic_sha256,
            "query_gradient_sha256": self.query_gradient_sha256,
            "inverse_application_sha256": self.inverse_application_sha256,
            "score_vector": list(self.score_vector),
        }


@dataclass(frozen=True)
class EKFACBackendObservation:
    implementation: str
    runtime_versions: dict[str, str]
    fixture_sha256: str
    model_sha256: str
    training_data_sha256: str
    query_data_sha256: str
    query_observations: tuple[EKFACQueryObservation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "runtime_versions": dict(sorted(self.runtime_versions.items())),
            "fixture_sha256": self.fixture_sha256,
            "model_sha256": self.model_sha256,
            "training_data_sha256": self.training_data_sha256,
            "query_data_sha256": self.query_data_sha256,
            "query_observations": [item.to_dict() for item in self.query_observations],
        }


@dataclass(frozen=True)
class EKFACSmokeReceipt:
    payload: dict[str, object]
    receipt_sha256: str

    @property
    def output_release(self) -> str:
        releases = cast(dict[str, object], self.payload["releases"])
        return cast(str, releases["output_release"])

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "receipt_sha256": self.receipt_sha256}


BackendRunner = Callable[[EKFACSmokePlan, Path], EKFACBackendObservation]


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EKFACSmokeError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _fields(value: dict[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise EKFACSmokeError(
            f"{label} unexpected fields={unexpected}, missing fields={missing}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EKFACSmokeError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if _SHA256.fullmatch(result) is None:
        raise EKFACSmokeError(f"{label} must be a lowercase SHA-256")
    return result


def _load_payload(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw = read_isolated_bytes(path, "EKFAC smoke")
        parsed = json_loads_unique(raw.decode("utf-8"))
        payload = _object(parsed, "EKFAC smoke")
        identity = canonical_sha256(payload)
    except (
        IsolatedIOError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        RecursionError,
    ) as error:
        raise EKFACSmokeError(f"cannot load EKFAC smoke: {error}") from error
    _fields(payload, _TOP_FIELDS, "EKFAC smoke")
    return payload, identity


def _validate_envelope(payload: dict[str, object]) -> None:
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise EKFACSmokeError("schema_version must be the integer 1")
    _string(payload["smoke_id"], "smoke_id")
    if payload["phase"] != "pre_feature":
        raise EKFACSmokeError("phase must remain pre_feature")
    if _object(payload["implementation"], "implementation") != _IMPLEMENTATION:
        raise EKFACSmokeError("implementation identity differs")


def _bindings(value: object) -> tuple[str, str, str]:
    payload = _object(value, "bindings")
    _fields(payload, _BINDING_FIELDS, "bindings")
    values = tuple(_sha256(payload[field], field) for field in sorted(payload))
    if len(set(values)) != len(values):
        raise EKFACSmokeError("binding hash roles must remain distinct")
    return (
        _sha256(payload["reconstruction_config_sha256"], "reconstruction config"),
        _sha256(payload["reconstruction_receipt_sha256"], "reconstruction receipt"),
        _sha256(payload["attribution_plan_sha256"], "attribution plan"),
    )


def _exact_object(value: object, expected: Mapping[str, object], label: str) -> None:
    if _object(value, label) != expected:
        raise EKFACSmokeError(f"{label} differs from the frozen fixture")


def _fixture(value: object) -> dict[str, object]:
    fixture = _object(value, "fixture")
    _fields(
        fixture,
        {"fixture_id", "model", "training_dataset", "query_dataset"},
        "fixture",
    )
    if fixture["fixture_id"] != "tiny-classification-ekfac-v1":
        raise EKFACSmokeError("fixture_id differs")
    _validate_fixture_shapes(fixture)
    return fixture


def _matrix(value: object, rows: int, columns: int, label: str) -> None:
    if not isinstance(value, list):
        raise EKFACSmokeError(f"{label} shape differs")
    matrix = cast(list[object], value)
    if len(matrix) != rows:
        raise EKFACSmokeError(f"{label} shape differs")
    for row in matrix:
        if not isinstance(row, list):
            raise EKFACSmokeError(f"{label} shape differs")
        row_values = cast(list[object], row)
        if len(row_values) != columns:
            raise EKFACSmokeError(f"{label} shape differs")
        if any(not _finite_number(item) for item in row_values):
            raise EKFACSmokeError(f"{label} must contain finite numbers")


def _vector(value: object, size: int, label: str) -> None:
    if not isinstance(value, list):
        raise EKFACSmokeError(f"{label} shape differs")
    values = cast(list[object], value)
    if len(values) != size:
        raise EKFACSmokeError(f"{label} shape differs")
    if any(not _finite_number(item) for item in values):
        raise EKFACSmokeError(f"{label} must contain finite numbers")


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_model(value: object) -> None:
    model = _object(value, "fixture.model")
    expected = {
        "architecture",
        "dtype",
        "first_weight",
        "first_bias",
        "second_weight",
        "second_bias",
        "tracked_modules",
    }
    _fields(model, expected, "fixture.model")
    if (model["architecture"], model["dtype"], model["tracked_modules"]) != (
        "linear-tanh-linear",
        "float64",
        ["0", "2"],
    ):
        raise EKFACSmokeError("fixture.model identity differs")
    _matrix(model["first_weight"], 4, 3, "first_weight")
    _vector(model["first_bias"], 4, "first_bias")
    _matrix(model["second_weight"], 2, 4, "second_weight")
    _vector(model["second_bias"], 2, "second_bias")


def _validate_dataset(value: object, size: int, label: str) -> None:
    dataset = _object(value, label)
    _fields(dataset, {"inputs", "labels"}, label)
    _matrix(dataset["inputs"], size, 3, f"{label}.inputs")
    labels = dataset["labels"]
    if not isinstance(labels, list):
        raise EKFACSmokeError(f"{label}.labels shape differs")
    label_values = cast(list[object], labels)
    if len(label_values) != size:
        raise EKFACSmokeError(f"{label}.labels shape differs")
    if any(type(item) is not int or item not in {0, 1} for item in label_values):
        raise EKFACSmokeError(f"{label}.labels must contain binary integers")


def _validate_fixture_shapes(fixture: dict[str, object]) -> None:
    _validate_model(fixture["model"])
    _validate_dataset(fixture["training_dataset"], 4, "training_dataset")
    _validate_dataset(fixture["query_dataset"], 1, "query_dataset")


def _runtime(value: object) -> dict[str, str]:
    payload = _object(value, "runtime")
    if payload != _RUNTIME:
        raise EKFACSmokeError("runtime differs from the frozen fixture")
    return cast(dict[str, str], payload["versions"])


def _determinism(
    value: object,
) -> tuple[int, float, dict[str, tuple[float, ...]]]:
    payload = _object(value, "determinism")
    _fields(
        payload,
        {"repeat_count", "absolute_tolerance", "expected_score_vectors"},
        "determinism",
    )
    tolerance = payload["absolute_tolerance"]
    scores = _object(payload["expected_score_vectors"], "expected_score_vectors")
    _fields(scores, {"behavior_loss", "synthetic_scalar"}, "expected_score_vectors")
    if payload["repeat_count"] != 2 or type(payload["repeat_count"]) is not int:
        raise EKFACSmokeError("repeat_count must be the integer 2")
    tolerance_value = _number(tolerance, "absolute_tolerance")
    if not 0 < tolerance_value <= 1e-6:
        raise EKFACSmokeError("absolute_tolerance must be in (0, 1e-6]")
    for kind, vector in scores.items():
        _vector(vector, 4, f"expected_score_vectors.{kind}")
    expected = {
        kind: tuple(
            _number(item, "expected score") for item in cast(list[object], vector)
        )
        for kind, vector in scores.items()
    }
    return 2, tolerance_value, expected


def _number(value: object, label: str) -> float:
    if not _finite_number(value):
        raise EKFACSmokeError(f"{label} must be finite")
    return float(cast(int | float, value))


def _boundaries(value: object) -> dict[str, bool]:
    payload = _object(value, "boundaries")
    _fields(payload, set(_BOUNDARY_VALUES), "boundaries")
    for field, expected in _BOUNDARY_VALUES.items():
        if payload[field] is not expected:
            raise EKFACSmokeError(f"{field} must be {expected}")
    return cast(dict[str, bool], payload)


def _divergences(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise EKFACSmokeError("divergences must contain non-empty strings")
    items = cast(list[object], value)
    if any(not isinstance(item, str) or not item for item in items):
        raise EKFACSmokeError("divergences must contain non-empty strings")
    return tuple(cast(list[str], items))


def load_ekfac_smoke_plan(path: Path) -> EKFACSmokePlan:
    payload, config_sha256 = _load_payload(path)
    _validate_envelope(payload)
    reconstruction, receipt, attribution = _bindings(payload["bindings"])
    _exact_object(payload["factor_arguments"], _FACTOR_ARGUMENTS, "factor_arguments")
    _exact_object(payload["score_arguments"], _SCORE_ARGUMENTS, "score_arguments")
    repeat, tolerance, scores = _determinism(payload["determinism"])
    boundaries = _boundaries(payload["boundaries"])
    return EKFACSmokePlan(
        _string(payload["smoke_id"], "smoke_id"),
        path.absolute(),
        config_sha256,
        "1.0.1",
        _string(_IMPLEMENTATION["source_revision"], "implementation source revision"),
        reconstruction,
        receipt,
        attribution,
        _fixture(payload["fixture"]),
        dict(_FACTOR_ARGUMENTS),
        dict(_SCORE_ARGUMENTS),
        _runtime(payload["runtime"]),
        repeat,
        tolerance,
        scores,
        boundaries["local_backend_smoke_execution"],
        boundaries["scientific_use"],
        boundaries["concept_scoring_allowed"],
        boundaries["scheduler_execution_allowed"],
        _divergences(payload["divergences"]),
    )


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        raw = read_isolated_bytes(path, label)
        value = json_loads_unique(raw.decode("utf-8"))
    except (
        IsolatedIOError,
        OSError,
        UnicodeError,
        ValueError,
        RecursionError,
    ) as error:
        raise EKFACSmokeError(f"cannot load {label}: {error}") from error
    return _object(value, label)


def _verify_reconstruction_receipt(
    plan: EKFACSmokePlan, path: Path
) -> dict[str, object]:
    payload = _load_json_object(path, "reconstruction receipt")
    declared = _sha256(
        payload.get("receipt_sha256"), "reconstruction receipt receipt_sha256"
    )
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        declared != canonical_sha256(body)
        or declared != plan.reconstruction_receipt_sha256
    ):
        raise EKFACSmokeError("reconstruction receipt identity differs")
    expected = (
        body.get("config_sha256"),
        body.get("implementation"),
        body.get("implementation_version"),
        body.get("implementation_source_revision"),
        body.get("scientific_use"),
        body.get("concept_scoring_allowed"),
    )
    if expected != _reconstruction_contract(plan):
        raise EKFACSmokeError("reconstruction receipt contract differs")
    return body


def _reconstruction_contract(plan: EKFACSmokePlan) -> tuple[object, ...]:
    return (
        plan.reconstruction_config_sha256,
        "kronfluence",
        "1.0.1",
        plan.implementation_source_revision,
        False,
        False,
    )


def _backend_config_hashes(
    reconstruction: EKFACReconstruction,
) -> tuple[str, str, str]:
    decisions = {item.name: item.value for item in reconstruction.decisions}
    parameter = decisions["parameter_and_layer_coverage"]
    factor = {
        "factor_sample": decisions["factor_sample"],
        "damping": decisions["damping"],
    }
    training = {
        "parameter_and_layer_coverage": parameter,
        "sequence_policy": decisions["sequence_policy"],
    }
    return (
        canonical_sha256(parameter),
        canonical_sha256(factor),
        canonical_sha256(training),
    )


def _verify_attribution_plan(
    plan: EKFACSmokePlan, path: Path, reconstruction: EKFACReconstruction
) -> None:
    loaded = load_causal_gradient_plan(path)
    if loaded.plan_sha256 != plan.attribution_plan_sha256:
        raise EKFACSmokeError("attribution plan identity differs")
    expected = (loaded.backend_id, loaded.implementation_version)
    if expected != ("public-rank32-kronfluence-reconstruction-v1", "1.0.1"):
        raise EKFACSmokeError("attribution backend identity differs")
    observed_hashes = (
        loaded.parameter_selection_sha256,
        loaded.factor_config_sha256,
        loaded.training_gradient_config_sha256,
    )
    if observed_hashes != _backend_config_hashes(reconstruction):
        raise EKFACSmokeError("attribution backend config differs")


def _verify_bound_inputs(
    plan: EKFACSmokePlan,
    paths: tuple[Path, Path, Path],
) -> None:
    reconstruction_config, reconstruction_receipt, attribution_plan = paths
    reconstruction = load_ekfac_reconstruction(reconstruction_config)
    if reconstruction.config_sha256 != plan.reconstruction_config_sha256:
        raise EKFACSmokeError("reconstruction config identity differs")
    _verify_reconstruction_receipt(plan, reconstruction_receipt)
    _verify_attribution_plan(plan, attribution_plan, reconstruction)


def _validate_observation(
    plan: EKFACSmokePlan, observation: EKFACBackendObservation
) -> None:
    if observation.implementation != "kronfluence":
        raise EKFACSmokeError("backend implementation differs")
    if observation.runtime_versions != plan.runtime_versions:
        raise EKFACSmokeError("backend runtime versions differ")
    if observation.fixture_sha256 != plan.fixture_sha256:
        raise EKFACSmokeError("backend fixture identity differs")
    queries = observation.query_observations
    kinds = tuple(item.objective_kind for item in queries)
    if kinds != ("behavior_loss", "synthetic_scalar"):
        raise EKFACSmokeError("backend query objective order differs")
    hash_values = _observation_hashes(observation)
    if any(_SHA256.fullmatch(value) is None for value in hash_values):
        raise EKFACSmokeError("backend observation contains a malformed SHA-256")
    factors = {item.factor_store_semantic_sha256 for item in queries}
    gradients = {item.training_gradient_sha256 for item in queries}
    if len(factors) != 1 or len(gradients) != 1:
        raise EKFACSmokeError("backend queries must reuse factors and train gradients")
    if len({item.query_gradient_sha256 for item in queries}) != len(queries):
        raise EKFACSmokeError("backend query gradients must remain distinct")
    if len(set(_distinct_role_hashes(observation))) != 12:
        raise EKFACSmokeError("backend observation hash roles must remain distinct")
    for query in queries:
        _validate_scores(plan, query)


def _observation_hashes(observation: EKFACBackendObservation) -> tuple[str, ...]:
    shared = (
        observation.fixture_sha256,
        observation.model_sha256,
        observation.training_data_sha256,
        observation.query_data_sha256,
    )
    query_hashes = tuple(
        value
        for query in observation.query_observations
        for value in (
            query.factor_store_semantic_sha256,
            query.training_gradient_sha256,
            query.score_store_semantic_sha256,
            query.query_gradient_sha256,
            query.inverse_application_sha256,
        )
    )
    return shared + query_hashes


def _distinct_role_hashes(observation: EKFACBackendObservation) -> tuple[str, ...]:
    first, second = observation.query_observations
    return (
        observation.fixture_sha256,
        observation.model_sha256,
        observation.training_data_sha256,
        observation.query_data_sha256,
        first.factor_store_semantic_sha256,
        first.training_gradient_sha256,
        first.score_store_semantic_sha256,
        first.query_gradient_sha256,
        first.inverse_application_sha256,
        second.score_store_semantic_sha256,
        second.query_gradient_sha256,
        second.inverse_application_sha256,
    )


def _validate_scores(plan: EKFACSmokePlan, query: EKFACQueryObservation) -> None:
    scores = query.score_vector
    expected = plan.expected_score_vectors[query.objective_kind]
    if len(scores) != len(expected):
        raise EKFACSmokeError("backend score count differs")
    if any(not math.isfinite(value) for value in scores):
        raise EKFACSmokeError("backend scores must be finite")
    differences = [
        abs(observed - target)
        for observed, target in zip(scores, expected, strict=True)
    ]
    if max(differences) > plan.absolute_tolerance:
        raise EKFACSmokeError("backend expected score tolerance exceeded")


def _compare_observations(
    plan: EKFACSmokePlan,
    first: EKFACBackendObservation,
    second: EKFACBackendObservation,
) -> float:
    first_payload, first_scores = _without_scores(first)
    second_payload, second_scores = _without_scores(second)
    if first_payload != second_payload:
        raise EKFACSmokeError("backend observation is not deterministic")
    differences = [
        abs(left - right)
        for left_vector, right_vector in zip(first_scores, second_scores, strict=True)
        for left, right in zip(left_vector, right_vector, strict=True)
    ]
    maximum = max(differences)
    if maximum > plan.absolute_tolerance:
        raise EKFACSmokeError("backend score vector is not deterministic")
    return maximum


def _without_scores(
    observation: EKFACBackendObservation,
) -> tuple[dict[str, object], tuple[tuple[float, ...], ...]]:
    payload = observation.to_dict()
    queries = cast(list[dict[str, object]], payload["query_observations"])
    scores = tuple(
        tuple(cast(list[float], item.pop("score_vector"))) for item in queries
    )
    return payload, scores


def _output_release(
    plan: EKFACSmokePlan, code_revision: str, observation: EKFACBackendObservation
) -> str:
    return canonical_sha256(
        {
            "executor_release": code_revision,
            "coordinator_release": code_revision,
            "input_artifact_identities": {
                "reconstruction_receipt": plan.reconstruction_receipt_sha256,
                "attribution_plan": plan.attribution_plan_sha256,
            },
            "config_sha256": plan.config_sha256,
            "result_sha256": canonical_sha256(observation.to_dict()),
        }
    )


def _receipt_payload(
    plan: EKFACSmokePlan,
    code_revision: str,
    observation: EKFACBackendObservation,
    maximum_difference: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "smoke_id": plan.smoke_id,
        "config_sha256": plan.config_sha256,
        **_receipt_boundaries(),
        "bindings": _binding_payload(plan),
        "releases": _release_payload(plan, code_revision, observation),
        "observation": observation.to_dict(),
        "determinism": {
            "repeat_count": plan.repeat_count,
            "absolute_tolerance": plan.absolute_tolerance,
            "maximum_observed_score_difference": maximum_difference,
        },
        "divergences": list(plan.divergences),
    }


def _receipt_boundaries() -> dict[str, bool]:
    return dict(_BOUNDARY_VALUES)


def _binding_payload(plan: EKFACSmokePlan) -> dict[str, str]:
    return {
        "reconstruction_config_sha256": plan.reconstruction_config_sha256,
        "reconstruction_receipt_sha256": plan.reconstruction_receipt_sha256,
        "attribution_plan_sha256": plan.attribution_plan_sha256,
    }


def _release_payload(
    plan: EKFACSmokePlan,
    code_revision: str,
    observation: EKFACBackendObservation,
) -> dict[str, object]:
    return {
        "executor_release": code_revision,
        "coordinator_release": code_revision,
        "input_artifact_identities": {
            "reconstruction_receipt": plan.reconstruction_receipt_sha256,
            "attribution_plan": plan.attribution_plan_sha256,
        },
        "output_release": _output_release(plan, code_revision, observation),
    }


def _write_receipt(
    receipt: EKFACSmokeReceipt,
    output_path: Path,
    output_parents: tuple[tuple[int, int], ...],
) -> None:
    try:
        value = json.dumps(receipt.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
        publish_new_bytes(
            output_path,
            value,
            "EKFAC smoke receipt",
            expected_parents=output_parents,
        )
    except IsolatedIOError as error:
        raise EKFACSmokeError(str(error)) from error


def _prepare_run_directory(
    run_dir: Path, output_path: Path
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    if run_dir.exists():
        raise EKFACSmokeError(f"refusing to reuse run directory: {run_dir}")
    if output_path.exists():
        raise EKFACSmokeError(f"refusing to overwrite output: {output_path}")
    try:
        parents = prepare_plain_parent_directories(
            run_dir / ".ekfac-run-root", "EKFAC run directory"
        )
        output_parents = prepare_plain_parent_directories(
            output_path, "EKFAC smoke receipt"
        )
    except IsolatedIOError as error:
        raise EKFACSmokeError(str(error)) from error
    return parents, output_parents


def _recheck_execution_context(
    plan: EKFACSmokePlan,
    inputs: tuple[Path, Path, Path],
    revision: str,
    run_dir: Path,
    run_parents: tuple[tuple[int, int], ...],
    output_path: Path,
    output_parents: tuple[tuple[int, int], ...],
) -> None:
    _validate_plan_identity(plan)
    _verify_bound_inputs(plan, inputs)
    try:
        verify_plain_parent_directories(
            run_dir / ".ekfac-run-root", run_parents, "EKFAC run directory"
        )
        verify_plain_parent_directories(
            output_path, output_parents, "EKFAC smoke receipt"
        )
    except IsolatedIOError as error:
        raise EKFACSmokeError(str(error)) from error
    if verify_executor_revision(revision) != revision:
        raise EKFACSmokeError("executor revision changed during EKFAC smoke")


def _validate_plan_identity(plan: EKFACSmokePlan) -> None:
    reloaded = load_ekfac_smoke_plan(plan.config_path)
    if reloaded != plan:
        raise EKFACSmokeError("EKFAC smoke differs from its canonical file")


def _load_backend_runner() -> BackendRunner:
    try:
        module = importlib.import_module("concept_filtering.ekfac_kronfluence")
        runner = module.run_kronfluence_ekfac_smoke
    except (ImportError, AttributeError) as error:
        raise EKFACSmokeError(
            "EKFAC smoke dependencies are unavailable; install the attribution extra"
        ) from error
    if not callable(runner):
        raise EKFACSmokeError("the installed EKFAC backend entry point is invalid")
    return cast(BackendRunner, runner)


def _run_backend(
    plan: EKFACSmokePlan, run: BackendRunner, run_dir: Path
) -> tuple[EKFACBackendObservation, float]:
    observations = tuple(
        run(plan, run_dir / f"run-{index}") for index in range(1, plan.repeat_count + 1)
    )
    for observation in observations:
        _validate_observation(plan, observation)
    maximum = _compare_observations(plan, observations[0], observations[1])
    return observations[0], maximum


def execute_ekfac_backend_smoke(
    plan: EKFACSmokePlan,
    *,
    reconstruction_config: Path,
    reconstruction_receipt: Path,
    attribution_plan: Path,
    code_revision: str,
    run_dir: Path,
    output_path: Path,
) -> EKFACSmokeReceipt:
    _validate_plan_identity(plan)
    revision = verify_executor_revision(code_revision)
    inputs = (reconstruction_config, reconstruction_receipt, attribution_plan)
    _verify_bound_inputs(plan, inputs)
    run = _load_backend_runner()
    run_parents, output_parents = _prepare_run_directory(run_dir, output_path)
    observation, maximum = _run_backend(plan, run, run_dir)
    _recheck_execution_context(
        plan,
        inputs,
        revision,
        run_dir,
        run_parents,
        output_path,
        output_parents,
    )
    payload = _receipt_payload(plan, revision, observation, maximum)
    receipt = EKFACSmokeReceipt(payload, canonical_sha256(payload))
    _write_receipt(receipt, output_path, output_parents)
    return receipt
