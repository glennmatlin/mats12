"""Pre-feature causal-gradient attribution plan summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from concept_filtering.isolated_io import IsolatedIOError, read_isolated_bytes
from concept_filtering.serialization import canonical_sha256, json_loads_unique


class CausalGradientPlanError(ValueError):
    """Raised when a causal-gradient plan violates its frozen contract."""


_PRE_FEATURE_BOUNDARIES = {
    "scientific_use",
    "sae_feature_selection_allowed",
    "concept_scoring_allowed",
    "provider_execution_allowed",
    "scheduler_execution_allowed",
    "concept_selected_training_allowed",
}
_PLAN_FIELDS = {
    "schema_version",
    "plan_id",
    "phase",
    "method",
    "artifacts",
    "backend",
    "objectives",
    "boundaries",
}
_ARTIFACT_FIELDS = {
    "organism_sha256",
    "training_manifest_sha256",
    "training_recipe_sha256",
}
_BACKEND_FIELDS = {
    "backend_id",
    "implementation",
    "implementation_version",
    "factor_store_id",
    "training_gradient_store_id",
    "parameter_selection_sha256",
    "factor_config_sha256",
    "training_gradient_config_sha256",
}
_OBJECTIVE_FIELDS = {
    "objective_id",
    "kind",
    "behavior",
    "status",
    "prompt_bank_sha256",
    "causal_feature_receipt_sha256",
}
_OBJECTIVE_IDENTITIES = {
    (behavior, kind)
    for behavior in ("validate_feelings", "refusal")
    for kind in ("behavior_loss", "causal_concept_activation")
}


@dataclass(frozen=True)
class QueryObjective:
    objective_id: str
    kind: str
    status: str


@dataclass(frozen=True)
class CausalGradientPlan:
    plan_id: str
    plan_sha256: str
    phase: str
    method: str
    backend_id: str
    implementation_version: str
    parameter_selection_sha256: str
    factor_config_sha256: str
    training_gradient_config_sha256: str
    objectives: tuple[QueryObjective, ...]
    scientific_use: bool

    @property
    def executable_objective_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.objective_id for item in self.objectives if item.status == "ready"
            )
        )

    @property
    def blocked_objective_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.objective_id for item in self.objectives if item.status != "ready"
            )
        )


def _check_fields(value: dict[str, object], expected: set[str], label: str) -> None:
    unexpected = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unexpected or missing:
        raise CausalGradientPlanError(
            f"{label} unexpected fields={unexpected}, missing fields={missing}"
        )


def _expect_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CausalGradientPlanError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _expect_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise CausalGradientPlanError(f"{label} must be a list")
    return cast(list[object], value)


def _expect_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalGradientPlanError(f"{label} must be a non-empty string")
    return value


def _validate_plan_identity(payload: dict[str, object]) -> None:
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise CausalGradientPlanError("schema_version must be the integer 1")
    _expect_nonempty_string(payload["plan_id"], "plan_id")
    if payload["phase"] != "pre_feature":
        raise CausalGradientPlanError("phase must be pre_feature")
    if payload["method"] != "ekfac":
        raise CausalGradientPlanError("method must be ekfac")


def _validate_hashes(values: dict[str, object], fields: set[str], label: str) -> None:
    for field in fields:
        value = values[field]
        if not _is_sha256(value):
            raise CausalGradientPlanError(
                f"{label}.{field} must be a lowercase SHA-256"
            )


def _validate_distinct_hash_roles(
    values: dict[str, object], fields: set[str], label: str
) -> None:
    hashes = [values[field] for field in fields]
    if len(set(hashes)) != len(hashes):
        raise CausalGradientPlanError(f"{label} hash roles must remain distinct")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_backend(backend: dict[str, object]) -> None:
    for field in (
        "backend_id",
        "implementation_version",
        "factor_store_id",
        "training_gradient_store_id",
    ):
        _expect_nonempty_string(backend[field], field)
    implementation = _expect_nonempty_string(
        backend["implementation"], "backend implementation"
    )
    if implementation not in {"kronfluence", "curvlinops"}:
        raise CausalGradientPlanError("backend implementation is not approved")


def _validate_objective_identities(
    objectives: tuple[dict[str, object], ...],
) -> None:
    objective_ids = [
        _expect_nonempty_string(item["objective_id"], "objective_id")
        for item in objectives
    ]
    if len(set(objective_ids)) != len(objective_ids):
        raise CausalGradientPlanError("objective_id values must be unique")
    identities = [(item["behavior"], item["kind"]) for item in objectives]
    if len(identities) != len(_OBJECTIVE_IDENTITIES) or set(identities) != (
        _OBJECTIVE_IDENTITIES
    ):
        raise CausalGradientPlanError(
            "objective identities must cover both kinds for both behaviors exactly once"
        )
    prompt_hashes = [
        item["prompt_bank_sha256"]
        for item in objectives
        if item["kind"] == "behavior_loss"
    ]
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise CausalGradientPlanError("objective prompt-bank hash roles must differ")


def _validate_objective_state(item: dict[str, object]) -> None:
    if item["kind"] == "behavior_loss":
        if (
            item["status"] != "ready"
            or not _is_sha256(item["prompt_bank_sha256"])
            or item["causal_feature_receipt_sha256"] is not None
        ):
            raise CausalGradientPlanError("behavior objective contract is invalid")
        return
    if (
        item["status"] != "blocked_until_causal_feature_receipt"
        or item["prompt_bank_sha256"] is not None
        or item["causal_feature_receipt_sha256"] is not None
    ):
        raise CausalGradientPlanError("causal concept objective must remain blocked")


def _read_payload(path: Path) -> dict[str, object]:
    try:
        raw = read_isolated_bytes(path, "causal-gradient plan")
        raw_payload = json_loads_unique(raw.decode("utf-8"))
    except (IsolatedIOError, OSError, UnicodeError, ValueError) as error:
        raise CausalGradientPlanError(f"cannot load plan: {error}") from error
    payload = _expect_object(raw_payload, "plan")
    _check_fields(payload, _PLAN_FIELDS, "plan")
    _validate_plan_identity(payload)
    return payload


def _validate_artifacts(payload: dict[str, object]) -> None:
    artifacts = _expect_object(payload["artifacts"], "artifacts")
    _check_fields(artifacts, _ARTIFACT_FIELDS, "artifacts")
    _validate_hashes(artifacts, _ARTIFACT_FIELDS, "artifacts")
    _validate_distinct_hash_roles(artifacts, _ARTIFACT_FIELDS, "artifact")


def _validate_backend_payload(payload: dict[str, object]) -> None:
    backend = _expect_object(payload["backend"], "backend")
    _check_fields(backend, _BACKEND_FIELDS, "backend")
    _validate_backend(backend)
    hash_fields = {
        "parameter_selection_sha256",
        "factor_config_sha256",
        "training_gradient_config_sha256",
    }
    _validate_hashes(backend, hash_fields, "backend")
    _validate_distinct_hash_roles(backend, hash_fields, "backend")


def _validate_boundaries(payload: dict[str, object]) -> dict[str, object]:
    boundaries = _expect_object(payload["boundaries"], "boundaries")
    _check_fields(boundaries, _PRE_FEATURE_BOUNDARIES, "boundaries")
    for field in _PRE_FEATURE_BOUNDARIES:
        if boundaries.get(field) is not False:
            raise CausalGradientPlanError(f"{field} must remain false")
    return boundaries


def _validate_objective(item: dict[str, object]) -> None:
    _check_fields(item, _OBJECTIVE_FIELDS, "objective")
    kind = _expect_nonempty_string(item["kind"], "objective kind")
    behavior = _expect_nonempty_string(item["behavior"], "objective behavior")
    if kind not in {"behavior_loss", "causal_concept_activation"}:
        raise CausalGradientPlanError("objective kind is not approved")
    if behavior not in {"validate_feelings", "refusal"}:
        raise CausalGradientPlanError("objective behavior is not approved")
    _validate_objective_state(item)


def _validate_objectives(payload: dict[str, object]) -> tuple[dict[str, object], ...]:
    raw_objectives = _expect_list(payload["objectives"], "objectives")
    checked_objectives = tuple(
        _expect_object(item, "objective") for item in raw_objectives
    )
    for item in checked_objectives:
        _validate_objective(item)
    _validate_objective_identities(checked_objectives)
    return checked_objectives


def _build_plan(
    payload: dict[str, object],
    boundaries: dict[str, object],
    raw_objectives: tuple[dict[str, object], ...],
) -> CausalGradientPlan:
    backend = _expect_object(payload["backend"], "backend")
    objectives = tuple(
        QueryObjective(
            objective_id=cast(str, item["objective_id"]),
            kind=cast(str, item["kind"]),
            status=cast(str, item["status"]),
        )
        for item in raw_objectives
    )
    return CausalGradientPlan(
        plan_id=cast(str, payload["plan_id"]),
        plan_sha256=canonical_sha256(payload),
        phase=cast(str, payload["phase"]),
        method=cast(str, payload["method"]),
        backend_id=cast(str, backend["backend_id"]),
        implementation_version=cast(str, backend["implementation_version"]),
        parameter_selection_sha256=cast(str, backend["parameter_selection_sha256"]),
        factor_config_sha256=cast(str, backend["factor_config_sha256"]),
        training_gradient_config_sha256=cast(
            str, backend["training_gradient_config_sha256"]
        ),
        objectives=objectives,
        scientific_use=cast(bool, boundaries["scientific_use"]),
    )


def load_causal_gradient_plan(path: Path) -> CausalGradientPlan:
    """Load one attribution plan without performing side effects."""
    payload = _read_payload(path)
    _validate_artifacts(payload)
    _validate_backend_payload(payload)
    boundaries = _validate_boundaries(payload)
    objectives = _validate_objectives(payload)
    return _build_plan(payload, boundaries, objectives)
