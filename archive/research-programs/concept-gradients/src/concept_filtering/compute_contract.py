"""No-submit compute and deployment contract planning."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from datetime import datetime
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import cast

from concept_filtering.serialization import (
    bytes_sha256,
    canonical_sha256,
    json_loads_unique,
)


class ComputeContractError(ValueError):
    """Raised when a compute contract cannot be planned safely."""


_CONTRACT_FIELDS = {
    "schema_version",
    "contract_id",
    "mode",
    "site",
    "deployment",
    "storage",
    "ledger",
    "stages",
    "boundaries",
}
_BOUNDARY_FIELDS = {
    "offline_plan_allowed",
    "deployment_execution_allowed",
    "scheduler_submission_allowed",
    "provider_execution_allowed",
    "destructive_cleanup_allowed",
    "scientific_use",
    "sae_feature_selection_allowed",
    "concept_scoring_allowed",
    "concept_selected_training_allowed",
}
_STAGE_FIELDS = {
    "stage_id",
    "name",
    "command_id",
    "command_receipt_sha256",
    "work_units",
    "dependencies",
    "release_tuple",
    "resources",
    "recovery",
}
_EXPECTED_DEPENDENCIES = {
    "G2_EKFAC_SMOKE": (),
    "G3_FACTOR_FIT": ("G2_EKFAC_SMOKE",),
    "G3_TRAINING_GRADIENTS": ("G3_FACTOR_FIT",),
    "G3_BEHAVIOR_QUERY_SCORES": ("G3_TRAINING_GRADIENTS",),
    "G3_ORDINARY_REMOVAL_MANIFEST": ("G3_BEHAVIOR_QUERY_SCORES",),
    "B7_TRAIN_REPLICATION_CONDITIONS": ("G3_ORDINARY_REMOVAL_MANIFEST",),
    "B8_EVALUATE_REPLICATION_CONDITIONS": ("B7_TRAIN_REPLICATION_CONDITIONS",),
}
_EXPECTED_COMMAND_IDS = {
    "G2_EKFAC_SMOKE": "run-ekfac-backend-smoke",
    "G3_FACTOR_FIT": "fit-ekfac-factors",
    "G3_TRAINING_GRADIENTS": "build-training-gradient-store",
    "G3_BEHAVIOR_QUERY_SCORES": "score-behavior-queries",
    "G3_ORDINARY_REMOVAL_MANIFEST": "freeze-ordinary-ekfac-manifest",
    "B7_TRAIN_REPLICATION_CONDITIONS": "train-replication-conditions",
    "B8_EVALUATE_REPLICATION_CONDITIONS": "evaluate-replication-conditions",
}
_EXPECTED_INPUT_ROLES = {
    "G2_EKFAC_SMOKE": {
        "organism",
        "training_data_package",
        "attribution_plan",
        "ekfac_backend_config",
    },
    "G3_FACTOR_FIT": {
        "training_data_package",
        "factor_config",
        "parameter_selection",
    },
    "G3_TRAINING_GRADIENTS": {
        "training_data_package",
        "training_gradient_config",
    },
    "G3_BEHAVIOR_QUERY_SCORES": {
        "validation_query_bank",
        "refusal_query_bank",
        "score_config",
    },
    "G3_ORDINARY_REMOVAL_MANIFEST": {
        "training_data_package",
        "selection_policy",
        "condition_multiplicity_decision",
    },
    "B7_TRAIN_REPLICATION_CONDITIONS": {
        "training_data_package",
        "training_recipe",
        "dense_probe_manifest",
        "matched_random_manifest",
        "condition_multiplicity_decision",
    },
    "B8_EVALUATE_REPLICATION_CONDITIONS": {
        "evaluator_contract",
        "accepted_evaluator_receipt",
        "validation_prompt_bank",
        "refusal_prompt_bank",
        "condition_multiplicity_decision",
    },
}
_VALIDATION_ONLY_CONDITION_UNITS = [
    "full_data",
    "dense_probe_removal",
    "matched_random_removal",
    "ordinary_ekfac_removal",
]
_SEPARATE_BEHAVIOR_CONDITION_UNITS = [
    "full_data",
    "validation_dense_probe_removal",
    "validation_matched_random_removal",
    "validation_ordinary_ekfac_removal",
    "refusal_dense_probe_removal",
    "refusal_matched_random_removal",
    "refusal_ordinary_ekfac_removal",
]
_CONDITION_WORK_UNIT_OPTIONS = {
    "G3_ORDINARY_REMOVAL_MANIFEST": [["validation"], ["validation", "refusal"]],
    "B7_TRAIN_REPLICATION_CONDITIONS": [
        _VALIDATION_ONLY_CONDITION_UNITS,
        _SEPARATE_BEHAVIOR_CONDITION_UNITS,
    ],
    "B8_EVALUATE_REPLICATION_CONDITIONS": [
        _VALIDATION_ONLY_CONDITION_UNITS,
        _SEPARATE_BEHAVIOR_CONDITION_UNITS,
    ],
}
_EXPECTED_STATIC_WORK_UNITS = {
    "G2_EKFAC_SMOKE": ["bounded_smoke"],
    "G3_FACTOR_FIT": ["factor_fit"],
    "G3_BEHAVIOR_QUERY_SCORES": ["validation", "refusal"],
}
_RELEASE_FIELDS = {
    "executor_release",
    "coordinator_release",
    "input_producer_releases",
    "output_release_policy",
    "pre_side_effect_release_gate",
}
_RESOURCE_FIELDS = {
    "nodes",
    "tasks_per_node",
    "cpus_per_task",
    "gpus_per_node",
    "gpu_type",
    "memory_per_node_gib",
    "wall_time_minutes",
    "array_size",
    "max_concurrency",
}
_RECOVERY_FIELDS = {
    "checkpoint_cadence_minutes",
    "max_attempts",
    "retry_policy",
    "retryable_states",
}
_SITE_FIELDS = {"cluster", "host", "account", "qos", "constraint", "module_names"}
_DEPLOYMENT_FIELDS = {
    "receipt_contract",
    "origin",
    "revision",
    "release_root",
    "receipt_sha256",
    "tracked_identities",
    "verification_mode",
}
_STORAGE_FIELDS = {
    "shared_project_root",
    "node_local_root",
    "retained_outputs_relative_root",
    "checkpoints_relative_root",
    "logs_relative_root",
    "cleanup_policy",
}
_LEDGER_FIELDS = {
    "summary_path",
    "detailed_path",
    "accounting_mode",
    "include_failed",
    "include_retried",
    "include_debug",
    "fields",
}
_ACCOUNTING_FIELDS = [
    "JobIDRaw",
    "JobName",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocTRES",
]
_RECEIPT_FIELDS = {
    "receipt_version",
    "contract",
    "origin",
    "revision",
    "deployment_path",
    "tracked_identities",
    "git_tree",
    "verified_at",
}
_TRACKED_IDENTITY_FIELDS = {"path", "git_object", "object_type"}


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ComputeContractError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ComputeContractError(f"{label} must be a list")
    return cast(list[object], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComputeContractError(f"{label} must be a non-empty string")
    return value


def _check_fields(value: dict[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ComputeContractError(
            f"{label} unexpected fields={unexpected}, missing fields={missing}"
        )


def _parse_json_bytes(value: bytes, label: str) -> dict[str, object]:
    try:
        parsed = json_loads_unique(value.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ComputeContractError(f"cannot load {label}: {error}") from error
    return _object(parsed, label)


def _input_identity(status: os.stat_result, label: str) -> tuple[int, ...]:
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise ComputeContractError(f"{label} is not an isolated regular file")
    return (
        status.st_dev,
        status.st_ino,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _read_isolated_bytes(path: Path, label: str) -> bytes:
    message = f"{label} is not an isolated regular file"
    parents_before = _parent_identities(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise ComputeContractError(message) from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = _input_identity(os.fstat(handle.fileno()), label)
            value = handle.read()
            after = _input_identity(os.fstat(handle.fileno()), label)
        path_identity = _input_identity(path.lstat(), label)
        parents_after = _parent_identities(path, label)
    except OSError as error:
        raise ComputeContractError(message) from error
    if before != after or after != path_identity or parents_before != parents_after:
        raise ComputeContractError(f"{label} identity changed during read")
    return value


def _parent_identities(path: Path, label: str) -> tuple[tuple[int, int], ...]:
    identities: list[tuple[int, int]] = []
    try:
        for parent in path.absolute().parents:
            status = parent.lstat()
            if not stat.S_ISDIR(status.st_mode):
                raise ComputeContractError(
                    f"{label} must have plain parent directories"
                )
            identities.append((status.st_dev, status.st_ino))
    except OSError as error:
        raise ComputeContractError(
            f"{label} must have plain parent directories"
        ) from error
    return tuple(identities)


def _load_contract(path: Path) -> tuple[dict[str, object], str]:
    raw = _read_isolated_bytes(path, "compute contract")
    payload = _parse_json_bytes(raw, "compute contract")
    _check_fields(payload, _CONTRACT_FIELDS, "compute contract")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ComputeContractError("schema_version must be the integer 1")
    _string(payload.get("contract_id"), "contract_id")
    try:
        contract_sha256 = canonical_sha256(payload)
    except (TypeError, ValueError, RecursionError) as error:
        raise ComputeContractError(
            f"compute contract is not canonical JSON: {error}"
        ) from error
    return payload, contract_sha256


def _validate_boundaries(payload: dict[str, object]) -> None:
    boundaries = _object(payload["boundaries"], "boundaries")
    _check_fields(boundaries, _BOUNDARY_FIELDS, "boundaries")
    for field in _BOUNDARY_FIELDS:
        expected = field == "offline_plan_allowed"
        if boundaries[field] is not expected:
            raise ComputeContractError(f"boundaries.{field} must be {expected}")


def _safe_relative_path(value: object, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if (
        _has_control_character(text)
        or "\\" in text
        or path.is_absolute()
        or not _path_components_are_canonical(text, absolute=False)
    ):
        raise ComputeContractError(f"{label} must be a safe relative path")
    return text


def _has_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _path_components_are_canonical(value: str, *, absolute: bool) -> bool:
    components = value.split("/")
    if absolute:
        components = components[1:]
    return bool(components) and all(item not in {"", ".", ".."} for item in components)


def _safe_absolute_path(value: object, label: str) -> str | None:
    if value == "VERIFY_ON_PACE":
        return None
    text = _string(value, label)
    path = PurePosixPath(text)
    if not path.is_absolute() or path == PurePosixPath("/"):
        raise ComputeContractError(f"{label} must be a scoped absolute path")
    if (
        _has_control_character(text)
        or "\\" in text
        or not _path_components_are_canonical(text, absolute=True)
    ):
        raise ComputeContractError(f"{label} contains an unsafe path component")
    return text


def _validate_site(payload: dict[str, object]) -> None:
    site = _object(payload["site"], "site")
    _check_fields(site, _SITE_FIELDS, "site")
    expected = {"cluster": "pace-phoenix", "host": "pace-phoenix-agent"}
    for field, value in expected.items():
        if site[field] != value:
            raise ComputeContractError(f"site.{field} must be {value}")
    account = site["account"]
    if account != "VERIFY_ON_PACE" and (
        not isinstance(account, str)
        or re.fullmatch(r"gts-[A-Za-z0-9_-]+", account) is None
    ):
        raise ComputeContractError("site.account is invalid")
    if site["qos"] not in {"inferno", "embers"}:
        raise ComputeContractError("site.qos is not an approved Phoenix QOS")
    if site["constraint"] != "gpu-h100":
        raise ComputeContractError("site.constraint must be gpu-h100")
    if site["module_names"] != ["anaconda3", "cuda"]:
        raise ComputeContractError("site.module_names must use documented base names")


def _validate_deployment(payload: dict[str, object]) -> None:
    deployment = _object(payload["deployment"], "deployment")
    _check_fields(deployment, _DEPLOYMENT_FIELDS, "deployment")
    if deployment["receipt_contract"] != "pace-code-release-v3":
        raise ComputeContractError("deployment.receipt_contract must be v3")
    origin = "https://github.com/glennmatlin/mats12-concept-filtering.git"
    if deployment["origin"] != origin:
        raise ComputeContractError("deployment.origin does not match the repository")
    revision = deployment["revision"]
    if revision != "VERIFY_ON_PACE" and not is_git_revision(revision):
        raise ComputeContractError("deployment.revision must be a Git revision")
    _safe_absolute_path(deployment["release_root"], "deployment.release_root")
    receipt_hash = deployment["receipt_sha256"]
    if receipt_hash != "VERIFY_ON_PACE" and not _is_sha256(receipt_hash):
        raise ComputeContractError("deployment.receipt_sha256 must be a SHA-256")
    _validate_tracked_identities(deployment)
    if deployment["verification_mode"] != "pace-code-release-v3-verify-only":
        raise ComputeContractError("deployment.verification_mode does not match")


def _validate_tracked_identities(deployment: dict[str, object]) -> None:
    raw = _list(deployment["tracked_identities"], "deployment.tracked_identities")
    paths = [_safe_relative_path(item, "tracked identity") for item in raw]
    if not paths or len(set(paths)) != len(paths):
        raise ComputeContractError(
            "deployment.tracked_identities must be nonempty and unique"
        )


def _validate_storage(payload: dict[str, object]) -> None:
    storage = _object(payload["storage"], "storage")
    _check_fields(storage, _STORAGE_FIELDS, "storage")
    _safe_absolute_path(storage["shared_project_root"], "storage.shared_project_root")
    if storage["node_local_root"] != "${TMPDIR}":
        raise ComputeContractError("storage.node_local_root must be ${TMPDIR}")
    relative_fields = _STORAGE_FIELDS - {
        "shared_project_root",
        "node_local_root",
        "cleanup_policy",
    }
    paths = [
        _safe_relative_path(storage[field], f"storage.{field}")
        for field in relative_fields
    ]
    if any(_paths_overlap(first, second) for first, second in combinations(paths, 2)):
        raise ComputeContractError("storage relative roots must not overlap")
    if storage["cleanup_policy"] != "node-local-after-receipted-copy-v1":
        raise ComputeContractError("storage.cleanup_policy does not match")


def _paths_overlap(first: str, second: str) -> bool:
    first_path = PurePosixPath(first)
    second_path = PurePosixPath(second)
    return (
        first_path == second_path
        or first_path in second_path.parents
        or second_path in first_path.parents
    )


def _validate_ledger(payload: dict[str, object]) -> None:
    ledger = _object(payload["ledger"], "ledger")
    _check_fields(ledger, _LEDGER_FIELDS, "ledger")
    expected_paths = {
        "summary_path": "docs/COMPUTE_LEDGER.md",
        "detailed_path": "reports/compute/ledger.jsonl",
    }
    for field, expected in expected_paths.items():
        if ledger[field] != expected:
            raise ComputeContractError(f"ledger.{field} must be {expected}")
    if ledger["accounting_mode"] != "allocation-only-sacct-v1":
        raise ComputeContractError("ledger.accounting_mode does not match")
    for field in ("include_failed", "include_retried", "include_debug"):
        if ledger[field] is not True:
            raise ComputeContractError(f"ledger.{field} must be true")
    if ledger["fields"] != _ACCOUNTING_FIELDS:
        raise ComputeContractError("ledger.fields do not match")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive_or_marker(value: object, label: str) -> int | None:
    if value == "VERIFY_ON_PACE":
        return None
    if type(value) is not int or value <= 0:
        raise ComputeContractError(f"{label} must be a positive integer")
    return value


def _nonnegative_or_marker(value: object, label: str) -> int | None:
    if value == "VERIFY_ON_PACE":
        return None
    if type(value) is not int or value < 0:
        raise ComputeContractError(f"{label} must be a nonnegative integer")
    return value


def _validate_release_tuple(
    stage_id: str, stage: dict[str, object], revision: object
) -> None:
    release = _object(stage["release_tuple"], f"{stage_id}.release_tuple")
    _check_fields(release, _RELEASE_FIELDS, f"{stage_id}.release_tuple")
    for field in ("executor_release", "coordinator_release"):
        value = release[field]
        if value != "VERIFY_ON_PACE" and not is_git_revision(value):
            raise ComputeContractError(f"{stage_id}.{field} must be a Git revision")
        if revision != "VERIFY_ON_PACE" and value != revision:
            raise ComputeContractError(f"{stage_id}.{field} must match deployment")
    _validate_input_releases(stage_id, release)
    if release["output_release_policy"] != "compute-stage-output-release-v1":
        raise ComputeContractError(f"{stage_id}.output_release_policy does not match")
    if release["pre_side_effect_release_gate"] != "compute-release-tuple-v1":
        raise ComputeContractError(
            f"{stage_id}.pre_side_effect_release_gate does not match"
        )


def _validate_input_releases(stage_id: str, release: dict[str, object]) -> None:
    values = _object(
        release["input_producer_releases"],
        f"{stage_id}.input_producer_releases",
    )
    if set(values) != _EXPECTED_INPUT_ROLES[stage_id]:
        raise ComputeContractError(f"{stage_id}.input producer roles do not match")
    if any(
        value != "VERIFY_ON_PACE" and not _is_sha256(value) for value in values.values()
    ):
        raise ComputeContractError(f"{stage_id}.input_producer_releases are invalid")
    exact_values = [value for value in values.values() if value != "VERIFY_ON_PACE"]
    if len(set(exact_values)) != len(exact_values):
        raise ComputeContractError(f"{stage_id}.input_producer_releases must be unique")


def _validate_resources(stage_id: str, stage: dict[str, object]) -> None:
    resources = _object(stage["resources"], f"{stage_id}.resources")
    _check_fields(resources, _RESOURCE_FIELDS, f"{stage_id}.resources")
    numeric_fields = _RESOURCE_FIELDS - {
        "gpus_per_node",
        "gpu_type",
    }
    values = {
        field: _positive_or_marker(resources[field], f"{stage_id}.{field}")
        for field in numeric_fields
    }
    _validate_gpu_resources(stage_id, resources)
    array_size = values["array_size"]
    concurrency = values["max_concurrency"]
    if array_size is not None and concurrency is not None and concurrency > array_size:
        raise ComputeContractError(f"{stage_id}.max_concurrency exceeds array_size")
    _validate_work_units(stage_id, stage, array_size)


def _validate_work_units(
    stage_id: str, stage: dict[str, object], array_size: int | None
) -> None:
    work_units = stage["work_units"]
    if stage_id == "G3_TRAINING_GRADIENTS":
        if work_units == "VERIFY_ON_PACE" and array_size is None:
            return
        items = _list(work_units, f"{stage_id}.work_units")
        expected = [
            f"training_gradient_shard_{index:04d}" for index in range(len(items))
        ]
        if items != expected or array_size != len(items):
            raise ComputeContractError(f"{stage_id}.work_units do not match")
        return
    if stage_id in _CONDITION_WORK_UNIT_OPTIONS:
        if work_units == "VERIFY_ON_PACE" and array_size is None:
            return
        items = _list(work_units, f"{stage_id}.work_units")
        if items not in _CONDITION_WORK_UNIT_OPTIONS[stage_id]:
            raise ComputeContractError(f"{stage_id}.work_units do not match")
        if array_size != len(items):
            raise ComputeContractError(
                f"{stage_id}.array_size does not match work units"
            )
        return
    if work_units != _EXPECTED_STATIC_WORK_UNITS[stage_id]:
        raise ComputeContractError(f"{stage_id}.work_units do not match")
    if array_size != len(_EXPECTED_STATIC_WORK_UNITS[stage_id]):
        raise ComputeContractError(f"{stage_id}.array_size does not match work units")


def _validate_gpu_resources(stage_id: str, resources: dict[str, object]) -> None:
    gpu_count = _nonnegative_or_marker(
        resources["gpus_per_node"], f"{stage_id}.gpus_per_node"
    )
    gpu_type = resources["gpu_type"]
    if gpu_count == 0:
        if gpu_type is not None:
            raise ComputeContractError(f"{stage_id} CPU resources must omit GPU fields")
        return
    if gpu_type != "H100":
        raise ComputeContractError(f"{stage_id}.gpu_type must be H100")


def _validate_recovery(stage_id: str, stage: dict[str, object]) -> None:
    recovery = _object(stage["recovery"], f"{stage_id}.recovery")
    _check_fields(recovery, _RECOVERY_FIELDS, f"{stage_id}.recovery")
    cadence = _positive_or_marker(
        recovery["checkpoint_cadence_minutes"],
        f"{stage_id}.checkpoint_cadence_minutes",
    )
    _positive_or_marker(recovery["max_attempts"], f"{stage_id}.max_attempts")
    if recovery["retry_policy"] != "resume-from-verified-checkpoint-v1":
        raise ComputeContractError(f"{stage_id}.retry_policy does not match")
    states = _list(recovery["retryable_states"], f"{stage_id}.retryable_states")
    if states != ["PREEMPTED", "TIMEOUT", "NODE_FAIL"]:
        raise ComputeContractError(f"{stage_id}.retryable_states do not match")
    resources = _object(stage["resources"], f"{stage_id}.resources")
    walltime = _positive_or_marker(resources["wall_time_minutes"], "walltime")
    if cadence is not None and walltime is not None and cadence >= walltime:
        raise ComputeContractError(
            f"{stage_id}.checkpoint cadence must be shorter than walltime"
        )


def _validate_stage_contracts(payload: dict[str, object]) -> None:
    deployment = _object(payload["deployment"], "deployment")
    revision = deployment.get("revision")
    stages = [_object(item, "stage") for item in _list(payload["stages"], "stages")]
    for stage in stages:
        stage_id = _string(stage["stage_id"], "stage_id")
        _string(stage["name"], f"{stage_id}.name")
        if stage["command_id"] != _EXPECTED_COMMAND_IDS[stage_id]:
            raise ComputeContractError(f"{stage_id}.command_id does not match")
        command_receipt = stage["command_receipt_sha256"]
        if command_receipt != "VERIFY_ON_PACE" and not _is_sha256(command_receipt):
            raise ComputeContractError(f"{stage_id}.command receipt is invalid")
        _validate_release_tuple(stage_id, stage, revision)
        _validate_resources(stage_id, stage)
        _validate_recovery(stage_id, stage)
    _validate_command_receipts(stages)
    _validate_shared_producer_releases(stages)
    _validate_condition_multiplicity(stages)


def _validate_command_receipts(stages: list[dict[str, object]]) -> None:
    receipts = [
        stage["command_receipt_sha256"]
        for stage in stages
        if stage["command_receipt_sha256"] != "VERIFY_ON_PACE"
    ]
    if len(set(receipts)) != len(receipts):
        raise ComputeContractError("exact command receipt identities must be unique")


def _validate_shared_producer_releases(stages: list[dict[str, object]]) -> None:
    seen: dict[str, object] = {}
    for stage in stages:
        release = _object(stage["release_tuple"], "release_tuple")
        inputs = _object(release["input_producer_releases"], "input releases")
        for role, value in inputs.items():
            if role in seen and seen[role] != value:
                raise ComputeContractError(f"shared producer role {role} conflicts")
            seen[role] = value


def _validate_condition_multiplicity(stages: list[dict[str, object]]) -> None:
    manifest, training, evaluation = (
        stages[index]["work_units"] for index in (4, 5, 6)
    )
    if manifest == training == evaluation == "VERIFY_ON_PACE":
        return
    expected = (
        _VALIDATION_ONLY_CONDITION_UNITS
        if manifest == ["validation"]
        else _SEPARATE_BEHAVIOR_CONDITION_UNITS
        if manifest == ["validation", "refusal"]
        else None
    )
    if expected is None or training != expected or evaluation != expected:
        raise ComputeContractError("condition multiplicity work units conflict")


def _stage_order(payload: dict[str, object]) -> tuple[str, ...]:
    stages = [_object(item, "stage") for item in _list(payload["stages"], "stages")]
    for stage in stages:
        _check_fields(stage, _STAGE_FIELDS, "stage")
    stage_ids = tuple(_string(stage.get("stage_id"), "stage_id") for stage in stages)
    if len(set(stage_ids)) != len(stage_ids):
        raise ComputeContractError("stage_id values must be unique")
    if stage_ids != tuple(_EXPECTED_DEPENDENCIES):
        raise ComputeContractError(
            "stage graph must contain the complete frozen stage order"
        )
    for stage_id, stage in zip(stage_ids, stages, strict=True):
        dependencies = tuple(
            _string(item, "dependency")
            for item in _list(stage.get("dependencies"), f"{stage_id}.dependencies")
        )
        if dependencies != _EXPECTED_DEPENDENCIES[stage_id]:
            raise ComputeContractError(f"{stage_id}.dependencies do not match")
    return stage_ids


def _load_receipt(
    deployment: dict[str, object], receipt_path: Path
) -> tuple[dict[str, object], str]:
    receipt_bytes = _read_isolated_bytes(receipt_path, "deployment receipt")
    receipt_sha256 = bytes_sha256(receipt_bytes)
    receipt = _parse_json_bytes(receipt_bytes, "deployment receipt")
    _validate_receipt_structure(deployment, receipt)
    return receipt, receipt_sha256


def _validate_deployment_match(
    deployment: dict[str, object], receipt: dict[str, object]
) -> None:
    revision = deployment.get("revision")
    release_root = deployment.get("release_root")
    expected = {"revision": revision, "deployment_path": f"{release_root}/{revision}"}
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ComputeContractError(f"deployment receipt conflicts on {field}")


def _validate_receipt_structure(
    deployment: dict[str, object], receipt: dict[str, object]
) -> None:
    _check_fields(receipt, _RECEIPT_FIELDS, "deployment receipt")
    if type(receipt["receipt_version"]) is not int or receipt["receipt_version"] != 3:
        raise ComputeContractError("deployment receipt version must be 3")
    expected = {
        "contract": deployment.get("receipt_contract"),
        "origin": deployment.get("origin"),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ComputeContractError(f"deployment receipt conflicts on {field}")
    if not is_git_revision(receipt["revision"]):
        raise ComputeContractError("deployment receipt revision is invalid")
    _safe_absolute_path(receipt["deployment_path"], "receipt deployment_path")
    _validate_receipt_identities(deployment, receipt)
    if not is_git_revision(receipt["git_tree"]):
        raise ComputeContractError("deployment receipt git_tree is invalid")
    _validate_verified_at(receipt["verified_at"])


def _validate_receipt_identities(
    deployment: dict[str, object], receipt: dict[str, object]
) -> None:
    items = [
        _object(item, "deployment receipt tracked identity")
        for item in _list(receipt["tracked_identities"], "receipt identities")
    ]
    paths: list[str] = []
    for item in items:
        _check_fields(item, _TRACKED_IDENTITY_FIELDS, "deployment receipt identity")
        paths.append(_safe_relative_path(item["path"], "receipt identity path"))
        if not is_git_revision(item["git_object"]):
            raise ComputeContractError("deployment receipt git_object is invalid")
        if item["object_type"] not in {"blob", "tree"}:
            raise ComputeContractError("deployment receipt object_type is invalid")
    if paths != deployment["tracked_identities"] or len(set(paths)) != len(paths):
        raise ComputeContractError("deployment receipt tracked identities conflict")


def _validate_verified_at(value: object) -> None:
    text = _string(value, "deployment receipt verified_at")
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ComputeContractError(
            "deployment receipt verified_at is invalid"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ComputeContractError(
            "deployment receipt verified_at must include timezone"
        )


def _stage_output_releases(
    payload: dict[str, object], receipt_sha256: str
) -> dict[str, str]:
    stages = [_object(item, "stage") for item in _list(payload["stages"], "stages")]
    outputs: dict[str, str] = {}
    context = {
        field: payload[field]
        for field in ("site", "deployment", "storage", "ledger", "boundaries")
    }
    context_sha256 = canonical_sha256(context)
    for stage in stages:
        stage_id = _string(stage.get("stage_id"), "stage_id")
        dependencies = cast(list[str], stage["dependencies"])
        outputs[stage_id] = canonical_sha256(
            {
                "contract": "compute-stage-output-release-v1",
                "stage": stage,
                "execution_context_sha256": context_sha256,
                "deployment_receipt_sha256": receipt_sha256,
                "dependency_output_releases": [outputs[item] for item in dependencies],
            }
        )
    return outputs


def _blocked_fields(value: object, path: str = "") -> list[str]:
    if value == "VERIFY_ON_PACE":
        return [path]
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return [
            item
            for key in sorted(mapping)
            for item in _blocked_fields(mapping[key], f"{path}.{key}" if path else key)
        ]
    if isinstance(value, list):
        values = cast(list[object], value)
        return [
            item
            for index, child in enumerate(values)
            for item in _blocked_fields(child, f"{path}[{index}]")
        ]
    return []


def _marker_path_allowed(path: str) -> bool:
    exact = {
        "site.account",
        "deployment.revision",
        "deployment.release_root",
        "deployment.receipt_sha256",
        "storage.shared_project_root",
    }
    if path in exact:
        return True
    patterns = (
        r"stages\[\d+\]\.command_receipt_sha256",
        r"stages\[\d+\]\.work_units",
        r"stages\[\d+\]\.release_tuple\.(executor|coordinator)_release",
        r"stages\[\d+\]\.release_tuple\.input_producer_releases\.[a-z][a-z0-9_]*",
        r"stages\[\d+\]\.resources\.(nodes|tasks_per_node|cpus_per_task)",
        r"stages\[\d+\]\.resources\.(gpus_per_node|memory_per_node_gib)",
        r"stages\[\d+\]\.resources\.(wall_time_minutes|array_size|max_concurrency)",
    )
    return any(re.fullmatch(pattern, path) is not None for pattern in patterns)


def _validate_marker_paths(payload: dict[str, object]) -> None:
    invalid = [
        path for path in _blocked_fields(payload) if not _marker_path_allowed(path)
    ]
    if invalid:
        raise ComputeContractError(
            f"VERIFY_ON_PACE is not allowed at these fields: {sorted(invalid)}"
        )


def _blocked_summary(
    payload: dict[str, object],
    contract_sha256: str,
    stage_order: tuple[str, ...],
    receipt_missing: bool,
) -> dict[str, object]:
    blocked = _blocked_fields(payload)
    if receipt_missing:
        blocked.append("deployment.receipt_file")
    outputs = dict.fromkeys(stage_order)
    return _plan_summary(payload, contract_sha256, outputs, sorted(set(blocked)))


def _plan_summary(
    payload: dict[str, object],
    contract_sha256: str,
    outputs: Mapping[str, str | None],
    blocked: list[str],
) -> dict[str, object]:
    stages = [_object(item, "stage") for item in _list(payload["stages"], "stages")]
    stage_plans = [
        {**stage, "output_release": outputs[_string(stage["stage_id"], "stage_id")]}
        for stage in stages
    ]
    return {
        "contract_id": payload["contract_id"],
        "contract_sha256": contract_sha256,
        "mode": payload["mode"],
        "site": payload["site"],
        "deployment": payload["deployment"],
        "storage": payload["storage"],
        "ledger": payload["ledger"],
        "boundaries": payload["boundaries"],
        "ready_for_no_submit": not blocked,
        "blocked_fields": blocked,
        "stage_order": list(outputs),
        "stage_plans": stage_plans,
        "stage_output_releases": outputs,
        "submission_allowed": False,
    }


def _validate_contract_payload(
    payload: dict[str, object],
) -> tuple[tuple[str, ...], list[str]]:
    if payload["mode"] not in {"fixture", "production"}:
        raise ComputeContractError("mode must be fixture or production")
    _validate_site(payload)
    _validate_deployment(payload)
    _validate_storage(payload)
    _validate_ledger(payload)
    _validate_boundaries(payload)
    stage_order = _stage_order(payload)
    _validate_stage_contracts(payload)
    _validate_marker_paths(payload)
    return stage_order, _blocked_fields(payload)


def plan_compute_contract(
    config_path: Path, deployment_receipt_path: Path | None
) -> dict[str, object]:
    """Validate and summarize a no-submit compute plan."""
    payload, contract_sha256 = _load_contract(config_path)
    stage_order, blocked = _validate_contract_payload(payload)
    deployment = _object(payload["deployment"], "deployment")
    receipt = None
    receipt_sha256 = None
    if deployment_receipt_path is not None:
        receipt, receipt_sha256 = _load_receipt(deployment, deployment_receipt_path)
    if blocked or deployment_receipt_path is None:
        deployment_blocked = any(field.startswith("deployment.") for field in blocked)
        if receipt is not None and not deployment_blocked:
            _validate_receipt_hash(deployment, receipt_sha256)
            _validate_deployment_match(deployment, receipt)
        return _blocked_summary(
            payload,
            contract_sha256,
            stage_order,
            deployment_receipt_path is None or deployment_blocked,
        )
    assert receipt is not None
    _validate_receipt_hash(deployment, receipt_sha256)
    _validate_deployment_match(deployment, receipt)
    assert receipt_sha256 is not None
    outputs = _stage_output_releases(payload, receipt_sha256)
    return _plan_summary(payload, contract_sha256, outputs, [])


def _validate_receipt_hash(
    deployment: dict[str, object], receipt_sha256: str | None
) -> None:
    if receipt_sha256 != deployment.get("receipt_sha256"):
        raise ComputeContractError("deployment receipt SHA-256 does not match")


def is_git_revision(value: object) -> bool:
    """Return whether a value is a lowercase full Git commit identity."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None
