"""Receipt-bound orchestration for the replication backbone."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from concept_filtering.serialization import (
    canonical_sha256,
    json_equal_exact,
    json_loads_unique,
)

_STAGE_IDS = tuple(f"B{index}" for index in range(10))
_DEPENDENCIES = {
    "B0": (),
    "B1": ("B0",),
    "B2": ("B1",),
    "B3": ("B1",),
    "B4": ("B2",),
    "B5": ("B4",),
    "B6": ("B5",),
    "B7": ("B2", "B4", "B5", "B6"),
    "B8": ("B3", "B7"),
    "B9": ("B8",),
}
_CONFIG_FIELDS = {
    "schema_version",
    "backbone_id",
    "mode",
    "novel_boundary",
    "scientific_use",
    "authority",
    "stages",
}
_AUTHORITY_FIELDS = {
    "offline_execution_allowed",
    "provider_execution_allowed",
    "scheduler_submission_allowed",
    "feature_selection_allowed",
    "concept_scoring_allowed",
    "concept_selected_training_allowed",
    "authority_receipt_sha256",
}
_STAGE_FIELDS = {
    "stage_id",
    "name",
    "dependencies",
    "capability",
    "adapter",
    "executor_release",
    "coordinator_release",
    "input_producer_releases",
    "dual_release_policy",
    "pre_side_effect_release_gate",
    "parameters",
}
_SAFE_ID_FIRST = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
_SAFE_ID_CHARS = _SAFE_ID_FIRST | frozenset("-_.")


class BackboneValidationError(ValueError):
    """Raised when the replication-backbone contract is ambiguous."""


@dataclass(frozen=True)
class BackboneAuthority:
    offline_execution_allowed: bool
    provider_execution_allowed: bool
    scheduler_submission_allowed: bool
    authority_receipt_sha256: str | None


@dataclass(frozen=True)
class BackboneStage:
    stage_id: str
    name: str
    dependencies: tuple[str, ...]
    capability: str
    adapter: str
    executor_release: str
    coordinator_release: str
    input_producer_releases: tuple[str, ...]
    parameters: dict[str, object]


@dataclass(frozen=True)
class BackboneConfig:
    backbone_id: str
    config_sha256: str
    authority: BackboneAuthority
    stages: tuple[BackboneStage, ...]


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BackboneValidationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_fields(payload: dict[str, Any], expected: set[str], field: str) -> None:
    unexpected = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unexpected or missing:
        raise BackboneValidationError(
            f"{field} unexpected fields={unexpected}, missing fields={missing}"
        )


def _string(payload: dict[str, Any], field: str, context: str | None = None) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        label = context if context is not None else field
        raise BackboneValidationError(f"{label} must be a non-empty string")
    return value


def _backbone_id(payload: dict[str, Any]) -> str:
    value = _string(payload, "backbone_id")
    if value[0] not in _SAFE_ID_FIRST or any(
        character not in _SAFE_ID_CHARS for character in value
    ):
        raise BackboneValidationError("backbone_id must be a safe path component")
    return value


def _hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise BackboneValidationError(f"{field} must be {length} lowercase hex digits")
    if any(character not in "0123456789abcdef" for character in value):
        raise BackboneValidationError(f"{field} must be {length} lowercase hex digits")
    return value


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        value = json_loads_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise BackboneValidationError(
            f"cannot load backbone config: {error}"
        ) from error
    return _mapping(value, "backbone config")


def _load_authority(value: object) -> BackboneAuthority:
    payload = _mapping(value, "authority")
    _exact_fields(payload, _AUTHORITY_FIELDS, "authority")
    boolean_fields = tuple(_AUTHORITY_FIELDS - {"authority_receipt_sha256"})
    if any(type(payload.get(field)) is not bool for field in boolean_fields):
        raise BackboneValidationError("authority flags must be booleans")
    for field in (
        "feature_selection_allowed",
        "concept_scoring_allowed",
        "concept_selected_training_allowed",
    ):
        if payload[field] is not False:
            raise BackboneValidationError(f"{field} crosses the novel boundary")
    return _build_authority(payload)


def _build_authority(payload: dict[str, Any]) -> BackboneAuthority:
    receipt = payload["authority_receipt_sha256"]
    if receipt is not None:
        receipt = _hex(receipt, 64, "authority_receipt_sha256")
    side_effect_allowed = (
        payload["provider_execution_allowed"] or payload["scheduler_submission_allowed"]
    )
    if side_effect_allowed and receipt is None:
        raise BackboneValidationError("side-effect authority requires a receipt")
    return BackboneAuthority(
        offline_execution_allowed=payload["offline_execution_allowed"],
        provider_execution_allowed=payload["provider_execution_allowed"],
        scheduler_submission_allowed=payload["scheduler_submission_allowed"],
        authority_receipt_sha256=receipt,
    )


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BackboneValidationError(f"{field} must be a non-empty list")
    items = cast(list[object], value)
    releases = tuple(_hex(item, 64, field) for item in items)
    if len(set(releases)) != len(releases):
        raise BackboneValidationError(f"{field} must be unique")
    return releases


def _dependencies(value: object, stage_id: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BackboneValidationError(
            f"{stage_id}.dependencies must be a list of stage IDs"
        )
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise BackboneValidationError(
            f"{stage_id}.dependencies must be a list of stage IDs"
        )
    observed = tuple(cast(list[str], items))
    if observed != _DEPENDENCIES[stage_id]:
        raise BackboneValidationError(f"{stage_id} dependencies do not match roadmap")
    return observed


def _load_stage(value: object, expected_id: str) -> BackboneStage:
    payload = _mapping(value, expected_id)
    _exact_fields(payload, _STAGE_FIELDS, expected_id)
    if payload.get("stage_id") != expected_id:
        raise BackboneValidationError("stages must remain ordered B0 through B9")
    _validate_stage_policy(payload, expected_id)
    return _build_stage(payload, expected_id)


def _validate_stage_policy(payload: dict[str, Any], stage_id: str) -> None:
    if payload.get("capability") != "offline" or payload.get("adapter") != "fixture":
        raise BackboneValidationError(f"{stage_id} fixture stage must remain offline")
    if payload.get("dual_release_policy") != "none":
        raise BackboneValidationError(f"{stage_id}.dual_release_policy must be none")
    if payload.get("pre_side_effect_release_gate") != "backbone-release-tuple-v1":
        raise BackboneValidationError(
            f"{stage_id}.pre_side_effect_release_gate does not match"
        )
    if payload.get("coordinator_release") != payload.get("executor_release"):
        raise BackboneValidationError(
            f"{stage_id}.coordinator_release and executor_release must match"
        )


def _build_stage(payload: dict[str, Any], stage_id: str) -> BackboneStage:
    parameters = _mapping(payload.get("parameters"), f"{stage_id}.parameters")
    return BackboneStage(
        stage_id=stage_id,
        name=_string(payload, "name", f"{stage_id}.name"),
        dependencies=_dependencies(payload.get("dependencies"), stage_id),
        capability=cast(str, payload["capability"]),
        adapter=cast(str, payload["adapter"]),
        executor_release=_hex(
            payload.get("executor_release"), 40, f"{stage_id}.executor_release"
        ),
        coordinator_release=_hex(
            payload.get("coordinator_release"), 40, f"{stage_id}.coordinator_release"
        ),
        input_producer_releases=_string_list(
            payload.get("input_producer_releases"),
            f"{stage_id}.input_producer_releases",
        ),
        parameters=cast(dict[str, object], parameters),
    )


def load_backbone_config(path: Path) -> BackboneConfig:
    payload = _load_payload(path)
    _exact_fields(payload, _CONFIG_FIELDS, "backbone config")
    _validate_envelope(payload)
    stages_value = payload.get("stages")
    if not isinstance(stages_value, list):
        raise BackboneValidationError("stages must contain exactly B0 through B9")
    stage_values = cast(list[object], stages_value)
    if len(stage_values) != len(_STAGE_IDS):
        raise BackboneValidationError("stages must contain exactly B0 through B9")
    stages = tuple(
        _load_stage(value, stage_id)
        for value, stage_id in zip(stage_values, _STAGE_IDS, strict=True)
    )
    return BackboneConfig(
        backbone_id=_backbone_id(payload),
        config_sha256=canonical_sha256(payload),
        authority=_load_authority(payload.get("authority")),
        stages=stages,
    )


def _validate_envelope(payload: dict[str, Any]) -> None:
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise BackboneValidationError("schema_version must be the integer 1")
    if payload.get("mode") != "fixture":
        raise BackboneValidationError("mode must be fixture")
    if payload.get("novel_boundary") != "sae_feature_selection":
        raise BackboneValidationError("novel boundary must be sae_feature_selection")
    if payload.get("scientific_use") is not False:
        raise BackboneValidationError("fixture scientific_use must be false")


def plan_backbone(config: BackboneConfig, run_dir: Path) -> dict[str, object]:
    receipts = _load_receipts(config, run_dir)
    return _plan_summary(config, receipts)


def _plan_summary(
    config: BackboneConfig, receipts: dict[str, dict[str, Any]]
) -> dict[str, object]:
    completed = set(receipts)
    stages: list[dict[str, object]] = []
    for stage in config.stages:
        blockers = [item for item in stage.dependencies if item not in completed]
        reasons = _authority_blockers(config, stage)
        status = _stage_status(stage.stage_id, completed, blockers, reasons)
        stages.append(
            {
                "stage_id": stage.stage_id,
                "name": stage.name,
                "dependencies": list(stage.dependencies),
                "capability": stage.capability,
                "status": status,
                "blocking_stages": blockers,
                "blocking_reasons": reasons,
            }
        )
    return {
        "schema_version": 1,
        "backbone_id": config.backbone_id,
        "config_sha256": config.config_sha256,
        "novel_boundary": "sae_feature_selection",
        "stages": stages,
    }


def _authority_blockers(config: BackboneConfig, stage: BackboneStage) -> list[str]:
    if stage.capability == "offline" and not config.authority.offline_execution_allowed:
        return ["authority.offline_execution_allowed=false"]
    return []


def _stage_status(
    stage_id: str,
    completed: set[str],
    blockers: list[str],
    reasons: list[str],
) -> str:
    if stage_id in completed:
        return "completed"
    return "ready" if not blockers and not reasons else "blocked"


def _receipt_path(run_dir: Path, stage_id: str) -> Path:
    return run_dir / "receipts" / f"{stage_id}.json"


def _read_receipt(path: Path, stage_id: str) -> dict[str, Any]:
    try:
        value = json_loads_unique(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise BackboneValidationError(
            f"cannot load {stage_id} receipt: {error}"
        ) from error
    return _mapping(value, f"{stage_id} receipt")


def _dependency_bindings(
    stage: BackboneStage, receipts: dict[str, dict[str, Any]]
) -> list[dict[str, object]]:
    missing = [stage_id for stage_id in stage.dependencies if stage_id not in receipts]
    if missing:
        raise BackboneValidationError(
            f"{stage.stage_id} has incomplete dependencies: {', '.join(missing)}"
        )
    return [
        {
            "stage_id": stage_id,
            "receipt_sha256": receipts[stage_id]["receipt_sha256"],
        }
        for stage_id in stage.dependencies
    ]


def _output_release(config: BackboneConfig, stage: BackboneStage) -> str:
    return canonical_sha256(
        {
            "executor_release": stage.executor_release,
            "config_sha256": config.config_sha256,
            "stage_id": stage.stage_id,
            "parameters_sha256": canonical_sha256(stage.parameters),
        }
    )


def _receipt_body(
    config: BackboneConfig,
    stage: BackboneStage,
    receipts: dict[str, dict[str, Any]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backbone_id": config.backbone_id,
        "config_sha256": config.config_sha256,
        "stage_id": stage.stage_id,
        "adapter": stage.adapter,
        "capability": stage.capability,
        "scientific_use": False,
        "provider_execution": False,
        "scheduler_submission": False,
        "executor_release": stage.executor_release,
        "coordinator_release": stage.coordinator_release,
        "input_producer_releases": list(stage.input_producer_releases),
        "dependency_receipts": _dependency_bindings(stage, receipts),
        "parameters_sha256": canonical_sha256(stage.parameters),
        "output_release": _output_release(config, stage),
    }


def _expected_receipt(
    config: BackboneConfig,
    stage: BackboneStage,
    receipts: dict[str, dict[str, Any]],
) -> dict[str, object]:
    body = _receipt_body(config, stage, receipts)
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _validate_receipt(
    config: BackboneConfig,
    stage: BackboneStage,
    observed: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
) -> None:
    expected = _expected_receipt(config, stage, receipts)
    if not json_equal_exact(observed, expected):
        raise BackboneValidationError(f"{stage.stage_id} receipt does not match config")


def _validate_receipt_directory(run_dir: Path) -> None:
    receipt_dir = run_dir / "receipts"
    if not receipt_dir.exists():
        return
    if not receipt_dir.is_dir():
        raise BackboneValidationError("receipt path is not a directory")
    allowed = {f"{stage_id}.json" for stage_id in _STAGE_IDS}
    unexpected = sorted(
        path.name
        for path in receipt_dir.iterdir()
        if path.name not in allowed and not _is_internal_receipt_temp(path)
    )
    if unexpected:
        raise BackboneValidationError(
            f"unexpected stage receipts: {', '.join(unexpected)}"
        )


def _is_internal_receipt_temp(path: Path) -> bool:
    return path.name.endswith(".tmp") and any(
        path.name.startswith(f".{stage_id}.json.") for stage_id in _STAGE_IDS
    )


def _load_receipts(config: BackboneConfig, run_dir: Path) -> dict[str, dict[str, Any]]:
    if not run_dir.exists():
        return {}
    if not run_dir.is_dir():
        raise BackboneValidationError("run directory path is not a directory")
    _validate_receipt_directory(run_dir)
    receipts: dict[str, dict[str, Any]] = {}
    for stage in config.stages:
        path = _receipt_path(run_dir, stage.stage_id)
        if path.exists():
            observed = _read_receipt(path, stage.stage_id)
            _validate_receipt(config, stage, observed, receipts)
            receipts[stage.stage_id] = observed
    _validate_existing_readiness(config, run_dir, receipts)
    return receipts


def status_backbone(config: BackboneConfig, run_dir: Path) -> dict[str, object]:
    receipts = _load_receipts(config, run_dir)
    plan = _plan_summary(config, receipts)
    stages = cast(list[dict[str, object]], plan["stages"])
    return {
        **plan,
        "completed_stages": list(receipts),
        "next_ready_stages": [
            cast(str, stage["stage_id"])
            for stage in stages
            if stage["status"] == "ready"
        ],
        "ready_for_sae_feature_selection": len(receipts) == len(config.stages),
    }


def _prepare_run_directory(run_dir: Path) -> None:
    if run_dir.exists() and not run_dir.is_dir():
        raise BackboneValidationError("run directory path is not a directory")
    (run_dir / "receipts").mkdir(parents=True, exist_ok=True)


def _authorize_stage(config: BackboneConfig, stage: BackboneStage) -> None:
    if stage.capability == "offline" and not config.authority.offline_execution_allowed:
        raise BackboneValidationError(
            f"{stage.stage_id} offline execution is not allowed"
        )
    if stage.capability != "offline":
        raise BackboneValidationError(f"{stage.stage_id} capability is not implemented")


def _write_temporary_receipt(path: Path, receipt: dict[str, object]) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    temporary = _write_temporary_receipt(path, receipt)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise BackboneValidationError(f"refusing to overwrite {path.name}") from error
    finally:
        temporary.unlink(missing_ok=True)


def execute_backbone(
    config: BackboneConfig, run_dir: Path, *, through: str
) -> dict[str, object]:
    if through not in _STAGE_IDS:
        raise BackboneValidationError("through must name B0 through B9")
    receipts = _load_receipts(config, run_dir)
    target_stages = config.stages[: _STAGE_IDS.index(through) + 1]
    for stage in target_stages:
        if stage.stage_id not in receipts:
            _authorize_stage(config, stage)
    _prepare_run_directory(run_dir)
    created: list[str] = []
    replayed: list[str] = []
    for stage in target_stages:
        if stage.stage_id in receipts:
            replayed.append(stage.stage_id)
            continue
        receipt = _expected_receipt(config, stage, receipts)
        _write_receipt(_receipt_path(run_dir, stage.stage_id), receipt)
        receipts[stage.stage_id] = cast(dict[str, Any], receipt)
        created.append(stage.stage_id)
    return {
        "backbone_id": config.backbone_id,
        "config_sha256": config.config_sha256,
        "through": through,
        "created_stages": created,
        "replayed_stages": replayed,
    }


def _readiness_body(
    config: BackboneConfig, receipts: dict[str, dict[str, Any]]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backbone_id": config.backbone_id,
        "config_sha256": config.config_sha256,
        "stage_receipts": [
            {
                "stage_id": stage.stage_id,
                "receipt_sha256": receipts[stage.stage_id]["receipt_sha256"],
                "output_release": receipts[stage.stage_id]["output_release"],
            }
            for stage in config.stages
        ],
        "stage_count": len(receipts),
        "scientific_use": False,
        "feature_selection_executed": False,
        "causal_sae_interventions_executed": False,
        "concept_scoring_executed": False,
        "concept_selected_training_executed": False,
        "ready_for_sae_feature_selection": True,
        "next_scientific_action": "sae_feature_selection",
    }


def _expected_readiness(
    config: BackboneConfig, receipts: dict[str, dict[str, Any]]
) -> dict[str, object]:
    body = _readiness_body(config, receipts)
    return {**body, "readiness_receipt_sha256": canonical_sha256(body)}


def _validate_existing_readiness(
    config: BackboneConfig,
    run_dir: Path,
    receipts: dict[str, dict[str, Any]],
) -> None:
    path = run_dir / "readiness-receipt.json"
    if not path.exists():
        return
    observed = _read_receipt(path, "readiness")
    complete = len(receipts) == len(config.stages)
    if not complete or not json_equal_exact(
        observed, _expected_readiness(config, receipts)
    ):
        raise BackboneValidationError("readiness receipt contradicts stage evidence")


def _incomplete_readiness(
    config: BackboneConfig, receipts: dict[str, dict[str, Any]]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backbone_id": config.backbone_id,
        "config_sha256": config.config_sha256,
        "completed_stages": list(receipts),
        "missing_stages": [
            stage.stage_id for stage in config.stages if stage.stage_id not in receipts
        ],
        "ready_for_sae_feature_selection": False,
        "next_scientific_action": None,
    }


def verify_backbone(config: BackboneConfig, run_dir: Path) -> dict[str, object]:
    receipts = _load_receipts(config, run_dir)
    if len(receipts) != len(config.stages):
        return _incomplete_readiness(config, receipts)
    expected = _expected_readiness(config, receipts)
    path = run_dir / "readiness-receipt.json"
    if not path.exists():
        _write_receipt(path, expected)
    return expected
