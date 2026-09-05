"""Typed organism configurations that bind parameters to artifact receipts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from concept_filtering.receipts import (
    ArtifactReceipt,
    ReceiptValidationError,
    load_artifact_receipt,
)
from concept_filtering.serialization import bytes_sha256, canonical_sha256


class OrganismValidationError(ValueError):
    """Raised when an organism configuration is incomplete or ambiguous."""


class ObservationStatus(StrEnum):
    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ParameterObservation:
    status: ObservationStatus
    value: object
    evidence: str
    reason: str | None


@dataclass(frozen=True)
class OrganismConfig:
    schema_version: int
    organism_id: str
    config_sha256: str
    artifact_set_sha256: str
    organism_sha256: str
    artifacts: tuple[ArtifactReceipt, ...]
    parameters: dict[str, ParameterObservation]


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OrganismValidationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OrganismValidationError(f"{field} must be a non-empty string")
    return value


def _parse_observation(field: str, value: object) -> ParameterObservation:
    payload = _mapping(value, f"parameter {field}")
    allowed = {"status", "value", "evidence", "reason"}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise OrganismValidationError(
            f"parameter {field} has unexpected fields: {', '.join(unexpected)}"
        )
    status = _parse_observation_status(field, payload)
    evidence = _required_string(payload, "evidence")
    reason = _parse_reason(field, payload.get("reason"))
    parameter_value = payload.get("value")
    _validate_observation_value(field, status, parameter_value, reason, payload)
    return ParameterObservation(
        status=status,
        value=parameter_value,
        evidence=evidence,
        reason=reason,
    )


def _parse_observation_status(field: str, payload: dict[str, Any]) -> ObservationStatus:
    status_value = _required_string(payload, "status")
    try:
        return ObservationStatus(status_value)
    except ValueError as error:
        raise OrganismValidationError(
            f"parameter {field} status must be observed or unavailable"
        ) from error


def _parse_reason(field: str, value: object) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise OrganismValidationError(
            f"parameter {field} reason must be null or a non-empty string"
        )
    return value


def _validate_observation_value(
    field: str,
    status: ObservationStatus,
    value: object,
    reason: str | None,
    payload: dict[str, Any],
) -> None:
    if status is ObservationStatus.OBSERVED:
        if "value" not in payload or value is None:
            raise OrganismValidationError(
                f"observed parameter {field} requires a non-null value"
            )
        if reason is not None:
            raise OrganismValidationError(
                f"observed parameter {field} cannot include a reason"
            )
    else:
        if value is not None:
            raise OrganismValidationError(
                f"unavailable parameter {field} value must be null"
            )
        if reason is None:
            raise OrganismValidationError(
                f"unavailable parameter {field} requires a reason"
            )


def _resolve_receipt(config_path: Path, raw_path: object) -> ArtifactReceipt:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise OrganismValidationError(
            "artifact_receipts entries must be non-empty relative paths"
        )
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise OrganismValidationError("artifact receipt paths must be relative")
    config_dir = config_path.parent.resolve()
    receipt_path = (config_path.parent / relative_path).resolve()
    if not receipt_path.is_relative_to(config_dir):
        raise OrganismValidationError(
            f"artifact receipt path escapes config directory: {raw_path}"
        )
    try:
        return load_artifact_receipt(receipt_path)
    except ReceiptValidationError as error:
        raise OrganismValidationError(
            f"invalid artifact receipt {raw_path}: {error}"
        ) from error


def _load_organism_payload(path: Path) -> tuple[dict[str, Any], str]:
    try:
        source_bytes = path.read_bytes()
        value: object = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OrganismValidationError(
            f"cannot load organism {path}: {error}"
        ) from error
    return _mapping(value, "organism"), bytes_sha256(source_bytes)


def _validate_organism_envelope(payload: dict[str, Any]) -> None:
    allowed = {"schema_version", "organism_id", "artifact_receipts", "parameters"}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise OrganismValidationError(
            f"organism has unexpected fields: {', '.join(unexpected)}"
        )
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise OrganismValidationError("schema_version must be the integer 1")


def _load_artifacts(path: Path, value: object) -> tuple[ArtifactReceipt, ...]:
    receipt_paths = value
    if not isinstance(receipt_paths, list) or not receipt_paths:
        raise OrganismValidationError("artifact_receipts must be a non-empty array")
    artifacts = tuple(
        _resolve_receipt(path, item) for item in cast(list[object], receipt_paths)
    )
    artifact_ids = [artifact.artifact_id for artifact in artifacts]
    if len(set(artifact_ids)) != len(artifact_ids):
        duplicates = sorted(
            artifact_id
            for artifact_id in set(artifact_ids)
            if artifact_ids.count(artifact_id) > 1
        )
        raise OrganismValidationError(f"duplicate artifact_id: {', '.join(duplicates)}")
    return artifacts


def _parse_parameters(value: object) -> dict[str, ParameterObservation]:
    raw_parameters = _mapping(value, "parameters")
    parameters: dict[str, ParameterObservation] = {}
    for field, raw_observation in raw_parameters.items():
        if not field.strip():
            raise OrganismValidationError("parameter names must be non-empty strings")
        parameters[field] = _parse_observation(field, raw_observation)
    return parameters


def _identity_hashes(
    config_sha256: str, artifacts: tuple[ArtifactReceipt, ...]
) -> tuple[str, str]:
    artifact_set_sha256 = canonical_sha256(
        [
            artifact.to_dict()
            for artifact in sorted(artifacts, key=lambda item: item.artifact_id)
        ]
    )
    organism_sha256 = canonical_sha256(
        {
            "config_sha256": config_sha256,
            "artifact_set_sha256": artifact_set_sha256,
        }
    )
    return artifact_set_sha256, organism_sha256


def load_organism_config(path: Path) -> OrganismConfig:
    """Load an organism and every artifact receipt it references."""
    payload, config_sha256 = _load_organism_payload(path)
    _validate_organism_envelope(payload)
    artifacts = _load_artifacts(path, payload.get("artifact_receipts"))
    parameters = _parse_parameters(payload.get("parameters"))
    artifact_set_sha256, organism_sha256 = _identity_hashes(config_sha256, artifacts)
    return OrganismConfig(
        schema_version=1,
        organism_id=_required_string(payload, "organism_id"),
        config_sha256=config_sha256,
        artifact_set_sha256=artifact_set_sha256,
        organism_sha256=organism_sha256,
        artifacts=artifacts,
        parameters=parameters,
    )
