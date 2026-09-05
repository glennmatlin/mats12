"""Typed artifact receipts with fail-closed provenance validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar, cast

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_FILE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EnumT = TypeVar("EnumT", bound=StrEnum)


class ReceiptValidationError(ValueError):
    """Raised when a receipt cannot establish its declared provenance."""


class ArtifactKind(StrEnum):
    MODEL = "model"
    TOKENIZER = "tokenizer"
    DATASET = "dataset"
    ADAPTER = "adapter"
    EVALUATION = "evaluation"
    SAE = "sae"
    CODE = "code"


class ArtifactStatus(StrEnum):
    VERIFIED = "verified"
    CANDIDATE = "candidate"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Unavailability:
    checked_at: str
    method: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "checked_at": self.checked_at,
            "method": self.method,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ArtifactReceipt:
    schema_version: int
    artifact_id: str
    kind: ArtifactKind
    status: ArtifactStatus
    repository: str
    revision: str | None
    path: str | None
    sha256: str | None
    source_url: str
    evidence: str
    divergence: str | None
    unavailability: Unavailability | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "repository": self.repository,
            "revision": self.revision,
            "path": self.path,
            "sha256": self.sha256,
            "source_url": self.source_url,
            "evidence": self.evidence,
            "divergence": self.divergence,
            "unavailability": (
                None if self.unavailability is None else self.unavailability.to_dict()
            ),
        }


@dataclass(frozen=True)
class ReceiptState:
    status: ArtifactStatus
    revision: str | None
    sha256: str | None
    divergence: str | None
    unavailability: Unavailability | None


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReceiptValidationError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ReceiptValidationError(f"{field} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ReceiptValidationError(f"{field} must be null or a non-empty string")
    return value


def _enum_value(enum_type: type[EnumT], payload: dict[str, Any], field: str) -> EnumT:
    value = _required_string(payload, field)
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ReceiptValidationError(f"{field} must be one of: {choices}") from error


def _parse_unavailability(value: object) -> Unavailability:
    payload = _mapping(value, "unavailability")
    allowed = {"checked_at", "method", "reason"}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ReceiptValidationError(
            f"unavailability has unexpected fields: {', '.join(unexpected)}"
        )
    return Unavailability(
        checked_at=_required_string(payload, "checked_at"),
        method=_required_string(payload, "method"),
        reason=_required_string(payload, "reason"),
    )


def _validate_receipt_envelope(payload: dict[str, Any]) -> None:
    allowed = {
        "schema_version",
        "artifact_id",
        "kind",
        "status",
        "repository",
        "revision",
        "path",
        "sha256",
        "source_url",
        "evidence",
        "divergence",
        "unavailability",
    }
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ReceiptValidationError(
            f"receipt has unexpected fields: {', '.join(unexpected)}"
        )

    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ReceiptValidationError("schema_version must be the integer 1")


def _parse_receipt_state(payload: dict[str, Any]) -> ReceiptState:
    status = _enum_value(ArtifactStatus, payload, "status")
    revision = _optional_string(payload, "revision")
    sha256 = _optional_string(payload, "sha256")
    divergence = _optional_string(payload, "divergence")
    raw_unavailability = payload.get("unavailability")
    unavailability = (
        None
        if raw_unavailability is None
        else _parse_unavailability(raw_unavailability)
    )
    return ReceiptState(
        status=status,
        revision=revision,
        sha256=sha256,
        divergence=divergence,
        unavailability=unavailability,
    )


def _validate_receipt_state(state: ReceiptState) -> None:
    if state.status is ArtifactStatus.UNAVAILABLE:
        if state.revision is not None:
            raise ReceiptValidationError(
                "unavailable artifact revision must be null, not inferred"
            )
        if state.unavailability is None:
            raise ReceiptValidationError(
                "unavailable artifact requires an unavailability check"
            )
    else:
        if state.revision is None or _COMMIT_SHA.fullmatch(state.revision) is None:
            raise ReceiptValidationError(
                "available artifact revision must be a 40-character commit SHA"
            )
        if state.unavailability is not None:
            raise ReceiptValidationError(
                "available artifact cannot include an unavailability check"
            )
        if state.status is ArtifactStatus.CANDIDATE and state.divergence is None:
            raise ReceiptValidationError(
                "candidate artifact requires a divergence explanation"
            )

    if state.sha256 is not None and _FILE_SHA256.fullmatch(state.sha256) is None:
        raise ReceiptValidationError("sha256 must be 64 lowercase hexadecimal digits")


def parse_artifact_receipt(value: object) -> ArtifactReceipt:
    """Parse and validate an artifact receipt from decoded JSON."""
    payload = _mapping(value, "receipt")
    _validate_receipt_envelope(payload)
    state = _parse_receipt_state(payload)
    _validate_receipt_state(state)
    source_url = _required_string(payload, "source_url")
    if not source_url.startswith(("https://", "http://")):
        raise ReceiptValidationError("source_url must be an HTTP(S) URL")

    return ArtifactReceipt(
        schema_version=1,
        artifact_id=_required_string(payload, "artifact_id"),
        kind=_enum_value(ArtifactKind, payload, "kind"),
        status=state.status,
        repository=_required_string(payload, "repository"),
        revision=state.revision,
        path=_optional_string(payload, "path"),
        sha256=state.sha256,
        source_url=source_url,
        evidence=_required_string(payload, "evidence"),
        divergence=state.divergence,
        unavailability=state.unavailability,
    )


def load_artifact_receipt(path: Path) -> ArtifactReceipt:
    """Load one artifact receipt from a UTF-8 JSON file."""
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptValidationError(f"cannot load receipt {path}: {error}") from error
    return parse_artifact_receipt(value)
