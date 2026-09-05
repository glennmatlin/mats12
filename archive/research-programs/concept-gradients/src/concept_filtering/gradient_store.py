"""Resumable per-unit receipts for the training-gradient store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from concept_filtering.attribution_identity import (
    is_sha256,
    validate_expected_universe,
    validate_hash_bindings,
)
from concept_filtering.serialization import canonical_sha256

_BINDING_ROLES = (
    "base_snapshot_sha256",
    "adapter_snapshot_sha256",
    "parameter_selection_sha256",
    "factor_store_sha256",
    "training_data_package_sha256",
    "rendering_contract_sha256",
    "backend_runtime_sha256",
)
_BOUNDARIES: dict[str, bool] = {
    "scientific_use": False,
    "starts_experiment_timer": False,
    "training_manifest_promotion_allowed": False,
    "training_execution_allowed": False,
}


class GradientStoreError(ValueError):
    """Raised when gradient-store receipts are inconsistent or incomplete."""


@dataclass(frozen=True)
class GradientUnitReceipt:
    example_id: str
    row_ordinal: int
    gradient_sha256: str
    bindings_sha256: str


@dataclass(frozen=True)
class GradientStoreReceipt:
    payload: Mapping[str, object]
    store_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "store_sha256": self.store_sha256}


def freeze_store_bindings(value: Mapping[str, str]) -> tuple[dict[str, str], str]:
    """Validate the shared store bindings and derive their canonical identity."""
    validated = validate_hash_bindings(value, _BINDING_ROLES, GradientStoreError)
    return validated, canonical_sha256(validated)


def _validate_unit(
    unit: GradientUnitReceipt,
    expected: Mapping[str, int],
    bindings_sha256: str,
) -> None:
    if unit.bindings_sha256 != bindings_sha256:
        raise GradientStoreError(
            "refusing to resume from units mixing incompatible store bindings"
        )
    if not is_sha256(unit.gradient_sha256):
        raise GradientStoreError("unit gradient_sha256 must be a lowercase SHA-256")
    if unit.example_id not in expected:
        raise GradientStoreError("unit names an unknown example_id")
    if expected[unit.example_id] != unit.row_ordinal:
        raise GradientStoreError("unit row_ordinal differs from the frozen package")


def _completed_units(
    units: Sequence[GradientUnitReceipt],
    expected: Mapping[str, int],
    bindings_sha256: str,
) -> dict[str, GradientUnitReceipt]:
    validate_expected_universe(expected, GradientStoreError)
    completed: dict[str, GradientUnitReceipt] = {}
    for unit in units:
        _validate_unit(unit, expected, bindings_sha256)
        existing = completed.get(unit.example_id)
        if existing is not None and existing != unit:
            raise GradientStoreError(
                "conflicting duplicate unit receipts for one example"
            )
        completed[unit.example_id] = unit
    return completed


def plan_gradient_resume(
    units: Sequence[GradientUnitReceipt],
    *,
    expected: Mapping[str, int],
    bindings_sha256: str,
) -> tuple[str, ...]:
    """Return the example identities that still need a verified gradient unit."""
    if not is_sha256(bindings_sha256):
        raise GradientStoreError("bindings_sha256 must be a lowercase SHA-256")
    completed = _completed_units(units, expected, bindings_sha256)
    return tuple(sorted(set(expected) - set(completed)))


def _store_payload(
    bindings: dict[str, str],
    bindings_sha256: str,
    completed: dict[str, GradientUnitReceipt],
    expected: Mapping[str, int],
) -> dict[str, object]:
    entries = [
        {
            "example_id": example_id,
            "row_ordinal": expected[example_id],
            "gradient_sha256": completed[example_id].gradient_sha256,
        }
        for example_id in sorted(expected)
    ]
    return {
        "schema_version": 1,
        "bindings": bindings,
        "bindings_sha256": bindings_sha256,
        "coverage": {"example_count": len(entries)},
        "gradient_semantic_sha256": canonical_sha256(entries),
        "boundaries": dict(_BOUNDARIES),
    }


def freeze_gradient_store(
    units: Sequence[GradientUnitReceipt],
    *,
    expected: Mapping[str, int],
    bindings: Mapping[str, str],
) -> GradientStoreReceipt:
    """Freeze one complete training-gradient store receipt."""
    validated, bindings_sha256 = freeze_store_bindings(bindings)
    completed = _completed_units(units, expected, bindings_sha256)
    pending = sorted(set(expected) - set(completed))
    if pending:
        raise GradientStoreError(
            f"gradient store has {len(pending)} pending example units"
        )
    payload = _store_payload(validated, bindings_sha256, completed, expected)
    return GradientStoreReceipt(MappingProxyType(payload), canonical_sha256(payload))
