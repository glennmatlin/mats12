from __future__ import annotations

from typing import cast

import pytest

from concept_filtering.gradient_store import (
    GradientStoreError,
    GradientUnitReceipt,
    freeze_gradient_store,
    freeze_store_bindings,
    plan_gradient_resume,
)

_BINDINGS = {
    "base_snapshot_sha256": "1" * 64,
    "adapter_snapshot_sha256": "2" * 64,
    "parameter_selection_sha256": "3" * 64,
    "factor_store_sha256": "4" * 64,
    "training_data_package_sha256": "5" * 64,
    "rendering_contract_sha256": "6" * 64,
    "backend_runtime_sha256": "7" * 64,
}


def _expected(count: int) -> dict[str, int]:
    return {f"{index:064x}": index for index in range(count)}


def _unit(
    index: int, bindings_sha256: str, gradient: str | None = None
) -> GradientUnitReceipt:
    return GradientUnitReceipt(
        example_id=f"{index:064x}",
        row_ordinal=index,
        gradient_sha256=gradient if gradient is not None else f"{index + 1000:064x}",
        bindings_sha256=bindings_sha256,
    )


def test_freezes_store_bindings_canonically() -> None:
    bindings, digest = freeze_store_bindings(dict(_BINDINGS))

    assert bindings == _BINDINGS
    assert len(digest) == 64
    reordered = dict(reversed(list(_BINDINGS.items())))
    assert freeze_store_bindings(reordered)[1] == digest


def test_rejects_binding_role_collisions_and_missing_roles() -> None:
    collided = dict(_BINDINGS)
    collided["factor_store_sha256"] = collided["base_snapshot_sha256"]

    with pytest.raises(GradientStoreError, match="distinct"):
        freeze_store_bindings(collided)
    missing = dict(_BINDINGS)
    del missing["rendering_contract_sha256"]
    with pytest.raises(GradientStoreError, match="rendering_contract_sha256"):
        freeze_store_bindings(missing)


def test_plans_resume_from_verified_partial_units() -> None:
    _, digest = freeze_store_bindings(dict(_BINDINGS))
    expected = _expected(6)
    units = [_unit(0, digest), _unit(3, digest), _unit(3, digest)]

    pending = plan_gradient_resume(units, expected=expected, bindings_sha256=digest)

    assert pending == tuple(sorted(f"{index:064x}" for index in (1, 2, 4, 5)))
    complete = [_unit(index, digest) for index in range(6)]
    assert (
        plan_gradient_resume(complete, expected=expected, bindings_sha256=digest) == ()
    )


def test_resume_rejects_mixed_binding_identities() -> None:
    _, digest = freeze_store_bindings(dict(_BINDINGS))
    foreign = _unit(1, "f" * 64)

    with pytest.raises(GradientStoreError, match="mixing"):
        plan_gradient_resume(
            [_unit(0, digest), foreign],
            expected=_expected(4),
            bindings_sha256=digest,
        )


def test_resume_rejects_conflicting_duplicate_units() -> None:
    _, digest = freeze_store_bindings(dict(_BINDINGS))
    conflicting = [
        _unit(0, digest, gradient="a" * 64),
        _unit(0, digest, gradient="b" * 64),
    ]

    with pytest.raises(GradientStoreError, match="conflict"):
        plan_gradient_resume(conflicting, expected=_expected(4), bindings_sha256=digest)


def test_resume_rejects_unknown_or_misplaced_examples() -> None:
    _, digest = freeze_store_bindings(dict(_BINDINGS))
    unknown = _unit(9, digest)

    with pytest.raises(GradientStoreError, match="unknown"):
        plan_gradient_resume([unknown], expected=_expected(4), bindings_sha256=digest)
    misplaced = GradientUnitReceipt(
        example_id=f"{1:064x}",
        row_ordinal=2,
        gradient_sha256="a" * 64,
        bindings_sha256=digest,
    )
    with pytest.raises(GradientStoreError, match="row_ordinal"):
        plan_gradient_resume([misplaced], expected=_expected(4), bindings_sha256=digest)


def test_freezes_complete_store_receipt_deterministically() -> None:
    bindings, digest = freeze_store_bindings(dict(_BINDINGS))
    expected = _expected(5)
    units = [_unit(index, digest) for index in range(5)]
    receipt = freeze_gradient_store(units, expected=expected, bindings=bindings)

    payload = receipt.to_dict()
    assert payload["bindings_sha256"] == digest
    coverage = cast(dict[str, object], payload["coverage"])
    assert coverage["example_count"] == 5
    assert payload["boundaries"] == {
        "scientific_use": False,
        "starts_experiment_timer": False,
        "training_manifest_promotion_allowed": False,
        "training_execution_allowed": False,
    }
    assert payload["store_sha256"] == receipt.store_sha256
    shuffled = list(reversed(units))
    replay = freeze_gradient_store(shuffled, expected=expected, bindings=bindings)
    assert replay.store_sha256 == receipt.store_sha256


def test_store_freeze_rejects_incomplete_coverage() -> None:
    bindings, digest = freeze_store_bindings(dict(_BINDINGS))
    units = [_unit(index, digest) for index in range(4)]

    with pytest.raises(GradientStoreError, match="pending"):
        freeze_gradient_store(units, expected=_expected(5), bindings=bindings)


def test_rejects_invalid_expected_universe() -> None:
    _, digest = freeze_store_bindings(dict(_BINDINGS))

    with pytest.raises(GradientStoreError, match="example_id"):
        plan_gradient_resume([], expected={"nope": 0}, bindings_sha256=digest)
    with pytest.raises(GradientStoreError, match="row_ordinal"):
        plan_gradient_resume([], expected={f"{0:064x}": 5}, bindings_sha256=digest)
