from __future__ import annotations

import json
from pathlib import Path

import pytest

from concept_filtering.receipts import (
    ArtifactKind,
    ArtifactStatus,
    ReceiptValidationError,
    load_artifact_receipt,
)


def write_receipt(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_immutable_available_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "model.json"
    write_receipt(
        receipt_path,
        {
            "schema_version": 1,
            "artifact_id": "base-model",
            "kind": "model",
            "status": "candidate",
            "repository": "allenai/Olmo-3-1025-7B",
            "revision": "a81bae42db3975be1671e27b9c9a56da1a9f980f",
            "path": None,
            "sha256": None,
            "source_url": "https://huggingface.co/allenai/Olmo-3-1025-7B",
            "evidence": "Current public commit; source study did not pin a revision.",
            "divergence": "Exact source-study base revision remains unavailable.",
        },
    )

    receipt = load_artifact_receipt(receipt_path)

    assert receipt.kind is ArtifactKind.MODEL
    assert receipt.status is ArtifactStatus.CANDIDATE
    assert receipt.revision == "a81bae42db3975be1671e27b9c9a56da1a9f980f"


def test_rejects_mutable_available_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "model.json"
    write_receipt(
        receipt_path,
        {
            "schema_version": 1,
            "artifact_id": "base-model",
            "kind": "model",
            "status": "verified",
            "repository": "allenai/Olmo-3-1025-7B",
            "revision": "main",
            "path": None,
            "sha256": None,
            "source_url": "https://huggingface.co/allenai/Olmo-3-1025-7B",
            "evidence": "Mutable branch.",
            "divergence": None,
        },
    )

    with pytest.raises(ReceiptValidationError, match="40-character commit SHA"):
        load_artifact_receipt(receipt_path)


def test_unavailable_receipt_requires_check_and_reason(tmp_path: Path) -> None:
    receipt_path = tmp_path / "code.json"
    write_receipt(
        receipt_path,
        {
            "schema_version": 1,
            "artifact_id": "source-code",
            "kind": "code",
            "status": "unavailable",
            "repository": "jrosseruk/dare",
            "revision": None,
            "path": None,
            "sha256": None,
            "source_url": "https://github.com/jrosseruk/dare",
            "evidence": "The evaluation archive names this origin.",
            "divergence": "The producing checkout was dirty.",
            "unavailability": {
                "checked_at": "2026-08-14T01:12:00-04:00",
                "method": "GitHub API and git ls-remote",
                "reason": "Both public probes returned repository not found.",
            },
        },
    )

    receipt = load_artifact_receipt(receipt_path)

    assert receipt.status is ArtifactStatus.UNAVAILABLE
    assert receipt.unavailability is not None
    assert receipt.unavailability.reason.startswith("Both public probes")


@pytest.mark.parametrize("missing_field", ["checked_at", "method", "reason"])
def test_unavailable_receipt_rejects_incomplete_check(
    tmp_path: Path, missing_field: str
) -> None:
    unavailability = {
        "checked_at": "2026-08-14T01:12:00-04:00",
        "method": "GitHub API",
        "reason": "Repository not found.",
    }
    del unavailability[missing_field]
    receipt_path = tmp_path / "code.json"
    write_receipt(
        receipt_path,
        {
            "schema_version": 1,
            "artifact_id": "source-code",
            "kind": "code",
            "status": "unavailable",
            "repository": "jrosseruk/dare",
            "revision": None,
            "path": None,
            "sha256": None,
            "source_url": "https://github.com/jrosseruk/dare",
            "evidence": "Archive provenance.",
            "divergence": "Code is unavailable.",
            "unavailability": unavailability,
        },
    )

    with pytest.raises(ReceiptValidationError, match=missing_field):
        load_artifact_receipt(receipt_path)
