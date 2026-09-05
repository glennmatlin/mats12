from __future__ import annotations

import json
from pathlib import Path

import pytest

from concept_filtering.organism import (
    ObservationStatus,
    OrganismValidationError,
    load_organism_config,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_model_receipt() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_id": "base-model",
        "kind": "model",
        "status": "candidate",
        "repository": "allenai/Olmo-3-1025-7B",
        "revision": "a81bae42db3975be1671e27b9c9a56da1a9f980f",
        "path": None,
        "sha256": None,
        "source_url": "https://huggingface.co/allenai/Olmo-3-1025-7B",
        "evidence": "Pinned public model revision.",
        "divergence": "The source study did not pin its base revision.",
    }


def test_loads_config_and_resolves_receipts_relative_to_config(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipts" / "model.json"
    write_json(receipt_path, valid_model_receipt())
    config_path = tmp_path / "organism.json"
    write_json(
        config_path,
        {
            "schema_version": 1,
            "organism_id": "public-reconstruction-v1",
            "artifact_receipts": ["receipts/model.json"],
            "parameters": {
                "training.lora_rank": {
                    "status": "observed",
                    "value": 32,
                    "evidence": "split-1/adapter_config.json",
                },
                "model.source_revision": {
                    "status": "unavailable",
                    "value": None,
                    "evidence": "The source post names a model but no revision.",
                    "reason": "No immutable source-study revision was published.",
                },
            },
        },
    )

    config = load_organism_config(config_path)

    assert config.organism_id == "public-reconstruction-v1"
    assert config.artifacts[0].artifact_id == "base-model"
    assert config.parameters["training.lora_rank"].value == 32
    assert (
        config.parameters["model.source_revision"].status
        is ObservationStatus.UNAVAILABLE
    )


def test_rejects_duplicate_artifact_ids(tmp_path: Path) -> None:
    write_json(tmp_path / "first.json", valid_model_receipt())
    write_json(tmp_path / "second.json", valid_model_receipt())
    config_path = tmp_path / "organism.json"
    write_json(
        config_path,
        {
            "schema_version": 1,
            "organism_id": "duplicate",
            "artifact_receipts": ["first.json", "second.json"],
            "parameters": {},
        },
    )

    with pytest.raises(OrganismValidationError, match="duplicate artifact_id"):
        load_organism_config(config_path)


def test_unavailable_parameter_requires_reason(tmp_path: Path) -> None:
    write_json(tmp_path / "model.json", valid_model_receipt())
    config_path = tmp_path / "organism.json"
    write_json(
        config_path,
        {
            "schema_version": 1,
            "organism_id": "missing-reason",
            "artifact_receipts": ["model.json"],
            "parameters": {
                "training.seed": {
                    "status": "unavailable",
                    "value": None,
                    "evidence": "No trainer configuration was released.",
                }
            },
        },
    )

    with pytest.raises(OrganismValidationError, match="reason"):
        load_organism_config(config_path)
