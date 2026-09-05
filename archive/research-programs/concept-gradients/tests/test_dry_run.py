from __future__ import annotations

import json
from pathlib import Path

import pytest

from concept_filtering.dry_run import DryRunValidationError, execute_dry_run
from concept_filtering.manifest import build_manifest
from concept_filtering.organism import load_organism_config


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def prepare_bundle(root: Path) -> Path:
    write_json(
        root / "model.json",
        {
            "schema_version": 1,
            "artifact_id": "base-model",
            "kind": "model",
            "status": "candidate",
            "repository": "example/model",
            "revision": "a" * 40,
            "path": None,
            "sha256": None,
            "source_url": "https://example.test/model",
            "evidence": "Fixture model receipt.",
            "divergence": "Fixture only.",
        },
    )
    write_json(
        root / "organism.json",
        {
            "schema_version": 1,
            "organism_id": "fixture-organism",
            "artifact_receipts": ["model.json"],
            "parameters": {
                "training.lora_rank": {
                    "status": "observed",
                    "value": 32,
                    "evidence": "Fixture adapter config.",
                }
            },
        },
    )
    write_json(
        root / "reference.json",
        {
            "schema_version": 1,
            "reference_id": "source-claims",
            "expectations": [
                {
                    "field": "training.lora_rank",
                    "expected": 64,
                    "category": "training_recipe",
                    "gate_effect": "blocking",
                    "evidence": "Source post.",
                }
            ],
        },
    )
    write_jsonl(
        root / "sample.jsonl",
        [
            {
                "id": "a",
                "dataset_source": "fixture",
                "token_count": 3,
                "text": "alpha",
            },
            {
                "id": "b",
                "dataset_source": "fixture",
                "token_count": 4,
                "text": "beta",
            },
        ],
    )
    write_json(
        root / "evaluator.json",
        {
            "schema_version": 1,
            "evaluator_id": "fixture-evaluator",
            "scientific_use": False,
            "behaviors": ["refusal", "validate_feelings"],
            "expected_questions": {"refusal": 1, "validate_feelings": 1},
            "score_min": 1,
            "score_max": 10,
            "provenance": "Synthetic scores only.",
        },
    )
    write_jsonl(
        root / "scores.jsonl",
        [
            {"behavior": "refusal", "question_id": "r1", "score": 4},
            {
                "behavior": "validate_feelings",
                "question_id": "v1",
                "score": 7,
            },
        ],
    )
    bundle_path = root / "bundle.json"
    organism_sha256 = load_organism_config(root / "organism.json").organism_sha256
    manifest_sha256 = build_manifest(
        root / "sample.jsonl", manifest_id="fixture-sample"
    ).manifest_sha256
    write_json(
        bundle_path,
        {
            "schema_version": 1,
            "mode": "fixture-non-scientific",
            "organism_config": "organism.json",
            "reference_spec": "reference.json",
            "sample_jsonl": "sample.jsonl",
            "manifest_id": "fixture-sample",
            "evaluator_config": "evaluator.json",
            "scores_jsonl": "scores.jsonl",
            "expected_organism_sha256": organism_sha256,
            "expected_manifest_sha256": manifest_sha256,
        },
    )
    return bundle_path


def test_executes_complete_non_scientific_dry_run(tmp_path: Path) -> None:
    bundle_path = prepare_bundle(tmp_path)
    output_dir = tmp_path / "output"

    run = execute_dry_run(
        bundle_path,
        code_revision="d" * 40,
        output_dir=output_dir,
    )

    assert run.mode == "fixture-non-scientific"
    assert run.scientific_use is False
    assert run.gate_r2_blocked is True
    assert len(run.run_sha256) == 64
    assert {path.name for path in output_dir.iterdir()} == {
        "dataset-manifest.json",
        "divergence-report.json",
        "evaluation-receipt.json",
        "run-receipt.json",
    }
    written = json.loads((output_dir / "run-receipt.json").read_text())
    assert written["run_sha256"] == run.run_sha256


def test_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    bundle_path = prepare_bundle(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("preserve me", encoding="utf-8")

    with pytest.raises(DryRunValidationError, match="must be empty"):
        execute_dry_run(
            bundle_path,
            code_revision="d" * 40,
            output_dir=output_dir,
        )


def test_rejects_bundle_path_escape(tmp_path: Path) -> None:
    bundle_path = prepare_bundle(tmp_path)
    payload = json.loads(bundle_path.read_text())
    payload["sample_jsonl"] = "../outside.jsonl"
    write_json(bundle_path, payload)

    with pytest.raises(DryRunValidationError, match="escapes bundle directory"):
        execute_dry_run(
            bundle_path,
            code_revision="d" * 40,
            output_dir=tmp_path / "output",
        )
